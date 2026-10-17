# Engram: conditional memory via n-gram lookups

Engram is a *conditional memory* module attached to specific layers of a
transformer backbone. At each position, the preceding few tokens are hashed
into large static embedding tables, and the retrieved rows are fused into the
hidden state, gated by how well they match it. Where Mixture-of-Experts scales
capacity through conditional computation, Engram scales it through conditional
memory: parameters that are never computed over, only looked up, so capacity
grows at O(1) per-token cost. The mechanism and its design rationale are
described in [Conditional Memory via Scalable Lookup: A New Axis of Sparsity
for Large Language Models](https://arxiv.org/abs/2601.07372).

vLLM enables Engram automatically whenever the checkpoint declares Engram
layers, so basic serving needs no extra flags. This page explains what Engram
does and how to control its memory footprint and multi-GPU topology.

## Supported models

Engram layers are declared by the checkpoint, not chosen at serve time:

| Architecture | HF config field | Notes |
| --- | --- | --- |
| `DeepseekV41ForCausalLM` | `engram_layer_ids` | e.g. `deepseek-ai/DeepSeek-V4.1-Flash` |
| `Qwen4ExpForCausalLM` / `Qwen4ExpForConditionalGeneration` | `ple_layer_ids` | same mechanism, called PLE |

Passing `--engram-config` for a model without n-gram embedding layers fails at
startup, as does running on a non-CUDA-alike platform.

## How it works

Engram processes each position in two phases: *retrieval* and *fusion*.

**Retrieval** maps the local token context to static table rows:

1. **Tokenizer compression.** At startup, vLLM builds a many-to-one map from
   the tokenizer that collapses tokens which normalize alike (NFKC, accent
   stripping, lowercasing, whitespace collapsing), so `" The"`, `"the"` and
   `"THE"` hash identically. This concentrates semantic density into a much
   smaller id space; its size must match `engram_compressed_vocab_size` from
   the checkpoint, or startup fails — every hash multiplier derives from it.
2. **Multi-head hashing.** For each n-gram order `n` (up to
   `engram_max_ngram_size - 1`), several hash heads each apply an independent
   multiplicative-XOR hash over the compressed suffix n-gram. Every (n-gram
   order, head) pair indexes its own prime-sized bucket range in the layer's
   table — disjoint across pairs, so collisions cannot collide twice — and
   the retrieved rows are concatenated into one memory vector per position.

**Fusion** turns the retrieved rows into a context-aware contribution to the
hidden state. The rows are context-independent priors and may be noisy (hash
collisions, polysemy), so they are gated: a `wkv` projection produces one key
per hyper-connection branch plus a shared value, and each branch computes a
sigmoid gate over the normalized dot product between its hidden state —
which has already aggregated global context through preceding attention —
and its key. If the retrieved memory contradicts the current context, the
gate tends toward zero and suppresses it. The gated value is added
residually into the branch's stream (`hidden + gate * value`); positions that
take no part in an n-gram (e.g. image spans) are masked so they pass through
untouched.

Engram sits only on the layers listed in the checkpoint. Placement is a
deliberate hardware-algorithm trade-off: early enough that the backbone is
relieved from reconstructing static patterns in its first layers, deep enough
that vLLM can hide table lookups behind the compute of the preceding layers
(see below).

In vLLM, hashing, lookup, and gating run as fused Triton kernels over FP8
rows with ue8m0 block scales. The layer layout — table sizes, n-gram size,
head count — is fixed by the checkpoint and is not user-configurable. Only the
storage and sharding described below are configurable, via
`--engram-config`, which takes a JSON object and can also be set field by
field as `--engram-config.<field>`
([CLI reference](../cli/serve.md#-engram-config)).

## CPU offload

Engram tables dwarf typical embedding weights: [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
carries two Engram layers of roughly 384M table rows each, with 256-byte FP8
entries plus their ue8m0 block scales — about 200 GB of table weights alone.
Because lookups are deterministic — the indices are known from the token ids
alone, before the forward pass — the tables need not sit in GPU memory at
all. By default vLLM stores them in **pinned host memory** and serves lookups
directly from the Triton kernel through unified virtual addressing (UVA),
prefetching on a side CUDA stream so transfers overlap with the compute of
preceding layers. The freed GPU memory goes to the KV cache.

`cpu_offload` defaults to `true` (on) and is set with
`--engram-config.cpu_offload`.

To keep the tables resident on GPU instead (e.g. when host memory is
scarce):

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --engram-config.cpu_offload false
```

CPU offload requires a GPU with UVA support; vLLM fails fast when it is
unavailable.

## Disk offload (Qwen4Exp)

On unified-memory systems such as DGX Spark, pinned host memory and GPU
memory are the same physical pool, so `cpu_offload` saves nothing. With
`disk_offload`, vLLM never loads the PLE table: it skips the table shards at
load time and reads the rows each step needs straight from the checkpoint
safetensors files on the NVMe drive, into a pinned staging buffer that the
GPU reads through UVA. Reads are issued by CUDA-graph-capturable host nodes,
so disk offload works in eager mode and with piecewise, breakable, and full
CUDA graphs.

```bash
vllm serve <qwen4exp-checkpoint> \
  --engram-config.disk_offload true \
  --engram-config.disk_row_cache_rows 4194304
```

`disk_offload` takes precedence over `cpu_offload`. Its tuning knobs are:

| Field | Default | Meaning |
| --- | --- | --- |
| `disk_row_cache_rows` | `1048576` | Rows kept in a host-memory row cache; `0` disables it. |
| `disk_io_threads` | `16` | I/O worker threads. |
| `disk_max_inflight` | `256` | Maximum outstanding row reads. |
| `disk_direct_io` | `true` | Read with `O_DIRECT`, falling back to buffered reads where the filesystem does not support it. |
| `disk_prefetch_hints` | `true` | Prefetch the rows of upcoming prefill chunks while earlier steps run. |
| `disk_wait_timeout_s` | `60.0` | Seconds a step waits for its reads before failing. |
| `disk_stats_interval_s` | `60.0` | Seconds between reader statistics log lines; `0` disables them. |

To size `disk_row_cache_rows`, watch the periodic `PLE disk ...` log line: it
reports the row-cache hit rate, SSD reads, read latency, and the average time
each step waited for its rows. Raise the cache until the wait stops shrinking;
each million rows costs about 160 MB of host memory for an FP8 table.

Only `disk_offload` itself affects the compilation cache; the knobs can be
retuned without recompiling.

In containers, bind-mount the model directory from a real filesystem (for
example `-v /models:/models`) rather than reading it through an overlay
filesystem, which may not support `O_DIRECT`. No `--privileged` flag is
needed. The row reader is a small C++ extension compiled on first use, so the
image needs a C++ compiler and `ninja`; mount `TORCH_EXTENSIONS_DIR` on a
persistent volume to skip the rebuild on container restarts. NVMe autonomous power-state transitions can add millisecond wake-up
latency to sparse reads; if per-step latency is spiky, disable them on the
host with the kernel parameter `nvme_core.default_ps_max_latency_us=0`.

Disk offload requires CUDA (the ROCm Qwen4Exp implementation does not support
it) with tensor, data, and pipeline parallel sizes of 1, and is incompatible
with `dp_shared_memory` and `use_thp`.

### Remote memory offload

When another machine on the network has spare RAM, disk offload can borrow
it over UCX (RDMA over RoCEv2 in production) and serve rows from there
instead of from the SSD:

1. A PLE memory server runs on the machine lending the RAM. It knows nothing
   about models; it only hands out registered memory regions.
2. At startup, vLLM connects to it, borrows enough regions for the PLE table,
   and uploads the table from the local checkpoint files in the background.
   Serving starts right away; rows that are not uploaded yet are still read
   from the SSD.
3. Once uploaded, rows missing from the local row cache are fetched with
   one-sided RDMA reads, which need no CPU on the server. The row cache
   (`disk_row_cache_rows`) still applies in front of the remote reads.

The checkpoint files stay the upload source and the fallback: with
`remote_fallback_to_disk` (the default), vLLM keeps reading from the SSD if
the server cannot be reached at startup or fails later, for the rest of the
run.

Start the memory server first. It is a standalone program that depends only
on UCX, not on vLLM or Python: copy the
`vllm/models/qwen4_exp/common/ple_mem_server/` directory to the machine
lending the RAM and build it with a C++17 compiler and the UCX development
files:

```bash
make -C ple_mem_server            # finds UCX with pkg-config
make -C ple_mem_server UCX_HOME=/opt/ucx   # or from a UCX install prefix
./ple_mem_server/ple_mem_server --port 18515 --max-memory 64G --hugepages
```

| Server flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Address to listen on. |
| `--port` | `18515` | Port to listen on. |
| `--max-memory` | `0` | Maximum bytes lent in total (`K`/`M`/`G`/`T` suffixes); `0` means no limit. |
| `--busy-poll` | off | Spin instead of sleeping between events; lowers latency of the TCP transport at the cost of a CPU core. |
| `--hugepages` | off | Back regions with transparent huge pages. |
| `--verbose` | off | Log more detail to stderr. |

The same directory builds `ple_mem_bench`, a standalone client that measures
the link before vLLM uses it: upload bandwidth, large-read bandwidth, and the
rate and latency (p50/p99/p99.9) of random row-sized reads at several queue
depths. Run it from the vLLM machine with the same UCX environment:

```bash
./ple_mem_server/ple_mem_bench --server 10.0.0.2:18515 --bytes 4G \
  --row-bytes 160 --depth 1,16,64,256 --seconds 5
```

`--verify` checks every read against the uploaded pattern and `--json`
prints machine-readable results; see the directory's `README.md` for all
flags.

Then point vLLM at it:

```bash
vllm serve <qwen4exp-checkpoint> \
  --engram-config.disk_offload true \
  --engram-config.remote_memory_server 10.0.0.2:18515
```

| Field | Default | Meaning |
| --- | --- | --- |
| `remote_memory_server` | `None` | `host:port` (or `[ipv6]:port`) of the memory server; requires `disk_offload`. |
| `remote_max_inflight` | `256` | Maximum outstanding remote row reads. |
| `remote_region_bytes` | `1073741824` | Size of each borrowed memory region. |
| `remote_upload_chunk_bytes` | `67108864` | Size of each staging buffer used for the upload. |
| `remote_connect_timeout_s` | `30.0` | Seconds to wait for connecting and borrowing the regions. |
| `remote_fallback_to_disk` | `true` | Keep reading from the SSD when the server is unreachable or fails; otherwise such a failure is fatal. |

Like the disk knobs, these do not affect the compilation cache. vLLM logs
when the upload completes (size and duration) or when the remote source
fails, and the periodic `PLE disk ...` line adds remote reads, remote read
latency, and the upload progress while it runs.

**UCX.** With `remote_memory_server` set, the row reader is compiled against
UCX, so the vLLM machine needs the UCX headers and libraries (for example
the distribution's `libucx-dev` package, or the UCX bundled with
MLNX_OFED/DOCA). vLLM finds them through `UCX_HOME` (with `include/` and
`lib/` below it) or `pkg-config ucx`. The server needs the same UCX
libraries.

**RoCEv2.** UCX picks the transport and device from its environment; set
the same variables for vLLM and the server, for example:

```bash
export UCX_NET_DEVICES=mlx5_0:1   # the RDMA device and port to use
export UCX_IB_GID_INDEX=3         # the RoCEv2 GID of that port (see show_gids)
export UCX_TLS=rc,tcp
```

`ucx_info -d` lists the available devices and transports. `UCX_TLS=tcp`
alone also works, without RDMA hardware, but UCX then emulates the remote
reads in software: they are slower and need CPU on the server.

**Containers.** No `--privileged` flag is needed. Pass the RDMA devices with
`--device /dev/infiniband`, lift the locked-memory limit with
`--ulimit memlock=-1`, and typically use `--network host` so the RoCE
interfaces and their GIDs are visible, for both the vLLM and the server
containers.

**Memory sizing.** The server must be able to lend the whole table: about
51 GB for Qwen3.8-Flash-Next, plus a little rounding per region (each region
holds whole rows). Use `--max-memory` to cap what one server lends.

**Security.** Run the memory server only on a trusted network. It has no
authentication, UCX traffic is not encrypted, and any peer holding a
region's remote key can read and write the borrowed memory.

## Data-parallel topologies (DeepSeek V4.1)

By default (`embedding_across_dp: false`), each DP replica keeps its own
TP-sharded copy of the tables — no cross-replica communication, but one host
copy per replica.

With `embedding_across_dp: true`, the hash heads are sharded across all
TP × DP ranks into a single table copy. Each step gathers the hash ids of
every co-located DP replica, each rank looks up the heads it owns, and the
rows are exchanged back. This trades per-step DP collectives for a much
smaller table footprint:

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --tensor-parallel-size 2 --data-parallel-size 4 \
  --engram-config '{"embedding_across_dp": true}'
```

`embedding_across_dp` is not supported with elastic expert parallelism yet.
Qwen4Exp PLE tables are ETP-sharded instead and honor only `cpu_offload` and
`disk_offload`.

## Sharing host tables across DP replicas

When the tables are CPU-offloaded and DP replicas are co-located on one node,
`dp_shared_memory` stores **one copy of each TP shard in `/dev/shm`**, shared
by all co-located replicas through `cudaHostRegister`. Each replica then
prefetches only its own tokens — no per-step Engram DP collectives — and host
memory drops from one table copy per replica to one per node.

`dp_shared_memory` defaults to enabled whenever the other settings allow it,
and falls back to per-replica (or DP-sharded) tables with a warning when:

- the DP replicas are not co-located on a single node,
- `/dev/shm` cannot hold the full tables — in containers, raise the limit
  with `--shm-size` or `--ipc=host`,
- expert parallelism is elastic, which is unsupported.

Requirements: `cpu_offload` enabled, `--data-parallel-size > 1`, and a shared
IPC namespace across the co-located replicas.

## Limitations

- CUDA-alike platforms only.
- DeepSeek V4.1 Engram does not support DBO or microbatching; disable
  `--enable-dbo` and set `--ubatch-size 0`.
- `dp_shared_memory` requires `cpu_offload`, `data_parallel_size > 1`, and is
  unsupported with elastic expert parallelism.
- `disk_offload` is Qwen4Exp-only and requires CUDA with TP, DP, and PP sizes
  of 1.
- `remote_memory_server` requires `disk_offload`, UCX on both machines, and a
  trusted network; the memory server is unauthenticated.
