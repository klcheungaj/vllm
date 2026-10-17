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
| `disk_max_inflight` | `256` | Maximum outstanding row reads. Prefetch hints use at most half (rounded down) of `min(disk_io_threads, disk_max_inflight)`, so step reads always find a free one; with fewer than two, hints are not prefetched from the SSD. |
| `disk_direct_io` | `true` | Read with `O_DIRECT`, falling back to buffered reads where the filesystem does not support it. |
| `disk_prefetch_hints` | `true` | Prefetch the rows of upcoming prefill chunks into the row cache while earlier steps run; has no effect when the row cache is disabled. |
| `disk_wait_timeout_s` | `60.0` | Seconds a step waits for its reads before failing. |
| `disk_wait_spin_us` | `100.0` | Microseconds a step's wait spins on the CPU for its rows before sleeping, avoiding a thread wake-up on the decode critical path. |
| `disk_inline_plan_rows` | `1024` | Largest step (in rows) whose reads are planned on the latency path: by the busy-polling remote reader thread when there is one, else by the step's notify on the CUDA stream, which then waits for it (about 0.1 us per row plus one thread wake-up). Larger steps (prefill chunks) are planned by a background thread while the step computes; `0` sends every step there. |
| `disk_stats_interval_s` | `60.0` | Seconds between reader statistics log lines; `0` disables them. |

To size `disk_row_cache_rows`, watch the periodic `PLE disk ...` log line: it
reports the row-cache hit rate, SSD reads and their latency, how long each
step took to get all its rows (`step rows ready`, p50/p99/max over all
steps, including failed or timed-out ones; then p50/p99 over decode-size
steps of at most 1024 rows), and how long the step actually blocked waiting
for them, spinning included (`step wait`). Raise the cache until the wait
stops shrinking; each million rows costs about 160 MB of host memory for an
FP8 table.

During decode, each step reads 16 rows per token the model processes (one
per PLE n-gram head), all at once when the step starts: for example 4 users
with 3 MTP draft tokens read 4 x 4 x 16 = 256 rows. These rows cannot be
prefetched, so they must arrive while the embedding and the first decoder
layer compute; a `step wait` near zero means the reads are fully hidden.

Only `disk_offload` itself affects the compilation cache; the knobs (and
`cpu_offload`, which disk offload overrides) can be changed without
recompiling.

The table must stay unread while the weights load, so `disk_offload`
requires `--load-format` `auto`, `safetensors` or `dummy` (with which the
table rows are zeros) and safetensors checkpoint files. Every
`--safetensors-load-strategy` works: `eager` reads only the other tensors of
the file holding the table, and `prefetch` leaves that file out of the
prefetch. Other load formats, such as `fastsafetensors`, `instanttensor`,
`sharded_state`, `runai_streamer` or `tensorizer`, are rejected at startup.

**Failure handling.** Reads that fail are never replaced with zeros or stale
rows. If a step's rows cannot be read (an I/O error, a short read, a read
taking longer than `disk_wait_timeout_s`, or a remote failure with
`remote_fallback_to_disk` disabled), the worker prints the error to stderr
and aborts before the step uses the rows, and the engine reports that the
worker died.

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
it and serve rows from there instead of from the SSD. The transport is a
pluggable backend chosen with `remote_memory_backend`; currently `ucx` (RDMA
such as RoCEv2 or InfiniBand through UCX) is supported:

1. A PLE memory server runs on the machine lending the RAM. It knows nothing
   about models; it lends plain memory regions and only answers the
   requests of the clients that borrowed them.
2. At startup, vLLM connects to it, borrows enough regions for the PLE table,
   and uploads the table from the local checkpoint files in the background.
   Serving starts right away; rows that are not uploaded yet are still read
   from the SSD.
3. Once uploaded, rows missing from the local row cache are fetched
   remotely: vLLM batches them into read requests of up to
   `remote_max_batch_rows` rows, and the server copies the requested rows
   into its reply. The row cache (`disk_row_cache_rows`) still applies in
   front of the remote reads.

The checkpoint files stay the upload source and the fallback: with
`remote_fallback_to_disk` (the default), vLLM keeps reading from the SSD if
the server cannot be reached at startup or fails later, for the rest of the
run. Without it, a remote failure is fatal: startup fails, or a step that
needs an uploaded row aborts the worker (see the failure handling above).

Start the memory server first. The `ucx` backend's server is a standalone
program that depends only on UCX, not on vLLM or Python: copy the
`vllm/models/qwen4_exp/common/ple_mem_server/` directory to the machine
lending the RAM and build it with a C++17 compiler and the UCX development
files:

```bash
make -C ple_mem_server            # finds UCX with pkg-config
make -C ple_mem_server UCX_HOME=/opt/ucx   # or from a UCX install prefix
./ple_mem_server/ple_mem_server --host 10.0.0.2 --port 18515 \
  --max-memory 64G --token-file /etc/ple/token
```

| Server flag | Default | Meaning |
| --- | --- | --- |
| `--host` | required | Address to listen on: the address of the RDMA (storage) interface. There is no bind-all default. |
| `--port` | `18515` | Port to listen on. |
| `--max-memory` | `auto` | Maximum bytes lent in total (`K`/`M`/`G`/`T` suffixes); `auto` is 90% of `MemAvailable` at startup. |
| `--max-regions` | `16384` | Regions lent in total. |
| `--max-client-regions` | `4096` | Regions lent to one client. |
| `--token-file` | none | Require the shared secret in this file from clients. |
| `--allow-tcp` | off | Allow UCX's tcp transport; insecure, see **Security** below. |
| `--event-wait` | off | Sleep on UCX wakeup events between requests instead of busy-polling; saves a CPU core at the cost of wakeup latency. |
| `--no-hugepages` | off | Back regions with 4 KiB pages instead of 2 MiB-aligned transparent huge pages (the default, which saves a TLB miss on most random row reads). |
| `--verbose` | off | Log every allocation (address, populate time, share in huge pages) and release to stderr. |

Every region is populated when it is lent, and the server logs one line per
client borrow with its size, duration, populate time and the share backed by
huge pages. Populating tens of GB takes a while (more when the kernel has to
compact memory for huge pages, see `transparent_hugepage/defrag`); vLLM's
`remote_connect_timeout_s` bounds the wait for each region, not the whole
borrow.

The same directory builds `ple_mem_bench`, a standalone client that measures
the link before vLLM uses it: upload bandwidth; the idle round-trip latency
of a one-row read; the latency of fetching one step's rows the way the
reader does (N random rows posted at once as requests of at most
`--step-batch` rows, timed until the last row arrives); and the rate,
bandwidth and latency of batched reads across batch sizes and queue depths.
Run it from the vLLM machine with the same UCX environment:

```bash
./ple_mem_server/ple_mem_bench --server 10.0.0.2:18515 --bytes 4G \
  --step-rows 64,128,256 --step-batch 64,128,256 \
  --batch-rows 1,64,256,1024 --depth 1,4,16 --seconds 5 \
  --token-file /etc/ple/token
```

A decode step reads about 16 rows per scheduled token (for example 4 users
with 3 MTP draft tokens: 16 tokens, 256 rows). If the step latency stays
well below the compute before the PLE layer, remote reads are fully hidden;
in vLLM, the `PLE disk ...` log line's `step rows ready` and `step wait`
latencies confirm it.

`--verify` checks every read against the uploaded pattern, `--probe` checks
that the server rejects out-of-bounds and foreign-region requests, and
`--json` prints machine-readable results; see the directory's `README.md`
for all flags.

Then point vLLM at it:

```bash
vllm serve <qwen4exp-checkpoint> \
  --engram-config.disk_offload true \
  --engram-config.remote_memory_server 10.0.0.2:18515 \
  --engram-config.remote_token_file /etc/ple/token
```

| Field | Default | Meaning |
| --- | --- | --- |
| `remote_memory_server` | `None` | Address of the memory server, `host:port` (or `[ipv6]:port`) for `ucx`; requires `disk_offload`. |
| `remote_memory_backend` | `ucx` | Transport used to reach the server. |
| `remote_max_inflight` | `1024` | Maximum outstanding remote row reads; prefetch hints use at most half (rounded down), so decode reads never wait behind them; with fewer than two, hints are not prefetched remotely. |
| `remote_max_batch_rows` | `256` | Maximum rows in one read request (at most `1024`). |
| `remote_region_bytes` | `1073741824` | Size of each borrowed memory region. |
| `remote_upload_chunk_bytes` | `67108864` | Size of each staging buffer used for the upload. |
| `remote_connect_timeout_s` | `30.0` | Seconds to wait for connecting, and for each borrowed region (the server populates a region before granting it); not a bound on the whole borrow. |
| `remote_fallback_to_disk` | `true` | Keep reading from the SSD when the server is unreachable or fails; otherwise such a failure is fatal. |
| `remote_token_file` | `None` | File holding the shared secret of the server's `--token-file`. |
| `remote_allow_insecure_transport` | `false` | Let UCX use its tcp transport; insecure, see **Security** below. |
| `remote_busy_poll` | `true` | Keep the remote reader thread polling instead of sleeping, for the lowest decode latency: it also plans each decode step's reads, so the step's notify on the CUDA stream only hands the step over. Costs one CPU core per GPU worker while requests run (the server busy-polls one more); after about a second without remote work the thread parks, and the first step after that pays one thread wake-up. |

The other `remote_*` fields are only valid together with
`remote_memory_server`. Like the disk knobs, they do not affect the
compilation cache. vLLM logs when the upload completes (size and duration)
and when the remote source fails, also after it became ready, and the
periodic `PLE disk ...` line adds remote reads, remote read latency, and the
upload progress while it runs.

**Backends.** The row reader is a small C++ extension compiled on first use.
With `remote_memory_server` set, vLLM compiles a variant containing every
backend whose dependencies it finds on the machine (UCX is found through
`UCX_HOME` or `pkg-config`), and `remote_memory_backend` picks one of them
once at startup; selecting a backend that could not be compiled in fails
with the reason. The plain `disk_offload` variant contains no backend and
loads no backend library. Each backend's transport is a template parameter
of the generic remote source, so after startup the per-read path has no
dynamic dispatch.

**UCX.** The `ucx` variant is compiled against UCX, so the vLLM machine
needs the UCX headers and shared libraries (for example
the distribution's `libucx-dev` package, or the UCX bundled with
MLNX_OFED/DOCA). vLLM finds them through `UCX_HOME` (with `include/` and
`lib/` below it) or `pkg-config ucx`. The server needs the same UCX
libraries.

**RoCEv2.** UCX picks the transport and device from its environment; set
the same variables for vLLM and the server, for example:

```bash
export UCX_NET_DEVICES=mlx5_0:1   # the RDMA device and port to use
export UCX_IB_GID_INDEX=3         # the RoCEv2 GID of that port (see show_gids)
export UCX_TLS=rc                 # optional; RDMA transports only
```

`ucx_info -d` lists the available devices and transports. vLLM and the
server remove UCX's tcp transport from `UCX_TLS` (and fail if nothing is
left) unless `remote_allow_insecure_transport` and `--allow-tcp` are set.
Those make the remote source work without RDMA hardware, which is meant for
tests on an isolated host or network only.

**Containers.** No `--privileged` flag is needed. Pass the RDMA devices with
`--device /dev/infiniband`, lift the locked-memory limit with
`--ulimit memlock=-1`, and typically use `--network host` so the RoCE
interfaces and their GIDs are visible, for both the vLLM and the server
containers.

**Memory sizing.** The server must be able to lend the whole table: about
51 GB for Qwen3.8-Flash-Next, plus a little rounding per region (each region
holds whole rows). By default it lends at most 90% of the memory available
when it starts (within its cgroup memory limit when it runs in a
container); set `--max-memory` to cap it explicitly, and
`--max-regions`/`--max-client-regions` to bound the region count.
On a multi-socket server, run it on the NUMA node the NIC is attached to
(`cat /sys/class/net/<dev>/device/numa_node`), with its memory there too,
e.g. `numactl --cpunodebind=0 --membind=0 ./ple_mem_server ...`, so that row
reads do not cross the socket interconnect; that node then needs the whole
table in free memory.

**Security.** Run the memory server only on a trusted network, bound to the
RDMA interface. The server can only reach memory the client registered for
it (with RDMA, enforced by the NIC through memory keys); it cannot address
arbitrary host memory. The client does not enable UCX's remote-memory-access
feature, whose software handlers would accept arbitrary addresses, and
refuses UCX's tcp transport unless `remote_allow_insecure_transport` is set,
because UCX's tcp transport accepts writes to arbitrary addresses from any
peer. The server checks that every request stays inside a region of the
connection that borrowed it, and vLLM drops the connection on any reply it
did not ask for or that has the wrong size. Traffic is not encrypted: the
optional token (`--token-file`/`remote_token_file`) only keeps other clients
on the network from borrowing memory, and anyone who can observe the network
can read the token and the table. These guarantees assume the server runs
this protocol on an unmodified UCX: UCX's own wire protocol trusts its peer,
so a deliberately modified UCX peer may still crash or corrupt the vLLM
process. See
[the security guide](../usage/security.md#ple-memory-server) for details.

#### Docker images

Two images package this setup; build them from the repository root:

```bash
# On the DGX Spark (arm64; an x86 host would have to emulate it):
docker buildx bake -f docker/docker-bake.hcl dgx-spark
# For the memory server (x86_64, no CUDA):
docker buildx bake -f docker/docker-bake.hcl ple-mem-server
```

- `docker/Dockerfile.dgx_spark` layers on the `vllm-openai` image that
  `docker/Dockerfile` builds for linux/arm64 and CUDA 13 (the
  `openai-dgx-spark-base` bake target; `DGX_SPARK_CUDA_VERSION` picks the
  CUDA 13 release) and refuses a base image built for another CUDA major
  version; the host needs the Spark's CUDA 13 driver (R580 or newer). It
  adds UCX built by
  `tools/install_ucx_from_source.sh` (verbs, rdmacm and mlx5, without CUDA
  support, since rows travel through pinned host memory and need no
  GPUDirect RDMA), the RDMA user-space libraries, g++ for the reader's JIT
  build, `ple_mem_bench`, and the reader extensions prebuilt in
  `TORCH_EXTENSIONS_DIR=/opt/vllm/torch_extensions`. The reader finds that
  UCX through `PKG_CONFIG_PATH` (setting `UCX_HOME` instead makes UCX warn
  about an unknown `UCX_*` variable). Leave `INSTALL_KV_CONNECTORS` off for this
  image: connectors that bundle their own UCX would share one UCX library
  with the reader in the same process.
- `docker/Dockerfile.ple_mem_server` contains the same UCX, the RDMA
  user-space libraries, `ple_mem_server` and `ple_mem_bench` on Ubuntu,
  runs as an unprivileged user, and passes its arguments to
  `ple_mem_server`.

Both NICs, the Spark's ConnectX-7 and a ConnectX-6 Dx on the server, use the
mlx5 driver and UCX's `rc_mlx5` transport over RoCEv2, so the images need no
NIC-specific build. The kernel driver (inbox or DOCA-OFED) stays on the
host; the images carry only rdma-core's user-space part. Run both containers
with `--network host` (RoCE addresses and GIDs belong to the host's
interfaces), `--device /dev/infiniband` and `--ulimit memlock=-1`:

```bash
# Memory server, pinned to the NIC's NUMA node with Docker's cpuset flags
# (numactl's memory binding would need extra capabilities):
docker run -d --name ple-mem-server --network host \
    --device /dev/infiniband --ulimit memlock=-1 \
    --cpuset-cpus 0-9 --cpuset-mems 0 \
    -e UCX_NET_DEVICES=mlx5_0:1 \
    -v /etc/ple_mem.token:/run/ple_mem.token:ro \
    vllm-ple-mem-server --host 10.0.0.2 --max-memory 60G \
    --token-file /run/ple_mem.token

# DGX Spark: check the link first, then serve.
docker run --rm --network host --device /dev/infiniband \
    --ulimit memlock=-1 -e UCX_NET_DEVICES=<device>:1 \
    -v /etc/ple_mem.token:/run/ple_mem.token:ro --entrypoint ple_mem_bench \
    vllm:dgx-spark --server 10.0.0.2:18515 --bytes 4G \
    --token-file /run/ple_mem.token
docker run --gpus all --ipc=host --network host \
    --device /dev/infiniband --ulimit memlock=-1 \
    -e UCX_NET_DEVICES=<device>:1 -v /models:/models \
    -v /etc/ple_mem.token:/run/ple_mem.token:ro \
    vllm:dgx-spark /models/Qwen3.8-Flash-Next \
    --engram-config.disk_offload true \
    --engram-config.remote_memory_server 10.0.0.2:18515 \
    --engram-config.remote_token_file /run/ple_mem.token
```

The `PLE offload` GitHub workflow (`.github/workflows/ple-offload.yml`)
runs the PLE tests on a CPU runner (the remote reader over UCX's tcp
transport), builds and smoke-tests the memory server image, and builds the
Spark image on GitHub's hosted arm64 runner. That runner cannot compile
vLLM's kernels in time, so the Spark image takes them from the upstream
wheel built for the commit the branch is based on
(`DGX_SPARK_USE_PRECOMPILED=1` and `VLLM_MERGE_BASE_COMMIT` in the bake
file), which is only valid while the branch does not change them; it has no
GPU, so the GPU tests do not run. To build on a DGX Spark instead, compiling
the kernels and running the PLE tests on its GPU, register it as a
self-hosted runner and set the repository variable `DGX_SPARK_RUNNER` to its
labels as a JSON list, e.g. `["self-hosted", "linux", "ARM64",
"dgx-spark"]` (set `DGX_SPARK_USE_PRECOMPILED` to `1` to skip compiling
there too). Pull requests from forks never run this job.

Outside pull requests, the workflow pushes the tested images to GitHub
Container Registry as `ghcr.io/<owner>/vllm-ple-mem-server` and
`ghcr.io/<owner>/vllm-dgx-spark`, tagged with the short commit and the
branch name, or with the tag name and `latest` for `v*` tags; pull them
instead of building, e.g.
`docker pull ghcr.io/<owner>/vllm-dgx-spark:<branch>`. On a hosted runner,
which lacks the disk for the build cache and a local copy of the Spark
image at once, BuildKit pushes the image straight to GHCR as
`ci-<commit>`, and the job pulls it back for the checks before adding the
other tags; the build logs the disk usage every minute.

Host settings the containers cannot make:

- `UCX_NET_DEVICES`: the Spark's ConnectX-7 appears as several RDMA
  devices, one per port and PCIe link; `rdma link show` on the host maps
  each to its network interface, so pick the one that is cabled to the
  server. `ucx_info -d` inside a container lists what UCX sees. UCX selects
  a RoCEv2 GID by itself; set `UCX_IB_GID_INDEX` only if that fails.
- MTU: set the RoCE interfaces (and the switch) to MTU 9000 so RoCE runs
  with 4 KiB packets, and configure the network for RoCE (PFC or ECN) as for
  any RDMA traffic.
- The server's huge pages are transparent huge pages requested with
  `madvise`; `/sys/kernel/mm/transparent_hugepage/enabled` on the server
  host must be `madvise` or `always`.
- `--cpuset-mems` confines the server's memory to one NUMA node, which
  `--max-memory auto` does not account for: pass a `--max-memory` that fits
  in that node's free memory.

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
- `disk_offload` requires the `auto`, `safetensors` or `dummy` load format
  and safetensors checkpoint files.
- `remote_memory_server` requires `disk_offload` and a trusted network (the
  traffic is not encrypted and the optional token is a shared secret); the
  `ucx` backend needs UCX on both machines and RDMA unless the insecure tcp
  transport is explicitly allowed.
