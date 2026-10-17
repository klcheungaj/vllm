# PLE memory server

A standalone UCX server that lends its RAM to vLLM instances serving Qwen4Exp
with the PLE n-gram table offloaded (`--engram-config.disk_offload` plus
`--engram-config.remote_memory_server=HOST:PORT`). vLLM borrows regions at
startup, uploads the table from its local checkpoint, and reads rows back with
batched read requests. The server knows nothing about models.

This directory is self-contained: copy it to the memory machine and build it
with a C++17 compiler and UCX (no Python, PyTorch or vLLM needed).

| File | Purpose |
| --- | --- |
| `ple_mem_server.cpp` | The server |
| `ple_mem_bench.cpp` | Client benchmark (upload bandwidth, read latency, probes) |
| `ple_mem_protocol.h` | Wire protocol |
| `ple_mem_client.h` | Client, shared by the benchmark and vLLM's PLE reader |
| `ple_mem_util.h` | Size parsing, addresses, token, UCX transport policy |

## Build

```bash
make                         # UCX found with pkg-config
make UCX_HOME=/opt/ucx       # UCX install prefix (adds an rpath)
make OUT=/tmp/ple            # put the binaries elsewhere
```

## Run the server

```bash
./ple_mem_server --host 10.0.0.2 --port 18515 --max-memory 60G \
    --token-file /etc/ple_mem.token
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | required | Listen address: the RDMA / storage interface the vLLM hosts reach (`0.0.0.0` = every interface) |
| `--port` | `18515` | Listen port (`0` picks a free one, logged at startup) |
| `--max-memory` | `auto` | Cap on lent memory in mapped bytes (K/M/G/T suffixes); `auto` = 90% of MemAvailable at startup |
| `--max-regions` | `16384` | Regions lent in total |
| `--max-client-regions` | `4096` | Regions per client |
| `--token-file` | none | Shared secret clients must send (file content, surrounding whitespace trimmed, at most 256 bytes) |
| `--allow-tcp` | off | Allow UCX's tcp transport (insecure, see below) |
| `--event-wait` | off | Sleep on UCX wakeup events between requests: saves a core, adds wakeup latency (default busy-polls) |
| `--no-hugepages` | off | Back lent regions with 4 KiB pages only (by default, regions of 2 MiB or more get transparent huge pages; `--hugepages` is accepted as a no-op) |
| `--verbose` | off | Log every allocation (address, populate time, share backed by huge pages) and release |

Regions are prefaulted when lent, so a shortage fails the client's startup
instead of a later access. Regions of 2 MiB or more are mapped 2 MiB-aligned
and madvised (`MADV_HUGEPAGE`) so that transparent huge pages back them and
random row reads rarely miss the TLB; this needs
`/sys/kernel/mm/transparent_hugepage/enabled` set to `always` or `madvise`
(the server logs when it is `never`). With
`/sys/kernel/mm/transparent_hugepage/defrag` at `madvise` (a common default),
populating may compact memory synchronously, which can slow the startup of a
long-running, fragmented server. The server logs one line per client borrow
(regions, size, duration, time spent populating, share in huge pages);
`--verbose` adds one per region. Clients bound the wait for each region, not
the whole borrow (`remote_connect_timeout_s` in vLLM, `--timeout` here).
Every region is charged its mapped size:
rounded up to 2 MiB for those, to 4 KiB otherwise (and with
`--no-hugepages`). A read request prefetches all of its rows before copying
them into the reply, so their memory accesses overlap. A client's regions are freed when it
releases them, disconnects or dies. Connections, refusals and totals are
logged to stderr. SIGINT / SIGTERM shut the server down.

## Protocol and security model

The server and its clients exchange requests and replies as UCX active
messages (`ple_mem_protocol.h`): allocate, free, write (upload) and batched
row reads. The server only answers requests and never starts a transfer to a
client on its own:

- **The server exposes nothing.** Its regions are plain anonymous memory,
  never registered with UCX. Only the request handlers touch them, after
  checking that the region belongs to the requesting connection and that the
  access is in bounds. Every request is size-checked; a bad one gets an error
  reply (or, if malformed, the connection is closed).
- **The client exposes nothing it did not register for a transfer.** Its UCX
  context has active messages only. UCX's software RMA handlers, which write
  whatever address a peer sends, are registered only for contexts with RMA,
  so they are never active. Read replies are always sent eagerly. Large
  uploads may go by rendezvous, in which the server's UCX reads only the
  client's source buffer of that upload. The client copies reply bytes into
  its buffers only after the reply matched an outstanding request of the
  connection exactly (type, id, row count, length). Anything else counts as
  a protocol violation: the client stops using the server and falls back to
  the SSD.
- **UCX's tcp transport is refused** on both sides unless explicitly
  allowed (`--allow-tcp` here, `remote_allow_insecure_transport` in vLLM).
  Its PUT handler writes data to any address the peer sends, regardless of
  configuration, so any process that can reach a tcp-enabled UCX endpoint
  can write its memory. The tools remove `tcp` from `UCX_TLS` (honoring the
  rest of it) and verify the connection's lanes. Allow tcp only on a trusted
  network, for tests or for machines without RDMA.
- **Authentication** is an optional shared secret (`--token-file`), compared
  in constant time. The channel is not encrypted, so the token only stops
  accidental or unauthorized use on a network you already trust. Bind
  `--host` to the RDMA / storage interface, and firewall the port.
- **Residual risk.** UCX's own wire protocol trusts its peer: some internal
  handlers dereference request ids sent by the peer, and some copy
  fragments at offsets the peer chooses. A malicious peer that speaks UCX
  could crash, or possibly corrupt, the other side through UCX itself.
  Run the server and its clients only on an isolated, trusted network.

## Benchmark

With the server running, from the vLLM machine:

```bash
./ple_mem_bench --server 10.0.0.2:18515 --bytes 8G --verify --probe
```

It borrows `--bytes` (in `--region-bytes` regions) and uploads them in
`--chunk-bytes` writes (`--upload-depth` in flight, at most
`--write-window` messages of 4 MiB each in flight). Then it measures:

- **Ping**: the idle round trip of a one-row read, `--ping-iters` times
  (min / mean / p50 / p90 / p99 / p99.9 / max).
- **Step latency**: for each `--step-batch` size B and each `--step-rows`
  size, the time to fetch that many random rows the way vLLM's reader does:
  posted at once as requests of at most B rows (`remote_max_batch_rows`)
  and timed until the last row arrives, `--step-iters` times. `--verify`
  checks rows inside the timed loop, so leave it off when comparing batch
  sizes. A step reads about
  tokens x 16 rows (one per PLE head); compare its latency with the compute
  before the PLE layer.
- **Sweep**: batched row reads for each `--batch-rows` x `--depth` point
  (rows per request, requests in flight), for `--iters` requests or
  `--seconds`: rows/s, MB/s, and request latency p50 / p99 / p99.9 / max,
  also divided by the batch size (us per row). `--no-sweep` skips it.

- `--verify` fills the memory with an offset-derived pattern and checks
  every row.
- `--probe` checks that the server rejects reads and writes past a
  region's end, and reads and writes of another client's region.
- `--token-file` and `--allow-tcp` match the server's flags.
- `--json` prints a single JSON object.

The exit code is non-zero on any error, mismatch or accepted probe. `--help`
lists all flags.

For one engine step needing N rows, the reader sends N / `max_batch_rows`
requests at once. So the request latency at batch B and depth N / B
approximates the step's fetch time. Use this to tune
`--engram-config.remote_max_batch_rows` (default 256).

## Transports (RoCE)

Both sides use UCX defaults minus tcp. Useful environment variables:

- `UCX_NET_DEVICES=mlx5_0:1` selects the RDMA port; `ucx_info -d` lists them.
- `UCX_TLS=rc_x,sm,self` (or `rc,ud`) forces RDMA transports. With
  `UCX_TLS=tcp`, both tools refuse to start unless tcp is allowed.
- `UCX_IB_GID_INDEX=3` picks the RoCEv2 GID when the default one is RoCEv1.
- `UCX_RNDV_THRESH` tunes when uploads switch to rendezvous.

Check with `ucx_info -d` on both machines that the RDMA devices are visible,
then compare `ple_mem_bench` latency at batch 1, depth 1 against the tcp
numbers to confirm that RDMA is in use.
