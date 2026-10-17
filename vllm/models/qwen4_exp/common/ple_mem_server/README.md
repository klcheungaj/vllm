# PLE memory server

A standalone UCX server that lends its RAM to vLLM instances serving Qwen4Exp
with the PLE n-gram table offloaded (`--engram-config.disk_offload` plus
`--engram-config.remote_memory_server=HOST:PORT`). vLLM borrows regions at
startup, uploads the table from its local checkpoint, and reads rows back with
one-sided RDMA GETs. The server knows nothing about models.

This directory is self-contained: copy it to the memory machine and build it
with a C++17 compiler and UCX (no Python, PyTorch or vLLM needed).

| File | Purpose |
| --- | --- |
| `ple_mem_server.cpp` | The server |
| `ple_mem_bench.cpp` | Client benchmark (bandwidth, latency, verification) |
| `ple_mem_protocol.h` | Wire protocol, shared with vLLM's PLE reader |
| `ple_mem_util.h` | Size parsing and address helpers |

## Build

```bash
make                         # UCX found with pkg-config
make UCX_HOME=/opt/ucx       # UCX install prefix (adds an rpath)
make OUT=/tmp/ple            # put the binaries elsewhere
```

## Run the server

```bash
./ple_mem_server --host 0.0.0.0 --port 18515 --max-memory 60G
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Listen address |
| `--port` | `18515` | Listen port (`0` picks a free one, logged at startup) |
| `--max-memory` | `0` | Cap on lent memory, K/M/G/T suffixes (`0` = no limit) |
| `--busy-poll` | off | Spin instead of sleeping on UCX wakeup events |
| `--hugepages` | off | `madvise(MADV_HUGEPAGE)` on lent regions |
| `--verbose` | off | Log every request |

Regions are prefaulted when lent, so a shortage fails the client's startup
instead of a later access. A client's regions are freed when it releases them,
disconnects or dies. Connections, allocations and totals are logged to stderr.
SIGINT / SIGTERM shut the server down.

## Benchmark

With the server running, from the vLLM machine:

```bash
./ple_mem_bench --server 10.0.0.2:18515 --bytes 8G --verify
```

It borrows `--bytes` (in `--region-bytes` regions), uploads them with
`--chunk-bytes` PUTs (`--upload-depth` in flight) and a flush, reads them back
with large GETs, then runs random `--row-bytes` GETs at each `--depth` for
`--iters` operations (or `--seconds`), reporting ops/s, MB/s and per-op
latency p50 / p99 / p99.9 / max. `--verify` fills the memory with an
offset-derived pattern and checks every GET (and slows the upload).
`--json` prints a single JSON object. The exit code is non-zero on any error
or mismatch; `--help` lists all flags.

## Transports (RoCE)

Both sides use UCX defaults. Useful environment variables:

- `UCX_NET_DEVICES=mlx5_0:1` selects the RDMA port; `ucx_info -d` lists them.
- `UCX_TLS=rc_x,sm,self` (or `rc,ud`) forces RDMA transports;
  `UCX_TLS=tcp` runs over TCP without RDMA hardware (software RMA, the server
  then has to progress every access and is much slower).
- `UCX_IB_GID_INDEX=3` picks the RoCEv2 GID when the default one is RoCEv1.
- `UCX_RNDV_THRESH` and `UCX_MAX_RNDV_RAILS` tune large transfers.

Check with `ucx_info -d` on both machines that the RDMA devices are visible,
then compare `ple_mem_bench` small-GET latency at depth 1 with and without
`UCX_TLS=tcp` to confirm RDMA is in use.
