# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step latency of the Qwen4Exp PLE row reader on a CUDA stream.

Times what a step pays for its PLE rows the way the model issues them: a
notify host function at the top of the forward, `--overlap-us` of GPU work
(standing in for the embedding and layer 0), then the wait host function,
with the row cache disabled by default. A GPU sleep is queued before each
step, so the CPU's enqueue time is not measured. Per step size it reports:

- notify_us: the notify host function's time on the GPU timeline (anything
  it does delays the compute queued behind it);
- stall_us: step time minus the overlap work, i.e. what the PLE rows add to
  the step (notify + wait + dispatch gaps);
- step_us percentiles and the reader's demand / wait statistics.

With ``--source loopback`` (default) rows come from the in-process loopback
backend with no added delay, which isolates the reader's own overhead from
the network and the server. ``--source disk`` reads a synthetic table file
(with ``O_DIRECT`` by default when the filesystem supports it; otherwise, or
with ``--no-direct-io``, it measures the page cache). ``--source ucx
--server HOST:PORT`` reads from a running PLE memory server
(``vllm/models/qwen4_exp/common/ple_mem_server``).

Example:
    python benchmarks/benchmark_ple_disk_reader.py --rows 16,256,4096,131072

"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from vllm.models.qwen4_exp.common.ple_disk_reader import (
    PLEDiskReader,
    PLEDiskSegment,
    PLERemoteOptions,
)

HEADS = 16
STAT_KEYS = (
    "demand_us_p50",
    "demand_us_p99",
    "demand_small_us_p50",
    "wait_us_p50",
    "wait_us_p99",
)


def _p(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def _make_table(root: Path, num_rows: int, row_bytes: int) -> PLEDiskSegment:
    path = root / "ple_table.bin"
    rng = np.random.default_rng(0)
    with open(path, "wb") as f:
        chunk = 1 << 18
        for start in range(0, num_rows, chunk):
            n = min(chunk, num_rows - start)
            f.write(rng.integers(0, 256, n * row_bytes, dtype=np.uint8).tobytes())
    return PLEDiskSegment(str(path), 0, 0, num_rows)


def _remote_options(args: argparse.Namespace) -> PLERemoteOptions | None:
    if args.source == "disk":
        return None
    extra: dict = {}
    if args.busy_poll is not None:
        extra["busy_poll"] = args.busy_poll
    if args.remote_max_inflight is not None:
        extra["max_inflight"] = args.remote_max_inflight
    if args.max_batch_rows is not None:
        extra["max_batch_rows"] = args.max_batch_rows
    if args.source == "loopback":
        return PLERemoteOptions("loopback", "loopback", **extra)
    if not args.server:
        raise SystemExit("--source ucx needs --server HOST:PORT")
    return PLERemoteOptions(
        args.server,
        "ucx",
        token_file=args.token_file,
        allow_insecure_transport=args.allow_tcp,
        **extra,
    )


def _make_reader(args: argparse.Namespace, segment: PLEDiskSegment) -> PLEDiskReader:
    kwargs: dict = {}
    if args.wait_spin_us is not None:
        kwargs["wait_spin_us"] = args.wait_spin_us
    if args.inline_plan_rows is not None:
        kwargs["inline_plan_rows"] = args.inline_plan_rows
    reader = PLEDiskReader(
        [segment],
        args.row_bytes,
        cache_rows=args.cache_rows,
        io_threads=args.io_threads,
        max_inflight=args.max_inflight,
        direct_io=args.direct_io,
        wait_timeout_s=60.0,
        remote=_remote_options(args),
        **kwargs,
    )
    if args.source != "disk" and not reader.wait_remote_ready(600):
        raise SystemExit(f"remote source not ready: {reader.remote_status()}")
    return reader


def _time_steps(
    reader: PLEDiskReader,
    rows: int,
    args: argparse.Namespace,
    ids: torch.Tensor,
    num: torch.Tensor,
    rng: np.random.Generator,
) -> dict:
    tokens = rows // HEADS
    iters = args.iters if rows <= 4096 else max(20, args.iters // 20)
    warmup = min(args.warmup, max(3, iters // 10))
    stream = torch.cuda.current_stream()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    start, notified, computed, end = events
    lead_cycles = int(300 * args.cycles_per_us)
    overlap_cycles = int(args.overlap_us * args.cycles_per_us)
    step, notify, stall = [], [], []
    for it in range(warmup + iters):
        ids[:tokens].copy_(
            torch.from_numpy(rng.integers(0, args.num_rows, (tokens, HEADS)))
        )
        num.fill_(tokens)
        torch.cuda._sleep(lead_cycles)
        start.record(stream)
        reader.launch_notify(stream)
        notified.record(stream)
        if overlap_cycles:
            torch.cuda._sleep(overlap_cycles)
        computed.record(stream)
        reader.launch_wait(stream)
        end.record(stream)
        end.synchronize()
        if it == warmup - 1:
            reader.reset_stats()
        if it < warmup:
            continue
        total = start.elapsed_time(end) * 1e3
        step.append(total)
        notify.append(start.elapsed_time(notified) * 1e3)
        stall.append(total - notified.elapsed_time(computed) * 1e3)
    stats = reader.stats()
    reader.check_error()
    result = {
        "rows": tokens * HEADS,
        "iters": len(step),
        "notify_us_p50": _p(notify, 50),
        "notify_us_p99": _p(notify, 99),
        "stall_us_p50": _p(stall, 50),
        "stall_us_p99": _p(stall, 99),
        "step_us_p50": _p(step, 50),
        "step_us_p99": _p(step, 99),
    }
    result.update({k: stats[k] for k in STAT_KEYS if k in stats})
    return result


def _cycles_per_us() -> float:
    """Calibrates torch.cuda._sleep (cycles) against GPU time."""
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    cycles = 10_000_000
    torch.cuda._sleep(1000)
    start.record(stream)
    torch.cuda._sleep(cycles)
    end.record(stream)
    end.synchronize()
    return cycles / (start.elapsed_time(end) * 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", choices=("loopback", "disk", "ucx"), default="loopback"
    )
    parser.add_argument("--server", help="memory server HOST:PORT (ucx)")
    parser.add_argument("--token-file")
    parser.add_argument("--allow-tcp", action="store_true")
    parser.add_argument(
        "--rows", default="16,256,4096,131072", help="rows per step (x16 heads)"
    )
    parser.add_argument(
        "--iters", type=int, default=1000, help="steps per size (/20 above 4096)"
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--num-rows", type=int, default=1 << 21, help="table rows")
    parser.add_argument("--row-bytes", type=int, default=160)
    parser.add_argument("--cache-rows", type=int, default=0)
    parser.add_argument("--io-threads", type=int, default=16)
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument(
        "--direct-io", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--remote-max-inflight", type=int)
    parser.add_argument("--max-batch-rows", type=int)
    parser.add_argument(
        "--busy-poll", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--wait-spin-us", type=float)
    parser.add_argument("--inline-plan-rows", type=int)
    parser.add_argument(
        "--overlap-us",
        type=float,
        default=200.0,
        help="GPU work between notify and wait (the embedding + layer 0)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    args.cycles_per_us = _cycles_per_us()
    sizes = [int(r) for r in args.rows.split(",")]
    max_tokens = max(sizes) // HEADS

    with tempfile.TemporaryDirectory() as tmp:
        segment = _make_table(Path(tmp), args.num_rows, args.row_bytes)
        reader = _make_reader(args, segment)
        try:
            ids = torch.zeros(max_tokens, HEADS, dtype=torch.int64, pin_memory=True)
            num = torch.zeros(1, dtype=torch.int64, pin_memory=True)
            out = torch.zeros(
                max_tokens, HEADS * args.row_bytes, dtype=torch.uint8, pin_memory=True
            )
            reader.bind_demand(ids, num, out)
            rng = np.random.default_rng(1)
            results = [_time_steps(reader, rows, args, ids, num, rng) for rows in sizes]
            direct = reader.stats()["direct_io"]
        finally:
            reader.close()

    if args.json:
        print(json.dumps({"source": args.source, "steps": results}, indent=1))
        return
    print(
        f"source={args.source} overlap_us={args.overlap_us:g} "
        f"direct_io={direct} (us; stall = step - overlap)"
    )
    header = list(results[0])
    widths = [max(9, len(h)) for h in header]
    print(" ".join(f"{h:>{w}}" for h, w in zip(header, widths)))
    for r in results:
        cells = (
            f"{r[h]:>{w}}" if isinstance(r[h], int) else f"{r[h]:>{w}.1f}"
            for h, w in zip(header, widths)
        )
        print(" ".join(cells))


if __name__ == "__main__":
    main()
