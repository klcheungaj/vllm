# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-demand SSD reader for the PLE n-gram embedding table.

The engine (``csrc/ple_disk_reader.cpp``) is JIT-built on first use. Rows are
fetched with page-aligned ``pread`` calls (``O_DIRECT`` when the filesystem
allows it) by a pool of I/O threads, cached in a set-associative row cache,
and delivered either synchronously (:meth:`PLEDiskReader.read_rows`) or from
CUDA host functions acting on bound pinned buffers, which keeps the path
capturable in CUDA graphs.
"""

import functools
import threading
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_SOURCE = Path(__file__).parent / "csrc" / "ple_disk_reader.cpp"
_BUILD_LOCK = threading.Lock()


@functools.cache
def _build_extension() -> Any:
    from torch.utils.cpp_extension import load

    try:
        return load(
            name="vllm_ple_disk_reader",
            sources=[str(_SOURCE)],
            extra_cflags=["-O3", "-std=c++17"],
            extra_ldflags=["-ldl", "-lpthread"],
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to build the PLE disk reader extension. PLE disk offload "
            "JIT-compiles a small C++ extension and needs a C++ compiler "
            f"(g++) and ninja to be installed: {e}"
        ) from e


def load_extension() -> Any:
    """Builds (once per process) and returns the native extension module."""
    with _BUILD_LOCK:
        return _build_extension()


@dataclass(frozen=True)
class PLEDiskSegment:
    """A contiguous run of table rows stored in one file.

    Attributes:
        path: File holding the rows.
        offset: Absolute byte offset in the file of the segment's first row.
        row_start: Global row index of the segment's first row.
        num_rows: Number of rows in the segment.

    """

    path: str
    offset: int
    row_start: int
    num_rows: int


def _stream_handle(stream: "torch.cuda.Stream | int") -> int:
    return stream if isinstance(stream, int) else int(stream.cuda_stream)


class PLEDiskReader:
    """Reads rows of a disk-resident table on demand.

    Args:
        segments: Non-overlapping segments covering the table. Rows outside
            every segment read as zeros.
        row_bytes: Bytes per row (at most 4096).
        cache_rows: Row cache capacity in rows (0 disables the cache).
        io_threads: Number of blocking ``pread`` threads.
        max_inflight: Number of aligned read slots; the device queue depth is
            ``min(io_threads, max_inflight)``.
        direct_io: Use ``O_DIRECT``; falls back to buffered random reads when
            the filesystem rejects it.
        wait_timeout_s: Upper bound on any blocking wait for a demand job.

    """

    def __init__(
        self,
        segments: Sequence[PLEDiskSegment],
        row_bytes: int,
        *,
        cache_rows: int,
        io_threads: int,
        max_inflight: int,
        direct_io: bool,
        wait_timeout_s: float,
    ) -> None:
        if not segments:
            raise ValueError("PLEDiskReader needs at least one segment")
        paths: list[str] = []
        index: dict[str, int] = {}
        for seg in segments:
            if seg.path not in index:
                index[seg.path] = len(paths)
                paths.append(seg.path)
        self.row_bytes = row_bytes
        self.num_rows = max(s.row_start + s.num_rows for s in segments)
        self._requested_direct = direct_io
        self._warned_fallback = False
        self._bound: tuple[torch.Tensor, ...] | None = None
        ext = load_extension()
        self._reader = ext.Reader(
            paths,
            [index[s.path] for s in segments],
            [s.offset for s in segments],
            [s.row_start for s in segments],
            [s.num_rows for s in segments],
            row_bytes,
            cache_rows,
            io_threads,
            max_inflight,
            direct_io,
            float(wait_timeout_s),
        )
        self._finalizer = weakref.finalize(self, self._reader.close)
        self._maybe_warn_fallback()

    def read_rows(self, ids: torch.Tensor, out: torch.Tensor) -> None:
        """Blocking read of rows ``ids`` into ``out`` (row i at bytes
        ``[i * row_bytes, (i + 1) * row_bytes)``).

        Args:
            ids: CPU int64 contiguous tensor of any shape.
            out: CPU uint8 contiguous tensor with at least
                ``ids.numel() * row_bytes`` bytes.

        Raises:
            RuntimeError: On an I/O error or a timeout.

        """
        n = ids.numel()
        self._check_cpu(ids, torch.int64, "ids")
        self._check_cpu(out, torch.uint8, "out")
        if out.numel() < n * self.row_bytes:
            raise ValueError(f"out has {out.numel()} bytes, need {n * self.row_bytes}")
        if n:
            self._reader.read_rows(ids.data_ptr(), n, out.data_ptr())
        self._maybe_warn_fallback()

    def hint(self, ids: "torch.Tensor | np.ndarray") -> None:
        """Advisory, non-blocking prefetch of rows ``ids`` into the cache."""
        if isinstance(ids, torch.Tensor):
            arr = ids.detach().to("cpu", torch.int64).contiguous()
            n, ptr = arr.numel(), arr.data_ptr()
        else:
            arr = np.ascontiguousarray(ids, dtype=np.int64)
            n, ptr = arr.size, arr.ctypes.data
        if n:
            self._reader.hint(ptr, n)

    def bind_demand(
        self, ids: torch.Tensor, num_tokens: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Binds the pinned buffers used by :meth:`launch_notify`.

        Args:
            ids: Pinned int64 ``[max_tokens, heads]`` row ids.
            num_tokens: Pinned int64 ``[1]`` count of valid tokens.
            out: Pinned uint8 ``[max_tokens, heads * row_bytes]`` rows.

        """
        if ids.dim() != 2:
            raise ValueError("ids must be [max_tokens, heads]")
        max_tokens, heads = ids.shape
        for t, dtype, name in (
            (ids, torch.int64, "ids"),
            (num_tokens, torch.int64, "num_tokens"),
            (out, torch.uint8, "out"),
        ):
            self._check_cpu(t, dtype, name)
            if not t.is_pinned():
                raise ValueError(f"{name} must be pinned")
        if num_tokens.numel() < 1:
            raise ValueError("num_tokens must have one element")
        if out.numel() < max_tokens * heads * self.row_bytes:
            raise ValueError("out is too small for [max_tokens, heads] rows")
        self._bound = (ids, num_tokens, out)
        self._reader.bind(
            ids.data_ptr(), num_tokens.data_ptr(), out.data_ptr(), max_tokens, heads
        )

    def launch_notify(self, stream: "torch.cuda.Stream | int") -> None:
        """Enqueues a host function starting a demand job for the bound
        ``ids[:num_tokens]`` once ``stream`` reaches it."""
        self._reader.launch(_stream_handle(stream), False)

    def launch_wait(self, stream: "torch.cuda.Stream | int") -> None:
        """Enqueues a host function blocking ``stream`` until the latest
        notified demand job completes (or fails / times out)."""
        self._reader.launch(_stream_handle(stream), True)

    def check_error(self) -> None:
        """Raises RuntimeError if an I/O error or a timeout was recorded."""
        self._maybe_warn_fallback()
        err = self._reader.error()
        if err:
            raise RuntimeError(f"PLE disk reader error: {err}")

    def stats(self) -> dict[str, int | float | bool]:
        stats = self._reader.stats()
        self._maybe_warn_fallback(stats)
        return stats

    def reset_stats(self) -> None:
        self._reader.reset_stats()

    def close(self) -> None:
        """Stops the I/O threads and frees buffers. Idempotent."""
        self._finalizer()
        self._bound = None

    @staticmethod
    def _check_cpu(t: torch.Tensor, dtype: torch.dtype, name: str) -> None:
        if t.device.type != "cpu" or t.dtype != dtype or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous CPU {dtype} tensor")

    def _maybe_warn_fallback(self, stats: dict | None = None) -> None:
        if not self._requested_direct or self._warned_fallback:
            return
        direct = self._reader.direct_io() if stats is None else stats["direct_io"]
        if not direct:
            self._warned_fallback = True
            logger.warning(
                "O_DIRECT is not supported for the PLE table files; falling "
                "back to buffered random reads (the page cache may grow)."
            )
