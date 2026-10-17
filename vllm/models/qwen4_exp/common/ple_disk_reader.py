# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-demand SSD reader for the PLE n-gram embedding table.

The engine (``csrc/ple_disk_reader.cpp``) is JIT-built on first use. Rows are
fetched with page-aligned ``pread`` calls (``O_DIRECT`` when the filesystem
allows it) by a pool of I/O threads, cached in a set-associative row cache,
and delivered either synchronously (:meth:`PLEDiskReader.read_rows`) or from
CUDA host functions acting on bound pinned buffers, which keeps the path
capturable in CUDA graphs.

With :class:`PLERemoteOptions`, the table is also uploaded to RAM borrowed
from a memory server and rows are read back with batched read requests
(``csrc/ple_remote_source.h``). The server only answers requests and the
reader exposes none of its memory to it (see ``ple_mem_server/README.md``).
The remote variant of the extension compiles in every backend whose
dependencies are available (e.g. ``ucx``, reaching the standalone PLE memory
server); the reader picks one of them by name when it starts.
"""

import functools
import os
import shutil
import subprocess
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_SOURCE = Path(__file__).parent / "csrc" / "ple_disk_reader.cpp"
_BUILD_LOCK = threading.Lock()
_UCX_HINT = (
    "Install UCX (e.g. `apt install libucx-dev` or a source build) and set "
    "UCX_HOME to its install prefix, or make `pkg-config ucx` work."
)


def ucx_build_flags() -> tuple[list[str], list[str]]:
    """Returns ``(cflags, ldflags)`` for compiling against UCX.

    Looks at ``UCX_HOME`` (``include/`` and ``lib/``) first, then
    ``pkg-config ucx``, then the system headers. ``ldflags`` carry an rpath
    to the UCX library directory when it is known.

    Raises:
        RuntimeError: If UCX cannot be found.

    """
    libs = ["-lucp", "-luct", "-lucs"]
    home = os.environ.get("UCX_HOME")
    if home:
        inc, lib = Path(home) / "include", Path(home) / "lib"
        if not (inc / "ucp" / "api" / "ucp.h").is_file():
            raise RuntimeError(f"UCX_HOME={home} has no include/ucp/api/ucp.h")
        return [f"-I{inc}"], [f"-L{lib}", f"-Wl,-rpath,{lib}", *libs]
    if shutil.which("pkg-config"):
        try:
            cflags = subprocess.run(
                ["pkg-config", "--cflags", "ucx"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.split()
            lib_dir = subprocess.run(
                ["pkg-config", "--variable=libdir", "ucx"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except subprocess.CalledProcessError:
            pass
        else:
            ldflags = [f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"] if lib_dir else []
            return cflags, ldflags + libs
    if Path("/usr/include/ucp/api/ucp.h").is_file():
        return [], libs
    raise RuntimeError(f"UCX was not found. {_UCX_HINT}")


def _no_build_flags() -> tuple[list[str], list[str]]:
    return [], []


def _always_available() -> str:
    return ""


def _ucx_missing() -> str:
    try:
        ucx_build_flags()
    except RuntimeError as e:
        return str(e)
    return ""


@dataclass(frozen=True)
class _RemoteBackend:
    """A remote memory backend the extension can compile in.

    Attributes:
        define: Macro compiling the backend in.
        build_flags: Returns its extra ``(cflags, ldflags)``.
        missing: Availability probe; returns why the backend cannot be built,
            or ``""``.
        requires: Dependency named in build errors.
        hint: How to install it.

    """

    define: str
    build_flags: Callable[[], tuple[list[str], list[str]]] = _no_build_flags
    missing: Callable[[], str] = _always_available
    requires: str = ""
    hint: str = ""


_REMOTE_BACKENDS: dict[str, _RemoteBackend] = {
    "ucx": _RemoteBackend(
        "PLE_WITH_UCX",
        ucx_build_flags,
        missing=_ucx_missing,
        requires="UCX",
        hint=_UCX_HINT,
    ),
    # In-process regions with failure injection, for tests.
    "loopback": _RemoteBackend("PLE_WITH_LOOPBACK"),
}

REMOTE_BACKENDS = ("ucx",)
"""Remote memory backends that can be configured for serving."""


def available_remote_backends() -> tuple[str, ...]:
    """Returns the sorted names of the remote backends whose dependencies are
    available, i.e. those the remote variant of the extension compiles in."""
    return tuple(
        sorted(name for name, spec in _REMOTE_BACKENDS.items() if not spec.missing())
    )


@functools.cache
def _build_extension(backends: tuple[str, ...] | None) -> Any:
    from torch.utils.cpp_extension import load

    cflags = ["-O3", "-std=c++17"]
    ldflags = ["-ldl", "-lpthread"]
    name = "vllm_ple_disk_reader"
    specs = [_REMOTE_BACKENDS[b] for b in backends or ()]
    for backend, spec in zip(backends or (), specs):
        extra_cflags, extra_ldflags = spec.build_flags()
        cflags += [f"-D{spec.define}", *extra_cflags]
        ldflags += extra_ldflags
        name += f"_{backend}"
    try:
        return load(
            name=name,
            sources=[str(_SOURCE)],
            extra_cflags=cflags,
            extra_ldflags=ldflags,
        )
    except Exception as e:
        requires = ", ".join(s.requires for s in specs if s.requires)
        extra = f" plus {requires}" if requires else ""
        hint = "".join(f" {s.hint}" for s in specs if s.hint)
        raise RuntimeError(
            "Failed to build the PLE disk reader extension. PLE disk offload "
            "JIT-compiles a small C++ extension and needs a C++ compiler "
            f"(g++) and ninja to be installed{extra}: {e}{hint}"
        ) from e


def load_extension(remote: bool = False) -> Any:
    """Builds (once per process and variant) and returns the native extension.

    Args:
        remote: Build the variant with the remote memory source and every
            backend of :func:`available_remote_backends` compiled in; the
            module's ``REMOTE_BACKENDS`` lists them. Variants are named after
            their backend set, so differently configured builds never share a
            JIT cache entry.

    """
    with _BUILD_LOCK:
        return _build_extension(available_remote_backends() if remote else None)


def _load_remote_extension(backend: str) -> Any:
    """Returns the remote variant of the extension, checking it has `backend`."""
    if backend not in _REMOTE_BACKENDS:
        raise ValueError(
            f"Unknown PLE remote memory backend {backend!r}; supported: "
            f"{', '.join(_REMOTE_BACKENDS)}"
        )
    ext = load_extension(remote=True)
    if backend not in ext.REMOTE_BACKENDS:
        reason = _REMOTE_BACKENDS[backend].missing() or "not compiled in"
        raise RuntimeError(
            f"PLE remote memory backend {backend!r} is not available: {reason}"
        )
    return ext


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


@dataclass(frozen=True)
class PLERemoteOptions:
    """Remote memory source of a :class:`PLEDiskReader`.

    Attributes:
        server: Memory server address in the backend's format: ``"host:port"``
            or ``"[ipv6]:port"`` for ``ucx``; ``"loopback"`` optionally
            followed by ``?fail_after=N&refuse=1&delay_us=D&corrupt=M`` for
            ``loopback``.
        backend: Transport reaching the server, one of
            :data:`REMOTE_BACKENDS` or ``"loopback"`` (in-process, for tests);
            it must be in :func:`available_remote_backends`.
        max_inflight: Remote row reads in flight. Hint (prefetch) reads take
            at most half of them (rounded down), so demand reads always find
            a slot; with fewer than 2, hints are not prefetched remotely.
        region_bytes: Size of each region borrowed from the server.
        upload_chunk_bytes: Size of each staged upload read from the files.
        connect_timeout_s: Bound on connecting, and on the wait for each
            borrowed region (the server populates a region before granting
            it), not on the whole borrow.
        fallback_to_disk: On a remote failure, keep serving from the SSD
            instead of failing the affected reads.
        max_batch_rows: Most rows per read request (1 to 1024); queued rows
            are sent at once, split into requests of at most this many.
        token_file: File holding the shared secret the server was started
            with (``--token-file``), or None.
        allow_insecure_transport: Allow UCX's tcp transport. UCX's tcp
            transport writes data to any address a peer sends, so it is
            excluded unless allowed; only use it on a trusted network.
        busy_poll: The thread driving the transport spins instead of
            sleeping once set up (one core per reader, i.e. per GPU worker),
            so it plans small demand jobs, sends their reads and picks up
            replies without a wake-up. After about a second without remote
            work it parks until the next one, which then pays one wake-up.
            False sleeps whenever idle.

    """

    server: str
    backend: str = "ucx"
    max_inflight: int = 1024
    region_bytes: int = 1 << 30
    upload_chunk_bytes: int = 64 << 20
    connect_timeout_s: float = 30.0
    fallback_to_disk: bool = True
    max_batch_rows: int = 256
    token_file: str | None = None
    allow_insecure_transport: bool = False
    busy_poll: bool = True


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
            ``min(io_threads, max_inflight)``, of which hint reads take at most
            half (rounded down: none, and no SSD hint prefetch, below 2).
        direct_io: Use ``O_DIRECT``; falls back to buffered random reads when
            the filesystem rejects it.
        wait_timeout_s: Upper bound on any blocking wait for a demand job.
        wait_spin_us: How long the stream's wait host function spins on the
            demand job before sleeping until it completes.
        inline_plan_rows: Largest demand job (rows) planned on the latency
            path: by the busy-polling remote thread, or else by the notify
            host function on the stream. Larger jobs (prefill chunks) are
            planned by a planner thread so the stream does not wait for
            them; 0 sends every job there.
        remote: Upload the table to a PLE memory server and serve rows from
            it once uploaded. The constructor waits for the connection; the
            upload runs in the background (see :meth:`remote_status`).

    Raises:
        ValueError: If ``remote.backend`` is unknown.
        RuntimeError: If ``remote.backend`` is not available (its dependency
            is missing), or the server cannot be used and
            ``remote.fallback_to_disk`` is false.

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
        remote: PLERemoteOptions | None = None,
        wait_spin_us: float = 100.0,
        inline_plan_rows: int = 1024,
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
        if remote is None:
            ext = load_extension()
        else:
            ext = _load_remote_extension(remote.backend)
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
            remote.server if remote else "",
            remote.backend if remote else "",
            remote.max_inflight if remote else 1,
            remote.region_bytes if remote else 1,
            remote.upload_chunk_bytes if remote else 1,
            float(remote.connect_timeout_s) if remote else 1.0,
            remote.fallback_to_disk if remote else True,
            remote.max_batch_rows if remote else 1,
            (remote.token_file or "") if remote else "",
            remote.allow_insecure_transport if remote else False,
            remote.busy_poll if remote else False,
            float(wait_spin_us),
            inline_plan_rows,
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
        """Enqueues a host function blocking ``stream`` until the demand job
        of the matching :meth:`launch_notify` completes.

        Fails closed: if that job failed or timed out, or no notify preceded
        the wait, the stream cannot continue with valid rows, so the host
        function prints the error to stderr and aborts the process.
        """
        self._reader.launch(_stream_handle(stream), True)

    def check_error(self) -> None:
        """Raises RuntimeError if an I/O error or a timeout was recorded."""
        self._maybe_warn_fallback()
        err = self._reader.error()
        if err:
            raise RuntimeError(f"PLE disk reader error: {err}")

    def stats(self) -> dict[str, int | float | bool]:
        """Returns counters and latency percentiles (us) since the last
        :meth:`reset_stats`. ``demand_us_*``: per stream demand job, from its
        notify host function to its last row delivered (to its cancellation
        for a failed or timed-out one); ``demand_small_us_*``: the same for
        jobs of at most 1024 rows (decode steps); ``wait_us_*`` and
        ``demand_wait_us`` (total): time the stream's wait host function
        blocked, spinning included."""
        stats = self._reader.stats()
        self._maybe_warn_fallback(stats)
        return stats

    def reset_stats(self) -> None:
        self._reader.reset_stats()

    def remote_status(self) -> dict[str, Any]:
        """Returns the remote source status.

        Keys: ``state`` (``"disabled"``, ``"connecting"``, ``"uploading"``,
        ``"ready"`` or ``"failed"``), ``uploaded_bytes``, ``total_bytes``,
        ``regions`` and ``error`` (``""`` if none).
        """
        return self._reader.remote_status()

    def wait_remote_ready(self, timeout_s: float | None = None) -> bool:
        """Waits until the upload finished or the remote source failed.

        Returns:
            True if the remote source is ready.

        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            state = self._reader.remote_status()["state"]
            if state in ("ready", "failed", "disabled"):
                return state == "ready"
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

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
