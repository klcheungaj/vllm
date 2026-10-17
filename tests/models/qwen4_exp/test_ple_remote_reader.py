# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the PLE reader's remote memory source and the standalone memory
server / benchmark (``ple_mem_server/``).

The server runs as a subprocess over UCX's TCP transport (software RMA), so
no RDMA hardware is needed; the module is skipped when UCX, make or g++ is
missing.
"""

import os
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import regex as re
import torch

import vllm.models.qwen4_exp.common as qwen4_common
from vllm.models.qwen4_exp.common.ple_disk_reader import (
    PLEDiskReader,
    PLERemoteOptions,
    ucx_build_flags,
)

from .test_ple_disk_reader import NUM_ROWS, ROW_BYTES, _mixed_ids, _Table

SERVER_DIR = os.path.join(os.path.dirname(qwen4_common.__file__), "ple_mem_server")


def _ucx_missing() -> str:
    if not sys.platform.startswith("linux"):
        return "PLE disk reader is Linux-only"
    if not (shutil.which("make") and shutil.which("g++")):
        return "make and g++ are required"
    try:
        ucx_build_flags()
    except RuntimeError as e:
        return str(e)
    return ""


pytestmark = pytest.mark.skipif(bool(_ucx_missing()), reason=_ucx_missing())

REGION_BYTES = 100_003  # 625 rows, not aligned to pages, chunks or segments
REGION_ROWS = REGION_BYTES // ROW_BYTES
CHUNK_BYTES = 50_000  # 312 rows per staged upload
UCX_ENV = {"UCX_TLS": "tcp", "UCX_WARN_UNUSED_ENV_VARS": "n"}


class _Server:
    def __init__(self, bin_dir: str, *args: str) -> None:
        binary = os.path.join(bin_dir, "ple_mem_server")
        self.proc = subprocess.Popen(
            [binary, "--host", "127.0.0.1", "--port", "0", *args],
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **UCX_ENV},
        )
        self.lines: list[str] = []
        self._cv = threading.Condition()
        threading.Thread(target=self._drain, daemon=True).start()
        match = re.search(r":(\d+) \(", self.wait_for("listening on"))
        assert match, self.lines
        self.port = int(match.group(1))

    def _drain(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            with self._cv:
                self.lines.append(line)
                self._cv.notify_all()

    def wait_for(self, text: str, timeout_s: float = 10.0) -> str:
        with self._cv:
            found = self._cv.wait_for(
                lambda: any(text in s for s in self.lines), timeout_s
            )
            assert found, f"server never logged {text!r}: {self.lines}"
            return next(s for s in self.lines if text in s)

    @property
    def address(self) -> str:
        return f"127.0.0.1:{self.port}"

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGKILL)
        self.proc.wait(timeout=10)


@pytest.fixture(scope="module")
def table(tmp_path_factory) -> _Table:
    return _Table(tmp_path_factory.mktemp("ple_remote_table"))


@pytest.fixture(scope="session")
def bin_dir(tmp_path_factory) -> str:
    """Builds the server and benchmark with their standalone Makefile."""
    out = str(tmp_path_factory.mktemp("ple_mem_bin"))
    cmd = ["make", "-C", SERVER_DIR, f"OUT={out}"]
    if os.environ.get("UCX_HOME"):
        cmd.append(f"UCX_HOME={os.environ['UCX_HOME']}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


@pytest.fixture
def start_server(bin_dir, monkeypatch):
    for key, value in UCX_ENV.items():
        monkeypatch.setenv(key, value)
    servers: list[_Server] = []

    def start(*args: str) -> _Server:
        servers.append(_Server(bin_dir, *args))
        return servers[-1]

    yield start
    for server in servers:
        server.kill()


def _reader(segments, address: str, **kwargs) -> PLEDiskReader:
    remote = dict(
        region_bytes=REGION_BYTES,
        upload_chunk_bytes=CHUNK_BYTES,
        max_inflight=64,
        connect_timeout_s=10.0,
    )
    remote.update(kwargs.pop("remote", {}))
    args = dict(
        cache_rows=0,
        io_threads=8,
        max_inflight=64,
        direct_io=True,
        wait_timeout_s=30.0,
        remote=PLERemoteOptions(address, **remote),
    )
    args.update(kwargs)
    return PLEDiskReader(segments, ROW_BYTES, **args)


def _read(reader: PLEDiskReader, ids: np.ndarray) -> np.ndarray:
    out = torch.empty(ids.size * ROW_BYTES, dtype=torch.uint8)
    reader.read_rows(torch.from_numpy(ids), out)
    return out.numpy().reshape(ids.size, ROW_BYTES)


def _boundary_ids() -> np.ndarray:
    edges = np.arange(REGION_ROWS, NUM_ROWS, REGION_ROWS, dtype=np.int64)
    return np.concatenate([edges - 1, edges])


def _wait_state(reader: PLEDiskReader, state: str, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while reader.remote_status()["state"] != state:
        assert time.monotonic() < deadline, reader.remote_status()
        time.sleep(0.01)


def test_remote_reads_bit_exact(table, start_server):
    """After the upload every row (incl. ones at region edges, page-straddling
    and out-of-range ids) comes from the server, matching the files."""
    server = start_server()
    reader = _reader(table.segments, server.address)
    try:
        assert reader.wait_remote_ready(30)
        status = reader.remote_status()
        assert status["uploaded_bytes"] == status["total_bytes"]
        assert status["total_bytes"] == NUM_ROWS * ROW_BYTES
        assert status["regions"] == -(-NUM_ROWS // REGION_ROWS)
        reader.reset_stats()
        ids = np.concatenate([_mixed_ids(table, 4000, 0), _boundary_ids()])
        np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
        stats = reader.stats()
        assert stats["reads"] == 0
        assert stats["remote_reads"] > 0
        assert stats["remote_bytes"] == stats["remote_reads"] * ROW_BYTES
        reader.check_error()
    finally:
        reader.close()
    line = server.wait_for("disconnected")
    assert "releasing 0 regions" in line


def test_reads_during_upload(table, start_server):
    """Reads issued while the table is uploading mix SSD and remote rows and
    stay exact; the watermark skips the gap between two segments."""
    server = start_server()
    gap_segments = [
        table.segments[0],
        table.segments[2],
    ]
    reader = _reader(
        gap_segments, server.address, remote={"upload_chunk_bytes": 16_000}
    )
    try:
        assert reader.remote_status()["state"] == "uploading"
        rng = np.random.default_rng(1)
        ids = rng.integers(0, NUM_ROWS, 2000, dtype=np.int64)
        expected = table.expected(ids)
        gap = (ids >= table.segments[1].row_start) & (ids < table.segments[2].row_start)
        expected[gap] = 0
        for _ in range(50):
            np.testing.assert_array_equal(_read(reader, ids), expected)
            if reader.remote_status()["state"] == "ready":
                break
        assert reader.wait_remote_ready(30)
        reader.reset_stats()
        np.testing.assert_array_equal(_read(reader, ids), expected)
        assert reader.stats()["reads"] == 0
        reader.check_error()
    finally:
        reader.close()


def test_hints_prefetch_from_remote(table, start_server):
    server = start_server()
    reader = _reader(table.segments, server.address, cache_rows=4096)
    try:
        assert reader.wait_remote_ready(30)
        ids = np.random.default_rng(3).choice(NUM_ROWS, 64, replace=False)
        reader.hint(ids)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            stats = reader.stats()
            if stats["hint_reads"] and stats["remote_reads"] >= stats["hint_reads"]:
                break
            time.sleep(0.01)
        reader.reset_stats()
        np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
        stats = reader.stats()
        assert stats["cache_hits"] == ids.size
        assert stats["reads"] == stats["remote_reads"] == 0
    finally:
        reader.close()


def test_server_killed_falls_back_to_disk(table, start_server):
    """Killing the server under load moves queued and in-flight remote reads
    to the SSD; every read stays exact and no error is recorded."""
    server = start_server()
    reader = _reader(table.segments, server.address, remote={"max_inflight": 256})
    stop = threading.Event()
    failures: list[BaseException] = []

    def read_loop(seed: int) -> None:
        ids = _mixed_ids(table, 3000, seed)
        expected = table.expected(ids)
        while not stop.is_set():
            try:
                np.testing.assert_array_equal(_read(reader, ids), expected)
            except BaseException as e:
                failures.append(e)
                return

    threads = [threading.Thread(target=read_loop, args=(s,)) for s in (5, 6)]
    try:
        assert reader.wait_remote_ready(30)
        for t in threads:
            t.start()
        time.sleep(0.2)
        server.kill()
        _wait_state(reader, "failed")
        time.sleep(0.2)
        stop.set()
        for t in threads:
            t.join(30)
        assert not failures, failures[0]
        assert reader.remote_status()["error"]
        assert reader.stats()["remote_requeued"] > 0
        reader.reset_stats()
        ids = _mixed_ids(table, 2000, 7)
        np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
        stats = reader.stats()
        assert stats["reads"] > 0 and stats["remote_reads"] == 0
        reader.check_error()
    finally:
        stop.set()
        reader.close()


def test_server_killed_without_fallback_errors(table, start_server):
    server = start_server()
    reader = _reader(table.segments, server.address, remote={"fallback_to_disk": False})
    try:
        assert reader.wait_remote_ready(30)
        server.kill()
        _wait_state(reader, "failed")
        with pytest.raises(RuntimeError, match="remote memory failed"):
            reader.check_error()
    finally:
        reader.close()


@pytest.mark.parametrize("fallback", [True, False])
def test_server_out_of_memory(table, start_server, fallback):
    """A server too small for the table fails the setup: disk-only with
    fallback, a constructor error without."""
    server = start_server("--max-memory", "1M")
    remote = {"fallback_to_disk": fallback}
    if not fallback:
        with pytest.raises(RuntimeError, match="max-memory"):
            _reader(table.segments, server.address, remote=remote)
        return
    reader = _reader(table.segments, server.address, remote=remote)
    try:
        status = reader.remote_status()
        assert status["state"] == "failed"
        assert "max-memory" in status["error"]
        assert not reader.wait_remote_ready(1)
        ids = _mixed_ids(table, 500, 7)
        np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
        assert reader.stats()["remote_reads"] == 0
        reader.check_error()
    finally:
        reader.close()
    server.wait_for("0 clients, 0 regions")


def test_unreachable_server(table, start_server):
    server = start_server()
    address = server.address
    server.kill()
    with pytest.raises(RuntimeError, match="remote memory setup failed"):
        _reader(table.segments, address, remote={"fallback_to_disk": False})
    reader = _reader(table.segments, address)
    try:
        assert reader.remote_status()["state"] == "failed"
        ids = _mixed_ids(table, 500, 8)
        np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
    finally:
        reader.close()


def test_bench_verifies_against_server(bin_dir, start_server):
    """The standalone benchmark borrows, uploads, reads back (verified) and
    releases memory; regions straddled by chunks and rows included."""
    server = start_server()
    proc = subprocess.run(
        [
            os.path.join(bin_dir, "ple_mem_bench"),
            "--server",
            server.address,
            "--bytes",
            "2M",
            "--region-bytes",
            "700001",
            "--chunk-bytes",
            "300001",
            "--depth",
            "1,32",
            "--iters",
            "2000",
            "--verify",
            "--timeout",
            "20",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **UCX_ENV},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "(verified)" in proc.stdout
    server.wait_for("0 clients, 0 regions")
