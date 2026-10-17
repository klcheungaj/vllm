# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the PLE reader's remote memory source and the standalone memory
server / benchmark (``ple_mem_server/``).

Transport-agnostic behavior runs against the in-process ``loopback`` backend,
and against UCX when available. The UCX server runs as a subprocess over UCX's
tcp transport (explicitly allowed with ``--allow-tcp`` /
``allow_insecure_transport``), so no RDMA hardware is needed; UCX tests are
skipped when UCX, make or g++ is missing.
"""

import dataclasses
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest
import regex as re
import torch

import vllm.models.qwen4_exp.common as qwen4_common
import vllm.models.qwen4_exp.common.ple_disk_reader as reader_module
from vllm.models.qwen4_exp.common.ple_disk_reader import (
    REMOTE_BACKENDS,
    PLEDiskReader,
    PLERemoteOptions,
    available_remote_backends,
    load_extension,
    ucx_build_flags,
)

from .test_ple_disk_reader import HEADER_A, NUM_ROWS, ROW_BYTES, _mixed_ids, _Table

SERVER_DIR = os.path.join(os.path.dirname(qwen4_common.__file__), "ple_mem_server")


def _ucx_missing() -> str:
    if not (shutil.which("make") and shutil.which("g++")):
        return "make and g++ are required"
    try:
        ucx_build_flags()
    except RuntimeError as e:
        return str(e)
    return ""


pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="PLE disk reader is Linux-only"
)
_UCX_MISSING = _ucx_missing()
requires_ucx = pytest.mark.skipif(bool(_UCX_MISSING), reason=_UCX_MISSING)

REGION_BYTES = 100_003  # 625 rows, not aligned to pages, chunks or segments
REGION_ROWS = REGION_BYTES // ROW_BYTES
CHUNK_BYTES = 50_000  # 312 rows per staged upload
UCX_ENV = {"UCX_TLS": "tcp", "UCX_WARN_UNUSED_ENV_VARS": "n"}


class _Server:
    def __init__(self, bin_dir: str, *args: str, allow_tcp: bool = True) -> None:
        binary = os.path.join(bin_dir, "ple_mem_server")
        tcp = ["--allow-tcp"] if allow_tcp else []
        self.proc = subprocess.Popen(
            [binary, "--host", "127.0.0.1", "--port", "0", *tcp, *args],
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

    def start(*args: str, **kwargs) -> _Server:
        servers.append(_Server(bin_dir, *args, **kwargs))
        return servers[-1]

    yield start
    for server in servers:
        server.kill()


@dataclass
class _Remote:
    backend: str
    address: str
    server: _Server | None = None


@pytest.fixture(params=["loopback", pytest.param("ucx", marks=requires_ucx)])
def remote(request) -> _Remote:
    if request.param == "loopback":
        return _Remote("loopback", "loopback")
    server = request.getfixturevalue("start_server")()
    return _Remote("ucx", server.address, server)


def _reader(segments, address: str, backend: str = "ucx", **kwargs) -> PLEDiskReader:
    remote = dict(
        region_bytes=REGION_BYTES,
        upload_chunk_bytes=CHUNK_BYTES,
        max_inflight=64,
        connect_timeout_s=10.0,
        allow_insecure_transport=True,
    )
    remote.update(kwargs.pop("remote", {}))
    args = dict(
        cache_rows=0,
        io_threads=8,
        max_inflight=64,
        direct_io=True,
        wait_timeout_s=30.0,
        remote=PLERemoteOptions(address, backend, **remote),
    )
    args.update(kwargs)
    return PLEDiskReader(segments, ROW_BYTES, **args)


def _read(reader: PLEDiskReader, ids: np.ndarray) -> np.ndarray:
    out = torch.empty(ids.size * ROW_BYTES, dtype=torch.uint8)
    reader.read_rows(torch.from_numpy(ids), out)
    return out.numpy().reshape(ids.size, ROW_BYTES)


def _loopback(segments, query: str = "", **kwargs) -> PLEDiskReader:
    return _reader(segments, "loopback" + query, "loopback", **kwargs)


def _boundary_ids(region_rows: int = REGION_ROWS) -> np.ndarray:
    edges = np.arange(region_rows, NUM_ROWS, region_rows, dtype=np.int64)
    return np.concatenate([edges - 1, edges])


def _wait_state(reader: PLEDiskReader, state: str, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while reader.remote_status()["state"] != state:
        assert time.monotonic() < deadline, reader.remote_status()
        time.sleep(0.01)


def test_remote_reads_bit_exact(table, remote):
    """After the upload every row (incl. ones at region edges, page-straddling
    and out-of-range ids) comes from the server, matching the files."""
    reader = _reader(table.segments, remote.address, remote.backend)
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
        assert 0 < stats["remote_requests"] <= stats["remote_reads"]
        reader.check_error()
    finally:
        reader.close()
    if remote.server is not None:
        line = remote.server.wait_for("disconnected")
        assert "releasing 0 regions" in line


def test_reads_during_upload(table, remote):
    """Reads issued while the table is uploading mix SSD and remote rows and
    stay exact; the watermark skips the gap between two segments."""
    address = remote.address
    if remote.backend == "loopback":
        address += "?delay_us=200"  # keeps the upload running for a while
    gap_segments = [
        table.segments[0],
        table.segments[2],
    ]
    reader = _reader(
        gap_segments, address, remote.backend, remote={"upload_chunk_bytes": 16_000}
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


def test_hints_prefetch_from_remote(table, remote):
    reader = _reader(table.segments, remote.address, remote.backend, cache_rows=4096)
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


def test_chunks_spanning_many_regions(table):
    """Regions much smaller than an upload chunk (and not aligned to rows):
    each chunk is split into several puts, rows at every edge stay exact."""
    region_bytes = 7_001
    region_rows = region_bytes // ROW_BYTES
    reader = _loopback(table.segments, remote={"region_bytes": region_bytes})
    try:
        assert reader.wait_remote_ready(30)
        assert reader.remote_status()["regions"] == -(-NUM_ROWS // region_rows)
        reader.reset_stats()
        ids = _boundary_ids(region_rows)
        np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
        assert reader.stats()["reads"] == 0
    finally:
        reader.close()


# The upload of the test table posts fewer than 1000 puts (643 chunks plus
# region straddles), so 100 fails it mid-upload and 2000 under reads.
@pytest.mark.parametrize("fallback", [True, False])
@pytest.mark.parametrize("fail_after", [100, 2000], ids=["upload", "reads"])
def test_transport_failure(table, fail_after, fallback):
    """A transport failing mid-upload or under reads fails the remote source.
    With fallback, in-flight reads move to the SSD and stay exact; without,
    they fail and the error is recorded. Later reads come from the SSD."""
    reader = _loopback(
        table.segments,
        f"?fail_after={fail_after}",
        remote={"fallback_to_disk": fallback},
    )
    try:
        ids = _mixed_ids(table, 3000, 9)
        if fail_after == 2000:
            assert reader.wait_remote_ready(30)
            if fallback:
                np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
                assert reader.stats()["remote_requeued"] > 0
            else:
                with pytest.raises(RuntimeError, match="remote memory failed"):
                    _read(reader, ids)
        _wait_state(reader, "failed")
        assert "injected failure" in reader.remote_status()["error"]
        reader.reset_stats()
        np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
        stats = reader.stats()
        assert stats["reads"] > 0 and stats["remote_reads"] == 0
        if fallback:
            reader.check_error()
        else:
            with pytest.raises(RuntimeError, match="remote memory failed"):
                reader.check_error()
    finally:
        reader.close()


@pytest.mark.parametrize("fallback", [True, False])
@pytest.mark.parametrize(
    "query, error",
    [("?refuse=1", "refused 320 regions"), ("?bogus=1", "invalid loopback address")],
)
def test_setup_failure(table, query, error, fallback):
    """A failed setup leaves the reader disk-only with fallback and fails the
    constructor without."""
    remote = {"fallback_to_disk": fallback}
    if not fallback:
        with pytest.raises(
            RuntimeError, match=f"remote memory setup failed: .*{error}"
        ):
            _loopback(table.segments, query, remote=remote)
        return
    reader = _loopback(table.segments, query, remote=remote)
    try:
        status = reader.remote_status()
        assert status["state"] == "failed"
        assert error in status["error"]
        assert not reader.wait_remote_ready(1)
        ids = _mixed_ids(table, 500, 7)
        np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
        assert reader.stats()["remote_reads"] == 0
        reader.check_error()
    finally:
        reader.close()


def test_close_while_uploading(table):
    """Closing mid-upload with remote reads in flight stops the remote and
    upload threads promptly; reads still pending fail as closed."""
    reader = _loopback(
        table.segments, "?delay_us=2000", remote={"upload_chunk_bytes": 16_000}
    )
    errors: list[RuntimeError] = []

    def read_loop() -> None:
        ids = _mixed_ids(table, 2000, 10)
        expected = table.expected(ids)
        try:
            while True:
                np.testing.assert_array_equal(_read(reader, ids), expected)
        except RuntimeError as e:
            errors.append(e)

    thread = threading.Thread(target=read_loop)
    try:
        thread.start()
        deadline = time.monotonic() + 10
        while reader.stats()["remote_reads"] == 0:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert reader.remote_status()["state"] == "uploading"
        start = time.monotonic()
        reader.close()
        assert time.monotonic() - start < 5
        thread.join(10)
        assert not thread.is_alive()
        assert errors and "closed" in str(errors[0])
    finally:
        reader.close()
        thread.join(10)


def test_backend_selection(table):
    """The remote variant compiles in every available backend and the reader
    picks one by name at startup; unknown names, and backends missing from
    the variant, are rejected."""
    assert REMOTE_BACKENDS == ("ucx",)
    available = available_remote_backends()
    assert "loopback" in available
    assert ("ucx" in available) == (not _UCX_MISSING)
    assert sorted(load_extension(remote=True).REMOTE_BACKENDS) == list(available)
    with pytest.raises(ValueError, match="supported: ucx, loopback"):
        _reader(table.segments, "nope", "nope")
    disk_only = load_extension()
    assert disk_only.REMOTE_BACKENDS == ()
    args = [[table.path_a], [0], [HEADER_A], [0], [10], ROW_BYTES, 0, 1, 1, False]
    remote = [1.0, "loopback", "loopback", 1, ROW_BYTES, 1, 1.0, True, 1, "", False]
    with pytest.raises(ValueError, match=r"not compiled into this reader .*none"):
        disk_only.Reader(*args, *remote)


def test_ucx_requested_without_ucx(table, monkeypatch):
    """Without UCX the remote variant holds only loopback (built under its own
    name), and a reader asking for ucx fails with the missing dependency."""
    ucx = dataclasses.replace(
        reader_module._REMOTE_BACKENDS["ucx"], missing=lambda: "UCX was not found"
    )
    monkeypatch.setitem(reader_module._REMOTE_BACKENDS, "ucx", ucx)
    assert available_remote_backends() == ("loopback",)
    assert load_extension(remote=True).REMOTE_BACKENDS == ("loopback",)
    with pytest.raises(RuntimeError, match="'ucx' is not available: UCX was not"):
        _reader(table.segments, "127.0.0.1:1", "ucx")


@requires_ucx
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


@requires_ucx
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


@requires_ucx
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


@requires_ucx
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


@requires_ucx
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
            "--batch-rows",
            "1,64",
            "--depth",
            "1,8",
            "--iters",
            "500",
            "--ping-iters",
            "100",
            "--step-rows",
            "16,300",
            "--step-batch",
            "64",
            "--step-iters",
            "20",
            "--verify",
            "--timeout",
            "20",
            "--allow-tcp",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **UCX_ENV},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "(verified)" in proc.stdout
    latency_rows = [line.split()[:4] for line in proc.stdout.splitlines()]
    assert ["ping", "1", "1", "100"] in latency_rows
    assert ["step", "300", "64", "20"] in latency_rows
    server.wait_for("0 clients, 0 regions")


@pytest.mark.parametrize(
    "corrupt, error",
    [(1, "wrong data length"), (2, "unsolicited read reply")],
    ids=["short", "unsolicited"],
)
def test_malformed_reply_fails_remote(table, corrupt, error):
    """A malformed read reply is a protocol violation: the transport fails
    without touching the destination rows, and every read stays exact from
    the SSD."""
    reader = _loopback(table.segments, f"?corrupt={corrupt}")
    try:
        assert reader.wait_remote_ready(30)
        ids = _mixed_ids(table, 2000, 12)
        np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
        status = reader.remote_status()
        assert status["state"] == "failed"
        assert "protocol violation" in status["error"]
        assert error in status["error"]
        reader.check_error()
    finally:
        reader.close()


@requires_ucx
@pytest.mark.parametrize("fault", ["short-reply", "unsolicited"])
def test_ucx_client_rejects_malformed_replies(table, start_server, fault):
    """The UCX client validates every reply of a misbehaving server and falls
    back to the SSD; no bad row reaches the caller."""
    server = start_server("--inject-fault", fault)
    reader = _reader(table.segments, server.address)
    try:
        assert reader.wait_remote_ready(30)
        ids = _mixed_ids(table, 2000, 13)
        np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
        status = reader.remote_status()
        assert status["state"] == "failed"
        assert "protocol violation" in status["error"]
        reader.check_error()
    finally:
        reader.close()


@requires_ucx
def test_server_rejects_out_of_bounds_and_foreign_regions(bin_dir, start_server):
    """Reads and writes past a region's end, or of another client's region,
    are refused by the server (ple_mem_bench --probe)."""
    server = start_server()
    proc = subprocess.run(
        [
            os.path.join(bin_dir, "ple_mem_bench"),
            "--server",
            server.address,
            "--bytes",
            "64K",
            "--region-bytes",
            "64K",
            "--batch-rows",
            "1",
            "--depth",
            "1",
            "--iters",
            "10",
            "--ping-iters",
            "0",
            "--step-iters",
            "0",
            "--probe",
            "--allow-tcp",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **UCX_ENV},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    probes = [line for line in proc.stdout.splitlines() if line.startswith("probe")]
    assert len(probes) == 5 and all("rejected" in p for p in probes), probes
    server.wait_for("0 clients, 0 regions")


@requires_ucx
def test_token(table, start_server, tmp_path):
    token = tmp_path / "token"
    token.write_text("s3cret\n")
    wrong = tmp_path / "wrong"
    wrong.write_text("nope")
    server = start_server("--token-file", str(token))
    for token_file in (str(wrong), None):
        with pytest.raises(RuntimeError, match="authentication failed"):
            _reader(
                table.segments,
                server.address,
                remote={"token_file": token_file, "fallback_to_disk": False},
            )
    reader = _reader(table.segments, server.address, remote={"token_file": str(token)})
    try:
        assert reader.wait_remote_ready(30)
        ids = _mixed_ids(table, 500, 14)
        np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
    finally:
        reader.close()


@requires_ucx
def test_tcp_refused_without_opt_in(table, bin_dir, start_server):
    """With only UCX's tcp transport available, both the client and the
    server refuse to run unless tcp is explicitly allowed."""
    server = start_server()
    with pytest.raises(RuntimeError, match="selects only the tcp transport"):
        _reader(
            table.segments,
            server.address,
            remote={"allow_insecure_transport": False, "fallback_to_disk": False},
        )
    proc = subprocess.run(
        [os.path.join(bin_dir, "ple_mem_server"), "--host", "127.0.0.1"],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, **UCX_ENV},
    )
    assert proc.returncode != 0
    assert "selects only the tcp transport" in proc.stderr


def _bench(bin_dir: str, address: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            os.path.join(bin_dir, "ple_mem_bench"),
            "--server",
            address,
            "--allow-tcp",
            "--row-bytes",
            "1",
            "--batch-rows",
            "1",
            "--depth",
            "1",
            "--iters",
            "10",
            "--ping-iters",
            "0",
            "--step-iters",
            "0",
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, **UCX_ENV},
    )


@requires_ucx
def test_server_limits(bin_dir, start_server):
    """--max-memory charges mapped pages (a 1-byte region costs 4 KiB) and the
    region count per client is capped."""
    server = start_server("--max-memory", "64K", "--max-client-regions", "20")
    ok = _bench(bin_dir, server.address, "--bytes", "16", "--region-bytes", "1")
    assert ok.returncode == 0, ok.stderr
    over = _bench(bin_dir, server.address, "--bytes", "17", "--region-bytes", "1")
    assert over.returncode != 0 and "max-memory" in over.stderr
    server = start_server("--max-client-regions", "4")
    over = _bench(bin_dir, server.address, "--bytes", "5", "--region-bytes", "1")
    assert over.returncode != 0 and "region limit" in over.stderr


@requires_ucx
@pytest.mark.parametrize(
    "args, error",
    [
        ([], "--host is required"),
        (["--host", "127.0.0.1", "--max-memory", "0"], "invalid --max-memory"),
    ],
)
def test_server_cli_errors(bin_dir, args, error):
    proc = subprocess.run(
        [os.path.join(bin_dir, "ple_mem_server"), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 2
    assert error in proc.stderr
