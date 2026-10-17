# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the on-demand PLE n-gram table reader (disk offload)."""

import sys
import time

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.common.ple_disk_reader import (
    PLEDiskReader,
    PLEDiskSegment,
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="PLE disk reader is Linux-only"
)

ROW_BYTES = 160
NUM_ROWS = 200_000
SPLIT = 120_000  # rows [0, SPLIT) in file A (two segments), the rest in B
SEG_A2 = 70_001
HEADER_A = 1000
HEADER_B = 333


class _Table:
    def __init__(self, root):
        rng = np.random.default_rng(0)
        self.data = rng.integers(0, 256, (NUM_ROWS, ROW_BYTES), dtype=np.uint8)
        self.path_a = str(root / "a.bin")
        self.path_b = str(root / "b.bin")
        with open(self.path_a, "wb") as f:
            f.write(b"\x7f" * HEADER_A)
            f.write(self.data[:SPLIT].tobytes())
        with open(self.path_b, "wb") as f:
            f.write(b"\x7f" * HEADER_B)
            f.write(self.data[SPLIT:].tobytes())
        self.segments = [
            PLEDiskSegment(self.path_a, HEADER_A, 0, SEG_A2),
            PLEDiskSegment(
                self.path_a,
                HEADER_A + SEG_A2 * ROW_BYTES,
                SEG_A2,
                SPLIT - SEG_A2,
            ),
            PLEDiskSegment(self.path_b, HEADER_B, SPLIT, NUM_ROWS - SPLIT),
        ]

    def expected(self, ids: np.ndarray) -> np.ndarray:
        valid = (ids >= 0) & (ids < NUM_ROWS)
        out = np.zeros((ids.size, ROW_BYTES), dtype=np.uint8)
        out[valid] = self.data[ids[valid]]
        return out

    def straddling_rows(self, count: int) -> np.ndarray:
        rows = np.arange(NUM_ROWS, dtype=np.int64)
        base = np.where(rows < SPLIT, HEADER_A, HEADER_B - SPLIT * ROW_BYTES)
        start = base + rows * ROW_BYTES
        mask = start // 4096 != (start + ROW_BYTES - 1) // 4096
        return rows[mask][:count]


@pytest.fixture(scope="module")
def table(tmp_path_factory) -> _Table:
    return _Table(tmp_path_factory.mktemp("ple_table"))


def _reader(table: _Table, **kwargs) -> PLEDiskReader:
    args = dict(
        cache_rows=0,
        io_threads=8,
        max_inflight=64,
        direct_io=True,
        wait_timeout_s=30.0,
    )
    args.update(kwargs)
    return PLEDiskReader(table.segments, ROW_BYTES, **args)


def _read(reader: PLEDiskReader, ids: np.ndarray) -> np.ndarray:
    out = torch.empty(ids.size * ROW_BYTES, dtype=torch.uint8)
    reader.read_rows(torch.from_numpy(ids), out)
    return out.numpy().reshape(ids.size, ROW_BYTES)


def _mixed_ids(table: _Table, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, NUM_ROWS, n, dtype=np.int64)
    special = np.array(
        [
            -1,
            -(2**40),
            NUM_ROWS,
            NUM_ROWS + 7,
            0,
            NUM_ROWS - 1,
            SEG_A2 - 1,
            SEG_A2,
            SPLIT - 1,
            SPLIT,
        ],
        dtype=np.int64,
    )
    ids = np.concatenate([ids, special, table.straddling_rows(32), ids[:16]])
    return rng.permutation(ids)


@pytest.mark.parametrize("cache_rows", [0, 1024])
@pytest.mark.parametrize("direct_io", [True, False])
def test_read_rows_bit_exact(table, cache_rows, direct_io):
    """Rows across files/segments, straddling pages, duplicates and
    out-of-range ids match the source table; O_DIRECT may fall back."""
    reader = _reader(table, cache_rows=cache_rows, direct_io=direct_io)
    try:
        for seed in range(2):
            ids = _mixed_ids(table, 2000, seed)
            np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
        small = ids[:50]
        for _ in range(2):
            np.testing.assert_array_equal(_read(reader, small), table.expected(small))
        stats = reader.stats()
        if not direct_io:
            assert stats["direct_io"] is False
        assert stats["requests"] > 0 and stats["dedup_rows"] > 0
        if cache_rows:
            assert stats["cache_hits"] > 0
        reader.check_error()
    finally:
        reader.close()
        reader.close()


def test_hint_fills_cache(table):
    reader = _reader(table, cache_rows=4096)
    try:
        ids = np.random.default_rng(3).choice(NUM_ROWS, 64, replace=False)
        reader.hint(ids)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            stats = reader.stats()
            if stats["hint_reads"] > 0 and stats["reads"] >= stats["hint_reads"]:
                break
            time.sleep(0.01)
        reader.reset_stats()
        np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
        stats = reader.stats()
        assert stats["cache_hits"] == ids.size
        assert stats["reads"] == 0
    finally:
        reader.close()


def test_cache_keeps_hot_row(table):
    """CLOCK replacement: a row hit between every cold insert stays resident
    in a single 8-way set even after many more cold rows than ways."""
    reader = _reader(table, cache_rows=8)
    try:
        hot = np.array([7], dtype=np.int64)
        _read(reader, hot)
        reader.reset_stats()
        cold = np.arange(1000, NUM_ROWS, 4999, dtype=np.int64)[:32]
        for row in cold:
            np.testing.assert_array_equal(
                _read(reader, np.array([row])), table.expected(np.array([row]))
            )
            np.testing.assert_array_equal(_read(reader, hot), table.expected(hot))
        stats = reader.stats()
        assert stats["cache_hits"] == cold.size
        assert stats["reads"] == cold.size
    finally:
        reader.close()


def test_large_batch(table):
    reader = _reader(table, cache_rows=8192, io_threads=16, max_inflight=256)
    try:
        ids = _mixed_ids(table, 100_000, 7)
        reader.hint(ids[: ids.size // 2])
        np.testing.assert_array_equal(_read(reader, ids), table.expected(ids))
        reader.check_error()
    finally:
        reader.close()


def _bound_buffers(reader: PLEDiskReader, max_tokens: int, heads: int):
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    ids = torch.zeros(max_tokens, heads, dtype=torch.int64, pin_memory=True)
    num = torch.zeros(1, dtype=torch.int64, pin_memory=True)
    out = torch.zeros(max_tokens, heads * ROW_BYTES, dtype=torch.uint8, pin_memory=True)
    reader.bind_demand(ids, num, out)
    views = tuple(get_accelerator_view_from_cpu_tensor(t) for t in (ids, num, out))
    return (ids, num, out), views


def _expected_tokens(table: _Table, ids: np.ndarray) -> np.ndarray:
    return table.expected(ids.reshape(-1)).reshape(ids.shape[0], -1)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@requires_cuda
def test_notify_wait_on_stream(table):
    max_tokens, heads, n = 64, 16, 40
    reader = _reader(table, cache_rows=1024)
    try:
        _, (ids_v, num_v, out_v) = _bound_buffers(reader, max_tokens, heads)
        ids = _mixed_ids(table, n * heads, 11)[: n * heads].reshape(n, heads)
        src = torch.from_numpy(ids).cuda()
        stream = torch.cuda.current_stream()
        ids_v[:n].copy_(src)
        num_v.fill_(n)
        reader.launch_notify(stream)
        reader.launch_wait(stream)
        result = out_v[:n].clone()
        torch.accelerator.synchronize()
        np.testing.assert_array_equal(
            result.cpu().numpy(), _expected_tokens(table, ids)
        )
        reader.check_error()
    finally:
        reader.close()


@requires_cuda
def test_notify_wait_cuda_graph(table):
    """Captured host nodes act on the current buffer contents per replay."""
    max_tokens, heads = 32, 16
    reader = _reader(table, cache_rows=1024)
    try:
        _, (ids_v, num_v, out_v) = _bound_buffers(reader, max_tokens, heads)
        static_ids = torch.zeros(max_tokens, heads, dtype=torch.int64, device="cuda")
        static_num = torch.zeros(1, dtype=torch.int64, device="cuda")
        static_out = torch.zeros(
            max_tokens, heads * ROW_BYTES, dtype=torch.uint8, device="cuda"
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ids_v.copy_(static_ids)
            num_v.copy_(static_num)
            stream = torch.cuda.current_stream()
            reader.launch_notify(stream)
            reader.launch_wait(stream)
            static_out.copy_(out_v)
        for seed, n in ((21, 32), (22, 9)):
            ids = _mixed_ids(table, max_tokens * heads, seed)
            ids = ids[: max_tokens * heads].reshape(max_tokens, heads)
            static_ids.copy_(torch.from_numpy(ids))
            static_num.fill_(n)
            graph.replay()
            torch.accelerator.synchronize()
            np.testing.assert_array_equal(
                static_out[:n].cpu().numpy(), _expected_tokens(table, ids[:n])
            )
        reader.check_error()
    finally:
        reader.close()


def test_truncated_file_reports_error(tmp_path):
    """Reads past a truncated file fail without hanging any waiter."""
    rows = 1000
    data = np.random.default_rng(5).integers(0, 256, (rows, ROW_BYTES), np.uint8)
    path = tmp_path / "short.bin"
    path.write_bytes(b"\0" * 17 + data[: rows // 2].tobytes())
    seg = PLEDiskSegment(str(path), 17, 0, rows)
    reader = PLEDiskReader(
        [seg],
        ROW_BYTES,
        cache_rows=0,
        io_threads=4,
        max_inflight=16,
        direct_io=True,
        wait_timeout_s=10.0,
    )
    try:
        good = np.arange(0, 100, dtype=np.int64)
        out = torch.empty(good.size * ROW_BYTES, dtype=torch.uint8)
        reader.read_rows(torch.from_numpy(good), out)
        np.testing.assert_array_equal(out.numpy().reshape(-1, ROW_BYTES), data[good])
        reader.check_error()

        bad = np.array([1, rows - 1], dtype=np.int64)
        out = torch.empty(bad.size * ROW_BYTES, dtype=torch.uint8)
        with pytest.raises(RuntimeError):
            reader.read_rows(torch.from_numpy(bad), out)
        with pytest.raises(RuntimeError):
            reader.check_error()

        if torch.cuda.is_available():
            (ids, num, out), _ = _bound_buffers(reader, 4, 2)
            ids.fill_(rows - 1)
            num.fill_(4)
            stream = torch.cuda.current_stream()
            start = time.monotonic()
            reader.launch_notify(stream)
            reader.launch_wait(stream)
            torch.accelerator.synchronize()
            assert time.monotonic() - start < 5
    finally:
        reader.close()
