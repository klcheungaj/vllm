# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk offload of the Qwen4Exp PLE table."""

import json
from itertools import accumulate
from types import SimpleNamespace
from typing import get_args

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.model_loader.weight_utils as weight_utils
import vllm.model_executor.parameter as parameter_module
import vllm.models.qwen4_exp.common.ngram_embedding as ngram_embedding_module
import vllm.models.qwen4_exp.common.ple_disk as ple_disk_module
from vllm.config.engram import EngramConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)
from vllm.models.qwen4_exp.common.ple_disk import (
    Qwen4ExpPLEDiskEmbedding,
    find_ple_disk_segments,
    is_ple_shard_checkpoint_weight,
    ple_ngram_row_ids_np,
)
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding
from vllm.utils.platform_utils import is_uva_available

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
requires_uva = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_uva_available(), reason="needs UVA"
)

_SHARD_NAME = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
_EMBEDDING_PREFIX = "model.language_model.model.layers.1.ple.ple_embedding"
_NUM_ROWS = 1000
_HEAD_DIM = 16
_NUM_HEADS = 4
_SPLIT_PARTS = 3
_MAX_TOKENS = 32


def _hash_params(ngram_size: int, heads_per_ngram: int) -> dict:
    """Realistic multipliers and prime-sized head partitions."""
    num_heads = (ngram_size - 1) * heads_per_ngram
    multipliers = Qwen4ExpNGramEmbedding._make_layer_multipliers(
        ngram_size=ngram_size,
        unigram_vocab_size=248_320,
        seed=1234,
        ple_dense_layer_id=0,
    )
    sizes, offsets, _ = Qwen4ExpNGramEmbedding._make_vocab_layout(
        ngram_vocab_size_base=1000,
        ngram_heads=num_heads,
        ple_dense_layer_id=0,
    )
    return {
        "layer_multipliers": np.array(multipliers, dtype=np.int64),
        "ngram_heads_vocab_sizes": np.array(sizes, dtype=np.int64),
        "ngram_heads_offsets": np.array(offsets, dtype=np.int64),
        "eos_token_id": 248_044,
        "heads_per_ngram": heads_per_ngram,
    }


@pytest.mark.parametrize(
    "name, expected",
    [
        (f"{_SHARD_NAME}.shard_7.weight", True),
        ("model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", True),
        (f"{_SHARD_NAME}.weight_scale", False),
        ("model.language_model.layers.1.ple.ple_embedding.layer_multipliers", False),
        ("mtp.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight", False),
    ],
)
def test_ple_shard_names_select_only_backbone_table_rows(name, expected):
    """Hash buffers, scales and draft layers must still load from the file."""
    assert is_ple_shard_checkpoint_weight(name) is expected


def test_loader_never_reads_skipped_weights(tmp_path, monkeypatch):
    """The model hook keeps skipped tensors from being materialized at all."""
    save_file(
        {"kept.weight": torch.ones(2), "skipped.weight": torch.zeros(2)},
        str(tmp_path / "model.safetensors"),
    )
    read: list[str] = []
    real_safe_open = weight_utils.safe_open

    class SpySafeOpen:
        def __init__(self, *args, **kwargs):
            self._file = real_safe_open(*args, **kwargs)

        def __enter__(self):
            file = self._file.__enter__()
            real_get_tensor = file.get_tensor

            def get_tensor(name: str) -> torch.Tensor:
                read.append(name)
                return real_get_tensor(name)

            return SimpleNamespace(keys=file.keys, get_tensor=get_tensor)

        def __exit__(self, *exc):
            return self._file.__exit__(*exc)

    monkeypatch.setattr(weight_utils, "safe_open", SpySafeOpen)
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    model = SimpleNamespace(should_skip_checkpoint_weight=lambda n: n.startswith("s"))
    model_config = SimpleNamespace(model=str(tmp_path), revision=None)

    names = [name for name, _ in loader.get_all_weights(model_config, model)]

    assert names == ["kept.weight"]
    assert read == ["kept.weight"]


@pytest.mark.parametrize("ngram_size", [3, 4])
@requires_cuda
def test_numpy_row_ids_match_triton_kernel(ngram_size):
    """Prefetch hints hash on the CPU and must name the rows the GPU reads."""
    from vllm.models.qwen4_exp.nvidia.ops.ple import ple_ngram_ids

    params = _hash_params(ngram_size, heads_per_ngram=2)
    eos = params["eos_token_id"]
    context_len = ngram_size - 1
    generator = np.random.default_rng(0)
    prompts = [generator.integers(0, 248_320, size=n) for n in (37, 5, 64)]
    prompts[0][[0, 9, 10, 30]] = eos
    prompts[2][[1, 2, 40]] = eos
    # Chunks start mid-prompt so their predecessors come from ngram_context.
    chunks = [(4, 37), (0, 5), (17, 50)]

    expected = []
    context = []
    input_ids = []
    for tokens, (start, end) in zip(prompts, chunks):
        expected.append(ple_ngram_row_ids_np(tokens, start, end, **params))
        history = [
            tokens[pos] if pos >= 0 else eos
            for pos in range(start - context_len, start)
        ]
        context.append(history)
        input_ids.extend(tokens[start:end])
    query_lens = [end - start for start, end in chunks]
    device = torch.device("cuda")
    actual = ple_ngram_ids(
        input_ids=torch.tensor(input_ids, dtype=torch.int32, device=device),
        query_start_loc=torch.tensor(
            [0, *accumulate(query_lens)], dtype=torch.int32, device=device
        ),
        ngram_context=torch.tensor(context, dtype=torch.int32, device=device),
        layer_multipliers=torch.from_numpy(params["layer_multipliers"]).to(device),
        ngram_heads_vocab_sizes=torch.from_numpy(params["ngram_heads_vocab_sizes"]).to(
            device
        ),
        ngram_heads_offsets=torch.from_numpy(params["ngram_heads_offsets"]).to(device),
        eos_token_id=eos,
        heads_per_ngram=params["heads_per_ngram"],
    )

    np.testing.assert_array_equal(actual.cpu().numpy(), np.concatenate(expected))


def test_prefill_hints_cover_this_and_next_chunk():
    """Hints span this chunk plus a lookahead chunk, clipped to the prompt."""
    params = _hash_params(3, heads_per_ngram=2)
    hinted: list[np.ndarray] = []
    errors_checked: list[bool] = []
    embedding = SimpleNamespace(
        hint_rows=hinted.append,
        check_error=lambda: errors_checked.append(True),
    )
    state = Qwen4ExpModelState.__new__(Qwen4ExpModelState)
    state.ple_disk_hints = True
    state.ple_disk_layers = [(embedding, params)]
    state.ple_disk_layers_resolved = True
    state.ple_disk_stats_interval_s = 0.0
    state.ple_disk_stats_logged_at = 0.0
    tokens = np.arange(100, 140, dtype=np.int64)
    state.prefill_token_ids = {"prefill": tokens, "decode": tokens}
    batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["decode", "prefill"],
        is_prefilling_np=np.array([False, True]),
        num_computed_prefill_tokens_np=np.array([40, 16], dtype=np.int32),
        num_scheduled_tokens=np.array([1, 16], dtype=np.int32),
    )

    state._service_ple_disk(batch)

    assert errors_checked == [True]
    assert len(hinted) == 1
    np.testing.assert_array_equal(
        hinted[0], ple_ngram_row_ids_np(tokens, 16, 40, **params)
    )


def _write_checkpoint(path, table: torch.Tensor, missing_shard: int | None = None):
    """Split a table into checkpoint shards across two files plus an index."""
    shard_size = -(-_NUM_ROWS // _SPLIT_PARTS)
    files: list[dict[str, torch.Tensor]] = [
        {"model.language_model.embed_tokens.weight": torch.ones(2, 2)},
        {f"{_SHARD_NAME}.weight_scale": torch.tensor([0.5], dtype=torch.bfloat16)},
    ]
    for shard in range(_SPLIT_PARTS):
        if shard == missing_shard:
            continue
        rows = table[shard * shard_size : (shard + 1) * shard_size].contiguous()
        files[shard % 2][f"{_SHARD_NAME}.shard_{shard}.weight"] = rows
    weight_map = {}
    for index, tensors in enumerate(files):
        file_name = f"model-{index:05d}.safetensors"
        save_file(tensors, str(path / file_name))
        weight_map.update(dict.fromkeys(tensors, file_name))
    with open(path / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": weight_map}, f)


def test_missing_checkpoint_shard_is_rejected(tmp_path):
    table = torch.randn(_NUM_ROWS, _HEAD_DIM).to(torch.bfloat16)
    _write_checkpoint(tmp_path, table, missing_shard=1)

    with pytest.raises(ValueError, match="cover 334 of 1000 rows"):
        find_ple_disk_segments(
            str(tmp_path),
            layer_idx=1,
            num_rows=_NUM_ROWS,
            embedding_dim=_HEAD_DIM,
            dtype=torch.bfloat16,
            split_ngram_parts=_SPLIT_PARTS,
        )


@pytest.fixture
def single_rank(monkeypatch):
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(ngram_embedding_module, "get_etp_group", lambda: group)
    monkeypatch.setattr(
        ngram_embedding_module, "get_tp_group", lambda: SimpleNamespace(world_size=1)
    )
    for module in (embedding_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    engram_config = EngramConfig(disk_offload=True, disk_row_cache_rows=64)
    monkeypatch.setattr(
        ple_disk_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(engram_config=engram_config),
    )


def _make_disk_embedding(
    tmp_path, fp8: bool
) -> tuple[Qwen4ExpPLEDiskEmbedding, torch.Tensor]:
    """Build a disk embedding over a freshly written tiny checkpoint."""
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    table = torch.randn(_NUM_ROWS, _HEAD_DIM).to(dtype)
    _write_checkpoint(tmp_path, table)
    method = (
        Qwen4ExpPLEFp8EmbeddingMethod()
        if fp8
        else Qwen4ExpPLEUnquantizedEmbeddingMethod()
    )
    with torch.device("cuda"):
        embedding = Qwen4ExpPLEDiskEmbedding(
            _NUM_ROWS,
            _HEAD_DIM,
            params_dtype=torch.bfloat16,
            padding_size=1,
            prefix=f"{_EMBEDDING_PREFIX}.ngram_embedding",
            embedding_method=method,
            num_ngram_heads=_NUM_HEADS,
            max_total_tokens=_MAX_TOKENS,
        )
    if fp8:
        embedding.weight_scale.data.fill_(0.5)
    embedding.open_disk_source(str(tmp_path), split_ngram_parts=_SPLIT_PARTS)
    return embedding, table


def _expected_rows(table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return table[ids.cpu()].flatten(-2)


@requires_uva
@pytest.mark.parametrize("fp8", [False, True])
def test_disk_embedding_reads_checkpoint_rows(tmp_path, single_rank, fp8):
    """Rows match the checkpoint and the table is never a parameter."""
    embedding, table = _make_disk_embedding(tmp_path, fp8)
    params = dict(embedding.named_parameters())
    assert "weight" not in params
    assert ("weight_scale" in params) is fp8

    num_tokens = 11
    ids = torch.randint(0, _NUM_ROWS, (num_tokens, _NUM_HEADS), device="cuda")
    buffer = embedding.get_prefetch_ids_buffer(_MAX_TOKENS, _NUM_HEADS)
    buffer[:num_tokens].copy_(ids)
    query_start_loc = torch.tensor([0, 4, num_tokens], device="cuda")
    hidden_states = torch.empty(num_tokens, 1, device="cuda")

    embedding.start_prefetch(None, buffer[:num_tokens], query_start_loc=query_start_loc)
    rows = embedding(hidden_states)
    dequantized = embedding.dequantize(rows, torch.bfloat16)
    torch.accelerator.synchronize()

    expected = _expected_rows(table, ids)
    assert rows.dtype == table.dtype
    assert torch.equal(rows.cpu().view(torch.uint8), expected.view(torch.uint8))
    scale = 0.5 if fp8 else 1.0
    assert torch.equal(dequantized.cpu(), expected.to(torch.bfloat16) * scale)
    embedding.check_error()
    assert embedding.reader.stats()["requests"] == num_tokens * _NUM_HEADS
    embedding.log_and_reset_stats(1.0)
    assert embedding.reader.stats()["requests"] == 0


@requires_uva
def test_disk_embedding_replays_under_cuda_graph(tmp_path, single_rank):
    """Captured host nodes read the ids and token count current at replay."""
    embedding, table = _make_disk_embedding(tmp_path, fp8=True)
    padded_tokens = 16
    buffer = embedding.get_prefetch_ids_buffer(_MAX_TOKENS, _NUM_HEADS)
    source_ids = torch.zeros(padded_tokens, _NUM_HEADS, dtype=torch.long, device="cuda")
    query_start_loc = torch.zeros(2, dtype=torch.int32, device="cuda")
    hidden_states = torch.empty(padded_tokens, 1, device="cuda")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        buffer[:padded_tokens].copy_(source_ids)
        embedding.start_prefetch(
            None, buffer[:padded_tokens], query_start_loc=query_start_loc
        )
        output = embedding.dequantize(embedding(hidden_states), torch.bfloat16)

    for num_tokens in (padded_tokens, 5):
        ids = torch.randint(0, _NUM_ROWS, (padded_tokens, _NUM_HEADS), device="cuda")
        source_ids.copy_(ids)
        query_start_loc[1] = num_tokens
        graph.replay()
        torch.accelerator.synchronize()
        expected = _expected_rows(table, ids[:num_tokens]).to(torch.bfloat16) * 0.5
        assert torch.equal(output[:num_tokens].cpu(), expected)
    embedding.check_error()


_REMOTE_SERVER = "10.0.0.2:18515"


class _FakeRemoteReader:
    """Native reader stand-in whose remote source is scripted by the test."""

    def __init__(self, segments, row_bytes, **options):
        self.options = options
        self.status = {
            "state": "uploading",
            "uploaded_bytes": 4_000_000_000,
            "total_bytes": 16_000_000_000,
            "regions": 16,
            "error": "",
        }
        self.status_polls = 0
        self.resets = 0

    def bind_demand(self, *buffers):
        pass

    def check_error(self):
        pass

    def remote_status(self):
        self.status_polls += 1
        return dict(self.status)

    def stats(self):
        return {
            **dict.fromkeys(["requests", "cache_hits", "reads", "bytes"], 0),
            **dict.fromkeys(["demand_wait_us", "demand_jobs", "hint_rows"], 0),
            "hint_dropped": 0,
            "read_us_p50": 0.0,
            "read_us_p99": 0.0,
            "direct_io": True,
            "remote_reads": 40,
            "remote_bytes": 6400,
            "remote_read_us_p50": 5.0,
            "remote_read_us_p99": 9.0,
            "remote_requeued": 0,
        }

    def reset_stats(self):
        self.resets += 1


@pytest.fixture
def remote_embedding(tmp_path, single_rank, monkeypatch):
    import vllm.models.qwen4_exp.common.ple_disk_reader as reader_module

    engram_config = EngramConfig(
        disk_offload=True, remote_memory_server=_REMOTE_SERVER, remote_max_inflight=32
    )
    monkeypatch.setattr(
        ple_disk_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(engram_config=engram_config),
    )
    monkeypatch.setattr(reader_module, "PLEDiskReader", _FakeRemoteReader)
    embedding, _ = _make_disk_embedding(tmp_path, fp8=False)
    return embedding


def test_config_backends_match_reader_backends():
    """Every configurable backend must have a reader variant to build."""
    from vllm.config.engram import RemoteMemoryBackend
    from vllm.models.qwen4_exp.common.ple_disk_reader import REMOTE_BACKENDS

    assert set(get_args(RemoteMemoryBackend)) == set(REMOTE_BACKENDS)


@requires_uva
def test_remote_options_reach_reader(remote_embedding):
    from vllm.models.qwen4_exp.common.ple_disk_reader import PLERemoteOptions

    expected = PLERemoteOptions(server=_REMOTE_SERVER, max_inflight=32)
    assert remote_embedding.reader.options["remote"] == expected


@requires_uva
@pytest.mark.parametrize(
    "state, level, message",
    [
        ("ready", "INFO", "is ready: uploaded 16.0 GB"),
        ("failed", "WARNING", "failed: peer gone; continuing with SSD reads"),
    ],
)
def test_remote_transition_logged_once(
    remote_embedding, caplog_vllm, state, level, message
):
    """Steps stop polling the remote status once the upload has settled."""
    reader = remote_embedding.reader
    remote_embedding.check_error()
    reader.status.update(state=state, uploaded_bytes=16_000_000_000, error="peer gone")
    for _ in range(3):
        remote_embedding.check_error()
    polls = reader.status_polls
    remote_embedding.check_error()

    assert reader.status_polls == polls
    records = [r for r in caplog_vllm.records if message in r.getMessage()]
    assert [r.levelname for r in records] == [level]


@requires_uva
def test_remote_stats_report_upload_progress(remote_embedding, caplog_vllm):
    remote_embedding.log_and_reset_stats(60.0)

    assert remote_embedding.reader.resets == 1
    (message,) = [
        r.getMessage() for r in caplog_vllm.records if "PLE disk" in r.getMessage()
    ]
    assert "remote uploading: 40 reads" in message
    assert "p50 5 us p99 9 us" in message
    assert "uploaded 4.0 of 16.0 GB" in message
