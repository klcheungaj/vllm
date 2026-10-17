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
from torch import nn

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.models.qwen4_exp.common.ngram_embedding as ngram_embedding_module
import vllm.models.qwen4_exp.common.ple_disk as ple_disk_module
import vllm.models.qwen4_exp.nvidia.model as model_module
import vllm.models.qwen4_exp.nvidia.model_state as model_state_module
import vllm.models.qwen4_exp.nvidia.ngram_embedding as nvidia_ngram_module
from vllm.config.engram import EngramConfig
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
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
)
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding
from vllm.utils.platform_utils import is_uva_available
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

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


def _vllm_config(engram_config: EngramConfig, load_format: str = "auto"):
    return SimpleNamespace(
        engram_config=engram_config,
        load_config=SimpleNamespace(load_format=load_format),
    )


def _build_causal_lm(monkeypatch, disk_offload: bool) -> Qwen4ExpForCausalLM:
    """Run the real constructor with the decoder stack stubbed out."""
    monkeypatch.setattr(
        model_module,
        "Qwen4ExpModel",
        lambda **kwargs: SimpleNamespace(
            layers=[], make_empty_intermediate_tensors=None
        ),
    )
    monkeypatch.setattr(model_module, "ParallelLMHead", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_module, "LogitsProcessor", lambda *args: None)
    monkeypatch.setattr(
        model_module, "enable_qwen4_exp_low_latency_gemm", lambda *args: None
    )
    monkeypatch.setattr(
        Qwen4ExpForCausalLM, "set_moe_parameters", lambda self, layers: None
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                ple_layer_ids=[2], vocab_size=8, hidden_size=4
            ),
            dtype=torch.float16,
        ),
        quant_config=None,
        scheduler_config=None,
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        engram_config=EngramConfig(disk_offload=disk_offload),
    )
    return Qwen4ExpForCausalLM(vllm_config=vllm_config)


@pytest.mark.parametrize("disk_offload", [False, True])
def test_table_is_skipped_only_under_disk_offload(monkeypatch, disk_offload):
    """Skipping without disk offload would silently drop the resident table."""
    causal_lm = _build_causal_lm(monkeypatch, disk_offload)
    wrapper = Qwen4ExpForConditionalGeneration.__new__(Qwen4ExpForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.language_model = causal_lm
    shard = f"{_SHARD_NAME}.shard_0.weight"

    for model in (causal_lm, wrapper):
        assert model.should_skip_checkpoint_weight(shard) is disk_offload
        assert not model.should_skip_checkpoint_weight(f"{_SHARD_NAME}.weight_scale")
    assert causal_lm.requires_checkpoint_weight_skip is disk_offload


@pytest.mark.parametrize("wrapped", [False, True])
def test_disk_sources_open_once_after_all_weights(monkeypatch, wrapped):
    """The loader may enter the language model once per run of its names."""
    opened: list[nn.Module] = []
    monkeypatch.setattr(
        model_module,
        "open_ple_disk_sources",
        lambda model, vllm_config: opened.append(model),
    )

    class FakeAutoWeightsLoader:
        def __init__(self, module, **kwargs):
            self.module = module

        def load_weights(self, weights, mapper=None) -> set[str]:
            if isinstance(self.module, Qwen4ExpForConditionalGeneration):
                for _ in range(2):
                    self.module.language_model.load_weights([])
            return set()

    monkeypatch.setattr(model_module, "AutoWeightsLoader", FakeAutoWeightsLoader)
    causal_lm = _build_causal_lm(monkeypatch, disk_offload=True)
    model: nn.Module = causal_lm
    if wrapped:
        model = Qwen4ExpForConditionalGeneration.__new__(
            Qwen4ExpForConditionalGeneration
        )
        nn.Module.__init__(model)
        model.language_model = causal_lm
        model.language_model_only = True
        # As set by Qwen4ExpForConditionalGeneration.__init__.
        causal_lm.open_ple_disk_sources_after_load = False

    model.load_weights([])

    assert opened == [causal_lm]


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


def test_prefill_hints_follow_chunks_preemption_and_prefix_hits(monkeypatch):
    """Each prompt row is hinted once per request lifetime, a chunk ahead."""
    monkeypatch.setattr(MambaHybridModelState, "add_request", lambda *args: None)
    monkeypatch.setattr(MambaHybridModelState, "remove_request", lambda *args: None)
    monkeypatch.setattr(model_state_module, "_PLE_HINT_TOKENS", 5)
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
    state.ple_disk_stats_interval_s = 0.0
    state.ple_disk_stats_logged_at = 0.0
    state.ple_hint_requests = {}
    tokens = np.arange(100, 140)
    request = SimpleNamespace(req_id="prefill", prefill_token_ids=tokens.tolist())

    def step(start: int, chunk: int) -> list[np.ndarray]:
        hinted.clear()
        state._service_ple_disk(
            SimpleNamespace(
                num_reqs=2,
                req_ids=["decode", "prefill"],
                is_prefilling_np=np.array([False, True]),
                num_computed_prefill_tokens_np=np.array([40, start], dtype=np.int32),
                num_scheduled_tokens=np.array([1, chunk], dtype=np.int32),
            )
        )
        assert all(rows.shape[0] <= 5 for rows in hinted)
        return hinted

    def rows(start: int, end: int) -> np.ndarray:
        return ple_ngram_row_ids_np(tokens, start, end, **params)

    state.add_request(0, request)
    assert state.ple_hint_requests["prefill"][0].dtype == np.int32
    np.testing.assert_array_equal(np.concatenate(step(0, 8)), rows(0, 16))
    np.testing.assert_array_equal(np.concatenate(step(8, 8)), rows(16, 24))
    # Preemption re-adds the request; it resumes from a prefix-cache hit.
    state.remove_request("prefill")
    state.add_request(0, request)
    np.testing.assert_array_equal(np.concatenate(step(16, 8)), rows(16, 32))
    np.testing.assert_array_equal(np.concatenate(step(24, 16)), rows(32, 40))
    assert state.ple_hint_requests == {}
    assert step(40, 1) == []
    assert len(errors_checked) == 5


def _write_checkpoint(path, table: torch.Tensor, missing_shard: int | None = None):
    """Split a table into checkpoint shards across two files plus an index."""
    shard_size = -(-table.shape[0] // _SPLIT_PARTS)
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
    vllm_config = _vllm_config(EngramConfig(disk_offload=True, disk_row_cache_rows=64))
    for module in (ple_disk_module, nvidia_ngram_module):
        monkeypatch.setattr(module, "get_current_vllm_config", lambda: vllm_config)


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
            "demand_us_p50": 21.0,
            "demand_us_p99": 48.0,
            "demand_us_max": 70.0,
            "demand_small_us_p50": 20.0,
            "demand_small_us_p99": 40.0,
            "wait_us_p50": 0.0,
            "wait_us_p99": 12.0,
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

    vllm_config = _vllm_config(
        EngramConfig(
            disk_offload=True,
            remote_memory_server=_REMOTE_SERVER,
            remote_max_inflight=32,
            remote_max_batch_rows=64,
            remote_token_file="/etc/ple/token",
            remote_allow_insecure_transport=True,
            remote_busy_poll=False,
            disk_wait_spin_us=25.0,
            disk_inline_plan_rows=64,
        )
    )
    monkeypatch.setattr(ple_disk_module, "get_current_vllm_config", lambda: vllm_config)
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

    expected = PLERemoteOptions(
        server=_REMOTE_SERVER,
        max_inflight=32,
        max_batch_rows=64,
        token_file="/etc/ple/token",
        allow_insecure_transport=True,
        busy_poll=False,
    )
    assert remote_embedding.reader.options["remote"] == expected
    assert remote_embedding.reader.options["wait_spin_us"] == 25.0
    assert remote_embedding.reader.options["inline_plan_rows"] == 64


@requires_uva
def test_remote_failure_after_ready_is_logged_once(remote_embedding, caplog_vllm):
    """Losing the server after the upload must not go unnoticed."""
    reader = remote_embedding.reader
    remote_embedding.check_error()
    reader.status.update(state="ready", uploaded_bytes=16_000_000_000)
    for _ in range(3):
        remote_embedding.check_error()
    reader.status.update(state="failed", error="peer gone")
    for _ in range(3):
        remote_embedding.check_error()
    polls = reader.status_polls
    remote_embedding.check_error()

    assert reader.status_polls == polls
    logged = [
        (r.levelname, r.getMessage())
        for r in caplog_vllm.records
        if "PLE remote memory" in r.getMessage() and "state" not in r.getMessage()
    ]
    assert [level for level, _ in logged] == ["INFO", "WARNING"]
    assert "is ready: uploaded 16.0 GB" in logged[0][1]
    assert "failed: peer gone; continuing with SSD reads" in logged[1][1]


@requires_uva
def test_remote_stats_report_upload_progress(remote_embedding, caplog_vllm):
    remote_embedding.log_and_reset_stats(60.0)

    assert remote_embedding.reader.resets == 1
    (message,) = [
        r.getMessage() for r in caplog_vllm.records if "PLE disk" in r.getMessage()
    ]
    assert "step rows ready p50 21 us p99 48 us max 70 us" in message
    assert "step wait p50 0 us p99 12 us" in message
    assert "remote uploading: 40 reads" in message
    assert "p50 5 us p99 9 us" in message
    assert "uploaded 4.0 of 16.0 GB" in message


@requires_uva
@pytest.mark.parametrize("load_format", ["auto", "dummy"])
def test_closed_reader_fails_unless_weights_are_dummy(
    single_rank, monkeypatch, load_format
):
    """A loader that never opened the source must not serve zero rows."""
    vllm_config = _vllm_config(EngramConfig(disk_offload=True), load_format)
    monkeypatch.setattr(ple_disk_module, "get_current_vllm_config", lambda: vllm_config)
    embedding = _make_closed_disk_embedding(f"{_EMBEDDING_PREFIX}.ngram_embedding")
    ids = embedding.get_prefetch_ids_buffer(_MAX_TOKENS, _NUM_HEADS)[:4]
    query_start_loc = torch.tensor([0, 4], device="cuda")
    hidden_states = torch.empty(4, 1, device="cuda")

    if load_format == "dummy":
        embedding.start_prefetch(None, ids, query_start_loc=query_start_loc)
        assert not embedding(hidden_states).cpu().any()
    else:
        with pytest.raises(RuntimeError, match="is not open"):
            embedding.start_prefetch(None, ids, query_start_loc=query_start_loc)
        with pytest.raises(RuntimeError, match="is not open"):
            embedding(hidden_states)


def _make_closed_disk_embedding(prefix: str) -> Qwen4ExpPLEDiskEmbedding:
    with torch.device("cuda"):
        return Qwen4ExpPLEDiskEmbedding(
            _NUM_ROWS,
            _HEAD_DIM,
            params_dtype=torch.bfloat16,
            padding_size=1,
            prefix=prefix,
            embedding_method=Qwen4ExpPLEUnquantizedEmbeddingMethod(),
            num_ngram_heads=_NUM_HEADS,
            max_total_tokens=_MAX_TOKENS,
        )


class _RecordingReader:
    def __init__(self, layer_idx: int, events: list[tuple[str, int]]):
        self.layer_idx = layer_idx
        self.events = events

    def launch_notify(self, stream) -> None:
        self.events.append(("notify", self.layer_idx))

    def launch_wait(self, stream) -> None:
        self.events.append(("wait", self.layer_idx))


class _FakePLE(nn.Module):
    """The prefetch path of Qwen4ExpPLELayer over a disk embedding."""

    def __init__(self, embedding: Qwen4ExpPLEDiskEmbedding):
        super().__init__()
        self.ple_embedding = SimpleNamespace(ngram_embedding=embedding)

    def start_prefetch(self, hidden_states, input_ids, query_start_loc, context):
        embedding = self.ple_embedding.ngram_embedding
        ids = embedding.get_prefetch_ids_buffer(_MAX_TOKENS, _NUM_HEADS)
        embedding.start_prefetch(
            hidden_states, ids[: input_ids.numel()], query_start_loc=query_start_loc
        )


class _FakeDecoderLayer(nn.Module):
    def __init__(self, layer_idx: int, events, embedding=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.events = events
        if embedding is not None:
            self.ple = _FakePLE(embedding)

    def forward(self, hidden_states, **kwargs):
        if hasattr(self, "ple"):
            self.ple.ple_embedding.ngram_embedding(hidden_states)
        self.events.append(("layer", self.layer_idx))
        return hidden_states, None, None


@requires_uva
def test_forward_notifies_each_disk_layer_once_before_any_wait(
    single_rank, monkeypatch
):
    """A wait without its step's notify would hand the layer stale rows."""
    events: list[tuple[str, int]] = []
    layers = []
    for layer_idx in range(4):
        embedding = None
        if layer_idx in (1, 3):
            embedding = _make_closed_disk_embedding(
                f"model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding"
            )
            embedding.reader = _RecordingReader(layer_idx, events)
        layers.append(_FakeDecoderLayer(layer_idx, events, embedding))
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hc_count=1)
    model.layers = nn.ModuleList(layers)
    model.start_layer, model.end_layer = 0, len(layers)
    model._forward_start_prefetch_layers = (
        Qwen4ExpModel._find_forward_start_prefetch_layers(model.layers, 0, 4)
    )
    monkeypatch.setattr(
        model_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )
    num_tokens = 6

    model(
        input_ids=torch.zeros(num_tokens, dtype=torch.int32, device="cuda"),
        positions=torch.arange(num_tokens, device="cuda"),
        inputs_embeds=torch.zeros(num_tokens, 8, device="cuda"),
        query_start_loc=torch.tensor([0, num_tokens], device="cuda"),
        ngram_context=torch.zeros(1, 2, dtype=torch.int32, device="cuda"),
    )

    assert events == [
        ("notify", 1),
        ("notify", 3),
        ("layer", 0),
        ("wait", 1),
        ("layer", 1),
        ("layer", 2),
        ("wait", 3),
        ("layer", 3),
    ]


@requires_uva
def test_ngram_module_disk_lookup_replays_under_cuda_graph(tmp_path, single_rank):
    """Hashing, notify and wait captured together follow new inputs on replay."""
    eos = 7
    config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=eos,
        vocab_size=1000,
        split_ngram_parts=_SPLIT_PARTS,
        seed=1234,
        ngram_vocab_size_base=200,
        make_ngram_vocab_size_divisible_by=1,
    )
    with torch.device("cuda"):
        module = Qwen4ExpNGramEmbedding(
            config,
            _NUM_HEADS * _HEAD_DIM,
            ple_dense_layer_id=0,
            max_total_tokens=_MAX_TOKENS,
            data_parallel_rank=0,
            prefix=_EMBEDDING_PREFIX,
            params_dtype=torch.bfloat16,
        )
    table = torch.randn(module.ngram_embedding.org_vocab_size, _HEAD_DIM)
    table = table.to(torch.bfloat16)
    _write_checkpoint(tmp_path, table)
    module.open_disk_source(str(tmp_path))
    padded_tokens = 16
    input_ids = torch.zeros(padded_tokens, dtype=torch.int32, device="cuda")
    query_start_loc = torch.tensor([0, padded_tokens], dtype=torch.int32, device="cuda")
    ngram_context = torch.full((1, 2), eos, dtype=torch.int32, device="cuda")
    hidden_states = torch.empty(padded_tokens, 1, device="cuda")

    def lookup() -> torch.Tensor:
        module.start_prefetch(None, input_ids, query_start_loc, ngram_context)
        return module(hidden_states, input_ids, query_start_loc, ngram_context)

    lookup()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = lookup()

    for num_tokens in (padded_tokens, 5):
        input_ids.copy_(torch.randint(0, 1000, (padded_tokens,)))
        ngram_context.copy_(torch.randint(0, 1000, (1, 2)))
        query_start_loc[1] = num_tokens
        graph.replay()
        torch.accelerator.synchronize()
        ids = module.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        expected = _expected_rows(table, ids[:num_tokens])
        assert torch.equal(output[:num_tokens].cpu(), expected)
    module.ngram_embedding.check_error()
