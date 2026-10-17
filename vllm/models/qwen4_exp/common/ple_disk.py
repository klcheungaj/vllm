# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp PLE table served from the checkpoint files on demand.

The n-gram table is never materialized. Row ids are hashed into a pinned
buffer, a capturable host node hands them to the native reader, and a second
host node waits for the rows to land in a pinned staging buffer that the GPU
reads through UVA. Both host nodes act on the buffer contents at replay time,
so the backend needs no eager breaks under CUDA graphs.
"""

import glob
import json
import os
import struct
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import regex as re
import torch

from vllm.config import get_current_vllm_config
from vllm.config.engram import EngramConfig
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .ngram_embedding import Qwen4ExpPLEEmbedding, Qwen4ExpPLEEmbeddingMethod

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from .ple_disk_reader import PLEDiskReader

logger = init_logger(__name__)

_CHECKPOINT_LAYER_PREFIX = r"^(?:model\.)?(?:language_model\.)?(?:model\.)?layers\."
_PLE_SHARD_SUFFIX = r"\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
_PLE_SHARD_WEIGHT_RE = re.compile(rf"{_CHECKPOINT_LAYER_PREFIX}\d+{_PLE_SHARD_SUFFIX}")

_SAFETENSORS_DTYPES = {
    torch.float8_e4m3fn: "F8_E4M3",
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
}


def is_ple_shard_checkpoint_weight(name: str) -> bool:
    """Whether a checkpoint name is a row shard of a backbone PLE table."""
    return _PLE_SHARD_WEIGHT_RE.search(name) is not None


def resolve_local_model_dir(vllm_config: "VllmConfig") -> str:
    """Return the local directory holding the model checkpoint files."""
    model_config = vllm_config.model_config
    if os.path.isdir(model_config.model):
        return model_config.model
    from vllm.transformers_utils.repo_utils import hf_api

    return hf_api().snapshot_download(
        repo_id=model_config.model,
        revision=model_config.revision,
        cache_dir=vllm_config.load_config.download_dir,
        local_files_only=True,
    )


def _read_safetensors_header(path: str) -> tuple[int, dict[str, Any]]:
    """Return the absolute data offset and the JSON header of a file."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    return 8 + header_len, header


def _checkpoint_files(model_dir: str) -> dict[str, list[str]]:
    """Map each safetensors file to the tensor names it may contain."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        files: dict[str, list[str]] = {}
        for name, file_name in weight_map.items():
            files.setdefault(os.path.join(model_dir, file_name), []).append(name)
        return files
    paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not paths:
        raise FileNotFoundError(f"No safetensors checkpoint found in {model_dir}")
    return {path: [] for path in paths}


@dataclass(frozen=True)
class PLEDiskSegmentSpec:
    """One checkpoint shard of a PLE table: where its rows live on disk."""

    path: str
    offset: int
    row_start: int
    num_rows: int


def find_ple_disk_segments(
    model_dir: str,
    *,
    layer_idx: int,
    num_rows: int,
    embedding_dim: int,
    dtype: torch.dtype,
    split_ngram_parts: int,
) -> list[PLEDiskSegmentSpec]:
    """Locate every checkpoint row shard of one decoder layer's PLE table.

    Args:
        model_dir: Local checkpoint directory.
        layer_idx: Decoder layer index that owns the table.
        num_rows: Table rows the shards must cover.
        embedding_dim: Elements per table row.
        dtype: Expected storage dtype of the rows.
        split_ngram_parts: Number of checkpoint shards the table is split into.

    Returns:
        Non-empty segments ordered by their first row.

    Raises:
        ValueError: If a shard is malformed or the shards do not cover the
            table exactly.

    """
    st_dtype = _SAFETENSORS_DTYPES.get(dtype)
    if st_dtype is None:
        raise ValueError(f"PLE disk offload does not support dtype {dtype}")
    pattern = re.compile(f"{_CHECKPOINT_LAYER_PREFIX}{layer_idx}{_PLE_SHARD_SUFFIX}")
    shard_size = (num_rows + split_ngram_parts - 1) // split_ngram_parts
    row_bytes = embedding_dim * dtype.itemsize
    found: dict[int, PLEDiskSegmentSpec] = {}
    for path, names in _checkpoint_files(model_dir).items():
        if names and not any(pattern.search(name) for name in names):
            continue
        data_start, header = _read_safetensors_header(path)
        for name, info in header.items():
            match = pattern.search(name)
            if match is None:
                continue
            shard = int(match.group(1))
            if shard >= split_ngram_parts:
                raise ValueError(
                    f"PLE shard {name} exceeds split_ngram_parts={split_ngram_parts}"
                )
            if shard in found:
                raise ValueError(f"Duplicate PLE shard {name} in {path}")
            row_start = shard * shard_size
            rows = max(0, min(shard_size, num_rows - row_start))
            begin, end = info["data_offsets"]
            if (
                info["dtype"] != st_dtype
                or info["shape"] != [rows, embedding_dim]
                or end - begin != rows * row_bytes
            ):
                raise ValueError(
                    f"PLE shard {name} has dtype {info['dtype']} and shape "
                    f"{info['shape']}, expected {st_dtype} and "
                    f"{[rows, embedding_dim]}"
                )
            found[shard] = PLEDiskSegmentSpec(path, data_start + begin, row_start, rows)

    segments = sorted(
        (segment for segment in found.values() if segment.num_rows),
        key=lambda segment: segment.row_start,
    )
    covered = 0
    for segment in segments:
        if segment.row_start != covered:
            break
        covered += segment.num_rows
    if covered != num_rows:
        raise ValueError(
            f"PLE shards of layer {layer_idx} in {model_dir} cover {covered} of "
            f"{num_rows} rows ({len(found)} of {split_ngram_parts} shards found)"
        )
    return segments


def ple_ngram_row_ids_np(
    tokens: np.ndarray,
    start: int,
    end: int,
    *,
    layer_multipliers: np.ndarray,
    ngram_heads_vocab_sizes: np.ndarray,
    ngram_heads_offsets: np.ndarray,
    eos_token_id: int,
    heads_per_ngram: int,
) -> np.ndarray:
    """NumPy port of ``ple_ngram_ids`` for positions ``[start, end)``.

    Args:
        tokens: The request's token ids from position 0.
        start: First position to hash.
        end: One past the last position to hash.
        layer_multipliers: Hash multipliers, one per n-gram position.
        ngram_heads_vocab_sizes: Per-head table partition sizes.
        ngram_heads_offsets: Per-head table partition offsets.
        eos_token_id: Token that cuts the n-gram history.
        heads_per_ngram: Hash heads per n-gram order.

    Returns:
        int64 row ids of shape ``[end - start, num_heads]``, bit-exact with the
        Triton kernel, including int64 wraparound and the EOS cut.

    """
    positions = np.arange(start, end, dtype=np.int64)
    context_len = layer_multipliers.shape[0] - 1
    mixed = tokens[positions].astype(np.int64) * layer_multipliers[0]
    crossed = np.zeros(positions.shape, dtype=bool)
    mixed_by_order = []
    for shift in range(1, context_len + 1):
        source = positions - shift
        candidate = np.where(
            source >= 0, tokens[np.maximum(source, 0)], eos_token_id
        ).astype(np.int64)
        candidate[crossed] = eos_token_id
        crossed |= candidate == eos_token_id
        mixed = mixed ^ (candidate * layer_multipliers[shift])
        mixed_by_order.append(mixed)
    order = np.arange(ngram_heads_vocab_sizes.shape[0]) // heads_per_ngram
    hashed = np.stack(mixed_by_order, axis=1)[:, order]
    return hashed % ngram_heads_vocab_sizes + ngram_heads_offsets


class Qwen4ExpPLEDiskEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table read row by row from the checkpoint into pinned staging."""

    supports_prefetch: ClassVar[bool] = True
    prefetch_at_forward_start: ClassVar[bool] = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
    ) -> None:
        if not is_uva_available():
            raise RuntimeError("Engram disk offload requires UVA support")
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        if self.tp_size != 1:
            raise ValueError("Engram disk offload requires an ETP size of 1")
        self.storage_dtype = self.weight.dtype
        # Rows come from the checkpoint files, so the table is not a parameter.
        del self.weight
        vllm_config = get_current_vllm_config()
        self.engram_config = vllm_config.engram_config or EngramConfig()
        # Dummy weights never open the reader; their rows stay zero.
        self._zero_rows_while_closed = vllm_config.load_config.load_format == "dummy"
        self.row_bytes = self.embedding_dim * self.storage_dtype.itemsize
        self.num_ngram_heads = num_ngram_heads
        self._ids = torch.zeros(
            max_total_tokens,
            num_ngram_heads,
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        self._num_tokens = torch.zeros(
            1, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._staging = torch.zeros(
            max_total_tokens,
            num_ngram_heads * self.row_bytes,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self._ids_uva = get_accelerator_view_from_cpu_tensor(self._ids)
        self._num_tokens_uva = get_accelerator_view_from_cpu_tensor(self._num_tokens)
        self._staging_uva = get_accelerator_view_from_cpu_tensor(self._staging).view(
            self.storage_dtype
        )
        self.reader: PLEDiskReader | None = None
        self._disk_segments: list[PLEDiskSegmentSpec] = []
        self._remote_state = "disabled"
        self._remote_opened_at = 0.0

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Describe the table without allocating it."""
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="meta")

    def get_prefetch_ids_buffer(
        self,
        max_total_tokens: int,
        num_ngram_heads: int,
    ) -> torch.Tensor:
        """Hash row ids straight into the pinned buffer the reader consumes."""
        assert self._ids_uva.shape == (max_total_tokens, num_ngram_heads)
        return self._ids_uva

    def open_disk_source(self, model_dir: str, *, split_ngram_parts: int) -> None:
        """Index this layer's checkpoint shards and start the row reader."""
        from vllm.model_executor.models.utils import extract_layer_index

        from .ple_disk_reader import PLEDiskReader, PLEDiskSegment, PLERemoteOptions

        specs = find_ple_disk_segments(
            model_dir,
            layer_idx=extract_layer_index(self.prefix),
            num_rows=self.org_vocab_size,
            embedding_dim=self.embedding_dim,
            dtype=self.storage_dtype,
            split_ngram_parts=split_ngram_parts,
        )
        if self.reader is not None:
            if specs == self._disk_segments:
                return
            # Captured CUDA graphs hold host nodes bound to the open reader.
            raise NotImplementedError(
                "Re-pointing an open PLE disk source at different checkpoint "
                "files is not supported"
            )
        config = self.engram_config
        remote = None
        if config.remote_memory_server is not None:
            remote = PLERemoteOptions(
                server=config.remote_memory_server,
                backend=config.remote_memory_backend,
                max_inflight=config.remote_max_inflight,
                region_bytes=config.remote_region_bytes,
                upload_chunk_bytes=config.remote_upload_chunk_bytes,
                connect_timeout_s=config.remote_connect_timeout_s,
                fallback_to_disk=config.remote_fallback_to_disk,
                max_batch_rows=config.remote_max_batch_rows,
                token_file=config.remote_token_file,
                allow_insecure_transport=config.remote_allow_insecure_transport,
                busy_poll=config.remote_busy_poll,
            )
        self._remote_opened_at = time.monotonic()
        reader = PLEDiskReader(
            [
                PLEDiskSegment(spec.path, spec.offset, spec.row_start, spec.num_rows)
                for spec in specs
            ],
            self.row_bytes,
            cache_rows=config.disk_row_cache_rows,
            io_threads=config.disk_io_threads,
            max_inflight=config.disk_max_inflight,
            direct_io=config.disk_direct_io,
            wait_timeout_s=config.disk_wait_timeout_s,
            wait_spin_us=config.disk_wait_spin_us,
            inline_plan_rows=config.disk_inline_plan_rows,
            remote=remote,
        )
        reader.bind_demand(self._ids, self._num_tokens, self._staging)
        self.reader = reader
        self._disk_segments = specs
        logger.info(
            "PLE disk offload for %s: %d rows x %d bytes from %d file(s), "
            "row cache %d rows, direct_io=%s",
            self.prefix,
            self.org_vocab_size,
            self.row_bytes,
            len({spec.path for spec in specs}),
            config.disk_row_cache_rows,
            reader.stats().get("direct_io"),
        )
        if remote is not None:
            status = reader.remote_status()
            self._remote_state = "connecting"
            logger.info(
                "PLE remote memory for %s: %s server %s, %d region(s), %.1f GB "
                "to upload, state %s, fallback_to_disk=%s",
                self.prefix,
                remote.backend,
                remote.server,
                int(status["regions"]),
                int(status["total_bytes"]) / 1e9,
                status["state"],
                remote.fallback_to_disk,
            )

    def hint_rows(self, rows: np.ndarray) -> None:
        """Advise the reader of rows a later step will need."""
        if self.reader is not None and rows.size:
            self.reader.hint(rows.ravel())

    def check_error(self) -> None:
        """Raise any I/O failure recorded by the reader and log remote state
        changes."""
        if self.reader is None:
            return
        if self._remote_state not in ("disabled", "failed"):
            self._note_remote_status(self.reader.remote_status())
        self.reader.check_error()

    def _note_remote_status(self, status: dict[str, Any]) -> None:
        """Log the remote source becoming ready or failing, once each."""
        state = status["state"]
        if state == self._remote_state:
            return
        self._remote_state = state
        if state == "ready":
            logger.info(
                "PLE remote memory for %s is ready: uploaded %.1f GB in %.0fs",
                self.prefix,
                int(status["uploaded_bytes"]) / 1e9,
                time.monotonic() - self._remote_opened_at,
            )
        elif state == "failed":
            logger.warning(
                "PLE remote memory for %s failed: %s; %s",
                self.prefix,
                status["error"],
                "continuing with SSD reads"
                if self.engram_config.remote_fallback_to_disk
                else "SSD fallback is disabled, so the worker aborts on its next "
                "read of an uploaded row",
            )

    def log_and_reset_stats(self, interval_s: float) -> None:
        """Log the reader statistics gathered since the previous call."""
        if self.reader is None:
            return
        stats = self.reader.stats()
        self.reader.reset_stats()
        requests = int(stats["requests"])
        jobs = max(1, int(stats["demand_jobs"]))
        fmt = (
            "PLE disk %s over %.0fs: %d rows requested, %.1f%% row-cache hits, "
            "%d SSD reads (%.1f MB), read p50 %.0f us p99 %.0f us, "
            "step rows ready p50 %.0f us p99 %.0f us max %.0f us "
            "(decode-size p50 %.0f us p99 %.0f us), "
            "step wait p50 %.0f us p99 %.0f us (%.0f us average), "
            "%d hinted rows (%d dropped)"
        )
        args: list[Any] = [
            self.prefix,
            interval_s,
            requests,
            100.0 * int(stats["cache_hits"]) / max(1, requests),
            int(stats["reads"]),
            int(stats["bytes"]) / 1e6,
            stats["read_us_p50"],
            stats["read_us_p99"],
            stats["demand_us_p50"],
            stats["demand_us_p99"],
            stats["demand_us_max"],
            stats["demand_small_us_p50"],
            stats["demand_small_us_p99"],
            stats["wait_us_p50"],
            stats["wait_us_p99"],
            float(stats["demand_wait_us"]) / jobs,
            int(stats["hint_rows"]),
            int(stats["hint_dropped"]),
        ]
        if self._remote_state != "disabled":
            status = self.reader.remote_status()
            self._note_remote_status(status)
            fmt += (
                ", remote %s: %d reads (%.1f MB), p50 %.0f us p99 %.0f us, "
                "%d moved to SSD"
            )
            args += [
                status["state"],
                int(stats["remote_reads"]),
                int(stats["remote_bytes"]) / 1e6,
                stats["remote_read_us_p50"],
                stats["remote_read_us_p99"],
                int(stats["remote_requeued"]),
            ]
            if status["state"] == "uploading":
                fmt += ", uploaded %.1f of %.1f GB"
                args += [
                    int(status["uploaded_bytes"]) / 1e9,
                    int(status["total_bytes"]) / 1e9,
                ]
        logger.info(fmt, *args)

    def _open_reader(self) -> "PLEDiskReader | None":
        """Return the reader, or None while dummy weights leave it closed."""
        if self.reader is None and not self._zero_rows_while_closed:
            raise RuntimeError(
                f"PLE disk source of {self.prefix} is not open: the model loader "
                "did not run load_weights. Use load_format 'auto' or "
                "'safetensors' with Engram disk_offload."
            )
        return self.reader

    def start_prefetch(
        self,
        hidden_states: torch.Tensor | None,
        ngram_ids: torch.Tensor,
        *,
        query_start_loc: torch.Tensor | None = None,
    ) -> None:
        """Queue the demand read of this step's rows on the current stream."""
        del hidden_states
        if query_start_loc is None:
            raise ValueError("PLE disk offload requires query_start_loc")
        reader = self._open_reader()
        if reader is None:
            return
        if ngram_ids.data_ptr() != self._ids_uva.data_ptr():
            self._ids_uva[: ngram_ids.shape[0]].copy_(ngram_ids)
        self._num_tokens_uva.copy_(query_start_loc[-1:])
        reader.launch_notify(torch.cuda.current_stream())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Wait for this step's rows and return them in storage dtype."""
        reader = self._open_reader()
        if reader is not None:
            reader.launch_wait(torch.cuda.current_stream())
        return self._staging_uva[: hidden_states.shape[0]]
