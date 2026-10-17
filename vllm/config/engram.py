# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Literal

import regex as re
from pydantic import Field, model_validator
from typing_extensions import Self

from vllm.config.utils import config, get_hash_factors, hash_factors

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

# Architecture -> the hf_text_config field naming its n-gram layers. A model is
# only configurable here if it actually has such layers to store.
_NGRAM_LAYER_FIELDS = {
    "DeepseekV41ForCausalLM": "engram_layer_ids",
    "Qwen4ExpForCausalLM": "ple_layer_ids",
    "Qwen4ExpForConditionalGeneration": "ple_layer_ids",
}
_DISK_OFFLOAD_ARCHITECTURES = frozenset(
    {"Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"}
)
# Runtime tunables that never change the captured graph or compiled code.
_DISK_TUNABLE_FIELDS = frozenset(
    {
        "disk_row_cache_rows",
        "disk_io_threads",
        "disk_max_inflight",
        "disk_direct_io",
        "disk_prefetch_hints",
        "disk_wait_timeout_s",
        "disk_stats_interval_s",
        "remote_memory_backend",
        "remote_memory_server",
        "remote_max_inflight",
        "remote_region_bytes",
        "remote_upload_chunk_bytes",
        "remote_connect_timeout_s",
        "remote_fallback_to_disk",
    }
)
_HOST_PORT_RE = re.compile(r"\[([^\[\]\s]+)\]:(\d{1,5})|([^:\[\]\s]+):(\d{1,5})")


RemoteMemoryBackend = Literal["ucx"]


def _check_host_port(backend: str, address: str) -> None:
    match = _HOST_PORT_RE.fullmatch(address)
    port = int(match[2] or match[4]) if match else 0
    if match is None or not 1 <= port <= 65535:
        raise ValueError(
            f"remote_memory_server for the {backend} backend must be 'host:port' "
            f"or '[ipv6]:port' with a port in 1..65535, got {address!r}"
        )


# Backend -> validator of its remote_memory_server address.
_REMOTE_ADDRESS_CHECKS = {"ucx": _check_host_port}


def model_has_engram_layers(model_config: "ModelConfig | None") -> bool:
    """Whether the model carries n-gram embedding layers."""
    if model_config is None:
        return False
    field = _NGRAM_LAYER_FIELDS.get(model_config.architecture)
    if field is None:
        return False
    return bool(getattr(model_config.hf_text_config, field, None))


@config
class EngramConfig:
    """Configuration for Engram embedding storage and sharding."""

    cpu_offload: bool = True
    """Store embedding weights in pinned CPU memory for UVA lookup."""

    embedding_across_dp: bool = False
    """Shard embeddings across TP and all DP ranks when enabled.
    Otherwise, each DP rank has a separate TP-sharded embedding replica."""

    dp_shared_memory: bool | None = None
    """Share CPU-offloaded embedding weights between co-located
    DP replicas. Each node stores one copy of every TP shard, reducing host
    memory without per-step Engram DP collectives. Requires sufficient
    /dev/shm capacity and a shared IPC namespace. Defaults to enabled whenever
    the other settings allow it, falling back to per-replica tables when DP
    replicas are not co-located on one node or /dev/shm cannot hold them."""

    use_thp: bool = False
    """Back private CPU-offloaded tables with transparent huge pages (best
    effort, falls back to ordinary pinned pages). Prefaulting the tables at
    startup takes longer. Requires cpu_offload without dp_shared_memory."""

    disk_offload: bool = False
    """Read embedding rows from the checkpoint files on demand instead of
    keeping the tables in memory (Qwen4Exp only). Takes precedence over
    cpu_offload and is incompatible with dp_shared_memory and use_thp."""

    disk_row_cache_rows: int = Field(default=1 << 20, ge=0)
    """Rows kept in the host-memory row cache of disk offload; 0 disables it."""

    disk_io_threads: int = Field(default=16, ge=1)
    """I/O worker threads used by disk offload."""

    disk_max_inflight: int = Field(default=256, ge=1)
    """Maximum outstanding row reads of disk offload."""

    disk_direct_io: bool = True
    """Open checkpoint files with O_DIRECT for disk offload, falling back to
    buffered reads where the filesystem does not support it."""

    disk_prefetch_hints: bool = True
    """Issue advisory row prefetches for upcoming prefill chunks under disk
    offload."""

    disk_wait_timeout_s: float = Field(default=60.0, gt=0)
    """Seconds a step waits for its disk offload reads before failing."""

    disk_stats_interval_s: float = Field(default=60.0, ge=0)
    """Seconds between disk offload reader statistics log lines (row-cache
    hit rate, SSD reads, read latency, per-step wait); 0 disables them."""

    remote_memory_server: str | None = None
    """Address of a memory server to borrow RAM from, in the form the
    remote_memory_backend expects ("host:port" or "[ipv6]:port" for ucx). The
    table is uploaded there from the checkpoint in the background and rows are
    then read remotely instead of from the SSD. Requires disk_offload."""

    remote_memory_backend: RemoteMemoryBackend = "ucx"
    """Transport used to reach remote_memory_server: "ucx" (RDMA such as
    RoCEv2 or InfiniBand, or TCP, through UCX)."""

    remote_max_inflight: int = Field(default=256, ge=1)
    """Maximum outstanding row reads from the remote memory server."""

    remote_region_bytes: int = Field(default=1 << 30, ge=1 << 20)
    """Size of each memory region borrowed from the remote memory server."""

    remote_upload_chunk_bytes: int = Field(default=64 << 20, ge=1 << 20)
    """Size of each staging buffer used to upload the table to the remote
    memory server."""

    remote_connect_timeout_s: float = Field(default=30.0, gt=0)
    """Seconds to wait for connecting to the remote memory server and
    borrowing its regions."""

    remote_fallback_to_disk: bool = True
    """Keep serving rows from the SSD when the remote memory server cannot be
    reached or fails later; otherwise such a failure is fatal."""

    @model_validator(mode="after")
    def _validate_shared_memory(self) -> Self:
        if self.disk_offload and (self.dp_shared_memory or self.use_thp):
            raise ValueError(
                "disk_offload is incompatible with dp_shared_memory and use_thp"
            )
        if self.dp_shared_memory and not self.cpu_offload:
            raise ValueError("dp_shared_memory requires cpu_offload=True")
        if self.use_thp and (not self.cpu_offload or self.dp_shared_memory):
            raise ValueError(
                "use_thp requires cpu_offload=True and dp_shared_memory=False"
            )
        if self.remote_memory_server is not None:
            if not self.disk_offload:
                raise ValueError("remote_memory_server requires disk_offload=True")
            _REMOTE_ADDRESS_CHECKS[self.remote_memory_backend](
                self.remote_memory_backend, self.remote_memory_server
            )
        return self

    def verify_model_config(self, model_config: "ModelConfig | None") -> None:
        """Reject Engram configuration for models without n-gram embeddings."""
        from vllm.platforms import current_platform

        field = (
            _NGRAM_LAYER_FIELDS.get(model_config.architecture)
            if model_config is not None
            else None
        )
        if (
            model_config is None
            or field is None
            or not current_platform.is_cuda_alike()
            or not getattr(model_config.hf_text_config, field, None)
        ):
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                "embeddings, non-empty n-gram layer ids, and a CUDA-alike "
                "device (CUDA or ROCm)."
            )
        if not self.disk_offload:
            return
        if model_config.architecture not in _DISK_OFFLOAD_ARCHITECTURES:
            raise ValueError(
                "Engram disk_offload is only supported for Qwen4Exp models, got "
                f"{model_config.architecture}."
            )
        if not current_platform.is_cuda():
            raise ValueError(
                "Engram disk_offload requires CUDA; the ROCm Qwen4Exp "
                "implementation does not support it."
            )

    def resolve_dp_shared_memory(self, parallel_config: "ParallelConfig") -> None:
        """Share host tables by default wherever the configuration permits."""
        if self.dp_shared_memory is None:
            self.dp_shared_memory = (
                self.cpu_offload
                and not self.disk_offload
                and not self.use_thp
                and parallel_config.data_parallel_size > 1
                and not parallel_config.enable_elastic_ep
            )

    def verify_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        """Reject unsupported embedding parallel topologies."""
        if self.dp_shared_memory:
            if parallel_config.data_parallel_size <= 1:
                raise ValueError("dp_shared_memory requires data_parallel_size > 1.")
            if parallel_config.enable_elastic_ep:
                raise ValueError("dp_shared_memory is not supported with elastic EP.")
        if (
            self.embedding_across_dp
            and parallel_config.data_parallel_size > 1
            and parallel_config.enable_elastic_ep
        ):
            raise ValueError(
                "Engram embedding_across_dp is not supported with elastic EP yet."
            )
        if self.disk_offload and (
            parallel_config.tensor_parallel_size > 1
            or parallel_config.data_parallel_size > 1
            or parallel_config.pipeline_parallel_size > 1
        ):
            raise ValueError(
                "Engram disk_offload requires tensor_parallel_size=1, "
                "data_parallel_size=1, and pipeline_parallel_size=1."
            )

    def get_parallel_size(self, parallel_config: "ParallelConfig") -> int:
        """Derive the embedding group size from the parallel configuration."""
        size = parallel_config.tensor_parallel_size
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1:
            size *= parallel_config.data_parallel_size
        return size

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, set(_DISK_TUNABLE_FIELDS)))
