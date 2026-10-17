# docker-bake.hcl - vLLM Docker build configuration
#
# This file lives in vLLM repo at docker/docker-bake.hcl
#
# Usage:
#   cd docker && docker buildx bake        # Build default target (openai)
#   cd docker && docker buildx bake test   # Build test target
#   docker buildx bake --print             # Show resolved config
#
# Reference: https://docs.docker.com/build/bake/reference/

# Build configuration

variable "MAX_JOBS" {
  default = 16
}

variable "NVCC_THREADS" {
  default = 8
}

variable "TORCH_CUDA_ARCH_LIST" {
  default = "8.0 8.9 9.0 10.0 11.0 12.0"
}

variable "COMMIT" {
  default = ""
}

variable "VLLM_BUILD_COMMIT" {
  default = "unknown"
}

variable "VLLM_BUILD_PIPELINE" {
  default = "local"
}

variable "VLLM_BUILD_URL" {
  default = ""
}

variable "VLLM_IMAGE_TAG" {
  default = "local/vllm-openai:dev"
}

# Groups

group "default" {
  targets = ["openai"]
}

group "all" {
  targets = ["openai", "openai-ubuntu2404"]
}

# Base targets

target "_common" {
  dockerfile = "docker/Dockerfile"
  context    = "."
  args = {
    max_jobs             = MAX_JOBS
    nvcc_threads         = NVCC_THREADS
    torch_cuda_arch_list = TORCH_CUDA_ARCH_LIST
    VLLM_BUILD_COMMIT    = VLLM_BUILD_COMMIT != "unknown" ? VLLM_BUILD_COMMIT : (COMMIT != "" ? COMMIT : "unknown")
    VLLM_BUILD_PIPELINE  = VLLM_BUILD_PIPELINE
    VLLM_BUILD_URL       = VLLM_BUILD_URL
    VLLM_IMAGE_TAG       = VLLM_IMAGE_TAG
  }
}

target "_labels" {
  labels = {
    "org.opencontainers.image.source"      = "https://github.com/vllm-project/vllm"
    "org.opencontainers.image.vendor"      = "vLLM"
    "org.opencontainers.image.title"       = "vLLM"
    "org.opencontainers.image.description" = "vLLM: A high-throughput and memory-efficient inference and serving engine for LLMs"
    "org.opencontainers.image.licenses"    = "Apache-2.0"
    "org.opencontainers.image.revision"    = VLLM_BUILD_COMMIT != "unknown" ? VLLM_BUILD_COMMIT : (COMMIT != "" ? COMMIT : "unknown")
    "org.opencontainers.image.version"     = VLLM_IMAGE_TAG
    "org.opencontainers.image.url"         = VLLM_BUILD_URL
    "ai.vllm.build.commit"                 = VLLM_BUILD_COMMIT != "unknown" ? VLLM_BUILD_COMMIT : (COMMIT != "" ? COMMIT : "unknown")
    "ai.vllm.build.pipeline"               = VLLM_BUILD_PIPELINE
    "ai.vllm.build.url"                    = VLLM_BUILD_URL
    "ai.vllm.image.tag"                    = VLLM_IMAGE_TAG
  }
  annotations = [
    "index,manifest:org.opencontainers.image.revision=${VLLM_BUILD_COMMIT != "unknown" ? VLLM_BUILD_COMMIT : (COMMIT != "" ? COMMIT : "unknown")}",
  ]
}

# Build targets

target "test" {
  inherits = ["_common", "_labels"]
  target   = "test"
  tags     = ["vllm:test"]
  output   = ["type=docker"]
}

target "openai" {
  inherits = ["_common", "_labels"]
  target   = "vllm-openai"
  tags     = ["vllm:openai"]
  output   = ["type=docker"]
}

# Ubuntu 24.04 targets

target "test-ubuntu2404" {
  inherits = ["_common", "_labels"]
  target   = "test"
  tags     = ["vllm:test-ubuntu24.04"]
  args = {
    UBUNTU_VERSION          = "24.04"
    GDRCOPY_OS_VERSION      = "Ubuntu24_04"
  }
  output = ["type=docker"]
}

target "openai-ubuntu2404" {
  inherits = ["_common", "_labels"]
  target   = "vllm-openai"
  tags     = ["vllm:openai-ubuntu24.04"]
  args = {
    UBUNTU_VERSION          = "24.04"
    GDRCOPY_OS_VERSION      = "Ubuntu24_04"
  }
  output = ["type=docker"]
}

# DGX Spark (GB10, arm64, CUDA 13) with the PLE remote memory reader, and the
# x86_64 PLE memory server. Build dgx-spark on the Spark itself:
#   docker buildx bake -f docker/docker-bake.hcl dgx-spark
#   docker buildx bake -f docker/docker-bake.hcl ple-mem-server

# The repository the PLE images link to on GitHub Container Registry; set by
# GitHub Actions.
variable "GITHUB_REPOSITORY" {
  default = "vllm-project/vllm"
}

variable "DGX_SPARK_BUILD_BASE_IMAGE" {
  default = "pytorch/manylinuxaarch64-builder:cuda13.0-b8b5f17a7d9ccfc25bbc5cf17b3fcea12964a042"
}

# GB10 needs CUDA 13; the builder image above must match its major.minor.
variable "DGX_SPARK_CUDA_VERSION" {
  default = "13.0.3"
}

# sm_121 runs the sm_12x family kernels built for 12.0 under CUDA 13.
variable "DGX_SPARK_CUDA_ARCH_LIST" {
  default = "12.0"
}

# The Spark's 20 cores share 128 GB with the GPU; keep nvcc's memory in check.
variable "DGX_SPARK_MAX_JOBS" {
  default = 8
}

variable "DGX_SPARK_NVCC_THREADS" {
  default = 2
}

# 1 takes vLLM's compiled kernels from the upstream wheel built for
# VLLM_MERGE_BASE_COMMIT (the upstream commit this branch is based on)
# instead of compiling them; only valid while the branch does not change
# them.
variable "DGX_SPARK_USE_PRECOMPILED" {
  default = ""
}

variable "VLLM_MERGE_BASE_COMMIT" {
  default = ""
}

# Neither Spark target inherits _labels: its index annotation cannot be
# exported by the default docker driver for a single-platform image.
target "openai-dgx-spark-base" {
  inherits  = ["_common"]
  target    = "vllm-openai"
  platforms = ["linux/arm64"]
  tags      = ["vllm:openai-dgx-spark-base"]
  args = {
    BUILD_BASE_IMAGE       = DGX_SPARK_BUILD_BASE_IMAGE
    CUDA_VERSION           = DGX_SPARK_CUDA_VERSION
    torch_cuda_arch_list   = DGX_SPARK_CUDA_ARCH_LIST
    max_jobs               = DGX_SPARK_MAX_JOBS
    nvcc_threads           = DGX_SPARK_NVCC_THREADS
    VLLM_USE_PRECOMPILED   = DGX_SPARK_USE_PRECOMPILED
    VLLM_MERGE_BASE_COMMIT = VLLM_MERGE_BASE_COMMIT
    VLLM_MAIN_CUDA_VERSION = join(".", slice(split(".", DGX_SPARK_CUDA_VERSION), 0, 2))
  }
}

target "dgx-spark" {
  dockerfile = "docker/Dockerfile.dgx_spark"
  context    = "."
  target     = "vllm-dgx-spark"
  platforms  = ["linux/arm64"]
  contexts = {
    "vllm:openai-dgx-spark-base" = "target:openai-dgx-spark-base"
  }
  args = {
    max_jobs = DGX_SPARK_MAX_JOBS
  }
  labels = {
    "org.opencontainers.image.source"      = "https://github.com/${GITHUB_REPOSITORY}"
    "org.opencontainers.image.title"       = "vLLM for DGX Spark"
    "org.opencontainers.image.description" = "vLLM for NVIDIA DGX Spark (CUDA 13) with Qwen4Exp PLE offload to a remote memory server over UCX"
    "org.opencontainers.image.licenses"    = "Apache-2.0"
    "org.opencontainers.image.revision"    = VLLM_BUILD_COMMIT
  }
  tags   = ["vllm:dgx-spark"]
  output = ["type=docker"]
}

target "ple-mem-server" {
  dockerfile = "docker/Dockerfile.ple_mem_server"
  context    = "."
  target     = "ple-mem-server"
  platforms  = ["linux/amd64"]
  labels = {
    "org.opencontainers.image.source"      = "https://github.com/${GITHUB_REPOSITORY}"
    "org.opencontainers.image.title"       = "vLLM PLE memory server"
    "org.opencontainers.image.description" = "Lends RAM over UCX (RDMA) to vLLM hosts offloading the Qwen4Exp PLE n-gram table"
    "org.opencontainers.image.licenses"    = "Apache-2.0"
  }
  tags       = ["vllm-ple-mem-server:latest"]
  output     = ["type=docker"]
}
