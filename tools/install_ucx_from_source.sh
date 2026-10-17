#!/bin/bash
# Builds and installs UCX for the PLE memory server and vLLM's PLE remote
# reader: RDMA through verbs, rdmacm and mlx5 direct verbs (ConnectX NICs),
# no GPU support, and the release configuration (no debug logging,
# assertions or parameter checks) for the lowest latency.
#
# Usage: install_ucx_from_source.sh [PREFIX]   (default /opt/ucx)
#
# Environment:
#   UCX_VERSION          git tag or branch (default v1.22.0)
#   UCX_REPO             git URL (default https://github.com/openucx/ucx)
#   UCX_CONFIGURE_FLAGS  extra configure flags, e.g. --with-avx
#   MAX_JOBS             parallel make jobs (default: all CPUs)
#
# Needs git, make, a C/C++ compiler, autoconf, automake, libtool, pkg-config,
# libibverbs-dev and librdmacm-dev.

set -euo pipefail

prefix=${1:-/opt/ucx}
version=${UCX_VERSION:-v1.22.0}
repo=${UCX_REPO:-https://github.com/openucx/ucx}

src=$(mktemp -d)
trap 'rm -rf "${src}"' EXIT

git clone --depth 1 --branch "${version}" "${repo}" "${src}"
cd "${src}"
./autogen.sh
# shellcheck disable=SC2086
./contrib/configure-release \
    --prefix="${prefix}" \
    --enable-mt \
    --with-verbs \
    --with-rdmacm \
    --with-mlx5 \
    --without-cuda \
    --without-rocm \
    --without-ze \
    --without-gdrcopy \
    --without-knem \
    --without-xpmem \
    --without-java \
    --without-go \
    --disable-static \
    ${UCX_CONFIGURE_FLAGS:-}
make -j"${MAX_JOBS:-$(nproc)}"
make install

for module in libuct_ib_mlx5 libuct_rdmacm; do
    if [ ! -e "${prefix}/lib/ucx/${module}.so" ]; then
        echo "UCX was built without ${module}" >&2
        exit 1
    fi
done
find "${prefix}" -name '*.la' -delete
