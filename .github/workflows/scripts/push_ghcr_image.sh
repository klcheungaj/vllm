#!/bin/bash
# Publishes an image as ghcr.io/<owner>/<name>, tagged with the short commit,
# and with the branch name for branch pushes or the tag name plus `latest`
# for tag pushes. Expects `docker login ghcr.io` to be done.
#
# Usage: push_ghcr_image.sh [--remote] IMAGE NAME
#   IMAGE is a local image, or with --remote an image already in a registry,
#   which is tagged there without downloading or uploading its layers.
set -euo pipefail

remote=false
if [ "${1:-}" = --remote ]; then
    remote=true
    shift
fi
source_image=$1
image="ghcr.io/${GITHUB_REPOSITORY_OWNER,,}/$2"

tags=("${GITHUB_SHA::12}")
case "${GITHUB_REF}" in
    refs/tags/*) tags+=("${GITHUB_REF_NAME}" latest) ;;
    refs/heads/*) tags+=("$(printf %s "${GITHUB_REF_NAME}" | tr -c 'A-Za-z0-9_.-' '-')") ;;
esac

if [ "${remote}" = true ]; then
    args=()
    for tag in "${tags[@]}"; do
        args+=(--tag "${image}:${tag}")
    done
    docker buildx imagetools create "${args[@]}" "${source_image}"
else
    for tag in "${tags[@]}"; do
        docker tag "${source_image}" "${image}:${tag}"
        docker push "${image}:${tag}"
    done
fi
echo "Pushed ${image}: ${tags[*]}"
