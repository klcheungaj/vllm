#!/bin/bash
# Pushes a locally built image to ghcr.io/<owner>/<name>, tagged with the
# short commit, and with the branch name for branch pushes or the tag name
# plus `latest` for tag pushes. Expects `docker login ghcr.io` to be done.
#
# Usage: push_ghcr_image.sh LOCAL_IMAGE NAME
set -euo pipefail

local_image=$1
image="ghcr.io/${GITHUB_REPOSITORY_OWNER,,}/$2"

tags=("${GITHUB_SHA::12}")
case "${GITHUB_REF}" in
    refs/tags/*) tags+=("${GITHUB_REF_NAME}" latest) ;;
    refs/heads/*) tags+=("$(printf %s "${GITHUB_REF_NAME}" | tr -c 'A-Za-z0-9_.-' '-')") ;;
esac

for tag in "${tags[@]}"; do
    docker tag "${local_image}" "${image}:${tag}"
    docker push "${image}:${tag}"
done
echo "Pushed ${image}: ${tags[*]}"
