#!/bin/bash
# Frees disk space on a GitHub-hosted runner for large Docker builds: removes
# preinstalled toolchains the build does not use, and moves Docker's storage
# to /mnt when that is a separate file system with more free space than /.
# Prints the disk usage before and after.
set -euo pipefail

report() {
    echo "== disk usage: $1"
    df -h
    sudo du -sh "$(docker info -f '{{.DockerRootDir}}')" 2>/dev/null || true
}

report "before cleanup"

sudo rm -rf \
    /opt/az \
    /opt/ghc \
    /opt/google \
    /opt/hostedtoolcache \
    /opt/microsoft \
    /usr/lib/jvm \
    /usr/local/.ghcup \
    /usr/local/lib/android \
    /usr/local/lib/node_modules \
    /usr/local/share/boost \
    /usr/local/share/powershell \
    /usr/share/dotnet \
    /usr/share/swift
sudo apt-get clean
docker system prune -af

avail() { df --output=avail -B1 "$1" | tail -1; }
if mountpoint -q /mnt && [ "$(stat -c %d /mnt)" != "$(stat -c %d /)" ] \
    && [ "$(avail /mnt)" -gt "$(avail /)" ]; then
    echo "Moving Docker's storage to /mnt/docker"
    sudo systemctl stop docker docker.socket
    sudo mkdir -p /etc/docker /mnt/docker
    config=/etc/docker/daemon.json
    if [ -s "${config}" ]; then
        sudo jq '. + {"data-root": "/mnt/docker"}' "${config}" \
            | sudo tee "${config}.new" > /dev/null
    else
        echo '{"data-root": "/mnt/docker"}' | sudo tee "${config}.new" > /dev/null
    fi
    sudo mv "${config}.new" "${config}"
    sudo systemctl start docker
fi

report "after cleanup"
