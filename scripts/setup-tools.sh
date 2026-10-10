#!/usr/bin/env bash
# Official ARM64 binaries, installed only under this checkout; no daemon start.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ "$(uname -m)" == aarch64 ]] || { echo 'This baseline targets ARM64.' >&2; exit 1; }
mkdir -p .local/downloads .local/bin .local/docker-config/cli-plugins .local/reports
DOCKER_VERSION=29.9.0
# Digest recorded from the selected official HTTPS download.
DOCKER_SHA256=811f6eee271678fc98dbe5c4d896c00fc5fe2daa4f7a37069c17abc036eab85b
COMPOSE_VERSION=v5.6.0
COMPOSE_SHA256=733ec76717ceb59052a9609b9dadfb523b2df8eab57a54212872d10a58078ea2
fetch() {
  local url="$1" target="$2"
  if [[ ! -f "$target" ]]; then
    curl --fail --location --retry 3 --connect-timeout 15 --max-time 600 \
      "$url" -o "$target.part"
    mv "$target.part" "$target"
  fi
}
fetch "https://download.docker.com/linux/static/stable/aarch64/docker-${DOCKER_VERSION}.tgz" \
  ".local/downloads/docker-${DOCKER_VERSION}.tgz"
fetch "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-aarch64" \
  .local/downloads/docker-compose-linux-aarch64
echo "$DOCKER_SHA256  .local/downloads/docker-${DOCKER_VERSION}.tgz" | sha256sum --check
echo "$COMPOSE_SHA256  .local/downloads/docker-compose-linux-aarch64" | sha256sum --check
tar -xzf ".local/downloads/docker-${DOCKER_VERSION}.tgz" -C .local/bin --strip-components=1
install -m 755 .local/downloads/docker-compose-linux-aarch64 \
  .local/docker-config/cli-plugins/docker-compose
sha256sum ".local/downloads/docker-${DOCKER_VERSION}.tgz" \
  .local/downloads/docker-compose-linux-aarch64 > .local/reports/tool-download-sha256.txt
export DOCKER_CONFIG="$ROOT/.local/docker-config"
.local/bin/docker --version
.local/bin/docker compose version
