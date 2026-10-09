#!/usr/bin/env bash
# Invoke upstream Compose with this host's overlay; forward standard commands.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.local/bin:$PATH"
export DOCKER_CONFIG="${DOCKER_CONFIG:-$ROOT/.local/docker-config}"
exec docker compose \
  --env-file "${LAMP_ENV_FILE:-$ROOT/deploy/lamp/cm3588.env}" \
  -f deploy/docker-compose.conversation-rk3588-rk1828.yml \
  -f deploy/lamp/compose.override.yaml "$@"
