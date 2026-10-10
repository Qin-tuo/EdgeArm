#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.local/bin:$PATH"
export DOCKER_CONFIG="${DOCKER_CONFIG:-$ROOT/.local/docker-config}"
export EDGEARM_REPO_ROOT="$ROOT"
export EDGEARM_DATA_ROOT="${EDGEARM_DATA_ROOT:-$ROOT/.local/data}"
exec docker compose --env-file "${EDGEARM_ENV_FILE:-$ROOT/deploy/cm3588.env}" -f deploy/compose.yaml "$@"
