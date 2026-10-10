#!/usr/bin/env bash
# Run the prepared development daemon in the foreground (administrator only).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$EUID" != 0 ]]; then
  echo "Administrator access required: sudo $ROOT/scripts/start-docker.sh" >&2
  exit 1
fi
export PATH="$ROOT/.local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
if pgrep -x dockerd >/dev/null; then
  echo 'A Docker daemon is already running; inspect its data-root before using it.' >&2
  exit 1
fi
mkdir -p "$ROOT/.local/docker-tmp" "$ROOT/.local/docker-exec"
export TMPDIR="$ROOT/.local/docker-tmp"
export DOCKER_TMPDIR="$TMPDIR"
exec dockerd --config-file "$ROOT/deploy/docker-daemon.json"
