#!/usr/bin/env bash
# Source this file from Bash to activate the project development environment.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Use: source /userdata/EdgeArm/scripts/activate-env.sh" >&2
  exit 1
fi
export EDGEARM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ ! -f "$EDGEARM_REPO_ROOT/.venv/bin/activate" ]]; then
  echo "Create the environment first: $EDGEARM_REPO_ROOT/scripts/setup-dev.sh" >&2
  return 1
fi
# ROS injects its packages through PYTHONPATH; keep this environment isolated.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
source "$EDGEARM_REPO_ROOT/.venv/bin/activate"
export PATH="$EDGEARM_REPO_ROOT/.local/bin:$PATH"
export DOCKER_CONFIG="$EDGEARM_REPO_ROOT/.local/docker-config"
export TMPDIR="$EDGEARM_REPO_ROOT/.local/tmp"
export PIP_CACHE_DIR="$EDGEARM_REPO_ROOT/.local/cache/pip"
export CONFIG_DIR="$EDGEARM_REPO_ROOT/deploy"
export EDGEARM_CONTROL_SOCKET="$EDGEARM_REPO_ROOT/.local/run/edgearm/control.sock"
export OVS_SILERO_VAD_ONNX="$EDGEARM_REPO_ROOT/server/core/assets/silero_vad.onnx"
export OVS_DEBUG_DASHBOARD_BIND=127.0.0.1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$EDGEARM_REPO_ROOT/.local/run/edgearm"
chmod 700 "$EDGEARM_REPO_ROOT/.local/run/edgearm"
