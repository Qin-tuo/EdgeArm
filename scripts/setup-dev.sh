#!/usr/bin/env bash
# Prepare the upstream application for development on the selected CM3588.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
mkdir -p .local/tmp .local/cache/pip .local/reports
mkdir -p .local/data/{asr,tts,llm,voice-data,agent-state}
export TMPDIR="$ROOT/.local/tmp"
export PIP_CACHE_DIR="$ROOT/.local/cache/pip"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-3}"
git submodule update --init --depth=1 third_party/rkvoice-stream
python3 -m venv .venv
mkdir -p .local/run/edgearm
chmod 700 .local/run/edgearm
constraints=deploy/constraints.txt
if [[ -f deploy/python.lock.txt ]]; then
  constraints=deploy/python.lock.txt
fi
.venv/bin/python -m pip install \
  -c "$constraints" \
  -r server/requirements.txt \
  -e './agent[dev]' \
  -e './third_party/rkvoice-stream'
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze --all > .local/reports/python-freeze.txt
.venv/bin/python -c 'import voxedge, ovs_agent, onnxruntime; print("Application imports OK; ORT", onnxruntime.__version__)'
