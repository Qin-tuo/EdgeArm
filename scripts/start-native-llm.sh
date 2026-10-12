#!/usr/bin/env bash
# Administrator privileges are needed by the current RK1828 device permissions.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
[[ "$EUID" == 0 ]] || { echo "Run with sudo: RK1828 requires administrator privileges." >&2; exit 1; }
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 RK1828_START_ATTEMPTS=1 RK1828_READY_TIMEOUT=180
# Use the worker rebuilt with the same V1.1.0 headers as this board's Runtime.
RUNTIME="${EDGEARM_NATIVE_LLM_RUNTIME:-$ROOT/.local/runtime/rk1828-v1.1.0}"
# Tool requests still disconnect the device with a matched SDK; reject them
# before sending any schema or prompt to the worker. KV reuse is off for the
# ordinary-dialogue baseline tested on this board.
export RK1828_ALLOW_TOOLS="${RK1828_ALLOW_TOOLS:-0}"
export RK1828_KV_REUSE="${RK1828_KV_REUSE:-0}"
MODELS="${EDGEARM_NATIVE_LLM_MODELS:-$ROOT/.local/data/llm}"
MANIFEST="${EDGEARM_NATIVE_LLM_MANIFEST:-$MODELS/artifacts.local.json}"
[[ -x "$RUNTIME/rknn_qwen3_demo" ]] || { echo "Missing native LLM worker: $RUNTIME/rknn_qwen3_demo" >&2; exit 1; }
for file in "$MANIFEST" "$RUNTIME/MANIFEST.json" "$RUNTIME/lib/librknn3_api.so" "$RUNTIME/lib/librknn3_api_rkcp.so"; do
  [[ -f "$file" ]] || { echo "Missing paired LLM artifact: $file" >&2; exit 1; }
done
"$ROOT/.venv/bin/python" - "$RUNTIME" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

runtime = Path(sys.argv[1]).resolve()
manifest = json.loads((runtime / 'MANIFEST.json').read_text())
for name, entry in manifest['files'].items():
    path = (runtime / name).resolve()
    if not path.is_relative_to(runtime):
        raise SystemExit(f'Invalid Runtime manifest path: {name}')
    if path.stat().st_size != entry['size_bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise SystemExit(f'Runtime integrity check failed: {name}')
print('PASS native LLM Runtime hashes.', flush=True)
PY
"$ROOT/.venv/bin/python" services/rk1828-llm/verify_artifacts.py "$MANIFEST" "$MODELS"
scripts/setup-rk1828-mode.sh
exec "$ROOT/.venv/bin/python" -u services/rk1828-llm/rk1828_llm_server.py \
  --binary "$RUNTIME/rknn_qwen3_demo" --model-dir "$MODELS" \
  --core-mask ff --max-context 2048 --host 127.0.0.1 --port 1828
