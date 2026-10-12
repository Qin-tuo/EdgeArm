#!/usr/bin/env bash
# Native deployment using the selected profile and separately verified Runtime.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
source scripts/activate-env.sh
export EDGEARM_NATIVE_RUNTIME="${EDGEARM_NATIVE_RUNTIME:-$ROOT/.local/runtime/speech}"
export EDGEARM_NATIVE_DATA="${EDGEARM_NATIVE_DATA:-$ROOT/.local/data}"
export RK_RUNTIME_MANIFEST="$EDGEARM_NATIVE_RUNTIME/MANIFEST.json"
export RKNNRT_LIB_PATH="$EDGEARM_NATIVE_RUNTIME/librknnrt.so"
for file in "$RK_RUNTIME_MANIFEST" "$RKNNRT_LIB_PATH" "$EDGEARM_NATIVE_RUNTIME/librkllmrt.so"; do
  [[ -f "$file" ]] || { echo "Missing verified native Runtime: $file" >&2; exit 1; }
done
export OVS_PROFILE_JSON="$ROOT/.local/deploy/rk3588-native.json"
mkdir -p .local/deploy
python - <<'PY'
import json, os
from pathlib import Path
profile = json.loads(Path('configs/profiles/rk3588-default.json').read_text())
data = Path(os.environ['EDGEARM_NATIVE_DATA']).resolve()
runtime = Path(os.environ['EDGEARM_NATIVE_RUNTIME']).resolve()
profile['name'] = 'rk3588-native-inference'
profile['env'].update(ASR_MODEL_DIR=str(data / 'asr'), TTS_MODEL_DIR=str(data / 'tts'),
    SHERPA_TTS_MODEL_DIR=str(data / 'tts/matcha-icefall-zh-en'), MODEL_DIR=str(data / 'tts'),
    RKLLM_LIB_PATH=str(runtime / 'librkllmrt.so'))
Path(os.environ['OVS_PROFILE_JSON']).write_text(json.dumps(profile, ensure_ascii=False, indent=2))
PY
export RK_ARTIFACT_AUTO_DOWNLOAD=0 OVS_MAX_CONCURRENT_SESSIONS=1
export LD_LIBRARY_PATH="$EDGEARM_NATIVE_RUNTIME${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec python -u -m uvicorn server.main:app --host 127.0.0.1 --port 8621
