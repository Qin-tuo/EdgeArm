#!/usr/bin/env bash
# Chat-only bring-up until RK1828 tool calls have passed real board validation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
source scripts/activate-env.sh
if [[ "${1:-}" != --prepare-only ]]; then
  exec 9>"$EDGEARM_REPO_ROOT/.local/run/edgearm/native-agent.lock"
  flock -n 9 || { echo 'Native agent startup is already running.' >&2; exit 1; }
  python scripts/wait-model-services.py --check-agent-port
fi
mkdir -p .local/deploy
export EDGEARM_NATIVE_TOOLS="${EDGEARM_NATIVE_TOOLS:-0}"
python - <<'PY'
import os
from pathlib import Path
import yaml
config = yaml.safe_load(Path('deploy/agent-config.yaml').read_text())
config['metadata']['robot']['enabled'] = False
config['metadata']['robot']['actuator']['calibration_dir'] = str(
    Path('.local/data/agent-state/calibration').resolve())
if os.environ['EDGEARM_NATIVE_TOOLS'] != '1':
    config['tools_enabled'] = False
    config['tools_default_allowlist'] = []
    config['system_prompt'] = '你是本地语音助手，回答简短中文。'
Path('.local/deploy/agent-native.yaml').write_text(yaml.safe_dump(config, allow_unicode=True))
PY
[[ "${1:-}" != --prepare-only ]] || exit 0
python scripts/wait-model-services.py --timeout "${EDGEARM_SERVICE_WAIT_S:-180}"
python scripts/wait-model-services.py --check-agent-port
export OVS_AGENT_SERVER_LOOP=0 OVS_AGENT_SESSION_MAX_INPUT_TOKENS=1000
exec edgearm run edgearm --config "$ROOT/.local/deploy/agent-native.yaml"
