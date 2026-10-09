#!/usr/bin/env bash
# Configuration and upstream software checks only; no hardware operations.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.local/bin:$ROOT/.venv/bin:$PATH"
export TMPDIR="$ROOT/.local/tmp"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" .local/reports
.venv/bin/python -m pip check
dockerd --validate --config-file deploy/lamp/docker-daemon.json
scripts/lamp/compose.sh config --format json > .local/reports/compose.json
.venv/bin/python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
from ovs_agent.config import load_config

lock = json.loads(Path('deploy/lamp/baseline.lock.json').read_text())
config = json.loads(Path('.local/reports/compose.json').read_text())
services = config['services']
for service in ['speech', 'agent', 'llm']:
    expected = lock['images'][service]['immutable_reference']
    assert services[service]['image'] == expected, (service, 'image differs from baseline lock')
assert services['profile-init']['image'] == services['speech']['image']
for volume in config['volumes'].values():
    location = Path(volume['driver_opts']['device'])
    assert location.is_relative_to('/userdata'), ('volume outside /userdata', location)
    assert location.is_dir(), ('missing volume directory', location)
mounts = {v['target']: v['source'] for v in services['agent']['volumes']}
assert Path(mounts['/opt/voice/config/agent-config.yaml']).resolve() == Path('deploy/lamp/agent-config.yaml').resolve()
assert services['llm']['environment']['RK1828_MAX_CONTEXT'] == '2048'
assert services['agent']['environment']['OVS_DEBUG_DASHBOARD_BIND'] == '127.0.0.1'
os.environ.update({k: str(v) for k, v in services['agent']['environment'].items() if v is not None})
agent = load_config('deploy/lamp/agent-config.yaml')
assert agent.llm_base_url == 'http://127.0.0.1:1828/v1'
assert agent.llm_model == 'Qwen3-4B'
assert agent.barge_in_enabled is False
for file, expected in lock['model_manifests_sha256'].items():
    assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == expected, file
for path, spec in lock['submodules'].items():
    if spec['required']:
        actual = subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()
        assert actual == spec['commit'], (path, actual)
print('Baseline configuration, image digests, model manifests and submodule pin: OK')
PY
.venv/bin/python -m pytest -q \
  server/tests/test_rk_profile_contract.py \
  server/tests/test_rk_matrix_profile_manifest_consistency.py \
  server/tests/test_rk_profile_streaming_mode.py \
  server/tests/test_profile_loader.py \
  agent/tests/test_conversation_app.py \
  agent/tests/test_config_unknown_keys.py \
  agent/tests/test_slv_config_types.py \
  agent/tests/test_asr_segment_accumulation.py \
  agent/tests/test_llm_availability_health_probe.py \
  agent/tests/test_wake_phrase_strip.py \
  agent/tests/test_tools_registry.py \
  agent/tests/test_session_trim_budget.py \
  --junitxml=.local/reports/baseline-tests.xml \
  | tee .local/reports/baseline-tests.log
