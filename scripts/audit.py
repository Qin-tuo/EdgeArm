#!/usr/bin/env python3
"""Cross-check product scope, imports, deploy, locks and local Markdown links."""
import hashlib
import json
from pathlib import Path
import re
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'agent')]
from ovs_agent.config import load_config
from ovs_agent.cli import _load_app_class
from server.core import asr_backend, tts_backend

lock = json.loads((ROOT / 'deploy/baseline.lock.json').read_text())
compose = json.loads((ROOT / '.local/reports/so101-compose.json').read_text())
services = compose['services']
assert set(services) == {'speech', 'llm', 'agent'}
assert [p.name for p in (ROOT / 'configs/profiles').glob('*.json')] == ['rk3588-default.json']
assert {p.name for p in (ROOT / 'agent/ovs_agent/apps').iterdir() if p.is_dir() and not p.name.startswith('__')} == {'voice_so101'}
assert set(asr_backend._ASR_REGISTRY) == {'rk.asr'}
assert set(tts_backend._TTS_REGISTRY) == {'rk.tts'}
assert services['agent']['command'][2] == 'voice_so101'
assert services['speech']['environment']['OVS_MAX_CONCURRENT_SESSIONS'] == '1'
assert services['agent']['environment']['OVS_DEBUG_DASHBOARD_BIND'] == '127.0.0.1'
assert services['llm']['environment']['RK1828_MAX_CONTEXT'] == '2048'
assert services['llm']['environment']['RK1828_START_ATTEMPTS'] == '1'
for name in services:
    dockerfile = ROOT / services[name]['build']['dockerfile']
    assert f'ARG BASE_IMAGE={lock["images"][name]["immutable_reference"]}' in dockerfile.read_text()
    assert services[name]['memswap_limit'] == services[name]['mem_limit']
for path, expected in lock['model_manifests_sha256'].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
for path, spec in lock['submodules'].items():
    actual = subprocess.check_output(['git', '-C', str(ROOT / path), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == spec['commit'], path
for path in ['deploy/agent-config.yaml', 'agent/ovs_agent/apps/voice_so101/config.yaml']:
    config = load_config(ROOT / path)
    assert config.llm_backend == 'openai_compat' and config.llm_base_url == 'http://127.0.0.1:1828/v1'
    assert config.llm_model == 'Qwen3-4B' and config.barge_in_enabled is False
    assert config.pipeline_mode == 'push_to_talk' and config.metadata['robot']['enabled'] is False
    assert config.tools_enabled and config.tools_max_iterations == 3
    assert set(config.tools_default_allowlist) == {'robot_query', 'robot_grasp_place', 'robot_cancel'}
    app = _load_app_class('voice_so101')(config)
    assert app._client_vad is not None and app._client_vad.name == 'silero'
    assert app._client_vad.is_speech(b'\x00\x00' * 512) is False
    assert app.robot.arm is None
    assert set(app.tool_registry.list_names()) == {'robot_query', 'robot_grasp_place', 'robot_cancel'}
    assert [x['name'] for x in app.modes.list_all()] == ['chat']
    app.robot._worker.shutdown(wait=True)
assert 'sounddevice' not in sys.modules, 'configuration/import checks touched native audio'
assert 'torch' not in sys.modules, 'Silero ONNX must not import torch'
from server import main
assert any(getattr(route, 'path', None) == '/v2v/stream' for route in main.app.routes)
# Verify documented local links, ignoring upstream dependency docs and external URLs.
for doc in list(ROOT.glob('*.md')) + list((ROOT / 'docs').rglob('*.md')) + [ROOT / 'agent/README.md']:
    for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
        target = target.strip('<>').split('#', 1)[0]
        if not target or '://' in target:
            continue
        candidate = Path(target) if target.startswith('/') else doc.parent / target
        assert candidate.exists(), f'{doc.relative_to(ROOT)}: broken link {target}'
print('Product imports, tool surface, single deploy/profile, locks, source overlays and local links: OK')
