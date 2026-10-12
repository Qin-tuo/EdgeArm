"""Regression coverage for failures observed during native board inference."""
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from server.core import gpu_watchdog


@pytest.fixture
def npu_paths(monkeypatch, tmp_path):
    class DeniedDebugfs:
        def read_text(self, **kwargs):
            raise PermissionError('debugfs is root-only')

        def exists(self):
            raise PermissionError('debugfs is root-only')

    devfreq = tmp_path / 'devfreq'
    devfreq.mkdir()
    original = Path
    monkeypatch.setattr(gpu_watchdog, 'Path', lambda name: (
        DeniedDebugfs() if str(name).startswith('/sys/kernel/debug/') else
        devfreq if str(name) == '/sys/class/devfreq' else original(name)))
    return devfreq


def test_debugfs_permission_error_uses_npu_devfreq(npu_paths):
    npu = npu_paths / 'fdab0000.npu'
    npu.mkdir()
    (npu / 'cur_freq').write_text('1000000000\n')
    assert gpu_watchdog._check_rk() == (True, 'ok')


def test_gpu_and_ddr_frequencies_cannot_hide_missing_npu(npu_paths):
    for name in ('dmc', 'fb000000.gpu'):
        path = npu_paths / name
        path.mkdir()
        (path / 'name').write_text(name)
        (path / 'cur_freq').write_text('1000000000')
    assert gpu_watchdog._check_rk() == (False, 'probe_unavailable')


@pytest.mark.parametrize('frequency', ['0', '-1', 'invalid'])
def test_invalid_npu_frequency_is_not_healthy(npu_paths, frequency):
    path = npu_paths / 'fdab0000.npu'
    path.mkdir()
    (path / 'cur_freq').write_text(frequency)
    assert gpu_watchdog._check_rk() == (False, 'probe_unavailable')


@pytest.fixture
def llm_module(monkeypatch):
    monkeypatch.delenv('RK1828_ALLOW_TOOLS', raising=False)
    path = Path(__file__).resolve().parents[2] / 'services/rk1828-llm/rk1828_llm_server.py'
    spec = importlib.util.spec_from_file_location('rk_health_regression', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_transfer_failure_revokes_worker_readiness(llm_module):
    worker = llm_module.Qwen3Worker('/unused/demo', '/unused/models', 'ff', 2048)
    worker.proc = SimpleNamespace(
        stderr=io.BytesIO(b'READY 1\nE RKNNAPI: -9(ERROR_PIPE)\n'), poll=lambda: None)
    worker._drain_stderr()
    assert not worker.is_ready()


def test_already_message_is_not_ready_handshake(llm_module):
    worker = llm_module.Qwen3Worker('/unused/demo', '/unused/models', 'ff', 2048)
    worker.proc = SimpleNamespace(stderr=io.BytesIO(b'model already loaded\n'), poll=lambda: None)
    worker._drain_stderr()
    assert not worker.is_ready()


@pytest.mark.parametrize('ready,status', [(False, 503), (True, 200)])
def test_health_http_status_reflects_worker(llm_module, ready, status):
    llm_module.WORKER = SimpleNamespace(is_ready=lambda: ready)
    response = TestClient(llm_module.app).get('/health')
    assert response.status_code == status
    assert response.json()['status'] == ('ok' if ready else 'unavailable')


@pytest.mark.parametrize('stream', [False, True])
def test_empty_generation_cannot_report_success(llm_module, stream):
    llm_module.WORKER = SimpleNamespace(is_ready=lambda: True, generate=lambda *args, **kwargs: iter(()))
    response = TestClient(llm_module.app).post('/v1/chat/completions', json={
        'model': 'Qwen3-4B', 'messages': [{'role': 'user', 'content': '你好'}], 'stream': stream})
    if stream:
        assert 'worker produced no output' in response.text
        assert '"finish_reason": "stop"' not in response.text
        assert 'data: [DONE]' in response.text
    else:
        assert response.status_code == 502


def test_worker_failure_is_bad_gateway(llm_module):
    def failed(*args, **kwargs):
        raise llm_module.WorkerError('ERROR_PIPE')

    llm_module.WORKER = SimpleNamespace(is_ready=lambda: True, generate=failed)
    response = TestClient(llm_module.app).post('/v1/chat/completions', json={
        'model': 'Qwen3-4B', 'messages': [{'role': 'user', 'content': '你好'}]})
    assert response.status_code == 502


@pytest.mark.parametrize('stream', [False, True])
def test_tool_only_output_is_valid(llm_module, stream):
    text = '<tool_call>{"name":"robot_query","arguments":{}}</tool_call>'
    llm_module.WORKER = SimpleNamespace(
        is_ready=lambda: True, generate=lambda *args, **kwargs: iter([text]))
    response = TestClient(llm_module.app).post('/v1/chat/completions', json={
        'model': 'Qwen3-4B', 'messages': [{'role': 'user', 'content': '查询状态'}],
        'stream': stream, 'tools': [{'type': 'function', 'function': {
            'name': 'robot_query', 'parameters': {'type': 'object', 'properties': {}}}}]})
    assert response.status_code == 200
    if stream:
        assert '"finish_reason": "tool_calls"' in response.text
        assert '"error"' not in response.text
    else:
        choice = response.json()['choices'][0]
        assert choice['finish_reason'] == 'tool_calls'
        assert choice['message']['tool_calls'][0]['function']['name'] == 'robot_query'


@pytest.mark.parametrize('stream', [False, True])
def test_disabled_tools_never_reach_worker(llm_module, monkeypatch, stream):
    monkeypatch.setenv('RK1828_ALLOW_TOOLS', '0')

    def unexpected(*args, **kwargs):
        pytest.fail('Disabled tool request reached the NPU worker')

    llm_module.WORKER = SimpleNamespace(is_ready=lambda: True, generate=unexpected)
    client = TestClient(llm_module.app)
    response = client.post('/v1/chat/completions', json={
        'messages': [{'role': 'user', 'content': '调用工具'}], 'stream': stream,
        'tools': [{'type': 'function', 'function': {'name': 'echo'}}]})
    assert response.status_code == 400
    assert 'tool requests are disabled' in response.json()['detail']
    assert client.get('/health').status_code == 200


def test_disabled_tools_allow_ordinary_dialogue(llm_module, monkeypatch):
    monkeypatch.setenv('RK1828_ALLOW_TOOLS', '0')
    llm_module.WORKER = SimpleNamespace(
        is_ready=lambda: True, generate=lambda *args, **kwargs: iter(['你好']))
    response = TestClient(llm_module.app).post('/v1/chat/completions', json={
        'messages': [{'role': 'user', 'content': '你好'}]})
    assert response.status_code == 200
    assert response.json()['choices'][0]['message']['content'] == '你好'
