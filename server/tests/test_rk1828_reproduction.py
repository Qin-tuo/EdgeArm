"""A manufacturer diagnostic must stop at failure and respect worker counts."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import urllib.error

import pytest


@pytest.fixture
def case_runner(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'scripts/reproduce-rk1828-modes.py'
    spec = importlib.util.spec_from_file_location('rk1828_reproduction', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    case = tmp_path / '.local/reports/inference-check/vendor-logs/test-case'
    case.mkdir(parents=True)
    (case / 'case.json').write_text(json.dumps({'work_mode': 'efficient', 'prefill_mode': 'performance'}))
    monkeypatch.setattr(sys, 'argv', [str(path), str(case)])
    monkeypatch.setattr(module, 'proxy', lambda: ['test device proxy'])
    return module, case


def response(body):
    result = io.BytesIO(json.dumps(body).encode())
    result.status = 200
    return result


def test_failed_inference_stops_without_retry(case_runner, monkeypatch):
    module, case = case_runner
    sent = []

    def open_request(request, **kwargs):
        if isinstance(request, str):
            return response({'status': 'ok'})
        sent.append(request)
        raise urllib.error.HTTPError(request.full_url, 502, 'worker unavailable', {},
                                     io.BytesIO(b'{"error":"ERROR_PIPE"}'))

    monkeypatch.setattr(module.OPENER, 'open', open_request)
    assert module.main() == 1
    report = json.loads((case / 'requests.json').read_text())
    assert len(sent) == len(report['cases']) == 1
    assert report['status'] == 'failed'
    assert report['cases'][0]['http_status'] == 502
    assert 'ERROR_PIPE' in report['cases'][0]['error_body']
    assert report['auto_reset'] is False


def test_zero_api_usage_does_not_bypass_worker_history_bound(case_runner, monkeypatch):
    module, case = case_runner
    sent = []

    def open_request(request, **kwargs):
        if isinstance(request, str):
            return response({'status': 'ok'})
        sent.append(request)
        (case / 'rk3588-host.log').write_text(
            '[server] req#0 done prefill_tokens=1001 ttft_ms=100 decode_tokens=3\n')
        return response({'choices': [{'message': {'content': '你好！'}}],
                         'usage': {'prompt_tokens': 0, 'completion_tokens': 0}})

    monkeypatch.setattr(module.OPENER, 'open', open_request)
    assert module.main() == 0
    report = json.loads((case / 'requests.json').read_text())
    assert len(sent) == 1
    assert report['status'] == 'sample_passed'
    assert report['cases'][0]['worker_metrics']['prefill_tokens'] == 1001
    assert '1000-token' in report['stop_reason']


def test_application_initialization_failure_sends_no_inference(case_runner, monkeypatch):
    module, case = case_runner
    (case / 'host.exit-code').write_text('1\n')

    def unexpected_request(*args, **kwargs):
        pytest.fail('No HTTP request should be issued after application exit')

    monkeypatch.setattr(module.OPENER, 'open', unexpected_request)
    assert module.main() == 1
    report = json.loads((case / 'requests.json').read_text())
    assert report['cases'] == []
    assert 'initialization' in report['error']
