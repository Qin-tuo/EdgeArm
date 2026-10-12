"""Native startup tolerates delayed listening without accepting failed backends."""
import importlib.util
import io
import json
from pathlib import Path
import urllib.error

import pytest


@pytest.fixture
def waiter():
    spec = importlib.util.spec_from_file_location(
        'native_service_wait', Path(__file__).resolve().parents[2] / 'scripts/wait-model-services.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name,body,expected', [
    ('speech', {'asr': True, 'tts': True}, True),
    ('speech', {'asr': False, 'tts': True}, False),
    ('speech', {'asr': 1, 'tts': True}, False),
    ('llm', {'status': 'ok'}, True),
    ('llm', {'status': 'unavailable'}, False),
    ('llm', [], False),
])
def test_health_requires_ready_backends(waiter, monkeypatch, name, body, expected):
    monkeypatch.setattr(waiter.urllib.request, 'urlopen',
                        lambda *args, **kwargs: io.BytesIO(json.dumps(body).encode()))
    assert waiter.probe_service(name, 'http://localhost/health', 1)[0] is expected


def test_connection_refusal_is_waitable(waiter, monkeypatch):
    def refused(*args, **kwargs):
        raise urllib.error.URLError(ConnectionRefusedError(111, 'Connection refused'))
    monkeypatch.setattr(waiter.urllib.request, 'urlopen', refused)
    ready, detail = waiter.probe_service('llm', waiter.SERVICES['llm'], 1)
    assert not ready
    assert 'Connection refused' in detail


@pytest.fixture
def clock(waiter, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(waiter.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(waiter.time, 'sleep', lambda duration: now.__setitem__(0, now[0] + duration))
    return now


def test_waits_for_delayed_llm_listener(waiter, monkeypatch, clock):
    monkeypatch.setattr(waiter, 'probe_service', lambda name, url, timeout:
                        (name == 'speech' or clock[0] >= 2, 'starting' if clock[0] < 2 else 'ok'))
    assert waiter.wait_for_services(5)
    assert clock[0] == 2


def test_timeout_names_failed_service_without_traceback(waiter, monkeypatch, clock, capsys):
    monkeypatch.setattr(waiter, 'probe_service', lambda name, url, timeout:
                        (name == 'speech', 'ok' if name == 'speech' else 'HTTP 503'))
    assert not waiter.wait_for_services(3)
    output = capsys.readouterr().out
    assert '1828/health' in output and 'HTTP 503' in output
    assert 'Traceback' not in output
    assert clock[0] == 3


def test_services_must_be_ready_in_same_poll(waiter, monkeypatch, clock):
    monkeypatch.setattr(waiter, 'probe_service', lambda name, url, timeout:
                        ((name == 'speech') == (clock[0] < 1), 'changed'))
    assert not waiter.wait_for_services(3)


def test_existing_agent_listener_blocks_duplicate_start(waiter, monkeypatch, capsys):
    from unittest.mock import Mock
    connection = Mock()
    monkeypatch.setattr(waiter.socket, 'create_connection', lambda *args, **kwargs: connection)
    assert not waiter.agent_port_available()
    connection.close.assert_called_once()
    assert '18000 is already in use' in capsys.readouterr().out


def test_unused_agent_port_allows_start(waiter, monkeypatch):
    def refused(*args, **kwargs):
        raise ConnectionRefusedError(111, 'Connection refused')
    monkeypatch.setattr(waiter.socket, 'create_connection', refused)
    assert waiter.agent_port_available()
