from __future__ import annotations

import importlib.util
import io
import json
import sys
import wave
from pathlib import Path

import pytest


def _load_client_module():
    path = Path(__file__).resolve().parents[1] / "bench" / "perf" / "client.py"
    spec = importlib.util.spec_from_file_location("perf_client", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


class _RejectedWS:
    def __init__(self, websocket_module):
        self._websocket = websocket_module

    def settimeout(self, _timeout):
        pass

    def recv(self):
        raise self._websocket.WebSocketConnectionClosedException(
            "session_limiter rejected WS (slot busy, code 4429)"
        )

    def close(self):
        pass


class _GoodWS:
    def __init__(self, websocket_module):
        self._websocket = websocket_module
        self.timeout = None
        self.probed = False
        self.final_sent = False
        self.sent = []

    def settimeout(self, timeout):
        self.timeout = timeout

    def send(self, payload):
        self.sent.append(payload)

    def send_binary(self, payload):
        self.sent.append(payload)

    def recv(self):
        if not self.probed:
            self.probed = True
            raise self._websocket.WebSocketTimeoutException()
        if self.timeout and self.timeout < 0.01:
            raise self._websocket.WebSocketTimeoutException()
        if self.final_sent:
            raise self._websocket.WebSocketConnectionClosedException("closed")
        self.final_sent = True
        return json.dumps({"type": "asr_final", "text": "hello"})

    def close(self):
        pass


def test_run_v2v_stream_asr_retries_immediate_slot_reject(monkeypatch):
    client = _load_client_module()
    sockets = [_RejectedWS(client.websocket), _GoodWS(client.websocket)]
    opened = []

    def fake_create_connection(_url, timeout):
        opened.append(timeout)
        return sockets.pop(0)

    monkeypatch.setattr(client.websocket, "create_connection", fake_create_connection)
    monkeypatch.setattr(client.time, "sleep", lambda _seconds: None)

    result = client.run_v2v_stream_asr(
        "http://device:8621",
        _wav_bytes(),
        language="English",
        chunk_ms=100,
        realtime=False,
        timeout=2,
        eos_mode="client",
    )

    assert result.text == "hello"
    assert len(opened) == 2


def test_run_v2v_stream_asr_sends_endpoint_min_speech_config(monkeypatch):
    client = _load_client_module()
    ws = _GoodWS(client.websocket)

    monkeypatch.setattr(
        client.websocket,
        "create_connection",
        lambda _url, timeout: ws,
    )
    monkeypatch.setattr(client.time, "sleep", lambda _seconds: None)

    result = client.run_v2v_stream_asr(
        "http://device:8621",
        _wav_bytes(),
        language="Chinese",
        chunk_ms=100,
        realtime=False,
        timeout=2,
        eos_mode="vad",
        vad_silence_ms=300,
        asr_endpoint_min_speech_s=1.2,
        asr_endpoint_min_audio_s=2.0,
    )

    config = json.loads(ws.sent[0])
    assert result.text == "hello"
    assert config["vad_silence_ms"] == 300
    assert config["asr_endpoint_min_speech_s"] == 1.2
    assert config["asr_endpoint_min_audio_s"] == 2.0


class _EarlyFinalWS:
    def __init__(self, websocket_module):
        self._websocket = websocket_module
        self.timeout = None
        self.probed = False
        self.frames = [
            {"type": "asr_partial", "text": "do"},
            {"type": "asr_endpoint"},
            {"type": "asr_final", "text": "done"},
            {"type": "asr_final", "session_complete": True},
        ]

    def settimeout(self, timeout):
        self.timeout = timeout

    def send(self, _payload):
        pass

    def send_binary(self, _payload):
        pass

    def recv(self):
        if not self.probed:
            self.probed = True
            raise self._websocket.WebSocketTimeoutException()
        if not self.frames:
            raise self._websocket.WebSocketConnectionClosedException("closed")
        return json.dumps(self.frames.pop(0))

    def close(self):
        pass


def test_run_v2v_stream_asr_handles_final_during_audio_send(monkeypatch):
    client = _load_client_module()

    monkeypatch.setattr(
        client.websocket,
        "create_connection",
        lambda _url, timeout: _EarlyFinalWS(client.websocket),
    )
    monkeypatch.setattr(client.time, "sleep", lambda _seconds: None)

    result = client.run_v2v_stream_asr(
        "http://device:8621",
        _wav_bytes(),
        language="English",
        chunk_ms=100,
        realtime=False,
        timeout=2,
        eos_mode="client",
    )

    assert result.text == "done"
    assert result.partial_before_client_eos is True
    assert result.endpoint_before_client_eos is True
    assert result.final_before_client_eos is True
    assert result.endpoint_latency_ms == 0
    assert result.asr_finalize_ms == 0
    assert result.total_latency_ms == 0
