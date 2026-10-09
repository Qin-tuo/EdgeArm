"""Integration tests for /admin/* routes."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    # Reset overrides between tests
    from server.core import tts_runtime
    tts_runtime.reset_overrides()

    # Stub tts_service so admin routes can resolve a model_id without loading.
    from server.core import tts_service

    fake_backend = MagicMock()
    fake_backend.model_id = "qwen3-tts"
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: fake_backend)

    from server.main import app
    from server.core.admin_auth import require_admin

    async def _allow():
        return None

    app.dependency_overrides[require_admin] = _allow

    # Don't use `with` — avoids triggering the heavy startup event
    # (model downloader, profile load, etc) which needs /opt/models.
    c = TestClient(app)
    try:
        yield c
    finally:
        app.dependency_overrides.pop(require_admin, None)
        tts_runtime.reset_overrides()


def test_get_runtime_returns_shape(client):
    r = client.get("/admin/tts/runtime")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_id"] == "qwen3-tts"
    assert body["overrides"] == {
        "speaker_id": None, "speed": None, "pitch_shift": None,
        "updated_at": 0.0,
    }
    # Effective falls back to backend default for qwen3-tts (=0).
    assert body["effective"]["speaker_id"] == 0


def test_patch_runtime_sets_speaker_then_get_reflects(client):
    r = client.patch("/admin/tts/runtime", json={"speaker_id": 2301})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["overrides"]["speaker_id"] == 2301
    assert body["effective"]["speaker_id"] == 2301

    g = client.get("/admin/tts/runtime").json()
    assert g["overrides"]["speaker_id"] == 2301
    assert g["effective"]["speaker_id"] == 2301


def test_patch_runtime_explicit_null_clears(client):
    client.patch("/admin/tts/runtime", json={"speaker_id": 2301, "speed": 1.5})
    r = client.patch("/admin/tts/runtime", json={"speed": None})
    assert r.status_code == 200
    body = r.json()
    assert body["overrides"]["speaker_id"] == 2301  # untouched
    assert body["overrides"]["speed"] is None


def test_patch_runtime_invalid_speed_returns_422(client):
    r = client.patch("/admin/tts/runtime", json={"speed": 99.0})
    assert r.status_code == 422


def test_patch_runtime_invalid_speaker_returns_422(client):
    r = client.patch("/admin/tts/runtime", json={"speaker_id": 999999})
    assert r.status_code == 422


def test_reload_speakers_route(client):
    r = client.post("/admin/tts/speakers/reload")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reloaded"] is True
    assert body["model_id"] == "qwen3-tts"
    assert body["count"] >= 1


def test_remote_client_without_key_forbidden(monkeypatch):
    """Non-loopback client (TestClient host=testclient) without OVS_ADMIN_KEY → 403."""
    monkeypatch.delenv("OVS_ADMIN_KEY", raising=False)

    from server.core import tts_service, tts_runtime
    fake_backend = MagicMock()
    fake_backend.model_id = "qwen3-tts"
    monkeypatch.setattr(tts_service, "is_ready", lambda: True)
    monkeypatch.setattr(tts_service, "get_backend", lambda: fake_backend)
    tts_runtime.reset_overrides()

    from server.main import app
    # No dependency override → real require_admin runs. TestClient's default
    # client.host is "testclient", which fails the loopback check and triggers 403.
    c = TestClient(app)
    r = c.get("/admin/tts/runtime")
    assert r.status_code == 403
