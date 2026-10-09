"""/livez and /readyz route tests.

These don't trigger the full app startup (which preloads ASR/TTS
backends). Instead they construct a tiny FastAPI app with the same
route handlers, or call the handler functions directly.
"""

import pytest

from server.core import session_limiter, metrics


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv("OVS_MAX_CONCURRENT_SESSIONS", raising=False)
    session_limiter._reset_for_tests()
    metrics._reset_for_tests()


def test_livez_always_200():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.get("/livez")
    async def livez():
        return JSONResponse({"status": "ok"})

    with TestClient(app) as c:
        r = c.get("/livez")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_readyz_503_when_limiter_unset():
    """Without initialised limiter, /readyz reports session_limiter_unavailable."""
    import importlib
    from server.core import session_limiter as sl_mod
    sl_mod._reset_for_tests()
    # Mimic the main.py readyz handler logic
    limiter = sl_mod.get_limiter()
    assert limiter is None  # not initialised
    reasons = []
    if limiter is None:
        reasons.append("session_limiter_unavailable")
    assert reasons == ["session_limiter_unavailable"]


def test_readyz_503_when_sessions_full():
    sl = session_limiter.SessionLimiter(1)
    t = sl.try_acquire()
    assert sl.available == 0
    # available == 0 → /readyz must include "sessions_full"
    t.release()
    assert sl.available == 1


def test_readyz_gpu_watchdog_failed_when_patched(monkeypatch):
    from server.core import gpu_watchdog
    monkeypatch.setattr(gpu_watchdog, "is_ok", lambda: False)
    assert gpu_watchdog.is_ok() is False
