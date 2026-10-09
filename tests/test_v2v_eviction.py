"""Unit tests for /v2v admission-time eviction (server.main).

Covers the limit==1 single-client eviction path that reclaims a slot leaked
by a zombie holder (app-layer reader dead but WS protocol layer still alive).
"""

import asyncio

import pytest

from server.core import session_limiter, metrics
import server.main as main


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv("OVS_MAX_CONCURRENT_SESSIONS", raising=False)
    monkeypatch.delenv("OVS_V2V_EVICT_ON_FULL", raising=False)
    session_limiter._reset_for_tests()
    metrics._reset_for_tests()
    main._V2V_HOLDERS.clear()


class _FakeWS:
    def __init__(self):
        self.closed = False
        self.close_code = None

    async def close(self, code=None, reason=None):
        self.closed = True
        self.close_code = code


def test_evict_enabled_gating(monkeypatch):
    # OFF by default even at limit==1.
    session_limiter.init_limiter({"name": "orin-nano"})  # limit 1
    assert main._v2v_evict_enabled() is False
    # ON only when env set AND limit==1.
    monkeypatch.setenv("OVS_V2V_EVICT_ON_FULL", "1")
    session_limiter._reset_for_tests()
    session_limiter.init_limiter({"name": "orin-nano"})
    assert main._v2v_evict_enabled() is True
    # limit==2 (multi-tenant) → eviction stays off even with env on.
    session_limiter._reset_for_tests()
    session_limiter.init_limiter({"name": "orin-nx"})  # limit 2
    assert main._v2v_evict_enabled() is False


def test_evict_reclaims_zombie_slot(monkeypatch):
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "1")
    monkeypatch.setenv("OVS_V2V_EVICT_ON_FULL", "1")
    session_limiter.init_limiter({})

    async def scenario():
        # A zombie holder: holds the only slot; its handler task releases the
        # slot when cancelled (mirrors _v2v_release_early in the real finally).
        held_token = session_limiter.try_acquire_ws_token("/v2v/stream")[0]
        assert held_token is not None

        async def zombie_handler():
            try:
                await asyncio.sleep(3600)  # blocked like engine.run() on a dead peer
            except asyncio.CancelledError:
                held_token.release()       # finally-equivalent slot release
                raise

        task = asyncio.create_task(zombie_handler())
        await asyncio.sleep(0)  # let it start
        ws = _FakeWS()
        handle = main._WSHandle(websocket=ws, task=task)
        main._V2V_HOLDERS.add(handle)

        # Newcomer: slot is full → evict the zombie and reacquire.
        token, info = await main._v2v_evict_and_reacquire("/v2v/stream")
        assert token is not None, f"expected reacquired slot, got {info}"
        assert ws.closed and ws.close_code == 1012
        assert task.cancelled() or task.done()
        token.release()

    asyncio.run(scenario())


def test_evict_noop_when_no_holder(monkeypatch):
    """Slot full but held by a non-/v2v owner (e.g. /tts HTTP) → no eviction."""
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "1")
    monkeypatch.setenv("OVS_V2V_EVICT_ON_FULL", "1")
    sl = session_limiter.init_limiter({})
    foreign = sl.try_acquire()  # held by someone not in _V2V_HOLDERS

    async def scenario():
        token, info = await main._v2v_evict_and_reacquire("/v2v/stream", timeout_s=0.2)
        assert token is None
        assert info["reason"] == "too_many"

    asyncio.run(scenario())
    foreign.release()
