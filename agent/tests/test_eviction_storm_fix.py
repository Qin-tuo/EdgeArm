"""Regression tests for the eviction↔reconnect self-excitation storm fix.

Root cause (prod 2026-06-01): the dispatch reader-exit path reconnected
unconditionally; with admission eviction (limit=1) two reconnect paths
(dispatch + wake) opened overlapping WS that evicted each other (1012
"superseded"), each eviction's reader-exit triggering another reconnect →
a storm that tore the WS down mid-utterance and wedged the ASR worker.

Fix: (1) slv_client.reconnect() serialized by _reconnect_lock (no overlapping
opens → no eviction) WITHOUT short-circuiting on is_healthy (wake's idle
refresh must still force a fresh session); (2) _reader_loop captures the
close code/reason so the dispatch can tell an eviction from a real death;
(3) the dispatch guard resumes (continue) instead of reconnecting when the WS
is already healthy / a reconnect is in flight / the close was a 1012 eviction.
"""
from __future__ import annotations

import asyncio
import json

import pytest
import websockets
from websockets.asyncio.server import serve

from ovs_agent import Config, Session
from ovs_agent.app_base import BaseApp
from ovs_agent.slv_client import SLVClient, SLVReconnectError
from ovs_agent.state import ConvState
from ovs_agent.tools import ToolRegistry


# ── mock /v2v/stream server ───────────────────────────────────────────

async def _serve(handler):
    server = await serve(handler, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    return server, f"ws://{host}:{port}"


# ── 1. reader captures eviction close code/reason ─────────────────────

@pytest.mark.asyncio
async def test_reader_captures_eviction_close_code():
    async def handler(ws):
        await ws.recv()  # config
        # Send one event + survive the reconnect grace window (0.05s) so
        # connect() succeeds, THEN simulate an admission eviction mid-stream.
        await ws.send(json.dumps({"type": "asr_partial", "text": "x", "is_stable": False}))
        await asyncio.sleep(0.2)
        await ws.close(code=1012, reason="evicted: superseded by new session")

    server, url = await _serve(handler)
    client = SLVClient(url, {"asr_language": "zh"})
    try:
        await client.connect()
        # Drain events() until the reader exits (close observed).
        async def drain():
            async for _ in client.events():
                pass
        await asyncio.wait_for(drain(), timeout=3.0)
        assert client.last_close_code() == 1012
        assert "superseded" in (client.last_close_reason() or "").lower()
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


# ── 2. reconnect() still forces a fresh WS when healthy (wake refresh) ─
# Guards against the is_healthy() short-circuit regression that would have
# made wake()'s idle>30s refresh a no-op (silent-mute-after-idle bug).

@pytest.mark.asyncio
async def test_reconnect_forces_fresh_ws_even_when_healthy():
    conns = {"n": 0}

    async def handler(ws):
        conns["n"] += 1
        await ws.recv()  # config
        try:
            async for _ in ws:
                pass
        except websockets.ConnectionClosed:
            return

    server, url = await _serve(handler)
    client = SLVClient(url, {"asr_language": "zh"})
    try:
        await client.connect()
        assert client.is_healthy()
        assert conns["n"] == 1
        # Healthy connection — reconnect() MUST still open a fresh one.
        await asyncio.wait_for(client.reconnect(), timeout=5.0)
        assert client.is_healthy()
        assert conns["n"] == 2, "reconnect() no-op'd on a healthy WS (wake refresh broken)"
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


# ── 3. concurrent reconnect() calls are serialized (no overlap) ───────

@pytest.mark.asyncio
async def test_concurrent_reconnect_serialized():
    overlap = {"max": 0, "cur": 0}

    async def handler(ws):
        overlap["cur"] += 1
        overlap["max"] = max(overlap["max"], overlap["cur"])
        try:
            await ws.recv()  # config
            async for _ in ws:
                pass
        except websockets.ConnectionClosed:
            pass
        finally:
            overlap["cur"] -= 1

    server, url = await _serve(handler)
    client = SLVClient(url, {"asr_language": "zh"})
    try:
        await client.connect()
        # Fire two reconnects at once: the lock must serialize them so they
        # never hold two live server connections simultaneously (which is
        # exactly what triggers admission eviction → storm).
        await asyncio.gather(client.reconnect(), client.reconnect())
        assert client.is_healthy()
        assert overlap["max"] <= 1, f"reconnects overlapped (max={overlap['max']}) → eviction risk"
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


# ── 4. dispatch guard: resume vs reconnect decision ───────────────────

class _GuardSLV:
    """SLV double whose events() returns immediately (reader exited), with
    scriptable liveness so we can assert the dispatch guard's decision."""

    def __init__(self, *, healthy, code, reason, reconnecting):
        self._closed = False
        self._healthy = healthy
        self._code = code
        self._reason = reason
        self._reconnecting = reconnecting
        self.reconnect_calls = 0

    async def events(self):
        # Reader already exited — yield nothing. Sleep briefly so the loop
        # yields control (in production events() blocks on the queue; this
        # fake returns instantly, so without the await a `continue` guard
        # would tight-spin and starve the test's timeout).
        await asyncio.sleep(0.02)
        if False:
            yield  # pragma: no cover  (make this an async generator)
        return

    def is_healthy(self) -> bool:
        return self._healthy

    def is_reconnecting(self) -> bool:
        return self._reconnecting

    def last_close_code(self):
        return self._code

    def last_close_reason(self):
        return self._reason

    async def reconnect(self) -> None:
        self.reconnect_calls += 1
        # Raise so the dispatch's except-branch handles it (avoids running the
        # reconnect-success body, which needs more app wiring than this double).
        raise SLVReconnectError("test")

    async def advertise_tools(self, *a, **k):
        pass


def _make_app(slv) -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.config = Config(system_prompt="SYS", server_loop=True, llm_model="m")
    app.tool_registry = ToolRegistry()
    app.session = Session()
    app.modes = None
    app.slv = slv
    app._slv_reconnect_count = 0

    async def _bc(*a, **k):
        return None
    app._broadcast = _bc  # type: ignore[assignment]
    return app


async def _run_dispatch_briefly(app, dur=0.4):
    task = asyncio.create_task(app._slv_dispatch())
    await asyncio.sleep(dur)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("healthy,code,reason,reconnecting,expect_reconnect", [
    (True, None, "", False, False),                      # fresh WS already up → resume
    (False, 1012, "superseded by new session", False, False),  # eviction → resume
    (False, None, "", True, False),                      # reconnect in flight → resume
    (False, 1006, "", False, True),                      # genuine death → reconnect
])
async def test_dispatch_guard_decision(healthy, code, reason, reconnecting, expect_reconnect):
    slv = _GuardSLV(healthy=healthy, code=code, reason=reason, reconnecting=reconnecting)
    app = _make_app(slv)
    await _run_dispatch_briefly(app)
    if expect_reconnect:
        assert slv.reconnect_calls >= 1, "genuine death should reconnect"
    else:
        assert slv.reconnect_calls == 0, "must resume (continue), not reconnect → storm"
