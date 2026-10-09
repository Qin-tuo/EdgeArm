"""Regression tests for send-path session revival (#3) and stale tool_result
misdelivery (#6).

#3: the mic pump's ``send_audio`` revives a dead WS via ``slv.connect()`` —
which opens the socket + sends config but does NOT advertise tools. ``events()``
keeps streaming on the new reader without the dispatch loop reaching its
reconnect/readvertise guard, so the server-loop LLM on the fresh session has
zero tools and the first tool_call silently fails. Fix: ``SLVClient`` exposes a
monotonic ``session_gen()`` (bumped each healthy open); the app re-advertises in
``_dispatch_one`` the moment it sees an event from a generation it hasn't
advertised to.

#6: ``send_tool_result`` must NOT auto-connect when the WS is dead — the call_id
is bound to the session that issued the SERVER_TOOL_CALL; delivering it to a
fresh session is a silent no-op that stalls the turn. It must drop instead.
"""
from __future__ import annotations

import asyncio
import json

import pytest
import websockets
from websockets.asyncio.server import serve

from ovs_agent import Config, Session
from ovs_agent.app_base import BaseApp
from ovs_agent.slv_client import SLVClient
from ovs_agent.tools import ToolRegistry


async def _serve(handler):
    server = await serve(handler, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    return server, f"ws://{host}:{port}"


# ── 1. session_gen bumps on each healthy open ─────────────────────────

@pytest.mark.asyncio
async def test_session_gen_bumps_on_each_open():
    async def handler(ws):
        await ws.recv()  # config
        try:
            async for _ in ws:
                pass
        except websockets.ConnectionClosed:
            return

    server, url = await _serve(handler)
    client = SLVClient(url, {"asr_language": "zh"})
    try:
        assert client.session_gen() == 0
        await client.connect()
        assert client.session_gen() == 1
        await asyncio.wait_for(client.reconnect(), timeout=5.0)
        assert client.session_gen() == 2, "reconnect must advance session_gen"
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


# ── 2. send_tool_result drops (no auto-connect) on a dead WS (#6) ──────

@pytest.mark.asyncio
async def test_send_tool_result_drops_on_dead_ws():
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
        assert conns["n"] == 1
        # Simulate the WS having died mid-dispatch (send observed
        # ConnectionClosed and nulled the handle).
        client._ws = None
        # A tool_result for the dead session must NOT spin up a new connection.
        await client.send_tool_result("call_old", "arm", ok=True, result={"x": 1})
        await asyncio.sleep(0.1)
        assert conns["n"] == 1, "send_tool_result auto-connected → would misdeliver call_id"
        # A normal send_audio, by contrast, SHOULD revive (mic pump path).
        await client.send_audio(b"\x00\x00")
        await asyncio.sleep(0.1)
        assert conns["n"] == 2, "send_audio must still auto-revive the session"
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


# ── 3. dispatch re-advertises when session gen advanced (#3) ──────────

class _GenSLV:
    """SLV double with a scriptable session_gen and an advertise spy."""

    def __init__(self, gen: int):
        self._gen = gen
        self.advertise_calls = 0

    def session_gen(self) -> int:
        return self._gen

    def set_gen(self, g: int) -> None:
        self._gen = g

    async def advertise_tools(self, *a, **k):
        self.advertise_calls += 1


def _make_app(slv, advertised_gen):
    app = BaseApp.__new__(BaseApp)
    app.config = Config(system_prompt="SYS", server_loop=True, llm_model="m")
    app.tool_registry = ToolRegistry()
    app.session = Session()
    app.modes = None
    app.slv = slv
    app._advertised_gen = advertised_gen

    async def _bc(*a, **k):
        return None
    app._broadcast = _bc  # type: ignore[assignment]
    return app


@pytest.mark.asyncio
async def test_readvertise_only_when_gen_advanced():
    # Same generation as last advertised → no re-advertise (steady state).
    slv = _GenSLV(gen=1)
    app = _make_app(slv, advertised_gen=1)
    await app._readvertise_if_session_advanced()
    assert slv.advertise_calls == 0, "must not re-advertise on the same session"

    # A send-path revival bumped the gen past what we advertised → re-advertise,
    # and the marker advances so we don't keep re-advertising every event.
    slv.set_gen(2)
    await app._readvertise_if_session_advanced()
    assert slv.advertise_calls == 1, "must re-advertise on a fresh un-advertised session"
    assert app._advertised_gen == 2, "advertise must record the new generation"
    await app._readvertise_if_session_advanced()
    assert slv.advertise_calls == 1, "must not re-advertise again on the same generation"


@pytest.mark.asyncio
async def test_no_readvertise_when_server_loop_off():
    slv = _GenSLV(gen=5)
    app = _make_app(slv, advertised_gen=1)
    app.config = Config(system_prompt="SYS", server_loop=False, llm_model="m")
    await app._readvertise_if_session_advanced()
    assert slv.advertise_calls == 0, "client-loop mode never advertises"
