"""Regression: SLVClient must read the SERVER_TOOL_CALL correlation id from
the wire field the server actually sends.

voxedge tool_registry emits {"type":"tool_call","call_id":...}. The parser
previously read evt["id"] only → empty id → send_tool_result echoed no
call_id → server resolve_remote never matched → every remote tool dispatch
blocked the full 15s timeout (the round2 "挥手 → 已挥手" stall). This drives
the real JSON path (not a hand-built ServerToolCall) to lock the field name.
"""
from __future__ import annotations

import asyncio
import json

from ovs_agent.slv_client import SLVClient, ServerToolCall


def _client():
    c = SLVClient.__new__(SLVClient)
    c._queue = asyncio.Queue()
    c._touch_activity = lambda: None  # type: ignore[attr-defined]
    return c


def _drive(raw: str) -> ServerToolCall:
    c = _client()

    async def run():
        await c._handle_json(raw)
        return await asyncio.wait_for(c._queue.get(), timeout=1.0)

    return asyncio.run(run())


def test_tool_call_reads_call_id_field():
    # The field name the server (voxedge) actually emits.
    evt = _drive(json.dumps({
        "type": "tool_call", "call_id": "efc38533", "name": "wave", "arguments": {},
    }))
    assert isinstance(evt, ServerToolCall)
    assert evt.id == "efc38533", "must capture the server's call_id (else 15s timeout)"
    assert evt.name == "wave"


def test_tool_call_falls_back_to_id_field():
    # Backward-compat: a frame using "id" still works.
    evt = _drive(json.dumps({
        "type": "tool_call", "id": "legacy1", "name": "home", "arguments": {"x": 1},
    }))
    assert evt.id == "legacy1"
    assert evt.arguments == {"x": 1}


def test_tool_call_prefers_call_id_over_id():
    evt = _drive(json.dumps({
        "type": "tool_call", "call_id": "primary", "id": "secondary", "name": "nod",
    }))
    assert evt.id == "primary"
