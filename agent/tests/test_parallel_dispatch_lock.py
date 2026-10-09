"""Plan D item 5 — parallel-mode dispatch latency monitoring.

Contract being tested (registry.py + runner.py):
  * Tools declared with ``response_mode="parallel"`` MUST return their
    ``{"started": True}`` stub within ~200ms (hard ceiling ~500ms).
  * The runner logs a WARNING when a parallel-mode tool's dispatch
    exceeds ``_PARALLEL_DISPATCH_BUDGET_MS`` (currently 500ms). This is
    the monitoring hook that flags "you wrote a 'parallel' tool that
    actually blocks" — without it, parallel-mode design degrades
    silently to await-mode latency.

Scenario: two parallel-mode tools share an ``asyncio.Lock`` (e.g. a
single serial port to the arm). Tool A dispatches fast (acquires the
lock + kicks off a worker). Tool B's dispatch then blocks on the lock.
The runner serialises dispatches in a loop, so B's wait shows up as
``dt_ms`` and (once long enough) triggers the >500ms WARNING.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from ovs_agent.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_parallel_dispatch_fast_returns_started_stub():
    """Sanity: a well-behaved parallel tool returns quickly with started=True."""
    registry = ToolRegistry()
    serial_lock = asyncio.Lock()

    @registry.tool(
        name="fast_action",
        description="kick off a fast action",
        response_mode="parallel",
    )
    async def quick_dispatch(ctx=None) -> dict:
        # Acquire-then-release in microseconds, return the stub.
        async with serial_lock:
            pass
        return {"started": True}

    t0 = asyncio.get_event_loop().time()
    result = await registry.dispatch("fast_action", {}, None)
    dt_ms = (asyncio.get_event_loop().time() - t0) * 1000.0

    assert result == {"started": True}
    # Hard contract: parallel dispatch returns well within 200ms when
    # the side-effect handoff is non-blocking.
    assert dt_ms < 100.0, f"parallel dispatch unexpectedly slow: {dt_ms:.0f}ms"


@pytest.mark.asyncio
async def test_parallel_dispatch_blocked_by_shared_lock_exceeds_budget(caplog):
    """If tool B's dispatch waits on a lock held by A's background worker,
    B's dispatch time should exceed the 500ms parallel budget — that's
    the WARNING the runner is supposed to surface.

    We assert the *latency contract* directly here. The runner's WARNING
    emission is integration-tested via the runner; this test isolates
    the cause (shared-lock race) so a regression that, e.g., changes
    lock semantics surfaces here.
    """
    registry = ToolRegistry()
    serial_lock = asyncio.Lock()
    a_holding = asyncio.Event()

    @registry.tool(
        name="tool_a", description="hold the port", response_mode="parallel",
    )
    async def long_holder(ctx=None) -> dict:
        # Tool A: grab the lock, kick off a "background worker" that
        # holds it for 600ms (simulates a real arm motion).
        await serial_lock.acquire()
        a_holding.set()

        async def _release_after():
            await asyncio.sleep(0.6)
            serial_lock.release()

        asyncio.create_task(_release_after())
        # dispatch_action itself returns immediately.
        return {"started": True}

    @registry.tool(
        name="tool_b", description="needs the same port",
        response_mode="parallel",
    )
    async def needs_same_port(ctx=None) -> dict:
        # Tool B: needs the same lock. Its dispatch will block until A's
        # worker releases — which is exactly the latency we want to
        # detect.
        async with serial_lock:
            pass
        return {"started": True}

    # Dispatch A first: fast.
    t_a = asyncio.get_event_loop().time()
    r_a = await registry.dispatch("tool_a", {}, None)
    a_ms = (asyncio.get_event_loop().time() - t_a) * 1000.0
    assert r_a == {"started": True}
    assert a_ms < 100.0
    # A's worker is now holding the lock.
    assert a_holding.is_set()

    # Dispatch B: blocks on the lock until A's worker releases at ~600ms.
    t_b = asyncio.get_event_loop().time()
    r_b = await registry.dispatch("tool_b", {}, None)
    b_ms = (asyncio.get_event_loop().time() - t_b) * 1000.0

    assert r_b == {"started": True}
    # B's dispatch should be > 500ms (the runner's WARNING threshold).
    assert b_ms > 500.0, (
        f"expected B to wait >500ms on the shared lock; got {b_ms:.0f}ms — "
        "if the lock semantics changed, the runner WARNING gating may "
        "also need re-tuning."
    )
