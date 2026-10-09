"""silero-primary stall fallback (config.vad_stall_eos_ms).

Resets on each real asr_partial; forces ONE asr_eos if silero goes quiet
without a final while still awaiting a command. Inactivity-based, so a long
sentence (partials keep flowing) is never cut — only a true stall fires.
"""
from __future__ import annotations

import asyncio

from ovs_agent.app_base import BaseApp
from ovs_agent.state import ConvState


def _app(stall_ms: float, state: ConvState = ConvState.IDLE) -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app._state = state
    app._stall_watchdog_task = None
    app._asr_watchdog_task = None
    app._eos_sent_this_turn = False
    app._eos_calls = 0

    class _Cfg:
        vad_stall_eos_ms = stall_ms
    app.config = _Cfg()

    # Stub send_asr_eos_once to record calls (real one needs an slv + arms
    # another watchdog; we only assert the fallback *invokes* it once).
    async def _send():
        if app._eos_sent_this_turn:
            return False
        app._eos_sent_this_turn = True
        app._eos_calls += 1
        return True
    app.send_asr_eos_once = _send  # type: ignore[assignment]
    return app


def test_fires_eos_after_stall():
    app = _app(stall_ms=60)  # short for the test

    async def run():
        app._arm_stall_watchdog()
        await asyncio.sleep(0.15)  # well past 60ms with no further partial
        return app._eos_calls
    assert asyncio.run(run()) == 1


def test_partial_resets_timer_no_premature_fire():
    app = _app(stall_ms=80)

    async def run():
        # Simulate a long utterance: re-arm every 40ms (< 80ms) five times.
        for _ in range(5):
            app._arm_stall_watchdog()
            await asyncio.sleep(0.04)
        mid = app._eos_calls  # should still be 0 — kept resetting
        await asyncio.sleep(0.15)  # now go quiet → fires
        return mid, app._eos_calls
    mid, after = asyncio.run(run())
    assert mid == 0, "must not fire while partials keep arriving (long sentence)"
    assert after == 1, "fires once after silero goes quiet"


def test_disabled_when_zero():
    app = _app(stall_ms=0.0)

    async def run():
        app._arm_stall_watchdog()
        await asyncio.sleep(0.1)
        return app._eos_calls, app._stall_watchdog_task
    calls, task = asyncio.run(run())
    assert calls == 0 and task is None


def test_no_fire_if_state_left_listening():
    # If a barge-in / sleep moved state away before the timer fires, no EOS.
    app = _app(stall_ms=50, state=ConvState.IDLE)

    async def run():
        app._arm_stall_watchdog()
        app._state = ConvState.SPEAKING  # state moved on
        await asyncio.sleep(0.12)
        return app._eos_calls
    assert asyncio.run(run()) == 0


def test_cancel_stops_fire():
    app = _app(stall_ms=60)

    async def run():
        app._arm_stall_watchdog()
        await asyncio.sleep(0.02)
        app._cancel_stall_watchdog()
        await asyncio.sleep(0.1)
        return app._eos_calls
    assert asyncio.run(run()) == 0
