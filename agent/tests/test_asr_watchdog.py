"""Tests for the asr_final watchdog.

SLV's always_on pipeline filters empty-text finals server-side, so a
mic-noise-triggered EOS produces no client-visible ASRFinal. Without
the watchdog the FSM would stay in THINKING forever.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ovs_agent.app_base import BaseApp
from ovs_agent.slv_client import ASRFinal, SLVError
from ovs_agent.state import ConvState
from ovs_agent.event_bus import EventBus


def _fresh_app(timeout: float = 0.1) -> BaseApp:
    """BaseApp built bypass __init__, with a mock SLV + short watchdog."""
    app = BaseApp.__new__(BaseApp)
    app.events = EventBus()
    app.plugins = []
    app.config = SimpleNamespace(asr_final_timeout_s=timeout)
    app._state = ConvState.IDLE
    app._slv_reconnect_count = 0
    app._eos_sent_this_turn = False
    app._asr_watchdog_task = None
    app._llm_turn_task = None
    app._first_tts_seen = False
    app._ptt_explicit_eos_pending = False
    app.slv = MagicMock()
    app.slv.asr_eos = AsyncMock()
    app.slv.reconnect = AsyncMock()
    return app


@pytest.mark.asyncio
async def test_watchdog_resets_state_when_no_final_arrives():
    app = _fresh_app(timeout=0.05)
    app._set_state(ConvState.LISTENING)
    app._set_state(ConvState.THINKING)
    sent = await app.send_asr_eos_once()
    assert sent is True
    assert app._state == ConvState.THINKING
    # No ASRFinal — wait for watchdog to fire.
    await asyncio.sleep(0.15)
    assert app._state == ConvState.IDLE, (
        "watchdog should have reset state to IDLE after asr_final_timeout_s"
    )
    assert app._eos_sent_this_turn is False


@pytest.mark.asyncio
async def test_watchdog_cancelled_when_real_final_arrives():
    app = _fresh_app(timeout=0.05)
    app._set_state(ConvState.LISTENING)
    app._set_state(ConvState.THINKING)
    await app.send_asr_eos_once()
    # Real final arrives before the watchdog would fire. We only need
    # to verify the watchdog gets cancelled — the dispatch path below
    # `_cancel_asr_watchdog()` does its own work; here we just call the
    # cancel directly the way the ASRFinal branch does.
    app._cancel_asr_watchdog()
    # Wait past the watchdog timeout. The state must NOT have been
    # touched by the (cancelled) watchdog.
    app._set_state(ConvState.SPEAKING)
    await asyncio.sleep(0.15)
    assert app._state == ConvState.SPEAKING, (
        "cancelled watchdog must not have reset state out from under us"
    )


@pytest.mark.asyncio
async def test_duplicate_final_while_thinking_resets_to_idle():
    """A duplicate-only ASR final must not strand the FSM in THINKING.

    This happens when client VAD sends EOS for a noise blip and SLV reports
    the final as a duplicate of an already streamed/empty result. The ASRFinal
    path cancels the watchdog, so the duplicate branch itself must recover.
    """
    app = _fresh_app(timeout=5.0)
    app._set_state(ConvState.THINKING)
    await app.send_asr_eos_once()

    await app._dispatch_one(
        ASRFinal(text="", duplicate_of_streamed=True, session_complete=False)
    )

    assert app._state == ConvState.IDLE
    assert app._eos_sent_this_turn is False
    assert app._asr_watchdog_task is None


@pytest.mark.asyncio
async def test_watchdog_no_op_if_state_already_moved():
    """If the FSM left THINKING (e.g. via SLVError → IDLE) before the
    watchdog fires, the watchdog must not clobber the new state."""
    app = _fresh_app(timeout=0.05)
    app._set_state(ConvState.LISTENING)
    app._set_state(ConvState.THINKING)
    await app.send_asr_eos_once()
    # Simulate something else (e.g. a barge-in) moving the state.
    app._set_state(ConvState.SPEAKING)
    await asyncio.sleep(0.15)
    # Watchdog fired but should have seen state != THINKING and bailed.
    assert app._state == ConvState.SPEAKING
    assert app._eos_sent_this_turn is False


@pytest.mark.asyncio
async def test_second_eos_call_cancels_prior_watchdog():
    """A second send_asr_eos_once on a fresh turn must not leak the
    previous watchdog task."""
    app = _fresh_app(timeout=0.05)
    app._set_state(ConvState.LISTENING)
    app._set_state(ConvState.THINKING)
    await app.send_asr_eos_once()
    first_task = app._asr_watchdog_task
    # Simulate turn ended (real final arrived).
    app._eos_sent_this_turn = False
    app._cancel_asr_watchdog()
    # New turn.
    app._set_state(ConvState.IDLE)
    app._set_state(ConvState.LISTENING)
    app._set_state(ConvState.THINKING)
    await app.send_asr_eos_once()
    # Give the cancelled task a tick to finalise its CancelledError.
    await asyncio.sleep(0)
    assert first_task.cancelled() or first_task.done()
    assert app._asr_watchdog_task is not first_task


@pytest.mark.asyncio
async def test_new_speech_while_waiting_for_final_releases_eos_latch():
    """If another speech segment starts before the prior empty/dropped
    ASR final arrives, the next speech end must be allowed to send EOS.
    """

    class _Vad:
        def is_speech(self, chunk: bytes) -> bool:
            return True

        def reset(self) -> None:
            pass

    app = _fresh_app(timeout=5.0)
    app.config.client_vad_speech_min_ms = 100
    app.config.pipeline_mode = "always_on"
    app.config.push_to_talk_no_vad_silence = True
    app._client_vad = _Vad()
    app._vad_state = "idle"
    app._vad_speech_ms = 0
    app._vad_silence_ms = 0
    app._vad_eos_sent = False
    app.audio = SimpleNamespace(is_playing=False)
    app._set_state(ConvState.THINKING)
    await app.send_asr_eos_once()
    stale_task = app._asr_watchdog_task

    await app._update_vad(b"\x01\x00" * 1600, 100)
    await asyncio.sleep(0)

    assert app._eos_sent_this_turn is False
    assert stale_task is not None
    assert stale_task.cancelled() or stale_task.done()
    assert app._state == ConvState.LISTENING
