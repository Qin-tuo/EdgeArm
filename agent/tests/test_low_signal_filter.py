"""Regression: low-signal ASR finals (single chars, interjections,
filler tokens) must NOT reach the LLM.

These are the canonical trigger for an in-context echo loop on a small
quantised model: a one-char "你" / "嗯" / "you" gives the LLM almost
zero intent signal, it falls back to a safe canned reply, that reply
enters history, and within 3-4 turns the model latches and emits the
same line for every subsequent user input regardless of meaning.
"""
from __future__ import annotations

import pytest

from ovs_agent.app_base import (
    BaseApp,
    _INTERJECTIONS,
    _strip_for_signal,
)
from ovs_agent.slv_client import ASRFinal
from ovs_agent.state import ConvState
from types import SimpleNamespace


class _FakeAudio:
    def __init__(self) -> None:
        self.is_playing = False
        self.armed = 0

    async def stop_playback(self) -> None:
        pass

    def arm_for_next_turn(self) -> None:
        self.armed += 1

    def set_output_sample_rate(self, sr: int) -> None:
        pass


class _FakeSLV:
    def __init__(self) -> None:
        self.reconnects = 0
        self._closed = False

    async def reconnect(self) -> None:
        self.reconnects += 1


def _make_app() -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.audio = _FakeAudio()
    app.slv = _FakeSLV()
    app.plugins = []
    app._first_tts_seen = False
    app._llm_turn_task = None
    app._state = ConvState.THINKING
    app._slv_reconnect_count = 0
    app._eos_sent_this_turn = True
    app._vad_state = "idle"
    app._vad_speech_ms = 0
    app._vad_silence_ms = 0
    app._vad_eos_sent = False
    app._client_vad = None
    app._asr_watchdog_task = None
    app._thinking_watchdog_task = None
    app.config = SimpleNamespace(pipeline_mode="always_on", sleep_timeout_s=30.0)
    app._sleep_task = None
    app._ptt_explicit_eos_pending = False
    app._mic_rms_broadcast_task = None
    app._stop_words_cache = None
    return app


def test_strip_for_signal_drops_punctuation():
    assert _strip_for_signal(" 嗯! ") == "嗯"
    assert _strip_for_signal("you?") == "you"
    assert _strip_for_signal("...") == ""
    assert _strip_for_signal("  ") == ""


def test_strip_for_signal_keeps_chinese_and_alphanumerics():
    assert _strip_for_signal("你好 123") == "你好123"


@pytest.mark.parametrize(
    "text",
    ["嗯", "啊", "哦", "you", "the", "i", "a", "ok", "okay", "yeah"],
)
def test_interjection_set_contains_common_voice_garbage(text):
    assert text in _INTERJECTIONS or len(text) <= 1


@pytest.mark.asyncio
async def test_single_char_asr_final_is_dropped():
    """A 1-char ASR final ('你') must NOT trigger an LLM turn."""
    app = _make_app()
    on_utterance_calls = []
    await app._dispatch_one(ASRFinal(text="你", duplicate_of_streamed=False))
    # State must reset to IDLE, no LLM turn task spawned.
    assert app._state == ConvState.IDLE
    assert app._llm_turn_task is None
    assert app.audio.armed >= 1


@pytest.mark.asyncio
async def test_interjection_asr_final_is_dropped():
    """A pure interjection ('嗯') must NOT trigger an LLM turn."""
    app = _make_app()
    await app._dispatch_one(ASRFinal(text="嗯", duplicate_of_streamed=False))
    assert app._state == ConvState.IDLE
    assert app._llm_turn_task is None


@pytest.mark.asyncio
async def test_interjection_with_punctuation_is_dropped():
    """'嗯!' / ' you. ' must also be treated as low-signal."""
    app = _make_app()
    await app._dispatch_one(ASRFinal(text=" you. ", duplicate_of_streamed=False))
    assert app._state == ConvState.IDLE
    assert app._llm_turn_task is None


@pytest.mark.asyncio
async def test_real_utterance_passes_through():
    """A legitimate user utterance must still spawn an LLM turn."""
    app = _make_app()

    async def _noop(text: str) -> None:
        return None

    app.on_user_utterance = _noop  # type: ignore[assignment]
    await app._dispatch_one(
        ASRFinal(text="给我讲个故事吧", duplicate_of_streamed=False)
    )
    assert app._llm_turn_task is not None
    # Settle the spawned task.
    import asyncio
    await asyncio.wait_for(app._llm_turn_task, timeout=1.0)
