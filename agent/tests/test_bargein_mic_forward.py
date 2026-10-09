"""Regression guard for the barge-in mic-forwarding fix (2026-06-01).

Barge-in regressed because `mic_drop_while_speaking=true` dropped mic audio
while the agent was SPEAKING → SLV never received the user's interrupting
speech → never emitted ASRPartial → the barge-in trigger never fired. The
fix set it false (reSpeaker XVF3800 has hardware AEC). This locks the
mic-pump behaviour both ways so it can't silently regress.
"""
from __future__ import annotations

import pytest

from ovs_agent import Config, Session
from ovs_agent.app_base import BaseApp
from ovs_agent.state import ConvState


class _FakeAudio:
    chunk_ms = 100

    def __init__(self, chunks):
        self._chunks = chunks

    async def start_capture(self):
        for c in self._chunks:
            yield c


class _FakeSLV:
    def is_reconnecting(self) -> bool:
        return False


class _StatefulAudio:
    chunk_ms = 100

    def __init__(self, app: BaseApp, script):
        self._app = app
        self._script = script

    async def start_capture(self):
        for state, chunk in self._script:
            self._app._state = state
            yield chunk


def _mic_app(*, drop_while_speaking: bool, state: ConvState):
    app = BaseApp.__new__(BaseApp)
    app.config = Config(
        system_prompt="SYS",
        mic_drop_while_speaking=drop_while_speaking,
        energy_gate_enabled=False,   # keep the forward path simple
        mic_makeup_gain=1.0,
    )
    # 3 non-silent 100ms@16k chunks (3200 bytes each).
    app.audio = _FakeAudio([b"\x10\x00" * 1600 for _ in range(3)])
    app.slv = _FakeSLV()
    app._client_vad = None          # server-VAD mode (production server-loop)
    app._state = state
    app._advertise_ready = None     # skip the advertise barrier
    app._vad_state = "idle"
    app._last_mic_chunk_ts = 0.0
    app._local_output_mic_suppress_until = 0.0

    sent: list[bytes] = []

    async def _send(pcm):
        sent.append(pcm)

    app._send_audio_nonblocking = _send          # type: ignore[assignment]
    app._schedule_mic_rms_broadcast = lambda d: False  # type: ignore[assignment]
    return app, sent


@pytest.mark.asyncio
async def test_mic_forwards_during_speaking_when_drop_false():
    """barge-in enabler: with drop=false, mic audio is forwarded to SLV even
    while SPEAKING, so the server can detect an interrupt and emit ASRPartial."""
    app, sent = _mic_app(drop_while_speaking=False, state=ConvState.SPEAKING)
    await app._mic_pump()
    assert len(sent) == 3, "mic must keep forwarding during SPEAKING for barge-in"


@pytest.mark.asyncio
async def test_mic_dropped_during_speaking_when_drop_true():
    """The old (barge-in-breaking) behaviour: drop=true silences the mic while
    SPEAKING. Locked so the trade-off is explicit, not accidental."""
    app, sent = _mic_app(drop_while_speaking=True, state=ConvState.SPEAKING)
    await app._mic_pump()
    assert len(sent) == 0, "drop=true must drop mic during SPEAKING"


@pytest.mark.asyncio
async def test_mic_forwards_when_idle_regardless():
    """Sanity: in a listening/idle-ish state the mic forwards either way
    (the drop gate only applies to SPEAKING/THINKING)."""
    app, sent = _mic_app(drop_while_speaking=True, state=ConvState.LISTENING)
    await app._mic_pump()
    assert len(sent) == 3, "mic must forward when not SPEAKING/THINKING"


@pytest.mark.asyncio
async def test_mic_dropped_during_local_output_suppress_window(monkeypatch):
    """Wake feedback is local playback, not TTS, so ConvState can still be
    IDLE. The mic pump must nevertheless drop that short echo window instead
    of forwarding the beep/tail into ASR as a bogus utterance."""
    app, sent = _mic_app(drop_while_speaking=False, state=ConvState.IDLE)
    app._local_output_mic_suppress_until = 100.0
    monkeypatch.setattr("ovs_agent.app_base.time.monotonic", lambda: 1.0)

    await app._mic_pump()

    assert len(sent) == 0


@pytest.mark.asyncio
async def test_energy_gate_resets_across_sleep_boundary(monkeypatch):
    """A wake/sleep boundary must end the current command-capture turn.

    Without this, an already-open energy gate can survive while mic chunks are
    dropped in SLEEPING, then close on the first post-wake silence and send an
    EOS for stale audio from the previous turn.
    """
    app = BaseApp.__new__(BaseApp)
    app.config = Config(
        system_prompt="SYS",
        energy_gate_enabled=True,
        energy_gate_open_rms=0.010,
        energy_gate_close_rms=0.004,
        energy_gate_hangover_ms=100,
        mic_makeup_gain=1.0,
        gate_drive_eos=True,
        gate_eos_min_speech_ms=250,
        mic_drop_while_speaking=False,
    )
    loud = b"\xd0\x07" * 1600  # int16 2000, comfortably above gate_open.
    silence = b"\x00\x00" * 1600
    app.audio = _StatefulAudio(
        app,
        [
            (ConvState.IDLE, loud),
            (ConvState.IDLE, loud),
            (ConvState.SLEEPING, loud),
            (ConvState.IDLE, silence),
        ],
    )
    app.slv = _FakeSLV()
    app._client_vad = None
    app._state = ConvState.IDLE
    app._advertise_ready = None
    app._vad_state = "idle"
    app._last_mic_chunk_ts = 0.0
    app._wake_mic_skip_until = 0.0
    app._local_output_mic_suppress_until = 0.0
    app._schedule_mic_rms_broadcast = lambda d: False  # type: ignore[assignment]

    async def _send_audio(pcm):
        return None

    app._send_audio_nonblocking = _send_audio  # type: ignore[assignment]

    eos_count = 0

    async def _send_eos():
        nonlocal eos_count
        eos_count += 1
        return True

    app.send_asr_eos_once = _send_eos  # type: ignore[assignment]

    now = 0.0

    def _mono():
        nonlocal now
        now += 0.2
        return now

    monkeypatch.setattr("ovs_agent.app_base.time.monotonic", _mono)

    await app._mic_pump()

    assert eos_count == 0


@pytest.mark.asyncio
async def test_gate_eos_enters_thinking_and_cancels_wake_timeout(monkeypatch):
    """In wake-command mode, the gate close edge is the end of the command.

    The app must stop the wake timeout once it has sent ASR EOS. Otherwise the
    fixed wake timer can fire while SLV is still producing the final, turning a
    heard utterance into a bogus no-final / "try again" interaction.
    """
    app = BaseApp.__new__(BaseApp)
    app.config = Config(
        system_prompt="SYS",
        pipeline_mode="wake_word",
        wake_command_single_turn=True,
        energy_gate_enabled=True,
        energy_gate_open_rms=0.010,
        energy_gate_close_rms=0.004,
        energy_gate_hangover_ms=100,
        mic_makeup_gain=1.0,
        gate_drive_eos=True,
        gate_eos_min_speech_ms=250,
        mic_drop_while_speaking=False,
    )
    loud = b"\xd0\x07" * 1600
    silence = b"\x00\x00" * 1600
    app.audio = _FakeAudio([loud, loud, loud, silence, silence])
    app.slv = _FakeSLV()
    app._client_vad = None
    app._state = ConvState.IDLE
    app._advertise_ready = None
    app._vad_state = "idle"
    app._last_mic_chunk_ts = 0.0
    app._wake_mic_skip_until = 0.0
    app._local_output_mic_suppress_until = 0.0
    app._schedule_mic_rms_broadcast = lambda d: False  # type: ignore[assignment]

    calls: list[str] = []

    async def _send_audio(pcm):
        calls.append("audio:zero" if set(pcm) == {0} else "audio:real")
        return None

    app._send_audio_nonblocking = _send_audio  # type: ignore[assignment]

    eos_count = 0

    async def _send_eos():
        nonlocal eos_count
        eos_count += 1
        calls.append("eos")
        return True

    app.send_asr_eos_once = _send_eos  # type: ignore[assignment]

    cancel_count = 0

    def _cancel_timeout():
        nonlocal cancel_count
        cancel_count += 1

    app._cancel_wake_command_timeout = _cancel_timeout  # type: ignore[assignment]
    async def _broadcast(*a, **k):
        return None

    app._broadcast = _broadcast  # type: ignore[assignment]

    now = 0.0

    def _mono():
        nonlocal now
        now += 0.2
        return now

    monkeypatch.setattr("ovs_agent.app_base.time.monotonic", _mono)

    await app._mic_pump()

    assert eos_count == 1
    eos_idx = calls.index("eos")
    assert calls[eos_idx - 1] == "audio:zero"
    assert cancel_count == 1
    assert app._state == ConvState.THINKING
