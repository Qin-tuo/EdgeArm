"""Tests for client-side VAD + BaseApp asr_eos integration."""
from __future__ import annotations

import asyncio
import importlib.util
import os
import subprocess
import sys
import wave
from typing import Any

import numpy as np
import pytest

from ovs_agent.vad import EnergyVAD, SileroVAD, create_vad

_HERE = os.path.dirname(__file__)
_SPEECH_WAV = os.path.join(_HERE, "fixtures", "user_input.wav")


def _silero_available() -> bool:
    """True only if BOTH onnxruntime and the bundled silero onnx exist —
    i.e. the torch-free path can actually run. The dev test env usually has
    neither, so these tests skip there; the decisive proof is the on-device
    boot log (client VAD=silero)."""
    if importlib.util.find_spec("onnxruntime") is None:
        return False
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or spec.origin is None:
        return False
    onnx = os.path.join(os.path.dirname(spec.origin), "data", "silero_vad.onnx")
    return os.path.exists(onnx)


_silero_skip = pytest.mark.skipif(
    not _silero_available(),
    reason="onnxruntime + silero_vad.onnx not available in this env",
)


def _read_wav_bytes(path: str) -> bytes:
    with wave.open(path) as w:
        return w.readframes(w.getnframes())


# ── EnergyVAD unit tests ────────────────────────────────────────────────


def test_energy_vad_silence() -> None:
    vad = EnergyVAD(threshold=0.01)
    silence = np.zeros(1600, dtype=np.int16).tobytes()
    assert vad.is_speech(silence) is False


def test_energy_vad_loud() -> None:
    vad = EnergyVAD(threshold=0.01)
    loud = np.random.randint(
        -10000, 10000, size=1600, dtype=np.int16
    ).tobytes()
    assert vad.is_speech(loud) is True


def test_energy_vad_empty() -> None:
    vad = EnergyVAD(threshold=0.01)
    assert vad.is_speech(b"") is False


def test_create_vad_energy_explicit() -> None:
    v = create_vad("energy")
    assert v.name == "energy"


def test_create_vad_auto_falls_back() -> None:
    # On a clean test env silero-vad likely isn't installed; auto must
    # not raise and must yield *some* VAD.
    v = create_vad("auto")
    assert v.name in ("silero", "energy")


# ── SileroVAD (torch-free onnxruntime path) ─────────────────────────────


@_silero_skip
def test_silero_vad_speech_vs_silence() -> None:
    """Real speech buffer → detected; silence → not detected."""
    vad = SileroVAD()
    speech = _read_wav_bytes(_SPEECH_WAV)
    detected = any(
        vad.is_speech(speech[i : i + 3200]) for i in range(0, len(speech), 3200)
    )
    assert detected is True

    vad.reset()
    silence = np.zeros(16000, dtype=np.int16).tobytes()
    sil_detected = any(
        vad.is_speech(silence[i : i + 3200]) for i in range(0, len(silence), 3200)
    )
    assert sil_detected is False


@_silero_skip
def test_silero_vad_reset_restores_state() -> None:
    """reset() clears recurrent state/context/buffer so detection repeats."""
    vad = SileroVAD()
    speech = _read_wav_bytes(_SPEECH_WAV)

    def run() -> bool:
        vad.reset()
        return any(
            vad.is_speech(speech[i : i + 3200])
            for i in range(0, len(speech), 3200)
        )

    assert run() is True
    assert run() is True  # reset() must let it detect again


@_silero_skip
def test_create_vad_silero_explicit() -> None:
    v = create_vad("silero")
    assert v.name == "silero"


def test_silero_vad_is_torch_free() -> None:
    """The whole SileroVAD path (init + is_speech + reset) must NEVER import
    torch. We prove this in a fresh subprocess that fakes-out torch: any
    `import torch` raises, so if the path touched torch the run would crash.
    The subprocess loads vad.py by file path (bypassing the package __init__,
    which needs pyyaml) and uses a synthetic float buffer so the test runs
    even without the silero onnx model present (it just exercises the import
    boundary up to the onnxruntime InferenceSession)."""
    code = r"""
import sys, importlib.util, importlib.machinery
# Poison torch: any attempt to import it must blow up loudly.
class _Boom(importlib.machinery.PathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'torch' or name.startswith('torch.'):
            raise AssertionError('TORCH WAS IMPORTED')
        return None
sys.meta_path.insert(0, _Boom())
assert 'torch' not in sys.modules
spec = importlib.util.spec_from_file_location('vadmod', %r)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# Importing the module must not pull torch.
assert 'torch' not in sys.modules
# Constructing + driving SileroVAD must not pull torch either. If the
# onnx model / onnxruntime isn't installed this raises a non-torch error,
# which still proves the torch-free boundary up to that point.
try:
    v = m.SileroVAD()
    import numpy as np
    pcm = (np.zeros(3200, dtype='int16')).tobytes()
    v.is_speech(pcm); v.reset()
except AssertionError:
    raise  # torch import -> real failure
except Exception:
    pass  # missing onnxruntime/model is fine; not a torch import
assert 'torch' not in sys.modules
print('TORCH_FREE_OK')
""" % (os.path.join(os.path.dirname(_HERE), "ovs_agent", "vad.py"),)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert "TORCH_FREE_OK" in proc.stdout, (
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert proc.returncode == 0, proc.stderr


# ── BaseApp asr_eos integration ─────────────────────────────────────────


class _FakeSLV:
    """Minimal SLV stand-in capturing send_audio + asr_eos calls."""

    def __init__(self) -> None:
        self.audio_chunks: list[bytes] = []
        self.asr_eos_count = 0

    async def send_audio(self, pcm: bytes) -> None:
        self.audio_chunks.append(pcm)

    async def asr_eos(self) -> None:
        self.asr_eos_count += 1


class _FakeAudio:
    """Yields pre-canned mic chunks then halts."""

    is_playing = False

    def __init__(self, chunks: list[bytes], chunk_ms: int = 100) -> None:
        self._chunks = chunks
        self.chunk_ms = chunk_ms

    async def start_capture(self):
        for c in self._chunks:
            yield c
            await asyncio.sleep(0)


class _AlwaysSpeechThenSilenceVAD:
    """Deterministic VAD that reports N speech chunks then silence."""

    name = "test"
    threshold = 0.5

    def __init__(self, speech_n: int) -> None:
        self._left = speech_n

    def is_speech(self, pcm: bytes) -> bool:
        if self._left > 0:
            self._left -= 1
            return True
        return False

    def reset(self) -> None:
        pass


@pytest.mark.asyncio
async def test_mic_pump_fires_asr_eos_on_speech_end() -> None:
    """speech (>=200ms) followed by 600ms silence → exactly one asr_eos."""
    # Construct app without invoking __init__ (avoids LLM/audio setup).
    from ovs_agent.app_base import BaseApp

    app = BaseApp.__new__(BaseApp)

    class _Cfg:
        client_vad_backend = "off"  # we'll inject our own
        client_vad_threshold = None
        client_vad_speech_min_ms = 200
        client_vad_silence_ms = 600
        client_vad_drive_eos = True  # explicit for this test
        audio_input_sample_rate = 16000

    app.config = _Cfg()
    app.slv = _FakeSLV()
    # 100ms chunks: 3 speech (=300ms speech, crosses 200ms min), then 7
    # silence (=700ms, crosses 600ms silence threshold).
    chunk = b"\x00\x00" * 1600  # bytes content irrelevant to fake VAD
    chunks = [chunk] * 10
    app.audio = _FakeAudio(chunks, chunk_ms=100)
    app._client_vad = _AlwaysSpeechThenSilenceVAD(speech_n=3)
    app._vad_state = "idle"
    app._vad_speech_ms = 0
    app._vad_silence_ms = 0
    app._vad_eos_sent = False
    app._eos_sent_this_turn = False
    app._asr_watchdog_task = None
    app._thinking_watchdog_task = None
    app._mic_rms_broadcast_task = None
    app._state = __import__("ovs_agent.state", fromlist=["ConvState"]).ConvState.IDLE
    app.events = __import__("ovs_agent.event_bus", fromlist=["EventBus"]).EventBus()
    app.plugins = []
    app._last_mic_chunk_ts = 0.0

    await app._mic_pump()

    assert app.slv.asr_eos_count == 1, (
        f"expected exactly 1 asr_eos, got {app.slv.asr_eos_count}"
    )
    # mic_pump only sends audio when VAD state is "speech" (with the
    # idle pre-roll flushed at speech-start). It also stops sending the
    # moment VAD transitions back to idle on speech-end. With 3 speech
    # chunks + 7 silence chunks and speech_min=200ms, silence=600ms:
    #   chunk 1: idle, buffered
    #   chunk 2: state→speech, flush preroll(1) + send(1) = 2 sends
    #   chunks 3..8: state=speech, send each = 6 sends
    #   chunk 9: silence_ms hits 600 → asr_eos, state→idle, NOT sent
    #   chunk 10: idle, buffered
    # → 8 sends total
    assert len(app.slv.audio_chunks) == 8
