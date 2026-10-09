"""Unit tests for the reusable client-VAD segmenter (ovs_agent.audio.vad_gate).

Covers:
- PrerollRing: idle buffering, single onset-drain, clear.
- vad_gated: pre-roll replayed BEFORE the first live chunk at onset
  (the "first word not swallowed" guarantee), speech_end after trailing
  silence, return-to-idle for a second utterance.
"""
from __future__ import annotations

import pytest

from ovs_agent.audio.vad_gate import PrerollRing, vad_gated


def test_preroll_ring_buffers_and_drains_once():
    ring = PrerollRing(preroll_max=3)
    assert ring.drain() == []  # empty → no-op
    ring.append(b"a")
    ring.append(b"b")
    assert ring.drain() == [b"a", b"b"]
    assert ring.drain() == []  # drained → empty until refilled


def test_preroll_ring_respects_maxlen():
    ring = PrerollRing(preroll_max=2)
    for c in (b"a", b"b", b"c", b"d"):
        ring.append(c)
    # Only the most recent 2 survive.
    assert ring.drain() == [b"c", b"d"]


def test_preroll_ring_clear():
    ring = PrerollRing(preroll_max=3)
    ring.append(b"x")
    ring.clear()
    assert ring.drain() == []


class _ScriptedVAD:
    """is_speech() returns the next value from a scripted list (keyed by the
    chunk payload so the test reads naturally)."""

    def __init__(self, speech_chunks):
        self._speech = set(speech_chunks)
        self.resets = 0

    def is_speech(self, pcm: bytes) -> bool:
        return pcm in self._speech

    def reset(self) -> None:
        self.resets += 1


async def _drain(agen):
    return [x async for x in agen]


async def _from_list(items):
    for it in items:
        yield it


@pytest.mark.asyncio
async def test_vad_gated_preroll_before_first_live_chunk():
    """The core guarantee: the buffered pre-roll (audio captured BEFORE
    onset confirmed) is emitted as ("audio", ...) ahead of the live chunk,
    so the engine sees the leading edge of the first word."""
    # 2 silent chunks (pre-roll), then 2 speech chunks. chunk_ms=100,
    # min_speech_ms=100 so onset confirms on the first speech chunk.
    sil1, sil2 = b"s1", b"s2"
    sp1, sp2 = b"p1", b"p2"
    vad = _ScriptedVAD([sp1, sp2])
    events = await _drain(
        vad_gated(
            _from_list([sil1, sil2, sp1, sp2]),
            vad,
            chunk_ms=100,
            preroll_ms=400,   # ring holds up to 4 chunks
            silence_ms=600,
            min_speech_ms=100,
        )
    )
    # speech_start fires, THEN the pre-roll (sil1, sil2) is replayed,
    # THEN sp1 was the onset chunk (also buffered into the ring → replayed),
    # then sp2 streams live.
    kinds = [e for e, _ in events]
    assert kinds[0] == "speech_start"
    audio = [p for k, p in events if k == "audio"]
    # Pre-roll chunks (sil1, sil2) precede the speech chunks in the stream.
    assert audio[:2] == [sil1, sil2], audio
    assert sp1 in audio and sp2 in audio
    # And sp1 (first speech) comes out before sp2.
    assert audio.index(sp1) < audio.index(sp2)


@pytest.mark.asyncio
async def test_vad_gated_speech_end_after_silence():
    sp1, sp2 = b"p1", b"p2"
    s = [b"q%d" % i for i in range(8)]  # trailing silence chunks
    vad = _ScriptedVAD([sp1, sp2])
    events = await _drain(
        vad_gated(
            _from_list([sp1, sp2, *s]),
            vad,
            chunk_ms=100,
            preroll_ms=200,
            silence_ms=600,   # 6 silent chunks @100ms
            min_speech_ms=100,
        )
    )
    kinds = [e for e, _ in events]
    assert kinds.count("speech_start") == 1
    assert kinds.count("speech_end") == 1
    # speech_end fires only after >= silence_ms of trailing silence.
    assert vad.resets == 1  # reset on utterance end


@pytest.mark.asyncio
async def test_vad_gated_two_utterances():
    sp = b"p"
    sil = b"s"
    # speech, 6 silence (end), speech, 6 silence (end)
    stream = [sp] + [sil] * 6 + [sp] + [sil] * 6
    vad = _ScriptedVAD([sp])
    events = await _drain(
        vad_gated(
            _from_list(stream), vad,
            chunk_ms=100, preroll_ms=200, silence_ms=600, min_speech_ms=100,
        )
    )
    kinds = [e for e, _ in events]
    assert kinds.count("speech_start") == 2
    assert kinds.count("speech_end") == 2
