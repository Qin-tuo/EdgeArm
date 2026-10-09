"""Robot-agnostic, reusable client-VAD utterance segmenter.

Extracts the pre-roll ring + speech-onset drain + trailing-silence EOS
logic that historically lived inline in ``app_base._mic_pump`` and was
re-implemented a second time in reachy's ``conversation_plugin``. Both
now share this one pure helper.

The contract is intentionally tiny so it composes with any chunk source
and any VAD backend (see :func:`ovs_agent.vad.create_vad`):

* takes an ``AsyncIterator[bytes]`` of fixed-size PCM16 chunks,
* takes a ``vad`` exposing ``is_speech(pcm: bytes) -> bool`` and ``reset()``,
* yields ``(event, payload)`` tuples:

    - ``("speech_start", None)`` — emitted once, the moment a confirmed
      speech onset is detected (after ``min_speech_ms`` of speech).
    - ``("audio", pcm)`` — first the buffered pre-roll ring (the
      ``preroll_ms`` of audio captured *before* onset, so the engine
      sees the leading edge of the first word), then every live chunk
      until the utterance ends.
    - ``("speech_end", None)`` — emitted once, after ``silence_ms`` of
      trailing silence following a confirmed utterance. The segmenter
      then returns to idle and the cycle can repeat.

No app / SLV / pipeline-state dependencies — callers layer their own
echo gate, reconnect handling, RMS broadcast, state machine, etc. on top
by reacting to the yielded events.

Why pre-roll matters: a VAD needs a few chunks of audio to *confirm*
speech-start. Without a ring buffer, those leading chunks (the start of
the first word) are gone by the time onset fires, so the ASR hears a
clipped utterance and "swallows the first word". The ring replays them.
"""
from __future__ import annotations

from collections import deque
from typing import AsyncIterator, Callable, List, Optional, Tuple

__all__ = ["vad_gated", "PrerollRing"]


class PrerollRing:
    """The pre-roll ring + speech-onset drain, with no VAD ownership.

    This is the *minimal core* shared by callers that already drive their
    own VAD / EOS / state machine (e.g. ``app_base._mic_pump`` via
    ``_update_vad``). It owns only the "buffer idle chunks, replay them at
    onset, then pass-through while speaking" mechanic — exactly the
    behaviour that previously lived inline.

    Usage::

        ring = PrerollRing(preroll_max=4)
        # per chunk, after your VAD decides the current speech state:
        if vad_state == "speech":
            for buffered in ring.drain():   # replays pre-roll once at onset
                await send(buffered)
            await send(chunk)
        else:
            ring.append(chunk)              # idle: buffer, do not send
        # on reconnect / sleep / etc:
        ring.clear()

    ``drain()`` returns the buffered chunks in order and empties the ring,
    so it is a no-op (empty list) on every chunk after onset until the
    ring is re-filled during the next idle period.
    """

    __slots__ = ("_buf",)

    def __init__(self, preroll_max: int) -> None:
        self._buf: deque[bytes] = deque(maxlen=max(1, int(preroll_max)))

    @property
    def maxlen(self) -> int:
        return self._buf.maxlen or 1

    def append(self, chunk: bytes) -> None:
        self._buf.append(chunk)

    def drain(self) -> List[bytes]:
        """Return + clear all buffered chunks (the speech-onset replay)."""
        if not self._buf:
            return []
        out = list(self._buf)
        self._buf.clear()
        return out

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buf)


async def vad_gated(
    chunks: AsyncIterator[bytes],
    vad,
    *,
    chunk_ms: int,
    preroll_ms: int = 400,
    silence_ms: int = 600,
    min_speech_ms: int = 250,
) -> AsyncIterator[Tuple[str, Optional[bytes]]]:
    """Segment a PCM16 chunk stream into utterances using a client VAD.

    See the module docstring for the emitted ``(event, payload)`` grammar.

    Args:
        chunks: async iterator of fixed-size PCM16 little-endian chunks.
        vad: object with ``is_speech(pcm: bytes) -> bool`` and ``reset()``
            (e.g. from :func:`ovs_agent.vad.create_vad`).
        chunk_ms: nominal duration of each chunk in milliseconds. Used to
            size the pre-roll ring and accumulate speech/silence timers.
        preroll_ms: how much audio (ms) to retain ahead of speech onset.
        silence_ms: trailing silence (ms) that ends an utterance.
        min_speech_ms: minimum confirmed speech (ms) before onset fires —
            debounces transient noise into a false ``speech_start``.
    """
    chunk_ms = max(1, int(chunk_ms))
    preroll_max = max(1, int(round(preroll_ms / chunk_ms)))
    preroll: deque[bytes] = deque(maxlen=preroll_max)

    state = "idle"  # "idle" | "speech"
    speech_acc_ms = 0.0
    silence_acc_ms = 0.0

    async for chunk in chunks:
        try:
            is_speech = vad.is_speech(chunk)
        except Exception:  # pragma: no cover - defensive; treat as silence
            is_speech = False

        if state == "idle":
            if is_speech:
                speech_acc_ms += chunk_ms
                # Keep buffering this chunk too: if onset confirms below,
                # the ring (including this chunk) is drained in order.
                preroll.append(chunk)
                if speech_acc_ms >= min_speech_ms:
                    # Confirmed onset. Announce, then replay the pre-roll
                    # ring (the leading edge of the utterance) in order.
                    yield ("speech_start", None)
                    while preroll:
                        yield ("audio", preroll.popleft())
                    state = "speech"
                    silence_acc_ms = 0.0
            else:
                # Idle silence: keep a short rolling buffer, never emit.
                speech_acc_ms = 0.0
                preroll.append(chunk)
        else:  # state == "speech"
            # Stream every chunk (incl. trailing silence) so the tail of
            # the utterance reaches the engine before EOS.
            yield ("audio", chunk)
            if is_speech:
                silence_acc_ms = 0.0
            else:
                silence_acc_ms += chunk_ms
                if silence_acc_ms >= silence_ms:
                    yield ("speech_end", None)
                    # Return to idle for the next utterance.
                    state = "idle"
                    speech_acc_ms = 0.0
                    silence_acc_ms = 0.0
                    preroll.clear()
                    try:
                        vad.reset()
                    except Exception:  # pragma: no cover - defensive
                        pass
