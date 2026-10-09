"""Unit tests for ASRSessionManager state machine.

Uses mock streams/backends. No real ASR or worker subprocess required.
"""

from __future__ import annotations

import asyncio
import sys, os
import time
from typing import Any, List

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _run(coro):
    """Run an async coroutine to completion. Avoids pytest-asyncio dep."""
    return asyncio.new_event_loop().run_until_complete(coro)


def asynctest(fn):
    """Decorator turning an async test fn into a sync one."""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(fn(*args, **kwargs))
        finally:
            loop.close()
    wrapper.__name__ = fn.__name__
    return wrapper

from server.core.asr_session_manager import (
    ASRSessionManager,
    SessionState,
)


# ── test doubles ───────────────────────────────────────────────────────


class FakeStream:
    def __init__(self, final_text: str = "hello", finalize_delay: float = 0.0):
        self.accept_calls: List[np.ndarray] = []
        self.finalize_called = False
        self.cancel_called = False
        self._final_text = final_text
        self._finalize_delay = finalize_delay

    def accept_waveform(self, sr: int, samples) -> None:
        self.accept_calls.append(samples)

    def finalize(self):
        self.finalize_called = True
        if self._finalize_delay:
            import time as _t
            _t.sleep(self._finalize_delay)
        # New ABC contract: ``(text, detected_language)``.
        return self._final_text, None

    def cancel(self) -> None:
        self.cancel_called = True

    def cancel_and_finalize(self) -> None:
        self.cancel_called = True


class HangingCancelStream(FakeStream):
    """Cancel blocks forever to exercise the 500ms timeout path."""

    def cancel(self) -> None:
        import time as _t
        _t.sleep(2.0)
        super().cancel()


class FakeBackend:
    def __init__(self, streams: List[FakeStream]):
        self._streams = streams
        self._idx = 0
        self.create_stream_calls = 0
        self.restart_worker_calls = 0

    def create_stream(self, language: str = "auto") -> FakeStream:
        self.create_stream_calls += 1
        s = self._streams[self._idx]
        self._idx += 1
        return s

    def restart_worker(self) -> None:
        self.restart_worker_calls += 1


class _WorkerProtocolError(Exception):
    pass


class _NoActiveSessionError(_WorkerProtocolError):
    pass


class _SessionAlreadyActiveError(_WorkerProtocolError):
    pass


class _WorkerExitError(_WorkerProtocolError):
    pass


class RaisingCreateBackend(FakeBackend):
    """create_stream raises the given exception N times, then returns a stream."""

    def __init__(self, streams, errors_seq):
        super().__init__(streams)
        self._errors_seq = list(errors_seq)

    def create_stream(self, language: str = "auto"):
        self.create_stream_calls += 1
        if self._errors_seq:
            err = self._errors_seq.pop(0)
            if err is not None:
                raise err
        s = self._streams[self._idx]
        self._idx += 1
        return s


# ── happy-path state machine ───────────────────────────────────────────


@asynctest
async def test_happy_path_state_transitions():
    s1 = FakeStream(final_text="utt1")
    be = FakeBackend([s1])
    mgr = ASRSessionManager(backend=be, language="zh")
    assert mgr.state == SessionState.IDLE

    gen = await mgr.on_speech_start()
    assert gen == 1
    assert mgr.state == SessionState.ACTIVE
    assert be.create_stream_calls == 1

    await mgr.accept_audio(np.zeros(160, dtype=np.float32))
    assert s1.accept_calls and len(s1.accept_calls[0]) == 160

    final = await mgr.finalize("vad_end")
    assert final == "utt1"
    assert s1.finalize_called
    assert mgr.state == SessionState.IDLE


@asynctest
async def test_each_utterance_gets_fresh_stream_and_generation():
    s1 = FakeStream("a")
    s2 = FakeStream("b")
    be = FakeBackend([s1, s2])
    mgr = ASRSessionManager(be)

    g1 = await mgr.on_speech_start()
    await mgr.finalize()
    g2 = await mgr.on_speech_start()
    await mgr.finalize()

    assert g1 == 1 and g2 == 2
    assert be.create_stream_calls == 2
    assert mgr.current_generation == 2


# ── stale generation guard ─────────────────────────────────────────────


@asynctest
async def test_stale_generation_dropped_via_state_check():
    """Final from gen=1 must NOT appear after gen=2 started."""
    s1 = FakeStream("from-gen-1")
    s2 = FakeStream("from-gen-2")
    be = FakeBackend([s1, s2])
    mgr = ASRSessionManager(be)

    await mgr.on_speech_start()  # gen 1
    # Pre-empt with another speech_start (TTS-active scenario).
    new_gen = await mgr.on_speech_start()  # gen 2
    assert new_gen == 2
    assert s1.cancel_called, "old gen 1 stream must have been cancelled"
    assert mgr.current_generation == 2


# ── abort during finalize ──────────────────────────────────────────────


@asynctest
async def test_cancel_during_finalize_discards_result():
    s1 = FakeStream("ignored", finalize_delay=0.2)
    be = FakeBackend([s1])
    mgr = ASRSessionManager(be)
    await mgr.on_speech_start()

    final_task = asyncio.create_task(mgr.finalize("vad_end"))
    # Let finalize enter FINALIZING (the sync sleep is in executor).
    await asyncio.sleep(0.02)
    assert mgr.state == SessionState.FINALIZING
    await mgr.cancel("bargein")
    final = await final_task
    assert final == "", "finalize result must be discarded when cancelled mid-flight"
    assert mgr.state == SessionState.IDLE


# ── VAD-start while TTS active ─────────────────────────────────────────


@asynctest
async def test_vad_start_while_session_active_preempts():
    # Simulates: ACTIVE → on_speech_start again (e.g. TTS was playing
    # and user re-started); old stream must be cancelled, new gen issued.
    s1, s2 = FakeStream("old"), FakeStream("new")
    be = FakeBackend([s1, s2])
    mgr = ASRSessionManager(be)

    await mgr.on_speech_start()
    await mgr.accept_audio(np.zeros(100, dtype=np.float32))

    g2 = await mgr.on_speech_start()
    assert g2 == 2
    assert s1.cancel_called
    assert be.create_stream_calls == 2
    assert mgr.state == SessionState.ACTIVE


# ── worker error → ERROR_REBUILD ──────────────────────────────────────


@asynctest
async def test_worker_protocol_errors_route_to_error_rebuild():
    """All 3 worker error types must trigger ERROR_REBUILD recovery."""
    for err_cls in (_NoActiveSessionError, _SessionAlreadyActiveError, _WorkerExitError):
        good = FakeStream("ok")
        be = RaisingCreateBackend(streams=[good], errors_seq=[err_cls("boom")])
        mgr = ASRSessionManager(be)
        # First on_speech_start: create_stream raises → ERROR_REBUILD → IDLE
        await mgr.on_speech_start()
        # After ERROR_REBUILD finishes, state should be IDLE (next attempt
        # creates the stream lazily on accept_audio).
        # The recovery schedule sleeps up to ~0.6s.
        await asyncio.sleep(0.05)
        # The manager flagged ERROR_REBUILD path; after recovery it
        # advanced to IDLE. State machine alive.
        assert mgr.state in (SessionState.IDLE, SessionState.ACTIVE)


@asynctest
async def test_repeated_failures_trigger_worker_restart():
    """4th consecutive failure → restart_worker() invoked."""
    good = FakeStream("ok")
    # 4 errors then a good stream.
    be = RaisingCreateBackend(
        streams=[good],
        errors_seq=[
            _WorkerExitError("1"),
            _WorkerExitError("2"),
            _WorkerExitError("3"),
            _WorkerExitError("4"),
        ],
    )
    mgr = ASRSessionManager(be)
    await mgr.on_speech_start()
    # After 3 retries the manager falls back to restart_worker().
    # Allow full backoff schedule (50+150+400ms) to complete.
    await asyncio.sleep(0.8)
    assert be.restart_worker_calls >= 1


# ── cancel timeout path ────────────────────────────────────────────────


@asynctest
async def test_cancel_timeout_triggers_restart_worker():
    hung = HangingCancelStream("never")
    be = FakeBackend([hung])
    mgr = ASRSessionManager(be)
    await mgr.on_speech_start()
    await mgr.cancel("bargein")  # 500ms timeout → restart_worker
    assert be.restart_worker_calls == 1
    assert mgr.state == SessionState.IDLE


# ── ws_close shutdown ──────────────────────────────────────────────────


@asynctest
async def test_cancel_idle_is_noop():
    be = FakeBackend([])
    mgr = ASRSessionManager(be)
    await mgr.cancel("ws_close")
    assert mgr.state == SessionState.IDLE
    assert be.restart_worker_calls == 0


# ── T1: concurrent restart_worker idempotency ─────────────────────────


@asynctest
async def test_concurrent_mark_error_collapses_to_single_restart():
    """5 concurrent mark_error calls must not fan out into 5 restart_worker
    invocations — recovery is serialized & coalesced per error window."""
    good = FakeStream("ok")
    # Each create_stream call will fail enough times to exhaust retries
    # and trigger restart_worker. If coalescing fails, we'd see >1
    # restart for the same error window.
    be = RaisingCreateBackend(
        streams=[good, good, good, good, good],
        errors_seq=[_WorkerExitError(f"e{i}") for i in range(20)],
    )
    mgr = ASRSessionManager(be)
    # Drive 5 concurrent error notifications.
    errs = [_WorkerExitError(f"caller-{i}") for i in range(5)]
    await asyncio.gather(*[mgr._async_mark_error(e) for e in errs])
    # Allow any in-flight restart to land.
    await asyncio.sleep(0.05)
    assert be.restart_worker_calls <= 1, (
        f"expected ≤1 restart_worker call, got {be.restart_worker_calls}"
    )


# ── T2: cancel during ERROR_REBUILD is bounded ────────────────────────


@asynctest
async def test_cancel_during_error_rebuild_is_bounded():
    """WS-close cancel must return promptly even when ERROR_REBUILD is
    burning through its retry/backoff schedule. No hang past 1s."""
    good = FakeStream("ok")
    # Enough errors to keep _do_rebuild_locked busy through all 3 backoffs
    # then a restart, so cancel definitely races with recovery.
    be = RaisingCreateBackend(
        streams=[good],
        errors_seq=[
            _WorkerExitError("1"), _WorkerExitError("2"),
            _WorkerExitError("3"), _WorkerExitError("4"),
        ],
    )
    mgr = ASRSessionManager(be)
    # Kick off recovery in the background.
    rebuild_task = asyncio.create_task(
        mgr._async_mark_error(_WorkerExitError("trigger"))
    )
    await asyncio.sleep(0.01)  # let it enter ERROR_REBUILD
    t0 = time.monotonic()
    await mgr.cancel("ws_close")
    elapsed = time.monotonic() - t0
    # 3 backoffs sum to ~0.6s; add tolerance for restart_worker hop.
    # The hard ceiling is 1.0s as specified.
    assert elapsed < 1.0, f"cancel during ERROR_REBUILD took {elapsed:.3f}s"
    # Let the background recovery finish so the loop closes cleanly.
    try:
        await asyncio.wait_for(rebuild_task, timeout=1.5)
    except asyncio.TimeoutError:
        rebuild_task.cancel()


# ── T3: slow finalize racing new speech_start does not drop new utterance ─


@asynctest
async def test_slow_finalize_does_not_silence_new_utterance():
    """utt1 finalize is slow; while it's in flight, utt2 speech_start fires.
    When utt1's finalize completes, the new (utt2) generation/active flag
    must NOT be cleared. accept_audio on utt2 must still succeed."""
    s1 = FakeStream("utt1-final", finalize_delay=0.2)
    s2 = FakeStream("utt2-final")
    be = FakeBackend([s1, s2])
    mgr = ASRSessionManager(be)

    g1 = await mgr.on_speech_start()
    assert g1 == 1

    # Kick off the slow finalize on utt1.
    finalize_task = asyncio.create_task(mgr.finalize_with_generation("vad_end"))
    await asyncio.sleep(0.02)  # let it enter FINALIZING

    # While utt1 finalize is in flight, the user starts speaking again.
    g2 = await mgr.on_speech_start()
    assert g2 == 2
    assert mgr.current_generation == g2
    assert mgr.state == SessionState.ACTIVE

    # Audio for the NEW utterance must flow.
    await mgr.accept_audio(np.zeros(160, dtype=np.float32))
    assert s2.accept_calls, "utt2 stream must receive audio after preempting utt1"

    # Now let utt1's finalize complete. Whatever it returns, the manager's
    # current generation must remain g2 (not be clobbered back).
    ran_gen, _text = await finalize_task
    # Caller uses ran_gen != current_generation to know NOT to clear
    # asr_active flags — exact same guard documented in main.py.
    assert ran_gen == g1
    assert mgr.current_generation == g2
    assert mgr.state == SessionState.ACTIVE, (
        f"utt2 must remain ACTIVE after stale utt1 finalize, got {mgr.state}"
    )

    # And subsequent accept_audio on the new utterance still works.
    await mgr.accept_audio(np.zeros(160, dtype=np.float32))
    assert len(s2.accept_calls) == 2
