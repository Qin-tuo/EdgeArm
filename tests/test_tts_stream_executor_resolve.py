"""Regression tests for codex Week 3 BLOCKER 4.

`_get_tts_stream_executor()` previously resolved the backend name only
at first call. If the TTS service wasn't ready yet at first call (lazy
startup, early /v2v/stream warm-up), backend_name() returned "" and
the backend-specific `OVS_TTS_STREAM_MAX_WORKERS_{KOKORO,MATCHA,...}`
envs never activated — the executor stuck at the global default for
the lifetime of the process.

The fix: track whether the backend was resolved at executor-create
time, and re-resolve once when the TTS service first reports ready.
"""
from __future__ import annotations

import pytest

from server import main as appmod


@pytest.fixture(autouse=True)
def _reset_executor(monkeypatch):
    # Reset module-level executor + flag so each test starts clean.
    if appmod._tts_stream_executor is not None:
        try:
            appmod._tts_stream_executor.shutdown(wait=False, cancel_futures=False)
        except Exception:
            pass
    appmod._tts_stream_executor = None
    appmod._tts_stream_executor_resolved_backend = False
    yield
    if appmod._tts_stream_executor is not None:
        try:
            appmod._tts_stream_executor.shutdown(wait=False, cancel_futures=False)
        except Exception:
            pass
    appmod._tts_stream_executor = None
    appmod._tts_stream_executor_resolved_backend = False


def _patch_tts_service(monkeypatch, is_ready: bool, name: str):
    from server.core import tts_service as svc
    monkeypatch.setattr(svc, "is_ready", lambda: is_ready)
    monkeypatch.setattr(svc, "backend_name", lambda: name)


def test_executor_uses_global_default_when_backend_unknown(monkeypatch):
    """First call before TTS ready → uses global OVS_TTS_STREAM_MAX_WORKERS."""
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS", "3")
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS_KOKORO", "5")
    _patch_tts_service(monkeypatch, is_ready=False, name="")

    ex = appmod._get_tts_stream_executor()
    assert ex._max_workers == 3, (
        f"expected global default 3 when backend unknown, got {ex._max_workers}"
    )
    assert appmod._tts_stream_executor_resolved_backend is False


def test_executor_refreshes_when_backend_resolves(monkeypatch):
    """Second call after TTS becomes ready → refresh to backend-specific."""
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS", "3")
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS_KOKORO", "1")

    # Call 1: TTS not ready → executor caches default = 3
    _patch_tts_service(monkeypatch, is_ready=False, name="")
    ex1 = appmod._get_tts_stream_executor()
    assert ex1._max_workers == 3

    # TTS service becomes ready, advertises kokoro.
    _patch_tts_service(monkeypatch, is_ready=True, name="jetson.kokoro_trt.fp16")

    # Call 2: must refresh to kokoro-specific = 1
    ex2 = appmod._get_tts_stream_executor()
    assert ex2._max_workers == 1, (
        f"expected refresh to OVS_TTS_STREAM_MAX_WORKERS_KOKORO=1, "
        f"got {ex2._max_workers}"
    )
    assert appmod._tts_stream_executor_resolved_backend is True

    # Call 3: should NOT refresh again (one-shot).
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS_KOKORO", "7")
    ex3 = appmod._get_tts_stream_executor()
    assert ex3._max_workers == 1, "refresh must be one-shot, not on every call"


def test_executor_resolves_at_init_when_backend_ready(monkeypatch):
    """If TTS is ready at first call, no refresh needed."""
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS", "3")
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS_QWEN3", "2")
    _patch_tts_service(monkeypatch, is_ready=True, name="jetson.qwen3_trt.w8a16")

    ex = appmod._get_tts_stream_executor()
    assert ex._max_workers == 2
    assert appmod._tts_stream_executor_resolved_backend is True


def test_executor_falls_back_to_capability_when_no_env(monkeypatch):
    """Spec §5: without env vars, executor cap aligns with backend
    concurrency_capability so the WorkerIO semaphore and executor share
    the same ceiling source."""
    # Force the loaded profile to declare matcha_trt (cap default K=2).
    monkeypatch.delenv("OVS_TTS_STREAM_MAX_WORKERS", raising=False)
    monkeypatch.delenv("OVS_TTS_STREAM_MAX_WORKERS_MATCHA", raising=False)
    from server.core import profile_loader
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {
            "tts_backend": "jetson.matcha_trt",
            "asr_backend": "jetson.paraformer_trt",
        },
    )
    _patch_tts_service(monkeypatch, is_ready=True, name="jetson.matcha_trt.fp16")
    n, name, src = appmod._resolve_tts_stream_max_workers()
    assert n == 2
    assert src == "concurrency_capability"


def test_executor_env_clamped_to_capability(monkeypatch, caplog):
    """Spec §5: env-based override is clamped to the backend ceiling."""
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS_MATCHA", "16")
    from server.core import profile_loader
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {
            "tts_backend": "jetson.matcha_trt",
            "asr_backend": "jetson.paraformer_trt",
        },
    )
    _patch_tts_service(monkeypatch, is_ready=True, name="jetson.matcha_trt.fp16")
    n, _, _ = appmod._resolve_tts_stream_max_workers()
    assert n == 2, f"expected clamp to backend ceiling 2, got {n}"


def test_executor_no_change_when_global_equals_backend_specific(monkeypatch):
    """Refresh path picks correct value even when values match — no
    spurious replacement, but flag flips so subsequent calls skip the
    refresh check."""
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS", "2")
    monkeypatch.setenv("OVS_TTS_STREAM_MAX_WORKERS_MOSS", "2")

    _patch_tts_service(monkeypatch, is_ready=False, name="")
    ex1 = appmod._get_tts_stream_executor()
    assert ex1._max_workers == 2

    _patch_tts_service(monkeypatch, is_ready=True, name="jetson.moss_tts_nano")
    ex2 = appmod._get_tts_stream_executor()
    # Same instance (no refresh, values match)
    assert ex2 is ex1
    assert appmod._tts_stream_executor_resolved_backend is True
