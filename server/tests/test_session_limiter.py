"""Unit tests for session_limiter.resolve_limit() capability aggregation.

Spec docs/specs/concurrency-capability-framework.md §3 + §7. The limiter
ceiling now derives from ``min(asr.max_concurrent, tts.max_concurrent)``
(``None`` is treated as +inf). Profile/env overrides may only downgrade;
exceeding the ceiling is warn + silent clamp.
"""

from __future__ import annotations

import logging

import pytest

from server.core.session_limiter import resolve_limit


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("OVS_MAX_CONCURRENT_SESSIONS", raising=False)
    monkeypatch.delenv("OVS_TTS_WORKER_CONCURRENCY", raising=False)
    monkeypatch.delenv("OVS_TTS_STREAM_MAX_WORKERS", raising=False)


# ---------- Aggregation math ----------


def test_jetson_matcha_plus_paraformer_takes_min_finite():
    """matcha (tts max=2) + paraformer (asr max=None=+inf) -> min = 2."""
    profile = {
        "asr_backend": "jetson.paraformer_trt",
        "tts_backend": "jetson.matcha_trt",
    }
    assert resolve_limit(profile) == 2


def test_cpu_pair_yields_desktop_default():
    """CPU sherpa pair -> min(4, 4) = 4."""
    profile = {
        "asr_backend": "cpu.sherpa_asr",
        "tts_backend": "cpu.sherpa",
    }
    assert resolve_limit(profile) == 4


def test_rk_pair_yields_serial():
    """RK pair -> min(1, 1) = 1."""
    profile = {
        "asr_backend": "rk.asr",
        "tts_backend": "rk.tts",
    }
    assert resolve_limit(profile) == 1


# ---------- Profile override semantics ----------


def test_profile_downgrade_honored():
    """Setting profile.max_concurrent_sessions BELOW ceiling downgrades."""
    profile = {
        "asr_backend": "jetson.paraformer_trt",
        "tts_backend": "jetson.matcha_trt",
        "max_concurrent_sessions": 1,
    }
    assert resolve_limit(profile) == 1


def test_profile_upgrade_clamped_with_warning(caplog):
    """Trying to lift profile above the ceiling is clamped + warn."""
    profile = {
        "asr_backend": "rk.asr",
        "tts_backend": "rk.tts",
        "max_concurrent_sessions": 99,
    }
    with caplog.at_level(logging.WARNING, logger="server.core.session_limiter"):
        assert resolve_limit(profile) == 1
    assert any("exceeds backend ceiling" in r.message for r in caplog.records)


# ---------- Env override semantics ----------


def test_env_downgrade_honored(monkeypatch):
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "1")
    profile = {
        "asr_backend": "cpu.sherpa_asr",
        "tts_backend": "cpu.sherpa",
    }
    assert resolve_limit(profile) == 1


def test_env_upgrade_clamped_with_warning(monkeypatch, caplog):
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "16")
    profile = {
        "asr_backend": "jetson.paraformer_trt",
        "tts_backend": "jetson.matcha_trt",
    }
    with caplog.at_level(logging.WARNING, logger="server.core.session_limiter"):
        assert resolve_limit(profile) == 2
    assert any("OVS_MAX_CONCURRENT_SESSIONS" in r.message for r in caplog.records)


def test_env_bad_value_raises(monkeypatch):
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "0")
    with pytest.raises(ValueError):
        resolve_limit({})


def test_profile_bad_value_raises():
    with pytest.raises(ValueError):
        resolve_limit({"max_concurrent_sessions": -1})


# ---------- Fallback to target defaults when capability unresolvable ----------


def test_unknown_backend_falls_back_to_target_default():
    """Profile without resolvable asr/tts -> legacy _TARGET_DEFAULTS path."""
    profile = {"name": "desktop-mac"}
    # No asr_backend / tts_backend keys => ceiling unknown, target_default
    # for "desktop" is 4.
    assert resolve_limit(profile) == 4


def test_no_profile_uses_unknown_default():
    assert resolve_limit(None) == 1


# ---------- Env beats profile (downgrade path) ----------


def test_env_overrides_profile(monkeypatch):
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "1")
    profile = {
        "asr_backend": "cpu.sherpa_asr",
        "tts_backend": "cpu.sherpa",
        "max_concurrent_sessions": 3,
    }
    assert resolve_limit(profile) == 1
