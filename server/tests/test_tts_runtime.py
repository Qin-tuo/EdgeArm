"""Tests for server.core.tts_runtime override store and merge logic."""
from __future__ import annotations

import pytest

from server.core import tts_runtime


@pytest.fixture(autouse=True)
def _reset():
    tts_runtime.reset_overrides()
    yield
    tts_runtime.reset_overrides()


def test_default_overrides_all_none():
    snap = tts_runtime.get_overrides()
    assert snap.default_speaker_id is None
    assert snap.default_speed is None
    assert snap.default_pitch_shift is None


def test_update_partial_keeps_others():
    tts_runtime.update_overrides(speed=1.5)
    tts_runtime.update_overrides(speaker_id=0, model_id="qwen3-tts")
    snap = tts_runtime.get_overrides()
    assert snap.default_speaker_id == 0
    assert snap.default_speed == 1.5
    assert snap.default_pitch_shift is None


def test_update_with_explicit_none_clears():
    tts_runtime.update_overrides(speed=1.5, pitch_shift=2.0)
    assert tts_runtime.get_overrides().default_speed == 1.5
    tts_runtime.update_overrides(speed=None)
    snap = tts_runtime.get_overrides()
    assert snap.default_speed is None
    # untouched
    assert snap.default_pitch_shift == 2.0


def test_merge_request_overrides_runtime():
    tts_runtime.update_overrides(speaker_id=0, speed=1.5, model_id="qwen3-tts")
    merged = tts_runtime.merge_tts_request_kwargs(
        request_speaker_id=1,
        request_speed=2.0,
        request_pitch_shift=3.0,
        model_id="qwen3-tts",
    )
    assert merged["speaker_id"] == 1
    assert merged["speed"] == 2.0
    assert merged["pitch_shift"] == 3.0


def test_merge_runtime_overrides_default():
    tts_runtime.update_overrides(
        speaker_id=2301, speed=1.25, pitch_shift=1.0, model_id="qwen3-tts"
    )
    merged = tts_runtime.merge_tts_request_kwargs(
        request_speaker_id=None,
        request_speed=None,
        request_pitch_shift=None,
        model_id="qwen3-tts",
    )
    assert merged["speaker_id"] == 2301
    assert merged["speed"] == 1.25
    assert merged["pitch_shift"] == 1.0


def test_merge_falls_back_to_backend_default():
    from server.core.tts_speakers import default_speaker_id

    expected_default = default_speaker_id("qwen3-tts")
    merged = tts_runtime.merge_tts_request_kwargs(
        request_speaker_id=None,
        request_speed=None,
        request_pitch_shift=None,
        model_id="qwen3-tts",
    )
    assert merged["speaker_id"] == expected_default
    assert merged["speed"] is None
    assert merged["pitch_shift"] is None


def test_validate_unknown_speaker_id_raises(monkeypatch):
    # disable permissive fallback to be explicit
    monkeypatch.setenv("OVS_TTS_ALLOW_UNMAPPED_SPEAKER_ID", "0")
    with pytest.raises(ValueError):
        tts_runtime.validate_speaker_id(999999, model_id="qwen3-tts")


def test_update_with_unknown_speaker_id_raises():
    with pytest.raises(ValueError):
        tts_runtime.update_overrides(speaker_id=999999, model_id="qwen3-tts")


def test_validate_speed_out_of_range():
    with pytest.raises(ValueError):
        tts_runtime.update_overrides(speed=10.0)
    with pytest.raises(ValueError):
        tts_runtime.update_overrides(speed=0.0)


def test_validate_pitch_out_of_range():
    with pytest.raises(ValueError):
        tts_runtime.update_overrides(pitch_shift=100.0)
    with pytest.raises(ValueError):
        tts_runtime.update_overrides(pitch_shift=-100.0)


def test_reset_clears_all():
    tts_runtime.update_overrides(
        speaker_id=0, speed=1.5, pitch_shift=2.0, model_id="qwen3-tts"
    )
    tts_runtime.reset_overrides()
    snap = tts_runtime.get_overrides()
    assert snap.default_speaker_id is None
    assert snap.default_speed is None
    assert snap.default_pitch_shift is None


def test_update_returns_snapshot_with_timestamp():
    snap = tts_runtime.update_overrides(speed=1.5)
    assert snap.default_speed == 1.5
    assert snap.updated_at > 0
