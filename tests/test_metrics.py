"""Unit tests for server.core.metrics (Week 1 stub)."""

from server.core import metrics


def setup_function(_func):
    metrics._reset_for_tests()


def test_sessions_active_increments_and_decrements():
    assert metrics.get_sessions_active() == 0
    assert metrics.inc_sessions_active() == 1
    assert metrics.inc_sessions_active() == 2
    assert metrics.dec_sessions_active() == 1
    assert metrics.dec_sessions_active() == 0


def test_sessions_active_never_goes_negative():
    metrics.dec_sessions_active()
    metrics.dec_sessions_active()
    assert metrics.get_sessions_active() == 0


def test_sessions_rejected_per_reason():
    metrics.inc_sessions_rejected("http")
    metrics.inc_sessions_rejected("http")
    metrics.inc_sessions_rejected("ws")
    assert metrics.get_sessions_rejected("http") == 2
    assert metrics.get_sessions_rejected("ws") == 1
    counts = metrics.get_sessions_rejected()
    assert counts == {"http": 2, "ws": 1}


def test_auth_rejected_per_endpoint():
    metrics.inc_auth_rejected("/tts")
    metrics.inc_auth_rejected("/tts")
    metrics.inc_auth_rejected("/asr/stream")
    assert metrics.get_auth_rejected("/tts") == 2
    assert metrics.get_auth_rejected("/asr/stream") == 1


def test_snapshot_returns_full_state():
    metrics.inc_sessions_active()
    metrics.inc_sessions_rejected("http")
    metrics.inc_auth_rejected("/tts")
    snap = metrics.snapshot()
    assert snap["ovs_sessions_active"] == 1
    assert snap["ovs_sessions_rejected_total"] == {"http": 1}
    assert snap["ovs_auth_rejected_total"] == {"/tts": 1}
