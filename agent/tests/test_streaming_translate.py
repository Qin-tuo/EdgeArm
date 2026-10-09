"""Unit tests for the streaming-translate primitives + barge-in gate."""
from __future__ import annotations

from types import SimpleNamespace

from ovs_agent import BaseApp
from ovs_agent.streaming_translate import EchoFilter, SegmentCommitter


# ── SegmentCommitter ─────────────────────────────────────────────────
def test_committer_agreement():
    """Two partials sharing a common prefix commit that prefix (N=2).

    Uses monotonic to isolate the agreement commit (retranslation would also
    surface tail-preview events on each partial)."""
    c = SegmentCommitter(agreement_n=2, strategy="monotonic")
    assert c.push_partial("今天天气") == []  # only 1 partial, no clause → no commit
    events = c.push_partial("今天天气不错")
    assert len(events) == 1
    assert events[0].committed_source == "今天天气"
    assert events[0].source_text == "今天天气"
    assert events[0].is_final is False


def test_committer_clause():
    """A single partial containing clause punctuation commits immediately."""
    c = SegmentCommitter(agreement_n=2, strategy="monotonic")
    events = c.push_partial("你好，")
    assert len(events) == 1
    assert events[0].committed_source == "你好，"
    assert events[0].source_text == "你好，"


def test_committer_final_flush():
    """finalize() force-commits the remaining tail with is_final=True."""
    c = SegmentCommitter(agreement_n=2, strategy="monotonic")
    c.push_partial("你好，")  # commit "你好，"
    events = c.finalize("你好，世界")
    assert len(events) == 1
    assert events[0].is_final is True
    assert events[0].source_text == "世界"
    assert events[0].committed_source == "你好，世界"
    assert events[0].tail_source == ""


def test_committer_retranslation_tail_refresh():
    """retranslation strategy re-emits a changed tail with a bumped revision."""
    c = SegmentCommitter(agreement_n=2, strategy="retranslation")
    c.push_partial("你好，A")  # clause commit "你好，", tail "A", rev 0
    events = c.push_partial("你好，B")  # no new commit; tail A→B
    assert len(events) == 1
    ev = events[0]
    assert ev.committed_source == "你好，"  # locked
    assert ev.tail_source == "B"
    assert ev.source_text == "B"
    assert ev.revision == 1  # bumped


def test_committer_monotonic_no_revoke():
    """monotonic strategy never emits tail-refresh (revoke) events."""
    c = SegmentCommitter(agreement_n=2, strategy="monotonic")
    c.push_partial("你好，A")  # clause commit "你好，"
    events = c.push_partial("你好，B")  # tail changed, but monotonic stays silent
    assert events == []


# ── barge-in gate (BaseApp) ──────────────────────────────────────────
class _GateApp(BaseApp):
    """Bare BaseApp shell that skips heavy __init__; just exercises the gate."""

    def __init__(self, config) -> None:  # noqa: D401 - test stub
        self.config = config


def test_barge_in_gate_disabled():
    # Unconfigured (None) → legacy always-on behaviour preserved.
    assert _GateApp(SimpleNamespace(barge_in_enabled=None))._barge_in_enabled() is True
    # Missing attribute entirely → also defaults to True.
    assert _GateApp(SimpleNamespace())._barge_in_enabled() is True
    # Explicit False → disabled.
    assert _GateApp(SimpleNamespace(barge_in_enabled=False))._barge_in_enabled() is False
    # Explicit True → enabled.
    assert _GateApp(SimpleNamespace(barge_in_enabled=True))._barge_in_enabled() is True


def test_barge_in_gate_mode_override_wins():
    """Active-mode override takes precedence over config."""
    app = _GateApp(SimpleNamespace(barge_in_enabled=True))
    app._active_mode_barge_in_override = lambda: False  # type: ignore[method-assign]
    assert app._barge_in_enabled() is False


# ── EchoFilter ───────────────────────────────────────────────────────
def test_echo_filter_high_similarity():
    f = EchoFilter(threshold=0.82, window_s=4.0, min_len=4)
    f.add_tts("The weather is nice today", ts=0.0)
    assert f.is_echo("the weather is nice today", ts=1.0) is True


def test_echo_filter_window_expiry():
    f = EchoFilter(threshold=0.82, window_s=4.0, min_len=4)
    f.add_tts("The weather is nice today", ts=0.0)
    assert f.is_echo("the weather is nice today", ts=10.0) is False


def test_echo_filter_short_text():
    f = EchoFilter(threshold=0.82, window_s=4.0, min_len=4)
    f.add_tts("hello there friend", ts=0.0)
    assert f.is_echo("hi", ts=0.5) is False
