"""In-process tests for LiveCaptionApp (no SLV / network)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ovs_agent.apps.live_caption.app import LiveCaptionApp
from ovs_agent.streaming_translate import SegmentCommitter
from ovs_agent.translator import NoopTranslator


def _make_app(translator, debounce_ms=0):
    """Build a LiveCaptionApp without BaseApp.__init__ (no network)."""
    app = LiveCaptionApp.__new__(LiveCaptionApp)
    app.config = SimpleNamespace(
        translator_src_lang="zho_Hans",
        translator_tgt_lang="eng_Latn",
        committer_agreement_n=2,
        committer_min_commit_chars=1,
        translate_debounce_ms=debounce_ms,
    )
    app.translator = translator
    app._committer = SegmentCommitter(agreement_n=2, strategy="retranslation")
    app._debounce_s = debounce_ms / 1000.0
    app._committed_source = ""
    app._committed_translation = ""
    app._tail_task = None
    app._events: list[dict] = []

    async def _record(hook, data):  # noqa: ANN001
        if hook == "on_translation":
            app._events.append(data)

    app._broadcast = _record  # type: ignore[assignment]
    return app


class _UpperTranslator(NoopTranslator):
    async def translate(self, text, src_lang, tgt_lang):  # noqa: ANN001
        return text.upper()


@pytest.mark.asyncio
async def test_caption_transcription_passthrough():
    """noop translator → captions echo the committed source (pure transcribe)."""
    app = _make_app(NoopTranslator())
    await app.on_user_partial("你好，")  # clause commit
    assert app._events, "expected an on_translation event on clause commit"
    last = app._events[-1]
    assert last["original"] == "你好，"
    assert last["translated"] == "你好，"  # passthrough
    assert last["is_final"] is False


@pytest.mark.asyncio
async def test_caption_translation_and_final():
    """ctranslate2-style translator output accumulates; finalize flushes tail."""
    app = _make_app(_UpperTranslator())
    await app.on_user_partial("hello, ")  # clause commit "hello,"
    await app.on_user_utterance("hello, world")  # final flush "world"
    final = app._events[-1]
    assert final["is_final"] is True
    assert final["original"] == "hello, world"
    assert final["translated"] == "HELLO, WORLD"


@pytest.mark.asyncio
async def test_caption_tail_debounce_emits_preview():
    """A changed tail (no new commit) emits a debounced preview event."""
    app = _make_app(_UpperTranslator(), debounce_ms=0)
    await app.on_user_partial("你好，A")  # commit "你好，", tail "A"
    n_before = len(app._events)
    await app.on_user_partial("你好，B")  # tail A→B, no new commit
    # The tail task runs on the loop; give it a tick.
    import asyncio

    await asyncio.sleep(0.01)
    assert len(app._events) > n_before
    tail_ev = app._events[-1]
    assert tail_ev["tail_original"] == "B"
    assert tail_ev["tail_translated"] == "B"  # upper of "B"
    assert tail_ev["original"] == "你好，"  # committed prefix unchanged
