"""In-process tests for SimulInterpretApp (no SLV / network)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ovs_agent.apps.simul_interpret.app import SimulInterpretApp
from ovs_agent.streaming_translate import EchoFilter, SegmentCommitter
from ovs_agent.translator import NoopTranslator


class _FakeSLV:
    def __init__(self):
        self.sent: list[str] = []
        self.flushes = 0

    async def send_text(self, text):  # noqa: ANN001
        self.sent.append(text)

    async def flush_tts(self):
        self.flushes += 1


class _UpperTranslator(NoopTranslator):
    async def translate(self, text, src_lang, tgt_lang):  # noqa: ANN001
        return text.upper()


def _make_app(*, overlap, echo=False, translator=None):
    app = SimulInterpretApp.__new__(SimulInterpretApp)
    app.config = SimpleNamespace(
        translator_src_lang="zho_Hans",
        translator_tgt_lang="eng_Latn",
    )
    app.translator = translator or _UpperTranslator()
    app.slv = _FakeSLV()
    app._committer = SegmentCommitter(agreement_n=2, strategy="monotonic")
    app._overlap = overlap
    app._echo = EchoFilter(threshold=0.82, window_s=4.0) if echo else None
    app._pending_translation = ""
    app._pending_source = ""

    async def _record(hook, data):  # noqa: ANN001
        pass

    app._broadcast = _record  # type: ignore[assignment]
    return app


@pytest.mark.asyncio
async def test_interpret_clause_lag_speaks_on_final():
    """overlap=off: nothing spoken mid-utterance; one flush on final."""
    app = _make_app(overlap=False)
    await app.on_user_partial("hello, ")  # commits "hello," but buffers
    assert app.slv.sent == []  # clause-lag: not spoken yet
    await app.on_user_utterance("hello, world")
    assert len(app.slv.sent) == 1
    assert app.slv.sent[0] == "HELLO, WORLD"
    assert app.slv.flushes == 1


@pytest.mark.asyncio
async def test_interpret_overlap_speaks_per_clause():
    """overlap=on: each committed clause is spoken immediately."""
    app = _make_app(overlap=True)
    await app.on_user_partial("你好，")  # clause commit → speak now
    assert app.slv.sent == ["你好，".upper()] or app.slv.sent == ["你好，"]
    assert app.slv.flushes == 1


@pytest.mark.asyncio
async def test_interpret_echo_filter_drops_self_echo():
    """A partial matching a recently spoken translation is dropped."""
    app = _make_app(overlap=True, echo=True)
    await app.on_user_partial("你好，")  # commit + speak; echo records translation
    spoken = app.slv.sent[-1]
    n = len(app.slv.sent)
    # Feed back our own spoken translation as a partial → must be dropped.
    await app.on_user_partial(spoken)
    assert len(app.slv.sent) == n  # nothing new spoken
