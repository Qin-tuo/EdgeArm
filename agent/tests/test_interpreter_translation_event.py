"""InterpreterMode → on_translation broadcast event (Phase 2a).

Verifies:
  (a) Successful translator path emits exactly one ``on_translation``
      broadcast with the {original, translated, src_lang, tgt_lang,
      detected_language} payload.
  (b) LLM fallback path (noop backend) does NOT emit ``on_translation``.
"""
from __future__ import annotations

from typing import Any

import pytest

from ovs_agent import Config, Session
from ovs_agent.app_mode import ModeContext, ModeManager
from ovs_agent.modes import InterpreterMode


class _FakeSLV:
    def __init__(self) -> None:
        self.text_frames: list[str] = []
        self.flushed: int = 0

    async def send_text(self, text: str) -> None:
        self.text_frames.append(text)

    async def flush_tts(self) -> None:
        self.flushed += 1


class _FakeLLM:
    def __init__(self, tokens: list[str] | None = None) -> None:
        self.tokens = tokens if tokens is not None else ["LLM-FALLBACK"]
        self.calls: list[list[dict]] = []
        self.last_cache_metrics: dict | None = None

    async def stream(self, messages, **kw):
        self.calls.append(list(messages))
        for t in self.tokens:
            yield t

    async def aclose(self) -> None:
        pass


class _RecordingTranslator:
    def __init__(self, response: str = "translated text") -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._response = response

    async def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        self.calls.append((text, src_lang, tgt_lang))
        return self._response


def _make_ctx(
    *,
    detected_language: str | None,
    translator_backend: str = "ctranslate2",
    translator: Any = None,
    src_lang: str = "zho_Hans",
    tgt_lang: str = "eng_Latn",
) -> tuple[ModeContext, _FakeLLM, _FakeSLV, list[tuple]]:
    cfg = Config(
        system_prompt="SYS",
        translator_backend=translator_backend,
        translator_src_lang=src_lang,
        translator_tgt_lang=tgt_lang,
    )
    llm = _FakeLLM()
    slv = _FakeSLV()
    broadcasts: list[tuple[str, Any]] = []

    async def _br(name, payload=None):
        broadcasts.append((name, payload))

    events = type("E", (), {"emit": lambda *a, **k: None})()
    ctx = ModeContext(
        config=cfg,
        slv=slv,
        llm=llm,
        session=Session(),
        audio=None,
        events=events,
        broadcast=_br,
        translator=translator,
        detected_language=detected_language,
    )
    mgr = ModeManager(lambda: ctx)
    im = InterpreterMode()
    mgr.register(im)
    ctx.mode_manager = mgr
    mgr._current = im
    return ctx, llm, slv, broadcasts


@pytest.mark.asyncio
async def test_on_translation_event_emitted_on_translator_path():
    translator = _RecordingTranslator(response="Hello world")
    ctx, llm, slv, broadcasts = _make_ctx(
        detected_language="Chinese",
        translator=translator,
        tgt_lang="eng_Latn",
    )
    await InterpreterMode().on_user_utterance(ctx, "你好世界")

    # Translator ran and audio was spoken.
    assert translator.calls == [("你好世界", "zho_Hans", "eng_Latn")]
    assert slv.text_frames == ["Hello world"]

    # Exactly one on_translation event with all five expected fields.
    tx_events = [b for b in broadcasts if b[0] == "on_translation"]
    assert len(tx_events) == 1
    _, payload = tx_events[0]
    assert payload == {
        "original": "你好世界",
        "translated": "Hello world",
        "src_lang": "zho_Hans",
        "tgt_lang": "eng_Latn",
        "detected_language": "Chinese",
    }


@pytest.mark.asyncio
async def test_no_translation_event_on_llm_fallback():
    """When the backend is noop, InterpreterMode goes through the LLM
    fallback path and must NOT emit on_translation (those events are
    reserved for the real translator path)."""
    translator = _RecordingTranslator(response="ignored")
    ctx, llm, slv, broadcasts = _make_ctx(
        detected_language="Chinese",
        translator_backend="noop",
        translator=translator,
        tgt_lang="eng_Latn",
    )
    await InterpreterMode().on_user_utterance(ctx, "你好")

    assert translator.calls == []
    assert len(llm.calls) == 1
    assert slv.text_frames == ["LLM-FALLBACK"]

    tx_events = [b for b in broadcasts if b[0] == "on_translation"]
    assert tx_events == []


@pytest.mark.asyncio
async def test_no_translation_event_when_translator_raises():
    """Failure path: translator raises → LLM fallback → no event."""
    class _RaisingTranslator:
        async def translate(self, text, src, tgt):
            raise RuntimeError("boom")

    ctx, llm, slv, broadcasts = _make_ctx(
        detected_language="Chinese",
        translator=_RaisingTranslator(),
        tgt_lang="eng_Latn",
    )
    await InterpreterMode().on_user_utterance(ctx, "你好")

    tx_events = [b for b in broadcasts if b[0] == "on_translation"]
    assert tx_events == []
    assert len(llm.calls) == 1
