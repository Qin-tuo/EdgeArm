"""InterpreterMode → NLLB translator pipeline (Phase 1).

Verifies:
  (a) ASR-detected language maps to FLORES src and translator is called.
  (b) When src == tgt the LLM fallback path runs (no translator call).
  (c) When ``detected_language`` is None the config default src is used.
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
    """Records messages it's asked to stream; yields a single sentinel
    token so the LLM-fallback path produces something to assert on."""

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
    llm: _FakeLLM | None = None,
    slv: _FakeSLV | None = None,
) -> tuple[ModeContext, _FakeLLM, _FakeSLV, list[tuple]]:
    cfg = Config(
        system_prompt="SYS",
        translator_backend=translator_backend,
        translator_src_lang=src_lang,
        translator_tgt_lang=tgt_lang,
    )
    llm = llm or _FakeLLM()
    slv = slv or _FakeSLV()
    broadcasts: list[tuple] = []

    async def _br(name, *args):
        broadcasts.append((name, args))

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
    # Wire up a ModeManager so system_prompt resolution paths used by
    # the LLM fallback don't NPE.
    mgr = ModeManager(lambda: ctx)
    im = InterpreterMode()
    mgr.register(im)
    ctx.mode_manager = mgr
    mgr._current = im  # skip async start() ceremony
    return ctx, llm, slv, broadcasts


# (a) ASR-detected language drives translator src
@pytest.mark.asyncio
async def test_interpreter_uses_detected_language_as_src():
    translator = _RecordingTranslator(response="Hello world")
    ctx, llm, slv, _ = _make_ctx(
        detected_language="Chinese",
        translator=translator,
        tgt_lang="eng_Latn",
    )
    await InterpreterMode().on_user_utterance(ctx, "你好世界")

    assert translator.calls == [("你好世界", "zho_Hans", "eng_Latn")]
    # No LLM round-trip on the translator path.
    assert llm.calls == []
    # Translation spoken verbatim.
    assert slv.text_frames == ["Hello world"]
    assert slv.flushed == 1


# (b) src == tgt → LLM fallback (no translator call)
@pytest.mark.asyncio
async def test_interpreter_src_equals_tgt_falls_back_to_llm():
    translator = _RecordingTranslator(response="should not be used")
    ctx, llm, slv, _ = _make_ctx(
        detected_language="English",
        translator=translator,
        tgt_lang="eng_Latn",
    )
    await InterpreterMode().on_user_utterance(ctx, "hello world")

    # No translator call — src maps to eng_Latn which equals tgt.
    assert translator.calls == []
    # LLM path ran instead.
    assert len(llm.calls) == 1
    # The LLM fallback produces a streamed token (sentinel).
    assert slv.text_frames == ["LLM-FALLBACK"]


# (c) detected_language is None → fall back to config src
@pytest.mark.asyncio
async def test_interpreter_falls_back_to_config_src_when_detection_missing():
    translator = _RecordingTranslator(response="bonjour")
    ctx, llm, slv, _ = _make_ctx(
        detected_language=None,
        translator=translator,
        src_lang="zho_Hans",
        tgt_lang="fra_Latn",
    )
    await InterpreterMode().on_user_utterance(ctx, "你好")

    # Config src used because detection is missing.
    assert translator.calls == [("你好", "zho_Hans", "fra_Latn")]
    assert llm.calls == []
    assert slv.text_frames == ["bonjour"]
    assert slv.flushed == 1


# Sanity: noop translator backend always falls back to LLM, even with
# a viable detected language and a wired translator object.
@pytest.mark.asyncio
async def test_interpreter_noop_backend_falls_back_to_llm():
    translator = _RecordingTranslator(response="ignored")
    ctx, llm, slv, _ = _make_ctx(
        detected_language="Chinese",
        translator_backend="noop",
        translator=translator,
        tgt_lang="eng_Latn",
    )
    await InterpreterMode().on_user_utterance(ctx, "你好")

    assert translator.calls == []
    assert len(llm.calls) == 1
    assert slv.text_frames == ["LLM-FALLBACK"]


# (d) translator.translate() raises → fall back to LLM
class _RaisingTranslator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        self.calls.append((text, src_lang, tgt_lang))
        raise RuntimeError("nllb backend transient failure")


@pytest.mark.asyncio
async def test_interpreter_translator_raises_falls_back_to_llm():
    translator = _RaisingTranslator()
    ctx, llm, slv, _ = _make_ctx(
        detected_language="Chinese",
        translator=translator,
        tgt_lang="eng_Latn",
    )
    await InterpreterMode().on_user_utterance(ctx, "你好世界")

    # Translator was attempted exactly once before the LLM fallback fired.
    assert translator.calls == [("你好世界", "zho_Hans", "eng_Latn")]
    # LLM fallback ran.
    assert len(llm.calls) == 1
    assert slv.text_frames == ["LLM-FALLBACK"]


# (e) translator returns empty/whitespace → fall back to LLM
@pytest.mark.asyncio
async def test_interpreter_empty_translation_falls_back_to_llm():
    translator = _RecordingTranslator(response="   ")
    ctx, llm, slv, _ = _make_ctx(
        detected_language="Chinese",
        translator=translator,
        tgt_lang="eng_Latn",
    )
    await InterpreterMode().on_user_utterance(ctx, "你好")

    # Translator was called once; its empty response triggered fallback.
    assert translator.calls == [("你好", "zho_Hans", "eng_Latn")]
    assert len(llm.calls) == 1
    assert slv.text_frames == ["LLM-FALLBACK"]


# (f) detected_language is an unknown name (not in FLORES map) → use
# the config-pinned translator_src_lang; translator still called.
@pytest.mark.asyncio
async def test_interpreter_unknown_detected_language_falls_back_to_config():
    translator = _RecordingTranslator(response="xin chào")
    ctx, llm, slv, _ = _make_ctx(
        detected_language="Vietnamese",  # not in ASR_NAME_TO_FLORES
        translator=translator,
        src_lang="zho_Hans",
        tgt_lang="vie_Latn",
    )
    await InterpreterMode().on_user_utterance(ctx, "你好")

    # Unknown ASR name → fall back to config's translator_src_lang.
    assert translator.calls == [("你好", "zho_Hans", "vie_Latn")]
    # No LLM fallback when translator path is viable.
    assert llm.calls == []
    assert slv.text_frames == ["xin chào"]
    assert slv.flushed == 1
