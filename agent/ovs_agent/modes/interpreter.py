"""InterpreterMode — real-time translator backed by the NLLB translator.

Behavior (Phase 1 of the ASR-language → translator src pipeline):
  1. Read ``ctx.detected_language`` (ASR-reported language name) and map
     it to an NLLB FLORES code; on miss fall back to
     ``ctx.config.translator_src_lang``.
  2. Use ``ctx.config.translator_tgt_lang`` as the target.
  3. If src == tgt OR the translator backend is the no-op stub, fall
     back to the LLM dialogue turn (the legacy behavior) so the user
     still hears something useful.
  4. Otherwise call ``ctx.translator.translate(text, src, tgt)`` and
     speak the result with ``ctx.speak``.

Multi-language TTS and per-session target-language switching from the
dashboard are deferred to later phases of this work.
"""
from __future__ import annotations

import logging

from ..app_mode import AppMode, ModeContext
from ..translator.lang_map import asr_lang_to_flores

logger = logging.getLogger(__name__)


class InterpreterMode(AppMode):
    name = "interpreter"
    display_name = "同传"
    icon = "🌐"
    description = "把听到的话翻译成目标语言，不维护对话上下文"
    # Kept for the LLM fallback path; ignored when NLLB is in use.
    system_prompt = (
        "You are a real-time interpreter. Translate the user's input "
        "into natural spoken English. Output ONLY the translation, no "
        "preface or explanation."
    )
    max_history = 0  # interpreter is stateless

    async def on_user_utterance(self, ctx: ModeContext, text: str) -> None:
        # Interpreter is stateless — clear history every turn.
        ctx.session.history.clear()

        cfg = ctx.config
        tgt_lang = getattr(cfg, "translator_tgt_lang", None)
        config_src_lang = getattr(cfg, "translator_src_lang", None)
        backend_name = getattr(cfg, "translator_backend", "noop")

        # Map ASR-reported language → FLORES, fall back to config default.
        mapped_src = asr_lang_to_flores(ctx.detected_language)
        src_lang = mapped_src or config_src_lang

        # Decide whether to use the translator path or fall back to LLM.
        use_translator = True
        fallback_reason: str | None = None
        if backend_name == "noop" or ctx.translator is None:
            use_translator = False
            fallback_reason = "translator_backend=noop"
        elif not src_lang or not tgt_lang:
            use_translator = False
            fallback_reason = "missing src/tgt lang"
        elif src_lang == tgt_lang:
            use_translator = False
            fallback_reason = f"src==tgt ({src_lang})"

        if not use_translator:
            logger.info(
                "InterpreterMode: LLM fallback (detected=%r mapped_src=%r "
                "config_src=%r tgt=%r reason=%s)",
                ctx.detected_language, mapped_src, config_src_lang,
                tgt_lang, fallback_reason,
            )
            await ctx.run_default_dialogue_turn(text)
            return

        logger.info(
            "InterpreterMode: translate src=%s tgt=%s (detected=%r mapped=%s)",
            src_lang, tgt_lang, ctx.detected_language,
            "yes" if mapped_src else "fallback-to-config",
        )
        try:
            translated = await ctx.translator.translate(text, src_lang, tgt_lang)
        except Exception:
            logger.exception(
                "InterpreterMode: translator failed; falling back to LLM"
            )
            await ctx.run_default_dialogue_turn(text)
            return

        if not translated or not translated.strip():
            logger.warning(
                "InterpreterMode: translator returned empty; falling back to LLM"
            )
            await ctx.run_default_dialogue_turn(text)
            return

        # Phase 2a: tell the dashboard (and any other broadcast
        # subscriber) what we translated so it can render side-by-side
        # subtitles. Failures here must NOT block the spoken output —
        # broadcast is best-effort UX.
        try:
            await ctx.broadcast("on_translation", {
                "original": text,
                "translated": translated,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "detected_language": ctx.detected_language,
            })
        except Exception:
            logger.exception(
                "InterpreterMode: on_translation broadcast failed (non-fatal)"
            )

        await ctx.speak(translated)


__all__ = ["InterpreterMode"]
