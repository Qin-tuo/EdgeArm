"""Mapping from ASR-reported language names to NLLB-200 FLORES codes.

The ASR backends in this repo emit human-readable language names
(e.g. ``"Chinese"``, ``"English"``) per the Qwen3-ASR language ID. NLLB
expects FLORES-200 BCP-47-ish codes (e.g. ``"zho_Hans"``, ``"eng_Latn"``).
This module owns the translation between them. Keep the table in sync
with the language set declared in
``app/backends/jetson/trt_edge_llm_asr.py::_strip_language_prefix``.

The dashboard's ``GET /api/translator/runtime`` endpoint also reads from
this module to build its ``supported_targets`` list, so adding a new
language here automatically surfaces it as a dashboard-switchable
target. Keep ``FLORES_DISPLAY_NAMES`` in sync with
``ASR_NAME_TO_FLORES`` for that reason.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


ASR_NAME_TO_FLORES: dict[str, str] = {
    "Chinese": "zho_Hans",
    "Cantonese": "yue_Hant",
    "English": "eng_Latn",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Russian": "rus_Cyrl",
}

# Human-readable display names for each FLORES code we support as a
# translator target. Used by the dashboard ``/api/translator/runtime``
# endpoint to render a target-language picker. Keys MUST stay in lockstep
# with the FLORES codes in ``ASR_NAME_TO_FLORES`` (every value in that
# map should appear here as a key).
FLORES_DISPLAY_NAMES: dict[str, str] = {
    "zho_Hans": "中文 (简体)",
    "yue_Hant": "粵語 (繁體)",
    "eng_Latn": "English",
    "jpn_Jpan": "日本語",
    "kor_Hang": "한국어",
    "fra_Latn": "Français",
    "deu_Latn": "Deutsch",
    "spa_Latn": "Español",
    "ita_Latn": "Italiano",
    "por_Latn": "Português",
    "rus_Cyrl": "Русский",
}


def supported_target_languages() -> list[dict[str, str]]:
    """Return the list of FLORES targets the dashboard can switch to.

    Format matches what the ``GET /api/translator/runtime`` endpoint
    serves: ``[{"code": "eng_Latn", "name": "English"}, ...]``. Order
    is stable (insertion order of ``FLORES_DISPLAY_NAMES``).
    """
    return [{"code": code, "name": name}
            for code, name in FLORES_DISPLAY_NAMES.items()]


# Module-level dedupe set so an unmapped language only warns once per
# process — otherwise an open-mic always-on pipeline that keeps
# detecting an unsupported language would flood the log every turn.
_UNMAPPED_WARNED: set[str] = set()


def asr_lang_to_flores(name: Optional[str]) -> Optional[str]:
    """Look up an ASR-reported language name in the FLORES table.

    Returns the FLORES code (``"zho_Hans"`` etc.) or ``None`` if ``name``
    is falsy, the literal string ``"None"`` (the trt_edge_llm ASR bailout
    sentinel for silence/noise segments — see
    ``trt_edge_llm_asr.py::_strip_language_prefix``), or not a known
    ASR-emitted language. Unknown non-sentinel values are logged at
    WARNING once per value so a missing entry in the table surfaces.

    Callers should fall back to a config default on ``None``.
    """
    if not name:
        return None
    # Explicit bailout: trt_edge_llm emits the literal string "None" for
    # silence/noise segments (not the Python ``None`` value). Treat it
    # like a missing detection without spamming a warning every time.
    if name == "None":
        return None
    code = ASR_NAME_TO_FLORES.get(name)
    if code is None and name not in _UNMAPPED_WARNED:
        _UNMAPPED_WARNED.add(name)
        logger.warning(
            "asr_lang_to_flores: unmapped ASR language: %r "
            "(falling back to None; add to ASR_NAME_TO_FLORES if supported)",
            name,
        )
    return code


__all__ = [
    "ASR_NAME_TO_FLORES",
    "FLORES_DISPLAY_NAMES",
    "asr_lang_to_flores",
    "supported_target_languages",
]
