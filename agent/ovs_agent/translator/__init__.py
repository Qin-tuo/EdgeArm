"""Text translation backends."""
from .base import TranslatorBackend
from .http import CTranslate2Translator
from .lang_map import ASR_NAME_TO_FLORES, asr_lang_to_flores
from .noop import NoopTranslator

__all__ = [
    "TranslatorBackend",
    "NoopTranslator",
    "CTranslate2Translator",
    "ASR_NAME_TO_FLORES",
    "asr_lang_to_flores",
]
