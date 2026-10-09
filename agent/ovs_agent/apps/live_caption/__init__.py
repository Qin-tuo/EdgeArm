"""live_caption — real-time bilingual captions (transcription + translation)
from streaming ASR. Loaded by the CLI as ``apps.live_caption.app:App``.
"""
from .app import App, LiveCaptionApp

__all__ = ["App", "LiveCaptionApp"]
