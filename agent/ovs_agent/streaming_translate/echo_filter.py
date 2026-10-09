"""EchoFilter — software backstop for self-echo during full-duplex
simultaneous interpretation.

When ``overlap_mode`` is on, the device speaks the translation while the mic
stays open. The reSpeaker XVF3800 hardware AEC cancels most of it, but leaks
the first ~200-500ms of output (see app_base.py near the echo-blip barge-in
suppression). That residual is our own *translated* text — if ASR transcribes
it and we translate it again, we self-pollute. This filter drops partials that
closely match recently-spoken translations.

Pure logic + stdlib only (``difflib``); no external deps.
"""
from __future__ import annotations

import time
from collections import deque
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    """Lowercase and drop whitespace + punctuation for comparison."""
    return "".join(ch.lower() for ch in text if ch.isalnum())


class EchoFilter:
    def __init__(
        self,
        threshold: float = 0.82,
        window_s: float = 4.0,
        min_len: int = 4,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self.threshold = threshold
        self.window_s = window_s
        self.min_len = min_len
        # (normalized_text, ts)
        self._recent: deque[tuple[str, float]] = deque()

    def add_tts(self, text: str, ts: float | None = None) -> None:
        """Record a translation we just sent to TTS."""
        norm = _normalize(text)
        if not norm:
            return
        self._recent.append((norm, self._now(ts)))

    def is_echo(self, partial: str, ts: float | None = None) -> bool:
        now = self._now(ts)
        self._prune(now)
        norm = _normalize(partial)
        if len(norm) < self.min_len:
            return False
        for tts_norm, _ in self._recent:
            if not tts_norm:
                continue
            if norm in tts_norm or tts_norm in norm:
                return True
            if SequenceMatcher(None, norm, tts_norm).ratio() >= self.threshold:
                return True
        return False

    # ── internals ────────────────────────────────────────────────────
    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._recent and self._recent[0][1] < cutoff:
            self._recent.popleft()

    @staticmethod
    def _now(ts: float | None) -> float:
        return time.monotonic() if ts is None else ts
