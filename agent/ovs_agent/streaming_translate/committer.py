"""SegmentCommitter — turn a stream of ASR partials/finals into stable,
translate-ready segments.

Pure logic, no IO. Two consumption strategies:

- ``retranslation`` (live captions): the committed prefix is locked, but the
  volatile tail may be re-emitted (with a bumped ``revision``) so the renderer
  can refresh the not-yet-stable region.
- ``monotonic`` (simultaneous interpret): only committed segments are emitted;
  the tail is never re-emitted, because spoken audio can't be retracted.

Commit triggers:
- Local Agreement: the last ``agreement_n`` partials share a common prefix →
  commit that prefix.
- Clause boundary: a partial that contains a clause punctuation mark commits
  up to (and including) the last such mark, even on a single partial.
- finalize(): force-commit whatever source remains.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ASRChunk:
    """One ASR hypothesis (partial or final)."""

    text: str
    is_final: bool = False
    detected_language: str | None = None
    ts: float | None = None


@dataclass
class SegmentEvent:
    """A translate-ready signal.

    For a *commit* event ``source_text`` is the newly committed delta (what to
    translate now); ``committed_source`` is the cumulative locked prefix and
    ``tail_source`` the remaining volatile region.

    For a *tail-refresh* event (retranslation only) ``source_text`` is the new
    tail to re-translate, ``committed_source`` is unchanged, and ``revision``
    is bumped.
    """

    seq: int
    source_text: str
    committed_source: str
    tail_source: str
    is_final: bool
    strategy: str
    revision: int
    # "commit": a newly locked segment (translate + keep); also used for the
    # finalize() flush. "tail": a volatile re-translation of the unstable tail
    # (retranslation strategy only — render in the preview region).
    kind: str = "commit"


def _common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    head = strings[0]
    for s in strings[1:]:
        n = 0
        for a, b in zip(head, s):
            if a != b:
                break
            n += 1
        head = head[:n]
        if not head:
            break
    return head


class SegmentCommitter:
    def __init__(
        self,
        agreement_n: int = 2,
        strategy: Literal["retranslation", "monotonic"] = "retranslation",
        clause_punct: str = "。！？；，,.?;",
        min_commit_chars: int = 1,
    ) -> None:
        if agreement_n < 1:
            raise ValueError("agreement_n must be >= 1")
        if strategy not in ("retranslation", "monotonic"):
            raise ValueError(f"unknown strategy: {strategy!r}")
        self.agreement_n = agreement_n
        self.strategy = strategy
        self.clause_punct = set(clause_punct)
        self.min_commit_chars = min_commit_chars

        self._partials: list[str] = []
        self._committed_source: str = ""
        self._tail_source: str = ""
        self._seq: int = 0
        self._revision: int = 0

    # ── helpers ──────────────────────────────────────────────────────
    def _last_clause_end(self, text: str) -> int:
        """Index *after* the last clause punctuation char in ``text`` (0 if none)."""
        for i in range(len(text) - 1, -1, -1):
            if text[i] in self.clause_punct:
                return i + 1
        return 0

    def _commit_target(self, text: str) -> str:
        """The longest prefix of ``text`` we're confident to commit."""
        # Local Agreement over the last N partials (only once we have N).
        agreement = ""
        if len(self._partials) >= self.agreement_n:
            agreement = _common_prefix(self._partials[-self.agreement_n :])
        # Clause boundary in the current hypothesis.
        clause = text[: self._last_clause_end(text)]
        return agreement if len(agreement) >= len(clause) else clause

    # ── public API ───────────────────────────────────────────────────
    def push_partial(
        self, text: str, detected_language: str | None = None, ts: float | None = None
    ) -> list[SegmentEvent]:
        self._partials.append(text)
        if len(self._partials) > self.agreement_n:
            self._partials = self._partials[-self.agreement_n :]

        events: list[SegmentEvent] = []
        target = self._commit_target(text)

        if (
            len(target) > len(self._committed_source)
            and len(target) - len(self._committed_source) >= self.min_commit_chars
        ):
            delta = target[len(self._committed_source) :]
            self._committed_source = target
            self._tail_source = text[len(target) :]
            self._seq += 1
            events.append(
                SegmentEvent(
                    seq=self._seq,
                    source_text=delta,
                    committed_source=self._committed_source,
                    tail_source=self._tail_source,
                    is_final=False,
                    strategy=self.strategy,
                    revision=self._revision,
                    kind="commit",
                )
            )
            return events

        # No new commit. For retranslation, surface a changed tail.
        tail = text[len(self._committed_source) :]
        if self.strategy == "retranslation" and tail != self._tail_source:
            self._tail_source = tail
            self._revision += 1
            events.append(
                SegmentEvent(
                    seq=self._seq,
                    source_text=tail,
                    committed_source=self._committed_source,
                    tail_source=tail,
                    is_final=False,
                    strategy=self.strategy,
                    revision=self._revision,
                    kind="tail",
                )
            )
        return events

    def finalize(
        self, text: str, detected_language: str | None = None, ts: float | None = None
    ) -> list[SegmentEvent]:
        events: list[SegmentEvent] = []
        if len(text) > len(self._committed_source):
            delta = text[len(self._committed_source) :]
            self._seq += 1
            events.append(
                SegmentEvent(
                    seq=self._seq,
                    source_text=delta,
                    committed_source=text,
                    tail_source="",
                    is_final=True,
                    strategy=self.strategy,
                    revision=self._revision,
                    kind="commit",
                )
            )
        # Reset per-utterance text buffers; keep counters monotonic so seq/
        # revision stay globally unique across utterances on the same session.
        self._partials = []
        self._committed_source = ""
        self._tail_source = ""
        return events

    def reset(self) -> None:
        self._partials = []
        self._committed_source = ""
        self._tail_source = ""
        self._seq = 0
        self._revision = 0
