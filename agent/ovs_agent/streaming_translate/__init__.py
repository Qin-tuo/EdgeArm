"""Streaming-translation primitives shared by the live-caption and
simultaneous-interpret apps.

- ``SegmentCommitter`` turns ASR partial/final flow into stable, translate-ready
  segments (local-agreement + clause boundaries + final flush).
- ``EchoFilter`` drops self-echo partials during full-duplex interpret.
"""
from __future__ import annotations

from .committer import ASRChunk, SegmentCommitter, SegmentEvent
from .echo_filter import EchoFilter

__all__ = ["ASRChunk", "SegmentCommitter", "SegmentEvent", "EchoFilter"]
