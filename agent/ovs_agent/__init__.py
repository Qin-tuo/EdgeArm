"""ovs_agent — agent-layer client on top of SLV /v2v/stream.

Architecture (Phase 1):

    mic -> [SLV /v2v/stream] -> asr_final -> [LLM streaming] -> text frames
                                                                     |
                                                                     v
                                              [SLV server-side sentence split + TTS]
                                                                     |
                                                                     v
                                                            binary PCM -> speaker

HARD invariants (see agent/README.md):
  1. ONE persistent WS for the App lifetime, `multi_utterance: true`.
  2. LLM tokens stream DIRECTLY into SLV `text` frames -- no client-side
     sentence buffering. SLV runs SentenceBuffer server-side.
  3. Session history is sent FULL to the LLM, no client-side trimming.
  4. Barge-in: on `asr_partial` while playing -> send `abort`.
  5. Plugin hooks are observer broadcasts, NOT routers.
  6. Protocol constants live in `ovs_agent.protocol` (vendored from the
     server's `app/core/v2v.py`, kept as the single source of truth) — never
     redeclared ad-hoc elsewhere.

The agent package is fully self-contained: it has NO import dependency on the
SLV server source tree (the wire-protocol constants are vendored in
`ovs_agent/protocol.py`), so the image ships only `agent/` and survives the
server-side `app/`→`server/` rename.
"""
from __future__ import annotations

# Public re-exports.
from .config import Config, load_config  # noqa: E402
from .event_bus import EventBus  # noqa: E402
from .plugin import Plugin  # noqa: E402
from .session import Session  # noqa: E402
from .slv_client import (  # noqa: E402
    ASREndpoint,
    ASRFinal,
    ASRPartial,
    SLVClient,
    SLVError,
    TTSAudio,
    TTSDone,
    TTSSentenceDone,
    TTSStarted,
    V2VEvent,
)
from .audio_io import AudioIO  # noqa: E402
from .app_base import BaseApp  # noqa: E402

__all__ = [
    "AudioIO",
    "BaseApp",
    "Config",
    "EventBus",
    "Plugin",
    "Session",
    "SLVClient",
    "SLVError",
    "V2VEvent",
    "ASRPartial",
    "ASREndpoint",
    "ASRFinal",
    "TTSStarted",
    "TTSSentenceDone",
    "TTSDone",
    "TTSAudio",
    "load_config",
]
