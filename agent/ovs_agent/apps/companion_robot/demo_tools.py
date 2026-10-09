"""Throwaway mock robot tools for the server-loop round-trip proof.

These register two ``@tool`` handlers onto the SAME ``default_registry`` that
``CompanionRobotApp`` advertises in server-loop mode. They do NOT touch real
motors — each just logs what it *would* do and returns a small ack dict so the
engine's tool pump can continue to TTS. This is a proof slice, not the real
Reachy wiring (that lives in clawd-reachy-mini).

Importing this module has the side effect of registering the tools; the
companion_robot package ``__init__`` imports it so the tools exist before the
agent's ``_advertise_tools_if_server_loop()`` runs at session open.
"""
from __future__ import annotations

import logging
from typing import Any

from ovs_agent.tools import default_registry as _r

logger = logging.getLogger(__name__)


@_r.tool(
    description=(
        "Point the robot's head at a target orientation, in radians. "
        "yaw = left/right, pitch = up/down, roll = tilt."
    ),
    preamble_text="好的。",
)
def move_head(yaw: float, pitch: float, roll: float = 0.0) -> dict[str, Any]:
    """Mock head-move: log the target and return an ack (no real motors)."""
    logger.info(
        "[demo_tools] move_head FIRED yaw=%.3f pitch=%.3f roll=%.3f "
        "(mock — no motors)", yaw, pitch, roll,
    )
    return {"ok": True, "moved_to": {"yaw": yaw, "pitch": pitch, "roll": roll}}


@_r.tool(
    description=(
        "Play an emotion animation on the robot's face/antennae. "
        "emotion is a slug like 'happy', 'sad', 'curious', 'neutral'."
    ),
    preamble_text="好的。",
)
def play_emotion(emotion: str) -> dict[str, Any]:
    """Mock emotion playback: log the slug and return an ack."""
    logger.info(
        "[demo_tools] play_emotion FIRED emotion=%r (mock — no animation)",
        emotion,
    )
    return {"ok": True, "emotion": emotion}


__all__ = ["move_head", "play_emotion"]
