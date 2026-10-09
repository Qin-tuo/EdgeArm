"""Pluggable actuator abstraction for the OpenVoiceStream Agent.

An ``Actuator`` owns a physical motion device (a robot arm, a pan/tilt
head, a gripper, …) plus its observation cache. The voice pipeline and
the observation HTTP server talk to the actuator only through the
``Actuator`` ABC. This package holds ONLY the abstraction — the ABC plus a
``register_actuator`` / ``create_actuator`` registry. Concrete drivers live
in the app that owns the hardware (e.g. the SO-ARM driver in
``apps/voice_arm/so_arm.py``, which self-registers on import). A new motor
is therefore a new app, not a change here.
"""
from .base import Actuator
from .factory import create_actuator, register_actuator

__all__ = ["Actuator", "create_actuator", "register_actuator"]
