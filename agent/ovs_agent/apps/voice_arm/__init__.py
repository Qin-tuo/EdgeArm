"""voice_arm — actuator (SO-ARM) voice agent built on MultiModeApp.

Loaded by the ovs-agent CLI as ``apps.voice_arm.app:App``. The actuator
itself is pluggable: ``metadata.actuator.backend`` selects the concrete
driver (currently ``so_arm``); future motors implement the same
``ovs_agent.actuators.base.Actuator`` ABC.
"""
# Importing the concrete driver registers it with the framework's actuator
# factory (register_actuator("so_arm", ...)), so create_actuator("so_arm") works.
# This keeps the framework driver-agnostic — the app owns its hardware.
from . import so_arm  # noqa: F401  (import for side-effect: driver registration)
from .app import App, VoiceArmApp

__all__ = ["App", "VoiceArmApp"]
