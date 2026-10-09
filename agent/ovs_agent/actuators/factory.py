"""factory.py — build a concrete :class:`Actuator` from config by name.

The framework owns only the *abstraction*: the :class:`Actuator` ABC and
this registry. Concrete drivers live in the app that uses them — e.g. the
SO-ARM driver ships in ``apps/voice_arm/`` and registers itself via
``register_actuator("so_arm", ...)`` when that app is imported. The
framework never imports a concrete motor driver, so a new motor is a new
app, not a change here.

Config shape (lives under ``metadata.actuator`` in the agent YAML)::

    metadata:
      actuator:
        backend: so_arm          # registry name the app registered
        config:
          port: /dev/ttyACM0
          ...

``create_actuator("so_arm", {...})`` returns whatever the app's registered
builder produces; the caller (ArmPlugin) runs ``connect()`` in
``asyncio.to_thread`` so the sync serial setup doesn't block.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from .base import Actuator

# Populated by apps at import time via ``register_actuator``. The framework
# ships it EMPTY — it knows no concrete drivers.
_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Actuator]] = {}


def register_actuator(
    name: str, builder: Callable[[Dict[str, Any]], Actuator]
) -> None:
    """Register a concrete actuator builder under ``name``.

    Apps call this at import time (e.g. ``apps/voice_arm/so_arm.py``) so the
    framework stays driver-agnostic. Re-registering the same name overrides.
    """
    _REGISTRY[name] = builder


def create_actuator(name: str, config: Dict[str, Any]) -> Actuator:
    """Build an :class:`Actuator` by registry name.

    Raises ``ValueError`` for a backend name no app has registered (most
    likely the owning app wasn't imported before this call).
    """
    builder = _REGISTRY.get(name)
    if builder is None:
        raise ValueError(
            f"unknown actuator backend {name!r}; "
            f"registered: {sorted(_REGISTRY)} "
            f"(is the owning app imported so it can register its driver?)"
        )
    return builder(dict(config or {}))


__all__ = ["create_actuator", "register_actuator"]
