"""Lightweight async-safe pub/sub event bus.

Ported from clawd-reachy-mini/src/reachy_claw/event_bus.py; robot-specific
behaviour stripped. Callbacks may be sync or async; exceptions are logged
and swallowed.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventBus:
    """Simple publish/subscribe event bus with async support."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event: str, callback: Callable) -> None:
        self._subscribers[event].append(callback)

    def unsubscribe(self, event: str, callback: Callable) -> None:
        try:
            self._subscribers[event].remove(callback)
        except ValueError:
            pass

    def emit(self, event: str, data: Any = None) -> None:
        """Emit an event. Auto-awaits async callbacks via the running loop."""
        for cb in self._subscribers.get(event, []):
            try:
                if inspect.iscoroutinefunction(cb):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(cb(data))
                    except RuntimeError:
                        # No running loop -- run synchronously.
                        asyncio.run(cb(data))
                else:
                    cb(data)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("EventBus callback error on %s: %s", event, e)

    def emit_sync(self, event: str, data: Any = None) -> None:
        """Thread-safe emit: schedule async callbacks on the running loop."""
        for cb in self._subscribers.get(event, []):
            try:
                if inspect.iscoroutinefunction(cb):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(cb(data))
                    except RuntimeError:
                        try:
                            loop = asyncio.get_event_loop()
                            asyncio.run_coroutine_threadsafe(cb(data), loop)
                        except Exception:
                            pass
                else:
                    cb(data)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("EventBus emit_sync error on %s: %s", event, e)
