"""Lightweight event bus.

Subsystems emit events (run started, context compiled, tool called, memory
written...) and applications can subscribe for logging, metrics, or custom
hooks without coupling to subsystem internals.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from .utils import new_id, utcnow

__all__ = ["Event", "EventBus", "EventHandler"]

logger = logging.getLogger("vincio.events")


class Event(BaseModel):
    id: str = Field(default_factory=lambda: new_id("evt"))
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    created_at: Any = Field(default_factory=utcnow)


EventHandler = Callable[[Event], None] | Callable[[Event], Awaitable[None]]


class EventBus:
    """Pub/sub for runtime events.

    Handlers may be sync or async. Wildcard subscriptions use ``*`` (all
    events) or a prefix like ``tool.*``. Handler errors are logged, never
    propagated — observability must not break the run.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, name: str, handler: EventHandler) -> Callable[[], None]:
        """Register *handler* for events matching *name*. Returns unsubscribe."""
        self._handlers[name].append(handler)

        def unsubscribe() -> None:
            try:
                self._handlers[name].remove(handler)
            except ValueError:
                pass

        return unsubscribe

    def on(self, name: str) -> Callable[[EventHandler], EventHandler]:
        """Decorator form of :meth:`subscribe`."""

        def decorator(handler: EventHandler) -> EventHandler:
            self.subscribe(name, handler)
            return handler

        return decorator

    def _matching(self, name: str) -> list[EventHandler]:
        handlers = list(self._handlers.get(name, ()))
        handlers.extend(self._handlers.get("*", ()))
        for pattern, hs in self._handlers.items():
            if pattern.endswith(".*") and name.startswith(pattern[:-1]):
                handlers.extend(hs)
        return handlers

    def emit(self, name: str, payload: dict[str, Any] | None = None, *, trace_id: str | None = None) -> Event:
        """Emit synchronously. Async handlers are scheduled if a loop is running."""
        event = Event(name=name, payload=payload or {}, trace_id=trace_id)
        for handler in self._matching(name):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        asyncio.run(result)  # no loop: run to completion
                    else:
                        loop.create_task(result)
            except Exception:  # noqa: BLE001 - handlers must not break runs
                logger.exception("event handler failed for %s", name)
        return event

    async def emit_async(
        self, name: str, payload: dict[str, Any] | None = None, *, trace_id: str | None = None
    ) -> Event:
        """Emit and await all handlers (async handlers awaited in order)."""
        event = Event(name=name, payload=payload or {}, trace_id=trace_id)
        for handler in self._matching(name):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001
                logger.exception("event handler failed for %s", name)
        return event
