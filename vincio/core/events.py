"""Lightweight event bus.

Subsystems emit events (run started, context compiled, tool called, memory
written...) and applications can subscribe for logging, metrics, or custom
hooks without coupling to subsystem internals.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import warnings
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, ClassVar, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from ..stability import VincioDeprecationWarning
from .utils import new_id, utcnow

__all__ = [
    "Event",
    "EventBus",
    "EventHandler",
    "EventPayload",
    "EVENT_CATALOG",
    "EVENT_SCHEMA_VERSION",
    "RunCompleted",
    "BudgetExceeded",
    "CostBudgetExceeded",
    "CostAnomaly",
    "ModelUnknown",
    "ModelRouted",
    "PolicyChanged",
    "EgressDLP",
    "DriftDetected",
    "SelfImprovementPhaseEvent",
    "DeployCompleted",
    "SourceErased",
    "PlanRepaired",
    "payload_model_for",
]

logger = logging.getLogger("vincio.events")

# Bumped only when a payload model changes incompatibly; stamped on every Event
# so external sinks can bind to a stable schema and detect version skew. The
# unified spans + metrics + cost model is the single source of truth, and the
# catalog covers the self-improvement, deploy, and provable-erasure events.
EVENT_SCHEMA_VERSION = "3.1"


class Event(BaseModel):
    id: str = Field(default_factory=lambda: new_id("evt"))
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    schema_version: str = EVENT_SCHEMA_VERSION
    created_at: Any = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Typed, versioned event catalog
# ---------------------------------------------------------------------------


class EventPayload(BaseModel):
    """Base for documented event payloads.

    Events are a typed, versioned catalog rather than stringly-typed names +
    free-form dicts: each documented event has a payload model with a stable
    ``event`` name, so observers and external sinks bind to a schema instead of
    guessing dict keys. ``extra="allow"`` keeps it forward-compatible — emitting
    extra keys never rejects an event — while the documented fields are typed.
    """

    model_config = ConfigDict(extra="allow")
    # The canonical event name this payload is published under.
    event: ClassVar[str] = ""


class RunCompleted(EventPayload):
    event: ClassVar[str] = "run.completed"
    run_id: str
    status: str
    batch: bool = False


class BudgetExceeded(EventPayload):
    event: ClassVar[str] = "budget.exceeded"
    breaches: list[str] = Field(default_factory=list)
    stage: str = ""


class CostBudgetExceeded(EventPayload):
    event: ClassVar[str] = "cost.budget_exceeded"
    action: str
    scope: str = ""
    reason: str = ""


class CostAnomaly(EventPayload):
    event: ClassVar[str] = "cost.anomaly"
    scope: str = ""
    cost_usd: float = 0.0
    mean_usd: float = 0.0
    factor: float = 0.0
    run_id: str | None = None


class ModelUnknown(EventPayload):
    event: ClassVar[str] = "model.unknown"
    model: str


class ModelRouted(EventPayload):
    event: ClassVar[str] = "model.routed"
    # Routing decisions dump a richer record; the model id is the headline field
    # but stays optional so any decision shape validates without a false warning.
    model: str = ""
    strategy: str = ""
    reason: str = ""


class PolicyChanged(EventPayload):
    event: ClassVar[str] = "policy.changed"
    policy: str


class EgressDLP(EventPayload):
    event: ClassVar[str] = "security.egress_dlp"
    run_id: str | None = None
    blocked: bool = False
    findings: list[dict[str, Any]] = Field(default_factory=list)


class DriftDetected(EventPayload):
    event: ClassVar[str] = "drift.detected"
    metric: str = ""
    score: float = 0.0
    method: str = ""


class SelfImprovementPhaseEvent(EventPayload):
    """A phase of a unified self-improvement cycle.

    Published under ``self_improvement.<phase>`` (observe / proposal / meta /
    label / reeval / canary / promote / rollback), so a sink binds to one schema
    across the whole cycle.
    """

    event: ClassVar[str] = "self_improvement.promote"
    phase: str = ""
    action: str = ""
    reason: str = ""
    promoted_ref: str | None = None
    rolled_back_to: str | None = None
    budget_spent: float = 0.0


class DeployCompleted(EventPayload):
    event: ClassVar[str] = "deploy.completed"
    prompt: str = ""
    tag: str = ""
    metric: str = ""


class SourceErased(EventPayload):
    event: ClassVar[str] = "governance.source_erased"
    source: str = ""
    found: bool = False
    proven: bool = False
    # Old payloads (and the dual-key deprecation emit) validate via the alias.
    content_hash: str | None = Field(
        default=None, validation_alias=AliasChoices("content_hash", "content_sha256")
    )

    @model_serializer(mode="wrap")
    def _dual_emit_legacy_key(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Legacy wire key, emitted alongside the canonical one until 8.0 so an
        # event consumer keyed on ``content_sha256`` keeps reading a payload
        # round-tripped through this model across the rename runway.
        data: dict[str, Any] = handler(self)
        data["content_sha256"] = data["content_hash"]
        return data

    @property
    def content_sha256(self) -> str | None:
        """Deprecated since 7.5 (removal in 8.0): use :attr:`content_hash`."""
        warnings.warn(
            "SourceErased.content_sha256 is deprecated since Vincio 7.5 and "
            "will be removed in 8.0. Use content_hash instead.",
            VincioDeprecationWarning,
            stacklevel=2,
        )
        return self.content_hash

    @content_sha256.setter
    def content_sha256(self, value: str | None) -> None:
        warnings.warn(
            "SourceErased.content_sha256 is deprecated since Vincio 7.5 and "
            "will be removed in 8.0. Assign content_hash instead.",
            VincioDeprecationWarning,
            stacklevel=2,
        )
        self.content_hash = value


class PlanRepaired(EventPayload):
    """An agent repaired its running plan in place rather than restarting.

    Emitted when the plan-repair pass re-binds, substitutes, reorders, or drops
    steps after a tool failure, a validation contradiction, or a budget shock —
    so a repair is an auditable trajectory event on the bus, not a silent retry.
    """

    event: ClassVar[str] = "plan.repaired"
    action: str = ""
    trigger: str = ""
    step_name: str = ""
    detail: str = ""


# name -> payload model. Observers can look up the schema for an event name;
# the bus validates dict payloads against it (leniently) on emit.
EVENT_CATALOG: dict[str, type[EventPayload]] = {
    cls.event: cls
    for cls in (
        RunCompleted,
        BudgetExceeded,
        CostBudgetExceeded,
        CostAnomaly,
        ModelUnknown,
        ModelRouted,
        PolicyChanged,
        EgressDLP,
        DriftDetected,
        SelfImprovementPhaseEvent,
        DeployCompleted,
        SourceErased,
        PlanRepaired,
    )
}


def payload_model_for(name: str) -> type[EventPayload] | None:
    """The documented payload model for an event name, or ``None`` if untyped."""
    return EVENT_CATALOG.get(name)


# Handlers may be sync or async; emit() ignores any return value (it only checks
# for awaitables to schedule), so the return type is intentionally permissive.
EventHandler = Callable[[Event], Any] | Callable[[Event], Awaitable[Any]]


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

    def publish(self, payload: EventPayload, *, trace_id: str | None = None) -> Event:
        """Emit a typed payload from the catalog (preferred over :meth:`emit`).

        The event name and schema come from the payload model, so callers and
        sinks share one contract instead of an ad-hoc dict.
        """
        return self.emit(type(payload).event, payload.model_dump(), trace_id=trace_id)

    @staticmethod
    def _validate_payload(name: str, payload: dict[str, Any]) -> None:
        """Best-effort validation of a dict payload against the catalog. Never
        raises — a schema mismatch is logged, not allowed to break the run."""
        model = EVENT_CATALOG.get(name)
        if model is None:
            return
        try:
            model.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 - validation must not break emit
            logger.warning("event %s payload does not match catalog schema: %s", name, exc)

    def emit(self, name: str, payload: dict[str, Any] | None = None, *, trace_id: str | None = None) -> Event:
        """Emit synchronously. Async handlers are scheduled if a loop is running."""
        data = payload or {}
        self._validate_payload(name, data)
        event = Event(name=name, payload=data, trace_id=trace_id)
        for handler in self._matching(name):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    coro = cast("Coroutine[Any, Any, Any]", result)
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        asyncio.run(coro)  # no loop: run to completion
                    else:
                        loop.create_task(coro)
            except Exception:  # noqa: BLE001 - handlers must not break runs
                logger.exception("event handler failed for %s", name)
        return event

    async def emit_async(
        self, name: str, payload: dict[str, Any] | None = None, *, trace_id: str | None = None
    ) -> Event:
        """Emit and await all handlers (async handlers awaited in order)."""
        data = payload or {}
        self._validate_payload(name, data)
        event = Event(name=name, payload=data, trace_id=trace_id)
        for handler in self._matching(name):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001
                logger.exception("event handler failed for %s", name)
        return event
