"""Durable timers and scheduled steps (agents/timers).

A long-running process should be able to *wait* — for a wall-clock delay, an
approval, or an external webhook — without pinning a worker for the duration. A
durable graph already pauses and persists its state at an interrupt; this module
turns that into first-class timer nodes:

* :func:`sleep_until` / :func:`sleep_for` pause the graph until a wake time;
* :func:`wait_for_event` pause it until a named event is delivered.

The wake condition rides the checkpoint's interrupt payload, so it survives a
restart: a brand-new process can scan the store, find the timers that are now
due, and resume them. While paused, no worker is held — the thread is just a
durable checkpoint. A :class:`TimerService` polls a compiled graph and resumes
the timers that have come due; :func:`deliver_event` wakes a graph waiting on a
specific event. Both are deterministic under an injected clock.

The helpers are called from inside a node, exactly like
:func:`vincio.agents.graph.interrupt`::

    def wait_node(state):
        sleep_for(state, 3600)          # pause ~1h, durably
        return {"woke": True}

    graph.add_node("wait", wait_node)
    # elsewhere / after a restart:
    TimerService(compiled).tick()       # resumes due timers
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from ..core.utils import utcnow
from ..providers.base import run_sync
from .graph import _RESUME_KEY, CompiledGraph, GraphInterrupt, GraphResult

__all__ = [
    "DurableTimer",
    "PendingTimer",
    "TimerService",
    "sleep_until",
    "sleep_for",
    "wait_for_event",
    "pending_timers",
    "due_timers",
    "resume_due_timers",
    "aresume_due_timers",
    "deliver_event",
    "adeliver_event",
]

# Discriminator key marking an interrupt payload as a durable timer (vs a plain
# human-gate interrupt), and the resume markers the service / event delivery use.
_TIMER_MARKER = "__vincio_timer__"
_TIMER_FIRED = "__vincio_timer_fired__"
_EVENT_DELIVERY = "__vincio_event__"


class DurableTimer(BaseModel):
    """The wake condition a timer node persists into its checkpoint."""

    kind: str  # "sleep_until" | "wait_for_event"
    wake_at: str | None = None  # ISO-8601, for sleep timers
    event_name: str | None = None  # for event waits
    created_at: str = Field(default_factory=lambda: utcnow().isoformat())


class PendingTimer(BaseModel):
    """A paused timer found on a graph thread."""

    thread_id: str
    timer: DurableTimer
    step: int = 0


# -- node-side helpers ---------------------------------------------------------


def sleep_until(state: dict[str, Any], when: datetime | str, *, clock: Any = utcnow) -> None:
    """Pause the graph until ``when`` (a datetime or ISO string), durably.

    On the first encounter this raises a timer interrupt the checkpoint records;
    when the :class:`TimerService` resumes the thread after the wake time, the
    node re-runs and this call returns so the node proceeds. ``clock`` is unused
    here (the service decides due-ness) but accepted for a uniform signature.
    """
    if state.pop(_RESUME_KEY, None) == _TIMER_FIRED:
        return
    wake = when.isoformat() if isinstance(when, datetime) else str(when)
    raise GraphInterrupt(
        {_TIMER_MARKER: DurableTimer(kind="sleep_until", wake_at=wake).model_dump(mode="json")}
    )


def sleep_for(state: dict[str, Any], seconds: float, *, clock: Any = utcnow) -> None:
    """Pause the graph for ``seconds`` of wall-clock time, durably.

    The absolute wake time is computed once from ``clock`` and persisted, so the
    delay is honored across a restart rather than restarting on resume."""
    if state.pop(_RESUME_KEY, None) == _TIMER_FIRED:
        return
    wake = (clock() + timedelta(seconds=seconds)).isoformat()
    raise GraphInterrupt(
        {_TIMER_MARKER: DurableTimer(kind="sleep_until", wake_at=wake).model_dump(mode="json")}
    )


def wait_for_event(state: dict[str, Any], name: str) -> Any:
    """Pause the graph until an event named ``name`` is delivered; return its payload.

    Resumes only when :func:`deliver_event` delivers a matching event; a
    different event re-pauses the node, so a thread waits for exactly its event.
    """
    resumed = state.get(_RESUME_KEY)
    if isinstance(resumed, dict) and resumed.get(_EVENT_DELIVERY) == name:
        state.pop(_RESUME_KEY, None)
        return resumed.get("payload")
    raise GraphInterrupt(
        {_TIMER_MARKER: DurableTimer(kind="wait_for_event", event_name=name).model_dump(mode="json")}
    )


# -- service-side scanning / resuming ------------------------------------------


def pending_timers(graph: CompiledGraph) -> list[PendingTimer]:
    """Every paused timer across the graph's threads (sleep + event waits)."""
    out: list[PendingTimer] = []
    for ckpt in graph.checkpointer.latest_per_thread():
        if ckpt.status != "interrupted":
            continue
        payload = ckpt.interrupt_payload
        if isinstance(payload, dict) and _TIMER_MARKER in payload:
            out.append(
                PendingTimer(
                    thread_id=ckpt.thread_id,
                    timer=DurableTimer.model_validate(payload[_TIMER_MARKER]),
                    step=ckpt.step,
                )
            )
    return out


def due_timers(graph: CompiledGraph, *, now: datetime | None = None) -> list[PendingTimer]:
    """Paused sleep timers whose wake time is at or before ``now``."""
    moment = now or utcnow()
    due: list[PendingTimer] = []
    for pending in pending_timers(graph):
        timer = pending.timer
        if timer.kind == "sleep_until" and timer.wake_at:
            try:
                wake = datetime.fromisoformat(timer.wake_at)
            except ValueError:
                continue
            if moment >= wake:
                due.append(pending)
    return due


async def aresume_due_timers(
    graph: CompiledGraph, *, now: datetime | None = None
) -> list[GraphResult]:
    """Resume every sleep timer that has come due; returns their results."""
    results: list[GraphResult] = []
    for pending in due_timers(graph, now=now):
        results.append(await graph.aresume(pending.thread_id, value=_TIMER_FIRED))
    return results


def resume_due_timers(graph: CompiledGraph, *, now: datetime | None = None) -> list[GraphResult]:
    return run_sync(aresume_due_timers(graph, now=now))


async def adeliver_event(
    graph: CompiledGraph, thread_id: str, event_name: str, payload: Any = None
) -> GraphResult | None:
    """Deliver ``event_name`` to a thread waiting on it; ``None`` if it is not.

    The wait is verified against the thread's head checkpoint, so an event is
    only consumed by a thread actually blocked on it — never misrouted."""
    latest = graph.checkpointer.latest(thread_id)
    if latest is None or latest.status != "interrupted":
        return None
    marker = latest.interrupt_payload
    if not (isinstance(marker, dict) and _TIMER_MARKER in marker):
        return None
    timer = DurableTimer.model_validate(marker[_TIMER_MARKER])
    if timer.kind != "wait_for_event" or timer.event_name != event_name:
        return None
    return await graph.aresume(
        thread_id, value={_EVENT_DELIVERY: event_name, "payload": payload}
    )


def deliver_event(
    graph: CompiledGraph, thread_id: str, event_name: str, payload: Any = None
) -> GraphResult | None:
    return run_sync(adeliver_event(graph, thread_id, event_name, payload))


class TimerService:
    """Resumes due timers and delivers events for one compiled graph.

    A thin, restart-safe poller: construct it around a compiled graph (often in a
    fresh process), call :meth:`tick` on a schedule to wake due sleep timers, and
    :meth:`deliver` to wake an event wait. ``clock`` is injectable for
    deterministic tests.
    """

    def __init__(self, graph: CompiledGraph, *, clock: Any = utcnow) -> None:
        self.graph = graph
        self._clock = clock

    def pending(self) -> list[PendingTimer]:
        return pending_timers(self.graph)

    def due(self, *, now: datetime | None = None) -> list[PendingTimer]:
        return due_timers(self.graph, now=now or self._clock())

    def tick(self, *, now: datetime | None = None) -> list[GraphResult]:
        return resume_due_timers(self.graph, now=now or self._clock())

    async def atick(self, *, now: datetime | None = None) -> list[GraphResult]:
        return await aresume_due_timers(self.graph, now=now or self._clock())

    def deliver(self, thread_id: str, event_name: str, payload: Any = None) -> GraphResult | None:
        return deliver_event(self.graph, thread_id, event_name, payload)

    async def adeliver(
        self, thread_id: str, event_name: str, payload: Any = None
    ) -> GraphResult | None:
        return await adeliver_event(self.graph, thread_id, event_name, payload)
