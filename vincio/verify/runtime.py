"""Runtime verification & shielding.

The output-side certificate proves a *result*; this is its behavioural, online
analogue — a property over an agent's plan or tool trajectory, checked
step-by-step as it runs, with a shield that blocks or repairs a violating action
*before* it executes.

* :class:`BehaviorSpec` states a property as plain data — events that must never
  occur (:attr:`forbid`), an ordering one event must precede another
  (:attr:`require_before`), and an invariant that must hold of every event
  (:attr:`invariants`). It is the per-step, behavioural analogue of a governance
  invariant: *never call a write tool before approval*, *always retrieve before
  claiming*, *stay within residency*.
* :class:`RuntimeMonitor` observes a stream of :class:`BehaviorEvent`\\ s and
  reports a :class:`MonitorVerdict` incrementally, maintaining only the history it
  needs to decide an ordering property.
* :class:`Shield` wraps a monitor and **prevents** a violation: in ``block`` mode
  a violating action is refused; in ``repair`` mode a supplied repair maps it to a
  safe alternative that is re-checked. The shield drops into the tool runtime, so
  an unsafe tool call is structurally blocked, not merely logged.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr

__all__ = [
    "BehaviorEvent",
    "EventPattern",
    "BehaviorSpec",
    "Violation",
    "MonitorVerdict",
    "RuntimeMonitor",
    "ShieldMode",
    "ShieldDecision",
    "Shield",
]


class BehaviorEvent(BaseModel):
    """One observable step in an agent's trajectory.

    ``kind`` classifies the step (``tool_call``, ``claim``, ``retrieval``,
    ``approval``, ``action`` or a custom string), ``name`` identifies the concrete
    target (a tool name, an action verb), and ``attributes`` carries the typed
    facts a property reads — ``side_effects`` and ``approved`` for a tool call, a
    ``region`` for residency, the cited source ids for a claim.
    """

    kind: str
    name: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)


class EventPattern(BaseModel):
    """A predicate that matches a :class:`BehaviorEvent`.

    Matches when every supplied facet agrees: ``kind`` (exact), ``name`` (exact)
    or ``name_regex`` (regex), and ``where`` (each key/value must equal the event's
    attribute). An empty pattern matches every event.
    """

    kind: str | None = None
    name: str | None = None
    name_regex: str | None = None
    where: dict[str, Any] = Field(default_factory=dict)

    def matches(self, event: BehaviorEvent) -> bool:
        """True when ``event`` satisfies every facet of this pattern."""
        if self.kind is not None and event.kind != self.kind:
            return False
        if self.name is not None and event.name != self.name:
            return False
        if self.name_regex is not None and not re.search(self.name_regex, event.name):
            return False
        for key, value in self.where.items():
            if event.attributes.get(key) != value:
                return False
        return True


class _Ordering(BaseModel):
    """``later`` may not occur unless a matching ``earlier`` occurred before it."""

    earlier: EventPattern
    later: EventPattern
    description: str = ""


class BehaviorSpec(BaseModel):
    """A temporal-logic-lite property over an event trajectory, as plain data.

    Three clause families compose one spec:

    * :attr:`forbid` — patterns of events that must **never** occur (a *safety*
      property: ``never run a write tool without approval``).
    * :attr:`require_before` — orderings ``(earlier, later)``: a ``later`` event is
      a violation unless a matching ``earlier`` event preceded it (a *precedence*
      property: ``retrieve before you claim``).
    * :attr:`invariants` — named predicates ``(event) -> bool | str`` that must hold
      of every event (a *state* property: ``stay within residency``); a string
      return becomes the violation message.

    Predicates are registered by name via :meth:`invariant` so a spec stays plain
    data while still expressing a custom check.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str
    forbid: list[EventPattern] = Field(default_factory=list)
    require_before: list[_Ordering] = Field(default_factory=list)
    _predicates: dict[str, Callable[[BehaviorEvent], Any]] = PrivateAttr(default_factory=dict)

    @property
    def invariants(self) -> dict[str, Callable[[BehaviorEvent], Any]]:
        """The registered invariant predicates, keyed by name."""
        return self._predicates

    def invariant(self, name: str, predicate: Callable[[BehaviorEvent], Any]) -> BehaviorSpec:
        """Register an invariant predicate that must hold of every event."""
        self._predicates[name] = predicate
        return self

    def precede(
        self, earlier: EventPattern, later: EventPattern, *, description: str = ""
    ) -> BehaviorSpec:
        """Add a precedence ordering: ``earlier`` must occur before ``later``."""
        self.require_before.append(
            _Ordering(earlier=earlier, later=later, description=description)
        )
        return self


class Violation(BaseModel):
    """A single property breach pinned to the event that caused it."""

    spec: str
    rule: str  # "forbid" | "require_before" | "invariant"
    message: str
    event_index: int
    event: BehaviorEvent


class MonitorVerdict(BaseModel):
    """The outcome of checking one event (or a whole trajectory).

    ``ok`` is true when no property is breached. ``violations`` are the breaches
    newly raised by this step (for :meth:`RuntimeMonitor.observe`) or every breach
    over the trajectory (for :meth:`RuntimeMonitor.check_trajectory`).
    """

    ok: bool = True
    violations: list[Violation] = Field(default_factory=list)
    event_index: int = -1


class RuntimeMonitor:
    """Checks a :class:`BehaviorSpec` set against a trajectory, step-by-step.

    :meth:`observe` is incremental — it appends the event to the history and
    returns the violations *this* step raises — so a long-running agent is checked
    online without re-scanning. :meth:`check_trajectory` runs a fresh monitor over
    a recorded event list and returns the cumulative verdict.
    """

    def __init__(self, specs: BehaviorSpec | list[BehaviorSpec]) -> None:
        self.specs: list[BehaviorSpec] = [specs] if isinstance(specs, BehaviorSpec) else list(specs)
        self.history: list[BehaviorEvent] = []
        self.violations: list[Violation] = []

    def reset(self) -> None:
        """Clear the observed history and accumulated violations."""
        self.history = []
        self.violations = []

    def observe(self, event: BehaviorEvent) -> MonitorVerdict:
        """Append ``event`` and return the violations it newly raises."""
        index = len(self.history)
        self.history.append(event)
        new: list[Violation] = []
        for spec in self.specs:
            new.extend(self._check_event(spec, event, index))
        self.violations.extend(new)
        return MonitorVerdict(ok=not new, violations=new, event_index=index)

    def check_trajectory(self, events: list[BehaviorEvent]) -> MonitorVerdict:
        """Run a fresh monitor over ``events`` and return the cumulative verdict."""
        self.reset()
        for event in events:
            self.observe(event)
        return MonitorVerdict(
            ok=not self.violations,
            violations=list(self.violations),
            event_index=len(events) - 1,
        )

    def _check_event(
        self, spec: BehaviorSpec, event: BehaviorEvent, index: int
    ) -> list[Violation]:
        out: list[Violation] = []
        for pattern in spec.forbid:
            if pattern.matches(event):
                out.append(Violation(
                    spec=spec.name, rule="forbid",
                    message=f"forbidden event {event.kind}:{event.name}",
                    event_index=index, event=event,
                ))
        for ordering in spec.require_before:
            if ordering.later.matches(event):
                seen = any(
                    ordering.earlier.matches(prior) for prior in self.history[:index]
                )
                if not seen:
                    out.append(Violation(
                        spec=spec.name, rule="require_before",
                        message=ordering.description
                        or f"{event.kind}:{event.name} occurred before a required precondition",
                        event_index=index, event=event,
                    ))
        for inv_name, predicate in spec.invariants.items():
            result = predicate(event)
            if result is not True and result:
                message = result if isinstance(result, str) else f"invariant {inv_name!r} violated"
                out.append(Violation(
                    spec=spec.name, rule="invariant", message=message,
                    event_index=index, event=event,
                ))
            elif result is False:
                out.append(Violation(
                    spec=spec.name, rule="invariant",
                    message=f"invariant {inv_name!r} violated",
                    event_index=index, event=event,
                ))
        return out


ShieldMode = Literal["block", "repair", "monitor"]


class ShieldDecision(BaseModel):
    """A shield's ruling on a proposed event.

    ``allowed`` says whether the action may execute. In ``repair`` mode a blocked
    event may carry a ``repaired`` safe alternative (already re-checked clean);
    ``violations`` and ``reason`` explain a refusal.
    """

    allowed: bool = True
    repaired: BehaviorEvent | None = None
    violations: list[Violation] = Field(default_factory=list)
    reason: str = ""


class Shield:
    """Prevents a behavioural violation before the action executes.

    A shield wraps a :class:`RuntimeMonitor` and a ``mode``: ``block`` refuses a
    violating action, ``repair`` maps it through a supplied ``repair`` callback to
    a safe alternative that is re-checked (the action plane's analogue of saga
    compensation, applied *ahead* of the effect), and ``monitor`` records but does
    not stop. :meth:`guard` is the general entry; :meth:`guard_tool_call` adapts a
    tool call into an event so the shield drops into the tool runtime.

    Crucially, guarding is **non-committing**: a blocked event is rolled back out
    of the monitor's history, so refusing an action does not poison the precedence
    state for the actions that follow.
    """

    def __init__(
        self,
        specs: BehaviorSpec | list[BehaviorSpec] | RuntimeMonitor,
        *,
        mode: ShieldMode = "block",
        repair: Callable[[BehaviorEvent, list[Violation]], BehaviorEvent | None] | None = None,
    ) -> None:
        self.monitor = specs if isinstance(specs, RuntimeMonitor) else RuntimeMonitor(specs)
        self.mode = mode
        self._repair = repair
        self.blocked: list[Violation] = []

    def guard(self, event: BehaviorEvent) -> ShieldDecision:
        """Rule on ``event`` before it executes, committing only an allowed event."""
        verdict = self.monitor.observe(event)
        if verdict.ok or self.mode == "monitor":
            return ShieldDecision(allowed=True, violations=verdict.violations)
        # Roll the rejected event back out of history so it does not affect later
        # precedence checks, then attempt a repair or block.
        self._rollback()
        if self.mode == "repair" and self._repair is not None:
            replacement = self._repair(event, verdict.violations)
            if replacement is not None:
                repaired_verdict = self.monitor.observe(replacement)
                if repaired_verdict.ok:
                    return ShieldDecision(
                        allowed=True, repaired=replacement, violations=verdict.violations,
                        reason="repaired to a safe alternative",
                    )
                self._rollback()
        self.blocked.extend(verdict.violations)
        return ShieldDecision(
            allowed=False,
            violations=verdict.violations,
            reason="; ".join(v.message for v in verdict.violations),
        )

    def guard_tool_call(
        self, tool_name: str, *, side_effects: str = "read", approved: bool = False,
        arguments: dict[str, Any] | None = None,
    ) -> ShieldDecision:
        """Adapt a tool call into a :class:`BehaviorEvent` and guard it."""
        event = BehaviorEvent(
            kind="tool_call",
            name=tool_name,
            attributes={
                **(arguments or {}),
                # The security facets are authoritative — never shadowed by an
                # argument that happens to share a name.
                "side_effects": side_effects,
                "approved": approved,
            },
        )
        return self.guard(event)

    def _rollback(self) -> None:
        if self.monitor.history:
            removed = len(self.monitor.history) - 1
            self.monitor.history.pop()
            self.monitor.violations = [
                v for v in self.monitor.violations if v.event_index != removed
            ]
