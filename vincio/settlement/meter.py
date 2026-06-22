"""The metering primitive: usage accrued against a negotiated contract.

A :class:`Meter` accumulates the usage of work delivered under a
:class:`~vincio.negotiation.Contract` — each unit of delivery a :class:`UsageEvent`
attributed to the contract and the run — and rolls it up into a deterministic
:class:`MeterReading` the settlement step reconciles against the agreed terms. It
is the cross-org analogue of the cost report: where the cost report attributes an
app's own model spend to a run, the meter attributes *delivered* cost / latency /
quality to a contract, so a saga's steps accrue against the agreed price as they
complete.

Metering is pure accumulation — deterministic, dependency-free, side-effect-free.
The reading's totals are exactly the sum of the accrued events (no double-count,
no drop), which is what makes a settlement built from a reading a faithful,
auditable reconciliation rather than an estimate. Enforcement (a hard cap on the
agreed price / SLA) stays where it already lives —
:meth:`~vincio.negotiation.Contract.to_budget` and
:meth:`~vincio.negotiation.Contract.check`; the meter only records what was
delivered so the books can be closed on it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import SettlementError
from ..core.utils import to_jsonable, utcnow

__all__ = [
    "UsageEvent",
    "MeterReading",
    "Meter",
]


def _round(value: float | None) -> float | None:
    """Round a delivered metric so float drift never breaks reconciliation."""
    return None if value is None else round(float(value), 9)


class UsageEvent(BaseModel):
    """One unit of delivered usage accrued against a contract.

    ``units`` is the quantity delivered (calls, tokens, items — whatever the scope
    meters); ``cost_usd`` / ``latency_ms`` / ``quality`` are the delivered metrics
    the settlement reconciles against the contract's price / SLA / quality floor.
    ``step`` names the saga step (or sub-task) the usage came from, so a reading
    can be attributed step by step. The event is the metering analogue of a cost
    event: a typed, attributed record of what was delivered, never an estimate.
    """

    contract_id: str
    run_id: str | None = None
    step: str | None = None
    kind: str = "delivery"
    units: float = 1.0
    cost_usd: float | None = None
    latency_ms: float | None = None
    quality: float | None = None
    at: datetime = Field(default_factory=utcnow)
    details: dict[str, Any] = Field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> UsageEvent:
        return cls.model_validate(data)


class MeterReading(BaseModel):
    """The deterministic roll-up of a meter's accrued usage for one contract.

    Aggregation rules are fixed and total-preserving so a reading is a faithful
    sum of its events, not an estimate: ``cost_usd`` and ``latency_ms`` are
    **summed** (cumulative delivered cost and end-to-end latency), ``units`` is
    summed, and ``quality`` is the **minimum** observed (the weakest link — the
    safe figure to hold against a quality *floor*). ``per_step`` attributes the
    delivered cost step by step. ``events`` is the count, so the metering-accuracy
    invariant (reading totals equal the sum of the events) is mechanically checked.
    """

    contract_id: str
    run_id: str | None = None
    events: int = 0
    units: float = 0.0
    cost_usd: float | None = None
    latency_ms: float | None = None
    quality: float | None = None
    per_step: dict[str, float] = Field(default_factory=dict)

    @property
    def metered(self) -> bool:
        """Whether any usage was accrued (an empty reading settles to zero owed)."""
        return self.events > 0

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> MeterReading:
        return cls.model_validate(data)


class Meter:
    """Accumulates the usage of work delivered under one contract.

    Bind a meter to a contract id (and optionally the run), :meth:`accrue` a
    :class:`UsageEvent` as each unit of work completes, and read the deterministic
    :meth:`reading`. A meter is a pure accumulator — it records what was delivered
    so the settlement step can reconcile it against the agreed terms; it does not
    enforce a cap (that is the contract's budget) and has no side effects.

    The convenience :meth:`from_saga` builds a meter for every contract a saga
    dispatched under, accruing one event per contracted step from the durable
    journal's recorded outcomes, so a whole cross-org saga meters in one call.
    """

    def __init__(self, contract_id: str, *, run_id: str | None = None) -> None:
        if not contract_id:
            raise SettlementError("a meter must be bound to a contract id")
        self.contract_id = contract_id
        self.run_id = run_id
        self._events: list[UsageEvent] = []

    @property
    def events(self) -> list[UsageEvent]:
        """The accrued usage events, in accrual order."""
        return list(self._events)

    def accrue(
        self,
        *,
        units: float = 1.0,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
        quality: float | None = None,
        step: str | None = None,
        kind: str = "delivery",
        details: dict[str, Any] | None = None,
    ) -> UsageEvent:
        """Record one unit of delivered usage and return the event.

        ``units`` must be non-negative (a meter records delivery, never a refund —
        a reversal belongs in compensation, not metering). The delivered cost /
        latency / quality are reconciled against the contract at settlement time.
        """
        if units < 0:
            raise SettlementError(
                f"metered units must be non-negative; got {units}",
                details={"contract_id": self.contract_id, "step": step},
            )
        event = UsageEvent(
            contract_id=self.contract_id,
            run_id=self.run_id,
            step=step,
            kind=kind,
            units=float(units),
            cost_usd=_round(cost_usd),
            latency_ms=_round(latency_ms),
            quality=_round(quality),
            details=dict(details or {}),
        )
        self._events.append(event)
        return event

    def accrue_event(self, event: UsageEvent) -> UsageEvent:
        """Accrue a pre-built :class:`UsageEvent` (e.g. received over the wire)."""
        if event.contract_id != self.contract_id:
            raise SettlementError(
                f"usage event for contract {event.contract_id!r} cannot accrue on a "
                f"meter bound to {self.contract_id!r}",
                details={"meter": self.contract_id, "event": event.contract_id},
            )
        self._events.append(event)
        return event

    def reading(self) -> MeterReading:
        """The deterministic roll-up of every accrued event — total-preserving."""
        cost = [e.cost_usd for e in self._events if e.cost_usd is not None]
        latency = [e.latency_ms for e in self._events if e.latency_ms is not None]
        quality = [e.quality for e in self._events if e.quality is not None]
        per_step: dict[str, float] = {}
        for event in self._events:
            if event.step is not None and event.cost_usd is not None:
                per_step[event.step] = round(
                    per_step.get(event.step, 0.0) + event.cost_usd, 9
                )
        return MeterReading(
            contract_id=self.contract_id,
            run_id=self.run_id,
            events=len(self._events),
            units=round(sum(e.units for e in self._events), 9),
            cost_usd=round(sum(cost), 9) if cost else None,
            latency_ms=round(sum(latency), 9) if latency else None,
            quality=min(quality) if quality else None,
            per_step=per_step,
        )

    @classmethod
    def from_saga(cls, result: Any, *, run_id: str | None = None) -> dict[str, Meter]:
        """Build a meter per contract from a saga's durable journal.

        Reads every completed forward step that ran under a contract from the
        :class:`~vincio.choreography.SagaResult` (or its
        :class:`~vincio.choreography.SagaJournal`) and accrues one
        :class:`UsageEvent` per step on the meter for that contract, keyed by
        ``contract_id``. So a whole cross-org saga's delivery meters in one call,
        ready for :func:`~vincio.settlement.book.settle_saga` to reconcile against
        each contract.
        """
        journal = getattr(result, "journal", result)
        meters: dict[str, Meter] = {}
        for record in journal.completed_forward():
            contract_id = record.contract_id
            if not contract_id:
                continue
            meter = meters.get(contract_id)
            if meter is None:
                meter = cls(contract_id, run_id=run_id or journal.id)
                meters[contract_id] = meter
            meter.accrue(
                units=1.0,
                cost_usd=record.cost_usd,
                latency_ms=record.latency_ms,
                quality=record.quality,
                step=record.step,
                kind="saga_step",
            )
        return meters
