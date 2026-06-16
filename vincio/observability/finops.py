"""Cost attribution and budget SLOs (1.3).

Every model call records a :class:`CostEvent` carrying the request-time
attribution dimensions — ``tenant`` / ``user`` / ``feature`` / ``run`` — so cost
is counted *honestly* against whoever incurred it, captured at request creation
rather than retrofitted from logs (which loses long agentic traces). The
:class:`CostLedger` rolls events up by any dimension (``vincio cost report --by
tenant|feature``).

:class:`BudgetManager` enforces per-tenant/feature **budgets**. When a scope's
spend over its period reaches the limit, a :class:`BudgetDecision` says how to
react — **hard cap** (deny), **degrade** to a cheaper model, or **queue to
batch** — and the runtime applies it as a :class:`PolicyViolation` on the same
audit path as every other policy decision. A spend spike raises a
``cost.anomaly`` event on the bus.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.types import TokenUsage
from ..core.utils import new_id, utcnow
from .costs import PriceTable, default_price_table

__all__ = [
    "CostEvent",
    "CostRow",
    "CostReport",
    "CostLedger",
    "CostBudget",
    "BudgetDecision",
    "BudgetManager",
]

Dimension = Literal["tenant", "feature", "user", "model", "provider", "run"]
Period = Literal["run", "hour", "day", "month", "total"]


class CostEvent(BaseModel):
    """One attributed unit of model spend."""

    id: str = Field(default_factory=lambda: new_id("cost"))
    model: str = ""
    provider: str = ""
    tenant_id: str | None = None
    user_id: str | None = None
    feature: str | None = None
    run_id: str | None = None
    trace_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0
    batch: bool = False
    created_at: datetime = Field(default_factory=utcnow)

    def dimension_value(self, dimension: Dimension) -> str:
        return {
            "tenant": self.tenant_id,
            "feature": self.feature,
            "user": self.user_id,
            "model": self.model,
            "provider": self.provider,
            "run": self.run_id,
        }.get(dimension) or "∅"


class CostRow(BaseModel):
    key: str
    cost_usd: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


class CostReport(BaseModel):
    dimension: str
    rows: list[CostRow] = Field(default_factory=list)
    total_usd: float = 0.0

    def print_summary(self) -> None:
        print(f"cost by {self.dimension}  (total ${self.total_usd:.6f})")
        width = max((len(r.key) for r in self.rows), default=3)
        for row in self.rows:
            cached = f"  cached={row.cached_input_tokens}" if row.cached_input_tokens else ""
            print(
                f"  {row.key:<{width}}  ${row.cost_usd:.6f}  "
                f"calls={row.calls}  in={row.input_tokens} out={row.output_tokens}{cached}"
            )


def _period_start(period: Period, *, now: datetime) -> datetime | None:
    # Rolling windows (last hour / 24h / 30 days), not calendar boundaries — a
    # budget protects continuously rather than resetting at midnight or the 1st.
    if period == "hour":
        return now - timedelta(hours=1)
    if period == "day":
        return now - timedelta(days=1)
    if period == "month":
        return now - timedelta(days=30)
    return None  # "run" and "total" have no rolling window


class CostLedger:
    """In-process append-only ledger of attributed cost events.

    Events are kept in memory and, when a metadata ``store`` is given, persisted
    to a ``cost_events`` table so ``vincio cost report`` works across processes.
    """

    def __init__(
        self,
        *,
        price_table: PriceTable | None = None,
        store: Any | None = None,
        max_events: int = 100_000,
    ) -> None:
        self.price_table = price_table or default_price_table()
        self.store = store
        self.max_events = max_events
        self.events: list[CostEvent] = []

    def record(self, event: CostEvent) -> CostEvent:
        self.events.append(event)
        if len(self.events) > self.max_events:
            del self.events[: len(self.events) - self.max_events]
        if self.store is not None:
            try:  # persistence must never break a run
                self.store.save("cost_events", event.model_dump(mode="json"))
            except Exception:  # noqa: BLE001
                pass
        return event

    def record_model_call(
        self,
        *,
        model: str,
        usage: TokenUsage,
        cost_usd: float,
        provider: str = "",
        tenant_id: str | None = None,
        user_id: str | None = None,
        feature: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
        batch: bool = False,
    ) -> CostEvent:
        return self.record(
            CostEvent(
                model=model,
                provider=provider,
                tenant_id=tenant_id,
                user_id=user_id,
                feature=feature,
                run_id=run_id,
                trace_id=trace_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                cost_usd=cost_usd,
                batch=batch,
            )
        )

    def _filter(
        self,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        feature: str | None = None,
        since: datetime | None = None,
    ) -> list[CostEvent]:
        out = []
        for event in self.events:
            if tenant_id is not None and event.tenant_id != tenant_id:
                continue
            if user_id is not None and event.user_id != user_id:
                continue
            if feature is not None and event.feature != feature:
                continue
            if since is not None and event.created_at < since:
                continue
            out.append(event)
        return out

    def total(self, **filters: Any) -> float:
        return round(sum(e.cost_usd for e in self._filter(**filters)), 8)

    def report(self, dimension: Dimension = "tenant", *, since: datetime | None = None) -> CostReport:
        rows: dict[str, CostRow] = {}
        for event in self._filter(since=since):
            key = event.dimension_value(dimension)
            row = rows.setdefault(key, CostRow(key=key))
            row.cost_usd += event.cost_usd
            row.calls += 1
            row.input_tokens += event.input_tokens
            row.output_tokens += event.output_tokens
            row.cached_input_tokens += event.cached_input_tokens
        ordered = sorted(rows.values(), key=lambda r: r.cost_usd, reverse=True)
        for row in ordered:
            row.cost_usd = round(row.cost_usd, 8)
        return CostReport(
            dimension=dimension,
            rows=ordered,
            total_usd=round(sum(r.cost_usd for r in ordered), 8),
        )

    @classmethod
    def from_store(
        cls, store: Any, *, price_table: PriceTable | None = None, limit: int = 1_000_000
    ) -> CostLedger:
        ledger = cls(price_table=price_table)
        for row in store.query("cost_events", limit=limit):
            try:
                ledger.events.append(CostEvent.model_validate(row))
            except Exception:  # noqa: BLE001 - tolerate legacy rows
                continue
        return ledger


class CostBudget(BaseModel):
    """A spend limit on a scope, with an enforcement action on breach."""

    scope: Literal["tenant", "feature", "user", "global"] = "tenant"
    id: str | None = None  # the tenant/feature/user id; None applies to all of that scope
    limit_usd: float
    period: Period = "day"  # rolling window: run | hour | day (24h) | month (30d) | total
    on_breach: Literal["cap", "degrade", "queue_to_batch"] = "cap"
    degrade_model: str | None = None  # target for on_breach="degrade"
    anomaly_factor: float | None = None  # raise cost.anomaly above factor × mean

    def matches(
        self, *, tenant_id: str | None, user_id: str | None, feature: str | None
    ) -> bool:
        value = {"tenant": tenant_id, "feature": feature, "user": user_id, "global": None}[
            self.scope
        ]
        if self.scope == "global":
            return True
        if value is None:
            return False
        return self.id is None or self.id == value


class BudgetDecision(BaseModel):
    action: Literal["allow", "cap", "degrade", "queue_to_batch"] = "allow"
    spent_usd: float = 0.0
    limit_usd: float = 0.0
    model_override: str | None = None
    scope: str = ""
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.action in ("allow", "degrade")


class BudgetManager:
    """Enforces :class:`CostBudget`\\ s and detects spend anomalies."""

    def __init__(self, ledger: CostLedger, *, events: Any | None = None) -> None:
        self.ledger = ledger
        self.events = events
        self.budgets: list[CostBudget] = []
        self._anomaly_state: dict[str, tuple[int, float]] = {}  # key -> (count, mean)

    def add(self, budget: CostBudget) -> CostBudget:
        # Most specific scopes (user, feature, tenant) checked before global.
        self.budgets.append(budget)
        order = {"user": 0, "feature": 1, "tenant": 2, "global": 3}
        self.budgets.sort(key=lambda b: order.get(b.scope, 9))
        return budget

    def _scope_spend(self, budget: CostBudget, *, tenant_id, user_id, feature, now) -> float:
        since = _period_start(budget.period, now=now)
        if budget.period == "run":
            return 0.0  # per-run budgets only bound the projected cost
        kw: dict[str, Any] = {"since": since}
        if budget.scope == "tenant":
            kw["tenant_id"] = budget.id or tenant_id
        elif budget.scope == "feature":
            kw["feature"] = budget.id or feature
        elif budget.scope == "user":
            kw["user_id"] = budget.id or user_id
        return self.ledger.total(**kw)

    def check(
        self,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        feature: str | None = None,
        projected_usd: float = 0.0,
        now: datetime | None = None,
    ) -> BudgetDecision:
        """Decide whether a run may proceed under the active budgets."""
        now = now or utcnow()
        for budget in self.budgets:
            if not budget.matches(tenant_id=tenant_id, user_id=user_id, feature=feature):
                continue
            spent = self._scope_spend(
                budget, tenant_id=tenant_id, user_id=user_id, feature=feature, now=now
            )
            if spent + projected_usd < budget.limit_usd:
                continue
            scope_id = budget.id or {"tenant": tenant_id, "feature": feature, "user": user_id}.get(
                budget.scope
            )
            reason = (
                f"{budget.scope} {scope_id!r} spend ${spent:.4f} reached budget "
                f"${budget.limit_usd:.4f} ({budget.period})"
            )
            return BudgetDecision(
                action=budget.on_breach,
                spent_usd=round(spent, 8),
                limit_usd=budget.limit_usd,
                model_override=budget.degrade_model if budget.on_breach == "degrade" else None,
                scope=f"{budget.scope}:{scope_id}",
                reason=reason,
            )
        return BudgetDecision(action="allow")

    def observe(self, event: CostEvent) -> None:
        """Update anomaly baselines for an event; raise ``cost.anomaly`` on a
        spike against the matching budgets' ``anomaly_factor``."""
        factors = [b for b in self.budgets if b.anomaly_factor]
        if not factors:
            return
        for budget in factors:
            if not budget.matches(
                tenant_id=event.tenant_id, user_id=event.user_id, feature=event.feature
            ):
                continue
            key = f"{budget.scope}:{budget.id or event.dimension_value(budget.scope)}"  # type: ignore[arg-type]
            count, mean = self._anomaly_state.get(key, (0, 0.0))
            if (
                count >= 5
                and mean > 0
                and budget.anomaly_factor
                and event.cost_usd > budget.anomaly_factor * mean
            ):
                if self.events is not None:
                    self.events.emit(
                        "cost.anomaly",
                        {
                            "scope": key,
                            "cost_usd": round(event.cost_usd, 8),
                            "mean_usd": round(mean, 8),
                            "factor": budget.anomaly_factor,
                            "run_id": event.run_id,
                        },
                        trace_id=event.trace_id,
                    )
            new_count = count + 1
            self._anomaly_state[key] = (new_count, mean + (event.cost_usd - mean) / new_count)
