"""Cost attribution and budget SLOs.

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

import logging
import math
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..core.types import TokenUsage
from ..core.utils import new_id, utcnow
from .costs import PriceTable, default_price_table
from .exporters import Alert, AlertSink

if TYPE_CHECKING:
    from ..core.events import Event, EventBus
    from .store import IndexedTraceStore

logger = logging.getLogger("vincio.observability")

__all__ = [
    "CostEvent",
    "CostRow",
    "CostReport",
    "CostLedger",
    "CostBudget",
    "BudgetDecision",
    "BudgetManager",
    # energy & carbon accounting
    "EnergyRow",
    "EnergyReport",
    "EnergyBudget",
    "EnergyBudgetDecision",
    # served alerting rule engine
    "AlertRule",
    "AlertManager",
    "within_budget",
]

Dimension = Literal["tenant", "feature", "user", "model", "provider", "run"]
Period = Literal["run", "hour", "day", "month", "total"]


def within_budget(spent: float, projected: float, limit: float | None) -> bool:
    """Whether a run is admitted under a hard spend cap (the shared budget gate).

    The single predicate behind every Vincio budget gate — dollar cost
    (:meth:`BudgetManager.check`), energy, and carbon
    (:meth:`BudgetManager.check_energy`): a run proceeds only while the scope's
    accrued spend *plus* the run's projection stays strictly under the limit.
    ``None`` means no cap. Checking the *projected* total — not just what is already
    spent — is what makes the cap a guarantee: once the scope reaches its limit,
    every further run is refused. The governance verifier proves this predicate
    never admits an overspend across the whole (limit × spent × projected) space.
    """
    if limit is None:
        return True
    return spent + projected < limit


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
    energy_wh: float = 0.0
    co2e_grams: float = 0.0
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
    energy_wh: float = 0.0
    co2e_grams: float = 0.0
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


class EnergyRow(BaseModel):
    """One dimension's line in an :class:`EnergyReport`."""

    key: str
    energy_wh: float = 0.0
    co2e_grams: float = 0.0
    calls: int = 0


class EnergyReport(BaseModel):
    """Estimated energy + carbon rolled up by dimension.

    The energy analogue of :class:`CostReport`, on the same surface: each row is
    a tenant/feature/user/model/provider/run's accrued watt-hours and grams CO₂e.
    """

    dimension: str
    rows: list[EnergyRow] = Field(default_factory=list)
    total_energy_wh: float = 0.0
    total_co2e_grams: float = 0.0

    @property
    def total_co2e_kg(self) -> float:
        """Total carbon in kilograms CO₂e."""
        return round(self.total_co2e_grams / 1000.0, 9)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        print(
            f"energy by {self.dimension}  "
            f"(total {self.total_energy_wh:.3f} Wh, {self.total_co2e_grams:.3f} gCO₂e)"
        )
        width = max((len(r.key) for r in self.rows), default=3)
        for row in self.rows:
            print(
                f"  {row.key:<{width}}  {row.energy_wh:.4f} Wh  "
                f"{row.co2e_grams:.4f} gCO₂e  calls={row.calls}"
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
        energy_wh: float = 0.0,
        co2e_grams: float = 0.0,
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
                energy_wh=energy_wh,
                co2e_grams=co2e_grams,
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

    def total_energy(self, **filters: Any) -> float:
        """Total estimated energy (watt-hours) over the filtered events."""
        return round(sum(e.energy_wh for e in self._filter(**filters)), 6)

    def total_co2e(self, **filters: Any) -> float:
        """Total estimated carbon (grams CO₂e) over the filtered events."""
        return round(sum(e.co2e_grams for e in self._filter(**filters)), 6)

    def report(self, dimension: Dimension = "tenant", *, since: datetime | None = None) -> CostReport:
        rows: dict[str, CostRow] = {}
        for event in self._filter(since=since):
            key = event.dimension_value(dimension)
            row = rows.setdefault(key, CostRow(key=key))
            row.cost_usd += event.cost_usd
            row.energy_wh += event.energy_wh
            row.co2e_grams += event.co2e_grams
            row.calls += 1
            row.input_tokens += event.input_tokens
            row.output_tokens += event.output_tokens
            row.cached_input_tokens += event.cached_input_tokens
        ordered = sorted(rows.values(), key=lambda r: r.cost_usd, reverse=True)
        for row in ordered:
            row.cost_usd = round(row.cost_usd, 8)
            row.energy_wh = round(row.energy_wh, 6)
            row.co2e_grams = round(row.co2e_grams, 6)
        return CostReport(
            dimension=dimension,
            rows=ordered,
            total_usd=round(sum(r.cost_usd for r in ordered), 8),
        )

    def energy_report(
        self, dimension: Dimension = "tenant", *, since: datetime | None = None
    ) -> EnergyReport:
        """Roll up estimated energy + carbon by dimension — the energy analogue
        of :meth:`report`, ordered by energy descending."""
        rows: dict[str, EnergyRow] = {}
        for event in self._filter(since=since):
            key = event.dimension_value(dimension)
            row = rows.setdefault(key, EnergyRow(key=key))
            row.energy_wh += event.energy_wh
            row.co2e_grams += event.co2e_grams
            row.calls += 1
        ordered = sorted(rows.values(), key=lambda r: r.energy_wh, reverse=True)
        for row in ordered:
            row.energy_wh = round(row.energy_wh, 6)
            row.co2e_grams = round(row.co2e_grams, 6)
        return EnergyReport(
            dimension=dimension,
            rows=ordered,
            total_energy_wh=round(sum(r.energy_wh for r in ordered), 6),
            total_co2e_grams=round(sum(r.co2e_grams for r in ordered), 6),
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


class EnergyBudget(BaseModel):
    """An energy/carbon limit on a scope, refused on breach.

    The sustainability analogue of :class:`CostBudget`. Set ``limit_wh`` (energy,
    watt-hours), ``limit_co2e_grams`` (carbon, grams CO₂e), or both — a run is
    refused when the scope's accrued energy or carbon over ``period`` reaches
    either ceiling. ``on_breach`` is ``"cap"`` (refuse); a refused run is surfaced
    on the same audit path as a cost cap.
    """

    scope: Literal["tenant", "feature", "user", "global"] = "global"
    id: str | None = None  # the tenant/feature/user id; None applies to all of that scope
    limit_wh: float | None = None
    limit_co2e_grams: float | None = None
    period: Period = "day"  # rolling window: run | hour | day (24h) | month (30d) | total
    on_breach: Literal["cap"] = "cap"

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


class EnergyBudgetDecision(BaseModel):
    """An explainable verdict on whether a run fits its energy/carbon budget."""

    action: Literal["allow", "cap"] = "allow"
    metric: Literal["energy", "carbon"] = "energy"
    spent_wh: float = 0.0
    limit_wh: float | None = None
    spent_co2e_grams: float = 0.0
    limit_co2e_grams: float | None = None
    scope: str = ""
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.action == "allow"


class BudgetManager:
    """Enforces :class:`CostBudget`/:class:`EnergyBudget`\\ s and detects spend anomalies."""

    def __init__(self, ledger: CostLedger, *, events: Any | None = None) -> None:
        self.ledger = ledger
        self.events = events
        self.budgets: list[CostBudget] = []
        self.energy_budgets: list[EnergyBudget] = []
        self._anomaly_state: dict[str, tuple[int, float]] = {}  # key -> (count, mean)

    def add(self, budget: CostBudget) -> CostBudget:
        # Most specific scopes (user, feature, tenant) checked before global.
        self.budgets.append(budget)
        order = {"user": 0, "feature": 1, "tenant": 2, "global": 3}
        self.budgets.sort(key=lambda b: order.get(b.scope, 9))
        return budget

    def add_energy_budget(self, budget: EnergyBudget) -> EnergyBudget:
        # Most specific scopes (user, feature, tenant) checked before global.
        self.energy_budgets.append(budget)
        order = {"user": 0, "feature": 1, "tenant": 2, "global": 3}
        self.energy_budgets.sort(key=lambda b: order.get(b.scope, 9))
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
            if within_budget(spent, projected_usd, budget.limit_usd):
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

    def _energy_scope_spend(
        self, budget: EnergyBudget, *, tenant_id, user_id, feature, now
    ) -> tuple[float, float]:
        """Accrued (watt-hours, grams CO₂e) for a scope over the budget period."""
        since = _period_start(budget.period, now=now)
        if budget.period == "run":
            return 0.0, 0.0  # per-run energy budgets only bound the projected run
        kw: dict[str, Any] = {"since": since}
        if budget.scope == "tenant":
            kw["tenant_id"] = budget.id or tenant_id
        elif budget.scope == "feature":
            kw["feature"] = budget.id or feature
        elif budget.scope == "user":
            kw["user_id"] = budget.id or user_id
        return self.ledger.total_energy(**kw), self.ledger.total_co2e(**kw)

    def check_energy(
        self,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        feature: str | None = None,
        projected_wh: float = 0.0,
        projected_co2e_grams: float = 0.0,
        now: datetime | None = None,
    ) -> EnergyBudgetDecision:
        """Decide whether a run may proceed under the active energy budgets.

        The energy analogue of :meth:`check`: a run is refused when the scope's
        accrued energy or carbon over the period (plus any projection) reaches the
        matching budget's ``limit_wh`` / ``limit_co2e_grams``."""
        now = now or utcnow()
        for budget in self.energy_budgets:
            if not budget.matches(tenant_id=tenant_id, user_id=user_id, feature=feature):
                continue
            spent_wh, spent_co2e = self._energy_scope_spend(
                budget, tenant_id=tenant_id, user_id=user_id, feature=feature, now=now
            )
            scope_id = budget.id or {
                "tenant": tenant_id, "feature": feature, "user": user_id
            }.get(budget.scope)
            metric: Literal["energy", "carbon"] | None = None
            if budget.limit_wh is not None and not within_budget(
                spent_wh, projected_wh, budget.limit_wh
            ):
                metric = "energy"
            elif budget.limit_co2e_grams is not None and not within_budget(
                spent_co2e, projected_co2e_grams, budget.limit_co2e_grams
            ):
                metric = "carbon"
            if metric is None:
                continue
            if metric == "energy":
                reason = (
                    f"{budget.scope} {scope_id!r} energy {spent_wh:.3f} Wh reached budget "
                    f"{budget.limit_wh:.3f} Wh ({budget.period})"
                )
            else:
                reason = (
                    f"{budget.scope} {scope_id!r} carbon {spent_co2e:.3f} gCO₂e reached budget "
                    f"{budget.limit_co2e_grams:.3f} gCO₂e ({budget.period})"
                )
            return EnergyBudgetDecision(
                action="cap",
                metric=metric,
                spent_wh=round(spent_wh, 6),
                limit_wh=budget.limit_wh,
                spent_co2e_grams=round(spent_co2e, 6),
                limit_co2e_grams=budget.limit_co2e_grams,
                scope=f"{budget.scope}:{scope_id}",
                reason=reason,
            )
        return EnergyBudgetDecision(action="allow")

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


# ---------------------------------------------------------------------------
# Served alerting rule engine
# ---------------------------------------------------------------------------


class _EwmaTracker:
    """Online EWMA mean + variance for anomaly z-scores (Welford-style update)."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self.mean = 0.0
        self.var = 0.0
        self.count = 0

    def zscore(self, value: float) -> float:
        std = math.sqrt(self.var)
        if std > 0:
            return (value - self.mean) / std
        # No observed variance yet: a perfectly flat series that suddenly jumps
        # is maximally anomalous; an unchanged value is not.
        return 0.0 if value == self.mean else math.copysign(float("inf"), value - self.mean)

    def update(self, value: float) -> None:
        self.count += 1
        if self.count == 1:
            self.mean = value
            return
        delta = value - self.mean
        self.mean += self.alpha * delta
        self.var = (1 - self.alpha) * (self.var + self.alpha * delta * delta)


class AlertRule(BaseModel):
    """One alerting rule over a metric stream.

    * ``threshold`` — fire when ``value`` crosses ``threshold`` in ``direction``.
    * ``ewma`` — fire when ``value`` deviates from its EWMA mean by at least
      ``factor`` standard deviations (anomaly detection), after ``min_samples``.
    * ``burn_rate`` — fire when the SRE error-budget burn rate
      (``error_rate / (1 - slo_target)``) reaches ``threshold`` (e.g. ``14.4`` for
      a fast-burn page).
    """

    name: str
    metric: Literal["cost", "latency", "error_rate", "value"] = "value"
    kind: Literal["threshold", "ewma", "burn_rate"] = "threshold"
    threshold: float = 0.0
    direction: Literal["above", "below"] = "above"
    severity: Literal["info", "warning", "critical"] = "warning"
    alpha: float = 0.3
    factor: float = 3.0
    min_samples: int = 5
    slo_target: float = 0.99


class AlertManager:
    """Evaluates :class:`AlertRule`\\ s over a metric stream and dispatches alerts.

    Feed it samples with :meth:`observe` (or :meth:`check_store` to read the
    indexed store's percentiles/error-rate), and wire it to the event bus with
    :meth:`subscribe` so existing ``cost.anomaly`` / ``cost.budget_exceeded``
    events become alerts on the same sinks. Sinks
    (:class:`~vincio.observability.exporters.AlertSink`: webhook / Slack /
    PagerDuty / Prometheus) are best-effort — a delivery failure is logged, not
    raised, so alerting never breaks a run.
    """

    def __init__(self, *, sinks: list[AlertSink] | None = None) -> None:
        self.sinks: list[AlertSink] = list(sinks or [])
        self.rules: list[AlertRule] = []
        self._ewma: dict[str, _EwmaTracker] = {}

    def add_rule(self, rule: AlertRule) -> AlertRule:
        self.rules.append(rule)
        return rule

    def add_sink(self, sink: AlertSink) -> AlertSink:
        self.sinks.append(sink)
        return sink

    def _dispatch(self, alert: Alert) -> Alert:
        for sink in self.sinks:
            try:
                sink.send(alert)
            except Exception:  # noqa: BLE001 - alert delivery must not break runs
                logger.warning("alert sink %s failed", type(sink).__name__, exc_info=True)
        return alert

    def observe(
        self, metric: str, value: float, *, key: str = "∅", trace_id: str | None = None
    ) -> list[Alert]:
        """Feed one metric sample; fire and dispatch any matching rules."""
        fired: list[Alert] = []
        for rule in self.rules:
            if rule.metric != metric:
                continue
            alert = self._evaluate(rule, value, key=key, trace_id=trace_id)
            if alert is not None:
                fired.append(self._dispatch(alert))
        return fired

    def _evaluate(
        self, rule: AlertRule, value: float, *, key: str, trace_id: str | None
    ) -> Alert | None:
        if rule.kind == "threshold":
            crossed = value >= rule.threshold if rule.direction == "above" else value <= rule.threshold
            if not crossed:
                return None
            return Alert(
                rule=rule.name, severity=rule.severity, value=value, threshold=rule.threshold,
                dimension=rule.metric, key=key, trace_id=trace_id,
                message=f"{rule.metric} {value:g} crossed {rule.direction} {rule.threshold:g}",
            )
        if rule.kind == "ewma":
            tracker = self._ewma.setdefault(f"{rule.name}:{key}", _EwmaTracker(rule.alpha))
            anomaly = False
            z = 0.0
            if tracker.count >= rule.min_samples:
                z = tracker.zscore(value)
                anomaly = abs(z) >= rule.factor
            tracker.update(value)
            if not anomaly:
                return None
            z_disp = "∞" if math.isinf(z) else f"{z:.1f}"
            return Alert(
                rule=rule.name, severity=rule.severity, value=value, threshold=round(tracker.mean, 6),
                dimension=rule.metric, key=key, trace_id=trace_id,
                message=f"{rule.metric} {value:g} is {z_disp}σ from EWMA mean {tracker.mean:g}",
            )
        # burn_rate
        burn = value / (1 - rule.slo_target) if rule.slo_target < 1 else value
        if burn < rule.threshold:
            return None
        return Alert(
            rule=rule.name, severity=rule.severity, value=round(burn, 4), threshold=rule.threshold,
            dimension="burn_rate", key=key, trace_id=trace_id,
            message=f"error-budget burn rate {burn:.1f}x ≥ {rule.threshold:g}x (SLO {rule.slo_target:g})",
        )

    def check_store(
        self, store: IndexedTraceStore, *, since: datetime | None = None, tenant_id: str | None = None
    ) -> list[Alert]:
        """Read the indexed store's current p95 latency/cost + error rate and
        evaluate the matching rules — the periodic poll the served plane runs."""
        stats = store.stats()
        latency = store.percentiles("latency", since=since, tenant_id=tenant_id)
        cost = store.percentiles("cost", since=since, tenant_id=tenant_id)
        key = tenant_id or "∅"
        fired: list[Alert] = []
        fired += self.observe("error_rate", float(stats["error_rate"]), key=key)
        fired += self.observe("latency", latency.p95, key=key)
        fired += self.observe("cost", cost.p95, key=key)
        return fired

    def subscribe(self, bus: EventBus) -> None:
        """Turn existing cost events on the bus into alerts on the sinks."""
        bus.subscribe("cost.anomaly", self._on_cost_anomaly)
        bus.subscribe("cost.budget_exceeded", self._on_budget_exceeded)

    def _on_cost_anomaly(self, event: Event) -> None:
        payload = event.payload
        self._dispatch(
            Alert(
                rule="cost.anomaly", severity="warning",
                value=float(payload.get("cost_usd", 0.0)),
                threshold=float(payload.get("mean_usd", 0.0)),
                dimension="cost", key=str(payload.get("scope", "")), trace_id=event.trace_id,
                message=(
                    f"cost spike ${payload.get('cost_usd', 0):.4f} vs mean "
                    f"${payload.get('mean_usd', 0):.4f} (x{payload.get('factor', '?')})"
                ),
            )
        )

    def _on_budget_exceeded(self, event: Event) -> None:
        payload = event.payload
        self._dispatch(
            Alert(
                rule="cost.budget_exceeded", severity="critical",
                dimension="budget", key=str(payload.get("scope", "")), trace_id=event.trace_id,
                message=str(payload.get("reason", "budget exceeded")),
            )
        )
