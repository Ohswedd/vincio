"""Model-lifecycle watcher & migration proposals.

The :class:`~vincio.providers.registry.ModelRegistry` already binds GA /
deprecation / retirement dates and a suggested ``successor`` to every model.
:class:`LifecycleWatcher` reads them to:

* emit **early sunset warnings** as a pinned model nears retirement, and
* **propose a migration** — to the model's declared successor, or to a cheaper
  Pareto-dominating model that is at least as capable — that the caller can run
  through the :class:`~vincio.evals.swap.SwapGate` and a canary before adopting.

A proposal can rewrite a :class:`~vincio.optimize.routing.ModelCascade`,
:class:`~vincio.optimize.routing.RoutingPolicy`, or ``config.model`` in place, so
the rotation flows through the same promotion machinery as every other change.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel

from ..core.types import ModelCapabilities, ModelLifecycle, ModelProfile
from ..core.utils import utcnow

__all__ = [
    "LifecycleAlert",
    "MigrationProposal",
    "LifecycleWatcher",
]


class LifecycleAlert(BaseModel):
    """A lifecycle finding for one pinned model."""

    model: str
    lifecycle: ModelLifecycle = "ga"
    severity: str = "info"  # info | warn | critical
    deprecation_date: str | None = None
    retirement_date: str | None = None
    days_to_retirement: int | None = None
    successor: str | None = None
    message: str = ""


class MigrationProposal(BaseModel):
    """A proposed rotation off a sunsetting (or simply pricier) model."""

    from_model: str
    to_model: str | None = None
    kind: str = "none"  # successor | pareto | none
    reason: str = ""
    capability_superset: bool = False
    from_blended_cost_per_mtok: float = 0.0
    to_blended_cost_per_mtok: float = 0.0
    savings_pct: float = 0.0

    @property
    def actionable(self) -> bool:
        return self.kind != "none" and self.to_model is not None

    def apply_to_config(self, config: Any) -> bool:
        """Rewrite ``config.provider.model`` to the proposed model (if it matches)."""
        if not self.actionable:
            return False
        provider_cfg = getattr(config, "provider", None)
        if provider_cfg is not None and getattr(provider_cfg, "model", None) == self.from_model:
            provider_cfg.model = self.to_model
            return True
        return False

    def apply_to_cascade(self, cascade: Any) -> Any:
        """Return a copy of *cascade* with the from-model rung repointed."""
        if not self.actionable:
            return cascade
        rungs = []
        changed = False
        for rung in cascade.rungs:
            if rung.model == self.from_model:
                rungs.append(rung.model_copy(update={"model": self.to_model}))
                changed = True
            else:
                rungs.append(rung)
        if not changed:
            return cascade
        return cascade.model_copy(update={"rungs": rungs})

    def apply_to_policy(self, policy: Any) -> Any:
        """Return a copy of *policy* with any tier equal to the from-model repointed."""
        if not self.actionable:
            return policy
        update = {
            field: self.to_model
            for field in ("cheap_model", "default_model", "strong_model")
            if getattr(policy, field, None) == self.from_model
        }
        return policy.model_copy(update=update) if update else policy


def _blended_cost(profile: ModelProfile) -> float:
    return profile.input_cost_per_mtok + profile.output_cost_per_mtok


def _capability_superset(candidate: ModelCapabilities, base: ModelCapabilities) -> bool:
    """Whether *candidate* meets or exceeds *base* on every capability axis."""
    for flag in ("structured_output", "tool_calling", "vision", "audio", "reasoning",
                 "prompt_caching", "supports_developer_message"):
        if getattr(base, flag) and not getattr(candidate, flag):
            return False
    if candidate.max_context_tokens < base.max_context_tokens:
        return False
    if base.max_output_tokens and candidate.max_output_tokens < base.max_output_tokens:
        return False
    return set(base.input_modalities).issubset(set(candidate.input_modalities))


class LifecycleWatcher:
    """Watch pinned models for sunset and propose migrations off them."""

    def __init__(
        self,
        registry: Any | None = None,
        *,
        warn_within_days: int = 90,
        events: Any | None = None,
    ) -> None:
        self._registry = registry
        self.warn_within_days = warn_within_days
        self._events = events

    def _reg(self) -> Any:
        if self._registry is None:
            from .registry import default_model_registry

            self._registry = default_model_registry()
        return self._registry

    @staticmethod
    def _days_to(value: str | None, today: date) -> int | None:
        if not value:
            return None
        try:
            return (date.fromisoformat(value[:10]) - today).days
        except ValueError:
            return None

    def alert(self, model: str, *, as_of: date | None = None) -> LifecycleAlert:
        """The lifecycle alert for a single pinned model id."""
        today = as_of or utcnow().date()
        profile = self._reg().resolve(model)
        if profile is None:
            return LifecycleAlert(model=model, severity="info",
                                  message=f"{model!r} is not in the registry — lifecycle unknown")
        lifecycle = profile.lifecycle(as_of=today)
        days = self._days_to(profile.retirement_date, today)
        severity = "info"
        message = f"{model!r} is {lifecycle}"
        if lifecycle == "retired":
            severity = "critical"
            message = f"{model!r} is retired — rotate now"
        elif lifecycle == "deprecated":
            severity = "warn"
            message = f"{model!r} is deprecated" + (
                f", retires in {days} day(s)" if days is not None else ""
            )
        elif days is not None and 0 <= days <= self.warn_within_days:
            severity = "warn"
            message = f"{model!r} retires in {days} day(s) — plan a migration"
        return LifecycleAlert(
            model=model, lifecycle=lifecycle, severity=severity,
            deprecation_date=profile.deprecation_date, retirement_date=profile.retirement_date,
            days_to_retirement=days, successor=profile.successor, message=message,
        )

    def scan(
        self, models: list[str], *, as_of: date | None = None, min_severity: str = "warn"
    ) -> list[LifecycleAlert]:
        """Scan pinned models; return alerts at or above ``min_severity`` and emit
        a ``model.sunset`` event for each."""
        order = {"info": 0, "warn": 1, "critical": 2}
        floor = order.get(min_severity, 1)
        alerts: list[LifecycleAlert] = []
        for model in models:
            alert = self.alert(model, as_of=as_of)
            if order[alert.severity] >= floor:
                alerts.append(alert)
                if self._events is not None:
                    self._events.emit("model.sunset", alert.model_dump())
        return alerts

    def propose_migration(self, model: str, *, as_of: date | None = None) -> MigrationProposal:
        """Propose a rotation off *model*: its declared successor first, else the
        cheapest Pareto-dominating (at-least-as-capable, strictly cheaper) model."""
        registry = self._reg()
        profile = registry.resolve(model)
        if profile is None:
            return MigrationProposal(from_model=model, kind="none",
                                     reason="model not in registry")
        base_cost = _blended_cost(profile)

        # 1) Declared successor.
        if profile.successor:
            successor = registry.resolve(profile.successor)
            if successor is not None:
                to_cost = _blended_cost(successor)
                return MigrationProposal(
                    from_model=model, to_model=successor.model, kind="successor",
                    reason=f"declared successor of {model!r}",
                    capability_superset=_capability_superset(
                        successor.capabilities, profile.capabilities
                    ),
                    from_blended_cost_per_mtok=round(base_cost, 4),
                    to_blended_cost_per_mtok=round(to_cost, 4),
                    savings_pct=round(1 - to_cost / base_cost, 4) if base_cost > 0 else 0.0,
                )

        # 2) Cheapest Pareto-dominating, at-least-as-capable, GA model.
        today = as_of or utcnow().date()
        best: ModelProfile | None = None
        for candidate in registry.profiles():
            if candidate.model == model or candidate.lifecycle(as_of=today) != "ga":
                continue
            cand_cost = _blended_cost(candidate)
            if base_cost <= 0 or cand_cost <= 0 or cand_cost >= base_cost:
                continue
            if not _capability_superset(candidate.capabilities, profile.capabilities):
                continue
            if best is None or _blended_cost(candidate) < _blended_cost(best):
                best = candidate
        if best is not None:
            to_cost = _blended_cost(best)
            return MigrationProposal(
                from_model=model, to_model=best.model, kind="pareto",
                reason=f"cheaper, at-least-as-capable replacement for {model!r}",
                capability_superset=True,
                from_blended_cost_per_mtok=round(base_cost, 4),
                to_blended_cost_per_mtok=round(to_cost, 4),
                savings_pct=round(1 - to_cost / base_cost, 4) if base_cost > 0 else 0.0,
            )
        return MigrationProposal(from_model=model, kind="none",
                                 reason="no successor or Pareto-dominating model found")

    def propose_all(
        self, models: list[str], *, as_of: date | None = None
    ) -> list[MigrationProposal]:
        """Migration proposals for every pinned model that is deprecated/retired
        or nearing retirement."""
        proposals: list[MigrationProposal] = []
        for alert in self.scan(models, as_of=as_of, min_severity="warn"):
            proposal = self.propose_migration(alert.model, as_of=as_of)
            if proposal.actionable:
                proposals.append(proposal)
        return proposals
