"""Cost-aware action selection (agents/selection).

A planner that always reaches for the strongest model overpays on the easy
steps; one that always reaches for the cheapest fails the hard ones. The
:class:`CostAwareSelector` consults the :class:`~vincio.providers.registry.ModelRegistry`
— the single source of capability and pricing — and the live budget to pick, per
action, the **cheapest capable model that clears the quality bar**, and escalates
to a stronger model only when the last attempt's confidence was low.

It reuses the same capability guard the router uses, so a model that cannot serve
the call (missing tools, structured output, reasoning, or context) is never
chosen — cost never overrides capability. The decision is returned as a
:class:`SelectionDecision` for the trace, so every escalation is explained, not
silent.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..providers.capabilities import RequestNeeds, capability_check

__all__ = ["ActionCandidate", "SelectionDecision", "CostAwareSelector"]

# Tier ordering: a higher rank is a stronger (and usually pricier) model. Used to
# resolve cost ties deterministically and to drive single-step escalation.
_TIER_RANK = {"fast": 0, "default": 1, "strong": 2}


class ActionCandidate(BaseModel):
    model: str
    tier: str = "default"
    tier_rank: int = 1
    est_cost_usd: float = 0.0
    capable: bool = True
    skip_reason: str = ""


class SelectionDecision(BaseModel):
    """The record of one action-selection pick — stamped on the trace."""

    model: str
    tier: str = "default"
    est_cost_usd: float = 0.0
    reason: str = ""
    escalated: bool = False
    downgraded: bool = False
    confidence: float | None = None
    quality_floor: float = 0.0
    candidates: list[str] = Field(default_factory=list)
    skipped: dict[str, str] = Field(default_factory=dict)


class CostAwareSelector:
    """Picks the cheapest capable model per action, escalating on low confidence.

    ``models`` is the candidate set (any order — they are ranked by registry
    price). ``quality_floor`` is the confidence below which the selector escalates
    one tier; pass the prior step's confidence to :meth:`select` to drive it.
    Costs and capabilities both come from ``registry``, so the planner reads one
    source of truth.
    """

    def __init__(
        self,
        models: list[str],
        *,
        registry: Any | None = None,
        quality_floor: float = 0.6,
        events: Any | None = None,
    ) -> None:
        if not models:
            raise ValueError("CostAwareSelector requires at least one candidate model")
        self.models = models
        self._registry = registry
        self.quality_floor = quality_floor
        self._events = events
        self.last_decision: SelectionDecision | None = None

    def _reg(self) -> Any:
        if self._registry is None:
            from ..providers.registry import default_model_registry

            self._registry = default_model_registry()
        return self._registry

    def _est_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate the call's USD cost straight from the registry profile."""
        profile = self._reg().resolve(model)
        if profile is None:
            return 0.0
        return (
            input_tokens * profile.input_cost_per_mtok
            + output_tokens * profile.output_cost_per_mtok
        ) / 1_000_000

    def candidates(
        self, *, needs: RequestNeeds, input_tokens: int, output_tokens: int
    ) -> list[ActionCandidate]:
        registry = self._reg()
        out: list[ActionCandidate] = []
        for model in self.models:
            profile = registry.resolve(model)
            tier = profile.tier if profile is not None else "default"
            verdict = capability_check(needs, registry.guard_capabilities(model), model=model)
            out.append(
                ActionCandidate(
                    model=model,
                    tier=tier,
                    tier_rank=_TIER_RANK.get(tier, 1),
                    est_cost_usd=self._est_cost(model, input_tokens, output_tokens),
                    capable=verdict.ok,
                    skip_reason="" if verdict.ok else verdict.reason,
                )
            )
        return out

    def select(
        self,
        *,
        needs: RequestNeeds,
        input_tokens: int,
        output_tokens: int,
        remaining_budget_usd: float | None = None,
        confidence: float | None = None,
    ) -> SelectionDecision:
        """Choose a model for one action.

        Starts from the cheapest capable model within ``remaining_budget_usd``;
        when ``confidence`` is supplied and below :attr:`quality_floor`, escalates
        to the cheapest *stronger-tier* capable model (still budget-bounded). When
        nothing fits the budget the cheapest capable model is chosen and flagged
        ``downgraded`` — capability is never traded away for price.
        """
        cands = self.candidates(needs=needs, input_tokens=input_tokens, output_tokens=output_tokens)
        capable = [c for c in cands if c.capable]
        skipped = {c.model: c.skip_reason for c in cands if not c.capable}
        if not capable:
            from ..core.errors import CapabilityMismatchError

            raise CapabilityMismatchError(
                f"no capable model among {self.models} for needs {needs.summary()}; skipped {skipped}",
                missing=needs.summary(),
            )
        # Cheapest capable first; ties broken by the weaker (cheaper-by-tier) model.
        ranked = sorted(capable, key=lambda c: (c.est_cost_usd, c.tier_rank))
        affordable = (
            [c for c in ranked if c.est_cost_usd <= remaining_budget_usd]
            if remaining_budget_usd is not None
            else ranked
        )
        downgraded = False
        if affordable:
            chosen = affordable[0]
        else:  # nothing fits — cheapest capable, flagged
            chosen = ranked[0]
            downgraded = True

        escalated = False
        if confidence is not None and confidence < self.quality_floor:
            pool = affordable or ranked
            stronger = [c for c in pool if c.tier_rank > chosen.tier_rank]
            if stronger:
                chosen = min(stronger, key=lambda c: (c.tier_rank, c.est_cost_usd))
                escalated = True

        reason = (
            f"escalated to {chosen.tier} tier (confidence {confidence:.2f} < {self.quality_floor:.2f})"
            if escalated
            else f"cheapest capable ({chosen.tier} tier) of {len(capable)} candidate(s)"
        )
        if downgraded:
            reason += "; over per-step budget, chose cheapest capable"
        decision = SelectionDecision(
            model=chosen.model,
            tier=chosen.tier,
            est_cost_usd=chosen.est_cost_usd,
            reason=reason,
            escalated=escalated,
            downgraded=downgraded,
            confidence=confidence,
            quality_floor=self.quality_floor,
            candidates=[c.model for c in capable],
            skipped=skipped,
        )
        self.last_decision = decision
        if self._events is not None:
            from ..core.events import ModelRouted

            self._events.publish(
                ModelRouted(model=decision.model, strategy="cost_aware", reason=decision.reason)
            )
        return decision
