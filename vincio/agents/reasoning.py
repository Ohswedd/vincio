"""Reasoning-effort control (agents/reasoning).

A reasoning model's thinking budget is the cheapest quality lever left: spend
more tokens thinking on a hard step, none on an easy one. Set it by hand and you
either overpay on the easy steps or starve the hard ones. The
:class:`ReasoningController` makes the choice a *policy* over the signals the
platform already computes — the task classification and an estimated difficulty
(the same signals that drive speculative retrieval prefetch and the
capability-aware router) and the live token budget — and returns a
:class:`ReasoningDecision` so every choice is explained on the trace, never
silent.

Two guardrails keep it safe. A **hard reasoning-token ceiling** (held by an SLO)
means a hard task can never silently exhaust the run: the thinking budget is
clamped to the ceiling *and* to a fraction of the remaining output budget,
whichever is smaller. And when a :class:`~vincio.caching.ReasoningTraceCache`
shows the call's thinking prefix is already **warm** — the same stable prefix was
thought through on an earlier ask — the controller steps the effort down, because
the expensive part of the reasoning was already paid for.

The decision is deterministic given its inputs (it reuses the deterministic
difficulty estimator and the provider-neutral effort→budget mapping), so a seeded
run is fully reproducible.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..core.types import ReasoningEffort, TaskType
from ..providers.base import reasoning_budget_from_effort

__all__ = ["ReasoningPolicy", "ReasoningDecision", "ReasoningController"]

# Effort levels, weakest → strongest. The single ordering used to bucket by
# difficulty, escalate on low confidence, and step down on a warm prefix.
_EFFORT_LADDER: tuple[ReasoningEffort, ...] = ("minimal", "low", "medium", "high")
_EFFORT_RANK = {effort: rank for rank, effort in enumerate(_EFFORT_LADDER)}


def _clamp_effort(effort: ReasoningEffort, low: ReasoningEffort, high: ReasoningEffort) -> ReasoningEffort:
    rank = max(_EFFORT_RANK[low], min(_EFFORT_RANK[high], _EFFORT_RANK[effort]))
    return _EFFORT_LADDER[rank]


def _step(effort: ReasoningEffort, delta: int) -> ReasoningEffort:
    rank = max(0, min(len(_EFFORT_LADDER) - 1, _EFFORT_RANK[effort] + delta))
    return _EFFORT_LADDER[rank]


class ReasoningPolicy(BaseModel):
    """The effort policy: difficulty bands, guardrails, and reuse behavior.

    ``low_effort_below`` / ``high_effort_above`` are difficulty cut-points in
    ``[0, 1]``: below the first the task is easy (``min_effort``), above the
    second it is hard (toward ``max_effort``), in between it is the middle level.
    ``max_reasoning_tokens`` is the hard ceiling held by the SLO; ``budget_fraction``
    caps thinking at a share of the remaining output budget so reasoning cannot
    crowd out the answer. ``quality_floor`` is the confidence below which the
    controller escalates one level; ``reuse_warm_prefix`` steps effort down when
    the thinking prefix is already cached.
    """

    min_effort: ReasoningEffort = "minimal"
    max_effort: ReasoningEffort = "high"
    low_effort_below: float = 0.3
    high_effort_above: float = 0.65
    max_reasoning_tokens: int = 16_384
    budget_fraction: float = 0.5
    quality_floor: float = 0.6
    reuse_warm_prefix: bool = True

    def base_effort(self, difficulty: float) -> ReasoningEffort:
        """The effort level a difficulty maps to, before any adjustment."""
        if difficulty < self.low_effort_below:
            effort: ReasoningEffort = "minimal"
        elif difficulty > self.high_effort_above:
            effort = "high"
        else:
            effort = "medium"
        return _clamp_effort(effort, self.min_effort, self.max_effort)


class ReasoningDecision(BaseModel):
    """The record of one reasoning-effort pick — stamped on the trace."""

    effort: ReasoningEffort
    thinking_budget_tokens: int
    difficulty: float
    reason: str = ""
    escalated: bool = False
    warm_prefix: bool = False
    ceiling_capped: bool = False
    budget_capped: bool = False
    confidence: float | None = None


def _task_type(task: Any) -> TaskType:
    """Resolve a TaskType from a classification, an enum, a string, or None."""
    if task is None:
        return TaskType.GENERAL
    inner = getattr(task, "task_type", task)
    if isinstance(inner, TaskType):
        return inner
    try:
        return TaskType(str(getattr(inner, "value", inner)))
    except ValueError:
        return TaskType.GENERAL


class ReasoningController:
    """Pick a thinking effort + token budget per step from task + budget signals.

    ``policy`` configures the difficulty bands and guardrails; ``trace_cache``,
    when supplied, lets a warm thinking prefix step the effort down. The decision
    is returned (and cached as :attr:`last_decision`) so the runtime can stamp it
    on the trace and apply it to the model request.
    """

    def __init__(
        self,
        policy: ReasoningPolicy | None = None,
        *,
        trace_cache: Any | None = None,
    ) -> None:
        self.policy = policy or ReasoningPolicy()
        if trace_cache is None:
            from ..caching import ReasoningTraceCache

            trace_cache = ReasoningTraceCache()
        self.trace_cache = trace_cache
        self.last_decision: ReasoningDecision | None = None

    def estimate_difficulty(
        self, text: str, *, task: Any = None, evidence_count: int = 0
    ) -> float:
        """Deterministic difficulty in ``[0, 1]`` (reuses the router's estimator)."""
        from ..optimize.routing import estimate_difficulty

        return estimate_difficulty(
            text, task_type=_task_type(task), evidence_count=evidence_count
        )

    def decide(
        self,
        *,
        task: Any = None,
        text: str = "",
        remaining_output_tokens: int | None = None,
        confidence: float | None = None,
        evidence_count: int = 0,
        difficulty: float | None = None,
        prefix_hash: str | None = None,
        model: str | None = None,
    ) -> ReasoningDecision:
        """Decide the reasoning effort and thinking-token budget for one step.

        ``difficulty`` is computed from ``text``/``task``/``evidence_count`` when
        not given. A low ``confidence`` (from a prior attempt) escalates one
        level; a warm ``prefix_hash``/``model`` in the trace cache steps it down.
        The thinking budget is clamped to the hard ceiling and to a fraction of
        ``remaining_output_tokens``.
        """
        policy = self.policy
        if difficulty is None:
            difficulty = self.estimate_difficulty(
                text, task=task, evidence_count=evidence_count
            )
        difficulty = max(0.0, min(1.0, difficulty))
        effort = policy.base_effort(difficulty)
        reasons: list[str] = [f"difficulty {difficulty:.2f} → {effort}"]

        escalated = False
        if confidence is not None and confidence < policy.quality_floor:
            stronger = _clamp_effort(_step(effort, +1), policy.min_effort, policy.max_effort)
            if stronger != effort:
                effort = stronger
                escalated = True
                reasons.append(
                    f"escalated to {effort} (confidence {confidence:.2f} < {policy.quality_floor:.2f})"
                )

        warm = False
        if (
            policy.reuse_warm_prefix
            and self.trace_cache is not None
            and prefix_hash is not None
            and model is not None
        ):
            if self.trace_cache.lookup(prefix_hash, model, effort) is not None:
                stepped = _clamp_effort(_step(effort, -1), policy.min_effort, policy.max_effort)
                if stepped != effort:
                    effort = stepped
                    warm = True
                    reasons.append(f"warm thinking prefix → stepped down to {effort}")
                else:
                    warm = True
                    reasons.append("warm thinking prefix (already at floor)")

        # Hard ceiling and budget-share clamp on the thinking-token budget.
        budget = reasoning_budget_from_effort(effort, None)
        ceiling_capped = False
        if budget > policy.max_reasoning_tokens:
            budget = policy.max_reasoning_tokens
            ceiling_capped = True
            reasons.append(f"capped at hard ceiling {policy.max_reasoning_tokens} tokens")
        budget_capped = False
        if remaining_output_tokens is not None:
            share = int(remaining_output_tokens * policy.budget_fraction)
            if share < budget:
                budget = max(0, share)
                budget_capped = True
                reasons.append(
                    f"capped at {policy.budget_fraction:.0%} of remaining output budget "
                    f"({budget} tokens)"
                )

        decision = ReasoningDecision(
            effort=effort,
            thinking_budget_tokens=budget,
            difficulty=round(difficulty, 4),
            reason="; ".join(reasons),
            escalated=escalated,
            warm_prefix=warm,
            ceiling_capped=ceiling_capped,
            budget_capped=budget_capped,
            confidence=confidence,
        )
        self.last_decision = decision
        return decision

    def record_trace(
        self,
        *,
        prefix_hash: str,
        model: str,
        effort: str | None = None,
        reasoning_tokens: int = 0,
        response_id: str | None = None,
    ) -> None:
        """Note that a thinking prefix was paid for, so a re-ask can reuse it."""
        if self.trace_cache is None:
            return
        self.trace_cache.record(
            prefix_hash=prefix_hash,
            model=model,
            effort=effort,
            reasoning_tokens=reasoning_tokens,
            response_id=response_id,
        )
