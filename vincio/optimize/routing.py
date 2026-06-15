"""Model routing optimization.

- :class:`RoutingPolicy` — deterministic difficulty/risk-based routing.
- :class:`RoutingOptimizer` — learns the difficulty threshold offline from
  per-tier eval reports.
- :class:`EpsilonGreedyBandit` and :class:`UCB1Bandit` — live routing
  bandits, to be used only behind offline eval gates.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

from pydantic import BaseModel

from ..core.types import ModelResponse, TaskType
from ..evals.reports import EvalReport
from .search import FitnessWeights

__all__ = [
    "RoutingPolicy",
    "estimate_difficulty",
    "RoutingOptimizer",
    "EpsilonGreedyBandit",
    "UCB1Bandit",
    "CascadeRung",
    "ModelCascade",
    "response_confidence",
]

_REASONING_RE = re.compile(
    r"(?i)\b(why|prove|derive|step[- ]by[- ]step|trade-?offs?|compare and|multi-?hop|implications?|root cause)\b"
)


def estimate_difficulty(text: str, *, task_type: TaskType = TaskType.GENERAL, evidence_count: int = 0) -> float:
    """Deterministic difficulty estimate in [0,1]."""
    score = 0.15
    words = len(text.split())
    score += min(0.3, words / 400)
    if _REASONING_RE.search(text):
        score += 0.25
    if task_type in (TaskType.AGENT_WORKFLOW, TaskType.PLANNING, TaskType.COMPLIANCE_REVIEW, TaskType.DOCUMENT_COMPARISON):
        score += 0.2
    elif task_type in (TaskType.CLASSIFICATION, TaskType.EXTRACTION):
        score -= 0.1
    score += min(0.2, evidence_count / 40)
    return max(0.0, min(1.0, score))


class RoutingPolicy(BaseModel):
    """Routing policy."""

    cheap_model: str
    default_model: str
    strong_model: str
    difficulty_threshold_low: float = 0.3
    difficulty_threshold_high: float = 0.65

    def route(
        self,
        *,
        difficulty: float,
        risk: str = "low",
        requires_reasoning: bool = False,
        validation_failed: bool = False,
    ) -> str:
        if validation_failed or requires_reasoning or risk == "high":
            return self.strong_model
        if difficulty < self.difficulty_threshold_low and risk == "low":
            return self.cheap_model
        if difficulty > self.difficulty_threshold_high:
            return self.strong_model
        return self.default_model


def response_confidence(response: ModelResponse, *, expects_schema: bool = False) -> float:
    """Default runtime confidence signal for a model response, in [0, 1].

    A clean stop is high confidence; a truncated or content-filtered answer, or
    a structured request that failed to parse, is low — exactly the cases worth
    escalating to a stronger model. Apps can supply a custom signal (e.g. a
    confidence metric) to :meth:`ContextApp.use_cascade`.
    """
    if response.finish_reason in ("length", "content_filter", "error"):
        return 0.0
    if expects_schema and response.structured is None:
        return 0.2
    if not (response.text or response.structured or response.tool_calls):
        return 0.0
    return 1.0


class CascadeRung(BaseModel):
    """One step of a runtime cascade: a model and the confidence below which a
    response is escalated to the next rung."""

    model: str
    provider: str | None = None
    min_confidence: float = 0.5


class ModelCascade(BaseModel):
    """An ordered cheap→strong model ladder for confidence-based escalation.

    At run time the cascade starts on the first (cheapest) rung and escalates to
    the next only when a response's confidence falls below the current rung's
    threshold — so most runs finish cheap and only the hard ones pay for the
    stronger model. The offline :class:`RoutingOptimizer` keeps tuning the
    thresholds; this is its runtime counterpart.
    """

    rungs: list[CascadeRung]
    max_escalations: int | None = None  # default: walk the whole ladder

    def model_post_init(self, _ctx: Any) -> None:
        if not self.rungs:
            raise ValueError("ModelCascade requires at least one rung")
        models = [rung.model for rung in self.rungs]
        if len(set(models)) != len(models):
            raise ValueError(f"ModelCascade rungs must have unique model names, got {models}")

    @classmethod
    def from_models(
        cls, models: list[str], *, min_confidence: float = 0.5, max_escalations: int | None = None
    ) -> ModelCascade:
        """Build a cascade from a cheap→strong list of model names."""
        return cls(
            rungs=[CascadeRung(model=m, min_confidence=min_confidence) for m in models],
            max_escalations=max_escalations,
        )

    @property
    def escalation_cap(self) -> int:
        ladder = len(self.rungs) - 1
        return ladder if self.max_escalations is None else min(self.max_escalations, ladder)

    def first(self) -> CascadeRung:
        return self.rungs[0]

    def _index(self, model: str) -> int:
        for i, rung in enumerate(self.rungs):
            if rung.model == model:
                return i
        return -1

    def next_rung(self, model: str, confidence: float) -> CascadeRung | None:
        """The next stronger rung when ``confidence`` is below ``model``'s
        threshold, else ``None`` (stay where we are)."""
        i = self._index(model)
        if i < 0 or i + 1 >= len(self.rungs):
            return None
        if confidence >= self.rungs[i].min_confidence:
            return None
        return self.rungs[i + 1]


class RoutingOptimizer:
    """Learn the low/high thresholds from per-tier eval reports.

    Provide eval reports of the SAME dataset run with cheap/default/strong
    models, each case annotated with its difficulty in
    ``case.details['difficulty']`` (the app's eval target records it).
    """

    def __init__(self, weights: FitnessWeights | None = None) -> None:
        self.weights = weights or FitnessWeights()

    def optimize(
        self,
        policy: RoutingPolicy,
        reports: dict[str, EvalReport],  # tier name -> report ("cheap"/"default"/"strong")
        *,
        quality_metric: str = "semantic_similarity",
        min_quality_ratio: float = 0.97,
    ) -> RoutingPolicy:
        cheap = reports.get("cheap")
        default = reports.get("default")
        if cheap is None or default is None:
            return policy
        cheap_by_id = {c.case_id: c for c in cheap.cases}
        # Find the highest difficulty bucket where the cheap model keeps
        # >= min_quality_ratio of the default model's quality.
        buckets: dict[int, list[tuple[float, float]]] = {}
        for case in default.cases:
            cheap_case = cheap_by_id.get(case.case_id)
            if cheap_case is None:
                continue
            difficulty = float(
                case.details.get("difficulty")
                or cheap_case.details.get("difficulty")
                or 0.5
            )
            quality_default = case.metrics.get(quality_metric)
            quality_cheap = cheap_case.metrics.get(quality_metric)
            if quality_default is None or quality_cheap is None:
                continue
            buckets.setdefault(int(difficulty * 10), []).append((quality_cheap, quality_default))
        best_low = policy.difficulty_threshold_low
        for bucket in sorted(buckets):
            pairs = buckets[bucket]
            cheap_quality = sum(p[0] for p in pairs) / len(pairs)
            default_quality = sum(p[1] for p in pairs) / len(pairs)
            if default_quality <= 0:
                continue
            if cheap_quality / default_quality >= min_quality_ratio:
                best_low = max(best_low, (bucket + 1) / 10)
            else:
                break
        updated = policy.model_copy(update={"difficulty_threshold_low": min(best_low, policy.difficulty_threshold_high)})
        return updated


class EpsilonGreedyBandit:
    """Live routing bandit. Arms are model names; reward is the
    run-level fitness (caller computes it)."""

    def __init__(self, arms: list[str], *, epsilon: float = 0.1, seed: int | None = None) -> None:
        if not arms:
            raise ValueError("bandit requires at least one arm")
        self.arms = list(arms)
        self.epsilon = epsilon
        self.counts: dict[str, int] = {arm: 0 for arm in arms}
        self.values: dict[str, float] = {arm: 0.0 for arm in arms}
        self._rng = random.Random(seed)

    def select(self) -> str:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(self.arms)
        return max(self.arms, key=lambda arm: self.values[arm])

    def update(self, arm: str, reward: float) -> None:
        if arm not in self.counts:
            raise ValueError(f"unknown arm {arm!r}")
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def snapshot(self) -> dict[str, Any]:
        return {"counts": dict(self.counts), "values": {k: round(v, 4) for k, v in self.values.items()}}


class UCB1Bandit:
    def __init__(self, arms: list[str]) -> None:
        if not arms:
            raise ValueError("bandit requires at least one arm")
        self.arms = list(arms)
        self.counts: dict[str, int] = {arm: 0 for arm in arms}
        self.values: dict[str, float] = {arm: 0.0 for arm in arms}
        self.total = 0

    def select(self) -> str:
        for arm in self.arms:  # play each arm once first
            if self.counts[arm] == 0:
                return arm
        return max(
            self.arms,
            key=lambda arm: self.values[arm]
            + math.sqrt(2 * math.log(self.total) / self.counts[arm]),
        )

    def update(self, arm: str, reward: float) -> None:
        self.counts[arm] += 1
        self.total += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
