"""Shared optimization machinery.

Implements the evolution loop::

    baseline → generate candidate configs → run eval subset
    → select top candidates → run full eval → promote if gates pass

with the §3.9 fitness function and §22.8 promotion safety rules.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import EvalError
from ..evals.datasets import Dataset
from ..evals.experiments import ab_test
from ..evals.reports import EvalReport, evaluate_gates

__all__ = [
    "FitnessWeights",
    "fitness",
    "Candidate",
    "OptimizationResult",
    "evolution_loop",
    "significance_report",
    "apply_significance_gate",
    "MIN_SIGNIFICANCE_CASES",
]

# Statistical power floor: below this many shared cases, a t-test cannot reliably
# establish significance, so an improvement is *warned* as under-powered rather
# than silently trusted (the audit's "compare point estimates" complaint).
MIN_SIGNIFICANCE_CASES = 8


def significance_report(
    baseline_report: EvalReport | None,
    candidate_report: EvalReport | None,
    metric: str,
    *,
    alpha: float = 0.05,
) -> dict[str, Any] | None:
    """``ab_test`` on *metric*, or ``None`` when the metric is absent from either
    report (so callers can gate on significance when available and fall back
    gracefully when not)."""
    if baseline_report is None or candidate_report is None:
        return None
    try:
        return ab_test(baseline_report, candidate_report, metric, alpha=alpha)
    except EvalError:
        return None


def apply_significance_gate(
    result: Any,
    *,
    baseline_report: EvalReport | None,
    candidate_report: EvalReport | None,
    accuracy_metric: str,
    alpha: float = 0.05,
    require_significance: bool = True,
) -> tuple[bool, str | None]:
    """Call the t-test at a promotion gate and record its verdict on *result*.

    Sets ``result.significance`` and appends to ``result.warnings``. Returns
    ``(blocked, reason)``: ``blocked`` is True when the primary metric
    *significantly regresses* (the caller must refuse to promote). A
    non-significant or under-powered gain is warned (the warning's ``n`` is the
    test's actual paired sample size, not the dataset size), never blocked — the
    promotion may still rest on a fitness improvement (e.g. lower cost). Shared
    by the evolution and reflective promotion paths so the gate can't drift.
    """
    sig = (
        significance_report(baseline_report, candidate_report, accuracy_metric, alpha=alpha)
        if require_significance
        else None
    )
    result.significance = sig
    if sig is None:
        return False, None
    if sig["delta"] < 0 and sig["significant"]:
        return True, (
            f"primary metric significantly regressed "
            f"(Δ={sig['delta']}, p={sig['p_value']}, effect={sig['effect_size']})"
        )
    if sig["delta"] > 0 and not sig["significant"]:
        n = min(int(sig.get("n_a", 0)), int(sig.get("n_b", 0)))
        if n < MIN_SIGNIFICANCE_CASES:
            result.warnings.append(
                f"under-powered: {n} paired cases < {MIN_SIGNIFICANCE_CASES}; "
                f"cannot establish significance of the quality gain (p={sig['p_value']})"
            )
        else:
            result.warnings.append(
                f"quality gain not statistically significant "
                f"(Δ={sig['delta']}, p={sig['p_value']}, n={n}); "
                f"promotion rests on the fitness improvement"
            )
    return False, None


class FitnessWeights(BaseModel):
    """Fitness = α·Accuracy + β·Groundedness + γ·SchemaValidity +
    δ·Safety − ε·Cost − ζ·Latency − η·RetryRate."""

    accuracy: float = 1.0  # α
    groundedness: float = 1.0  # β
    schema_validity: float = 0.8  # γ
    safety: float = 1.0  # δ
    cost: float = 0.5  # ε  (per dollar-per-case)
    latency: float = 0.2  # ζ  (per second p50)
    retry_rate: float = 0.3  # η
    accuracy_metric: str = "semantic_similarity"  # which metric stands in for accuracy
    safety_metric: str = "safety"


def _mean(report: EvalReport, metric: str, default: float = 0.0) -> float:
    values = report.metric_values(metric)
    return sum(values) / len(values) if values else default


def fitness(report: EvalReport, weights: FitnessWeights | None = None) -> float:
    w = weights or FitnessWeights()
    accuracy = _mean(report, w.accuracy_metric)
    grounded = _mean(report, "groundedness", default=1.0)
    schema = _mean(report, "schema_validity", default=1.0)
    safety = _mean(report, w.safety_metric, default=1.0)
    cost = _mean(report, "cost")
    latency_s = _mean(report, "latency") / 1000.0
    retries = _mean(report, "retries")
    return (
        w.accuracy * accuracy
        + w.groundedness * grounded
        + w.schema_validity * schema
        + w.safety * safety
        - w.cost * cost
        - w.latency * latency_s
        - w.retry_rate * retries
    )


class Candidate(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    payload: Any = None  # e.g. a PromptVariant
    subset_fitness: float | None = None
    full_fitness: float | None = None
    subset_report: Any = None
    full_report: Any = None


class OptimizationResult(BaseModel):
    baseline_fitness: float
    baseline: Candidate | None = None
    best: Candidate | None = None
    promoted: bool = False
    reason: str = ""
    candidates: list[Candidate] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    # Statistical backing for the promotion decision (1.7): the ab_test verdict
    # on the primary metric (p-value, confidence interval, effect size), so a
    # promotion is defensible at a confidence level rather than a point estimate.
    significance: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


# evaluate_fn(candidate, dataset) -> EvalReport
EvaluateFn = Callable[[Candidate, Dataset], Awaitable[EvalReport]]


def _promotion_safe(
    candidate_report: EvalReport,
    baseline_report: EvalReport,
    *,
    gates: dict[str, str] | None,
    max_cost_per_case: float | None,
) -> tuple[bool, str]:
    """Never auto-promote when safety/schema regress, cost blows
    the budget, or eval gates fail."""
    for metric in ("safety", "schema_validity"):
        baseline_values = baseline_report.metric_values(metric)
        candidate_values = candidate_report.metric_values(metric)
        if baseline_values and candidate_values:
            baseline_mean = sum(baseline_values) / len(baseline_values)
            candidate_mean = sum(candidate_values) / len(candidate_values)
            if candidate_mean < baseline_mean - 1e-9:
                return False, f"{metric} regressed ({baseline_mean:.4f} → {candidate_mean:.4f})"
    if max_cost_per_case is not None:
        cost_values = candidate_report.metric_values("cost")
        if cost_values and sum(cost_values) / len(cost_values) > max_cost_per_case:
            return False, "cost exceeds budget"
    if gates:
        outcomes = evaluate_gates(candidate_report, gates)
        failed = [k for k, v in outcomes.items() if not v["passed"]]
        if failed:
            return False, f"gates failed: {failed}"
    return True, "ok"


async def evolution_loop(
    candidates: list[Candidate],
    evaluate_fn: EvaluateFn,
    dataset: Dataset,
    *,
    baseline: Candidate,
    weights: FitnessWeights | None = None,
    subset_size: int = 16,
    top_n: int = 3,
    gates: dict[str, str] | None = None,
    max_cost_per_case: float | None = None,
    min_improvement: float = 1e-6,
    min_dataset_coverage: int = 4,
    require_significance: bool = True,
    alpha: float = 0.05,
) -> OptimizationResult:
    weights = weights or FitnessWeights()
    if len(dataset) < min_dataset_coverage:
        return OptimizationResult(
            baseline_fitness=float("nan"),
            promoted=False,
            reason=f"dataset too small ({len(dataset)} cases < {min_dataset_coverage}); refusing to optimize",
        )
    subset = dataset.sample(subset_size)

    # Baseline on the full dataset.
    baseline.full_report = await evaluate_fn(baseline, dataset)
    baseline.full_fitness = fitness(baseline.full_report, weights)
    history: list[dict[str, Any]] = [
        {"phase": "baseline", "name": baseline.name, "fitness": baseline.full_fitness}
    ]

    # Phase 1: subset screening. Candidates that arrive pre-scored (e.g.
    # from a guided search strategy) keep their subset fitness.
    for candidate in candidates:
        if candidate.subset_fitness is None:
            candidate.subset_report = await evaluate_fn(candidate, subset)
            candidate.subset_fitness = fitness(candidate.subset_report, weights)
        history.append(
            {"phase": "subset", "name": candidate.name, "fitness": candidate.subset_fitness}
        )
    survivors = sorted(candidates, key=lambda c: c.subset_fitness or float("-inf"), reverse=True)[
        :top_n
    ]

    # Phase 2: full eval for survivors.
    for candidate in survivors:
        candidate.full_report = await evaluate_fn(candidate, dataset)
        candidate.full_fitness = fitness(candidate.full_report, weights)
        history.append(
            {"phase": "full", "name": candidate.name, "fitness": candidate.full_fitness}
        )

    best = max(survivors, key=lambda c: c.full_fitness or float("-inf"), default=None)
    result = OptimizationResult(
        baseline_fitness=baseline.full_fitness,
        baseline=baseline,
        best=best,
        candidates=candidates,
        history=history,
    )
    if best is None or (best.full_fitness or float("-inf")) <= baseline.full_fitness + min_improvement:
        result.reason = "no candidate beat the baseline"
        return result
    safe, reason = _promotion_safe(
        best.full_report, baseline.full_report, gates=gates, max_cost_per_case=max_cost_per_case
    )
    if not safe:
        result.promoted = False
        result.reason = reason
        return result

    # Significance gate (1.7): call the t-test at the gate so the promotion is
    # defensible at a confidence level, not a point estimate.
    blocked, gate_reason = apply_significance_gate(
        result,
        baseline_report=baseline.full_report,
        candidate_report=best.full_report,
        accuracy_metric=weights.accuracy_metric,
        alpha=alpha,
        require_significance=require_significance,
    )
    if blocked:
        result.promoted = False
        result.reason = gate_reason or "primary metric significantly regressed"
        return result

    result.promoted = True
    sig = result.significance
    detail = f" (p={sig['p_value']}, effect={sig['effect_size']})" if sig is not None else ""
    result.reason = (
        f"promoted {best.name}: fitness {baseline.full_fitness:.4f} → {best.full_fitness:.4f}{detail}"
    )
    return result
