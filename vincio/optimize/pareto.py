"""Cost/quality Pareto optimization.

Instead of collapsing accuracy, groundedness, latency, and cost into one
scalar, ``pareto_loop`` keeps the full multi-objective frontier: candidates
that are not dominated on any trade-off survive, and promotion picks from
the frontier (the knee point by default, or a constraint-filtered point)
through the same safety gates as the scalar evolution loop. The scalar
fitness is still used for cheap subset screening — Pareto dominance only
means something on full-dataset reports.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..evals.datasets import Dataset
from ..evals.reports import EvalReport
from .search import Candidate, EvaluateFn, FitnessWeights, _promotion_safe, fitness

__all__ = [
    "ObjectiveSpec",
    "DEFAULT_OBJECTIVES",
    "AGENTIC_OBJECTIVES",
    "objective_vector",
    "dominates",
    "ParetoPoint",
    "ParetoFrontier",
    "ParetoResult",
    "pareto_loop",
]


class ObjectiveSpec(BaseModel):
    """One axis of the multi-objective search."""

    name: str
    metric: str  # metric name aggregated from the EvalReport (mean)
    direction: Literal["max", "min"] = "max"
    scale: float = 1.0  # multiplier applied before comparison (e.g. ms → s)


DEFAULT_OBJECTIVES: list[ObjectiveSpec] = [
    ObjectiveSpec(name="accuracy", metric="semantic_similarity"),
    ObjectiveSpec(name="groundedness", metric="groundedness"),
    ObjectiveSpec(name="cost", metric="cost", direction="min"),
    ObjectiveSpec(name="latency_s", metric="latency", direction="min", scale=0.001),
]

# Frontier for agent optimization: trade goal success and tool correctness off
# against step count and cost. The 1.2 trajectory metrics are ordinary metrics,
# so they drop straight into the optimizer as fitness/Pareto terms.
AGENTIC_OBJECTIVES: list[ObjectiveSpec] = [
    ObjectiveSpec(name="goal", metric="goal_accuracy"),
    ObjectiveSpec(name="tools", metric="tool_call_accuracy"),
    ObjectiveSpec(name="efficiency", metric="step_efficiency"),
    ObjectiveSpec(name="cost", metric="cost", direction="min"),
]


def objective_vector(
    report: EvalReport, objectives: list[ObjectiveSpec] | None = None
) -> dict[str, float]:
    """Aggregate an eval report into one value per objective (means)."""
    objectives = objectives or DEFAULT_OBJECTIVES
    vector: dict[str, float] = {}
    for objective in objectives:
        values = report.metric_values(objective.metric)
        mean = sum(values) / len(values) if values else 0.0
        vector[objective.name] = round(mean * objective.scale, 6)
    return vector


def dominates(
    a: dict[str, float], b: dict[str, float], objectives: list[ObjectiveSpec]
) -> bool:
    """True when *a* is at least as good as *b* on every objective and
    strictly better on at least one."""
    strictly_better = False
    for objective in objectives:
        value_a, value_b = a.get(objective.name, 0.0), b.get(objective.name, 0.0)
        if objective.direction == "min":
            value_a, value_b = -value_a, -value_b
        if value_a < value_b - 1e-12:
            return False
        if value_a > value_b + 1e-12:
            strictly_better = True
    return strictly_better


class ParetoPoint(BaseModel):
    name: str
    objectives: dict[str, float] = Field(default_factory=dict)
    on_front: bool = False
    candidate: Any = None  # the Candidate this point was measured from


class ParetoFrontier(BaseModel):
    """All evaluated points plus the non-dominated subset."""

    specs: list[ObjectiveSpec] = Field(default_factory=list)
    points: list[ParetoPoint] = Field(default_factory=list)

    @classmethod
    def build(
        cls, points: list[ParetoPoint], *, specs: list[ObjectiveSpec] | None = None
    ) -> ParetoFrontier:
        specs = specs or DEFAULT_OBJECTIVES
        for point in points:
            point.on_front = not any(
                dominates(other.objectives, point.objectives, specs)
                for other in points
                if other is not point
            )
        return cls(specs=specs, points=points)

    @property
    def front(self) -> list[ParetoPoint]:
        return [point for point in self.points if point.on_front]

    def _normalized_goodness(self, point: ParetoPoint) -> float:
        """Sum of per-objective goodness normalized to [0, 1] across points."""
        total = 0.0
        for spec in self.specs:
            values = [p.objectives.get(spec.name, 0.0) for p in self.points]
            low, high = min(values), max(values)
            value = point.objectives.get(spec.name, 0.0)
            normalized = (value - low) / (high - low) if high > low else 0.5
            total += normalized if spec.direction == "max" else 1.0 - normalized
        return total

    def knee(self) -> ParetoPoint | None:
        """The balanced point: best summed normalized goodness on the front."""
        front = self.front
        if not front:
            return None
        return max(front, key=self._normalized_goodness)

    def select(
        self,
        *,
        constraints: dict[str, float] | None = None,
        prefer: str | None = None,
    ) -> ParetoPoint | None:
        """Pick a front point under constraints.

        ``constraints`` bound objectives by name: an upper bound for "min"
        objectives (``{"cost": 0.01}`` = at most a cent per case) and a lower
        bound for "max" objectives (``{"accuracy": 0.8}``). ``prefer`` names
        the objective to maximize/minimize among feasible points; without it
        the knee of the feasible subset wins.
        """
        directions = {spec.name: spec.direction for spec in self.specs}
        feasible = []
        for point in self.front:
            ok = True
            for name, bound in (constraints or {}).items():
                value = point.objectives.get(name, 0.0)
                if directions.get(name, "max") == "min":
                    ok = ok and value <= bound + 1e-12
                else:
                    ok = ok and value >= bound - 1e-12
            if ok:
                feasible.append(point)
        if not feasible:
            return None
        if prefer is not None:
            reverse = directions.get(prefer, "max") == "max"
            return sorted(
                feasible, key=lambda p: p.objectives.get(prefer, 0.0), reverse=reverse
            )[0]
        subset = ParetoFrontier.build(
            [point.model_copy() for point in feasible], specs=self.specs
        )
        knee = subset.knee()
        if knee is None:
            return None
        return next((p for p in feasible if p.name == knee.name), None)


class ParetoResult(BaseModel):
    baseline: ParetoPoint
    frontier: ParetoFrontier
    best: ParetoPoint | None = None
    promoted: bool = False
    reason: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)


async def pareto_loop(
    candidates: list[Candidate],
    evaluate_fn: EvaluateFn,
    dataset: Dataset,
    *,
    baseline: Candidate,
    objectives: list[ObjectiveSpec] | None = None,
    weights: FitnessWeights | None = None,
    subset_size: int = 16,
    top_n: int = 4,
    gates: dict[str, str] | None = None,
    max_cost_per_case: float | None = None,
    constraints: dict[str, float] | None = None,
    prefer: str | None = None,
    min_dataset_coverage: int = 4,
) -> ParetoResult:
    """Evolution loop with a multi-objective selection stage.

    Screening still uses scalar fitness (cheap), but the final pick comes
    from the Pareto frontier of full-dataset reports: the knee point, or
    the ``prefer``/``constraints`` selection. Promotion goes through the
    same safety rules as the scalar loop — a frontier point that regresses
    safety or fails a gate never wins.
    """
    objectives = objectives or DEFAULT_OBJECTIVES
    weights = weights or FitnessWeights()
    if len(dataset) < min_dataset_coverage:
        empty = ParetoPoint(name=baseline.name)
        return ParetoResult(
            baseline=empty,
            frontier=ParetoFrontier(specs=objectives),
            reason=(
                f"dataset too small ({len(dataset)} cases < {min_dataset_coverage}); "
                "refusing to optimize"
            ),
        )
    subset = dataset.sample(subset_size)

    baseline.full_report = await evaluate_fn(baseline, dataset)
    baseline.full_fitness = fitness(baseline.full_report, weights)
    baseline_point = ParetoPoint(
        name=baseline.name,
        objectives=objective_vector(baseline.full_report, objectives),
        candidate=baseline,
    )
    history: list[dict[str, Any]] = [
        {"phase": "baseline", "name": baseline.name, "objectives": baseline_point.objectives}
    ]

    for candidate in candidates:
        if candidate.subset_fitness is None:
            candidate.subset_report = await evaluate_fn(candidate, subset)
            candidate.subset_fitness = fitness(candidate.subset_report, weights)
        history.append(
            {"phase": "subset", "name": candidate.name, "fitness": candidate.subset_fitness}
        )
    survivors = sorted(
        candidates, key=lambda c: c.subset_fitness or float("-inf"), reverse=True
    )[:top_n]

    points: list[ParetoPoint] = [baseline_point]
    for candidate in survivors:
        candidate.full_report = await evaluate_fn(candidate, dataset)
        candidate.full_fitness = fitness(candidate.full_report, weights)
        point = ParetoPoint(
            name=candidate.name,
            objectives=objective_vector(candidate.full_report, objectives),
            candidate=candidate,
        )
        points.append(point)
        history.append({"phase": "full", "name": candidate.name, "objectives": point.objectives})

    frontier = ParetoFrontier.build(points, specs=objectives)
    pick = frontier.select(constraints=constraints, prefer=prefer)
    result = ParetoResult(
        baseline=baseline_point, frontier=frontier, best=pick, history=history
    )
    if pick is None:
        result.reason = "no frontier point satisfies the constraints"
        return result
    if pick.name == baseline.name:
        result.best = None
        result.reason = "baseline is the selected frontier point"
        return result
    candidate = pick.candidate
    if not dominates(pick.objectives, baseline_point.objectives, objectives) and (
        (candidate.full_fitness or float("-inf")) <= (baseline.full_fitness or 0.0)
    ):
        result.reason = "selected point neither dominates the baseline nor improves fitness"
        return result
    safe, reason = _promotion_safe(
        candidate.full_report,
        baseline.full_report,
        gates=gates,
        max_cost_per_case=max_cost_per_case,
    )
    result.promoted = safe
    result.reason = (
        reason
        if not safe
        else f"promoted {pick.name} from the frontier ({len(frontier.front)} non-dominated points)"
    )
    return result
