"""Learned context budgeting.

Per-task budget allocation tuned from eval outcomes instead of the fixed
tables in :mod:`vincio.context.budgeting`. The learner perturbs a baseline
allocation in bounded steps (move a slice of the budget from a donor block
to a receiver block, renormalize), evaluates every candidate allocation
through the shared evolution loop, and only adopts a learned table when it
beats the baseline through the same safety gates as every other optimizer.
Learned tables persist as JSON and load back into a
:class:`~vincio.context.budgeting.BudgetAllocator`.
"""

from __future__ import annotations

import json
import random
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..context.budgeting import DEFAULT_ALLOCATION, BudgetAllocator
from ..core.types import TaskType
from ..evals.datasets import Dataset
from ..evals.reports import EvalReport
from .search import Candidate, FitnessWeights, OptimizationResult, evolution_loop

__all__ = ["LearnedAllocations", "BudgetLearner"]

# evaluate_allocation(fractions, dataset) -> EvalReport; the app re-runs the
# dataset with the candidate allocation installed on its context compiler.
AllocationEvaluateFn = Callable[[dict[str, float], Dataset], Awaitable[EvalReport]]


class LearnedAllocations(BaseModel):
    """Per-task allocation tables learned from eval outcomes."""

    allocations: dict[str, dict[str, float]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def set(self, task_type: TaskType | str, fractions: dict[str, float]) -> None:
        key = task_type.value if isinstance(task_type, TaskType) else str(task_type)
        total = sum(fractions.values()) or 1.0
        self.allocations[key] = {k: round(v / total, 6) for k, v in fractions.items()}

    def get(self, task_type: TaskType | str) -> dict[str, float] | None:
        key = task_type.value if isinstance(task_type, TaskType) else str(task_type)
        fractions = self.allocations.get(key)
        return dict(fractions) if fractions else None

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> LearnedAllocations:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def allocator(self) -> BudgetAllocator:
        return BudgetAllocator(learned=self.allocations)


def _perturb(
    fractions: dict[str, float], rng: random.Random, *, step: float
) -> dict[str, float]:
    """Move *step* of the budget from one block to another, renormalized."""
    blocks = list(fractions)
    donors = [name for name in blocks if fractions[name] >= step] or blocks
    donor = rng.choice(donors)
    receiver = rng.choice([name for name in blocks if name != donor])
    candidate = dict(fractions)
    candidate[donor] = max(0.0, candidate[donor] - step)
    candidate[receiver] = candidate[receiver] + step
    total = sum(candidate.values()) or 1.0
    return {name: round(value / total, 6) for name, value in candidate.items()}


class BudgetLearner:
    """Tunes per-task budget allocation from eval outcomes (0.8)."""

    def __init__(
        self,
        evaluate_allocation: AllocationEvaluateFn,
        *,
        weights: FitnessWeights | None = None,
        gates: dict[str, str] | None = None,
        max_cost_per_case: float | None = None,
    ) -> None:
        self.evaluate_allocation = evaluate_allocation
        self.weights = weights
        self.gates = gates
        self.max_cost_per_case = max_cost_per_case

    async def learn(
        self,
        dataset: Dataset,
        *,
        task_type: TaskType = TaskType.GENERAL,
        baseline: dict[str, float] | None = None,
        candidates: int = 8,
        step: float = 0.05,
        subset_size: int = 8,
        top_n: int = 3,
        seed: int = 7,
    ) -> tuple[OptimizationResult, LearnedAllocations | None]:
        """Search bounded perturbations of the baseline allocation.

        Returns the optimization result and, when a candidate was promoted,
        a :class:`LearnedAllocations` holding the winning table for
        ``task_type`` — ``None`` when the baseline stands.
        """
        base = baseline or BudgetAllocator().allocation_for(task_type) or dict(DEFAULT_ALLOCATION)
        rng = random.Random(seed)
        seen: set[tuple] = {tuple(sorted(base.items()))}
        proposals: list[dict[str, float]] = []
        attempts = 0
        while len(proposals) < candidates and attempts < candidates * 20:
            attempts += 1
            knobs = 1 + rng.randrange(2)
            candidate = dict(base)
            for _ in range(knobs):
                candidate = _perturb(candidate, rng, step=step)
            key = tuple(sorted(candidate.items()))
            if key in seen:
                continue
            seen.add(key)
            proposals.append(candidate)

        def _name(fractions: dict[str, float]) -> str:
            top = sorted(fractions.items(), key=lambda kv: kv[1], reverse=True)[:3]
            return "budget:" + ",".join(f"{k}={v:.2f}" for k, v in top)

        baseline_candidate = Candidate(name="budget:baseline", params=base, payload=base)
        candidate_list = [
            Candidate(name=_name(fractions), params=fractions, payload=fractions)
            for fractions in proposals
        ]

        def evaluate(candidate, ds):
            return self.evaluate_allocation(candidate.payload, ds)

        result = await evolution_loop(
            candidate_list,
            evaluate,
            dataset,
            baseline=baseline_candidate,
            weights=self.weights,
            subset_size=subset_size,
            top_n=top_n,
            gates=self.gates,
            max_cost_per_case=self.max_cost_per_case,
        )
        learned: LearnedAllocations | None = None
        if result.promoted and result.best is not None:
            learned = LearnedAllocations(
                metadata={
                    "dataset": dataset.name,
                    "task_type": task_type.value,
                    "baseline_fitness": result.baseline_fitness,
                    "fitness": result.best.full_fitness,
                }
            )
            learned.set(task_type, dict(result.best.payload))
        return result, learned
