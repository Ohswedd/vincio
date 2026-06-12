"""Context optimization: tune retrieval/context parameters.

Search space: chunk size, top-K, reranker selection, evidence ledger vs raw
chunks, memory threshold, compression. Random search over the grid (cheap,
parallelizable, no gradient assumptions), screened on a subset and verified
on the full dataset through the shared evolution loop.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from ..evals.datasets import Dataset
from ..evals.reports import EvalReport
from .search import Candidate, FitnessWeights, OptimizationResult, evolution_loop

__all__ = ["ContextSearchSpace", "ContextOptimizer"]


class ContextSearchSpace(BaseModel):
    top_k: list[int] = Field(default_factory=lambda: [4, 8, 12])
    chunk_size_tokens: list[int] = Field(default_factory=lambda: [200, 400, 800])
    chunk_overlap_tokens: list[int] = Field(default_factory=lambda: [0, 50])
    reranker: list[str | None] = Field(default_factory=lambda: ["heuristic", None])
    use_evidence_ledger: list[bool] = Field(default_factory=lambda: [False, True])
    compress_evidence: list[bool] = Field(default_factory=lambda: [True])
    memory_min_confidence: list[float] = Field(default_factory=lambda: [0.25, 0.5])
    ordering: list[str] = Field(default_factory=lambda: ["relevance", "boundary_sandwich"])

    def sample(self, n: int, *, seed: int = 7) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        space = self.model_dump()
        seen: set[tuple] = set()
        configs: list[dict[str, Any]] = []
        max_unique = 1
        for values in space.values():
            max_unique *= max(1, len(values))
        attempts = 0
        while len(configs) < min(n, max_unique) and attempts < n * 20:
            attempts += 1
            config = {key: rng.choice(values) for key, values in space.items() if values}
            key = tuple(sorted((k, str(v)) for k, v in config.items()))
            if key in seen:
                continue
            seen.add(key)
            configs.append(config)
        return configs


# evaluate_config(config, dataset) -> EvalReport; the app rebuilds its
# retrieval/context pipeline with the config and runs the dataset.
ConfigEvaluateFn = Callable[[dict[str, Any], Dataset], Awaitable[EvalReport]]


class ContextOptimizer:
    def __init__(
        self,
        evaluate_config: ConfigEvaluateFn,
        *,
        weights: FitnessWeights | None = None,
        gates: dict[str, str] | None = None,
        max_cost_per_case: float | None = None,
    ) -> None:
        self.evaluate_config = evaluate_config
        self.weights = weights
        self.gates = gates
        self.max_cost_per_case = max_cost_per_case

    async def optimize(
        self,
        dataset: Dataset,
        *,
        space: ContextSearchSpace | None = None,
        baseline_config: dict[str, Any] | None = None,
        budget: int = 12,
        subset_size: int = 16,
        top_n: int = 3,
        seed: int = 7,
    ) -> OptimizationResult:
        space = space or ContextSearchSpace()
        configs = space.sample(budget, seed=seed)
        baseline = Candidate(name="context:baseline", params=baseline_config or {}, payload=baseline_config or {})
        candidates = [
            Candidate(
                name="context:" + ",".join(f"{k}={v}" for k, v in sorted(config.items())),
                params=config,
                payload=config,
            )
            for config in configs
        ]
        def evaluate(candidate, ds):
            return self.evaluate_config(candidate.payload, ds)

        return await evolution_loop(
            candidates,
            evaluate,
            dataset,
            baseline=baseline,
            weights=self.weights,
            subset_size=subset_size,
            top_n=top_n,
            gates=self.gates,
            max_cost_per_case=self.max_cost_per_case,
        )
