"""Prompt optimization: search prompt variants by fitness.

Strategies covered via variant generation: instruction rewrite hooks,
format selection (markdown/xml/json/minimal), example selection counts,
rule reordering, and reasoning-mode selection.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..evals.datasets import Dataset
from ..evals.reports import EvalReport
from ..prompts.compiler import RenderFormat
from ..prompts.optimizers import PromptVariant, generate_variants
from ..prompts.templates import PromptSpec
from .search import Candidate, FitnessWeights, OptimizationResult, evolution_loop

__all__ = ["PromptOptimizer"]

# The app supplies this: evaluate a prompt variant over a dataset.
VariantEvaluateFn = Callable[[PromptVariant, Dataset], Awaitable[EvalReport]]


class PromptOptimizer:
    def __init__(
        self,
        evaluate_variant: VariantEvaluateFn,
        *,
        weights: FitnessWeights | None = None,
        gates: dict[str, str] | None = None,
        max_cost_per_case: float | None = None,
    ) -> None:
        self.evaluate_variant = evaluate_variant
        self.weights = weights
        self.gates = gates
        self.max_cost_per_case = max_cost_per_case

    async def optimize(
        self,
        spec: PromptSpec,
        dataset: Dataset,
        *,
        formats: list[RenderFormat] | None = None,
        example_counts: list[int] | None = None,
        reasoning_modes: list[str] | None = None,
        max_variants: int = 12,
        subset_size: int = 16,
        top_n: int = 3,
        rewrites: list[PromptSpec] | None = None,
    ) -> OptimizationResult:
        from ..prompts.compiler import CompilerOptions

        variants = generate_variants(
            spec,
            formats=formats,
            example_counts=example_counts,
            reasoning_modes=reasoning_modes,
            max_variants=max_variants,
        )
        # Instruction-rewrite candidates supplied by the caller (or an LLM).
        for index, rewritten in enumerate(rewrites or []):
            variants.append(
                PromptVariant(
                    name=f"{spec.name}:rewrite{index}",
                    spec=rewritten,
                    compiler_options=CompilerOptions(),
                    dimensions={"rewrite": index},
                )
            )

        baseline = Candidate(
            name=f"{spec.name}:baseline",
            payload=PromptVariant(
                name=f"{spec.name}:baseline",
                spec=spec,
                compiler_options=CompilerOptions(),
                dimensions={},
            ),
        )
        candidates = [
            Candidate(name=v.name, params=v.dimensions, payload=v) for v in variants
        ]

        def evaluate(candidate, ds):
            return self.evaluate_variant(candidate.payload, ds)

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
