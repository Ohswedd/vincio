"""Faithfulness-gated adoption of learned compression (1.4).

A learned compressor only earns its place on the compiler if it pays its way:
it must shrink the prompt **and** keep answer quality and the cited-fact set
intact. This module measures both against the eval suite and adopts the
compressor only when the gate holds — the same measured, gated discipline as
:mod:`vincio.optimize.budget_learning`. A compressor that drops evidence, or
trades quality for tokens, is rejected with the reason, and the compiler stays
on extractive compression.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from ..evals.datasets import Dataset
from ..evals.reports import EvalReport
from .search import _promotion_safe

__all__ = ["CompressionTuningResult", "CompressionTuner"]


# evaluate(compressor, dataset) -> EvalReport (the app rebuilds its compiler
# with the given compressor and runs the dataset). ``None`` selects the
# baseline (extractive) compressor.
CompressorEvaluateFn = Callable[[Any, Dataset], Awaitable[EvalReport]]


class CompressionTuningResult(BaseModel):
    adopted: bool = False
    reason: str = ""
    baseline_quality: float = 0.0
    learned_quality: float = 0.0
    quality_ratio: float = 0.0
    baseline_faithfulness: float = 1.0
    learned_faithfulness: float = 1.0
    baseline_tokens: float = 0.0
    learned_tokens: float = 0.0
    token_savings: float = 0.0  # fraction fewer input tokens than the baseline


class CompressionTuner:
    """Adopt a learned compressor only when it holds quality and faithfulness.

    ``evaluate`` runs the dataset with a given compressor and returns an
    :class:`EvalReport`. The report should carry the quality metric and, when
    available, a ``faithfulness`` metric (the fraction of cited facts preserved)
    and ``input_tokens`` — the tuner uses whatever is present and is
    conservative about what is missing.
    """

    def __init__(
        self,
        evaluate: CompressorEvaluateFn,
        *,
        quality_metric: str = "lexical_overlap",
        faithfulness_metric: str = "faithfulness",
        token_metric: str = "input_tokens",
        min_quality_ratio: float = 0.98,
        min_faithfulness: float = 0.9,
        gates: dict[str, str] | None = None,
    ) -> None:
        self.evaluate = evaluate
        self.quality_metric = quality_metric
        self.faithfulness_metric = faithfulness_metric
        self.token_metric = token_metric
        self.min_quality_ratio = min_quality_ratio
        self.min_faithfulness = min_faithfulness
        self.gates = gates

    async def tune(
        self, compressor: Any, dataset: Dataset, *, min_dataset_coverage: int = 4
    ) -> tuple[CompressionTuningResult, Any | None]:
        """Return the gate result and the compressor to install (or ``None``).

        ``None`` for the returned compressor means stay on the baseline.
        """
        result = CompressionTuningResult()
        if len(dataset) < min_dataset_coverage:
            result.reason = (
                f"dataset too small ({len(dataset)} cases < {min_dataset_coverage}); "
                "refusing to tune compression"
            )
            return result, None

        baseline_report = await self.evaluate(None, dataset)
        learned_report = await self.evaluate(compressor, dataset)

        result.baseline_quality = round(_mean(baseline_report, self.quality_metric), 4)
        result.learned_quality = round(_mean(learned_report, self.quality_metric), 4)
        result.baseline_faithfulness = round(_mean(baseline_report, self.faithfulness_metric, 1.0), 4)
        result.learned_faithfulness = round(_mean(learned_report, self.faithfulness_metric, 1.0), 4)
        result.baseline_tokens = round(_mean(baseline_report, self.token_metric), 2)
        result.learned_tokens = round(_mean(learned_report, self.token_metric), 2)
        result.quality_ratio = (
            round(result.learned_quality / result.baseline_quality, 4)
            if result.baseline_quality > 0
            else 0.0
        )
        if result.baseline_tokens > 0:
            result.token_savings = round(1.0 - result.learned_tokens / result.baseline_tokens, 4)

        if result.learned_faithfulness < self.min_faithfulness:
            result.reason = (
                f"faithfulness {result.learned_faithfulness:.2f} below floor "
                f"{self.min_faithfulness:.2f}; cited facts would be lost"
            )
            return result, None
        if result.baseline_quality > 0 and result.quality_ratio < self.min_quality_ratio:
            result.reason = (
                f"quality holds only {result.quality_ratio:.1%} (< {self.min_quality_ratio:.0%}); "
                "compression not worth the loss"
            )
            return result, None
        # A token saving must be *verifiable*: without an input-token signal we
        # cannot show the compressor is worth its risk, so we do not adopt.
        if result.baseline_tokens <= 0:
            result.reason = (
                "no token signal (the eval did not report input_tokens); cannot verify "
                "that learned compression saves tokens"
            )
            return result, None
        if result.learned_tokens >= result.baseline_tokens:
            result.reason = "learned compression saved no tokens; not adopting"
            return result, None
        safe, reason = _promotion_safe(
            learned_report, baseline_report, gates=self.gates, max_cost_per_case=None
        )
        if not safe:
            result.reason = reason
            return result, None

        result.adopted = True
        result.reason = (
            f"adopted learned compression: {result.token_savings:.0%} fewer tokens, "
            f"quality {result.quality_ratio:.1%}, faithfulness {result.learned_faithfulness:.2f}"
        )
        return result, compressor


def _mean(report: EvalReport, metric: str, default: float = 0.0) -> float:
    values = report.metric_values(metric)
    return sum(values) / len(values) if values else default
