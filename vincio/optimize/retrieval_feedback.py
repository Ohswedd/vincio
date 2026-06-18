"""Retrieval feedback: eval-scored relevance tunes retrieval automatically.

Closes the retrieval half of the loop: relevance labels that already
live on eval cases (``rubric.relevant_ids``) drive deterministic, bounded
searches over the knobs retrieval exposes — per-index RRF fusion weights and
the heuristic reranker's blend — and a chunking recommendation picks the
chunking config whose eval report scored best. Every tuner is gated: weights
only change when the tuned configuration measurably beats the current one,
and the engine is restored untouched otherwise.
"""

from __future__ import annotations

import itertools
from typing import Any

from pydantic import BaseModel, Field

from ..evals.datasets import Dataset
from ..evals.reports import EvalReport
from ..retrieval.engine import RetrievalEngine
from ..retrieval.rerankers import HeuristicReranker

__all__ = [
    "RelevanceRecord",
    "records_from_dataset",
    "records_from_report",
    "RetrievalFeedbackResult",
    "RetrievalFeedback",
    "ChunkingRecommendation",
    "recommend_chunking",
]


class RelevanceRecord(BaseModel):
    """One labelled query: which chunks should come back."""

    query: str
    relevant_ids: list[str]
    case_id: str = ""
    observed: dict[str, float] = Field(default_factory=dict)  # metric values seen in evals


def _relevant_ids(case: Any) -> list[str]:
    ids = case.rubric.get("relevant_ids") or case.context.get("relevant_ids") or []
    return [str(item) for item in ids]


def records_from_dataset(dataset: Dataset) -> list[RelevanceRecord]:
    """Relevance records from every case that carries relevance labels."""
    records = []
    for case in dataset.cases:
        ids = _relevant_ids(case)
        if ids:
            records.append(
                RelevanceRecord(query=case.input_text, relevant_ids=ids, case_id=case.id)
            )
    return records


def records_from_report(report: EvalReport, dataset: Dataset) -> list[RelevanceRecord]:
    """Relevance records for the cases an eval report measured, carrying the
    observed retrieval metrics so feedback can prioritize weak queries."""
    measured = {result.case_id: result.metrics for result in report.cases}
    records = []
    for record in records_from_dataset(dataset):
        metrics = measured.get(record.case_id)
        if metrics is None:
            continue
        record.observed = {
            name: value
            for name, value in metrics.items()
            if name in ("recall_at_k", "precision_at_k", "mrr", "ndcg")
        }
        records.append(record)
    return records


class RetrievalFeedbackResult(BaseModel):
    baseline_score: float
    tuned_score: float
    applied: bool = False
    reason: str = ""
    index_weights_before: list[float] = Field(default_factory=list)
    index_weights_after: list[float] = Field(default_factory=list)
    reranker_weights_before: dict[str, float] = Field(default_factory=dict)
    reranker_weights_after: dict[str, float] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalFeedback:
    """Tunes a live :class:`RetrievalEngine` from relevance labels.

    The search is deterministic (fixed grids, no randomness) and gated:
    changes apply only when the tuned score beats the baseline by
    ``min_improvement``; otherwise the engine keeps its current weights.
    """

    _INDEX_WEIGHT_GRID = (0.5, 1.0, 1.5, 2.0)
    _RERANKER_GRID = {
        "weight_retrieval": (0.3, 0.5, 0.7),
        "weight_lexical": (0.15, 0.35, 0.5),
        "weight_structure": (0.0, 0.15, 0.3),
    }

    def __init__(
        self,
        engine: RetrievalEngine,
        records: list[RelevanceRecord],
        *,
        top_k: int = 8,
        min_improvement: float = 1e-6,
    ) -> None:
        if not records:
            raise ValueError("RetrievalFeedback needs at least one relevance record")
        self.engine = engine
        self.records = records
        self.top_k = top_k
        self.min_improvement = min_improvement

    async def score(self) -> float:
        """Mean of recall@k and MRR over the records, with current weights."""
        totals = 0.0
        for record in self.records:
            result = await self.engine.retrieve(record.query, top_k=self.top_k)
            relevant = set(record.relevant_ids)
            hits = 0
            reciprocal = 0.0
            for rank, item in enumerate(result.evidence, start=1):
                refs = {item.id, item.citation_ref, str(item.metadata.get("chunk_id"))}
                if refs & relevant:
                    hits += 1
                    if reciprocal == 0.0:
                        reciprocal = 1.0 / rank
            recall = hits / len(relevant) if relevant else 0.0
            totals += 0.5 * recall + 0.5 * reciprocal
        return round(totals / len(self.records), 6)

    async def tune_index_weights(self, *, rounds: int = 2) -> RetrievalFeedbackResult:
        """Coordinate search over per-index RRF fusion weights."""
        before = list(self.engine.index_weights)
        baseline = await self.score()
        best_score = baseline
        history: list[dict[str, Any]] = []
        for _round in range(rounds):
            improved = False
            for position in range(len(self.engine.index_weights)):
                current = self.engine.index_weights[position]
                for weight in self._INDEX_WEIGHT_GRID:
                    if weight == current:
                        continue
                    self.engine.index_weights[position] = weight
                    score = await self.score()
                    history.append(
                        {"index": position, "weight": weight, "score": score}
                    )
                    if score > best_score + self.min_improvement:
                        best_score, current, improved = score, weight, True
                    else:
                        self.engine.index_weights[position] = current
            if not improved:
                break
        applied = best_score > baseline + self.min_improvement
        if not applied:
            self.engine.index_weights = before
        return RetrievalFeedbackResult(
            baseline_score=baseline,
            tuned_score=best_score,
            applied=applied,
            reason="index weights tuned" if applied else "no improvement; weights unchanged",
            index_weights_before=before,
            index_weights_after=list(self.engine.index_weights),
            history=history,
        )

    async def tune_reranker(self) -> RetrievalFeedbackResult:
        """Grid search over the heuristic reranker's blend weights."""
        reranker = self.engine.reranker
        if not isinstance(reranker, HeuristicReranker):
            return RetrievalFeedbackResult(
                baseline_score=0.0,
                tuned_score=0.0,
                reason="engine has no HeuristicReranker to tune",
            )
        before = {
            "weight_retrieval": reranker.weight_retrieval,
            "weight_lexical": reranker.weight_lexical,
            "weight_structure": reranker.weight_structure,
        }
        baseline = await self.score()
        best_score, best_weights = baseline, dict(before)
        history: list[dict[str, Any]] = []
        names = list(self._RERANKER_GRID)
        for combo in itertools.product(*(self._RERANKER_GRID[name] for name in names)):
            weights = dict(zip(names, combo, strict=True))
            if weights == before:
                continue
            for name, value in weights.items():
                setattr(reranker, name, value)
            score = await self.score()
            history.append({**weights, "score": score})
            if score > best_score + self.min_improvement:
                best_score, best_weights = score, dict(weights)
        applied = best_score > baseline + self.min_improvement
        final = best_weights if applied else before
        for name, value in final.items():
            setattr(reranker, name, value)
        return RetrievalFeedbackResult(
            baseline_score=baseline,
            tuned_score=best_score if applied else baseline,
            applied=applied,
            reason="reranker weights tuned" if applied else "no improvement; weights unchanged",
            reranker_weights_before=before,
            reranker_weights_after=final,
            history=history,
        )

    async def tune(self) -> RetrievalFeedbackResult:
        """Tune fusion weights, then the reranker blend, in one gated pass."""
        index_result = await self.tune_index_weights()
        reranker_result = await self.tune_reranker()
        return RetrievalFeedbackResult(
            baseline_score=index_result.baseline_score,
            tuned_score=max(index_result.tuned_score, reranker_result.tuned_score),
            applied=index_result.applied or reranker_result.applied,
            reason="; ".join(
                part for part in (index_result.reason, reranker_result.reason) if part
            ),
            index_weights_before=index_result.index_weights_before,
            index_weights_after=index_result.index_weights_after,
            reranker_weights_before=reranker_result.reranker_weights_before,
            reranker_weights_after=reranker_result.reranker_weights_after,
            history=[*index_result.history, *reranker_result.history],
        )


class ChunkingRecommendation(BaseModel):
    baseline: str
    recommended: str
    improvement: float = 0.0
    metric: str = "recall_at_k"
    scores: dict[str, float] = Field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.recommended != self.baseline


def recommend_chunking(
    reports_by_config: dict[str, EvalReport],
    *,
    baseline: str,
    metric: str = "recall_at_k",
    min_improvement: float = 0.0,
) -> ChunkingRecommendation:
    """Pick the chunking config whose eval report scored best.

    ``reports_by_config`` maps a chunking description (e.g.
    ``"recursive:400/50"``) to the eval report measured with it — typically
    produced by running the same dataset through
    :class:`~vincio.optimize.ContextOptimizer` candidates or an experiment
    tracker. The recommendation stays on ``baseline`` unless a config beats
    it by ``min_improvement``.
    """
    if baseline not in reports_by_config:
        raise ValueError(f"baseline config {baseline!r} missing from reports")
    scores: dict[str, float] = {}
    for name, report in reports_by_config.items():
        values = report.metric_values(metric)
        scores[name] = round(sum(values) / len(values), 6) if values else 0.0
    best = max(scores, key=lambda name: scores[name])
    if scores[best] <= scores[baseline] + min_improvement:
        best = baseline
    return ChunkingRecommendation(
        baseline=baseline,
        recommended=best,
        improvement=round(scores[best] - scores[baseline], 6),
        metric=metric,
        scores=scores,
    )
