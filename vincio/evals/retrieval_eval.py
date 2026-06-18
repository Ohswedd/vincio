"""Retrieval evaluation harness + index-version regression.

A golden-set harness scoped to **retrieval**: it quantitatively benchmarks an
embedder / reranker / chunker / index configuration against a fixed query set on
the standard IR metrics (**recall@k**, **nDCG@k**, **MRR**, **precision@k**,
**context-precision**), reusing the metric implementations in
:mod:`vincio.evals.metrics`. Results render through :class:`EvalReport`, so the
offline-vs-real retrieval trade-offs become measurable instead of a vibe check.

The harness also closes the *regression* loop. Each evaluation is recorded as a
versioned artifact keyed on ``(embedder, chunker, corpus hash)`` (see
:mod:`vincio.storage.index_regression`); a later run on the same golden set is
compared against the stored baseline and **gated on recall/nDCG deltas using the
same significance machinery a model swap is gated on** (:func:`ab_test`). A
re-embed or chunking tweak that drops recall is caught as a first-class CI gate,
not discovered in production.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from ..core.types import EvidenceItem
from ..providers.base import run_sync
from ..storage.index_regression import IndexRegressionArtifact, IndexRegressionStore
from .datasets import EvalCase
from .experiments import ab_test
from .metrics import METRICS, RunOutput
from .reports import CaseResult, EvalReport, evaluate_gates

__all__ = [
    "RetrievalQuery",
    "RetrievalGoldenSet",
    "RetrievalConfig",
    "RetrievalEvaluator",
    "RetrievalRegressionVerdict",
    "retrieval_regression",
    "as_search_fn",
    "DEFAULT_K_VALUES",
]

# A search function: given a query and a depth, return ranked evidence. May be
# sync or async. The harness builds RunOutputs from the returned evidence so the
# existing retrieval metrics score it without re-instrumentation.
SearchFn = Callable[[str, int], "list[EvidenceItem] | Awaitable[list[EvidenceItem]]"]

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


class RetrievalQuery(BaseModel):
    """One golden query: the text plus the ids of the relevant corpus items."""

    id: str
    query: str
    relevant_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class RetrievalGoldenSet(BaseModel):
    """A fixed query set scored against a fixed corpus."""

    name: str = "retrieval_golden"
    queries: list[RetrievalQuery] = Field(default_factory=list)
    corpus_hash: str = ""

    @staticmethod
    def corpus_hash_of(items: list[Any]) -> str:
        """A stable hash over the corpus identity (``id`` + scorable text).

        Re-embedding the *same* corpus keeps the hash (so runs compare); editing
        the corpus changes it (so a new regression lineage starts).
        """
        parts: list[str] = []
        for item in items:
            ident = getattr(item, "id", None) or ""
            text = getattr(item, "scorable_text", None) or getattr(item, "text", "") or ""
            parts.append(f"{ident}:{text}")
        blob = "\n".join(parts)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class RetrievalConfig(BaseModel):
    """The identity of a retrieval pipeline, for regression keying."""

    embedder: str
    chunker: str = "default"
    reranker: str = ""
    index: str = ""

    def key(self, corpus_hash: str) -> str:
        return IndexRegressionArtifact.key_for(
            self.embedder, self.chunker, corpus_hash, reranker=self.reranker, index=self.index
        )


def as_search_fn(retriever: Any) -> SearchFn:
    """Adapt a :class:`RetrievalEngine` / :class:`Index` into a ``search_fn``.

    Accepts anything exposing ``retrieve(query, top_k=...)`` (returns a result
    with ``.evidence``) or ``search(query, top_k=...)`` (returns ranked hits with
    ``.chunk``).
    """

    async def search(query: str, top_k: int) -> list[EvidenceItem]:
        if hasattr(retriever, "retrieve"):
            result = retriever.retrieve(query, top_k=top_k)
            if hasattr(result, "__await__"):
                result = await result
            return list(getattr(result, "evidence", result))
        hits = retriever.search(query, top_k=top_k)
        if hasattr(hits, "__await__"):
            hits = await hits
        out: list[EvidenceItem] = []
        for hit in hits:
            chunk = getattr(hit, "chunk", hit)
            out.append(
                EvidenceItem(
                    id=str(getattr(chunk, "id", "") or ""),
                    source_id=str(getattr(chunk, "source_id", None) or getattr(chunk, "id", None) or "corpus"),
                    text=getattr(chunk, "text", "") or "",
                    relevance=float(getattr(hit, "score", 0.0) or 0.0),
                )
            )
        return out

    return search


class RetrievalEvaluator:
    """Score a retriever against a :class:`RetrievalGoldenSet` on the IR metrics."""

    def __init__(self, *, k_values: tuple[int, ...] = DEFAULT_K_VALUES) -> None:
        self.k_values = tuple(sorted(set(k_values)))

    async def aevaluate(
        self, search_fn: SearchFn, golden: RetrievalGoldenSet, *, top_k: int | None = None,
        name: str | None = None,
    ) -> EvalReport:
        depth = top_k or max(self.k_values)
        cases: list[CaseResult] = []
        for q in golden.queries:
            retrieved = search_fn(q.query, depth)
            if hasattr(retrieved, "__await__"):
                retrieved = await retrieved  # type: ignore[assignment]
            evidence = list(retrieved)
            case = EvalCase(id=q.id, input=q.query, rubric={"relevant_ids": q.relevant_ids}, tags=q.tags)
            metrics: dict[str, float] = {}
            # recall@k / nDCG@k at each cutoff (truncate the retrieved set to k).
            for k in self.k_values:
                run_k = RunOutput(evidence=evidence[:k])
                metrics[f"recall_at_{k}"] = METRICS["recall_at_k"](case, run_k).value
                metrics[f"ndcg_at_{k}"] = METRICS["ndcg"](case, run_k).value
            # Rank-order and precision metrics over the full retrieved set.
            run_full = RunOutput(evidence=evidence)
            metrics["mrr"] = METRICS["mrr"](case, run_full).value
            metrics["precision_at_k"] = METRICS["precision_at_k"](case, run_full).value
            metrics["context_precision"] = METRICS["context_precision"](case, run_full).value
            cases.append(CaseResult(case_id=q.id, metrics=metrics, tags=q.tags))
        return EvalReport(
            name=name or golden.name,
            dataset=golden.name,
            cases=cases,
            metadata={"corpus_hash": golden.corpus_hash, "k_values": list(self.k_values)},
        )

    def evaluate(self, search_fn: SearchFn, golden: RetrievalGoldenSet, **kwargs: Any) -> EvalReport:
        async def _async_search(query: str, top_k: int) -> list[EvidenceItem]:
            out = search_fn(query, top_k)
            if hasattr(out, "__await__"):
                return await out  # type: ignore[return-value]
            return list(out)

        return run_sync(self.aevaluate(_async_search, golden, **kwargs))


class RetrievalRegressionVerdict(BaseModel):
    """The outcome of comparing a fresh retrieval evaluation to its baseline."""

    passed: bool
    key: str
    is_baseline: bool = False
    current: dict[str, float] = Field(default_factory=dict)
    baseline: dict[str, float] = Field(default_factory=dict)
    deltas: dict[str, float] = Field(default_factory=dict)
    regressions: list[str] = Field(default_factory=list)
    significance: dict[str, Any] = Field(default_factory=dict)
    gates: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "key": self.key,
            "regressions": self.regressions,
            "deltas": self.deltas,
        }


async def retrieval_regression(
    search_fn: SearchFn,
    golden: RetrievalGoldenSet,
    config: RetrievalConfig,
    *,
    store: IndexRegressionStore | None = None,
    metrics: tuple[str, ...] = ("recall_at_3", "ndcg_at_5"),
    gates: dict[str, str] | None = None,
    top_k: int | None = None,
    alpha: float = 0.05,
    min_delta: float = 0.0,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> RetrievalRegressionVerdict:
    """Evaluate ``config`` on ``golden``, record an artifact, and gate vs. baseline.

    A metric counts as a regression when its mean drops by more than ``min_delta``
    **and** the drop is statistically significant (:func:`ab_test`, ``p < alpha``)
    against the recorded baseline — the same significance bar a model swap clears.
    Optional absolute ``gates`` (e.g. ``{"recall_at_3": ">= 0.6"}``) also apply.
    """
    store = store or IndexRegressionStore()
    evaluator = RetrievalEvaluator(k_values=k_values)
    report = await evaluator.aevaluate(search_fn, golden, top_k=top_k)
    key = config.key(golden.corpus_hash)

    summary = report.summary()
    current_means = {m: round(s["mean"], 6) for m, s in summary.items()}

    gate_results = evaluate_gates(report, gates) if gates else {}
    gates_pass = all(g["passed"] for g in gate_results.values()) if gate_results else True

    baseline_artifact = store.baseline(key)

    artifact = IndexRegressionArtifact(
        key=key,
        embedder=config.embedder,
        chunker=config.chunker,
        reranker=config.reranker,
        index=config.index,
        corpus_hash=golden.corpus_hash,
        n_queries=len(golden.queries),
        metrics=current_means,
        report=report.model_dump(mode="json"),
    )
    store.record(artifact)

    if baseline_artifact is None:
        return RetrievalRegressionVerdict(
            passed=gates_pass,
            key=key,
            is_baseline=True,
            current=current_means,
            gates=gate_results,
            reason="no baseline recorded; stored first measurement"
            + ("" if gates_pass else "; absolute gates failed"),
        )

    baseline_report = EvalReport.model_validate(baseline_artifact.report)
    deltas: dict[str, float] = {}
    regressions: list[str] = []
    significance: dict[str, Any] = {}
    for metric in metrics:
        cur = current_means.get(metric)
        base = baseline_artifact.metrics.get(metric)
        if cur is None or base is None:
            continue
        delta = round(cur - base, 6)
        deltas[metric] = delta
        test = ab_test(baseline_report, report, metric, alpha=alpha)
        significance[metric] = {
            "p_value": test.get("p_value"),
            "significant": test.get("significant"),
            "delta": delta,
        }
        if delta < -min_delta and test.get("significant"):
            regressions.append(metric)

    passed = not regressions and gates_pass
    reason = "no significant retrieval regression" if not regressions else f"regressions: {regressions}"
    if not gates_pass:
        reason += "; absolute gates failed"
    return RetrievalRegressionVerdict(
        passed=passed,
        key=key,
        current=current_means,
        baseline=dict(baseline_artifact.metrics),
        deltas=deltas,
        regressions=regressions,
        significance=significance,
        gates=gate_results,
        reason=reason,
    )
