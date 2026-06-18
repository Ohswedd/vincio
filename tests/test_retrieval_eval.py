"""Tests for the retrieval evaluation harness + index-version regression."""

from __future__ import annotations

import pytest

from vincio.core.types import EvidenceItem
from vincio.evals import (
    RetrievalConfig,
    RetrievalEvaluator,
    RetrievalGoldenSet,
    RetrievalQuery,
    retrieval_regression,
)
from vincio.storage.index_regression import IndexRegressionStore, config_key

# A tiny deterministic corpus: each doc is a bag of words.
CORPUS = {
    "d0": "refund policy window thirty days",
    "d1": "shipping address change order",
    "d2": "cancel order before dispatch",
    "d3": "warranty coverage twelve months",
    "d4": "password reset email link",
    "d5": "loyalty points redemption rewards",
    "d6": "invoice download tax pdf",
    "d7": "subscription renewal billing cycle",
}

QUERIES = [
    RetrievalQuery(id="q0", query="how long is the refund window", relevant_ids=["d0"]),
    RetrievalQuery(id="q1", query="change my shipping address", relevant_ids=["d1"]),
    RetrievalQuery(id="q2", query="cancel an order", relevant_ids=["d2"]),
    RetrievalQuery(id="q3", query="warranty coverage length", relevant_ids=["d3"]),
    RetrievalQuery(id="q4", query="reset my password", relevant_ids=["d4"]),
    RetrievalQuery(id="q5", query="redeem loyalty points", relevant_ids=["d5"]),
    RetrievalQuery(id="q6", query="download my invoice pdf", relevant_ids=["d6"]),
    RetrievalQuery(id="q7", query="subscription billing cycle", relevant_ids=["d7"]),
]


def _golden() -> RetrievalGoldenSet:
    corpus_items = [EvidenceItem(id=cid, source_id=cid, text=text) for cid, text in CORPUS.items()]
    return RetrievalGoldenSet(
        name="support_corpus",
        queries=QUERIES,
        corpus_hash=RetrievalGoldenSet.corpus_hash_of(corpus_items),
    )


def lexical_search(query: str, top_k: int) -> list[EvidenceItem]:
    """A decent retriever: rank by word overlap with the query."""
    q_words = set(query.lower().split())
    scored = sorted(
        CORPUS.items(),
        key=lambda kv: len(q_words & set(kv[1].split())),
        reverse=True,
    )
    return [EvidenceItem(id=cid, source_id=cid, text=text) for cid, text in scored[:top_k]]


def degraded_search(query: str, top_k: int) -> list[EvidenceItem]:
    """A regressed retriever: ignores the query, returns a fixed prefix."""
    fixed = list(CORPUS.items())[:top_k]
    return [EvidenceItem(id=cid, source_id=cid, text=text) for cid, text in fixed]


def test_evaluator_computes_ir_metrics():
    report = RetrievalEvaluator().evaluate(lexical_search, _golden(), top_k=5)
    summary = report.summary()
    assert "recall_at_3" in summary and "ndcg_at_5" in summary and "mrr" in summary
    assert summary["recall_at_3"]["mean"] == 1.0  # the right doc is always top-ranked


def test_config_key_changes_with_embedder_or_corpus():
    a = config_key("e1", "c1", "corpusA")
    assert a == config_key("e1", "c1", "corpusA")  # stable
    assert a != config_key("e2", "c1", "corpusA")  # embedder change
    assert a != config_key("e1", "c1", "corpusB")  # corpus change


@pytest.mark.asyncio
async def test_first_run_is_baseline():
    store = IndexRegressionStore()
    verdict = await retrieval_regression(
        lexical_search, _golden(), RetrievalConfig(embedder="hash", chunker="fixed"), store=store
    )
    assert verdict.is_baseline is True
    assert verdict.passed is True
    assert store.baseline(verdict.key) is not None


@pytest.mark.asyncio
async def test_stable_rerun_passes():
    store = IndexRegressionStore()
    config = RetrievalConfig(embedder="hash", chunker="fixed")
    await retrieval_regression(lexical_search, _golden(), config, store=store)
    verdict = await retrieval_regression(lexical_search, _golden(), config, store=store)
    assert verdict.is_baseline is False
    assert verdict.passed is True
    assert verdict.regressions == []


@pytest.mark.asyncio
async def test_recall_regression_is_caught():
    store = IndexRegressionStore()
    config = RetrievalConfig(embedder="hash", chunker="fixed")
    # Baseline = good retriever; candidate = degraded retriever.
    await retrieval_regression(lexical_search, _golden(), config, store=store)
    verdict = await retrieval_regression(
        degraded_search, _golden(), config, store=store, metrics=("recall_at_3", "ndcg_at_5")
    )
    assert verdict.passed is False
    assert "recall_at_3" in verdict.regressions
    assert verdict.deltas["recall_at_3"] < 0
    assert verdict.significance["recall_at_3"]["significant"] is True


@pytest.mark.asyncio
async def test_absolute_gate_failure_marks_not_passed():
    store = IndexRegressionStore()
    verdict = await retrieval_regression(
        degraded_search,
        _golden(),
        RetrievalConfig(embedder="hash", chunker="fixed"),
        store=store,
        gates={"recall_at_3": ">= 0.9"},
    )
    # First run is a baseline, but an absolute gate still applies.
    assert verdict.is_baseline is True
    assert verdict.passed is False


@pytest.mark.asyncio
async def test_artifacts_are_versioned_per_key():
    store = IndexRegressionStore()
    config = RetrievalConfig(embedder="hash", chunker="fixed")
    golden = _golden()
    await retrieval_regression(lexical_search, golden, config, store=store)
    await retrieval_regression(lexical_search, golden, config, store=store)
    key = config.key(golden.corpus_hash)
    history = store.history(key)
    assert [a.version for a in history] == [1, 2]
