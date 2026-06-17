"""2.0 breaking eval semantics: unscoreable metrics are skipped (not a neutral
1.0), and ``semantic_similarity`` is embedding-backed while the lexical metric
is honestly named ``lexical_overlap``."""

from __future__ import annotations

import pytest

from vincio.evals.datasets import Dataset, EvalCase
from vincio.evals.metrics import (
    METRICS,
    MetricResult,
    RunOutput,
    lexical_overlap,
    semantic_similarity,
    set_semantic_embedder,
)
from vincio.evals.reports import GateSpec, evaluate_gates
from vincio.evals.runners import EvalRunner


def _case(**kw):
    return EvalCase(id=kw.pop("id", "c1"), input=kw.pop("input", "q"), **kw)


def test_metric_result_has_skipped_field():
    assert "skipped" in MetricResult.model_fields
    assert MetricResult(name="m", value=1.0).skipped is False


def test_lexical_overlap_is_registered_and_lexical():
    assert "lexical_overlap" in METRICS
    case = _case(expected="the quick brown fox")
    run = RunOutput(output="the quick brown fox")
    assert lexical_overlap(case, run).value == pytest.approx(1.0)
    run2 = RunOutput(output="completely different unrelated tokens")
    assert lexical_overlap(case, run2).value < 0.3


def test_semantic_similarity_is_embedding_backed():
    # Identical text → cosine 1.0; unrelated text → low; deterministic offline.
    case = _case(expected="annual revenue grew twelve percent")
    same = semantic_similarity(case, RunOutput(output="annual revenue grew twelve percent"))
    assert same.value == pytest.approx(1.0, abs=1e-6)
    assert same.skipped is False
    a = semantic_similarity(case, RunOutput(output="the cat sat on the warm mat"))
    b = semantic_similarity(case, RunOutput(output="the cat sat on the warm mat"))
    assert a.value == b.value  # deterministic
    assert 0.0 <= a.value <= 1.0


def test_semantic_similarity_skipped_without_reference():
    case = _case(expected=None)
    result = semantic_similarity(case, RunOutput(output="anything"))
    assert result.skipped is True


def test_set_semantic_embedder_swaps_backend():
    class _ConstEmbedder:
        def embed_one(self, text: str):
            return [1.0, 0.0, 0.0]

    case = _case(expected="alpha")
    set_semantic_embedder(_ConstEmbedder())
    try:
        # Every vector identical → cosine 1.0 regardless of text.
        assert semantic_similarity(case, RunOutput(output="beta")).value == pytest.approx(1.0)
    finally:
        from vincio.retrieval.embeddings import LocalHashEmbedder

        set_semantic_embedder(LocalHashEmbedder(dim=256))


def test_unscoreable_metrics_are_skipped_not_neutral_one():
    # faithfulness on output with no verifiable claims is unscoreable.
    case = _case(expected="x")
    run = RunOutput(output="ok")  # no claims, no evidence
    result = METRICS["faithfulness"](case, run)
    assert result.skipped is True

    # recall_at_k with no relevant ids defined is unscoreable.
    rec = METRICS["recall_at_k"](case, run)
    assert rec.skipped is True


async def test_runner_excludes_skipped_from_aggregation_and_gates():
    # Two cases: one scoreable for recall_at_k, one not. The unscoreable case
    # must not contribute a 1.0 that inflates the mean.
    dataset = Dataset(
        name="t",
        cases=[
            EvalCase(id="scoreable", input="q1", rubric={"relevant_ids": ["E1"]}),
            EvalCase(id="unscoreable", input="q2"),  # no relevant_ids
        ],
    )

    async def target(case: EvalCase) -> RunOutput:
        # Never retrieves the relevant id → recall 0 for the scoreable case.
        return RunOutput(output="answer", evidence=[])

    runner = EvalRunner(target, metrics=["recall_at_k"])
    report = await runner.arun(dataset)
    by_id = {c.case_id: c for c in report.cases}
    assert "recall_at_k" in by_id["scoreable"].metrics
    assert by_id["scoreable"].metrics["recall_at_k"] == 0.0
    # The unscoreable case excludes the metric entirely (not a neutral 1.0).
    assert "recall_at_k" not in by_id["unscoreable"].metrics
    assert "recall_at_k" in by_id["unscoreable"].details.get("_skipped_metrics", [])

    # Aggregation sees only the one scored value (0.0), so a >=0.5 gate fails
    # instead of being inflated to a mean of 0.5 by the skipped 1.0.
    values = report.metric_values("recall_at_k")
    assert values == [0.0]
    passed, agg = GateSpec(metric="recall_at_k", expression=">= 0.5").check(values)
    assert passed is False
    assert agg == 0.0


async def test_gate_with_all_cases_skipped_does_not_silently_pass():
    dataset = Dataset(name="t", cases=[EvalCase(id="u", input="q")])

    async def target(case: EvalCase) -> RunOutput:
        return RunOutput(output="answer")

    runner = EvalRunner(target, metrics=["recall_at_k"])
    report = await runner.arun(dataset)
    # No scoreable values at all → the gate cannot pass on a phantom 1.0.
    outcomes = evaluate_gates(report, {"recall_at_k": ">= 0.5"})
    assert outcomes["recall_at_k"]["passed"] is False
