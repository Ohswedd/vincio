"""Real-behavior coverage tests for ``vincio.evals.metrics``.

Each test drives a metric function through its real public API on known inputs
and asserts the exact computed score (or exact state: skipped / passed /
details). No mocking — deterministic offline metrics only. The expected numbers
were derived by hand from the metric definitions and confirmed against the
implementation, so a regression in any formula or branch fails a test.
"""

from __future__ import annotations

import math

from vincio.core.types import EvidenceItem, TokenUsage
from vincio.evals.datasets import EvalCase
from vincio.evals.metrics import (
    METRICS,
    MetricResult,
    RunOutput,
    answer_relevance,
    bias,
    citation_accuracy,
    citation_coverage,
    citation_recall,
    claim_entailment,
    classification_accuracy,
    context_precision,
    context_recall,
    conversation_outcome,
    conversation_relevance,
    cost_metric,
    exact_match,
    extraction_f1,
    faithfulness,
    goal_accuracy,
    groundedness,
    hallucination,
    intent_resolution,
    knowledge_retention,
    latency_metric,
    lexical_overlap,
    mrr,
    ndcg,
    plan_adherence,
    plan_quality,
    precision_at_k,
    recall_at_k,
    register_metric,
    schema_validity,
    semantic_similarity,
    set_semantic_embedder,
    step_efficiency,
    summarization_quality,
    tool_call_accuracy,
    tool_call_f1,
    topic_adherence,
    unsupported_claim_rate,
)
from vincio.evals.trajectory import Trajectory, TrajectoryStep
from vincio.observability.spans import Trace


def case(input_text: str = "q", **kw) -> EvalCase:
    return EvalCase(id=kw.pop("id", "c"), input=input_text, **kw)


def ev(id_: str, text: str, **kw) -> EvidenceItem:
    return EvidenceItem(id=id_, source_id=kw.pop("source_id", "src"), text=text, **kw)


# -- RunOutput.output_text --------------------------------------------------


def test_output_text_str_passthrough():
    assert RunOutput(output="hello").output_text == "hello"


def test_output_text_none_falls_back_to_raw_text():
    assert RunOutput(output=None, raw_text="raw fallback").output_text == "raw fallback"


def test_output_text_pydantic_model_uses_model_dump_json():
    from pydantic import BaseModel

    class M(BaseModel):
        x: int = 5

    assert RunOutput(output=M()).output_text == '{"x":5}'


def test_output_text_dict_serialized_as_json():
    assert RunOutput(output={"a": 1}).output_text == '{"a": 1}'


def test_output_text_set_uses_json_default_str():
    # json.dumps(set, default=str) coerces the set to its repr string.
    assert RunOutput(output={1, 2}).output_text == '"{1, 2}"'


def test_from_trace_carries_output_id_and_cost():
    trace = Trace(id="t1", name="run", spans=[], attributes={"output": "hello world", "cost_usd": 0.5})
    ro = RunOutput.from_trace(trace)
    assert ro.output == "hello world"
    assert ro.trace_id == "t1"
    assert ro.cost_usd == 0.5
    assert ro.output_text == "hello world"


# -- exact_match ------------------------------------------------------------


def test_exact_match_normalized_equal():
    res = exact_match(case(expected="The Capital, France!"), RunOutput(output="the capital france"))
    assert res.value == 1.0
    assert res.passed is True


def test_exact_match_empty_expected_is_zero():
    # Empty expected must not pass even against empty output.
    res = exact_match(case(expected=None), RunOutput(output=""))
    assert res.value == 0.0
    assert res.passed is False


def test_exact_match_mismatch():
    assert exact_match(case(expected="paris"), RunOutput(output="london")).value == 0.0


# -- semantic_similarity ----------------------------------------------------


def test_semantic_similarity_identical_is_one():
    res = semantic_similarity(case(expected="the cat sat"), RunOutput(output="the cat sat"))
    assert res.value == 1.0


def test_semantic_similarity_no_expected_is_skipped():
    res = semantic_similarity(case(expected=None), RunOutput(output="anything"))
    assert res.skipped is True
    assert res.value == 1.0


def test_semantic_similarity_empty_output_is_zero():
    res = semantic_similarity(case(expected="hello"), RunOutput(output=""))
    assert res.value == 0.0
    assert res.skipped is False


def test_set_semantic_embedder_swaps_backend_and_restores():
    class FixedEmbedder:
        def embed_one(self, text: str) -> list[float]:
            return [1.0, 0.0]

    from vincio.evals import metrics as m

    original = m._SEMANTIC_EMBEDDER
    try:
        set_semantic_embedder(FixedEmbedder())
        # Both texts map to the same vector -> cosine 1.0 regardless of content.
        res = semantic_similarity(case(expected="apple"), RunOutput(output="orange"))
        assert res.value == 1.0
    finally:
        set_semantic_embedder(original)


# -- classification_accuracy ------------------------------------------------


def test_classification_accuracy_object_label_attr():
    class Pred:
        label = "positive"

    res = classification_accuracy(case(expected={"label": "positive"}), RunOutput(output=Pred()))
    assert res.value == 1.0
    assert res.passed is True
    assert res.details["got"] == "positive"


def test_classification_accuracy_string_is_stripped():
    res = classification_accuracy(case(expected="positive"), RunOutput(output="  positive "))
    assert res.value == 1.0


def test_classification_accuracy_mismatch():
    res = classification_accuracy(case(expected={"label": "positive"}), RunOutput(output="negative"))
    assert res.value == 0.0
    assert res.passed is False


# -- extraction_f1 ----------------------------------------------------------


def test_extraction_f1_partial_lists():
    # expected {a,b,c}, got {a,b}: precision 2/2=1, recall 2/3 -> f1 = 0.8
    res = extraction_f1(case(expected=["a", "b", "c"]), RunOutput(output=["a", "b"]))
    assert res.value == 0.8
    assert res.details["precision"] == 1.0
    assert res.details["recall"] == 0.6667


def test_extraction_f1_dict_keyvalue_match():
    res = extraction_f1(case(expected={"k": "v"}), RunOutput(output={"k": "v"}))
    assert res.value == 1.0


def test_extraction_f1_list_of_dict_items_match():
    spec = [{"name": "Alice", "age": "30"}]
    res = extraction_f1(case(expected=spec), RunOutput(output=list(spec)))
    assert res.value == 1.0


def test_extraction_f1_no_reference_is_skipped():
    res = extraction_f1(case(expected=None), RunOutput(output=["a"]))
    assert res.skipped is True
    assert res.value == 1.0


def test_extraction_f1_both_empty_passes():
    res = extraction_f1(case(expected=[]), RunOutput(output=[]))
    assert res.value == 1.0
    assert res.passed is True


def test_extraction_f1_model_dump_output():
    from pydantic import BaseModel

    class Out(BaseModel):
        city: str = "paris"

    res = extraction_f1(case(expected={"city": "paris"}), RunOutput(output=Out()))
    assert res.value == 1.0


# -- schema_validity --------------------------------------------------------


def test_schema_validity_true_and_false():
    assert schema_validity(case(), RunOutput(schema_valid=True)).value == 1.0
    assert schema_validity(case(), RunOutput(schema_valid=False)).value == 0.0


def test_schema_validity_none_uses_error_presence():
    assert schema_validity(case(), RunOutput()).value == 1.0
    assert schema_validity(case(), RunOutput(error="boom")).value == 0.0


# -- groundedness / unsupported_claim_rate ----------------------------------


def test_groundedness_no_verifiable_claims_skipped():
    res = groundedness(case(), RunOutput(output="Hi."))
    assert res.skipped is True
    assert res.details["claims"] == 0


def test_groundedness_supported_and_unsupported_mix():
    evidence = [ev("e1", "The refund period is 30 days for all customers")]
    run = RunOutput(
        output="The refund period is 30 days for all customers. The purple sky has nine moons tonight.",
        evidence=evidence,
    )
    res = groundedness(case(), run)
    # Two verifiable claims; only the first is supported by evidence.
    assert res.details["claims"] == 2
    assert res.details["supported"] == 1
    assert res.value == 0.5


def test_unsupported_claim_rate_half():
    evidence = [ev("e1", "The refund period is 30 days for all customers")]
    run = RunOutput(
        output="The refund period is 30 days for all customers. The purple sky has nine moons tonight.",
        evidence=evidence,
    )
    assert unsupported_claim_rate(case(), run).value == 0.5


def test_unsupported_claim_rate_no_claims_is_zero():
    assert unsupported_claim_rate(case(), RunOutput(output="Hi.")).value == 0.0


# -- citation metrics -------------------------------------------------------


def test_citation_accuracy_half():
    evidence = [ev("D1", "evidence one", citation_ref="C1")]
    run = RunOutput(citations=["D1", "X9"], evidence=evidence)
    res = citation_accuracy(case(), run)
    assert res.value == 0.5
    assert res.details["correct"] == 1


def test_citation_accuracy_no_citations_is_zero():
    res = citation_accuracy(case(), RunOutput())
    assert res.value == 0.0
    assert res.details["citations"] == 0


def test_citation_recall_from_rubric():
    c = case(rubric={"required_evidence": ["D1", "D2"]})
    assert citation_recall(c, RunOutput(citations=["D1"])).value == 0.5


def test_citation_recall_skipped_when_no_required():
    assert citation_recall(case(), RunOutput()).skipped is True


def test_citation_recall_fallback_to_run_evidence():
    evidence = [ev("D1", "x")]
    res = citation_recall(case(), RunOutput(citations=["D1"], evidence=evidence))
    assert res.value == 1.0


def test_citation_coverage_cited_claim():
    evidence = [ev("D1", "The capital of France is Paris with population 2 million", citation_ref="C1")]
    run = RunOutput(
        output="The capital of France is Paris [D1] with population 2 million.", evidence=evidence
    )
    assert citation_coverage(case(), run).value == 1.0


def test_citation_coverage_no_claims_skipped():
    assert citation_coverage(case(), RunOutput(output="Hi.")).skipped is True


def test_claim_entailment_supported_cited_claim():
    evidence = [ev("D1", "The capital of France is Paris with population 2 million")]
    run = RunOutput(
        output="The capital of France is Paris [D1] with population 2 million.", evidence=evidence
    )
    assert claim_entailment(case(), run).value == 1.0


def test_claim_entailment_skipped_when_nothing_cited():
    evidence = [ev("D1", "some evidence text here")]
    run = RunOutput(output="The fact here is that nothing was cited at all.", evidence=evidence)
    assert claim_entailment(case(), run).skipped is True


def test_claim_entailment_number_contradiction_unsupported():
    # Strict entailment: cited evidence says 2 million, claim says 9 million.
    evidence = [ev("D1", "The capital of France is Paris with population 2 million")]
    run = RunOutput(
        output="The capital of France is Paris [D1] with population 9 million.", evidence=evidence
    )
    assert claim_entailment(case(), run).value == 0.0


# -- context metrics --------------------------------------------------------


def test_context_precision_no_evidence_is_zero():
    assert context_precision(case(), RunOutput()).value == 0.0


def test_context_recall_skipped_without_facts():
    assert context_recall(case(), RunOutput()).skipped is True


def test_context_recall_covered_fact():
    c = case(rubric={"facts": ["the price is 30 dollars per unit always"]})
    evidence = [ev("e1", "the price is 30 dollars per unit always here")]
    res = context_recall(c, RunOutput(evidence=evidence))
    assert res.value == 1.0
    assert res.details["covered"] == 1


# -- faithfulness -----------------------------------------------------------


def test_faithfulness_reference_from_context():
    c = case(context={"reference": ["The capital of France is Paris indeed"]})
    run = RunOutput(output="The capital of France is Paris indeed.")
    assert faithfulness(c, run).value == 1.0


def test_faithfulness_no_claims_skipped():
    assert faithfulness(case(), RunOutput(output="Hi.")).skipped is True


# -- answer_relevance -------------------------------------------------------


def test_answer_relevance_empty_answer_is_zero():
    res = answer_relevance(case(), RunOutput(output="   "))
    assert res.value == 0.0
    assert res.details["empty"] is True


def test_answer_relevance_noncommittal_penalized():
    c = case("what is the refund policy duration")
    res = answer_relevance(c, RunOutput(output="I don't know the refund policy duration honestly."))
    assert res.details["noncommittal"] is True
    # The 0.25 multiplier means the value cannot exceed 0.25.
    assert res.value <= 0.25


# -- hallucination ----------------------------------------------------------


def test_hallucination_no_claims_is_zero_and_passes():
    res = hallucination(case(), RunOutput(output="Hi."))
    assert res.value == 0.0
    assert res.passed is True


def test_hallucination_number_mismatch_is_one():
    evidence = [ev("e", "The refund window is 30 days for customers")]
    run = RunOutput(output="The refund window is 90 days for customers", evidence=evidence)
    res = hallucination(case(), run)
    assert res.value == 1.0
    assert res.passed is False


def test_hallucination_exact_number_match_is_zero():
    evidence = [ev("e", "The refund window is 30 days for customers")]
    run = RunOutput(output="The refund window is 30 days for customers", evidence=evidence)
    assert hallucination(case(), run).value == 0.0


# -- toxicity / bias --------------------------------------------------------


def test_toxicity_one_of_two_sentences():
    res = toxicity_run("You are an idiot. Have a nice day.")
    assert res.value == 0.5
    assert res.passed is False


def toxicity_run(text: str) -> MetricResult:
    return METRICS["toxicity"](case(), RunOutput(output=text))


def test_toxicity_clean_is_zero():
    assert toxicity_run("Have a nice day.").value == 0.0


def test_bias_sweeping_generalization():
    res = bias(case(), RunOutput(output="All women are bad drivers. The weather is fine."))
    assert res.value == 0.5


# -- summarization_quality --------------------------------------------------


def test_summarization_quality_missing_source_is_zero():
    res = summarization_quality(case(), RunOutput(output=""))
    assert res.value == 0.0
    assert res.details["missing"] == "source or summary"


def test_summarization_quality_full_coverage_and_faithful():
    src = (
        "The annual report shows revenue grew twenty percent over the prior fiscal year. "
        "The company expanded into three new international markets during the period."
    )
    c = case(context={"source": src})
    summary = (
        "Revenue grew twenty percent over the prior fiscal year and expanded "
        "into three new international markets."
    )
    assert summarization_quality(c, RunOutput(output=summary)).value == 1.0


# -- knowledge_retention ----------------------------------------------------


def test_knowledge_retention_violation_when_reasking():
    c = case(
        context={
            "messages": [
                {"role": "user", "content": [{"text": "My order number is 12345 and it shipped today"}]},
                {"role": "assistant", "content": "ok"},
            ]
        }
    )
    run = RunOutput(output="What is your order number 12345 that shipped today?")
    assert knowledge_retention(c, run).value == 0.0


def test_knowledge_retention_skipped_without_facts():
    assert knowledge_retention(case(), RunOutput(output="hi")).skipped is True


def test_knowledge_retention_no_violation_keeps_one():
    c = case(rubric={"session_facts": ["user is allergic to peanuts always"]})
    assert knowledge_retention(c, RunOutput(output="Here is your meal plan.")).value == 1.0


# -- conversation metrics ---------------------------------------------------


def test_conversation_relevance_falls_back_to_input():
    res = conversation_relevance(
        case("tell me about cats"),
        RunOutput(output="cats are great pets and cats love to sleep"),
    )
    assert res.details["turns"] == 0
    assert res.value > 0.0


def test_conversation_outcome_skipped_without_goal():
    assert conversation_outcome(case(), RunOutput(output="hi")).skipped is True


def test_conversation_outcome_keyword_matching():
    c = case(rubric={"goal_keywords": ["refund", "approved"]})
    assert conversation_outcome(c, RunOutput(output="Your refund has been approved.")).value == 1.0
    partial = conversation_outcome(c, RunOutput(output="Your refund is pending."))
    assert partial.value == 0.5
    assert partial.passed is True


def test_conversation_outcome_goal_similarity():
    c = case(context={"goal": "book a flight to paris next week"})
    res = conversation_outcome(c, RunOutput(output="I booked a flight to paris next week for you"))
    assert res.value == 1.0


def test_intent_resolution_no_messages_uses_input():
    res = intent_resolution(
        case("tell me about cats"),
        RunOutput(output="cats are wonderful pets and cats are independent animals"),
    )
    assert res.details["turns"] == 0


def test_intent_resolution_all_resolved():
    c = case(
        context={
            "messages": [
                {"role": "user", "content": "what is the capital of france"},
                {"role": "assistant", "content": "the capital of france is paris"},
                {"role": "user", "content": "what about germany"},
            ]
        }
    )
    res = intent_resolution(c, RunOutput(output="the capital of germany is berlin"))
    assert res.value == 1.0
    assert res.details["intents"] == 2
    assert res.details["resolved"] == 2


# -- tool-call metrics ------------------------------------------------------


def _tool(name: str, args: dict | None = None, status: str = "done") -> TrajectoryStep:
    return TrajectoryStep(type="tool", tool_name=name, tool_arguments=args or {}, status=status)


def test_tool_call_accuracy_with_expected_ordered_args():
    traj = Trajectory(steps=[_tool("search", {"q": "paris"}), _tool("fetch")], success=True)
    c = case(rubric={"expected_tools": [{"tool": "search", "arguments": {"q": "paris"}}, "fetch"]})
    res = tool_call_accuracy(c, RunOutput(trajectory=traj))
    assert res.value == 1.0
    assert res.passed is True


def test_tool_call_accuracy_success_rate_fallback():
    traj = Trajectory(steps=[_tool("a"), _tool("b", status="failed")])
    res = tool_call_accuracy(case(), RunOutput(trajectory=traj))
    assert res.value == 0.5
    assert res.details["mode"] == "success_rate"


def test_tool_call_accuracy_no_tools_skipped():
    assert tool_call_accuracy(case(), RunOutput()).skipped is True


def test_tool_call_accuracy_expected_from_case_expected_dict():
    traj = Trajectory(steps=[_tool("search", {"q": "paris"}), _tool("fetch")])
    c = case(expected={"tool_calls": ["search", "fetch"]})
    assert tool_call_accuracy(c, RunOutput(trajectory=traj)).value == 1.0


def test_tool_call_f1_skipped_without_reference():
    traj = Trajectory(steps=[_tool("search")])
    assert tool_call_f1(case(), RunOutput(trajectory=traj)).skipped is True


def test_tool_call_f1_partial():
    traj = Trajectory(steps=[_tool("search"), _tool("fetch")])
    c = case(rubric={"expected_tools": ["search", "fetch", "extra"]})
    res = tool_call_f1(c, RunOutput(trajectory=traj))
    # tp=2, precision 2/2=1, recall 2/3 -> f1 0.8
    assert res.value == 0.8
    assert res.details["tp"] == 2


# -- goal_accuracy ----------------------------------------------------------


def test_goal_accuracy_no_expected_uses_error():
    assert goal_accuracy(case(), RunOutput(error=None)).value == 1.0
    assert goal_accuracy(case(), RunOutput(error="boom")).value == 0.0


def test_goal_accuracy_success_and_answer_match():
    traj = Trajectory(steps=[TrajectoryStep(type="finalize")], success=True)
    c = case(expected="the capital of france is paris")
    full = goal_accuracy(c, RunOutput(output="the capital of france is paris", trajectory=traj))
    assert full.value == 1.0
    # success but wrong answer -> only half credit.
    half = goal_accuracy(c, RunOutput(output="something unrelated entirely", trajectory=traj))
    assert half.value == 0.5


# -- plan metrics -----------------------------------------------------------


def test_plan_adherence_skipped_without_plan():
    assert plan_adherence(case(), RunOutput()).skipped is True


def test_plan_adherence_full_and_partial():
    steps = [TrajectoryStep(type="retrieve"), _tool("search"), TrajectoryStep(type="finalize")]
    traj = Trajectory(steps=steps)
    full = plan_adherence(case(rubric={"plan": ["retrieve", "search", "finalize"]}), RunOutput(trajectory=traj))
    assert full.value == 1.0
    partial = plan_adherence(case(rubric={"plan": ["retrieve", "missing", "finalize"]}), RunOutput(trajectory=traj))
    assert partial.value == 0.6667


def test_plan_quality_skipped_without_steps():
    assert plan_quality(case(), RunOutput()).skipped is True


def test_plan_quality_penalizes_failed_and_redundant():
    steps = [_tool("a"), _tool("a"), _tool("b", status="failed")]
    res = plan_quality(case(), RunOutput(trajectory=Trajectory(steps=steps)))
    # 1 redundant + 1 failed over 3 steps -> 1 - 2/3 = 0.3333
    assert res.value == 0.3333
    assert res.details["failed"] == 1
    assert res.details["redundant"] == 1


def test_step_efficiency_skipped_without_steps():
    assert step_efficiency(case(), RunOutput()).skipped is True


def test_step_efficiency_with_optimal():
    traj = Trajectory(steps=[TrajectoryStep(type=t) for t in "abcd"])
    res = step_efficiency(case(rubric={"optimal_steps": 2}), RunOutput(trajectory=traj))
    assert res.value == 0.5
    assert res.passed is False


def test_step_efficiency_fallback_counts_useful_steps():
    steps = [
        TrajectoryStep(type="a", name="x"),
        TrajectoryStep(type="a", name="x"),  # redundant repeat
        TrajectoryStep(type="b", name="y", status="failed"),  # failed
    ]
    res = step_efficiency(case(), RunOutput(trajectory=Trajectory(steps=steps)))
    assert res.value == 0.3333
    assert res.details["useful"] == 1


def test_topic_adherence_skipped_without_steps():
    assert topic_adherence(case(), RunOutput()).skipped is True


def test_topic_adherence_on_topic_steps():
    traj = Trajectory(
        objective="research the climate of paris",
        steps=[_tool("search", {"q": "paris climate"}), TrajectoryStep(type="think")],
    )
    res = topic_adherence(case(), RunOutput(trajectory=traj))
    assert res.value == 1.0
    assert res.details["on_topic"] == 2


# -- operational metrics ----------------------------------------------------


def test_operational_metrics_exact_values():
    assert METRICS["input_tokens"](case(), RunOutput(usage=TokenUsage(input_tokens=42))).value == 42.0
    assert METRICS["output_tokens"](case(), RunOutput(usage=TokenUsage(output_tokens=7))).value == 7.0
    assert METRICS["retries"](case(), RunOutput(retries=3)).value == 3.0
    assert latency_metric(case(), RunOutput(latency_ms=250)).value == 250.0


def test_cost_metric_rounds_to_eight_places():
    assert cost_metric(case(), RunOutput(cost_usd=0.000123456)).value == 0.00012346


# -- retrieval ranking metrics ----------------------------------------------


def _ranked() -> list[EvidenceItem]:
    return [ev("D1", "a"), ev("D2", "b"), ev("D3", "c")]


def test_recall_at_k_value():
    c = case(rubric={"relevant_ids": ["D2", "D3"]})
    assert recall_at_k(c, RunOutput(evidence=_ranked())).value == 1.0


def test_recall_at_k_skipped_without_relevant():
    assert recall_at_k(case(), RunOutput(evidence=_ranked())).skipped is True


def test_precision_at_k_value_and_no_evidence():
    c = case(rubric={"relevant_ids": ["D2", "D3"]})
    assert precision_at_k(c, RunOutput(evidence=_ranked())).value == 0.6667
    assert precision_at_k(c, RunOutput()).value == 0.0


def test_precision_at_k_skipped_without_relevant():
    assert precision_at_k(case(), RunOutput(evidence=_ranked())).skipped is True


def test_mrr_first_relevant_at_rank_two():
    c = case(rubric={"relevant_ids": ["D2", "D3"]})
    assert mrr(c, RunOutput(evidence=_ranked())).value == 0.5


def test_mrr_miss_is_zero():
    c = case(rubric={"relevant_ids": ["ZZ"]})
    assert mrr(c, RunOutput(evidence=_ranked())).value == 0.0


def test_mrr_skipped_without_relevant():
    assert mrr(case(), RunOutput(evidence=_ranked())).skipped is True


def test_ndcg_value_matches_manual_computation():
    c = case(rubric={"relevant_ids": ["D2", "D3"]})
    res = ndcg(c, RunOutput(evidence=_ranked()))
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)  # ranks 2 and 3
    ideal = 1.0 / math.log2(2) + 1.0 / math.log2(3)  # ranks 1 and 2
    assert res.value == round(dcg / ideal, 4)


def test_ndcg_miss_is_zero_and_skip_without_relevant():
    c_miss = case(rubric={"relevant_ids": ["ZZ"]})
    assert ndcg(c_miss, RunOutput(evidence=_ranked())).value == 0.0
    assert ndcg(case(), RunOutput(evidence=_ranked())).skipped is True


# -- registry ---------------------------------------------------------------


def test_register_metric_adds_to_registry():
    name = "__cov_test_dummy_metric__"

    @register_metric(name)
    def _dummy(c, run):  # pragma: no cover - body trivial, registration is the point
        return MetricResult(name=name, value=0.5)

    try:
        assert METRICS[name] is _dummy
        assert METRICS[name](case(), RunOutput()).value == 0.5
    finally:
        METRICS.pop(name, None)


# -- additional branch coverage --------------------------------------------


def test_lexical_overlap_identical_is_one():
    c = case(expected="the quick brown fox")
    assert lexical_overlap(c, RunOutput(output="the quick brown fox")).value == 1.0


def test_lexical_overlap_disjoint_is_zero():
    c = case(expected="alpha beta gamma")
    assert lexical_overlap(c, RunOutput(output="delta epsilon zeta")).value == 0.0


def test_exact_match_expected_dict_serialized_then_matched():
    # _expected_text JSON-encodes a dict expected; exact_match compares normalized.
    c = case(expected={"answer": "paris"})
    assert exact_match(c, RunOutput(output='{"answer": "paris"}')).value == 1.0


def test_classification_accuracy_dict_output_label():
    res = classification_accuracy(case(expected="positive"), RunOutput(output={"label": "positive"}))
    assert res.value == 1.0


def test_extraction_f1_scalar_value():
    # Non-dict, non-collection expected/output go through the scalar branch.
    assert extraction_f1(case(expected="paris"), RunOutput(output="paris")).value == 1.0


def test_context_precision_counts_relevant_evidence():
    evidence = [
        ev("e1", "the capital of france is paris indeed"),
        ev("e2", "unrelated weather report cloudy skies"),
    ]
    c = case(expected="the capital of france is paris indeed")
    assert context_precision(c, RunOutput(evidence=evidence)).value == 0.5


def test_citation_coverage_no_evidence_uses_marker_presence():
    # With no evidence, a present marker counts (resolution unavailable).
    run = RunOutput(output="The capital of France is Paris [D1] with population 2 million.")
    assert citation_coverage(case(), run).value == 1.0


def test_faithfulness_unsupported_without_any_reference():
    # No run evidence and no context reference -> nothing supports the claim.
    run = RunOutput(output="The capital of France is Paris indeed today.")
    res = faithfulness(case(), run)
    assert res.value == 0.0
    assert res.details["supported"] == 0


def test_answer_relevance_non_noncommittal_branch():
    c = case("what is the capital of france")
    res = answer_relevance(c, RunOutput(output="the capital of france is paris"))
    assert res.details["noncommittal"] is False
    assert res.value > 0.0


def test_conversation_relevance_with_messages_content_blocks():
    c = case(
        context={
            "messages": [
                {"role": "user", "content": [{"text": "tell me about the paris climate today"}]}
            ]
        }
    )
    res = conversation_relevance(c, RunOutput(output="the paris climate today is mild and pleasant"))
    assert res.details["turns"] == 1
    assert res.value > 0.0


def test_intent_resolution_falls_back_to_run_output_when_no_reply():
    c = case(context={"messages": [{"role": "user", "content": "what is the capital of france"}]})
    res = intent_resolution(c, RunOutput(output="the capital of france is paris"))
    assert res.value == 1.0
    assert res.details["intents"] == 1
    assert res.details["resolved"] == 1


def test_tool_call_accuracy_arg_mismatch_and_missing_actual():
    # Actual search has wrong arg (london vs paris); fetch is missing entirely.
    traj = Trajectory(steps=[_tool("search", {"q": "london"})])
    c = case(rubric={"expected_tools": [{"tool": "search", "arguments": {"q": "paris"}}, "fetch"]})
    res = tool_call_accuracy(c, RunOutput(trajectory=traj))
    assert res.value == 0.0
    assert res.details["correct"] == 0


def test_topic_adherence_empty_text_step_counts_on_topic():
    # A step with no free text cannot be off-topic, so it counts as on-topic.
    traj = Trajectory(objective="research", steps=[TrajectoryStep(type="think")])
    assert topic_adherence(case(), RunOutput(trajectory=traj)).value == 1.0
