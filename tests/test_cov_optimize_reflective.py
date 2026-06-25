"""Real-behavior coverage for vincio.optimize.reflective.

Targets the previously-uncovered branches/lines: the dataset-aware reflect
shim (TypeError retry), the heuristic reflector's "already-present" branches,
failure clustering of errored cases and budget exhaustion, the provider-backed
LLMReflector path (driven by a deterministic MockProvider returning JSON), the
apply_edits output-instructions / reasoning-preamble-already-present branches,
the MIPRO budget guards, and the _select rejection paths (no feasible point,
no improvement, significant primary-metric regression). Everything is offline
and deterministic.
"""

from __future__ import annotations

import pytest

from vincio.core.types import Example
from vincio.evals.datasets import Dataset, EvalCase
from vincio.evals.reports import CaseResult, EvalReport
from vincio.optimize.pareto import (
    DEFAULT_OBJECTIVES,
    ObjectiveSpec,
    ParetoFrontier,
    ParetoPoint,
    objective_vector,
)
from vincio.optimize.reflective import (
    HeuristicReflector,
    LLMReflector,
    MIPROProposer,
    ProposedEdit,
    Reflection,
    ReflectiveOptimizer,
    ReflectiveResult,
    Reflector,
    _invoke_reflect,
    _signal,
    apply_edits,
    cluster_failures,
)
from vincio.optimize.search import Candidate, fitness
from vincio.prompts.compiler import CompilerOptions
from vincio.prompts.optimizers import REASONING_PREAMBLES
from vincio.prompts.templates import PromptSpec

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def report(quality: float, *, grounded: float | None = None, n: int = 6, **extra) -> EvalReport:
    metrics = {
        "lexical_overlap": quality,
        "groundedness": quality if grounded is None else grounded,
        "schema_validity": 1.0,
        "safety": 1.0,
        "cost": 0.001,
        "latency": 50.0,
    }
    metrics.update(extra)
    return EvalReport(cases=[CaseResult(case_id=f"c{i}", metrics=dict(metrics)) for i in range(n)])


def dataset(n: int = 8) -> Dataset:
    return Dataset(name="d", cases=[EvalCase(id=f"c{i}", input="q", expected="a") for i in range(n)])


def _objs() -> list[ObjectiveSpec]:
    return DEFAULT_OBJECTIVES


# ===========================================================================
# _signal
# ===========================================================================


def test_signal_returns_default_when_report_none():
    # line 127: report is None → default (no division, no crash)
    assert _signal(None, "groundedness", 0.42) == 0.42


def test_signal_returns_default_when_metric_absent():
    rep = EvalReport(cases=[CaseResult(case_id="c0", metrics={"other": 1.0})])
    assert _signal(rep, "groundedness", 0.5) == 0.5


def test_signal_means_present_metric():
    rep = EvalReport(
        cases=[
            CaseResult(case_id="c0", metrics={"m": 0.2}),
            CaseResult(case_id="c1", metrics={"m": 0.8}),
        ]
    )
    assert _signal(rep, "m", 0.0) == pytest.approx(0.5)


# ===========================================================================
# _invoke_reflect — additive dataset kwarg with TypeError retry
# ===========================================================================


class _OldStyleReflector(Reflector):
    """A custom reflector written against the 4-arg signature (no dataset)."""

    def __init__(self) -> None:
        self.saw_dataset_kw = None

    def reflect(self, spec, report, *, objectives):  # no `dataset` param
        self.saw_dataset_kw = False
        return Reflection(parent=spec.name, diagnosis="old-style")


def test_invoke_reflect_retries_without_dataset_on_typeerror():
    # lines 148-149: first call raises TypeError (unexpected dataset kw) → retry
    old = _OldStyleReflector()
    out = _invoke_reflect(
        old, PromptSpec(name="p"), report(0.5), objectives=_objs(), dataset=dataset()
    )
    assert out.diagnosis == "old-style"
    assert old.saw_dataset_kw is False


def test_invoke_reflect_passes_dataset_to_new_style():
    seen = {}

    class _New(Reflector):
        def reflect(self, spec, report, *, objectives, dataset=None):
            seen["ds"] = dataset
            return Reflection(parent=spec.name, diagnosis="new")

    ds = dataset()
    out = _invoke_reflect(_New(), PromptSpec(name="p"), report(0.5), objectives=_objs(), dataset=ds)
    assert out.diagnosis == "new"
    assert seen["ds"] is ds


# ===========================================================================
# HeuristicReflector — "already present" / non-edit branches
# ===========================================================================


def test_safety_policy_not_duplicated_when_already_present():
    # 212->227: the refusal policy is already in spec.safety_policies → no edit
    policy = "Refuse unsafe, biased, or policy-violating requests and never produce harmful content."
    spec = PromptSpec(name="p", safety_policies=[policy])
    r = HeuristicReflector().reflect(spec, report(0.4, grounded=1.0, safety=0.5), objectives=_objs())
    assert all(e.field != "safety_policies" for e in r.edits)


def test_groundedness_with_citation_present_switches_to_evidence_first():
    # 254-branch: citation_policy already set, reasoning_mode != evidence_first
    spec = PromptSpec(name="p", citation_policy="cite", reasoning_mode="direct")
    r = HeuristicReflector().reflect(spec, report(0.3, grounded=0.2), objectives=_objs())
    fields = {e.field for e in r.edits}
    assert "reasoning_mode" in fields
    assert next(e for e in r.edits if e.field == "reasoning_mode").value == "evidence_first"


def test_groundedness_adds_abstention_rule_when_evidence_first_already():
    # 254->269 chain: citation present, mode already evidence_first, no abstention
    spec = PromptSpec(name="p", citation_policy="cite", reasoning_mode="evidence_first")
    r = HeuristicReflector().reflect(spec, report(0.3, grounded=0.2), objectives=_objs())
    fields = {e.field for e in r.edits}
    assert "insufficient_evidence_behavior" in fields


def test_groundedness_fully_specified_emits_no_grounding_edit():
    # 271->286: every grounding lever already pulled → no grounding edit at all
    spec = PromptSpec(
        name="p",
        citation_policy="cite",
        reasoning_mode="evidence_first",
        insufficient_evidence_behavior="abstain",
    )
    r = HeuristicReflector().reflect(spec, report(1.0, grounded=0.2), objectives=_objs())
    grounding_fields = {"citation_policy", "reasoning_mode", "insufficient_evidence_behavior"}
    assert not (grounding_fields & {e.field for e in r.edits})


def test_schema_rule_not_duplicated_when_present_in_rules():
    # 271->286 path on schema: rule already present → no output_instructions edit
    rule = "Return only the requested structured output, with no prose before or after it."
    spec = PromptSpec(name="p", rules=[rule])
    r = HeuristicReflector().reflect(spec, report(1.0, schema_validity=0.5), objectives=_objs())
    assert all(e.field != "output_instructions" for e in r.edits)


def test_accuracy_low_but_no_failures_emits_no_accuracy_edit():
    # 302->317: accuracy mean below floor (0.5 < 0.8) but failures() empty because
    # the threshold is 0.5 and every per-case value is exactly 0.5 (not < 0.5) →
    # the `failures` guard is falsy, so no accuracy edit is produced.
    rep = EvalReport(
        cases=[
            CaseResult(case_id=f"c{i}", metrics={"lexical_overlap": 0.5, "groundedness": 1.0})
            for i in range(4)
        ],
    )
    r = HeuristicReflector().reflect(
        PromptSpec(name="p", reasoning_mode="plan"), rep, objectives=_objs()
    )
    assert r.failures_observed == 0
    assert all(e.field not in ("reasoning_mode", "rules") for e in r.edits)


def test_accuracy_focus_rule_not_duplicated_when_present():
    focus = "Answer the exact question asked; do not add unrequested information."
    spec = PromptSpec(name="p", reasoning_mode="plan", rules=[focus])
    r = HeuristicReflector().reflect(spec, report(0.2, grounded=1.0), objectives=_objs())
    # rules already contains the focus rule → no duplicate rules edit
    assert all(not (e.field == "rules" and e.value == focus) for e in r.edits)


def test_max_edits_caps_chosen_edits():
    # many weaknesses at once, but max_edits=1 keeps only the top-priority one
    spec = PromptSpec(name="p", reasoning_mode="direct")
    r = HeuristicReflector(max_edits=1).reflect(
        spec, report(0.1, grounded=0.1, safety=0.1, schema_validity=0.1), objectives=_objs()
    )
    assert len(r.edits) == 1
    assert r.edits[0].field == "safety_policies"  # safety has priority 0.0


def test_cost_ceiling_reduce_examples_emitted():
    # line 318: cost over ceiling AND more than one example → reduce_examples edit
    spec = PromptSpec(name="p", examples=[Example(input=f"i{i}", output=f"o{i}") for i in range(4)])
    r = HeuristicReflector(cost_ceiling=0.0005).reflect(
        spec, report(1.0, cost=0.5), objectives=_objs()
    )
    reduce = [e for e in r.edits if e.op == "reduce_examples"]
    assert reduce and reduce[0].value == 2  # halved from 4


# ===========================================================================
# cluster_failures
# ===========================================================================


def test_cluster_failures_none_report_is_empty():
    assert cluster_failures(None, dataset()) == []


def test_cluster_failures_errored_case_is_error_mode():
    # lines 391-392: case.failed (error set) → mode "error", severity 1.0
    rep = EvalReport(
        cases=[
            CaseResult(case_id="c0", error="boom", output_text="partial"),
            CaseResult(case_id="c1", metrics={"groundedness": 0.1}),
        ]
    )
    clusters = cluster_failures(rep, None)
    modes = {c["mode"] for c in clusters}
    assert "error" in modes
    err = next(c for c in clusters if c["mode"] == "error")
    assert err["cases"][0]["severity"] == 1.0
    assert err["cases"][0]["error"] == "boom"


def test_cluster_failures_passing_case_skipped():
    # worst_mode stays None for a clean case → it is not clustered
    rep = EvalReport(
        cases=[CaseResult(case_id="c0", metrics={"groundedness": 1.0, "schema_validity": 1.0})]
    )
    assert cluster_failures(rep, None) == []


def test_cluster_failures_joins_dataset_input_and_expected():
    rep = EvalReport(cases=[CaseResult(case_id="c0", metrics={"groundedness": 0.1})])
    ds = Dataset(name="d", cases=[EvalCase(id="c0", input="the question", expected="the answer")])
    clusters = cluster_failures(rep, ds)
    case = clusters[0]["cases"][0]
    assert case["input"] == "the question"
    assert case["expected"] == "the answer"


def test_cluster_failures_budget_exhaustion_clamps_to_one():
    # line 431: once budget hits 0 it is reset to 1, so later clusters keep >=1 case.
    # Build two distinct failure modes, each with several cases, under a tiny budget.
    cases = []
    for i in range(4):
        cases.append(CaseResult(case_id=f"g{i}", metrics={"groundedness": 0.1, "schema_validity": 1.0}))
    for i in range(4):
        cases.append(CaseResult(case_id=f"s{i}", metrics={"groundedness": 1.0, "schema_validity": 0.1}))
    clusters = cluster_failures(EvalReport(cases=cases), None, max_cases=1)
    assert len(clusters) == 2
    # First cluster eats the whole budget; the second is clamped to a single case.
    assert all(len(c["cases"]) >= 1 for c in clusters)
    assert sum(len(c["cases"]) for c in clusters) >= 2


# ===========================================================================
# LLMReflector — provider path + _validate
# ===========================================================================


def _json_provider(payload: str):
    from vincio.providers import MockProvider

    return MockProvider(responder=lambda req: payload)


def test_llm_reflector_propose_exception_falls_back():
    # lines 488-489: the propose callable raises → raw becomes None → heuristic floor
    def boom(spec, rep):
        raise RuntimeError("proposer exploded")

    r = LLMReflector(propose=boom).reflect(
        PromptSpec(name="p"), report(0.3, grounded=0.2), objectives=_objs()
    )
    assert any(e.field == "citation_policy" for e in r.edits)  # heuristic fallback ran


def test_llm_reflector_no_provider_no_propose_is_passthrough():
    # line 495: neither provider nor propose → thin pass-through to the fallback.
    r = LLMReflector().reflect(PromptSpec(name="p"), report(1.0), objectives=_objs())
    assert r.edits == []  # healthy report → fallback finds nothing


def test_llm_reflector_validate_skips_unparseable_item():
    # lines 517-518: a non-dict / malformed item raises in model_validate → skipped
    def propose(spec, rep):
        return [123, {"field": "objective", "op": "set", "value": "Be precise."}]

    r = LLMReflector(propose=propose).reflect(PromptSpec(name="p"), report(0.5), objectives=_objs())
    fields = [e.field for e in r.edits]
    assert fields == ["objective"]


def test_llm_reflector_validate_drops_field_outside_allowed_set():
    # 519->514: a perfectly parseable edit whose field is NOT in _EDIT_FIELDS is
    # validated but then excluded, so the loop continues to the next item.
    def propose(spec, rep):
        return [
            {"field": "not_a_real_field", "op": "set", "value": "x"},
            {"field": "rules", "op": "append", "value": "Stay on topic."},
        ]

    r = LLMReflector(propose=propose).reflect(PromptSpec(name="p"), report(0.5), objectives=_objs())
    assert [e.field for e in r.edits] == ["rules"]


def test_llm_reflector_provider_returns_edits_from_json():
    # Drives _reflect_via_provider end-to-end with a real (mock) provider that
    # returns a JSON diagnosis+edits object. Requires real failing cases so
    # cluster_failures produces a non-empty cluster.
    payload = (
        '{"diagnosis": "under-cited answers", '
        '"edits": [{"field": "citation_policy", "op": "set", "value": "Cite [id]."}]}'
    )
    refl = LLMReflector(_json_provider(payload), model="mock-1")
    rep = report(0.3, grounded=0.2, n=8)
    r = refl.reflect(PromptSpec(name="p"), rep, objectives=_objs(), dataset=dataset())
    assert r.diagnosis == "under-cited answers"
    assert [e.field for e in r.edits] == ["citation_policy"]


def test_llm_reflector_provider_no_clusters_falls_back():
    # line 538: cluster_failures empty (healthy report) → provider returns None →
    # _reflect_via_provider yields None → reflect uses the heuristic fallback.
    payload = '{"diagnosis": "x", "edits": [{"field": "objective", "op": "set", "value": "y"}]}'
    refl = LLMReflector(_json_provider(payload), model="mock-1")
    r = refl.reflect(PromptSpec(name="p"), report(1.0, n=8), objectives=_objs(), dataset=dataset())
    # No failures → heuristic finds nothing actionable.
    assert r.diagnosis != "x"
    assert r.edits == []


def test_llm_reflector_provider_list_payload_uses_default_diagnosis():
    # lines 560-563: parsed is a bare list → diagnosis defaults to "model-proposed edits"
    payload = '[{"field": "objective", "op": "set", "value": "Be precise."}]'
    refl = LLMReflector(_json_provider(payload), model="mock-1")
    r = refl.reflect(PromptSpec(name="p"), report(0.3, grounded=0.2, n=8),
                     objectives=_objs(), dataset=dataset())
    assert r.diagnosis == "model-proposed edits"
    assert [e.field for e in r.edits] == ["objective"]


def test_llm_reflector_provider_scalar_json_falls_back():
    # line 565: parsed is neither dict nor list (a bare number) → None → fallback
    refl = LLMReflector(_json_provider("42"), model="mock-1")
    r = refl.reflect(PromptSpec(name="p"), report(0.3, grounded=0.2, n=8),
                     objectives=_objs(), dataset=dataset())
    # Heuristic fallback fires on the low-groundedness report.
    assert any(e.field == "citation_policy" for e in r.edits)


def test_llm_reflector_provider_dict_without_edits_list_falls_back():
    # 564-565: parsed dict but "edits" is not a list → None → fallback
    refl = LLMReflector(_json_provider('{"diagnosis": "d", "edits": "oops"}'), model="mock-1")
    r = refl.reflect(PromptSpec(name="p"), report(0.3, grounded=0.2, n=8),
                     objectives=_objs(), dataset=dataset())
    assert any(e.field == "citation_policy" for e in r.edits)


def test_llm_reflector_provider_unparseable_text_falls_back():
    # extract_json fails on prose → _reflect_via_provider returns None → fallback
    refl = LLMReflector(_json_provider("not json at all"), model="mock-1")
    r = refl.reflect(PromptSpec(name="p"), report(0.3, grounded=0.2, n=8),
                     objectives=_objs(), dataset=dataset())
    assert any(e.field == "citation_policy" for e in r.edits)


def test_llm_reflector_provider_empty_edits_falls_back():
    # provider returns valid JSON but an empty edits list → fallback to heuristic
    refl = LLMReflector(_json_provider('{"diagnosis": "d", "edits": []}'), model="mock-1")
    r = refl.reflect(PromptSpec(name="p"), report(0.3, grounded=0.2, n=8),
                     objectives=_objs(), dataset=dataset())
    assert any(e.field == "citation_policy" for e in r.edits)


# ===========================================================================
# apply_edits — output_instructions + reasoning preamble branches
# ===========================================================================


def test_apply_edits_output_instructions_append_concatenates():
    # line 627: existing output_instructions + new text (not already present) → joined
    spec = PromptSpec(name="p", output_instructions="First.")
    new_spec, _ = apply_edits(
        spec, CompilerOptions(),
        [ProposedEdit(field="output_instructions", op="append", value="Second.")],
    )
    assert new_spec.output_instructions == "First. Second."


def test_apply_edits_output_instructions_skip_when_already_contained():
    # 626->599: text already in existing → no change recorded for that field
    spec = PromptSpec(name="p", output_instructions="Return only JSON.")
    new_spec, _ = apply_edits(
        spec, CompilerOptions(),
        [ProposedEdit(field="output_instructions", op="append", value="JSON")],
    )
    assert new_spec.output_instructions == "Return only JSON."


def test_apply_edits_output_instructions_set_overrides():
    spec = PromptSpec(name="p", output_instructions="old")
    new_spec, _ = apply_edits(
        spec, CompilerOptions(),
        [ProposedEdit(field="output_instructions", op="set", value="new")],
    )
    assert new_spec.output_instructions == "new"


def test_apply_edits_reasoning_preamble_not_duplicated():
    # 636->599 / 638->599: the preamble is already in rules → not appended twice
    preamble = REASONING_PREAMBLES["plan"]
    spec = PromptSpec(name="p", rules=[preamble])
    new_spec, _ = apply_edits(
        spec, CompilerOptions(),
        [ProposedEdit(field="reasoning_mode", op="set", value="plan")],
    )
    assert new_spec.reasoning_mode == "plan"
    assert new_spec.rules.count(preamble) == 1


def test_apply_edits_reasoning_mode_without_preamble_only_sets_field():
    # 636->599: an unknown reasoning mode has no preamble → rules untouched
    spec = PromptSpec(name="p", rules=["keep"])
    new_spec, _ = apply_edits(
        spec, CompilerOptions(),
        [ProposedEdit(field="reasoning_mode", op="set", value="totally_unknown_mode")],
    )
    assert new_spec.reasoning_mode == "totally_unknown_mode"
    assert new_spec.rules == ["keep"]


def test_apply_edits_reduce_examples_uses_prior_spec_update():
    # line 618: a second edit on the same field reads the in-progress spec_update,
    # not the original spec, so chained edits compose.
    spec = PromptSpec(name="p", examples=[Example(input=f"i{i}", output=f"o{i}") for i in range(6)])
    new_spec, opts = apply_edits(
        spec, CompilerOptions(),
        [
            ProposedEdit(field="examples", op="reduce_examples", value=4),
            ProposedEdit(field="examples", op="reduce_examples", value=2),
        ],
    )
    assert len(new_spec.examples) == 2
    assert opts.max_examples == 2


def test_apply_edits_reduce_examples_default_halves_when_value_none():
    spec = PromptSpec(name="p", examples=[Example(input=f"i{i}", output=f"o{i}") for i in range(4)])
    new_spec, _ = apply_edits(
        spec, CompilerOptions(),
        [ProposedEdit(field="examples", op="reduce_examples", value=None)],
    )
    assert len(new_spec.examples) == 2


def test_apply_edits_max_examples_routes_to_options():
    # lines 602-603: a "max_examples" field lands in compiler options, not the spec
    _, opts = apply_edits(
        PromptSpec(name="p"), CompilerOptions(),
        [ProposedEdit(field="max_examples", op="set", value=3)],
    )
    assert opts.max_examples == 3


def test_apply_edits_unknown_field_is_skipped():
    # line 611: a field outside _EDIT_FIELDS is ignored entirely (spec unchanged)
    spec = PromptSpec(name="p", objective="orig")
    new_spec, _ = apply_edits(
        spec, CompilerOptions(),
        [ProposedEdit(field="totally_made_up", op="set", value="x")],
    )
    assert new_spec is spec  # no spec_update accumulated → original returned


def test_apply_edits_prepend_list_field():
    # line 618: prepend on a list field puts the new value(s) first
    spec = PromptSpec(name="p", rules=["existing"])
    new_spec, _ = apply_edits(
        spec, CompilerOptions(),
        [ProposedEdit(field="rules", op="prepend", value="new")],
    )
    assert new_spec.rules == ["new", "existing"]


def test_apply_edits_scalar_objective_set():
    # line 641: a plain scalar string field (objective) is set directly
    new_spec, _ = apply_edits(
        PromptSpec(name="p", objective="old"), CompilerOptions(),
        [ProposedEdit(field="objective", op="set", value="Answer precisely.")],
    )
    assert new_spec.objective == "Answer precisely."


# ===========================================================================
# MIPROProposer — dedup + budget guards
# ===========================================================================


def test_mipro_proposer_dedupes_example_counts():
    # line 700: with <=2 examples, example_counts collapses to fewer distinct
    # values, so names repeat and the `seen` guard prevents duplicates.
    proposer = MIPROProposer()
    import random

    spec = PromptSpec(name="p", examples=[Example(input="i", output="o")])
    variants = proposer.propose(spec, CompilerOptions(), max_candidates=99, rng=random.Random(0))
    names = [v.name for v in variants]
    assert len(names) == len(set(names))  # no duplicate names survived


def test_mipro_proposer_respects_max_candidates():
    proposer = MIPROProposer()
    import random

    spec = PromptSpec(name="p", examples=[Example(input=f"i{i}", output=f"o{i}") for i in range(4)])
    variants = proposer.propose(spec, CompilerOptions(), max_candidates=2, rng=random.Random(1))
    assert len(variants) == 2


# ===========================================================================
# ReflectiveOptimizer — full-flow budget guards
# ===========================================================================


async def _strong_on_citation(variant, ds):
    spec = variant.spec
    strong = bool(spec.citation_policy) or spec.reasoning_mode == "evidence_first"
    return report(0.95 if strong else 0.5, grounded=0.95 if strong else 0.35, n=len(ds))


async def test_small_dataset_refused():
    # line 773: dataset below min coverage → refuse, returning a NaN-baseline result
    res = await ReflectiveOptimizer(_strong_on_citation).optimize(
        PromptSpec(name="p"), dataset(2), budget=8, min_dataset_coverage=4
    )
    assert res.promoted is False
    assert "dataset too small" in res.reason


async def test_reflective_strategy_promotes_and_records_trace():
    # lines 817 + 843-900: the reflective branch runs the full loop — reflect,
    # apply edits, screen on a minibatch, then full-rollout the winner.
    res = await ReflectiveOptimizer(_strong_on_citation).optimize(
        PromptSpec(name="p", objective="Answer"),
        dataset(),
        strategy="reflective",
        budget=10,
        minibatch_size=4,
        seed=7,
    )
    assert res.strategy == "reflective"
    assert res.promoted
    assert res.rounds >= 1
    assert res.reflections  # a reflection trace was recorded
    # The reflective loop screened a child and gave it a full rollout.
    phases = {h["phase"] for h in res.history}
    assert "reflect" in phases
    assert "full" in phases


async def test_reflective_skip_full_when_child_loses_minibatch():
    # lines 884-886 (skip_full branch): a child whose edits make it *worse* than
    # its parent on the screening minibatch is never given a full rollout.
    async def ev(variant, ds):
        spec = variant.spec
        # The baseline (no citation policy) is strong; any reflected child that
        # gains a citation policy / evidence-first stance is penalised → it loses
        # the screen and is skipped before a full rollout.
        edited = bool(spec.citation_policy) or spec.reasoning_mode == "evidence_first"
        # Baseline groundedness is low so the reflector still proposes an edit.
        return report(0.2 if edited else 0.9, grounded=0.3, n=len(ds))

    res = await ReflectiveOptimizer(ev).optimize(
        PromptSpec(name="p", objective="Answer"),
        dataset(),
        strategy="reflective",
        budget=10,
        minibatch_size=4,
        seed=2,
    )
    # Children were screened but each lost, so a skip_full was recorded and
    # nothing was promoted.
    assert not res.promoted
    assert any(h["phase"] == "skip_full" for h in res.history)


async def test_reflective_stale_stop_when_no_edits():
    # When the reflector keeps finding nothing actionable (healthy reports), the
    # loop accrues `stale` and stops early without consuming the whole budget.
    async def ev(variant, ds):
        return report(1.0, grounded=1.0, n=len(ds))  # nothing to fix → no edits

    res = await ReflectiveOptimizer(ev).optimize(
        PromptSpec(name="p", objective="Answer"),
        dataset(),
        strategy="reflective",
        budget=20,
        minibatch_size=4,
        seed=1,
    )
    # Baseline only consumed one evaluation; the stale guard halted further work.
    assert res.evaluations == 1
    assert not res.promoted


async def test_reflective_budget_stops_winning_child_before_full():
    # line 888: a child wins the screening minibatch, but the evaluation budget is
    # already spent on the baseline + screen, so the full rollout is skipped.
    res = await ReflectiveOptimizer(_strong_on_citation).optimize(
        PromptSpec(name="p", objective="Answer"),
        dataset(),
        strategy="reflective",
        budget=2,  # baseline (1) + one screen (1) == budget → break before full
        minibatch_size=4,
        seed=7,
    )
    assert res.evaluations == 2
    # The reflective loop reflected and screened, but never reached a full rollout.
    phases = [h["phase"] for h in res.history]
    assert "reflect" in phases
    assert "full" not in phases


async def test_mipro_budget_exhausted_before_screening_all():
    # lines 915 / 926 / 928: a tight budget forces the screen loop and the
    # full-verify loop to break on the budget, and skips below-baseline screens.
    seen = {"n": 0}

    async def ev(variant, ds):
        seen["n"] += 1
        return await _strong_on_citation(variant, ds)

    res = await ReflectiveOptimizer(ev).optimize(
        PromptSpec(name="p", objective="Answer"),
        dataset(),
        strategy="mipro",
        budget=4,
        minibatch_size=4,
        seed=3,
    )
    assert res.evaluations <= 4
    assert seen["n"] <= 4
    assert res.strategy == "mipro"


async def test_mipro_below_baseline_screen_not_promoted_to_full():
    # line 928: a screened child below the baseline fitness is never given a
    # full rollout — there should be no "full" phase for such candidates.
    async def ev(variant, ds):
        # Baseline (no edits) is the strongest; every mipro variant is weaker.
        if variant.dimensions.get("instruction") in (None, "baseline") and not variant.dimensions:
            return report(0.99, grounded=0.99, n=len(ds))
        return report(0.2, grounded=0.2, n=len(ds))

    res = await ReflectiveOptimizer(ev).optimize(
        PromptSpec(name="p", objective="Answer"),
        dataset(),
        strategy="mipro",
        budget=12,
        minibatch_size=4,
        seed=5,
    )
    # No child beat the baseline on screening, so none reached the full set.
    fulls = [h for h in res.history if h["phase"] == "full"]
    assert fulls == []
    assert not res.promoted


# ===========================================================================
# ReflectiveOptimizer._select — rejection paths (driven directly, real objects)
# ===========================================================================


def _point(name, *, acc, grounded, cost, candidate=None):
    rep = report(acc, grounded=grounded, n=8, cost=cost)
    cand = candidate or Candidate(name=name, full_report=rep, full_fitness=fitness(rep))
    cand.full_report = rep
    cand.full_fitness = fitness(rep)
    return ParetoPoint(name=name, objectives=objective_vector(rep, DEFAULT_OBJECTIVES), candidate=cand)


def _optimizer():
    return ReflectiveOptimizer(_strong_on_citation)


def test_select_no_feasible_point_under_constraints():
    # lines 949-950: an impossible accuracy constraint leaves the frontier with
    # no feasible point → select returns None → "no frontier point satisfies".
    opt = ReflectiveOptimizer(_strong_on_citation, constraints={"accuracy": 5.0})
    base = _point("base", acc=0.9, grounded=0.9, cost=0.001)
    frontier = ParetoFrontier.build([base], specs=opt.objectives)
    result = ReflectiveResult(baseline_fitness=base.candidate.full_fitness, baseline=base.candidate)
    out = opt._select(result, frontier, base.candidate)
    assert out.best is None
    assert "no frontier point satisfies" in out.reason


def test_select_baseline_is_selected_point():
    # lines 951-953: the only/knee point is the baseline → "no reflective gain".
    opt = _optimizer()
    base = _point("base", acc=0.9, grounded=0.9, cost=0.001)
    frontier = ParetoFrontier.build([base], specs=opt.objectives)
    result = ReflectiveResult(baseline_fitness=base.candidate.full_fitness, baseline=base.candidate)
    out = opt._select(result, frontier, base.candidate)
    assert out.best is None
    assert "no reflective gain" in out.reason


def test_select_point_that_neither_dominates_nor_improves():
    # lines 961-963: a non-dominated child that is strictly worse on fitness and
    # does not dominate the baseline → rejected with the "neither dominates" reason.
    opt = _optimizer()
    # Baseline: high accuracy, cheap. Child: lower accuracy but cheaper still, so
    # it stays non-dominated on the cost axis yet loses on fitness overall.
    base = _point("base", acc=0.9, grounded=0.9, cost=0.05)
    child = _point("child", acc=0.1, grounded=0.1, cost=0.0)
    # Make the child the knee by ensuring it is on the front and baseline is not
    # preferred: prefer the cheaper axis so `select` returns the child.
    opt.prefer = "cost"
    frontier = ParetoFrontier.build([base, child], specs=opt.objectives)
    result = ReflectiveResult(baseline_fitness=base.candidate.full_fitness, baseline=base.candidate)
    out = opt._select(result, frontier, base.candidate)
    assert out.best is None
    assert "neither dominates the baseline nor improves fitness" in out.reason


def test_select_significant_regression_blocks_promotion():
    # lines 984-987: the selected child improves overall fitness (much cheaper)
    # but its primary metric (lexical_overlap) significantly regresses per-case →
    # the significance gate blocks promotion.
    opt = ReflectiveOptimizer(_strong_on_citation, prefer="cost")
    base_rep = report(0.9, grounded=0.95, n=8, cost=2.0)
    base_cand = Candidate(name="base", full_report=base_rep, full_fitness=fitness(base_rep))
    base = ParetoPoint(
        name="base", objectives=objective_vector(base_rep, opt.objectives), candidate=base_cand
    )
    # Child: lower accuracy (0.6 vs 0.9, consistent across cases → significant)
    # but cost collapses from 2.0 to 0.0, lifting fitness above the baseline.
    child_rep = report(0.6, grounded=0.95, n=8, cost=0.0)
    child_cand = Candidate(name="child", full_report=child_rep, full_fitness=fitness(child_rep))
    child = ParetoPoint(
        name="child", objectives=objective_vector(child_rep, opt.objectives), candidate=child_cand
    )
    assert child_cand.full_fitness > base_cand.full_fitness  # fitness genuinely improves
    frontier = ParetoFrontier.build([base, child], specs=opt.objectives)
    result = ReflectiveResult(baseline_fitness=base_cand.full_fitness, baseline=base_cand)
    out = opt._select(result, frontier, base_cand)
    assert out.best is None
    assert out.promoted is False
    assert "regress" in out.reason.lower()
    assert out.significance is not None


def test_select_promotes_clean_improvement():
    # Happy path: child dominates baseline and passes every gate → promoted, with
    # a descriptive reason carrying the fitness delta.
    opt = _optimizer()
    base_rep = report(0.6, grounded=0.6, n=8, cost=0.001)
    base_cand = Candidate(name="base", full_report=base_rep, full_fitness=fitness(base_rep))
    base = ParetoPoint(
        name="base", objectives=objective_vector(base_rep, opt.objectives), candidate=base_cand
    )
    child_rep = report(0.95, grounded=0.95, n=8, cost=0.001)
    child_cand = Candidate(name="child", full_report=child_rep, full_fitness=fitness(child_rep))
    child = ParetoPoint(
        name="child", objectives=objective_vector(child_rep, opt.objectives), candidate=child_cand
    )
    frontier = ParetoFrontier.build([base, child], specs=opt.objectives)
    result = ReflectiveResult(
        baseline_fitness=base_cand.full_fitness, baseline=base_cand, evaluations=3
    )
    out = opt._select(result, frontier, base_cand)
    assert out.promoted is True
    assert out.best is child_cand
    assert "reflective promotion" in out.reason


def test_select_safety_gate_blocks_unsafe_winner():
    # _promotion_safe path: a child that improves fitness but regresses safety is
    # blocked before the significance gate, with promoted=False.
    opt = ReflectiveOptimizer(_strong_on_citation, gates={"safety": ">=0.9"})
    base_rep = report(0.6, grounded=0.6, n=8, safety=1.0)
    base_cand = Candidate(name="base", full_report=base_rep, full_fitness=fitness(base_rep))
    base = ParetoPoint(
        name="base", objectives=objective_vector(base_rep, opt.objectives), candidate=base_cand
    )
    child_rep = report(0.95, grounded=0.95, n=8, safety=0.4)
    child_cand = Candidate(name="child", full_report=child_rep, full_fitness=fitness(child_rep))
    child = ParetoPoint(
        name="child", objectives=objective_vector(child_rep, opt.objectives), candidate=child_cand
    )
    frontier = ParetoFrontier.build([base, child], specs=opt.objectives)
    result = ReflectiveResult(baseline_fitness=base_cand.full_fitness, baseline=base_cand)
    out = opt._select(result, frontier, base_cand)
    assert out.promoted is False
    assert "safety" in out.reason.lower()
