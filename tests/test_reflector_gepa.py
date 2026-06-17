"""1.10 — the real provider-backed reflective optimizer (GEPA proper):
failure clustering, the LLMReflector provider path, and its offline fallback."""

import json
import warnings

from vincio.evals import Dataset, EvalCase, EvalReport
from vincio.evals.reports import CaseResult
from vincio.optimize.pareto import objectives_from_weights
from vincio.optimize.reflective import (
    HeuristicReflector,
    LLMReflector,
    ProposedEdit,
    cluster_failures,
)
from vincio.optimize.search import FitnessWeights
from vincio.prompts.templates import PromptSpec
from vincio.providers import MockProvider

warnings.simplefilter("ignore")

_OBJ = objectives_from_weights(FitnessWeights())


def _report() -> EvalReport:
    return EvalReport(
        name="r",
        dataset="d",
        cases=[
            CaseResult(case_id="c1", metrics={"lexical_overlap": 0.3, "groundedness": 0.2,
                                              "schema_validity": 1.0}, output_text="wrong, no cite"),
            CaseResult(case_id="c2", metrics={"lexical_overlap": 0.9, "groundedness": 0.95,
                                              "schema_validity": 0.4}, output_text="{bad json"),
            CaseResult(case_id="c3", metrics={"lexical_overlap": 0.95, "groundedness": 0.95,
                                              "schema_validity": 1.0}, output_text="good"),
        ],
    )


def _dataset() -> Dataset:
    return Dataset(name="d", cases=[
        EvalCase(id="c1", input="what is the refund window?", expected="30 days"),
        EvalCase(id="c2", input="extract the totals", expected={"total": 5}),
        EvalCase(id="c3", input="say hi", expected="hi"),
    ])


def _spec() -> PromptSpec:
    return PromptSpec(name="qa", objective="Answer questions from the evidence.")


class TestClusterFailures:
    def test_groups_by_dominant_failure_mode(self):
        clusters = cluster_failures(_report(), _dataset())
        modes = {c["mode"]: c["count"] for c in clusters}
        # c1 worst on groundedness, c2 worst on schema; c3 passes everything.
        assert modes.get("groundedness") == 1
        assert modes.get("schema_validity") == 1
        assert "c3" not in json.dumps(clusters)  # passing case excluded

    def test_joins_dataset_for_input_and_expected(self):
        clusters = cluster_failures(_report(), _dataset())
        grounded = next(c for c in clusters if c["mode"] == "groundedness")
        case = grounded["cases"][0]
        assert case["input"] == "what is the refund window?"
        assert case["expected"] == "30 days"

    def test_empty_when_no_report(self):
        assert cluster_failures(None, _dataset()) == []


class TestLLMReflectorProviderPath:
    def test_provider_proposes_validated_edits(self):
        def responder(request):
            # The reflector must have surfaced the failing case in the prompt.
            text = "\n".join(m.text for m in request.messages)
            assert "groundedness" in text and "refund window" in text
            return json.dumps({
                "diagnosis": "answers were under-cited",
                "edits": [{"field": "citation_policy", "op": "set",
                           "value": "Cite [Ek] for every claim.", "rationale": "low groundedness"}],
            })

        reflector = LLMReflector(MockProvider(responder=responder), "mock-1")
        reflection = reflector.reflect(_spec(), _report(), objectives=_OBJ, dataset=_dataset())
        assert reflection.diagnosis == "answers were under-cited"
        assert any(e.field == "citation_policy" for e in reflection.edits)

    def test_invalid_fields_are_filtered(self):
        def responder(request):
            return json.dumps({"edits": [
                {"field": "not_a_field", "op": "set", "value": "x"},
                {"field": "reasoning_mode", "op": "set", "value": "evidence_first"},
            ]})

        reflector = LLMReflector(MockProvider(responder=responder), "mock-1")
        reflection = reflector.reflect(_spec(), _report(), objectives=_OBJ, dataset=_dataset())
        fields = {e.field for e in reflection.edits}
        assert fields == {"reasoning_mode"}

    def test_garbage_response_falls_back_to_heuristic(self):
        reflector = LLMReflector(MockProvider(responder=lambda r: "not json at all"), "mock-1")
        reflection = reflector.reflect(_spec(), _report(), objectives=_OBJ, dataset=_dataset())
        # Heuristic floor still proposes a grounded-ness fix from the low signal.
        assert reflection.edits
        assert reflection.diagnosis != "model-proposed edits"

    def test_no_provider_no_callable_is_heuristic(self):
        reflector = LLMReflector()
        reflection = reflector.reflect(_spec(), _report(), objectives=_OBJ, dataset=_dataset())
        heuristic = HeuristicReflector().reflect(_spec(), _report(), objectives=_OBJ)
        assert [e.field for e in reflection.edits] == [e.field for e in heuristic.edits]

    def test_explicit_propose_callable_overrides_provider(self):
        reflector = LLMReflector(
            propose=lambda spec, report: [{"field": "rules", "op": "append", "value": "Be concise."}]
        )
        reflection = reflector.reflect(_spec(), _report(), objectives=_OBJ)
        assert reflection.edits == [ProposedEdit(field="rules", op="append", value="Be concise.")]


class TestAppWiring:
    def test_reflective_optimize_llm_offline_falls_back(self, tmp_path):
        from vincio import ContextApp, VincioConfig

        config = VincioConfig()
        config.storage.metadata = f"sqlite:///{tmp_path}/v.db"
        config.observability.exporter = "memory"
        app = ContextApp(name="r", provider=MockProvider(), model="mock-1", config=config)
        dataset = Dataset(name="d", cases=[
            EvalCase(id=f"c{i}", input="q", expected="a") for i in range(6)
        ])
        # reflector="llm" wires the app provider; MockProvider's non-JSON text
        # makes it fall back to the heuristic — the run must still complete.
        result = app.reflective_optimize(dataset, budget=4, reflector="llm")
        assert result.strategy == "reflective"
        assert isinstance(result.reflections, list)
