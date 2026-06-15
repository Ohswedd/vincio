"""1.4 — Reflective optimization & the data flywheel.

Covers the reflective optimizer (GEPA + MIPRO strategies), the distillation /
fine-tune flywheel (grounded export + gated teacher→student loop), learned
prompt compression (LLMLingua pass + faithfulness gate), and optimizer-judge
calibration — all offline and deterministic.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vincio import ContextApp, VincioConfig
from vincio.context.compression import extractive_compress
from vincio.context.llmlingua import (
    LLMLinguaCompressor,
    TokenImportanceScorer,
    compression_faithfulness,
    faithfulness_preserved,
    salient_units,
)
from vincio.core.tokens import count_tokens
from vincio.core.types import EvidenceItem
from vincio.evals import Dataset, EvalCase, EvalReport
from vincio.evals.datasets import EvalCase as Case
from vincio.evals.judges import GEvalJudge
from vincio.evals.metrics import RunOutput
from vincio.evals.reports import CaseResult
from vincio.optimize import (
    BootstrapFinetune,
    CompressionTuner,
    FitnessWeights,
    HeuristicReflector,
    ImprovementLoop,
    JudgeCalibrator,
    JudgeStepReflector,
    LLMReflector,
    ProposedEdit,
    ReflectiveOptimizer,
    ReflectiveResult,
    TrainingExample,
    TrainingSet,
    apply_edits,
    export_training_set,
)
from vincio.optimize.search import OptimizationResult
from vincio.prompts.compiler import CompilerOptions
from vincio.prompts.optimizers import PromptVariant
from vincio.prompts.templates import PromptSpec
from vincio.providers import MockProvider

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def report(quality: float, *, grounded: float | None = None, n: int = 6, **extra) -> EvalReport:
    metrics = {
        "semantic_similarity": quality,
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


def fake_trace(tid, inp, out, *, evidence=None, status="ok", feedback=None):
    return SimpleNamespace(
        id=tid,
        run_id=tid,
        session_id=None,
        status=status,
        feedback=feedback or [],
        attributes={
            "input": inp,
            "output": out,
            "evidence": [e.model_dump() for e in (evidence or [])],
        },
    )


# ===========================================================================
# Reflective optimizer
# ===========================================================================


class TestApplyEdits:
    def test_set_scalar_and_append_list(self):
        spec = PromptSpec(name="p", rules=["r1"])
        edits = [
            ProposedEdit(field="citation_policy", op="set", value="Cite [id]."),
            ProposedEdit(field="rules", op="append", value="r2"),
        ]
        new_spec, _ = apply_edits(spec, CompilerOptions(), edits)
        assert new_spec.citation_policy == "Cite [id]."
        assert new_spec.rules == ["r1", "r2"]
        # Original spec is untouched (immutability).
        assert spec.citation_policy == ""

    def test_reduce_examples_shrinks_and_sets_max(self):
        from vincio.core.types import Example

        spec = PromptSpec(name="p", examples=[Example(input=f"i{i}", output=f"o{i}") for i in range(4)])
        new_spec, opts = apply_edits(
            spec, CompilerOptions(), [ProposedEdit(field="examples", op="reduce_examples", value=2)]
        )
        assert len(new_spec.examples) == 2
        assert opts.max_examples == 2

    def test_reasoning_mode_injects_preamble(self):
        spec = PromptSpec(name="p")
        new_spec, _ = apply_edits(
            spec, CompilerOptions(), [ProposedEdit(field="reasoning_mode", op="set", value="plan")]
        )
        assert new_spec.reasoning_mode == "plan"
        # The mode's preamble is added to rules so the prompt actually changes.
        assert any("plan" in r.lower() for r in new_spec.rules)

    def test_unknown_field_ignored(self):
        spec = PromptSpec(name="p", objective="orig")
        new_spec, _ = apply_edits(
            spec, CompilerOptions(), [ProposedEdit(field="nonexistent", op="set", value="x")]
        )
        assert new_spec.objective == "orig"

    def test_format_routes_to_compiler_options(self):
        new_spec, opts = apply_edits(
            PromptSpec(name="p"), CompilerOptions(), [ProposedEdit(field="format", op="set", value="xml")]
        )
        assert opts.format == "xml"


class TestHeuristicReflector:
    def test_low_groundedness_adds_citation_policy(self):
        r = HeuristicReflector().reflect(
            PromptSpec(name="p"), report(0.9, grounded=0.3), objectives=_objs()
        )
        assert any(e.field == "citation_policy" for e in r.edits)
        assert r.failures_observed == 0  # accuracy is high → no accuracy failures

    def test_low_accuracy_adds_reasoning(self):
        r = HeuristicReflector().reflect(
            PromptSpec(name="p", reasoning_mode="direct"),
            report(0.3, grounded=1.0),
            objectives=_objs(),
        )
        assert any(e.field == "reasoning_mode" for e in r.edits)

    def test_low_safety_adds_policy_first(self):
        r = HeuristicReflector().reflect(
            PromptSpec(name="p"), report(0.4, grounded=0.4, safety=0.5), objectives=_objs()
        )
        assert r.edits[0].field == "safety_policies"

    def test_healthy_report_no_edits(self):
        r = HeuristicReflector().reflect(PromptSpec(name="p"), report(1.0), objectives=_objs())
        assert r.edits == []
        assert "no actionable" in r.diagnosis

    def test_cost_ceiling_trims_examples(self):
        from vincio.core.types import Example

        spec = PromptSpec(name="p", examples=[Example(input=f"i{i}", output=f"o{i}") for i in range(4)])
        r = HeuristicReflector(cost_ceiling=0.0005).reflect(
            spec, report(1.0, cost=0.01), objectives=_objs()
        )
        assert any(e.op == "reduce_examples" for e in r.edits)


class TestLLMReflector:
    def test_falls_back_without_propose(self):
        r = LLMReflector().reflect(PromptSpec(name="p"), report(0.9, grounded=0.3), objectives=_objs())
        assert any(e.field == "citation_policy" for e in r.edits)

    def test_falls_back_on_error(self):
        def boom(spec, rep):
            raise RuntimeError("provider down")

        r = LLMReflector(boom).reflect(PromptSpec(name="p"), report(0.9, grounded=0.3), objectives=_objs())
        assert r.edits  # used the heuristic fallback

    def test_honors_valid_model_edits(self):
        def propose(spec, rep):
            return [{"field": "objective", "op": "set", "value": "Be precise."},
                    {"field": "bogus", "op": "set", "value": "x"}]

        r = LLMReflector(propose).reflect(PromptSpec(name="p"), report(0.5), objectives=_objs())
        fields = {e.field for e in r.edits}
        assert "objective" in fields and "bogus" not in fields


def _objs():
    from vincio.optimize import DEFAULT_OBJECTIVES

    return DEFAULT_OBJECTIVES


class TestReflectiveOptimizer:
    async def _evaluate(self):
        # Quality is high when the spec gains a citation policy or evidence-first
        # reasoning; otherwise weak. Gives the reflector a real signal.
        async def ev(variant, ds):
            spec = variant.spec
            strong = bool(spec.citation_policy) or spec.reasoning_mode == "evidence_first"
            q = 0.95 if strong else 0.5
            return report(q, grounded=0.95 if strong else 0.35, n=len(ds))

        return ev

    async def test_promotes_reflective_winner(self):
        opt = ReflectiveOptimizer(await self._evaluate())
        spec = PromptSpec(name="p", objective="Answer")
        res = await opt.optimize(spec, dataset(), budget=10, minibatch_size=4, seed=7)
        assert res.promoted
        assert isinstance(res, ReflectiveResult)
        assert isinstance(res, OptimizationResult)  # drop-in for the loop
        assert isinstance(res.best.payload, PromptVariant)
        assert res.frontier is not None and len(res.frontier.front) >= 1

    async def test_deterministic_under_seed(self):
        ev = await self._evaluate()
        spec = PromptSpec(name="p", objective="Answer")
        a = await ReflectiveOptimizer(ev).optimize(spec, dataset(), budget=10, minibatch_size=4, seed=7)
        b = await ReflectiveOptimizer(ev).optimize(spec, dataset(), budget=10, minibatch_size=4, seed=7)
        assert a.promoted == b.promoted
        assert (a.best.params if a.best else None) == (b.best.params if b.best else None)

    async def test_evaluation_budget_is_hard_bound(self):
        calls = {"n": 0}

        async def ev(variant, ds):
            calls["n"] += 1
            return report(0.5, grounded=0.3, n=len(ds))

        res = await ReflectiveOptimizer(ev).optimize(
            PromptSpec(name="p"), dataset(), budget=6, minibatch_size=4, seed=1
        )
        assert res.evaluations <= 6
        assert calls["n"] <= 6

    async def test_small_dataset_refused(self):
        res = await ReflectiveOptimizer(await self._evaluate()).optimize(
            PromptSpec(name="p"), dataset(2), budget=8
        )
        assert not res.promoted
        assert "dataset too small" in res.reason

    async def test_mipro_strategy_promotes(self):
        res = await ReflectiveOptimizer(await self._evaluate()).optimize(
            PromptSpec(name="p", objective="Answer"), dataset(), strategy="mipro", budget=12,
            minibatch_size=4, seed=3,
        )
        assert res.strategy == "mipro"
        # The strategy ran joint proposals and verified some on the full set.
        assert any(h["phase"] == "screen" for h in res.history)

    async def test_safety_regression_blocks_promotion(self):
        # A child raises accuracy but the evaluator regresses safety → blocked.
        async def ev(variant, ds):
            if variant.spec.citation_policy:
                return report(0.99, grounded=0.99, n=len(ds), safety=0.5)
            return report(0.5, grounded=0.3, n=len(ds), safety=1.0)

        res = await ReflectiveOptimizer(ev).optimize(
            PromptSpec(name="p"), dataset(), budget=10, minibatch_size=4
        )
        assert not res.promoted
        assert "safety" in res.reason


class TestReflectiveLoopIntegration:
    def _app(self, tmp_path):
        cfg = VincioConfig()
        cfg.storage.metadata = "memory://"
        cfg.observability.exporter = "memory"
        cfg.security.audit_log = False

        def responder(req):
            text = "\n".join(m.text for m in req.messages)
            if "plan the steps" in text.lower() or "briefly plan" in text.lower():
                return "The Pro plan refund window is 30 days."
            return "I am not sure."

        return ContextApp(name="refl", provider=MockProvider(responder=responder), model="mock-1", config=cfg)

    def test_improvement_loop_reflective_promotes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = self._app(tmp_path)
        ds = Dataset(
            name="refunds",
            cases=[EvalCase(id=f"c{i}", input="What is the Pro plan refund window?",
                            expected="The Pro plan refund window is 30 days.") for i in range(6)],
        )
        loop = ImprovementLoop(
            app, metrics=["semantic_similarity", "cost", "latency"],
            weights=FitnessWeights(latency=0.0), optimizer="reflective", experiment="refl_exp",
        )
        result = loop.run(dataset=ds, max_variants=8, subset_size=3)
        assert result.promoted
        assert result.promoted_ref is not None
        assert result.optimization.strategy == "reflective"

    def test_app_reflective_optimize_apply(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = self._app(tmp_path)
        before = app.prompt_spec.spec_hash
        ds = Dataset(
            name="d",
            cases=[EvalCase(id=f"c{i}", input="What is the Pro plan refund window?",
                            expected="The Pro plan refund window is 30 days.") for i in range(6)],
        )
        res = app.reflective_optimize(
            ds, metrics=["semantic_similarity", "cost", "latency"],
            weights=FitnessWeights(latency=0.0), budget=8, minibatch_size=3, apply=True,
        )
        assert res.promoted
        assert app.prompt_spec.spec_hash != before  # winner applied


# ===========================================================================
# Distillation / fine-tune flywheel
# ===========================================================================

REFUND_EVIDENCE = [
    EvidenceItem(id="D1:C0", source_id="D1",
                 text="Customers on the Pro plan may request refunds within 30 days of purchase.",
                 provenance=0.9)
]


class TestExportTrainingSet:
    def test_keeps_grounded_drops_ungrounded(self):
        traces = [
            fake_trace("t1", "Refund window?", "The Pro plan refund window is 30 days.", evidence=REFUND_EVIDENCE),
            fake_trace("t2", "Mascot?", "The mascot is a purple axolotl with 12 legs.", evidence=REFUND_EVIDENCE),
        ]
        ts = export_training_set(traces, require_grounding=True, min_support=0.4)
        assert len(ts) == 1
        assert ts.metadata["dropped_ungrounded"] == 1
        assert ts.grounded_fraction == 1.0
        assert ts.examples[0].provenance["trace_id"] == "t1"

    def test_dedupes_identical_io(self):
        traces = [
            fake_trace("t1", "Refund?", "30 days for Pro.", evidence=REFUND_EVIDENCE),
            fake_trace("t2", "Refund?", "30 days for Pro.", evidence=REFUND_EVIDENCE),
        ]
        ts = export_training_set(traces, require_grounding=False)
        assert len(ts) == 1

    def test_feedback_filter(self):
        ok = fake_trace("t1", "q", "30 days for the Pro plan.", evidence=REFUND_EVIDENCE,
                        feedback=[SimpleNamespace(score=1.0)])
        bad = fake_trace("t2", "q2", "30 days for the Pro plan.", evidence=REFUND_EVIDENCE,
                         feedback=[SimpleNamespace(score=0.0)])
        ts = export_training_set([ok, bad], min_feedback_score=0.5, require_grounding=False)
        ids = {e.provenance["trace_id"] for e in ts.examples}
        assert ids == {"t1"}

    def test_prefers_untruncated_capture(self):
        t = fake_trace("t1", "short", "trunc", evidence=REFUND_EVIDENCE)
        t.attributes["input_full"] = "the full input question about refunds"
        t.attributes["output_full"] = "The Pro plan refund window is 30 days of purchase."
        ts = export_training_set([t], require_grounding=False)
        msgs = ts.examples[0].messages
        assert any("full input" in m["content"] for m in msgs)
        assert any("30 days of purchase" in m["content"] for m in msgs)

    def test_jsonl_formats(self):
        ex = TrainingExample(messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ])
        ts = TrainingSet(name="t", examples=[ex])
        openai = ex.to_record("openai")
        assert openai["messages"][0]["role"] == "system"
        anthropic = ex.to_record("anthropic")
        assert anthropic["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in anthropic["messages"])
        assert ts.to_jsonl(format="openai").count("\n") == 0  # one example, no trailing newline

    def test_save_writes_file(self, tmp_path):
        ts = TrainingSet(name="t", examples=[TrainingExample(messages=[{"role": "user", "content": "x"}])])
        path = ts.save(tmp_path / "train.jsonl", format="openai")
        assert path.read_text(encoding="utf-8").strip().startswith("{")


class TestBootstrapFinetune:
    def _evaluator(self, quality_by_model):
        async def ev(model, ds):
            q, cost = quality_by_model[model]
            return report(q, grounded=q, n=len(ds), cost=cost)

        return ev

    @pytest.fixture()
    def grounded_set(self):
        return export_training_set(
            [fake_trace("t1", "q", "The Pro plan refund window is 30 days.", evidence=REFUND_EVIDENCE)],
            require_grounding=True, min_support=0.4,
        )

    async def test_promotes_cheaper_student_holding_quality(self, grounded_set):
        loop = BootstrapFinetune(
            self._evaluator({"teacher": (0.95, 0.01), "student": (0.93, 0.002)}), min_quality_ratio=0.9
        )
        res = await loop.distill(grounded_set, dataset(), teacher="teacher", student="student")
        assert res.promoted
        assert res.cascade.rungs[0].model == "student"
        assert res.cascade.rungs[1].model == "teacher"
        assert res.cost_savings > 0

    async def test_rejects_quality_drop(self, grounded_set):
        loop = BootstrapFinetune(
            self._evaluator({"teacher": (0.95, 0.01), "weak": (0.5, 0.002)}), min_quality_ratio=0.95
        )
        res = await loop.distill(grounded_set, dataset(), teacher="teacher", student="weak")
        assert not res.promoted and "quality" in res.reason

    async def test_rejects_not_cheaper(self, grounded_set):
        loop = BootstrapFinetune(
            self._evaluator({"teacher": (0.95, 0.002), "student": (0.95, 0.01)}), min_quality_ratio=0.9
        )
        res = await loop.distill(grounded_set, dataset(), teacher="teacher", student="student")
        assert not res.promoted and "cheaper" in res.reason

    async def test_rejects_when_no_cost_signal(self, grounded_set):
        # Unpriced/mock models report cost 0 — a cost win cannot be verified, so
        # the student must not be promoted on quality alone.
        loop = BootstrapFinetune(
            self._evaluator({"teacher": (0.95, 0.0), "student": (0.95, 0.0)}), min_quality_ratio=0.9
        )
        res = await loop.distill(grounded_set, dataset(), teacher="teacher", student="student")
        assert not res.promoted and "cost signal" in res.reason

    async def test_trainer_hook_used(self, grounded_set):
        seen = {}

        async def trainer(ts, base):
            seen["base"] = base
            return f"{base}-ft"

        loop = BootstrapFinetune(
            self._evaluator({"teacher": (0.95, 0.01), "student-ft": (0.93, 0.002)}),
            min_quality_ratio=0.9, trainer=trainer,
        )
        res = await loop.distill(grounded_set, dataset(), teacher="teacher", student="student")
        assert seen["base"] == "student"
        assert res.trained_student == "student-ft"
        assert res.promoted

    async def test_empty_training_set_refused(self):
        loop = BootstrapFinetune(self._evaluator({"teacher": (0.9, 0.01), "student": (0.9, 0.002)}))
        res = await loop.distill(TrainingSet(), dataset(), teacher="teacher", student="student")
        assert not res.promoted and "no grounded" in res.reason

    async def test_same_model_refused(self, grounded_set):
        loop = BootstrapFinetune(self._evaluator({"teacher": (0.9, 0.01)}))
        res = await loop.distill(grounded_set, dataset(), teacher="teacher", student="teacher")
        assert not res.promoted and "same model" in res.reason


def fake_run(rid, inp, out, *, evidence=None, status="succeeded", trace_id="tr"):
    return SimpleNamespace(
        run_id=rid, trace_id=trace_id, status=status, raw_text=out, output=out,
        evidence=list(evidence or []), citations=[], metadata={"input": inp},
    )


class TestExportTrainingSetFromRuns:
    def test_faithful_grounded_export_no_flag(self):
        from vincio.optimize import export_training_set_from_runs

        runs = [
            fake_run("r1", "Refund window?", "The Pro plan refund window is 30 days.", evidence=REFUND_EVIDENCE),
            fake_run("r2", "Mascot?", "The mascot is a purple axolotl with 12 legs.", evidence=REFUND_EVIDENCE),
        ]
        ts = export_training_set_from_runs(runs, require_grounding=True, min_support=0.4)
        assert ts.metadata["source"] == "runs"
        assert len(ts) == 1 and ts.metadata["dropped_ungrounded"] == 1
        assert ts.examples[0].provenance["run_id"] == "r1"
        assert ts.grounded_fraction == 1.0

    def test_dedupes_and_filters_status(self):
        from vincio.optimize import export_training_set_from_runs

        runs = [
            fake_run("r1", "q", "30 days for Pro.", evidence=REFUND_EVIDENCE),
            fake_run("r2", "q", "30 days for Pro.", evidence=REFUND_EVIDENCE),  # dup
            fake_run("r3", "q2", "30 days for Pro.", evidence=REFUND_EVIDENCE, status="failed"),
        ]
        ts = export_training_set_from_runs(runs, require_grounding=False)
        assert len(ts) == 1  # dup collapsed, failed dropped

    def test_uses_full_untruncated_output(self):
        from vincio.optimize import export_training_set_from_runs

        long_answer = "The Pro plan refund window is 30 days. " + "Extra detail. " * 60
        runs = [fake_run("r1", "q", long_answer, evidence=REFUND_EVIDENCE)]
        ts = export_training_set_from_runs(runs, require_grounding=True, min_support=0.4)
        # The full answer (well over the 500-char span truncation) is preserved.
        assert len(ts.examples[0].messages[-1]["content"]) > 500


class TestDistillAppIntegration:
    def test_export_from_runs_flag_free(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = VincioConfig()
        cfg.storage.metadata = "memory://"
        cfg.observability.exporter = "memory"
        cfg.security.audit_log = False
        app = ContextApp(
            name="d", provider=MockProvider(responder=lambda r: "The Pro plan refund window is 30 days."),
            model="teacher", config=cfg,
        )
        app.pending_evidence = list(REFUND_EVIDENCE)
        # No enable_training_capture() — RunResults are faithful by construction.
        results = [app.run("What is the Pro plan refund window?")]
        assert results[0].metadata.get("input")  # runtime stamped the input
        ts = app.export_training_set(runs=results, require_grounding=True, min_support=0.4)
        assert len(ts) >= 1 and ts.grounded_fraction == 1.0
        assert ts.metadata["source"] == "runs"

    async def test_streaming_capture_records_full_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = VincioConfig()
        cfg.storage.metadata = "memory://"
        cfg.observability.exporter = "memory"
        cfg.security.audit_log = False
        app = ContextApp(
            name="s", provider=MockProvider(responder=lambda r: "The Pro plan refund window is 30 days."),
            model="mock-1", config=cfg,
        )
        app.enable_training_capture()
        app.pending_evidence = list(REFUND_EVIDENCE)
        async for ev in app.astream("Refund window for Pro plan?"):
            if ev.type == "done":
                break
        captured = [t for t in app.tracer.exporter.traces if t.attributes.get("output_full")]
        assert captured, "streaming run should record full output when training_capture is on"
        assert captured[-1].attributes.get("evidence")

    def test_export_and_distill(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = VincioConfig()
        cfg.storage.metadata = "memory://"
        cfg.observability.exporter = "memory"
        cfg.security.audit_log = False

        def responder(req):
            return "The Pro plan refund window is 30 days."

        app = ContextApp(name="distill", provider=MockProvider(responder=responder), model="teacher", config=cfg)
        app.enable_training_capture()
        app.pending_evidence = list(REFUND_EVIDENCE)  # cite-able evidence on the run
        app.run("What is the Pro plan refund window?")
        ts = app.export_training_set(require_grounding=False)
        assert len(ts) >= 1
        path = tmp_path / "train.jsonl"
        ts.save(path, format="openai")
        assert path.is_file()


# ===========================================================================
# Learned prompt compression
# ===========================================================================

LONG_TEXT = (
    "The Pro plan offers a refund window of 30 days from the date of purchase. "
    "Customers who are not satisfied may contact support to request a full refund. "
    "The Enterprise plan, by contrast, provides a 90 day evaluation period and onboarding. "
    "All refunds are processed within 5 business days to the original payment method."
)


class TestLLMLinguaCompressor:
    def test_hits_budget_and_keeps_numbers(self):
        budget = max(8, count_tokens(LONG_TEXT) // 2)
        res = LLMLinguaCompressor()(LONG_TEXT, "Pro plan refund window", budget)
        assert res.method == "llmlingua"
        assert res.compressed_tokens <= budget
        assert "30" in res.text  # query-relevant number protected

    def test_passthrough_when_under_budget(self):
        res = LLMLinguaCompressor()("short text", "q", 100)
        assert res.method == "none"
        assert res.text == "short text"

    def test_protected_tokens_never_dropped(self):
        text = "alpha beta gamma delta epsilon $4,200 zeta eta theta iota kappa"
        res = LLMLinguaCompressor()(text, "amount", count_tokens(text) // 2)
        assert "$4,200" in res.text

    def test_multiword_citation_kept_atomic(self):
        # A bracketed citation with spaces must survive whole, not fragmented.
        text = ("According to the detailed quarterly report per [Smith and Jones 2020] "
                "the revenue grew substantially across every region last year.")
        res = LLMLinguaCompressor()(text, "revenue", count_tokens(text) // 2)
        assert "[Smith and Jones 2020]" in res.text

    def test_drop_in_signature_matches_extractive(self):
        # Both callables accept (text, query, max_tokens) positionally.
        budget = count_tokens(LONG_TEXT) // 2
        a = extractive_compress(LONG_TEXT, "refund", budget)
        b = LLMLinguaCompressor()(LONG_TEXT, "refund", budget)
        assert a.compressed_tokens <= budget and b.compressed_tokens <= budget


class TestTokenImportanceScorer:
    def test_protected_scores_top_stopwords_bottom(self):
        scorer = TokenImportanceScorer()
        scores = scorer.score(["the", "30", "Refund"], "refund")
        assert scores[0] < 0.2  # stopword
        assert scores[1] == 1.0  # number protected
        assert scores[2] > 0.5  # query-relevant entity

    def test_learned_hook_overrides(self):
        scorer = TokenImportanceScorer(learned=lambda tokens, q: [0.5] * len(tokens))
        assert scorer.score(["a", "b"], "q") == [0.5, 0.5]


class TestFaithfulness:
    def test_salient_units_extracts_numbers_entities_citations(self):
        units = salient_units("Pro plan 30 days [D1:C0]")
        assert "30" in units and "pro" in units and "[d1:c0]" in units

    def test_faithfulness_full_when_preserved(self):
        assert compression_faithfulness("Pro plan 30 days", "Pro plan 30 days") == 1.0

    def test_faithfulness_drops_when_number_lost(self):
        assert compression_faithfulness("Pro plan 30 days", "Pro plan days") < 1.0

    def test_faithfulness_preserved_gate(self):
        assert faithfulness_preserved(["The Pro plan refund is 30 days."], "Pro plan refund 30 days", threshold=0.8)
        assert not faithfulness_preserved(["The Pro plan refund is 30 days."], "the plan refund window", threshold=0.8)


class TestCompilerCompressorIntegration:
    def test_default_is_extractive(self):
        from vincio.context.compiler import ContextCompiler

        assert ContextCompiler().compressor is extractive_compress

    def test_pluggable_compressor_used(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = VincioConfig()
        cfg.storage.metadata = "memory://"
        cfg.observability.exporter = "memory"
        cfg.security.audit_log = False
        app = ContextApp(name="c", provider=MockProvider(), model="mock-1", config=cfg)
        app.use_learned_compression()
        assert isinstance(app.context_compiler.compressor, LLMLinguaCompressor)


class TestCompressionTuner:
    def _evaluator(self, baseline, learned):
        async def ev(compressor, ds):
            cfg = learned if compressor is not None else baseline
            return report(cfg["q"], n=len(ds), **{"faithfulness": cfg["f"], "input_tokens": cfg["t"]})

        return ev

    async def test_adopts_when_faithful_and_cheaper(self):
        tuner = CompressionTuner(
            self._evaluator({"q": 1.0, "f": 1.0, "t": 100.0}, {"q": 0.99, "f": 0.95, "t": 60.0})
        )
        res, chosen = await tuner.tune(LLMLinguaCompressor(), dataset())
        assert res.adopted and chosen is not None
        assert res.token_savings > 0

    async def test_rejects_when_faithfulness_drops(self):
        tuner = CompressionTuner(
            self._evaluator({"q": 1.0, "f": 1.0, "t": 100.0}, {"q": 0.99, "f": 0.5, "t": 60.0}),
            min_faithfulness=0.9,
        )
        res, chosen = await tuner.tune(LLMLinguaCompressor(), dataset())
        assert not res.adopted and chosen is None and "faithfulness" in res.reason

    async def test_rejects_when_no_savings(self):
        tuner = CompressionTuner(
            self._evaluator({"q": 1.0, "f": 1.0, "t": 100.0}, {"q": 1.0, "f": 1.0, "t": 110.0})
        )
        res, chosen = await tuner.tune(LLMLinguaCompressor(), dataset())
        assert not res.adopted and chosen is None

    async def test_rejects_when_no_token_signal(self):
        # No input_tokens metric in the report → a token win cannot be verified.
        async def ev(compressor, ds):
            learned = compressor is not None
            return report(0.99 if learned else 1.0, n=len(ds), **{"faithfulness": 1.0})

        res, chosen = await CompressionTuner(ev).tune(LLMLinguaCompressor(), dataset())
        assert not res.adopted and chosen is None and "token signal" in res.reason


# ===========================================================================
# Optimizer-judge calibration
# ===========================================================================


def _grounding_sensitive_judge():
    def responder(req):
        text = "\n".join(m.text for m in req.messages)
        grounded_steps = "does not support" in text.lower()
        is_good = "GOOD_OUTPUT" in text
        if grounded_steps:
            return {"score": 5 if is_good else 1, "reasoning": "r"}
        return {"score": 3, "reasoning": "r"}

    return GEvalJudge(
        MockProvider(responder=responder), model="mock-1", criteria="faithful?",
        steps=["Read the output.", "Give a 1-5 score."],
    )


def _labeled_samples(n=6):
    samples = []
    for i in range(n):
        good = i % 2 == 0
        samples.append((Case(id=f"c{i}", input="q"),
                        RunOutput(raw_text="GOOD_OUTPUT" if good else "BAD_OUTPUT"),
                        1.0 if good else 0.0))
    return samples


class TestJudgeCalibration:
    def test_step_reflector_proposes_distinct(self):
        proposals = JudgeStepReflector().propose(["Read the output."], budget=3)
        assert len(proposals) == 3
        assert all(len(p.steps) > 1 for p in proposals)
        assert len({p.name for p in proposals}) == 3

    async def test_adopts_better_procedure_and_lifts_gating(self):
        judge = _grounding_sensitive_judge()
        res = await JudgeCalibrator(judge).acalibrate(_labeled_samples(), budget=4)
        assert res.adopted
        assert res.kappa_after > res.kappa_before
        assert res.gating_weight_after == 1.0 and res.gating_weight_before == 0.0
        assert judge.gating_weight(threshold=0.6) == 1.0

    async def test_keeps_incumbent_when_already_best(self):
        judge = GEvalJudge(
            _grounding_sensitive_judge().provider, model="mock-1", criteria="faithful?",
            steps=["Penalize claims the context does not support.", "Give a 1-5 score."],
        )
        res = await JudgeCalibrator(judge).acalibrate(_labeled_samples(), budget=4)
        assert not res.adopted
        assert res.kappa_before == pytest.approx(1.0)

    async def test_deterministic(self):
        a = await JudgeCalibrator(_grounding_sensitive_judge()).acalibrate(_labeled_samples(), budget=4)
        b = await JudgeCalibrator(_grounding_sensitive_judge()).acalibrate(_labeled_samples(), budget=4)
        assert a.adopted == b.adopted and a.kappa_after == b.kappa_after

    def test_requires_two_samples(self):
        with pytest.raises(ValueError):
            JudgeCalibrator(_grounding_sensitive_judge()).calibrate(_labeled_samples(1))

    def test_app_calibrate_judge(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = VincioConfig()
        cfg.storage.metadata = "memory://"
        cfg.observability.exporter = "memory"
        cfg.security.audit_log = False
        app = ContextApp(name="j", provider=MockProvider(), model="mock-1", config=cfg)
        res = app.calibrate_judge(_grounding_sensitive_judge(), _labeled_samples())
        assert res.adopted


# ===========================================================================
# CLI
# ===========================================================================


class TestCLI:
    def test_distill_writes_grounded_jsonl(self, tmp_path, monkeypatch):
        from vincio.cli.main import main

        traces_dir = tmp_path / "traces"
        # Produce traces with the JSONL exporter + training capture.
        cfg = VincioConfig()
        cfg.storage.metadata = "memory://"
        cfg.observability.exporter = "jsonl"
        cfg.observability.traces_dir = str(traces_dir)
        cfg.security.audit_log = False
        app = ContextApp(
            name="d",
            provider=MockProvider(responder=lambda r: "The Pro plan refund window is 30 days."),
            model="mock-1",
            config=cfg,
        )
        app.enable_training_capture()
        app.pending_evidence = list(REFUND_EVIDENCE)
        app.run("What is the Pro plan refund window?")
        out = tmp_path / "train.jsonl"
        code = main(["distill", "--traces-dir", str(traces_dir), "--output", str(out), "--min-support", "0.4"])
        assert code == 0
        assert out.is_file()
        body = out.read_text(encoding="utf-8").strip()
        assert body and '"messages"' in body

    def test_optimize_reflective_cli(self, tmp_path):
        from vincio.cli.main import main

        app_py = tmp_path / "app.py"
        app_py.write_text(
            "from vincio import ContextApp, VincioConfig\n"
            "from vincio.providers import MockProvider\n"
            "def responder(req):\n"
            "    text = '\\n'.join(m.text for m in req.messages)\n"
            "    if 'plan the steps' in text.lower() or 'briefly plan' in text.lower():\n"
            "        return 'The Pro plan refund window is 30 days.'\n"
            "    return 'I am not sure.'\n"
            "cfg = VincioConfig(); cfg.storage.metadata='memory://'\n"
            "cfg.observability.exporter='memory'; cfg.security.audit_log=False\n"
            "app = ContextApp(name='r', provider=MockProvider(responder=responder), model='mock-1', config=cfg)\n",
            encoding="utf-8",
        )
        import json as _json

        ds = tmp_path / "golden.jsonl"
        ds.write_text(
            "\n".join(
                _json.dumps({
                    "id": f"c{i}",
                    "input": "What is the Pro plan refund window?",
                    "expected": "The Pro plan refund window is 30 days.",
                })
                for i in range(6)
            ),
            encoding="utf-8",
        )
        code = main(["optimize", "reflective", "--app", str(app_py), "--dataset", str(ds), "--budget", "8"])
        assert code == 0
