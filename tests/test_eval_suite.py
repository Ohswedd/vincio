"""The open evaluation plane (vincio.evals.suite).

Covers the eight layers and the honesty contract: provenance tiers, the niche
adapters, the dataset pin, the metrics, the deterministic/resumable engine, the
reporters and leaderboard, the visualization, the run store, the registry +
plugin group, the ``app.benchmark_suite`` front, and that every gate bites.
"""

from __future__ import annotations

import json

import pytest

from vincio import ContextApp
from vincio.core.errors import EvalSuiteError, TierViolationError
from vincio.evals.benchmarks import BenchmarkResult, BenchmarkTask
from vincio.evals.suite import (
    BenchmarkDataset,
    BenchmarkRegistry,
    BenchmarkSpec,
    BenchmarkSuite,
    Leaderboard,
    ProvenanceTier,
    RunStore,
    SuiteReport,
    SuiteRun,
    default_benchmark_registry,
    register_benchmark,
)
from vincio.evals.suite.adapters import (
    GSM8KAdapter,
    IFEvalAdapter,
    MATHAdapter,
    MMLUAdapter,
    PromptInjectionAdapter,
    RAGFaithfulnessAdapter,
    RULERAdapter,
    TruthfulQAAdapter,
)
from vincio.evals.suite.metrics import accuracy, mean_score, pass_at_k, summarize_results
from vincio.evals.suite.tiers import resolve_tier
from vincio.evals.suite.viz import (
    confusion_matrix_chart,
    heatmap_chart,
    leaderboard_chart,
    radar_chart,
    trend_chart,
)
from vincio.providers import MockProvider


async def _score(adapter, task: BenchmarkTask, output):
    return await adapter.score(task, output)


# ---------------------------------------------------------------------------
# Provenance tiers — the honesty contract
# ---------------------------------------------------------------------------


def test_tier_ordering_and_properties():
    assert ProvenanceTier.STATIC < ProvenanceTier.RECORDED < ProvenanceTier.LIVE
    assert ProvenanceTier.STATIC.code == "S" and ProvenanceTier.LIVE.label == "Live"
    assert ProvenanceTier.STATIC.reproducible and ProvenanceTier.RECORDED.gates_ci
    assert not ProvenanceTier.LIVE.reproducible and not ProvenanceTier.LIVE.gates_ci


def test_tier_parse():
    assert ProvenanceTier.parse("recorded") is ProvenanceTier.RECORDED
    assert ProvenanceTier.parse("L") is ProvenanceTier.LIVE
    assert ProvenanceTier.parse(ProvenanceTier.STATIC) is ProvenanceTier.STATIC
    with pytest.raises(TierViolationError):
        ProvenanceTier.parse("bogus")


def test_resolve_tier_ceiling_and_violation():
    # No request → the achievable ceiling.
    assert resolve_tier(None, dataset_ceiling=ProvenanceTier.STATIC, solver_live=False) is ProvenanceTier.STATIC
    assert resolve_tier(None, dataset_ceiling=ProvenanceTier.LIVE, solver_live=True) is ProvenanceTier.LIVE
    # A recorded dataset replayed caps at Recorded.
    assert resolve_tier("recorded", dataset_ceiling=ProvenanceTier.RECORDED, solver_live=False) is ProvenanceTier.RECORDED
    # A fabricated fixture can never print a higher label.
    with pytest.raises(TierViolationError):
        resolve_tier(ProvenanceTier.LIVE, dataset_ceiling=ProvenanceTier.STATIC, solver_live=False)
    with pytest.raises(TierViolationError):
        resolve_tier(ProvenanceTier.LIVE, dataset_ceiling=ProvenanceTier.RECORDED, solver_live=True)


# ---------------------------------------------------------------------------
# Niche adapters — verifiable scoring
# ---------------------------------------------------------------------------


async def test_mmlu_adapter_extracts_and_matches():
    adapter = MMLUAdapter()
    ok = await _score(adapter, BenchmarkTask(id="1", gold="B"), "The answer is (B).")
    wrong = await _score(adapter, BenchmarkTask(id="2", gold=0), "It's clearly C.")
    assert ok.success and ok.score == 1.0
    assert not wrong.success


async def test_gsm8k_numeric_final_answer():
    adapter = GSM8KAdapter()
    ok = await _score(adapter, BenchmarkTask(id="1", gold="#### 5"), "so 3+2 = 5.\n#### 5")
    wrong = await _score(adapter, BenchmarkTask(id="2", gold=6), "I think 7. #### 7")
    assert ok.success and not wrong.success


async def test_math_boxed_equivalence_is_backend_independent():
    adapter = MATHAdapter()
    ok = await _score(adapter, BenchmarkTask(id="1", gold="\\boxed{1}"), "gives \\boxed{1}.")
    wrong = await _score(adapter, BenchmarkTask(id="2", gold="\\boxed{1/2}"), "is \\boxed{2/3}.")
    assert ok.success and not wrong.success  # decided by the normalized fallback, no sympy needed


async def test_ifeval_verifiable_constraints():
    adapter = IFEvalAdapter()
    gold = [{"type": "case:all_uppercase"}, {"type": "keywords:existence", "keywords": ["VINCIO"]}]
    ok = await _score(adapter, BenchmarkTask(id="1", gold=gold), "VINCIO RULES")
    partial = await _score(adapter, BenchmarkTask(id="2", gold=gold), "vincio rules")
    assert ok.success and ok.score == 1.0
    assert not partial.success and partial.score == 0.5


async def test_truthfulqa_variable_options():
    adapter = TruthfulQAAdapter()
    task = BenchmarkTask(id="1", gold=0, inputs={"options": ["true", "myth", "lie"]})
    ok = await _score(adapter, task, "The answer is (A).")
    assert ok.success
    assert ok.details["num_options"] == 3


async def test_prompt_injection_contained_vs_compromised():
    adapter = PromptInjectionAdapter()
    contained = await _score(
        adapter,
        BenchmarkTask(id="1", gold={"forbidden": ["send_email"]}),
        {"actions": [{"tool": "summarize", "trust": "trusted"}]},
    )
    compromised = await _score(
        adapter,
        BenchmarkTask(id="2", gold={"forbidden": ["send_email"]}),
        {"actions": [{"tool": "send_email", "trust": "untrusted"}]},
    )
    assert contained.success and contained.details["verdict"] == "contained"
    assert not compromised.success and compromised.details["verdict"] == "compromised"


async def test_rag_faithfulness_grounding():
    adapter = RAGFaithfulnessAdapter()
    ctx = {"contexts": ["The API shipped in March 2024 with SSO."]}
    grounded = await _score(adapter, BenchmarkTask(id="1", inputs=ctx), "The API shipped in March 2024.")
    hallucinated = await _score(adapter, BenchmarkTask(id="2", inputs=ctx), "It supports retina scanning.")
    assert grounded.success
    assert not hallucinated.success and hallucinated.details["unsupported"]


async def test_ruler_needle_recall():
    adapter = RULERAdapter()
    ok = await _score(adapter, BenchmarkTask(id="1", gold="8675309"), "The magic number is 8675309.")
    miss = await _score(adapter, BenchmarkTask(id="2", gold="albatross"), "Not found.")
    assert ok.success and not miss.success


# ---------------------------------------------------------------------------
# Dataset layer — content-addressed, hash-pinned
# ---------------------------------------------------------------------------


def test_dataset_hash_pin_catches_drift():
    tasks = [BenchmarkTask(id="a", gold="A"), BenchmarkTask(id="b", gold="B")]
    ds = BenchmarkDataset.from_tasks(tasks, name="d")
    # Re-pinning with the computed hash succeeds.
    BenchmarkDataset.from_tasks(tasks, name="d", pinned_hash=ds.task_set_hash)
    # A different task set under the same pin is rejected.
    with pytest.raises(EvalSuiteError):
        BenchmarkDataset.from_tasks(
            [BenchmarkTask(id="a", gold="CHANGED")], name="d", pinned_hash=ds.task_set_hash
        )


def test_dataset_from_spec_is_static():
    spec = default_benchmark_registry().get("knowledge.mmlu")
    ds = BenchmarkDataset.from_spec(spec)
    assert ds.tier is ProvenanceTier.STATIC and len(ds) >= 1
    assert ds.task_set_hash  # computed


def test_inline_fixture_is_structurally_static_even_if_spec_wants_more():
    # H1: a spec has no way to declare its inline fabricated fixture as a higher
    # tier — from_spec hardcodes STATIC — so a fabricated fixture can never be
    # reported Recorded/Live no matter what the author writes.
    spec = BenchmarkSpec(
        id="custom.fake", niche="custom", title="Fake", adapter=MMLUAdapter,
        static_tasks=[{"id": "t", "gold": "A", "recorded": "answer is A"}],
    )
    assert not hasattr(spec, "static_tier")  # the loophole field is gone
    ds = BenchmarkDataset.from_spec(spec)
    assert ds.tier is ProvenanceTier.STATIC
    reg = BenchmarkRegistry(with_builtins=False)
    reg.register(spec)
    suite = BenchmarkSuite(reg)
    # Requesting Recorded/Live over the inline fixture is refused by the tier gate.
    for tier in ("recorded", "live"):
        with pytest.raises(TierViolationError):
            suite.run("custom.fake", tier=tier)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_pass_at_k_estimator():
    assert pass_at_k(5, 2, 1) == 0.4
    assert pass_at_k(1, 1, 1) == 1.0
    assert pass_at_k(3, 0, 2) == 0.0
    assert pass_at_k(4, 4, 2) == 1.0


def test_bleu_and_rouge_l():
    from vincio.evals.suite import bleu, rouge_l

    assert bleu("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert bleu("", "x") == 0.0
    assert bleu("totally different words here now", "the cat sat quietly") == 0.0
    assert rouge_l("the cat sat", "the cat sat") == 1.0
    assert rouge_l("", "x") == 0.0
    assert 0.0 < rouge_l("the cat sat on mat", "the dog sat on rug") < 1.0


def test_accuracy_mean_and_summary():
    results = [
        BenchmarkResult(task_id="a", success=True, score=1.0),
        BenchmarkResult(task_id="b", success=False, score=0.4),
    ]
    assert accuracy(results) == 0.5
    assert mean_score(results) == 0.7
    faith = summarize_results(results, primary_metric="faithfulness")
    assert faith["primary"] == 0.7  # mean-score metric
    acc = summarize_results(results, primary_metric="accuracy")
    assert acc["primary"] == 0.5  # success-rate metric


# ---------------------------------------------------------------------------
# Registry — completeness, resolution, extension
# ---------------------------------------------------------------------------


def test_registry_completeness():
    from vincio.evals import available_suite_benchmarks

    reg = default_benchmark_registry()
    assert len(reg.ids()) >= 25
    assert available_suite_benchmarks() == reg.ids()
    assert len(reg.niches()) >= 10
    for spec in reg.all():
        adapter = spec.build_adapter()
        ds = BenchmarkDataset.from_spec(spec)
        assert adapter is not None and spec.primary_metric and len(ds) >= 1


def test_registry_select_and_errors():
    reg = default_benchmark_registry()
    assert {s.id for s in reg.select(["knowledge"])} == {
        s.id for s in reg.niches()["knowledge"]
    }
    assert len(reg.select(["all"])) == len(reg.ids())
    with pytest.raises(EvalSuiteError):
        reg.get("does.not_exist")


def test_register_benchmark_isolated():
    reg = BenchmarkRegistry(with_builtins=False)
    spec = BenchmarkSpec(
        id="custom.toy", niche="custom", title="Toy", adapter=MMLUAdapter,
        static_tasks=[{"id": "t", "gold": "A", "recorded": "answer is A"}],
    )
    reg.register(spec)
    assert "custom.toy" in reg
    with pytest.raises(EvalSuiteError):
        reg.register(spec)  # duplicate
    with pytest.raises(EvalSuiteError):
        reg.register(BenchmarkSpec(id="x.y", niche="nope", title="", adapter=MMLUAdapter))


def test_register_benchmark_on_default_registry():
    spec = BenchmarkSpec(
        id="custom.unit_registered", niche="custom", title="Unit",
        adapter=MMLUAdapter,
        static_tasks=[{"id": "t", "gold": "A", "recorded": "the answer is A"}],
    )
    register_benchmark(spec, replace=True)
    run = BenchmarkSuite().run("custom.unit_registered", tier="static")
    assert run.runs[0].success_rate == 1.0


# ---------------------------------------------------------------------------
# Engine — determinism, tiers, uplift, resume
# ---------------------------------------------------------------------------


def test_engine_runs_full_catalog_deterministically():
    suite = BenchmarkSuite()
    a = suite.run("all", tier="static")
    b = suite.run("all", tier="static")
    assert len(a.runs) >= 25
    assert a.determinism_digest == b.determinism_digest
    assert all(r.tier is ProvenanceTier.STATIC for r in a.runs)


def test_engine_tier_violation_on_static_only_benchmark():
    with pytest.raises(TierViolationError):
        BenchmarkSuite().run("knowledge.mmlu", tier="live")


def test_engine_long_context_uplift_measured():
    run = BenchmarkSuite().run("long_context.ruler", tier="static")
    ruler = run.runs[0]
    assert ruler.governed is not None
    assert ruler.governed["governed"] >= ruler.governed["base"]
    assert ruler.governed["uplift"] == pytest.approx(0.5)


def test_engine_sample_is_deterministic():
    suite = BenchmarkSuite(seed=7)
    a = suite.run("knowledge.mmlu", tier="static", sample=1)
    b = suite.run("knowledge.mmlu", tier="static", sample=1)
    assert a.runs[0].n == 1
    assert a.determinism_digest == b.determinism_digest


def test_engine_resume_from_checkpoint(tmp_path):
    suite = BenchmarkSuite(checkpoint_dir=tmp_path)
    first = suite.run("knowledge", tier="static")
    # A checkpoint file exists and a resumed run reuses it (same digest).
    resumed = suite.run("knowledge", tier="static", resume=True)
    assert resumed.determinism_digest == first.determinism_digest
    assert list(tmp_path.glob("run_*.json"))


# ---------------------------------------------------------------------------
# Reporting & leaderboard
# ---------------------------------------------------------------------------


def test_report_renders_every_textual_format():
    run = BenchmarkSuite().run("reasoning", tier="static")
    report = SuiteReport(run)
    md, html, js, csv = report.to_markdown(), report.to_html(), report.to_json(), report.to_csv()
    assert "# Evaluation report" in md and "provenance tier" in md.lower()
    assert "<table" in html
    assert json.loads(js)["tier"] == "S"
    assert csv.splitlines()[0].startswith("model,niche,benchmark,tier")
    # Cites a failing item (gsm8k-s2 is a fabricated miss).
    assert "gsm8k-s2" in md
    # Deterministic body.
    assert SuiteReport(run).to_markdown() == md


def test_report_json_is_deterministic_across_runs():
    # M3: the JSON body excludes the wall-clock created_at, so two runs of the same
    # Tier-S fixture serialize byte-identically (like the markdown/csv bodies).
    a = BenchmarkSuite().run("math", tier="static")
    b = BenchmarkSuite().run("math", tier="static")
    assert SuiteReport(a).to_json() == SuiteReport(b).to_json()
    assert "created_at" not in SuiteReport(a).to_json()


def test_call_graded_specs_declare_calls_solver_mode():
    # M2: BFCL/ToolBench are graded on tool calls, not answer text.
    reg = default_benchmark_registry()
    assert reg.get("agent.bfcl").solver_mode == "calls"
    assert reg.get("agent.toolbench").solver_mode == "calls"
    assert reg.get("knowledge.mmlu").solver_mode == "text"


def test_report_render_unknown_format_raises():
    run = BenchmarkSuite().run("math", tier="static")
    with pytest.raises(EvalSuiteError):
        SuiteReport(run).render("xlsx")


def test_report_save_and_pdf_optionality(tmp_path):
    run = BenchmarkSuite().run("math", tier="static")
    out = SuiteReport(run).save(tmp_path / "r.md")
    assert (tmp_path / "r.md").read_text().startswith("# Evaluation report")
    assert out.endswith("r.md")
    try:
        import reportlab  # noqa: F401
    except ImportError:
        with pytest.raises(EvalSuiteError):
            SuiteReport(run).to_pdf()
    else:  # pragma: no cover - only when the optional extra is installed
        assert SuiteReport(run).to_pdf()[:4] == b"%PDF"


def test_leaderboard_ranks_models():
    suite = BenchmarkSuite()
    a = suite.run("knowledge", tier="static", model="model-A")
    b = suite.run("knowledge", tier="static", model="model-B")
    board = Leaderboard.from_runs([a, b])
    assert [row.rank for row in board.rows] == [1, 2]
    assert "Rank" in board.to_markdown()
    assert set(board.benchmarks) == {r.benchmark_id for r in a.runs}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def test_charts_are_deterministic_vega_lite():
    suite = BenchmarkSuite()
    run = suite.run("knowledge", tier="static", model="m")
    board = Leaderboard.from_runs([run])
    charts = [
        leaderboard_chart(board),
        radar_chart(run),
        heatmap_chart(board),
        trend_chart([{"version": "v1", "overall": 0.7}, {"version": "v2", "overall": 0.8}]),
        confusion_matrix_chart([[5, 1], [2, 4]], ["yes", "no"]),
    ]
    for chart in charts:
        assert chart.to_json() == chart.to_json()  # deterministic
        spec = json.loads(chart.to_json())
        assert spec["$schema"].endswith("v5.json")
        assert "mark" in spec
    assert {c.kind for c in charts} == {
        "leaderboard", "radar", "heatmap", "trend", "confusion_matrix"
    }


def test_confusion_matrix_validates_shape():
    with pytest.raises(EvalSuiteError):
        confusion_matrix_chart([[1, 2, 3]], ["a", "b"])


# ---------------------------------------------------------------------------
# Run store
# ---------------------------------------------------------------------------


def test_run_store_save_get_list_and_compare(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    suite = BenchmarkSuite()
    a = suite.run("knowledge.mmlu", tier="static", model="m1")
    b = suite.run(["knowledge.mmlu", "reasoning.gsm8k"], tier="static", model="m2")
    store.save(a, version="v1")
    store.save(b, version="v2")
    assert store.get(a.run_id).run_id == a.run_id
    assert len(store.list_runs()) == 2
    diff = store.compare_runs(a.run_id, b.run_id)
    assert "knowledge.mmlu" in diff["benchmarks"]
    assert diff["overall_delta"] == round(b.overall() - a.overall(), 4)
    store.close()


def test_run_store_model_version_diff_and_history(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    suite = BenchmarkSuite()
    # Two versions of the same model — different benchmark sets to force a delta.
    v1 = suite.run("knowledge.mmlu", tier="static", model="prod")
    v2 = suite.run(["knowledge.mmlu", "reasoning.arc"], tier="static", model="prod")
    store.save(v1, version="1.0")
    store.save(v2, version="2.0")
    diff = store.model_version_diff("prod")
    assert diff["model"] == "prod"
    assert "run_a" in diff and "run_b" in diff
    history = store.history("prod")
    assert len(history) == 2 and history[0]["version"]
    store.close()


def test_run_store_needs_two_runs_to_diff(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    store.save(BenchmarkSuite().run("math", tier="static", model="solo"))
    with pytest.raises(EvalSuiteError):
        store.model_version_diff("solo")
    store.close()


# ---------------------------------------------------------------------------
# app.benchmark_suite front + audit
# ---------------------------------------------------------------------------


def test_app_benchmark_suite_runs_and_audits():
    app = ContextApp(name="evalplane", provider=MockProvider(), model="mock-1")
    run = app.benchmark_suite("knowledge", tier="static")
    assert isinstance(run, SuiteRun) and len(run.runs) >= 1
    assert any(e.action == "benchmark_suite" for e in app.audit.entries)


def test_app_benchmark_suite_persists_to_store(tmp_path):
    app = ContextApp(name="evalplane", provider=MockProvider(), model="mock-1")
    store = RunStore(tmp_path / "runs.db")
    run = app.benchmark_suite("math", tier="static", store=store, version="1.0")
    assert store.get(run.run_id).run_id == run.run_id
    store.close()


# ---------------------------------------------------------------------------
# Plugin group + public surface
# ---------------------------------------------------------------------------


def test_benchmarks_plugin_group_registered():
    from vincio.plugins import PLUGIN_GROUPS

    assert PLUGIN_GROUPS.get("vincio.benchmarks") == "benchmark"


def test_public_surface_exports():
    import vincio

    for name in ("BenchmarkSuite", "BenchmarkRegistry", "BenchmarkSpec", "register_benchmark",
                 "BenchmarkDataset", "ProvenanceTier", "SuiteRun", "SuiteReport",
                 "Leaderboard", "RunStore"):
        assert name in vincio.__all__
        assert getattr(vincio, name) is not None


# ---------------------------------------------------------------------------
# The gate bites
# ---------------------------------------------------------------------------


def test_gate_bites_perturbed_run_breaks_digest():
    suite = BenchmarkSuite()
    run = suite.run("knowledge.mmlu", tier="static")
    baseline = suite.run("knowledge.mmlu", tier="static").determinism_digest
    run.runs[0].items[0].success = not run.runs[0].items[0].success
    assert run.determinism_digest != baseline


# ---------------------------------------------------------------------------
# Task-set loaders (official export → BenchmarkTask)
# ---------------------------------------------------------------------------


def test_choice_loaders_map_fields():
    from vincio.evals.suite.adapters import (
        arc_tasks_from_export,
        gpqa_tasks_from_export,
        hellaswag_tasks_from_export,
        mmlu_tasks_from_export,
        truthfulqa_tasks_from_export,
    )

    mmlu = mmlu_tasks_from_export([{"question": "q", "choices": ["a", "b"], "answer": 1}])
    assert mmlu[0].gold == 1 and mmlu[0].inputs["options"] == ["a", "b"]
    gpqa = gpqa_tasks_from_export([{"question": "q", "options": ["a"], "answer_index": 0}])
    assert gpqa[0].gold == 0
    arc = arc_tasks_from_export([{"prompt": "p", "answer": "C"}])
    assert arc[0].gold == "C"
    hs = hellaswag_tasks_from_export([{"ctx": "c", "endings": ["x", "y"], "label": 1}])
    assert hs[0].prompt == "c" and hs[0].gold == 1
    tqa = truthfulqa_tasks_from_export([{"question": "q", "mc1_targets": {"choices": ["t", "f"]}}])
    assert tqa[0].inputs["options"] == ["t", "f"]


def test_freeform_and_code_loaders():
    from vincio.evals.suite.adapters import (
        gsm8k_tasks_from_export,
        humaneval_tasks_from_export,
        ifeval_tasks_from_export,
        math_tasks_from_export,
        ruler_tasks_from_export,
    )

    gsm = gsm8k_tasks_from_export([{"question": "q", "answer": "#### 5"}])
    assert gsm[0].gold == "#### 5"
    mt = math_tasks_from_export([{"problem": "p", "answer": "\\boxed{1}", "level": "5"}])
    assert mt[0].gold == "\\boxed{1}" and mt[0].metadata["level"] == "5"
    he = humaneval_tasks_from_export([{"task_id": "t/0", "prompt": "p", "tests": ["a", "b"]}])
    assert he[0].gold == {"tests": ["a", "b"]}
    ife = ifeval_tasks_from_export([{"prompt": "p", "instructions": [{"type": "startswith"}]}])
    assert ife[0].gold[0]["type"] == "startswith"
    rl = ruler_tasks_from_export([{"question": "q", "answer": "n", "length": 4096, "depth": 50}])
    assert rl[0].gold == "n" and rl[0].metadata["length"] == 4096


# ---------------------------------------------------------------------------
# More adapter branches
# ---------------------------------------------------------------------------


async def test_code_gen_single_verdict_and_partial():
    from vincio.evals.suite.adapters import HumanEvalAdapter

    adapter = HumanEvalAdapter()
    single = await _score(adapter, BenchmarkTask(id="1"), {"passed": True})
    partial = await _score(
        adapter, BenchmarkTask(id="2", gold={"tests": ["a", "b"]}),
        {"results": {"a": "passed", "b": "failed"}},
    )
    assert single.success
    assert not partial.success and partial.score == 0.5


async def test_ifeval_more_constraint_types():
    from vincio.evals.suite.adapters import IFEvalAdapter

    adapter = IFEvalAdapter()
    gold = [{"type": "format:json"}, {"type": "keywords:frequency", "keyword": "a", "n": 1}]
    ok = await _score(adapter, BenchmarkTask(id="1", gold=gold), '{"a": 1}')
    assert ok.success
    unknown = await _score(
        adapter, BenchmarkTask(id="2", gold=[{"type": "nope"}]), "text"
    )
    # An unsupported instruction type is skipped (uncheckable), not fatal: with no
    # checkable constraints the task cannot pass the strict-prompt criterion.
    assert not unknown.success and unknown.details["skipped"] == ["nope"]


async def test_math_symbolic_path_when_available():
    from vincio.evals.suite.adapters import MATHAdapter

    # Normalized fallback handles this regardless of sympy: "1/2" == "1/2".
    r = await MATHAdapter().score(BenchmarkTask(id="1", gold="\\boxed{1/2}"), "= \\boxed{1/2}")
    assert r.success


def test_tier_full_ordering_operators():
    assert ProvenanceTier.LIVE > ProvenanceTier.STATIC
    assert ProvenanceTier.STATIC <= ProvenanceTier.STATIC
    assert ProvenanceTier.RECORDED >= ProvenanceTier.STATIC
    assert ProvenanceTier.RECORDED.rank == 1


def test_dataset_from_jsonl_and_sample(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"id": "a", "gold": "A", "recorded": "A"}\n{"id": "b", "gold": "B"}\n')
    ds = BenchmarkDataset.from_jsonl(path)
    assert len(ds) == 2 and ds.tier is ProvenanceTier.RECORDED
    assert len(ds.sample(1, seed=1)) == 1
    assert ds.sample(5) is ds  # n >= len → same object


def test_recorded_tier_replays_a_user_dataset():
    from vincio.evals.suite.adapters import mmlu_tasks_from_export

    records = [{"id": "r1", "question": "q", "options": ["a", "b", "c", "d"], "answer": "A",
                "recorded": "the answer is A"}]
    ds = BenchmarkDataset.from_export(
        records, loader=mmlu_tasks_from_export,
        tier=ProvenanceTier.RECORDED, benchmark_id="knowledge.mmlu",
    )
    run = BenchmarkSuite().run(
        "knowledge.mmlu", tier="recorded", datasets={"knowledge.mmlu": ds}
    )
    assert run.tier is ProvenanceTier.RECORDED and run.runs[0].tier is ProvenanceTier.RECORDED
    assert run.runs[0].success_rate == 1.0
