"""Evaluation & observability — measure, judge, gate, and trace an AI app.

A single offline program covering the full quality lifecycle Vincio gives you:
golden datasets and metrics, three flavours of judge (deterministic / model /
G-Eval) plus ensembles, synthetic data and red-teaming, regression gates with a
baseline diff, trajectory eval over a real agent run, drift detection, the
versioned prompt registry, full trace span-trees, the indexed trace store +
local viewer, and OpenTelemetry export. Everything runs on the deterministic
mock provider — no API keys, no network.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from _shared import citing_responder, example_provider, write_sample_docs

from vincio import ContextApp
from vincio.evals import Dataset as EvalDataset
from vincio.evals import (
    DeterministicJudge,
    DriftMonitor,
    EvalCase,
    EvalRunner,
    GEvalJudge,
    JudgeEnsemble,
    ModelJudge,
    RedTeamSuite,
    RunOutput,
    SyntheticGenerator,
)
from vincio.evals.metrics import METRICS
from vincio.observability import (
    IndexedTraceStore,
    ViewerApp,
    render_trace_text,
)
from vincio.prompts import PromptRegistry, PromptSpec
from vincio.testing import assert_grounded, assert_safe

# ---------------------------------------------------------------------------
# Shared fixtures: a docs corpus + a grounded app whose mock answer cites its
# first evidence ref, so groundedness/citation metrics have something to score.
# ---------------------------------------------------------------------------
WORKDIR = Path(tempfile.mkdtemp(prefix="vincio_eval_"))
DOCS_DIR = write_sample_docs(WORKDIR / "docs")


def build_app() -> ContextApp:
    provider, model = example_provider(
        citing_responder("The refund window for the Pro plan is 30 days. [{ref}]")
    )
    app = ContextApp(name="eval_obs_demo", provider=provider, model=model)
    app.add_source("docs", path=str(DOCS_DIR))
    # answer_only_from_sources makes the groundedness/faithfulness evaluators meaningful.
    app.set_policy("answer_only_from_sources", True)
    app.add_evaluator("groundedness")
    app.add_evaluator("faithfulness")
    return app


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


# A responder for the LLM-as-judge sections. The judges issue *schema-named*
# requests (steps generation, then scoring); we return a sensible fixed payload
# per schema so judge scores are deterministic instead of schema-default zeros.
def judge_responder(request):
    name = request.output_schema_name or ""
    if name == "evaluation_steps":
        return json.dumps(
            {"steps": ["Read the output.", "Check it is grounded.", "Score 1-5."]}
        )
    if name == "g_eval_score":  # GEvalJudge fills a 1-5 form
        return json.dumps({"score": 5, "reasoning": "Grounded and complete."})
    if name == "judgment":  # ModelJudge returns a 0-1 score
        return json.dumps(
            {"score": 0.95, "reasoning": "Accurate and grounded.", "failures": []}
        )
    return "ok"


# ---------------------------------------------------------------------------
# 1. Golden JSONL dataset + the metric families on a gated run.
# ---------------------------------------------------------------------------
def golden_dataset_and_metrics(app: ContextApp) -> EvalDataset:
    banner("1. Golden JSONL dataset + task/grounding/quality metrics")

    dataset = EvalDataset(
        name="refunds",
        cases=[
            EvalCase(
                id="c1",
                input="What is the refund window for the Pro plan?",
                expected="Refunds within 30 days for the Pro plan.",
                tags=["refund"],
            ),
            EvalCase(
                id="c2",
                input="How long is the Pro refund window?",
                expected="30 days",
                tags=["refund"],
                difficulty="easy",
            ),
        ],
    )

    # Datasets serialize as one JSON object per line — the portable "golden" format.
    golden_path = WORKDIR / "golden.jsonl"
    dataset.save(golden_path)
    reloaded = EvalDataset.load(golden_path)
    print(f"  saved golden JSONL: {golden_path.name} ({len(reloaded)} cases round-tripped)")

    # metrics span the families: grounding, quality/overlap, and cost/latency (task/ops).
    runner = EvalRunner(
        app,
        metrics=["groundedness", "citation_accuracy", "lexical_overlap", "cost", "latency"],
        gates={"groundedness": ">= 0.9", "p95_latency": "<= 10000"},
        concurrency=4,
    )
    report = runner.run(reloaded, name="golden_run")
    report.print_summary()
    return dataset


# ---------------------------------------------------------------------------
# 2. Inline quality assertions — eval scores as unit-test assertions.
# ---------------------------------------------------------------------------
def quality_assertions(app: ContextApp) -> None:
    banner("2. Per-run quality + safety assertions")
    result = app.run("What is the refund window for the Pro plan?", session_id="sess_demo")
    print("  eval scores carried on the run:", result.eval_scores)
    # assert_grounded / assert_safe raise on failure, like a unit test would.
    assert_grounded(result, threshold=0.5)
    assert_safe(result)
    print("  assertions passed (grounded, safe)")


# ---------------------------------------------------------------------------
# 3. Three judges + an ensemble with disagreement detection.
# ---------------------------------------------------------------------------
async def judges_and_ensemble() -> None:
    banner("3. Deterministic / model / G-Eval judges + ensemble")
    provider, model = example_provider(default_responder=judge_responder)
    case = EvalCase(id="j", input="What is the refund window?", expected="30 days")
    out = RunOutput(output="The refund window for the Pro plan is 30 days.")

    # Deterministic judge: a pure metric, no model call. Reproducible and free.
    det = DeterministicJudge(METRICS["lexical_overlap"], name="overlap")
    # Model judge: an LLM scores against a rubric (0-1).
    model_judge = ModelJudge(provider, model=model, name="rubric")
    # G-Eval: derives explicit eval steps from criteria, then form-fills a 1-5 score.
    geval = GEvalJudge(provider, model=model, criteria="Is the answer grounded and complete?")

    for judge in (det, model_judge, geval):
        verdict = await judge.score(case, out)
        print(f"  {judge.name:8s} -> {verdict.value}")

    # An ensemble averages judges and flags disagreement as an uncertainty signal.
    panel = JudgeEnsemble([det, model_judge, geval], disagreement_threshold=0.25)
    ev = await panel.averdict(case, out)
    print(f"  ensemble value={ev.value} uncertain={ev.uncertain} spread={ev.spread}")


# ---------------------------------------------------------------------------
# 4. Synthetic dataset generation + red-teaming.
# ---------------------------------------------------------------------------
def synthetic_and_redteam(app: ContextApp) -> None:
    banner("4. Synthetic dataset generation + red-teaming")
    from vincio.documents import load_document

    documents = [load_document(p) for p in sorted(DOCS_DIR.glob("*.md"))]
    # Bootstrap a golden set straight from the corpus (seeded → reproducible).
    synth = SyntheticGenerator(seed=7).generate(documents, n=6, name="synth_refunds")
    print(
        f"  synthetic: {len(synth)} cases from {synth.metadata['sources']} docs "
        f"({synth.metadata['generator']})"
    )

    # Adversarial probes judged by the built-in security detectors — gate-able.
    report = RedTeamSuite().run(app)
    summary = report.summary()
    print(
        f"  red team: {summary['probes']} probes, "
        f"attack_success_rate={summary['attack_success_rate']}, "
        f"detector_coverage={summary['detector_coverage']}"
    )


# ---------------------------------------------------------------------------
# 5. Regression gates with a baseline diff — catch quality drops in CI.
# ---------------------------------------------------------------------------
def regression_gate_and_baseline(app: ContextApp, dataset: EvalDataset) -> None:
    banner("5. Regression gates + baseline diff")
    runner = EvalRunner(
        app,
        metrics=["groundedness", "lexical_overlap", "latency"],
        gates={"groundedness": ">= 0.9"},
    )
    baseline = runner.run(dataset, name="baseline")
    # Passing baseline= computes a per-case diff so CI can fail on *regressions*.
    current = runner.run(dataset, baseline=baseline, name="current")
    print("  gates:", {k: v["passed"] for k, v in current.gates.items()})
    print("  regressed cases vs baseline:", current.metadata["baseline_diff"]["regressed_cases"])


# ---------------------------------------------------------------------------
# 6. Trajectory & agentic eval — score *how* an agent worked, not just output.
# ---------------------------------------------------------------------------
def trajectory_eval() -> None:
    banner("6. Trajectory / agentic eval over a real ReAct run")
    # Script the agent's tool-call plan deterministically: search -> summarize -> answer.
    provider, model = example_provider(
        script=[
            {"tool_call": {"name": "search", "arguments": {"query": "refund window"}}},
            {"tool_call": {"name": "summarize", "arguments": {"text": "30 days"}}},
            "Refunds on the Pro plan are available within 30 days.",
        ]
    )
    app = ContextApp(name="agent_eval", provider=provider, model=model)

    def search(query: str) -> dict:
        """Search the knowledge base."""
        return {"hits": ["Pro plan: refunds within 30 days"]}

    def summarize(text: str) -> str:
        """Summarize text."""
        return "Refunds within 30 days."

    agent = app.agent(tools=[search, summarize], planner="react", max_steps=6)
    state = agent.run("What is the refund window for the Pro plan?")

    # Project the AgentState onto a RunOutput that carries the trajectory — no
    # re-instrumentation needed to score the run.
    run = RunOutput.from_agent_state(state)
    case = EvalCase(
        id="refund",
        input="What is the refund window for the Pro plan?",
        expected="Refunds on the Pro plan are available within 30 days.",
        rubric={
            "expected_tools": ["search", "summarize"],
            "plan": ["tool", "tool", "finalize"],
            "optimal_steps": 3,
        },
    )
    print("  trajectory metrics:")
    for name in ("goal_accuracy", "tool_call_accuracy", "tool_call_f1", "plan_adherence", "step_efficiency"):
        print(f"    {name:20s} {METRICS[name](case, run).value}")


# ---------------------------------------------------------------------------
# 7. Drift detection — alert when live scores wander off the baseline.
# ---------------------------------------------------------------------------
def drift_detection() -> None:
    banner("7. Drift detection on a score window")
    monitor = DriftMonitor(score_threshold=0.1)
    monitor.set_score_baseline("goal_accuracy", [0.9, 0.91, 0.89, 0.92])
    stable = monitor.check_scores("goal_accuracy", [0.9, 0.9, 0.91])
    regressed = monitor.check_scores("goal_accuracy", [0.6, 0.62, 0.58])
    print(f"  stable window drifted:    {stable.drifted}")
    print(f"  regressed window drifted: {regressed.drifted} (delta {regressed.delta})")


# ---------------------------------------------------------------------------
# 8. Versioned prompt registry — version prompts like code (push/diff/tag).
# ---------------------------------------------------------------------------
def prompt_registry() -> None:
    banner("8. Versioned prompt registry")
    registry = PromptRegistry(WORKDIR / "prompts")
    registry.push(
        PromptSpec(name="support", role="support agent", objective="Answer from the docs only."),
        tags=["production"],
        message="initial",
    )
    v2 = registry.push(
        PromptSpec(
            name="support",
            role="support agent",
            objective="Answer from the docs only, citing sources.",
        ),
        message="require citations",
    )
    diff = registry.diff("support", 1, 2)
    print(
        f"  {len(registry.versions('support'))} versions, "
        f"changed={list(diff['changed_fields'])}"
    )
    registry.tag("support", v2.version, "production")  # move the production pointer
    print("  production ->", registry.get("support", tag="production").ref)


# ---------------------------------------------------------------------------
# 9. Trace span-tree + indexed store + viewer + OpenTelemetry export.
# ---------------------------------------------------------------------------
def tracing_store_and_otel(app: ContextApp) -> None:
    banner("9. Trace span-tree, indexed store, viewer, OTel export")
    app.run("How long do refunds take?", session_id="sess_demo")

    # Every run captures a structured trace. Pull the latest from the exporter.
    exporter = app.tracer.exporter
    traces = exporter.traces if hasattr(exporter, "traces") else exporter.load_all()
    trace = traces[-1]
    trace.add_feedback(score=1.0, comment="clear answer")  # feedback becomes data

    # The span tree shows the full pipeline nested by parent.
    top = [s["name"] for s in trace.span_tree()]
    print(f"  span tree ({len(top)} top-level spans): {top}")
    print(render_trace_text(trace, show_attributes=False))

    # The indexed store is the queryable backend behind the local viewer; it
    # rolls up cost/latency percentiles across recorded traces. We record every
    # trace captured so far this run (eval runs, red-team probes, agent steps).
    store = IndexedTraceStore(WORKDIR / "observability.db")
    for tr in traces:
        store.record(tr)
    stats = store.stats()
    print(f"  indexed store: {store.count()} traces recorded, error_rate={stats['error_rate']}")
    # ViewerApp(store) is a WSGI app; serve_viewer(store) would bind 127.0.0.1:8043.
    ViewerApp(store)
    print("  local viewer ready — serve_viewer(store) opens it at http://127.0.0.1:8043")
    store.close()

    # OpenTelemetry export is optional (pip install 'vincio[otel]'); export to any
    # OTLP collector. We attempt it and degrade gracefully when the SDK is absent.
    try:
        from vincio.observability.otel import OTelExporter

        OTelExporter(service_name="vincio-eval-demo").export(trace)
        print("  OTel: exported trace to the configured tracer provider")
    except Exception as exc:  # missing optional dependency, in this offline env
        print(f"  OTel: skipped ({type(exc).__name__}) — install 'vincio[otel]' to enable")


# ---------------------------------------------------------------------------
# A note on the pytest plugin (no test run here, just the how-to).
# ---------------------------------------------------------------------------
def pytest_plugin_note() -> None:
    banner("10. Pytest plugin (reference)")
    # Installed via the pytest11 entry point, so `import vincio` registers it.
    # In a test module you get the `vincio_snapshot` fixture and can assert
    # eval reports directly:
    #
    #   def test_refunds(vincio_snapshot):
    #       report = EvalRunner(app, metrics=["groundedness"],
    #                           gates={"groundedness": ">= 0.9"}).run(dataset)
    #       assert all(g["passed"] for g in report.gates.values())
    #       vincio_snapshot.assert_match(report.model_dump())  # golden snapshot
    #
    # Refresh accepted snapshots with: pytest --vincio-update-snapshots
    print("  vincio_snapshot fixture + assert_* helpers wire evals into pytest")
    print("  refresh goldens with: pytest --vincio-update-snapshots")


async def main() -> None:
    app = build_app()
    dataset = golden_dataset_and_metrics(app)
    quality_assertions(app)
    await judges_and_ensemble()
    synthetic_and_redteam(app)
    regression_gate_and_baseline(app, dataset)
    trajectory_eval()
    drift_detection()
    prompt_registry()
    tracing_store_and_otel(app)
    pytest_plugin_note()
    print(f"\nartifacts written under: {WORKDIR}")


if __name__ == "__main__":
    asyncio.run(main())
