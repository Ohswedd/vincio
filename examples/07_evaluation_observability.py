"""Evaluation & observability — measure, judge, gate, and trace an AI app.

The full quality lifecycle, offline on the mock: golden datasets + metric
families, three flavours of judge (deterministic / model / G-Eval) plus an
ensemble, synthetic data + red-teaming, regression gates with a baseline diff,
trajectory eval over a real agent run, drift detection, the versioned prompt
registry, and the trace span-tree + indexed store + OpenTelemetry export.
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
from vincio.observability import IndexedTraceStore, ViewerApp, render_trace_text
from vincio.prompts import PromptRegistry, PromptSpec
from vincio.testing import assert_grounded, assert_safe

WORKDIR = Path(tempfile.mkdtemp(prefix="vincio_eval_"))
DOCS_DIR = write_sample_docs(WORKDIR / "docs")


def build_app() -> ContextApp:
    # A grounded app whose mock answer cites its first evidence ref, so the
    # groundedness/citation metrics have something concrete to score.
    provider, model = example_provider(
        citing_responder("The refund window for the Pro plan is 30 days. [{ref}]"))
    app = ContextApp(name="eval_obs_demo", provider=provider, model=model)
    app.add_source("docs", path=str(DOCS_DIR))
    app.set_policy("answer_only_from_sources", True)  # makes groundedness meaningful
    app.add_evaluator("groundedness")
    app.add_evaluator("faithfulness")
    return app


def judge_responder(request):
    # The LLM-as-judge sections issue schema-named requests; return a fixed payload
    # per schema so judge scores are deterministic instead of schema-default zeros.
    name = request.output_schema_name or ""
    if name == "evaluation_steps":
        return json.dumps({"steps": ["Read the output.", "Check it is grounded.", "Score 1-5."]})
    if name == "g_eval_score":  # GEvalJudge fills a 1-5 form
        return json.dumps({"score": 5, "reasoning": "Grounded and complete."})
    if name == "judgment":  # ModelJudge returns a 0-1 score
        return json.dumps({"score": 0.95, "reasoning": "Accurate and grounded.", "failures": []})
    return "ok"


def golden_dataset_and_metrics(app: ContextApp) -> EvalDataset:
    # A golden dataset is the portable contract for quality. It serializes as one
    # JSON object per line (JSONL) so it round-trips and lives in version control.
    dataset = EvalDataset(name="refunds", cases=[
        EvalCase(id="c1", input="What is the refund window for the Pro plan?",
                 expected="Refunds within 30 days for the Pro plan.", tags=["refund"]),
        EvalCase(id="c2", input="How long is the Pro refund window?",
                 expected="30 days", tags=["refund"], difficulty="easy"),
    ])
    golden = WORKDIR / "golden.jsonl"
    dataset.save(golden)
    EvalDataset.load(golden)  # round-trips losslessly

    # metrics span families (grounding, overlap, cost/latency); gates are the CI
    # contract — a run FAILS when a gated metric misses its threshold.
    report = EvalRunner(app, metrics=["groundedness", "citation_accuracy", "lexical_overlap", "cost", "latency"],
                        gates={"groundedness": ">= 0.9", "p95_latency": "<= 10000"}, concurrency=4
                        ).run(dataset, name="golden_run")
    print("1. golden eval: gates", {k: v["passed"] for k, v in report.gates.items()})
    return dataset


def quality_assertions(app: ContextApp) -> None:
    # Eval scores can be asserted like unit tests: assert_grounded / assert_safe
    # raise on failure, so a single answer can be gated inline in a test.
    result = app.run("What is the refund window for the Pro plan?", session_id="sess_demo")
    assert_grounded(result, threshold=0.5)
    assert_safe(result)
    print("2. inline assertions passed; scores", result.eval_scores)


async def judges_and_ensemble() -> None:
    # Three judge flavours over one output. Deterministic = a pure metric (free,
    # reproducible); Model = an LLM scores against a rubric (0-1); G-Eval derives
    # explicit eval steps from criteria then form-fills a 1-5 score. An ensemble
    # averages them and flags disagreement as an uncertainty signal to escalate on.
    provider, model = example_provider(default_responder=judge_responder)
    case = EvalCase(id="j", input="What is the refund window?", expected="30 days")
    out = RunOutput(output="The refund window for the Pro plan is 30 days.")
    det = DeterministicJudge(METRICS["lexical_overlap"], name="overlap")
    model_judge = ModelJudge(provider, model=model, name="rubric")
    geval = GEvalJudge(provider, model=model, criteria="Is the answer grounded and complete?")
    scores = {j.name: (await j.score(case, out)).value for j in (det, model_judge, geval)}
    ev = await JudgeEnsemble([det, model_judge, geval], disagreement_threshold=0.25).averdict(case, out)
    print(f"3. judges {scores} | ensemble={ev.value} uncertain={ev.uncertain}")


def synthetic_and_redteam(app: ContextApp) -> None:
    # Bootstrap a golden set straight from the corpus (seeded -> reproducible), and
    # probe the app with adversarial red-team attacks judged by the security
    # detectors — both feed the same gate-able report a human dataset would.
    from vincio.documents import load_document
    documents = [load_document(p) for p in sorted(DOCS_DIR.glob("*.md"))]
    synth = SyntheticGenerator(seed=7).generate(documents, n=6, name="synth_refunds")
    summary = RedTeamSuite().run(app).summary()
    print(f"4. synthetic {len(synth)} cases | red team {summary['probes']} probes, "
          f"attack_success_rate={summary['attack_success_rate']}")


def regression_gate_and_baseline(app: ContextApp, dataset: EvalDataset) -> None:
    # Passing baseline= computes a per-case diff so CI fails on REGRESSIONS, not on
    # an absolute bar — the right shape for catching quality drops between commits.
    runner = EvalRunner(app, metrics=["groundedness", "lexical_overlap", "latency"],
                        gates={"groundedness": ">= 0.9"})
    baseline = runner.run(dataset, name="baseline")
    current = runner.run(dataset, baseline=baseline, name="current")
    print("5. regression: gates", {k: v["passed"] for k, v in current.gates.items()},
          "| regressed", current.metadata["baseline_diff"]["regressed_cases"])


def trajectory_eval() -> None:
    # Trajectory eval scores HOW an agent worked, not just its final answer. Project
    # the AgentState onto a RunOutput that carries the trajectory (no
    # re-instrumentation), then score tool choice / plan adherence / step efficiency
    # against a rubric — essential for agents where the path is the product.
    provider, model = example_provider(script=[
        {"tool_call": {"name": "search", "arguments": {"query": "refund window"}}},
        {"tool_call": {"name": "summarize", "arguments": {"text": "30 days"}}},
        "Refunds on the Pro plan are available within 30 days.",
    ])
    app = ContextApp(name="agent_eval", provider=provider, model=model)
    app.add_tool(lambda query: {"hits": ["Pro plan: refunds within 30 days"]}, name="search")
    app.add_tool(lambda text: "Refunds within 30 days.", name="summarize")
    state = app.agent(tools=["search", "summarize"], planner="react", max_steps=6).run(
        "What is the refund window for the Pro plan?")
    run = RunOutput.from_agent_state(state)
    case = EvalCase(id="refund", input="What is the refund window for the Pro plan?",
                    expected="Refunds on the Pro plan are available within 30 days.",
                    rubric={"expected_tools": ["search", "summarize"],
                            "plan": ["tool", "tool", "finalize"], "optimal_steps": 3})
    metrics = {n: METRICS[n](case, run).value for n in
               ("tool_call_f1", "plan_adherence", "step_efficiency")}
    print("6. trajectory metrics:", metrics)


def drift_detection() -> None:
    # Alert when live scores wander off a baseline window — the production analogue
    # of a regression gate, run continuously against a rolling score sample.
    monitor = DriftMonitor(score_threshold=0.1)
    monitor.set_score_baseline("goal_accuracy", [0.9, 0.91, 0.89, 0.92])
    stable = monitor.check_scores("goal_accuracy", [0.9, 0.9, 0.91])
    regressed = monitor.check_scores("goal_accuracy", [0.6, 0.62, 0.58])
    print(f"7. drift: stable={stable.drifted} regressed={regressed.drifted} (delta {regressed.delta})")


def prompt_registry() -> None:
    # Version prompts like code: push new versions, diff them, and MOVE a
    # `production` tag to promote — so a rollback is just re-tagging the last good one.
    registry = PromptRegistry(WORKDIR / "prompts")
    registry.push(PromptSpec(name="support", role="support agent",
                             objective="Answer from the docs only."), tags=["production"], message="initial")
    v2 = registry.push(PromptSpec(name="support", role="support agent",
                                  objective="Answer from the docs only, citing sources."),
                       message="require citations")
    diff = registry.diff("support", 1, 2)
    registry.tag("support", v2.version, "production")
    print(f"8. registry: {len(registry.versions('support'))} versions, changed {list(diff['changed_fields'])}, "
          f"production -> {registry.get('support', tag='production').ref}")


def tracing_store_and_otel(app: ContextApp) -> None:
    # Every run captures a structured trace: a span tree nested by parent, with
    # feedback attachable as data. The IndexedTraceStore is the queryable backend
    # behind the local viewer, rolling up cost/latency/error-rate across traces.
    app.run("How long do refunds take?", session_id="sess_demo")
    exporter = app.tracer.exporter
    traces = exporter.traces if hasattr(exporter, "traces") else exporter.load_all()
    trace = traces[-1]
    trace.add_feedback(score=1.0, comment="clear answer")
    print("9. trace span tree:", [s["name"] for s in trace.span_tree()])
    print(render_trace_text(trace, show_attributes=False))

    store = IndexedTraceStore(WORKDIR / "observability.db")
    for tr in traces:
        store.record(tr)
    print(f"   indexed store: {store.count()} traces, error_rate={store.stats()['error_rate']} "
          f"| serve_viewer(store) opens http://127.0.0.1:8043")
    ViewerApp(store)  # a WSGI app; serve_viewer(store) would bind it
    store.close()

    # OpenTelemetry export is optional (pip install 'vincio[otel]'); degrade
    # gracefully when the SDK is absent, as it is in this offline env.
    try:
        from vincio.observability.otel import OTelExporter
        OTelExporter(service_name="vincio-eval-demo").export(trace)
        print("   OTel: exported to the configured tracer provider")
    except Exception as exc:  # noqa: BLE001 - optional dependency may be missing
        print(f"   OTel: skipped ({type(exc).__name__}) — install 'vincio[otel]' to enable")

    # The pytest plugin (registered via the pytest11 entry point) adds a
    # `vincio_snapshot` fixture + assert_* helpers so eval reports become golden
    # snapshots in CI; refresh them with `pytest --vincio-update-snapshots`.


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
    print(f"\nartifacts under: {WORKDIR}")


if __name__ == "__main__":
    asyncio.run(main())
