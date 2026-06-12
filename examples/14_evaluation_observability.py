"""Evaluation & observability (0.5): quality metrics, synthetic data,
red-teaming, experiments with significance, prompt registry, sessions,
feedback, and the local trace viewer — all offline, no platform."""

import tempfile
from pathlib import Path

from _shared import citing_responder, example_provider, write_sample_docs

from vincio import ContextApp
from vincio.evals import (
    EvalRunner,
    ExperimentTracker,
    RedTeamSuite,
    SyntheticGenerator,
    ab_test,
    dataset_from_traces,
)
from vincio.observability import (
    render_trace_text,
    sessions_from_traces,
    trace_to_html,
)
from vincio.prompts import PromptRegistry, PromptSpec
from vincio.testing import assert_grounded, assert_safe

workdir = Path(tempfile.mkdtemp())
docs_dir = write_sample_docs(workdir / "docs")
provider, model = example_provider(
    citing_responder("The refund window for the Pro plan is 30 days. [{ref}]")
)

app = ContextApp(name="eval_obs_demo", provider=provider, model=model)
app.add_source("docs", path=str(docs_dir))
app.set_policy("answer_only_from_sources", True)
app.add_evaluator("groundedness")
app.add_evaluator("faithfulness")
app.add_evaluator("hallucination")


def quality_metrics_and_assertions() -> None:
    """New 0.5 metrics ride on every run; assert them like unit tests."""
    result = app.run("What is the refund window for the Pro plan?", session_id="sess_demo")
    print("eval scores on the run:", result.eval_scores)
    assert_grounded(result, threshold=0.5)
    assert_safe(result)
    print("assertions passed (grounded, safe)")


def synthetic_dataset_and_gates() -> None:
    """Bootstrap a golden dataset from the corpus, then gate the app on it."""
    from vincio.documents import load_document

    documents = [load_document(path) for path in sorted(docs_dir.glob("*.md"))]
    dataset = SyntheticGenerator(seed=7).generate(documents, n=6, name="synth_refunds")
    print(f"synthetic dataset: {len(dataset)} cases from {dataset.metadata['sources']} docs "
          f"({dataset.metadata['generator']})")
    runner = EvalRunner(app, metrics=["groundedness", "answer_relevance", "latency"],
                        gates={"groundedness": ">= 0.5"})
    report = runner.run(dataset, name="synth_run")
    print("gates:", {name: value["passed"] for name, value in report.gates.items()})
    return report


def experiments_with_significance(report) -> None:
    """Track variants locally; test the A/B for statistical significance."""
    tracker = ExperimentTracker(workdir / "experiments.db")
    tracker.log("prompt_ab", report, variant="baseline", params={"prompt": "v1"})
    tracker.log("prompt_ab", report, variant="candidate", params={"prompt": "v2"})
    comparison = tracker.compare("prompt_ab")
    print("experiment best-per-metric:", comparison["best"])
    test = ab_test(report, report, "groundedness")
    print(f"A/B {test['test']}: delta={test['delta']} p={test['p_value']} "
          f"significant={test['significant']}")


def red_team_the_app() -> None:
    """Adversarial probes judged by the security detectors — gate-able."""
    report = RedTeamSuite().run(app)
    summary = report.summary()
    print(f"red team: {summary['probes']} probes, "
          f"attack_success_rate={summary['attack_success_rate']}, "
          f"detector_coverage={summary['detector_coverage']}")


def prompt_registry_versions() -> None:
    """Version prompts like code: push, tag, diff, link eval runs."""
    registry = PromptRegistry(workdir / "prompts")
    registry.push(PromptSpec(name="support", role="support agent",
                             objective="Answer from the documentation only."),
                  tags=["production"], message="initial")
    v2 = registry.push(PromptSpec(name="support", role="support agent",
                                  objective="Answer from the documentation only, citing sources."),
                       message="require citations")
    diff = registry.diff("support", 1, 2)
    print(f"prompt registry: {len(registry.versions('support'))} versions, "
          f"changed={list(diff['changed_fields'])}")
    registry.tag("support", v2.version, "production")
    print("production →", registry.get("support", tag="production").ref)


def sessions_feedback_and_viewer() -> None:
    """Sessions group runs; feedback becomes data; traces become datasets."""
    app.run("How long do refunds take?", session_id="sess_demo")
    exporter = app.tracer.exporter
    traces = exporter.traces if hasattr(exporter, "traces") else exporter.load_all()
    trace = traces[-1]
    trace.add_feedback(score=1.0, comment="clear answer")
    print()
    print(render_trace_text(trace))
    session = next(s for s in sessions_from_traces(traces) if s.id == "sess_demo")
    print("\nsession summary:", session.summary())
    html_path = workdir / "trace.html"
    html_path.write_text(trace_to_html(trace), encoding="utf-8")
    print("self-contained trace HTML:", html_path)
    golden = dataset_from_traces(traces, name="from_production", min_feedback_score=0.5)
    print(f"traces → dataset: {len(golden)} case(s) with provenance")


if __name__ == "__main__":
    quality_metrics_and_assertions()
    report = synthetic_dataset_and_gates()
    experiments_with_significance(report)
    red_team_the_app()
    prompt_registry_versions()
    sessions_feedback_and_viewer()
