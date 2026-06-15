"""Tests for 1.2 — agentic evaluation & continuous quality.

Trajectory & tool-use metrics, the multi-turn simulator + conversation metrics,
online/continuous eval, drift detection, human annotation with Cohen's κ,
production A/B, and the metric-as-guardrail interconnection. Plus a regression
test for the Gemini embedding cost-table fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vincio import ContextApp
from vincio.core.config import StorageConfig, VincioConfig
from vincio.evals import (
    AnnotationQueue,
    Dataset,
    DriftMonitor,
    EvalCase,
    GEvalJudge,
    Persona,
    RunOutput,
    Simulator,
    cohens_kappa,
    dataset_from_traces,
    metric_guardrail,
)
from vincio.evals.metrics import METRICS
from vincio.evals.trajectory import Trajectory, TrajectoryStep
from vincio.observability.spans import Trace
from vincio.providers import MockProvider

GOLDEN = Path(__file__).resolve().parent / "golden" / "agentic_eval.jsonl"


def _memory_app(name: str, **kwargs) -> ContextApp:
    return ContextApp(
        name=name,
        config=VincioConfig(storage=StorageConfig(metadata="memory://")),
        provider=MockProvider(default_text=kwargs.pop("text", "Refunds take 30 days.")),
        model="mock-1",
        **kwargs,
    )


def _trajectory(**kwargs) -> Trajectory:
    steps = [
        TrajectoryStep(type="tool", tool_name="search", tool_arguments={"q": "refund"}, status="done"),
        TrajectoryStep(type="tool", tool_name="summarize", status="done"),
        TrajectoryStep(type="finalize", name="finalize", status="done"),
    ]
    defaults = dict(objective="find the refund window", success=True, steps=steps,
                    final_answer="30 days", source="agent_state")
    defaults.update(kwargs)
    return Trajectory(**defaults)


# -- trajectory & tool-use metrics ------------------------------------------


class TestTrajectoryMetrics:
    def _case(self) -> EvalCase:
        return EvalCase(
            id="c", input="find the refund window", expected="30 days",
            rubric={"expected_tools": ["search", "summarize"], "plan": ["tool", "tool", "finalize"],
                    "optimal_steps": 3},
        )

    def test_tool_call_accuracy_perfect(self):
        run = RunOutput(output="30 days", trajectory=_trajectory())
        assert METRICS["tool_call_accuracy"](self._case(), run).value == 1.0

    def test_tool_call_accuracy_wrong_order(self):
        traj = _trajectory(steps=[
            TrajectoryStep(type="tool", tool_name="summarize", status="done"),
            TrajectoryStep(type="tool", tool_name="search", status="done"),
        ])
        run = RunOutput(output="x", trajectory=traj)
        # positions both wrong → 0.0
        assert METRICS["tool_call_accuracy"](self._case(), run).value == 0.0

    def test_tool_call_f1_order_insensitive(self):
        traj = _trajectory(steps=[
            TrajectoryStep(type="tool", tool_name="summarize", status="done"),
            TrajectoryStep(type="tool", tool_name="search", status="done"),
        ])
        run = RunOutput(output="x", trajectory=traj)
        assert METRICS["tool_call_f1"](self._case(), run).value == 1.0

    def test_tool_call_f1_penalizes_missing_and_spurious(self):
        traj = _trajectory(steps=[TrajectoryStep(type="tool", tool_name="search", status="done"),
                                  TrajectoryStep(type="tool", tool_name="delete", status="done")])
        run = RunOutput(output="x", trajectory=traj)
        result = METRICS["tool_call_f1"](self._case(), run)
        assert 0.0 < result.value < 1.0

    def test_goal_accuracy_requires_answer_match(self):
        traj = _trajectory(success=True, final_answer="totally wrong")
        run = RunOutput(output="totally wrong answer about nothing", trajectory=traj)
        # success but answer mismatch → 0.5
        assert METRICS["goal_accuracy"](self._case(), run).value == 0.5

    def test_goal_accuracy_no_trajectory_neutral(self):
        run = RunOutput(output="x")
        assert METRICS["goal_accuracy"](EvalCase(id="c", input="q"), run).value == 1.0

    def test_plan_adherence(self):
        run = RunOutput(output="30 days", trajectory=_trajectory())
        assert METRICS["plan_adherence"](self._case(), run).value == 1.0

    def test_plan_quality_penalizes_redundancy(self):
        traj = _trajectory(steps=[
            TrajectoryStep(type="tool", tool_name="search", status="done"),
            TrajectoryStep(type="tool", tool_name="search", status="done"),
            TrajectoryStep(type="finalize", status="done"),
        ])
        run = RunOutput(output="x", trajectory=traj)
        assert METRICS["plan_quality"](self._case(), run).value == pytest.approx(0.6667, abs=0.01)

    def test_step_efficiency_optimal(self):
        run = RunOutput(output="30 days", trajectory=_trajectory())
        assert METRICS["step_efficiency"](self._case(), run).value == 1.0

    def test_step_efficiency_penalizes_extra_steps(self):
        steps = [TrajectoryStep(type="tool", tool_name=f"t{i}", status="done") for i in range(6)]
        run = RunOutput(output="x", trajectory=_trajectory(steps=steps))
        assert METRICS["step_efficiency"](self._case(), run).value == pytest.approx(0.5, abs=0.01)

    def test_topic_adherence_with_arguments(self):
        traj = _trajectory(objective="refund window pro plan", steps=[
            TrajectoryStep(type="tool", tool_name="search", tool_arguments={"q": "refund window pro plan"}, status="done"),
        ])
        run = RunOutput(output="x", trajectory=traj)
        assert METRICS["topic_adherence"](EvalCase(id="c", input="refund window pro plan"), run).value == 1.0


class TestTrajectoryAdapters:
    def test_from_trace(self):
        trace = Trace(status="ok", attributes={"input": "q", "output": "a"})
        with_tool = trace
        from vincio.observability.spans import Span
        trace.spans.append(Span(name="search", type="tool_call", status="ok",
                                attributes={"tool": "search", "arguments": {"q": "x"}}))
        run = RunOutput.from_trace(with_tool)
        assert run.trajectory is not None
        assert run.trajectory.tool_names() == ["search"]
        assert run.trajectory.success is True

    def test_from_agent_state_uses_tool_results(self):
        from vincio.agents.state import AgentState, AgentStep
        from vincio.core.types import Objective, ToolResult

        state = AgentState(
            objective=Objective("find the refund window"),
            steps=[AgentStep(type="think", name="react_0", status="done")],
            tool_results=[ToolResult(call_id="1", tool_name="search", status="ok"),
                          ToolResult(call_id="2", tool_name="summarize", status="ok")],
            final_answer="30 days",
            terminated=True,
            termination_reason="objective_complete",
        )
        run = RunOutput.from_agent_state(state)
        assert run.trajectory.tool_names() == ["search", "summarize"]
        assert run.trajectory.success is True


def test_golden_trajectory_agreement():
    """New metrics agree with the labeled traces in tests/golden/."""
    dataset = Dataset.load(GOLDEN)
    assert len(dataset) >= 5
    checked = 0
    for case in dataset:
        traj = case.context.get("trajectory")
        if traj:
            run = RunOutput(output=traj.get("final_answer"), trajectory=Trajectory.model_validate(traj))
        else:
            messages = case.context.get("messages", [])
            last = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")
            run = RunOutput(output=last)
        for metric, expected in case.rubric.get("labels", {}).items():
            assert METRICS[metric](case, run).value == pytest.approx(expected, abs=0.02), (
                f"{case.id}:{metric}"
            )
            checked += 1
    assert checked >= 10


# -- simulator & conversation metrics ---------------------------------------


class TestSimulator:
    def test_determinism(self):
        def agent(messages):
            return "Open Settings then Security then Reset your password."

        persona = Persona(name="sam", goal="reset my password", max_turns=3)
        a = Simulator(seed=7).simulate(agent, persona)
        b = Simulator(seed=7).simulate(agent, persona)
        assert [t["content"] for t in a.turns] == [t["content"] for t in b.turns]

    def test_goal_achieved_and_eval_case(self):
        def agent(messages):
            return "To reset your password go to settings and reset. Done in 5 minutes."

        convo = Simulator(seed=1).simulate(agent, Persona(goal="reset password", max_turns=3))
        assert convo.goal_achieved is True
        case = convo.to_eval_case(id="s")
        assert "messages" in case.context
        assert METRICS["conversation_outcome"](case, RunOutput(output=convo.turns[-1]["content"])).value > 0.5

    def test_max_turns_bound(self):
        def agent(messages):
            return "I am not going to help."  # never satisfies goal

        convo = Simulator(seed=3).simulate(agent, Persona(goal="reset password", max_turns=2))
        assert convo.rounds <= 2


def test_conversation_outcome_keywords():
    case = EvalCase(id="c", input="reset password",
                    context={"messages": [{"role": "assistant", "content": "go to settings and reset"}]},
                    rubric={"goal_keywords": ["settings", "reset", "missing"]})
    result = METRICS["conversation_outcome"](case, RunOutput(output=""))
    assert result.value == pytest.approx(2 / 3, abs=0.01)


def test_dataset_from_traces_multi_turn():
    t1 = Trace(session_id="s1", status="ok", attributes={"input": "hi", "output": "hello"})
    t2 = Trace(session_id="s1", status="ok", attributes={"input": "reset password", "output": "open settings"})
    t3 = Trace(session_id="s2", status="ok", attributes={"input": "bye", "output": "goodbye"})
    dataset = dataset_from_traces([t1, t2, t3], group_by_session=True)
    assert len(dataset) == 2
    s1 = next(c for c in dataset if c.id == "s1")
    assert len(s1.context["messages"]) == 4
    assert "multi_turn" in s1.tags


# -- online / continuous eval -----------------------------------------------


class TestOnlineEval:
    async def test_writes_time_series(self):
        app = _memory_app("online")
        app.add_online_evaluator("answer_relevance", sample_rate=1.0)
        await app.arun("How long are refunds?")
        await app.arun("What is the fee?")
        await app.aflush_online()
        series = app.online_evaluators[0].series()
        assert len(series) == 2
        assert all(0.0 <= r["metric_value"] <= 1.0 for r in series)

    async def test_sampling(self):
        app = _memory_app("online_sampled")
        app.add_online_evaluator("cost", sample_rate=0.5)
        for _ in range(4):
            await app.arun("q")
        await app.aflush_online()
        # 1-in-2 deterministic sampling → 2 of 4 runs scored.
        assert len(app.online_evaluators[0].series()) == 2

    def test_sync_run_flushes(self):
        app = _memory_app("online_sync")
        app.add_online_evaluator("latency", sample_rate=1.0)
        app.run("hello")
        assert app.online_evaluators[0].series()


# -- drift -------------------------------------------------------------------


class TestDrift:
    def test_score_drift(self):
        monitor = DriftMonitor(score_threshold=0.1)
        monitor.set_score_baseline("goal_accuracy", [0.9, 0.91, 0.89, 0.92])
        assert monitor.check_scores("goal_accuracy", [0.9, 0.9, 0.91]).drifted is False
        assert monitor.check_scores("goal_accuracy", [0.6, 0.62, 0.58]).drifted is True

    def test_embedding_drift(self):
        monitor = DriftMonitor(embedding_threshold=0.15)
        monitor.set_embedding_baseline([[1, 0, 0], [0.9, 0.1, 0], [0.95, 0.05, 0]])
        assert monitor.check_embeddings([[0.92, 0.08, 0]]).drifted is False
        assert monitor.check_embeddings([[0, 1, 0], [0, 0, 1]]).drifted is True

    def test_emits_event(self):
        from vincio.core.events import EventBus

        bus = EventBus()
        seen = []
        bus.subscribe("drift.detected", lambda e: seen.append(e.payload))
        monitor = DriftMonitor(bus=bus, score_threshold=0.1)
        monitor.set_score_baseline("m", [0.9, 0.9, 0.9])
        monitor.check_scores("m", [0.5, 0.5, 0.5])
        assert seen and seen[0]["metric"] == "m"


# -- annotation & Cohen's kappa ---------------------------------------------


class TestAnnotation:
    def test_cohens_kappa_perfect(self):
        assert cohens_kappa([(0.9, 0.9), (0.1, 0.1), (0.8, 0.7), (0.2, 0.3)], bins=2) == 1.0

    def test_cohens_kappa_disagreement(self):
        assert cohens_kappa([(0.9, 0.1), (0.1, 0.9), (0.8, 0.2), (0.2, 0.8)], bins=2) < 0.0

    def test_queue_lifecycle_and_trust(self):
        queue = AnnotationQueue(name="q")
        items = [queue.add(run_id=f"r{i}", judge_score=j) for i, j in enumerate([0.9, 0.2, 0.8, 0.1, 0.95])]
        assert len(queue.pending()) == 5
        for item, human in zip(items, [1.0, 0.0, 0.7, 0.0, 0.9], strict=True):
            queue.label(item.id, human)
        assert not queue.pending()
        assert queue.judge_trusted(threshold=0.6) is True
        assert queue.gating_weight(threshold=0.6) == 1.0

    def test_geval_calibrate_reports_kappa(self):
        judge = GEvalJudge(provider=None, model="x", criteria="relevance")
        fit = judge.calibrate([(0.9, 1.0), (0.2, 0.0), (0.8, 0.7), (0.1, 0.0)])
        assert "cohens_kappa" in fit
        assert judge.gating_weight(threshold=0.6) == 1.0


# -- A/B experiment ----------------------------------------------------------


class TestExperiment:
    def _dataset(self) -> Dataset:
        return Dataset(name="g", cases=[
            EvalCase(id="c1", input="refund window?", expected="30 days"),
            EvalCase(id="c2", input="renewal?", expected="60 days"),
        ])

    def test_compare_cost_significance(self):
        app = _memory_app("ab")
        exp = app.experiment(
            "ab1",
            variants={"baseline": {"model": "mock-1"}, "variant_b": {"model": "mock-1"}},
            dataset=self._dataset(),
            metrics=["semantic_similarity", "cost"],
        )
        comparison = exp.compare()
        assert set(comparison["variants"]) == {"baseline", "variant_b"}
        assert set(exp.cost()) == {"baseline", "variant_b"}
        sig = exp.significance("semantic_similarity")
        assert "variant_b" in sig and "p_value" in sig["variant_b"]


# -- metric as guardrail -----------------------------------------------------


class TestMetricGuardrail:
    def test_toxicity_guard(self):
        guard = metric_guardrail("toxicity", threshold=0.0)
        assert guard("a helpful answer", {}) is None
        assert guard("you are an idiot", {}) is not None

    def test_relevance_guard_direction(self):
        guard = metric_guardrail("answer_relevance", threshold=0.3)
        # higher-is-better → fires when BELOW threshold
        assert guard("completely unrelated", {"input": "what is the refund window"}) is not None

    def test_add_metric_rail(self):
        app = _memory_app("rails")
        app.add_metric_rail("toxicity", threshold=0.0)
        assert any(r.name == "toxicity_guard" for r in app.rail_engine.rails)


# -- optimizer interconnection & cost fix -----------------------------------


def test_agentic_objectives_use_trajectory_metrics():
    from vincio.optimize import AGENTIC_OBJECTIVES

    metrics = {o.metric for o in AGENTIC_OBJECTIVES}
    assert {"goal_accuracy", "tool_call_accuracy"} <= metrics


def test_gemini_embedding_cost_is_billed():
    """Regression: the default Gemini embedding model must have a non-zero price
    (it was absent from the table, so embedding cost resolved to $0)."""
    from vincio.observability.costs import CostTracker, default_price_table

    assert default_price_table().lookup("gemini-embedding-001").input_per_mtok > 0
    tracker = CostTracker()
    assert tracker.record_embedding("gemini-embedding-001", 1_000_000) > 0
