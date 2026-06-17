"""Agentic evaluation & continuous quality (1.2).

Vincio can run and trace a crew, a graph, and a tool loop — 1.2 makes it *score*
them, over the trajectory, a multi-turn conversation, and live traffic, reusing
the same metric objects offline, as runtime guardrails, and as optimizer fitness:

  1. Trajectory & tool-use metrics over a real ReAct run (no re-instrumentation),
     shown alongside final-output-only evaluation.
  2. A deterministic multi-turn user Simulator + conversation-outcome scoring.
  3. Online/continuous evaluation writing a score time series, plus drift detection.
  4. Human-in-the-loop annotation with Cohen's-κ judge calibration.
  5. A production A/B over variants, and a metric reused as a runtime guardrail.

Runs fully offline on the deterministic mock provider.
"""

from __future__ import annotations

from _shared import example_provider

from vincio import ContextApp
from vincio.core.config import StorageConfig, VincioConfig
from vincio.evals import (
    AnnotationQueue,
    Dataset,
    DriftMonitor,
    EvalCase,
    Persona,
    RunOutput,
    Simulator,
    metric_guardrail,
)
from vincio.evals.metrics import METRICS


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _memory_app(name: str) -> ContextApp:
    provider, model = example_provider()
    return ContextApp(
        name=name,
        config=VincioConfig(storage=StorageConfig(metadata="memory://")),
        provider=provider,
        model=model,
    )


def trajectory_eval() -> None:
    _section("1. Trajectory & tool-use metrics")
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

    # Score the trajectory WITHOUT re-instrumenting the agent: project the
    # AgentState onto a RunOutput that carries the trajectory.
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
    # Output-only eval is shown alongside trajectory eval (agents pass more
    # output-only cases than the trajectory reveals).
    print("  final-output-only:")
    for name in ("lexical_overlap", "answer_relevance"):
        print(f"    {name:20s} {METRICS[name](case, run).value}")
    print("  trajectory eval:")
    for name in (
        "goal_accuracy", "tool_call_accuracy", "tool_call_f1",
        "plan_adherence", "step_efficiency",
    ):
        print(f"    {name:20s} {METRICS[name](case, run).value}")


def simulator_eval() -> None:
    _section("2. Multi-turn simulator & conversation outcome")

    def assistant(messages: list[dict]) -> str:
        return "To reset your password, open Settings then Security then Reset. It takes 5 minutes."

    persona = Persona(name="sam", goal="reset my password", max_turns=3)
    convo = Simulator(seed=11).simulate(assistant, persona)
    case = convo.to_eval_case(id="reset")
    run = RunOutput(output=convo.turns[-1]["content"])
    print(f"  rounds={convo.rounds} goal_achieved={convo.goal_achieved}")
    for name in ("conversation_outcome", "intent_resolution", "knowledge_retention"):
        print(f"  {name:22s} {METRICS[name](case, run).value}")


def online_and_drift() -> None:
    _section("3. Online eval + drift detection")
    app = _memory_app("online_demo")
    app.add_online_evaluator("answer_relevance", sample_rate=1.0)
    for question in ("How long are refunds?", "What is the fee?", "When does it renew?"):
        app.run(question)
    series = [round(r["metric_value"], 3) for r in app.online_evaluators[0].series()]
    print(f"  online answer_relevance series: {series}")

    monitor = DriftMonitor(score_threshold=0.1)
    monitor.set_score_baseline("goal_accuracy", [0.9, 0.91, 0.89, 0.92])
    stable = monitor.check_scores("goal_accuracy", [0.9, 0.9, 0.91])
    regressed = monitor.check_scores("goal_accuracy", [0.6, 0.62, 0.58])
    print(f"  drift on stable window:    {stable.drifted}")
    print(f"  drift on regressed window: {regressed.drifted} (delta {regressed.delta})")


def annotation_eval() -> None:
    _section("4. Human annotation & Cohen's kappa")
    queue = AnnotationQueue(name="judge_cal")
    for i, (judge, human) in enumerate([(0.9, 1.0), (0.2, 0.0), (0.8, 0.7), (0.1, 0.0), (0.95, 0.9)]):
        item = queue.add(run_id=f"r{i}", judge_score=judge)
        queue.label(item.id, human)
    agreement = queue.agreement()
    print(f"  cohens_kappa={agreement['cohens_kappa']}  n={agreement['n']}")
    print(f"  judge earns CI-gating weight: {queue.judge_trusted(threshold=0.6)}")


def ab_and_guardrail() -> None:
    _section("5. A/B experiment & metric-as-guardrail")
    app = _memory_app("ab_demo")
    dataset = Dataset(
        name="golden",
        cases=[
            EvalCase(id="c1", input="refund window?", expected="30 days"),
            EvalCase(id="c2", input="renewal notice?", expected="60 days"),
        ],
    )
    exp = app.experiment(
        "prompt_ab",
        variants={"baseline": {"model": "mock-1"}, "variant_b": {"model": "mock-1"}},
        dataset=dataset,
        metrics=["lexical_overlap", "cost"],
    )
    print(f"  cost per variant: {exp.cost()}")
    print(f"  significance test: {exp.significance('lexical_overlap')['variant_b']['test']}")

    # The same metric, now a runtime guardrail.
    guard = metric_guardrail("toxicity", threshold=0.0)
    print(f"  toxicity guard (clean): {guard('A helpful answer.', {})}")
    print(f"  toxicity guard (toxic): {guard('you are an idiot', {})}")


def main() -> None:
    trajectory_eval()
    simulator_eval()
    online_and_drift()
    annotation_eval()
    ab_and_guardrail()


if __name__ == "__main__":
    main()
