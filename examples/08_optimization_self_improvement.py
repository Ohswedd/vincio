"""Optimization & the self-improving loop.

How a Vincio app gets *better over time* without leaving the process or breaking
its safety discipline. The spine is one gated, audited loop —
trace -> dataset -> eval -> optimize -> promote — where every promotion must
clear a no-regression gate, so improvement is monotonic by construction.

This tour focuses on four rungs of that loop, shown deeply:
  1. The closed loop itself: real traffic -> feedback -> dataset -> gated promote.
  2. The distillation flywheel: grounded traces -> a gated, cheaper student.
  3. RLVR: on-policy reinforcement from a VERIFIABLE reward (app.learn).
  4. On-device LoRA adaptation: in-process, gated, reversible.
Runs fully offline. (Closing note points to the rest of the plane.)
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from _shared import example_provider, write_sample_docs

from vincio import AdapterRegistry, ContextApp, LocalAdaptationPolicy, VincioConfig
from vincio.evals.datasets import Dataset, EvalCase
from vincio.evals.environment import (
    EnvAction,
    EnvironmentSimulator,
    build_retail_environment,
    scripted_policy,
)
from vincio.observability.sessions import record_feedback
from vincio.optimize import (
    BootstrapFinetune,
    CandidateOutcome,
    FitnessWeights,
    ImprovementLoop,
    LearningTask,
    OracleReward,
    RewardModel,
    RewardSample,
    export_training_set,
)
from vincio.optimize.distill import TrainingExample, TrainingSet

DOCS_DIR = write_sample_docs(Path(tempfile.mkdtemp()) / "docs")


def _config() -> VincioConfig:
    # In-memory metadata + in-process exporter, no audit file: the demo is
    # reproducible on re-run and never touches disk or network.
    config = VincioConfig()
    config.storage.metadata = "memory://"
    config.observability.exporter = "memory"
    config.security.audit_log = False
    return config


def _format_sensitive_responder(request):
    """Answers correctly everywhere but only CITES when the prompt is XML-rendered
    (contains a closing tag) — a genuine, learnable signal the optimizer can find."""
    import re
    text = "\n".join(m.text for m in request.messages)
    answer = "The refund window for the Pro plan is 30 days."
    if "</" not in text:
        return answer  # plain prompt: correct but uncited
    match = re.search(r"\[([\w.:-]+:C\d+)\]", text)
    return f"{answer} [{match.group(1) if match else 'E1'}]"


def _qa_app() -> ContextApp:
    provider, model = example_provider(_format_sensitive_responder)
    config = _config()
    config.memory.write_back = ["facts"]  # grounded run claims become candidate memories
    app = ContextApp(name="optimize_demo", provider=provider, model=model, config=config)
    app.add_source("docs", path=str(DOCS_DIR))
    app.add_memory()
    return app


async def section_closed_loop() -> None:
    app = _qa_app()

    # Real traffic: run a few paraphrases, then mark the answers approved. Traces
    # with positive feedback become the eval dataset — the loop mines production,
    # it doesn't need a hand-authored golden set.
    for q in ["What is the refund window for the Pro plan?",
              "How long do Pro customers have to request a refund?",
              "Within how many days can a Pro plan be refunded?", "Pro plan refund period?"]:
        await app.arun(q, user_id="u1")
    for trace in app.tracer.exporter.traces:
        record_feedback(trace, score=1.0)  # users approved these answers

    # The loop optimizes citation quality. The baseline answers correctly but does
    # not cite; promotion requires a variant that cites WITHOUT regressing
    # groundedness (the explicit gate) or safety/schema (built-in rules). The
    # winner is pushed to the prompt registry and tagged production.
    loop = ImprovementLoop(
        app, metrics=["lexical_overlap", "groundedness", "citation_accuracy", "cost", "latency"],
        weights=FitnessWeights(accuracy_metric="citation_accuracy"),
        gates={"groundedness": ">= 0.5"}, experiment="refund_qa")
    result = await loop.arun(min_feedback_score=0.5, max_variants=6, subset_size=4)
    print(f"1. closed loop: dataset {result.dataset_name} ({result.dataset_size} cases from traces) "
          f"-> promoted={result.promoted} ({result.reason})")
    if result.promoted_ref:
        version = loop.registry.get(loop.prompt_name, tag="production")
        print(f"   registry: {result.promoted_ref} tags={version.tags} eval_runs={len(version.eval_runs)}")


async def section_distillation() -> None:
    from types import SimpleNamespace

    from vincio.core.types import EvidenceItem
    from vincio.evals.reports import CaseResult, EvalReport

    evidence = [EvidenceItem(id="D1:C0", source_id="D1",
                text="Customers on the Pro plan may request refunds within 30 days.", provenance=0.9)]

    def trace(tid, inp, out):
        return SimpleNamespace(id=tid, run_id=tid, session_id=None, status="ok", feedback=[],
                               attributes={"input": inp, "output": out,
                                           "evidence": [e.model_dump() for e in evidence]})

    # Only GROUNDED traces become training data — the ungrounded axolotl claim is
    # dropped, so the student never learns a hallucination.
    traces = [trace("t1", "Refund window?", "The Pro plan refund window is 30 days."),
              trace("t2", "Mascot?", "The mascot is a purple axolotl with 12 legs.")]
    training_set = export_training_set(traces, require_grounding=True, min_support=0.4)

    # Teacher -> student gate: the student is promoted ONLY if it HOLDS quality
    # (>= min_quality_ratio of the teacher) while being cheaper.
    async def evaluate_model(model, ds):
        q, cost = (0.95, 0.01) if model == "teacher" else (0.93, 0.002)
        return EvalReport(cases=[CaseResult(case_id=f"c{i}", metrics={"lexical_overlap": q, "cost": cost})
                                 for i in range(len(ds))])

    dataset = Dataset(name="held", cases=[EvalCase(id=f"c{i}", input="q") for i in range(6)])
    result = await BootstrapFinetune(evaluate_model, min_quality_ratio=0.9).distill(
        training_set, dataset, teacher="teacher", student="student")
    print(f"2. distillation: exported {len(training_set)} grounded (dropped "
          f"{training_set.metadata['dropped_ungrounded']} ungrounded) -> "
          f"student promoted={result.promoted}, holds {result.quality_ratio:.0%} quality "
          f"at {result.cost_savings:.0%} lower cost")


def _run_env(actions):
    env = build_retail_environment("cancel_refund")
    return EnvironmentSimulator().run(env, scripted_policy([EnvAction(**a) for a in actions]))


def section_rlvr() -> None:
    # The correct trajectory cancels BEFORE refunding; the violation refunds a
    # still-processing order, which the task-success oracle rejects.
    good = _run_env([{"kind": "tool", "tool": "cancel_order", "arguments": {"order_id": "O1002"}},
                     {"kind": "tool", "tool": "refund_order", "arguments": {"order_id": "O1002"}}])
    bad = _run_env([{"kind": "tool", "tool": "refund_order", "arguments": {"order_id": "O1002"}}])

    # The reward is not a learned guess: it is the task-success ORACLE the platform
    # already computes — the database end-state becomes a checkable reward. app.learn
    # runs a GRPO-style update: group-relative advantage, a KL-to-reference clamp so
    # the policy stays near reference, and a no-regression gate on served reward.
    task = LearningTask(id="refund", prompt="Cancel order O1002 and refund it.", candidates=[
        CandidateOutcome(action="cancel_then_refund",
                         sample=RewardSample(task_id="refund", verification=good.verification),
                         text="cancel order O1002, then issue the refund"),
        CandidateOutcome(action="refund_only",
                         sample=RewardSample(task_id="refund", verification=bad.verification),
                         text="issue the refund"),
    ])
    result = ContextApp(name="rlvr", config=_config()).learn(
        [task], reward=RewardModel([OracleReward()]), kl_max=0.5, iterations=6, learning_rate=0.8)
    print(f"3. RLVR: promoted={result.promoted}, reward {result.baseline_reward} -> "
          f"{result.policy_reward} (Δ{result.reward_delta:+.4f}), KL {result.kl_to_reference} "
          f"within bound={result.kl_within_bound}, monotonic={result.reward_monotonic}")


_QA = [("what is the refund policy", "Refunds are processed within 30 days."),
       ("how do I reset my password", "Use the reset link on the login page."),
       ("what are the shipping options", "We ship worldwide via DHL in 5-7 days."),
       ("how do I contact support", "Email support@example.com any time.")]


def section_local_adaptation() -> None:
    from vincio.providers.mock import MockProvider

    # A base model that does not know the grounded answers. The continual loop fits
    # a parameter-efficient low-rank adapter ON-DEVICE from the grounded set — pure
    # Python, no network — and promotes it ONLY when the adapted model is
    # at-least-as-good as the base on a held-out set (the same no-regression
    # discipline a hosted fine-tune clears). Unloading restores the base exactly.
    app = ContextApp(name="edge", provider=MockProvider(default_text="I am not sure about that."),
                     config=_config())
    training = TrainingSet(name="local-adapter", examples=[TrainingExample(messages=[
        {"role": "user", "content": q}, {"role": "assistant", "content": a}]) for q, a in _QA])
    golden = Dataset(name="golden", cases=[EvalCase(id=f"c{i}", input=q, expected=a)
                                           for i, (q, a) in enumerate(_QA)])
    result = app.adapt_locally(golden, training_set=training, registry=AdapterRegistry(),
                               policy=LocalAdaptationPolicy(min_examples=4, min_samples=4,
                                                            require_significance=False))
    adapted = app.run(_QA[0][0]).raw_text
    app.use_local_adapter(None)  # reversible: unload restores the base
    print(f"4. local LoRA: promoted={result.promoted}, base {result.verdict.baseline:.2f} -> "
          f"adapted {result.verdict.candidate:.2f} | live answer {adapted!r}, "
          f"after unload {app.run(_QA[0][0]).raw_text!r}")


async def main() -> None:
    await section_closed_loop()
    await section_distillation()
    section_rlvr()
    section_local_adaptation()
    # The SAME gated, audited path also powers the rest of the plane:
    #   * reflective GEPA-style search that reads WHY the baseline lost — app.reflective_optimize
    #   * one declarative governed contract — optimize.SelfImprovementPolicy / app.self_improvement
    #   * canary-gated deploy with automatic rollback — app.deploy(..., canary=, rollback_on_fail=)
    #   * federated cross-org improvement under a differential-privacy accountant —
    #     app.adopt_federated + app.use_privacy_accountant (per-subject (ε, δ) budget)
    #   * open-ended skill acquisition — app.cultivate over an AutoCurriculum
    print("\nOne gated, audited path — improvement is monotonic by construction.")


if __name__ == "__main__":
    asyncio.run(main())
