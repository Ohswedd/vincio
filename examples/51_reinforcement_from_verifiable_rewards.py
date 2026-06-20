"""On-policy reinforcement from verifiable rewards (RLVR).

The closed loop already reaches trace → dataset → eval → optimize → promote
through reflective prompt optimization and the distillation flywheel. This closes
it on a *policy*, not just a prompt: the verifiable signals the platform already
computes — the task-success oracle and the judge ensembles — become a reward, and
a GRPO-style update improves the policy behind the same safety discipline prompt
optimization uses.

Four steps, all offline on the deterministic reference environment (no GPU):

  1. Verifiable rewards: the stateful-environment task-success oracle turns the
     database end state into a dense reward; a judge ensemble's *disagreement*
     down-weights its own contribution so the blend leans on what is checkable.
  2. Step-level credit: a Shapley counterfactual replay attributes the outcome
     reward back to the steps that earned it — the cancel that the refund
     depended on carries the larger share.
  3. The trajectory optimizer (app.learn): a group-relative advantage update with
     a KL-to-reference clamp and a monotonic no-regression gate. The served
     policy never regresses the baseline reward.
  4. Emit a fine-tune job: the on-policy winners feed the existing distillation
     flywheel — turning the reward signal into a cheaper model under the same gate.

Everything here is opt-in and additive; nothing below is required to run Vincio.
"""

from __future__ import annotations

import asyncio

from vincio import ContextApp
from vincio.evals.datasets import Dataset, EvalCase
from vincio.evals.ensemble import JudgeEnsemble
from vincio.evals.environment import (
    EnvAction,
    EnvironmentSimulator,
    make_retail_environment,
    scripted_policy,
)
from vincio.evals.judges import Judge
from vincio.evals.metrics import MetricResult, RunOutput
from vincio.evals.reports import CaseResult, EvalReport
from vincio.optimize import (
    CandidateOutcome,
    JudgeEnsembleReward,
    LearningTask,
    OracleReward,
    RewardModel,
    RewardSample,
    TrajectoryAdvantage,
    environment_step_value,
)
from vincio.optimize.distill import BootstrapFinetune


def _run(actions: list[dict]):
    """Drive the deterministic retail environment through a fixed action list."""
    env = make_retail_environment("cancel_refund")
    policy = scripted_policy([EnvAction(**a) for a in actions])
    return EnvironmentSimulator().run(env, policy)


# The correct trajectory cancels before refunding; the violation refunds a
# still-processing order, which the oracle rejects.
CORRECT = [
    {"kind": "tool", "tool": "cancel_order", "arguments": {"order_id": "O1002"}},
    {"kind": "tool", "tool": "refund_order", "arguments": {"order_id": "O1002"}},
]
VIOLATION = [{"kind": "tool", "tool": "refund_order", "arguments": {"order_id": "O1002"}}]


class _Fixed(Judge):
    """A deterministic judge returning a constant score (to build a panel)."""

    def __init__(self, value: float, name: str) -> None:
        self.value = value
        self.name = name

    async def score(self, case: EvalCase, output: RunOutput) -> MetricResult:
        return MetricResult(name=self.name, value=self.value)


async def verifiable_rewards() -> None:
    print("1. Verifiable rewards")
    good, bad = _run(CORRECT), _run(VIOLATION)

    oracle = OracleReward()
    g = await oracle.aevaluate(RewardSample(verification=good.verification))
    b = await oracle.aevaluate(RewardSample(verification=bad.verification))
    print(f"   oracle reward — correct: {g.value} (success={g.success})")
    print(f"   oracle reward — violation: {b.value} (success={b.success})")

    # A split judge panel down-weights itself rather than rewarding noise.
    split = await JudgeEnsembleReward(
        JudgeEnsemble([_Fixed(0.1, "a"), _Fixed(0.9, "b")])
    ).aevaluate(RewardSample(prompt="q", output="answer"))
    agree = await JudgeEnsembleReward(
        JudgeEnsemble([_Fixed(0.9, "a"), _Fixed(0.92, "b")])
    ).aevaluate(RewardSample(prompt="q", output="answer"))
    print(f"   judge weight — agreeing panel: {agree.weight} | split panel: {split.weight}")


async def step_level_credit() -> None:
    print("\n2. Step-level credit assignment (Shapley counterfactual replay)")
    good = _run(CORRECT)
    advantage = TrajectoryAdvantage(
        environment_step_value(lambda: make_retail_environment("cancel_refund"))
    )
    for credit in advantage.credit(good.trajectory):
        print(f"   {credit.name:14s} credit={credit.credit:+.3f} share={credit.share}")
    print(f"   credits reconstruct the outcome value (efficiency): {advantage.explained}")


def _refund_task() -> LearningTask:
    good, bad = _run(CORRECT), _run(VIOLATION)
    return LearningTask(
        id="refund",
        prompt="Cancel order O1002 and refund it.",
        candidates=[
            CandidateOutcome(
                action="cancel_then_refund",
                sample=RewardSample(task_id="refund", verification=good.verification),
                text="cancel order O1002, then issue the refund",
            ),
            CandidateOutcome(
                action="refund_only",
                sample=RewardSample(task_id="refund", verification=bad.verification),
                text="issue the refund",
            ),
        ],
    )


async def trajectory_optimizer() -> None:
    print("\n3. The trajectory optimizer (app.learn)")
    app = ContextApp(name="rlvr")
    result = app.learn(
        [_refund_task()],
        reward=RewardModel([OracleReward()]),
        kl_max=0.5,
        iterations=6,
        learning_rate=0.8,
    )
    print(f"   promoted={result.promoted} reason={result.reason}")
    print(f"   expected reward: {result.baseline_reward} → {result.policy_reward} "
          f"(Δ={result.reward_delta:+.4f})")
    print(f"   KL to reference: {result.kl_to_reference} (bound {result.kl_bound}, "
          f"within={result.kl_within_bound})")
    print(f"   monotonic (never regresses baseline): {result.reward_monotonic}")
    print(f"   learned recommendation: {result.recommended}")
    print(f"   verdict (same shape a prompt deploy produces): passed={result.verdict.passed}")


async def emit_fine_tune_job() -> None:
    print("\n4. Emit a fine-tune job through the existing flywheel")

    async def evaluate_model(model: str, dataset: Dataset) -> EvalReport:
        cost = 0.001 if "student" in model else 0.01  # the student is cheaper
        cases = [
            CaseResult(case_id=f"c{i}", metrics={"lexical_overlap": 0.9, "cost": cost})
            for i in range(6)
        ]
        return EvalReport(name=model, dataset="held", cases=cases)

    async def trainer(training_set, base_model: str) -> str:
        return f"{base_model}-student"  # offline: a faithful no-op fine-tune

    app = ContextApp(name="rlvr_flywheel")
    flywheel = BootstrapFinetune(evaluate_model, trainer=trainer, min_quality_ratio=0.9)
    held = Dataset(name="held", cases=[EvalCase(id=f"c{i}", input="q") for i in range(6)])
    result = app.learn(
        [_refund_task()],
        reward=RewardModel([OracleReward()]),
        learning_rate=0.8,
        flywheel=flywheel,
        held_out=held,
        teacher="teacher",
        student="student",
    )
    print(f"   on-policy training examples: {len(result.training_set.examples)}")
    distilled = result.distillation
    print(f"   flywheel promoted a cheaper student: {distilled.promoted}")
    print(f"   reason: {distilled.reason}")


async def main() -> None:
    await verifiable_rewards()
    await step_level_credit()
    await trajectory_optimizer()
    await emit_fine_tune_job()


if __name__ == "__main__":
    asyncio.run(main())
