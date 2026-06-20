"""On-policy reinforcement from verifiable rewards (RLVR).

Covers the reward model (verifiable + judge-ensemble down-weighting), the
GRPO group-relative advantage, step-level Shapley credit assignment, and the
trajectory optimizer's math: advantage normalization, the KL-to-reference clamp,
and the monotonic no-regression gate. All offline and deterministic.
"""

from __future__ import annotations

import math

import pytest

from vincio import ContextApp
from vincio.core.shapley import is_efficient, shapley_values
from vincio.evals.datasets import EvalCase
from vincio.evals.ensemble import JudgeEnsemble
from vincio.evals.environment import (
    EnvAction,
    EnvironmentSimulator,
    make_retail_environment,
    scripted_policy,
)
from vincio.evals.judges import Judge
from vincio.evals.metrics import MetricResult, RunOutput
from vincio.optimize import (
    BenchmarkReward,
    CandidateOutcome,
    JudgeEnsembleReward,
    LearningTask,
    OracleReward,
    RewardError,
    RewardModel,
    RewardSample,
    SoftmaxPolicy,
    TrajectoryAdvantage,
    TrajectoryOptimizer,
    compute_group_advantages,
    environment_step_value,
    kl_divergence,
    no_regression_gate,
)
from vincio.optimize.distill import BootstrapFinetune, DistillationResult

# ---------------------------------------------------------------------------
# Fixtures: deterministic retail-environment trajectories
# ---------------------------------------------------------------------------


def _run(actions: list[dict]):
    env = make_retail_environment("cancel_refund")
    policy = scripted_policy([EnvAction(**a) for a in actions])
    return EnvironmentSimulator().run(env, policy)


CORRECT = [
    {"kind": "tool", "tool": "cancel_order", "arguments": {"order_id": "O1002"}},
    {"kind": "tool", "tool": "refund_order", "arguments": {"order_id": "O1002"}},
]
POLICY_VIOLATION = [  # refund before cancel — the oracle rejects it
    {"kind": "tool", "tool": "refund_order", "arguments": {"order_id": "O1002"}},
]


# ---------------------------------------------------------------------------
# Group-relative advantage + divergence
# ---------------------------------------------------------------------------


def test_group_advantages_are_standardized():
    adv = compute_group_advantages([0.0, 0.5, 1.0])
    assert abs(sum(adv)) < 1e-9  # mean-centered
    # standardized to unit population std
    assert pytest.approx(adv[2], abs=1e-6) == math.sqrt(1.5)
    assert adv[0] < 0 < adv[2]


def test_group_advantages_degenerate_group_is_zero():
    assert compute_group_advantages([0.7, 0.7, 0.7]) == [0.0, 0.0, 0.0]


def test_group_advantages_mean_centered_when_not_normalized():
    adv = compute_group_advantages([0.0, 1.0], normalize=False)
    assert adv == [-0.5, 0.5]


def test_kl_divergence_is_zero_iff_identical_and_positive_otherwise():
    assert kl_divergence([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert kl_divergence([0.9, 0.1], [0.5, 0.5]) > 0.0


# ---------------------------------------------------------------------------
# Verifiable rewards
# ---------------------------------------------------------------------------


async def test_oracle_reward_dense_and_binary():
    good, bad = _run(CORRECT), _run(POLICY_VIOLATION)
    dense = OracleReward(dense=True)
    binary = OracleReward(dense=False)
    g = await dense.aevaluate(RewardSample(verification=good.verification))
    b = await dense.aevaluate(RewardSample(verification=bad.verification))
    assert g.value == 1.0 and g.success
    assert 0.0 < b.value < 1.0 and not b.success  # partial credit, not a pass
    bb = await binary.aevaluate(RewardSample(verification=bad.verification))
    assert bb.value == 0.0  # binary collapses partial credit


async def test_oracle_reward_falls_back_to_trajectory_success():
    good = _run(CORRECT)
    reward = OracleReward()
    signal = await reward.aevaluate(RewardSample(trajectory=good.trajectory))
    assert signal.success and signal.value == 1.0


async def test_oracle_reward_requires_a_verifiable_signal():
    with pytest.raises(RewardError):
        await OracleReward().aevaluate(RewardSample(task_id="empty"))


async def test_benchmark_reward_uses_the_adapter_scorer():
    from vincio.evals.benchmarks import GAIAAdapter

    reward = BenchmarkReward(GAIAAdapter())
    hit = await reward.aevaluate(RewardSample(task_id="q1", output="42", gold="42"))
    miss = await reward.aevaluate(RewardSample(task_id="q2", output="7", gold="42"))
    assert hit.success and hit.value == 1.0
    assert not miss.success and miss.value == 0.0


class _FixedJudge(Judge):
    """A deterministic judge that returns a fixed score (to build a panel)."""

    def __init__(self, value: float, *, name: str) -> None:
        self.value = value
        self.name = name

    async def score(self, case: EvalCase, output: RunOutput) -> MetricResult:
        return MetricResult(name=self.name, value=self.value)


async def test_judge_ensemble_reward_downweights_on_disagreement():
    agree = JudgeEnsembleReward(
        JudgeEnsemble([_FixedJudge(0.9, name="a"), _FixedJudge(0.92, name="b")])
    )
    split = JudgeEnsembleReward(
        JudgeEnsemble([_FixedJudge(0.1, name="a"), _FixedJudge(0.9, name="b")])
    )
    sample = RewardSample(prompt="q", output="ans", gold="ans")
    agreed = await agree.aevaluate(sample)
    disagreed = await split.aevaluate(sample)
    # An agreeing panel keeps near-full confidence; a split panel down-weights.
    assert agreed.weight > 0.9
    assert disagreed.weight < agreed.weight
    assert disagreed.details["uncertain"] is True


async def test_judge_reward_requires_calibration_when_asked():
    reward = JudgeEnsembleReward(
        JudgeEnsemble([_FixedJudge(0.9, name="a"), _FixedJudge(0.9, name="b")]),
        require_calibration=True,
    )
    signal = await reward.aevaluate(RewardSample(prompt="q", output="a"))
    assert signal.weight == 0.0  # uncalibrated panel cannot earn reward weight


async def test_reward_model_blend_leans_on_verifiable_when_judge_splits():
    good = _run(CORRECT)
    split_judge = JudgeEnsembleReward(
        JudgeEnsemble([_FixedJudge(0.0, name="a"), _FixedJudge(1.0, name="b")])
    )
    model = RewardModel([(OracleReward(), 1.0), (split_judge, 1.0)])
    sample = RewardSample(prompt="q", output="ans", verification=good.verification)
    signal = await model.aevaluate(sample)
    # Oracle says 1.0 with full confidence; the split judge (low confidence,
    # value 0.5) is discounted, so the blend stays close to the verifiable 1.0.
    assert signal.value > 0.8
    assert signal.success


# ---------------------------------------------------------------------------
# Softmax policy
# ---------------------------------------------------------------------------


def test_softmax_policy_uniform_then_skewed():
    policy = SoftmaxPolicy()
    assert policy.probabilities("t", ["a", "b"]) == [0.5, 0.5]
    policy.set_logit("t", "a", 2.0)
    probs = policy.probabilities("t", ["a", "b"])
    assert probs[0] > probs[1]
    assert policy.best_action("t", ["a", "b"]) == "a"


def test_softmax_policy_copy_is_independent():
    policy = SoftmaxPolicy()
    policy.set_logit("t", "a", 1.0)
    clone = policy.copy()
    clone.set_logit("t", "a", 5.0)
    assert policy.logit("t", "a") == 1.0


# ---------------------------------------------------------------------------
# Step-level Shapley credit assignment
# ---------------------------------------------------------------------------


def test_trajectory_advantage_credits_the_enabling_step():
    good = _run(CORRECT)
    advantage = TrajectoryAdvantage(
        environment_step_value(lambda: make_retail_environment("cancel_refund"))
    )
    credits = advantage.credit(good.trajectory)
    by_name = {c.name: c.credit for c in credits}
    # Cancelling is the precondition for the refund, so it earns the larger share.
    assert by_name["cancel_order"] > by_name["refund_order"] > 0
    assert advantage.explained  # credits sum to the attributable value


def test_trajectory_advantage_guards_against_too_many_players():
    good = _run(CORRECT)
    advantage = TrajectoryAdvantage(
        environment_step_value(lambda: make_retail_environment("cancel_refund")),
        max_players=1,
    )
    with pytest.raises(ValueError, match="exceeds max_players"):
        advantage.credit(good.trajectory)


def test_shapley_kernel_efficiency_holds():
    # v(S) = number of players in S squared — superadditive, with interactions.
    shapley, cache = shapley_values(["x", "y", "z"], lambda s: float(len(s) ** 2))
    assert is_efficient(["x", "y", "z"], shapley, cache)


# ---------------------------------------------------------------------------
# The no-regression gate (pure)
# ---------------------------------------------------------------------------


def test_no_regression_gate_blocks_a_regressor():
    promoted, reason = no_regression_gate(0.8, 0.5, 0.1, kl_max=0.5)
    assert not promoted and "regress" in reason


def test_no_regression_gate_blocks_kl_drift():
    promoted, reason = no_regression_gate(0.5, 0.9, 0.9, kl_max=0.5)
    assert not promoted and "KL" in reason


def test_no_regression_gate_promotes_a_safe_improvement():
    promoted, _ = no_regression_gate(0.5, 0.8, 0.2, kl_max=0.5)
    assert promoted


def test_no_regression_gate_respects_min_improvement():
    promoted, reason = no_regression_gate(0.5, 0.51, 0.1, kl_max=0.5, min_improvement=0.1)
    assert not promoted and "bar" in reason


# ---------------------------------------------------------------------------
# The trajectory optimizer
# ---------------------------------------------------------------------------


def _refund_task() -> LearningTask:
    good, bad = _run(CORRECT), _run(POLICY_VIOLATION)
    return LearningTask(
        id="refund",
        prompt="Cancel order O1002 and refund it.",
        candidates=[
            CandidateOutcome(
                action="cancel_then_refund",
                sample=RewardSample(task_id="refund", verification=good.verification),
                text="cancel then refund",
            ),
            CandidateOutcome(
                action="refund_only",
                sample=RewardSample(task_id="refund", verification=bad.verification),
                text="refund only",
            ),
        ],
    )


async def test_optimizer_improves_reward_within_kl_bound():
    optimizer = TrajectoryOptimizer(
        RewardModel([OracleReward()]), kl_max=0.5, iterations=6, learning_rate=0.8
    )
    result = await optimizer.alearn([_refund_task()])
    assert result.promoted
    assert result.policy_reward > result.baseline_reward  # monotone improvement
    assert result.reward_monotonic
    assert result.kl_to_reference <= result.kl_bound + 1e-9  # KL clamp held
    assert result.kl_within_bound
    assert result.recommended["refund"] == "cancel_then_refund"
    assert result.verdict is not None and result.verdict.passed


async def test_optimizer_kl_clamp_binds_under_a_tight_bound():
    optimizer = TrajectoryOptimizer(
        RewardModel([OracleReward()]), kl_max=0.02, iterations=10, learning_rate=2.0
    )
    result = await optimizer.alearn([_refund_task()])
    # The clamp must keep the policy inside the tight trust region.
    assert result.kl_to_reference <= 0.02 + 1e-9
    assert result.kl_within_bound


async def test_optimizer_never_serves_a_regression():
    # A degenerate group (equal rewards) yields zero advantage and no real gain;
    # with a positive improvement bar the gate refuses to promote and the served
    # policy reverts to (never regresses) the baseline.
    good = _run(CORRECT)
    flat = LearningTask(
        id="flat",
        prompt="q",
        candidates=[
            CandidateOutcome(action="a", sample=RewardSample(verification=good.verification)),
            CandidateOutcome(action="b", sample=RewardSample(verification=good.verification)),
        ],
    )
    optimizer = TrajectoryOptimizer(
        RewardModel([OracleReward()]), iterations=5, min_reward_improvement=0.1
    )
    result = await optimizer.alearn([flat])
    assert not result.promoted
    assert result.policy_reward >= result.baseline_reward - 1e-9  # never regresses
    assert result.training_set is None  # nothing emitted on a blocked run


async def test_optimizer_emits_on_policy_training_set():
    optimizer = TrajectoryOptimizer(
        RewardModel([OracleReward()]), iterations=5, learning_rate=0.8
    )
    result = await optimizer.alearn([_refund_task()])
    assert result.training_set is not None
    assert len(result.training_set.examples) == 1
    example = result.training_set.examples[0]
    assert example.messages[-1]["content"] == "cancel then refund"  # the winner


async def test_optimizer_runs_the_flywheel_on_promotion():
    # A trained student that holds quality at lower cost promotes through the
    # existing flywheel from the on-policy winners.
    from vincio.evals.datasets import Dataset
    from vincio.evals.reports import CaseResult, EvalReport

    async def evaluate_model(model: str, dataset: Dataset) -> EvalReport:
        quality = 0.9 if "student" in model else 0.9
        cost = 0.001 if "student" in model else 0.01
        cases = [
            CaseResult(case_id=f"c{i}", metrics={"lexical_overlap": quality, "cost": cost})
            for i in range(6)
        ]
        return EvalReport(name=model, dataset="held", cases=cases)

    async def trainer(training_set, base_model: str) -> str:
        return f"{base_model}-student"

    flywheel = BootstrapFinetune(evaluate_model, trainer=trainer, min_quality_ratio=0.9)
    held = Dataset(name="held", cases=[EvalCase(id=f"c{i}", input="q") for i in range(6)])
    optimizer = TrajectoryOptimizer(
        RewardModel([OracleReward()]), iterations=5, learning_rate=0.8
    )
    result = await optimizer.alearn(
        [_refund_task()], flywheel=flywheel, held_out=held, teacher="teacher", student="student"
    )
    assert result.promoted
    assert isinstance(result.distillation, DistillationResult)
    assert result.distillation.promoted


# ---------------------------------------------------------------------------
# app.learn integration
# ---------------------------------------------------------------------------


def test_app_learn_promotes_and_audits():
    app = ContextApp(name="rl_app")
    before = len(app.audit.entries) if hasattr(app.audit, "entries") else 0
    result = app.learn([_refund_task()], reward=RewardModel([OracleReward()]), learning_rate=0.8)
    assert result.promoted
    # the decision is recorded on the audit chain
    learn_records = [e for e in app.audit.entries if e.action == "learn"]
    assert learn_records and learn_records[-1].decision == "allow"
    assert before <= len(app.audit.entries)


def test_app_learn_accepts_a_bare_verifiable_reward():
    app = ContextApp(name="rl_bare")
    result = app.learn([_refund_task()], reward=OracleReward(), learning_rate=0.8)
    assert result.promoted
