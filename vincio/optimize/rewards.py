"""Verifiable rewards for on-policy reinforcement.

On-policy reinforcement learning from verifiable rewards (RLVR) turns the
signals the platform *already* computes — the stateful-environment task-success
oracle, the nine agentic benchmark scorers, and the calibrated judge ensembles —
into a reward a policy can be optimized against. The point of *verifiable* is
that the reward is grounded in a checkable end state (a database mutation, a test
transition, an exact match, a solvable tool path), not a free-floating
model opinion.

A :class:`VerifiableReward` maps one :class:`RewardSample` (a scored unit of
work) to a :class:`RewardSignal` — a dense value in ``[0, 1]``, a verifiable
``success`` flag, and a **confidence weight**. A :class:`RewardModel` composes
several into one blended signal, weighting each by its own confidence. The
judge-ensemble reward is the reason confidence matters: a *split* panel
(:meth:`~vincio.evals.ensemble.JudgeEnsemble.averdict` flags it ``uncertain``)
**down-weights itself** so the blend leans on the verifiable scorers rather than
rewarding noise — disagreement lowers influence, it does not earn reward.

Everything runs offline and deterministically against the reference
environments, adapters, and the mock-backed judges.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import VincioError
from ..evals.environment import TaskVerification
from ..evals.trajectory import Trajectory
from ..providers.base import run_sync

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..evals.benchmarks import BenchmarkAdapter
    from ..evals.ensemble import JudgeEnsemble

__all__ = [
    "RewardError",
    "RewardSignal",
    "RewardSample",
    "VerifiableReward",
    "OracleReward",
    "BenchmarkReward",
    "JudgeEnsembleReward",
    "RewardModel",
]


class RewardError(VincioError):
    """A reward could not be derived (no verifiable signal on the sample)."""

    code = "REWARD_ERROR"


class RewardSignal(BaseModel):
    """One verifiable reward: a dense value, a success flag, and a confidence.

    ``value`` is the dense reward in ``[0, 1]`` (a partial-credit fraction, not
    just pass/fail). ``success`` is the verifiable binary outcome. ``weight`` is
    the signal's confidence in ``[0, 1]`` — ``1.0`` for a deterministic scorer,
    lower for a judge panel that disagrees with itself — and is what a
    :class:`RewardModel` uses to discount a noisy component.
    """

    value: float = 0.0
    success: bool = False
    weight: float = 1.0
    source: str = ""
    components: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class RewardSample(BaseModel):
    """A scored unit of work, carrying whatever verifiable artifacts it has.

    A reward reads only the fields it needs: :class:`OracleReward` reads
    ``verification`` (or a trajectory's success), :class:`BenchmarkReward` reads
    ``output`` / ``gold`` / ``inputs`` against an adapter's scorer, and
    :class:`JudgeEnsembleReward` reads ``output`` / ``gold`` / ``prompt``. One
    sample can therefore feed a whole reward panel.
    """

    model_config = {"arbitrary_types_allowed": True}

    task_id: str = ""
    prompt: str = ""
    output: Any = None  # the agent's answer / action list / function call(s)
    gold: Any = None
    inputs: dict[str, Any] = Field(default_factory=dict)  # adapter inputs (env, apis, …)
    trajectory: Trajectory | None = None
    verification: TaskVerification | None = None  # the environment oracle's verdict
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifiableReward(ABC):
    """Base contract: map a :class:`RewardSample` to a :class:`RewardSignal`."""

    name: str = "verifiable_reward"

    @abstractmethod
    async def aevaluate(self, sample: RewardSample) -> RewardSignal:
        """Derive the reward for one sample (async)."""

    def evaluate(self, sample: RewardSample) -> RewardSignal:
        """Synchronous wrapper over :meth:`aevaluate`."""
        return run_sync(self.aevaluate(sample))


class OracleReward(VerifiableReward):
    """Dense reward from the stateful-environment task-success oracle.

    Reads the :class:`~vincio.evals.environment.TaskVerification` produced by
    ``Environment.verify()`` (or driven by the :class:`EnvironmentSimulator`):
    the reward is the fraction of end-state checks satisfied (``dense=True``) or a
    bare pass/fail (``dense=False``), and ``success`` is the oracle's verdict. A
    sample with neither a verification nor a trajectory has nothing verifiable to
    score, so it raises rather than inventing a reward.
    """

    name = "oracle"

    def __init__(self, *, dense: bool = True) -> None:
        self.dense = dense

    async def aevaluate(self, sample: RewardSample) -> RewardSignal:
        verification = sample.verification
        if verification is None:
            if sample.trajectory is not None:
                passed = bool(sample.trajectory.success)
                return RewardSignal(
                    value=1.0 if passed else 0.0, success=passed, source=self.name,
                    details={"from": "trajectory.success"},
                )
            raise RewardError(
                "OracleReward needs a sample.verification (or a trajectory); "
                "drive the EnvironmentSimulator and pass its verification"
            )
        value = verification.score if self.dense else (1.0 if verification.passed else 0.0)
        return RewardSignal(
            value=round(float(value), 6),
            success=verification.passed,
            source=self.name,
            components={c.name: 1.0 if c.passed else 0.0 for c in verification.checks},
            details={"reason": verification.reason},
        )


class BenchmarkReward(VerifiableReward):
    """Reward from a benchmark adapter's verifiable scorer.

    Wraps any :class:`~vincio.evals.benchmarks.BenchmarkAdapter` so its
    leaderboard criterion — SWE-bench's test transition, τ-bench's database end
    state, GAIA's exact match, BFCL's AST match, ToolBench's solvable path — *is*
    the reward (``score`` for dense partial credit, ``success`` for the binary
    pass). The same scorer that ranks the agent on the board now improves it.
    """

    def __init__(self, adapter: BenchmarkAdapter, *, dense: bool = True) -> None:
        self.adapter = adapter
        self.dense = dense
        self.name = f"benchmark:{adapter.name}"

    async def aevaluate(self, sample: RewardSample) -> RewardSignal:
        from ..evals.benchmarks import BenchmarkTask

        task = BenchmarkTask(
            id=sample.task_id or "reward-task",
            prompt=sample.prompt,
            inputs=dict(sample.inputs),
            gold=sample.gold,
        )
        result = await self.adapter.score(task, sample.output)
        value = result.score if self.dense else (1.0 if result.success else 0.0)
        return RewardSignal(
            value=round(float(value), 6),
            success=result.success,
            source=self.name,
            details=dict(result.details),
        )


class JudgeEnsembleReward(VerifiableReward):
    """Reward from a judge ensemble whose **disagreement down-weights itself**.

    The panel's aggregate score is the reward; its score *spread* is the inverse
    of the confidence. When the judges agree the signal carries full weight; when
    they split (the verdict is flagged ``uncertain``) the weight drops to
    ``1 − spread`` (floored at ``min_weight``), so a :class:`RewardModel` leans on
    the verifiable scorers instead of a coin-flip. The panel still only earns
    reward weight at all once it is κ-calibrated against human labels — the same
    bar :class:`~vincio.evals.ensemble.JudgeEnsemble` holds for CI gating; pass
    ``require_calibration=True`` to enforce it.
    """

    name = "judge_ensemble"

    def __init__(
        self,
        ensemble: JudgeEnsemble,
        *,
        min_weight: float = 0.0,
        require_calibration: bool = False,
    ) -> None:
        self.ensemble = ensemble
        self.min_weight = max(0.0, min(1.0, min_weight))
        self.require_calibration = require_calibration

    async def aevaluate(self, sample: RewardSample) -> RewardSignal:
        from ..evals.datasets import EvalCase
        from ..evals.metrics import RunOutput

        if self.require_calibration and self.ensemble.gating_weight() <= 0.0:
            return RewardSignal(
                value=0.0, weight=0.0, source=self.name,
                details={"uncalibrated": True, "kappa": self.ensemble.kappa},
            )
        case = EvalCase(id=sample.task_id or "reward-case", input=sample.prompt, expected=sample.gold)
        output_text = sample.output if isinstance(sample.output, str) else ""
        run_output = RunOutput(output=sample.output, raw_text=output_text)
        verdict = await self.ensemble.averdict(case, run_output)
        weight = max(self.min_weight, 1.0 - verdict.spread)
        return RewardSignal(
            value=verdict.value,
            success=verdict.value >= 0.5 and not verdict.uncertain,
            weight=round(weight, 6),
            source=self.name,
            details={
                "disagreement": verdict.disagreement,
                "uncertain": verdict.uncertain,
                "scores": verdict.scores,
                "calibrated": verdict.calibrated,
            },
        )


class RewardModel:
    """Compose verifiable rewards into one dense, confidence-weighted signal.

    Each component contributes ``static_weight · confidence · value``; the
    aggregate is the confidence-weighted mean of the component values, so a
    judge ensemble that disagrees (low confidence) is pulled toward the
    verifiable scorers automatically. ``success`` is reported two ways: the
    aggregate clears ``success_threshold``, *and* the OR of every component's own
    verifiable success — so a deterministic pass is never masked by a cautious
    judge. The components run concurrently.
    """

    def __init__(
        self,
        rewards: list[Any],
        *,
        success_threshold: float = 0.5,
        name: str = "reward_model",
    ) -> None:
        if not rewards:
            raise ValueError("RewardModel requires at least one VerifiableReward")
        self.rewards: list[VerifiableReward] = []
        self.weights: list[float] = []
        for entry in rewards:
            if isinstance(entry, tuple):
                reward, weight = entry
            else:
                reward, weight = entry, 1.0
            self.rewards.append(reward)
            self.weights.append(float(weight))
        self.success_threshold = success_threshold
        self.name = name

    async def aevaluate(self, sample: RewardSample) -> RewardSignal:
        signals = await asyncio.gather(*(r.aevaluate(sample) for r in self.rewards))
        numerator = 0.0
        denominator = 0.0
        components: dict[str, float] = {}
        any_success = False
        for static_weight, signal in zip(self.weights, signals, strict=True):
            effective = static_weight * signal.weight
            numerator += effective * signal.value
            denominator += effective
            components[signal.source or "reward"] = round(signal.value, 6)
            any_success = any_success or signal.success
        value = numerator / denominator if denominator > 0 else 0.0
        return RewardSignal(
            value=round(value, 6),
            success=value >= self.success_threshold or any_success,
            weight=1.0,
            source=self.name,
            components=components,
            details={"effective_weight": round(denominator, 6)},
        )

    def evaluate(self, sample: RewardSample) -> RewardSignal:
        """Synchronous wrapper over :meth:`aevaluate`."""
        return run_sync(self.aevaluate(sample))
