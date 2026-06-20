"""On-policy reinforcement: group-relative advantage and the trajectory optimizer.

The self-improvement loop already reaches ``trace → dataset → eval → optimize →
promote`` through reflective prompt optimization, MIPRO, learned compression, and
the distillation flywheel. The rung this module adds is **on-policy
reinforcement from verifiable rewards (RLVR)**: it turns the
:mod:`~vincio.optimize.rewards` signals into a GRPO-style update that improves a
*policy*, not just a prompt — without adding a trainer dependency to the default
path.

Three pieces:

* :func:`compute_group_advantages` — the GRPO group-relative advantage:
  ``A_i = (r_i − mean(r)) / (std(r) + ε)``. Credit is *relative* within a group
  of candidates for the same task, so no separate value network is needed.
* :class:`TrajectoryAdvantage` — step-level credit assignment over a
  trajectory's steps by Shapley counterfactual replay, reusing the shared
  :mod:`vincio.core.shapley` kernel that backs causal regression attribution.
  It attributes the outcome reward back to the tool / retrieval / reasoning steps
  that earned it (drop a step, re-verify the end state, measure the marginal).
* :class:`TrajectoryOptimizer` (``app.learn``) — a GRPO update over a
  deterministic :class:`SoftmaxPolicy`, behind the **same** safety discipline the
  rest of the loop uses: advantage normalization, a **KL-to-reference clamp** so
  the policy never drifts past a trust-region bound, and a **monotonic
  no-regression gate** (the served policy is promoted only if its expected reward
  does not regress the reference — otherwise it reverts). The promoted policy can
  emit a fine-tune job through the existing flywheel and is recorded with the
  same :class:`~vincio.optimize.self_improvement.CanaryVerdict` a prompt deploy
  produces.

The offline path runs the deterministic mock :class:`SoftmaxPolicy` so the
optimizer's *math* is fully tested without a GPU.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.shapley import is_efficient, shapley_values
from ..providers.base import run_sync
from .rewards import RewardModel, RewardSample, VerifiableReward
from .self_improvement import CanaryVerdict

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..evals.datasets import Dataset
    from ..evals.environment import Environment
    from ..evals.trajectory import Trajectory, TrajectoryStep
    from .distill import BootstrapFinetune, DistillationResult, TrainingSet

__all__ = [
    "compute_group_advantages",
    "kl_divergence",
    "StepCredit",
    "TrajectoryAdvantage",
    "environment_step_value",
    "SoftmaxPolicy",
    "CandidateOutcome",
    "LearningTask",
    "PolicyUpdate",
    "LearningResult",
    "no_regression_gate",
    "TrajectoryOptimizer",
]


# ---------------------------------------------------------------------------
# Group-relative advantage (GRPO) and divergence
# ---------------------------------------------------------------------------


def compute_group_advantages(
    rewards: list[float], *, normalize: bool = True, eps: float = 1e-8
) -> list[float]:
    """GRPO group-relative advantage of each reward within its group.

    ``A_i = (r_i − mean) / (std + ε)`` with ``normalize=True`` (the standardized
    form GRPO uses, so the update is scale-free across tasks); mean-centered
    ``r_i − mean`` with ``normalize=False``. A degenerate group (all rewards
    equal, ``std≈0``) yields all-zero advantages — no spurious update from noise.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    centered = [r - mean for r in rewards]
    if not normalize:
        return centered
    var = sum(c * c for c in centered) / n
    std = math.sqrt(var)
    if std < eps:
        return [0.0] * n
    return [c / (std + eps) for c in centered]


def kl_divergence(p: list[float], q: list[float], *, eps: float = 1e-12) -> float:
    """KL(p ‖ q) over two discrete distributions (nats). Non-negative; ``0`` iff
    the distributions are identical."""
    total = 0.0
    for pi, qi in zip(p, q, strict=True):
        if pi <= 0.0:
            continue
        total += pi * math.log((pi + eps) / (qi + eps))
    return max(0.0, total)


# ---------------------------------------------------------------------------
# Step-level credit assignment by Shapley counterfactual replay
# ---------------------------------------------------------------------------


class StepCredit(BaseModel):
    """One step's share of a trajectory's outcome value (its Shapley credit)."""

    index: int
    name: str
    type: str = "step"
    credit: float  # signed, in the value function's units; sums to the total
    share: float  # |credit| / Σ|credit|


# A coalition value function: the value attainable from a subset of kept steps.
StepValueFn = Callable[[list["TrajectoryStep"]], float]


class TrajectoryAdvantage:
    """Attribute a trajectory's outcome reward to the steps that earned it.

    Step-level credit assignment by **Shapley counterfactual replay**: each step
    is a player, a coalition is the ordered subset of *kept* steps, and the
    characteristic function is ``value_fn(kept_steps)`` — for an environment, the
    end state re-verified with only those tool actions applied (see
    :func:`environment_step_value`). The exact Shapley decomposition (the shared
    :mod:`vincio.core.shapley` kernel) gives each step its average marginal
    contribution, and the credits sum to the full trajectory value (efficiency),
    so a step that was *necessary* for success carries its share and a no-op
    carries ~0. By default only tool steps are players (the actions that mutate
    the world); pass ``include`` to widen the set.
    """

    def __init__(
        self,
        value_fn: StepValueFn,
        *,
        include: tuple[str, ...] = ("tool", "tool_call"),
        max_players: int = 12,
    ) -> None:
        self.value_fn = value_fn
        self.include = include
        self.max_players = max_players
        # Whether the last credit() decomposition reconstructed the outcome value
        # (the Shapley efficiency axiom). Set on each credit() call.
        self.explained = True

    def _players(self, trajectory: Trajectory) -> list[int]:
        idx = [
            i
            for i, step in enumerate(trajectory.steps)
            if step.type in self.include or step.is_tool
        ]
        return idx

    def credit(self, trajectory: Trajectory) -> list[StepCredit]:
        """Return the per-step Shapley credit for ``trajectory``."""
        player_idx = self._players(trajectory)
        if len(player_idx) > self.max_players:
            raise ValueError(
                f"TrajectoryAdvantage: {len(player_idx)} attributable steps exceeds "
                f"max_players={self.max_players} (2**k replay would be too large); "
                "raise max_players deliberately or pre-summarize the trajectory"
            )
        steps_by_index = {i: trajectory.steps[i] for i in player_idx}

        def value(coalition: frozenset[int]) -> float:
            kept = [steps_by_index[i] for i in sorted(coalition)]
            return float(self.value_fn(kept))

        shapley, cache = shapley_values(player_idx, value)
        total_abs = sum(abs(v) for v in shapley.values()) or 1.0
        credits = [
            StepCredit(
                index=i,
                name=trajectory.steps[i].name or trajectory.steps[i].type,
                type=trajectory.steps[i].type,
                credit=round(shapley[i], 6),
                share=round(abs(shapley[i]) / total_abs, 4),
            )
            for i in player_idx
        ]
        credits.sort(key=lambda c: abs(c.credit), reverse=True)
        # Efficiency is a property of the decomposition, not an assertion we can
        # silently drop: surface it for callers that gate on it.
        self.explained = is_efficient(player_idx, shapley, cache)
        return credits


def environment_step_value(env_factory: Callable[[], Environment]) -> StepValueFn:
    """A coalition value function that re-verifies an environment's end state
    using only the kept tool steps.

    ``env_factory()`` must return a *fresh* environment each call (so coalitions
    don't leak state). Replaying a subset of the trajectory's tool actions and
    returning the oracle's ``verification.score`` makes credit genuinely
    counterfactual: a step the success depended on (e.g. cancelling before
    refunding) earns its marginal because dropping it drops the score.
    """

    def value(kept_steps: list[TrajectoryStep]) -> float:
        from ..evals.environment import EnvAction, EnvironmentSimulator, scripted_policy

        env = env_factory()
        actions = [
            EnvAction(
                kind="tool",
                tool=step.tool_name or step.name,
                arguments=dict(step.tool_arguments),
            )
            for step in kept_steps
        ]
        result = EnvironmentSimulator().run(env, scripted_policy(actions))
        return result.verification.score

    return value


# ---------------------------------------------------------------------------
# The deterministic mock policy
# ---------------------------------------------------------------------------


class SoftmaxPolicy:
    """A deterministic, GPU-free tabular softmax policy over discrete candidate
    actions per task — the offline mock the optimizer's math is tested on.

    The policy holds a logit per ``(task_id, action)``; unseen entries start at
    ``0`` (a uniform prior over a task's candidates). ``probabilities`` is the
    temperature-scaled softmax; the optimizer ascends the logits on advantage and
    clamps them back toward a reference. A real policy (a fine-tuned model, a
    learned router) implements the same ``probabilities`` surface; the
    optimizer's update math is identical.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self._logits: dict[str, dict[str, float]] = {}

    def logit(self, task_id: str, action: str) -> float:
        return self._logits.get(task_id, {}).get(action, 0.0)

    def set_logit(self, task_id: str, action: str, value: float) -> None:
        self._logits.setdefault(task_id, {})[action] = value

    def probabilities(self, task_id: str, actions: list[str]) -> list[float]:
        zs = [self.logit(task_id, a) / self.temperature for a in actions]
        m = max(zs)
        exps = [math.exp(z - m) for z in zs]
        total = sum(exps) or 1.0
        return [e / total for e in exps]

    def best_action(self, task_id: str, actions: list[str]) -> str:
        probs = self.probabilities(task_id, actions)
        return max(zip(actions, probs, strict=True), key=lambda ap: ap[1])[0]

    def copy(self) -> SoftmaxPolicy:
        clone = SoftmaxPolicy(temperature=self.temperature)
        clone._logits = {task: dict(actions) for task, actions in self._logits.items()}
        return clone


# ---------------------------------------------------------------------------
# Tasks, results, and the gate
# ---------------------------------------------------------------------------


class CandidateOutcome(BaseModel):
    """One candidate action for a task: the discrete action the policy chooses,
    its verifiable :class:`RewardSample`, and the assistant text it would emit
    (for the fine-tune example, defaulting to the action label)."""

    model_config = {"arbitrary_types_allowed": True}

    action: str
    sample: RewardSample
    text: str = ""


class LearningTask(BaseModel):
    """A task the policy learns on: a prompt and the group of candidate outcomes
    the GRPO advantage is computed across."""

    model_config = {"arbitrary_types_allowed": True}

    id: str
    prompt: str = ""
    candidates: list[CandidateOutcome] = Field(default_factory=list)


class PolicyUpdate(BaseModel):
    """The per-task record of one learning run: rewards, advantages, and the
    probability mass moved from before to after."""

    task_id: str
    actions: list[str]
    rewards: list[float]
    advantages: list[float]
    prob_before: list[float]
    prob_after: list[float]


class LearningResult(BaseModel):
    """The outcome of a :class:`TrajectoryOptimizer` run.

    ``policy_reward`` is the *served* policy's expected reward (post-gate, so it
    never regresses ``baseline_reward``); ``candidate_reward`` is the pre-gate
    update's reward (which a blocked run reverts away from). ``verdict`` is the
    same :class:`~vincio.optimize.self_improvement.CanaryVerdict` a prompt deploy
    produces, so a policy promotion lands on the same governance surface.
    """

    model_config = {"arbitrary_types_allowed": True}

    promoted: bool = False
    reason: str = ""
    iterations: int = 0
    tasks: int = 0
    baseline_reward: float = 0.0
    candidate_reward: float = 0.0
    policy_reward: float = 0.0
    reward_delta: float = 0.0
    kl_to_reference: float = 0.0
    kl_bound: float = 0.0
    kl_within_bound: bool = True
    reward_monotonic: bool = True
    step_credit_explained: bool = True
    verdict: CanaryVerdict | None = None
    updates: list[PolicyUpdate] = Field(default_factory=list)
    recommended: dict[str, str] = Field(default_factory=dict)
    policy: Any = None  # the served SoftmaxPolicy
    training_set: Any = None  # the on-policy TrainingSet of winners
    distillation: Any = None  # DistillationResult when a flywheel is wired


def no_regression_gate(
    baseline_reward: float,
    candidate_reward: float,
    kl: float,
    *,
    kl_max: float,
    min_improvement: float = 0.0,
    tol: float = 1e-9,
) -> tuple[bool, str]:
    """The monotonic no-regression gate for a policy update.

    Promote the candidate only when (1) it does not regress the reference's
    expected reward beyond ``tol``, (2) it clears ``min_improvement``, and (3) it
    stays within the KL trust region. A regressing or drifting candidate is
    blocked — the optimizer then serves the reference, so the deployed policy is
    monotone in reward by construction. Pure and side-effect-free so the gate can
    be unit-tested directly with a constructed regressor.
    """
    if candidate_reward < baseline_reward - tol:
        return False, (
            f"candidate regresses expected reward "
            f"({baseline_reward:.6f} → {candidate_reward:.6f}); reverting to reference"
        )
    if kl > kl_max + tol:
        return False, f"policy drifted beyond KL bound (KL={kl:.6f} > {kl_max})"
    if (candidate_reward - baseline_reward) < min_improvement - tol:
        return False, (
            f"reward gain {candidate_reward - baseline_reward:+.6f} below the "
            f"{min_improvement} promotion bar; not promoting"
        )
    return True, (
        f"reward held and improved ({baseline_reward:.6f} → {candidate_reward:.6f}, "
        f"KL={kl:.6f} ≤ {kl_max})"
    )


# ---------------------------------------------------------------------------
# The trajectory optimizer
# ---------------------------------------------------------------------------


class TrajectoryOptimizer:
    """GRPO-style on-policy update over a deterministic policy, safety-gated.

    Given a set of :class:`LearningTask`\\ s — each a group of candidate outcomes
    with verifiable rewards — the optimizer (1) scores every candidate with the
    :class:`~vincio.optimize.rewards.RewardModel`, (2) computes group-relative
    advantages, (3) ascends the policy logits on the advantage-weighted
    objective, (4) **clamps** the policy back inside a KL trust region around the
    reference each iteration, and (5) gates the result on a **monotonic
    no-regression** check before serving it. A promoted policy can emit a
    fine-tune job through the existing distillation flywheel from its on-policy
    winners. All deterministic and offline against the mock policy.
    """

    def __init__(
        self,
        reward_model: RewardModel | VerifiableReward,
        *,
        policy: SoftmaxPolicy | None = None,
        learning_rate: float = 0.5,
        kl_max: float = 0.5,
        iterations: int = 3,
        group_normalize: bool = True,
        min_reward_improvement: float = 0.0,
    ) -> None:
        if isinstance(reward_model, VerifiableReward):
            reward_model = RewardModel([reward_model])
        self.reward_model = reward_model
        self.policy = policy or SoftmaxPolicy()
        self.learning_rate = learning_rate
        self.kl_max = kl_max
        self.iterations = max(1, iterations)
        self.group_normalize = group_normalize
        self.min_reward_improvement = min_reward_improvement

    # -- expected reward / KL over the task set ------------------------------

    @staticmethod
    def _expected_reward(
        policy: SoftmaxPolicy, tasks: list[LearningTask], rewards: dict[str, list[float]]
    ) -> float:
        totals: list[float] = []
        for task in tasks:
            actions = [c.action for c in task.candidates]
            probs = policy.probabilities(task.id, actions)
            totals.append(sum(p * r for p, r in zip(probs, rewards[task.id], strict=True)))
        return sum(totals) / len(totals) if totals else 0.0

    def _mean_kl(
        self, policy: SoftmaxPolicy, reference: SoftmaxPolicy, tasks: list[LearningTask]
    ) -> float:
        kls: list[float] = []
        for task in tasks:
            actions = [c.action for c in task.candidates]
            kls.append(
                kl_divergence(
                    policy.probabilities(task.id, actions),
                    reference.probabilities(task.id, actions),
                )
            )
        return sum(kls) / len(kls) if kls else 0.0

    def _blend(
        self, reference: SoftmaxPolicy, policy: SoftmaxPolicy, t: float, tasks: list[LearningTask]
    ) -> SoftmaxPolicy:
        """A policy whose logits are ``(1−t)·reference + t·policy`` per action."""
        blended = reference.copy()
        for task in tasks:
            for c in task.candidates:
                ref_l = reference.logit(task.id, c.action)
                cur_l = policy.logit(task.id, c.action)
                blended.set_logit(task.id, c.action, (1.0 - t) * ref_l + t * cur_l)
        return blended

    def _clamp_to_reference(
        self, policy: SoftmaxPolicy, reference: SoftmaxPolicy, tasks: list[LearningTask]
    ) -> None:
        """Project the policy back so its mean KL to the reference ≤ ``kl_max``.

        Binary-searches the interpolation factor between the reference and the
        current logits — KL is monotone in the blend, so the largest factor that
        satisfies the bound is found exactly (to numerical tolerance).
        """
        if self._mean_kl(policy, reference, tasks) <= self.kl_max:
            return
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            candidate = self._blend(reference, policy, mid, tasks)
            if self._mean_kl(candidate, reference, tasks) <= self.kl_max:
                lo = mid
            else:
                hi = mid
        final = self._blend(reference, policy, lo, tasks)
        for task in tasks:
            for c in task.candidates:
                policy.set_logit(task.id, c.action, final.logit(task.id, c.action))

    # -- the learning run ----------------------------------------------------

    async def alearn(
        self,
        tasks: list[LearningTask],
        *,
        flywheel: BootstrapFinetune | None = None,
        held_out: Dataset | None = None,
        teacher: str | None = None,
        student: str | None = None,
    ) -> LearningResult:
        """Run the GRPO update over ``tasks`` and return the gated result."""
        if not tasks:
            raise ValueError("TrajectoryOptimizer.alearn requires at least one LearningTask")

        # Score every candidate once — verifiable rewards are fixed signals.
        rewards: dict[str, list[float]] = {}
        for task in tasks:
            if not task.candidates:
                raise ValueError(f"task {task.id!r} has no candidates to learn from")
            signals = [await self.reward_model.aevaluate(c.sample) for c in task.candidates]
            rewards[task.id] = [s.value for s in signals]

        reference = self.policy.copy()
        baseline_reward = self._expected_reward(reference, tasks, rewards)

        prob_before = {
            task.id: self.policy.probabilities(task.id, [c.action for c in task.candidates])
            for task in tasks
        }

        # Gradient ascent on Σ_i π_i · A_i (the GRPO surrogate). The gradient
        # w.r.t. a logit is π_j·(A_j − Ā), where Ā is the policy-averaged
        # advantage — pushing mass toward above-average candidates.
        for _ in range(self.iterations):
            for task in tasks:
                actions = [c.action for c in task.candidates]
                advantages = compute_group_advantages(
                    rewards[task.id], normalize=self.group_normalize
                )
                probs = self.policy.probabilities(task.id, actions)
                baseline_adv = sum(p * a for p, a in zip(probs, advantages, strict=True))
                for action, prob, adv in zip(actions, probs, advantages, strict=True):
                    grad = prob * (adv - baseline_adv)
                    self.policy.set_logit(
                        task.id, action, self.policy.logit(task.id, action) + self.learning_rate * grad
                    )
            self._clamp_to_reference(self.policy, reference, tasks)

        candidate_reward = self._expected_reward(self.policy, tasks, rewards)
        kl = self._mean_kl(self.policy, reference, tasks)
        promoted, reason = no_regression_gate(
            baseline_reward,
            candidate_reward,
            kl,
            kl_max=self.kl_max,
            min_improvement=self.min_reward_improvement,
        )

        if not promoted:
            # Revert the live policy to the reference — never serve a regressor.
            self.policy = reference.copy()

        served_reward = self._expected_reward(self.policy, tasks, rewards)
        updates = [
            PolicyUpdate(
                task_id=task.id,
                actions=[c.action for c in task.candidates],
                rewards=[round(r, 6) for r in rewards[task.id]],
                advantages=[
                    round(a, 6)
                    for a in compute_group_advantages(rewards[task.id], normalize=self.group_normalize)
                ],
                prob_before=[round(p, 6) for p in prob_before[task.id]],
                prob_after=[
                    round(p, 6)
                    for p in self.policy.probabilities(task.id, [c.action for c in task.candidates])
                ],
            )
            for task in tasks
        ]
        recommended = {
            task.id: self.policy.best_action(task.id, [c.action for c in task.candidates])
            for task in tasks
        }
        verdict = CanaryVerdict(
            passed=promoted,
            metric="reward",
            baseline=round(baseline_reward, 6),
            candidate=round(candidate_reward, 6),
            delta=round(candidate_reward - baseline_reward, 6),
            samples=len(tasks),
            reason=reason,
        )

        result = LearningResult(
            promoted=promoted,
            reason=reason,
            iterations=self.iterations,
            tasks=len(tasks),
            baseline_reward=round(baseline_reward, 6),
            candidate_reward=round(candidate_reward, 6),
            policy_reward=round(served_reward, 6),
            reward_delta=round(candidate_reward - baseline_reward, 6),
            kl_to_reference=round(kl, 6),
            kl_bound=self.kl_max,
            kl_within_bound=kl <= self.kl_max + 1e-9,
            reward_monotonic=served_reward >= baseline_reward - 1e-9,
            verdict=verdict,
            updates=updates,
            recommended=recommended,
            policy=self.policy,
        )

        # Emit a fine-tune job through the existing flywheel from the on-policy
        # winners — the highest-reward candidate per task — only on a promotion.
        if promoted:
            result.training_set = self._winning_training_set(tasks, rewards)
            if flywheel is not None and held_out is not None and teacher and student:
                result.distillation = await self._run_flywheel(
                    flywheel, result.training_set, held_out, teacher=teacher, student=student
                )
        return result

    def learn(self, tasks: list[LearningTask], **kwargs: Any) -> LearningResult:
        """Synchronous wrapper over :meth:`alearn`."""
        return run_sync(self.alearn(tasks, **kwargs))

    # -- flywheel emission ---------------------------------------------------

    def _winning_training_set(
        self, tasks: list[LearningTask], rewards: dict[str, list[float]]
    ) -> TrainingSet:
        """The on-policy training corpus: the best-reward candidate per task,
        rendered as a grounded (prompt → chosen action) fine-tuning example."""
        from .distill import TrainingExample, TrainingSet

        examples: list[TrainingExample] = []
        for task in tasks:
            group = rewards[task.id]
            best_idx = max(range(len(group)), key=lambda i: group[i])
            if group[best_idx] <= 0.0:
                continue  # nothing verifiably good to learn from
            best = task.candidates[best_idx]
            content = best.text or (
                best.sample.output if isinstance(best.sample.output, str) else best.action
            )
            examples.append(
                TrainingExample(
                    messages=[
                        {"role": "user", "content": task.prompt or best.sample.prompt},
                        {"role": "assistant", "content": str(content)},
                    ],
                    support=round(group[best_idx], 4),
                    grounded=True,
                    provenance={"task_id": task.id, "source": "on_policy_rl"},
                )
            )
        return TrainingSet(
            name="on_policy_rl",
            examples=examples,
            metadata={"source": "trajectory_optimizer", "tasks": len(tasks)},
        )

    async def _run_flywheel(
        self,
        flywheel: BootstrapFinetune,
        training_set: TrainingSet,
        held_out: Dataset,
        *,
        teacher: str,
        student: str,
    ) -> DistillationResult:
        return await flywheel.distill(training_set, held_out, teacher=teacher, student=student)
