"""The unified self-improvement contract.

The closed loop is built from many organs — gated re-optimization
(:class:`~vincio.optimize.loop.ImprovementLoop`), autonomous experiment proposal
(:class:`~vincio.optimize.loop.ExperimentProposer`), online drift→action
(:class:`~vincio.optimize.controller.ContinuousImprovementController`), guarded
online bandits, and shadow/canary providers. This contract unifies them under
one declarative, governed surface. A
:class:`SelfImprovementPolicy` is the spec: *what to watch, when to propose, how
to meta-optimize, whether to canary, and how much to spend*. A
:class:`SelfImprovementController` is the single engine that drives it — one
streaming controller whose :meth:`SelfImprovementController.astream` emits the
life of a self-improvement cycle as ``observe → proposal → meta → label → canary
→ promote/rollback`` events. Every promotion still passes the *same* gated path
(significance + safety + golden non-regression) the loop always used, and every
decision lands on the same hash-chained audit log and event bus. The culmination
is fewer, truer abstractions — not more features.

Meta-optimization (learned fitness weights + successive-halving over the
strategy/budget grid) and active-learning label acquisition are first-class here
rather than ad-hoc knobs, so the system also decides *how* to tune itself.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .loop import DEFAULT_LOOP_METRICS, ExperimentProposal, ExperimentProposer, ImprovementLoop
from .search import FitnessWeights, significance_report

if TYPE_CHECKING:
    from ..core.app import ContextApp
    from ..evals.datasets import Dataset, GoldenRegressionSuite
    from ..evals.reports import EvalReport
    from ..prompts.registry import PromptRegistry

__all__ = [
    "CanarySpec",
    "MetaSpec",
    "SelfImprovementPolicy",
    "SelfImprovementEvent",
    "SelfImprovementController",
    "CanaryVerdict",
    "DeployResult",
    "LiveCanary",
    "successive_halving",
    "learn_fitness_weights",
    "select_for_labeling",
    "deploy_candidate",
]

SelfImprovementPhase = Literal[
    "observe", "proposal", "meta", "label", "reeval", "canary", "promote", "rollback", "exhausted"
]


# ---------------------------------------------------------------------------
# Declarative policy
# ---------------------------------------------------------------------------


class CanarySpec(BaseModel):
    """How a candidate is qualified before it is deployed live.

    Two qualification modes, same verdict:

    * **Offline gated comparison** (the default `app.deploy(dataset=...)` path) —
      the candidate is evaluated against the current baseline on a held-out set
      and only clears when it does not regress the watched metric beyond
      ``regression_threshold`` (a tie passes — deploying a no-regression change is
      safe) and no significant regression is detected.
    * **Live-traffic canary** (`app.deploy(live_inputs=..., score_fn=...)`) —
      ``percent`` of live runs ramp onto the candidate, each arm is scored online,
      and once ``min_samples`` candidate observations land the same no-regression
      verdict promotes the candidate or freezes + rolls it back (the prompt-layer
      analog of the ``CanaryRouter``).
    """

    metric: str = "lexical_overlap"
    regression_threshold: float = 0.05
    min_samples: int = 4
    require_significance: bool = True
    alpha: float = 0.05
    # Fraction of live traffic ramped onto the candidate in the live-canary path.
    percent: float = 50.0


class MetaSpec(BaseModel):
    """Meta-optimization: let the system choose *how* to optimize.

    ``strategies`` × ``budgets`` is the grid; successive-halving screens it on a
    minibatch and keeps the best config for the full run. ``learn_weights`` lets
    the fitness weights adapt toward the metric with the most headroom.
    """

    enabled: bool = True
    strategies: list[str] = Field(default_factory=lambda: ["reflective", "evolution"])
    budgets: list[int] = Field(default_factory=lambda: [4, 8])
    halving_rounds: int = 2
    learn_weights: bool = True


class SelfImprovementPolicy(BaseModel):
    """One declarative, governed contract for continual self-improvement."""

    metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_LOOP_METRICS))
    # Scheduling: which live signals trigger a cycle when the controller is
    # attached to the bus. (Manual ``step``/``astream`` always works.)
    triggers: list[str] = Field(default_factory=lambda: ["drift", "online_eval", "manual"])
    propose: bool = True  # autonomous experiment proposal ranks the weakest metric
    online: bool = True  # act on sustained live drift (debounced, budgeted)
    canary: CanarySpec | None = Field(default_factory=CanarySpec)
    active_learning: bool = False  # acquire human labels for the most-uncertain cases
    label_budget: int = 5
    meta: MetaSpec | None = Field(default_factory=MetaSpec)
    eval_budget: float = 48.0
    gates: dict[str, str] | None = None
    sustain: int = 2
    cooldown_s: float = 300.0
    dry_run: bool = False
    promote_tag: str = "production"


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------


class CanaryVerdict(BaseModel):
    """The gated comparison that qualifies (or blocks) a deploy."""

    passed: bool
    metric: str = ""
    baseline: float = 0.0
    candidate: float = 0.0
    delta: float = 0.0
    samples: int = 0
    significance: dict[str, Any] | None = None
    reason: str = ""


class DeployResult(BaseModel):
    """Outcome of a canary-gated prompt/policy deployment."""

    deployed: bool = False
    ref: str | None = None
    rolled_back_to: str | None = None
    verdict: CanaryVerdict | None = None
    reason: str = ""


class SelfImprovementEvent(BaseModel):
    """One event in a self-improvement cycle — stamped on the audit chain."""

    phase: SelfImprovementPhase = "observe"
    metric: str = ""
    action: str = ""
    reason: str = ""
    proposal: dict[str, Any] | None = None
    chosen_strategy: str | None = None
    chosen_budget: int | None = None
    label_case_ids: list[str] = Field(default_factory=list)
    verdict: CanaryVerdict | None = None
    promoted_ref: str | None = None
    rolled_back_to: str | None = None
    budget_spent: float = 0.0
    budget_remaining: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Meta-optimization primitives
# ---------------------------------------------------------------------------

# evaluate(config) -> score (higher is better)
_ScoreFn = Callable[[Any], Awaitable[float]]


def _loop_optimizer(strategy: str) -> tuple[str, str]:
    """Map a meta-selected strategy to ``(optimizer, inner_strategy)`` for the
    ``ImprovementLoop``.

    ``"evolution"`` runs the blind variant search (its inner strategy is
    unused); ``"reflective"`` / ``"mipro"`` both run the GEPA-style reflective
    optimizer, differing only in its inner strategy (instruction-edit vs. joint
    instruction+example proposal)."""
    if strategy == "evolution":
        return "evolution", "reflective"
    return "reflective", strategy


async def successive_halving(
    configs: list[Any],
    evaluate: _ScoreFn,
    *,
    rounds: int = 2,
    keep_fraction: float = 0.5,
) -> tuple[Any, list[dict[str, Any]]]:
    """Successive-halving over a config grid: cheaply screen, keep the best half.

    Returns ``(best_config, history)``. Deterministic given a deterministic
    ``evaluate`` — the screen runs every surviving config each round and keeps
    ``ceil(n·keep_fraction)`` (always at least one), so a grid converges to a
    single winner within ``rounds`` halvings."""
    import math

    survivors = list(configs)
    history: list[dict[str, Any]] = []
    if not survivors:
        raise ValueError("successive_halving requires at least one config")
    for round_idx in range(max(1, rounds)):
        scored: list[tuple[Any, float]] = []
        for config in survivors:
            score = await evaluate(config)
            scored.append((config, score))
            history.append({"round": round_idx, "config": config, "score": round(score, 6)})
        scored.sort(key=lambda cs: cs[1], reverse=True)
        keep = max(1, math.ceil(len(scored) * keep_fraction))
        survivors = [config for config, _ in scored[:keep]]
        if len(survivors) == 1:
            break
    return survivors[0], history


def learn_fitness_weights(
    report: EvalReport | None,
    *,
    base: FitnessWeights | None = None,
    targets: dict[str, float] | None = None,
    boost: float = 0.5,
) -> FitnessWeights:
    """Adapt fitness weights toward the metric with the most headroom.

    Reads the current report's means and raises the weight of whichever
    higher-is-better quality metric sits furthest below its target, so the next
    optimization round pushes hardest where the system is weakest. Returns a new
    :class:`~vincio.optimize.search.FitnessWeights`; the input is never mutated.
    """
    weights = (base or FitnessWeights()).model_copy(deep=True)
    if report is None:
        return weights
    targets = targets or {
        "lexical_overlap": 0.8,
        "groundedness": 0.85,
        "faithfulness": 0.85,
        "answer_relevance": 0.8,
        "schema_validity": 0.98,
    }
    field_for = {
        "lexical_overlap": "accuracy",
        "groundedness": "groundedness",
        "schema_validity": "schema_validity",
    }
    worst_metric: str | None = None
    worst_gap = 0.0
    for metric, target in targets.items():
        values = report.metric_values(metric)
        if not values:
            continue
        gap = target - (sum(values) / len(values))
        if gap > worst_gap:
            worst_gap, worst_metric = gap, metric
    if worst_metric is not None:
        field = field_for.get(worst_metric)
        if field is not None and hasattr(weights, field):
            setattr(weights, field, getattr(weights, field) + boost * worst_gap)
    return weights


def select_for_labeling(
    report: EvalReport,
    *,
    metric: str = "lexical_overlap",
    budget: int = 5,
) -> list[str]:
    """Active-learning acquisition: pick the most *uncertain* cases to label.

    Uncertainty is distance from the decision midpoint (0.5): a case scoring near
    0.5 is the one a human label resolves most. Returns up to ``budget`` case ids,
    most-uncertain first — the queue a human (or the annotation tool) works next.
    """
    scored: list[tuple[str, float]] = []
    for case in report.cases:
        value = case.metrics.get(metric)
        if value is None:
            continue
        uncertainty = 1.0 - abs(float(value) - 0.5) * 2.0
        scored.append((case.case_id, uncertainty))
    scored.sort(key=lambda cu: cu[1], reverse=True)
    return [case_id for case_id, _ in scored[:budget]]


# ---------------------------------------------------------------------------
# Canary-gated deployment
# ---------------------------------------------------------------------------


async def _evaluate_spec(
    app: ContextApp, spec: Any, compiler_options: Any, dataset: Dataset, metrics: list[str]
) -> EvalReport:
    """Evaluate one prompt spec against a dataset without polluting memory or
    leaving the app's live prompt changed (same discipline the loop uses)."""
    from ..evals.runners import EvalRunner

    original_spec = app.prompt_spec
    original_options = app.prompt_compiler.options
    original_write_back = app.config.memory.write_back
    app.prompt_spec = spec
    if compiler_options is not None:
        app.prompt_compiler.options = compiler_options
    app.config.memory.write_back = []
    try:
        return await EvalRunner(app, metrics=metrics).arun(dataset, name="canary")
    finally:
        app.prompt_spec = original_spec
        app.prompt_compiler.options = original_options
        app.config.memory.write_back = original_write_back


def _canary_verdict(
    baseline_report: EvalReport,
    candidate_report: EvalReport,
    spec: CanarySpec,
) -> CanaryVerdict:
    """Decide a canary: no metric regression beyond the threshold, and no
    significant regression at the configured confidence."""
    metric = spec.metric
    base_vals = baseline_report.metric_values(metric)
    cand_vals = candidate_report.metric_values(metric)
    baseline = sum(base_vals) / len(base_vals) if base_vals else 0.0
    candidate = sum(cand_vals) / len(cand_vals) if cand_vals else 0.0
    delta = candidate - baseline
    samples = min(len(base_vals), len(cand_vals))
    # Too few canary samples to trust the comparison: refuse rather than deploy
    # on noise.
    if samples < spec.min_samples:
        return CanaryVerdict(
            passed=False, metric=metric, baseline=round(baseline, 6),
            candidate=round(candidate, 6), delta=round(delta, 6), samples=samples,
            reason=f"insufficient canary samples ({samples} < {spec.min_samples})",
        )
    sig = (
        significance_report(baseline_report, candidate_report, metric, alpha=spec.alpha)
        if spec.require_significance
        else None
    )
    # Block a significant regression outright; otherwise pass when the candidate
    # holds within the no-regression band.
    if sig is not None and sig.get("significant") and sig.get("delta", 0.0) < 0:
        return CanaryVerdict(
            passed=False, metric=metric, baseline=round(baseline, 6),
            candidate=round(candidate, 6), delta=round(delta, 6), samples=samples,
            significance=sig, reason="candidate significantly regresses the canary metric",
        )
    passed = delta >= -spec.regression_threshold
    return CanaryVerdict(
        passed=passed, metric=metric, baseline=round(baseline, 6),
        candidate=round(candidate, 6), delta=round(delta, 6), samples=samples,
        significance=sig,
        reason=(
            f"{metric} held within {spec.regression_threshold} (Δ={delta:+.4f})"
            if passed
            else f"{metric} regressed beyond {spec.regression_threshold} (Δ={delta:+.4f})"
        ),
    )


def _finalize_deploy(
    app: ContextApp,
    spec: Any,
    compiler_options: Any,
    verdict: CanaryVerdict,
    *,
    registry: PromptRegistry,
    prompt_name: str,
    tag: str,
    candidate_report: Any = None,
    dataset_name: str = "",
    rollback_on_fail: bool = True,
    dry_run: bool = False,
    live: bool = False,
) -> DeployResult:
    """Refuse + (optionally) roll back, or promote — the shared decision both the
    offline and live-canary deploy paths take once a verdict is in hand."""
    result = DeployResult(verdict=verdict, reason=verdict.reason)

    if not verdict.passed:
        app.audit.record(
            "deploy", decision="deny", resource=prompt_name,
            details={"reason": verdict.reason, "verdict": verdict.model_dump(), "live": live},
        )
        if rollback_on_fail:
            try:
                versions = registry.versions(prompt_name)
            except Exception:  # noqa: BLE001 - no history is a no-op
                versions = []
            known_good = next((v for v in reversed(versions) if tag in v.tags), None)
            if known_good is not None:
                new_head = registry.rollback(prompt_name, to_version=known_good.version)
                app.prompt_spec = new_head.spec
                result.rolled_back_to = known_good.ref
                result.reason = f"{verdict.reason}; rolled back to {known_good.ref}"
                app.events.emit(
                    "deploy.rolled_back", {"prompt": prompt_name, "restored": known_good.ref}
                )
        return result

    if dry_run:
        result.reason = f"dry run: would deploy ({verdict.reason})"
        return result

    registry.push(app.prompt_spec, name=prompt_name, message="deploy baseline")
    version = registry.push(
        spec, name=prompt_name, tags=[tag], message=f"canary-gated deploy: {verdict.reason}"
    )
    if candidate_report is not None:
        registry.link_eval(prompt_name, version.version, candidate_report, dataset=dataset_name)
    app.prompt_spec = spec
    if compiler_options is not None:
        app.prompt_compiler.options = compiler_options
    result.deployed = True
    result.ref = version.ref
    result.reason = f"deployed {version.ref}: {verdict.reason}"
    app.audit.record(
        "deploy", decision="allow", resource=prompt_name,
        details={
            "ref": version.ref, "tag": tag, "metric": verdict.metric,
            "delta": verdict.delta, "significance": verdict.significance, "live": live,
        },
    )
    app.events.emit(
        "deploy.completed",
        {"prompt": version.ref, "tag": tag, "metric": verdict.metric, "live": live},
    )
    return result


class LiveCanary:
    """Qualify a candidate prompt/policy on **live traffic** and auto-roll-back.

    The prompt-layer analog of the :class:`~vincio.providers.shadow.CanaryRouter`:
    it ramps ``canary.percent`` of live runs onto the candidate spec (a
    deterministic accumulator, like the provider canary), scores each arm online
    with ``score_fn``, and — once ``min_samples`` candidate observations land —
    decides with the same no-regression verdict: promote, or freeze + roll back if
    the candidate regresses beyond ``regression_threshold``. The user always gets a
    real answer (baseline or candidate); the canary only governs whether the
    candidate is promoted to the live default. Observations are processed
    sequentially (feed it a sampled live stream), so the transient per-run prompt
    swap is race-free.
    """

    def __init__(
        self,
        app: ContextApp,
        candidate: Any,
        *,
        score_fn: Callable[[Any], float],
        canary: CanarySpec | None = None,
        registry: PromptRegistry | None = None,
        prompt_name: str | None = None,
    ) -> None:
        from ..prompts.registry import PromptRegistry

        self.app = app
        self.spec = getattr(candidate, "spec", candidate)
        self.compiler_options = getattr(candidate, "compiler_options", None)
        self.score_fn = score_fn
        self.canary = canary or CanarySpec()
        self.registry = registry or getattr(app, "prompt_registry", None) or PromptRegistry()
        self.prompt_name = prompt_name or app.prompt_spec.name
        self._accum = 0.0
        self._base: list[float] = []
        self._cand: list[float] = []
        self.frozen = False  # stopped ramping to the candidate (post-rollback)
        self.rolled_back = False
        self.calls = 0

    def _route_to_candidate(self) -> bool:
        if self.frozen or self.rolled_back:
            return False
        self._accum += self.canary.percent
        if self._accum >= 100.0:
            self._accum -= 100.0
            return True
        return False

    async def aobserve(self, user_input: Any, **run_kwargs: Any) -> Any:
        """Serve one live run through the chosen arm, score it, and update state.

        Returns the `RunResult` (the caller still gets a real answer)."""
        self.calls += 1
        to_candidate = self._route_to_candidate()
        original_spec = self.app.prompt_spec
        original_options = self.app.prompt_compiler.options
        if to_candidate:
            self.app.prompt_spec = self.spec
            if self.compiler_options is not None:
                self.app.prompt_compiler.options = self.compiler_options
        try:
            result = await self.app.arun(user_input, **run_kwargs)
        finally:
            self.app.prompt_spec = original_spec
            self.app.prompt_compiler.options = original_options
        score = float(self.score_fn(result))
        (self._cand if to_candidate else self._base).append(score)
        self._maybe_rollback()
        return result

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _maybe_rollback(self) -> None:
        if self.rolled_back or len(self._cand) < self.canary.min_samples:
            return
        if self._mean(self._cand) < self._mean(self._base) - self.canary.regression_threshold:
            self.rolled_back = True
            self.frozen = True
            self.app.events.emit(
                "canary.rollback",
                {
                    "prompt": self.prompt_name,
                    "candidate_mean": round(self._mean(self._cand), 6),
                    "baseline_mean": round(self._mean(self._base), 6),
                },
            )

    def verdict(self) -> CanaryVerdict:
        baseline = self._mean(self._base)
        candidate = self._mean(self._cand)
        delta = candidate - baseline
        samples = len(self._cand)
        if samples < self.canary.min_samples:
            passed, reason = (
                False,
                f"insufficient live canary samples ({samples} < {self.canary.min_samples})",
            )
        elif self.rolled_back:
            passed, reason = False, "candidate regressed on live traffic; rolled back"
        else:
            passed = delta >= -self.canary.regression_threshold
            reason = (
                f"{self.canary.metric} held on live traffic (Δ={delta:+.4f}, n={samples})"
                if passed
                else f"{self.canary.metric} regressed on live traffic (Δ={delta:+.4f})"
            )
        return CanaryVerdict(
            passed=passed, metric=self.canary.metric, baseline=round(baseline, 6),
            candidate=round(candidate, 6), delta=round(delta, 6), samples=samples, reason=reason,
        )

    async def afinalize(
        self, *, tag: str = "production", rollback_on_fail: bool = True, dry_run: bool = False
    ) -> DeployResult:
        """Decide the deploy from the live verdict — promote or refuse/roll back."""
        return _finalize_deploy(
            self.app, self.spec, self.compiler_options, self.verdict(),
            registry=self.registry, prompt_name=self.prompt_name, tag=tag,
            rollback_on_fail=rollback_on_fail, dry_run=dry_run, live=True,
        )


async def deploy_candidate(
    app: ContextApp,
    candidate: Any,
    *,
    dataset: Dataset | None = None,
    live_inputs: list[Any] | None = None,
    score_fn: Callable[[Any], float] | None = None,
    canary: CanarySpec | None = None,
    metrics: list[str] | None = None,
    gates: dict[str, str] | None = None,
    tag: str = "production",
    prompt_name: str | None = None,
    registry: PromptRegistry | None = None,
    rollback_on_fail: bool = True,
    dry_run: bool = False,
) -> DeployResult:
    """Canary-gate a prompt/policy candidate and deploy it only if it clears.

    The candidate may be a ``PromptSpec`` or any object carrying ``.spec`` (a
    ``PromptVariant``). Two qualification modes:

    * **Offline gated comparison** (``dataset=``) — the candidate is evaluated
      against the current live prompt on the canary ``dataset`` (this is the
      default).
    * **Live-traffic canary** (``live_inputs=`` + ``score_fn=``) — ``percent`` of
      the supplied live runs ramp onto the candidate and each arm is scored online
      with ``score_fn(RunResult) -> float``; the same no-regression verdict
      decides, with auto-rollback if the candidate regresses mid-ramp (see
      :class:`LiveCanary`).

    On a pass the candidate is pushed to the prompt registry, tagged, applied
    live, and audited (``deploy``); on a fail it is refused and — when
    ``rollback_on_fail`` — the live prompt is rolled back to the last known-good
    registry version. This is the canary-driven promotion surface for prompt and
    policy candidates, in both its offline and live forms.
    """
    from ..prompts.registry import PromptRegistry

    canary = canary or CanarySpec()
    prompt_name = prompt_name or app.prompt_spec.name
    registry = registry or getattr(app, "prompt_registry", None) or PromptRegistry()

    # Live-traffic canary path: ramp, score online, then decide.
    if live_inputs is not None:
        if score_fn is None:
            raise ValueError("live-canary deploy requires score_fn(RunResult) -> float")
        live = LiveCanary(
            app, candidate, score_fn=score_fn, canary=canary,
            registry=registry, prompt_name=prompt_name,
        )
        for user_input in live_inputs:
            await live.aobserve(user_input)
        return await live.afinalize(tag=tag, rollback_on_fail=rollback_on_fail, dry_run=dry_run)

    if dataset is None:
        raise ValueError(
            "deploy_candidate requires a dataset (offline canary) or "
            "live_inputs + score_fn (live canary)"
        )

    metrics = metrics or [canary.metric]
    spec = getattr(candidate, "spec", candidate)
    compiler_options = getattr(candidate, "compiler_options", None)

    baseline_report = await _evaluate_spec(
        app, app.prompt_spec, app.prompt_compiler.options, dataset, metrics
    )
    candidate_report = await _evaluate_spec(app, spec, compiler_options, dataset, metrics)
    verdict = _canary_verdict(baseline_report, candidate_report, canary)

    # Safety/gate overlay: a failing gate blocks the deploy regardless of metric.
    if verdict.passed and gates:
        from ..evals.reports import evaluate_gates

        outcomes = evaluate_gates(candidate_report, gates)
        failed = [k for k, v in outcomes.items() if not v["passed"]]
        if failed:
            verdict.passed = False
            verdict.reason = f"deploy gates failed: {failed}"

    return _finalize_deploy(
        app, spec, compiler_options, verdict, registry=registry, prompt_name=prompt_name,
        tag=tag, candidate_report=candidate_report, dataset_name=dataset.name,
        rollback_on_fail=rollback_on_fail, dry_run=dry_run, live=False,
    )


# ---------------------------------------------------------------------------
# The unified controller
# ---------------------------------------------------------------------------


class SelfImprovementController:
    """Drive a :class:`SelfImprovementPolicy` as one streaming controller.

    Composes the existing organs — the experiment proposer, the gated
    improvement loop, the continuous-improvement controller, and canary-gated
    deployment — under one budget, one audit trail, and one event stream. It
    never reaches inside any of them; promotion still passes their gates.
    """

    def __init__(
        self,
        app: ContextApp,
        policy: SelfImprovementPolicy | None = None,
        *,
        dataset: Dataset | None = None,
        golden: GoldenRegressionSuite | None = None,
        registry: PromptRegistry | None = None,
        prompt_name: str | None = None,
    ) -> None:
        from ..prompts.registry import PromptRegistry

        self.app = app
        self.policy = policy or SelfImprovementPolicy()
        self.dataset = dataset
        self.golden = golden
        self.registry = registry or getattr(app, "prompt_registry", None) or PromptRegistry()
        self.prompt_name = prompt_name or app.prompt_spec.name
        self._budget_spent = 0.0
        self.events: list[SelfImprovementEvent] = []
        self.proposer = ExperimentProposer(
            app, eval_budget=int(self.policy.eval_budget), gates=self.policy.gates,
            golden_suite=golden,
        )
        # The online drift→action controller is the policy's "online" arm; it is
        # only attached to the bus when the policy opts in.
        self._controller: Any = None

    # -- live attachment (scheduling) ----------------------------------------

    def attach(self) -> SelfImprovementController:
        """Subscribe the online arm to the event bus (drift/online_eval)."""
        if not self.policy.online:
            return self
        from .controller import ContinuousImprovementController

        self._controller = ContinuousImprovementController(
            self.app,
            metrics=self.policy.metrics,
            golden=self.dataset,
            registry=self.registry,
            prompt_name=self.prompt_name,
            sustain=self.policy.sustain,
            cooldown_s=self.policy.cooldown_s,
            eval_budget=self.policy.eval_budget,
            gates=self.policy.gates,
        )
        self._controller.attach()
        return self

    def detach(self) -> None:
        if self._controller is not None:
            self._controller.detach()

    # -- the streaming cycle -------------------------------------------------

    async def astream(self) -> AsyncIterator[SelfImprovementEvent]:
        """Run one full self-improvement cycle, yielding each phase as it lands.

        Sequence: observe → (proposal) → (meta) → (label) → reeval → canary →
        promote / rollback. Bounded by the policy's eval budget; emits
        ``exhausted`` and stops when the budget runs out."""
        policy = self.policy
        yield self._emit(SelfImprovementEvent(phase="observe", reason="cycle started"))

        # 1. Autonomous proposal: rank the weakest metric and what would move it.
        proposal: ExperimentProposal | None = None
        if policy.propose:
            ranked = self.proposer.propose()
            proposal = ranked[0] if ranked else None
            yield self._emit(
                SelfImprovementEvent(
                    phase="proposal",
                    metric=proposal.target_metric if proposal else "",
                    action="proposed" if proposal else "none",
                    reason=proposal.rationale if proposal else "no weakness above target",
                    proposal=proposal.model_dump() if proposal else None,
                )
            )

        if self.dataset is None:
            yield self._emit(
                SelfImprovementEvent(phase="reeval", action="skipped", reason="no dataset bound")
            )
            return

        # 2. Meta-optimization: choose strategy + budget (successive-halving) and
        #    learned fitness weights.
        strategy = "reflective"
        budget = 8
        weights: FitnessWeights | None = None
        if policy.meta is not None and policy.meta.enabled:
            strategy, budget, weights = await self._meta_select()
            yield self._emit(
                SelfImprovementEvent(
                    phase="meta", action="selected", chosen_strategy=strategy,
                    chosen_budget=budget,
                    reason=f"successive-halving chose {strategy}@{budget}",
                    details={"learned_weights": weights.model_dump() if weights else None},
                )
            )

        # 3. Active-learning label acquisition: queue the most-uncertain cases.
        if policy.active_learning:
            base_report = await _evaluate_spec(
                self.app, self.app.prompt_spec, self.app.prompt_compiler.options,
                self.dataset, policy.metrics,
            )
            self._budget_spent += float(len(self.dataset))
            case_ids = select_for_labeling(
                base_report, metric=policy.metrics[0], budget=policy.label_budget
            )
            yield self._emit(
                SelfImprovementEvent(
                    phase="label", action="queued", label_case_ids=case_ids,
                    reason=f"queued {len(case_ids)} most-uncertain cases for labeling",
                )
            )

        if self._budget_spent >= policy.eval_budget:
            yield self._emit(
                SelfImprovementEvent(phase="exhausted", reason="eval budget exhausted before re-eval")
            )
            return

        # 4. Gated re-optimization through the existing improvement loop.
        optimizer, inner_strategy = _loop_optimizer(strategy)
        loop = ImprovementLoop(
            self.app, registry=self.registry, gates=policy.gates,
            prompt_name=self.prompt_name, weights=weights, golden_suite=self.golden,
            optimizer=optimizer, strategy=inner_strategy,
        )
        loop_result = await loop.arun(
            dataset=self.dataset, max_variants=min(budget, 8), subset_size=4,
            promote_tag=policy.promote_tag, dry_run=True,  # decide here, deploy below
        )
        self._budget_spent += float(budget)
        candidate = (
            loop_result.optimization.best.payload
            if loop_result.optimization and loop_result.optimization.best
            else None
        )
        yield self._emit(
            SelfImprovementEvent(
                phase="reeval", action="optimized" if candidate else "no_candidate",
                reason=loop_result.reason,
                details={"baseline_fitness": (
                    loop_result.optimization.baseline_fitness if loop_result.optimization else None
                )},
            )
        )

        if candidate is None:
            return

        # 5. Canary-gate the winner, then deploy or roll back.
        if policy.canary is not None:
            deploy_result = await deploy_candidate(
                self.app, candidate, dataset=self.dataset, canary=policy.canary,
                metrics=policy.metrics, gates=policy.gates, tag=policy.promote_tag,
                prompt_name=self.prompt_name, registry=self.registry, dry_run=policy.dry_run,
            )
            self._budget_spent += float(len(self.dataset)) * 2
            if deploy_result.deployed:
                yield self._emit(
                    SelfImprovementEvent(
                        phase="promote", action="deployed", promoted_ref=deploy_result.ref,
                        verdict=deploy_result.verdict, reason=deploy_result.reason,
                    )
                )
            else:
                yield self._emit(
                    SelfImprovementEvent(
                        phase="rollback", action="refused",
                        rolled_back_to=deploy_result.rolled_back_to,
                        verdict=deploy_result.verdict, reason=deploy_result.reason,
                    )
                )

    async def step(self) -> list[SelfImprovementEvent]:
        """Run one cycle and return the events it produced (non-streaming)."""
        return [event async for event in self.astream()]

    def run(self) -> list[SelfImprovementEvent]:
        """Sync wrapper over :meth:`step`."""
        from ..providers.base import run_sync

        return run_sync(self.step())

    # -- internals -----------------------------------------------------------

    async def _meta_select(self) -> tuple[str, int, FitnessWeights | None]:
        """Pick (strategy, budget) by successive-halving and learn weights."""
        assert self.dataset is not None  # noqa: S101 - the re-eval phase returns early when no dataset is bound
        meta = self.policy.meta
        assert meta is not None  # noqa: S101 - _meta_select runs only when policy.meta is enabled
        configs = [(s, b) for s in meta.strategies for b in meta.budgets]
        minibatch = self.dataset.sample(min(4, len(self.dataset)))

        async def score(config: tuple[str, int]) -> float:
            strat, _budget = config
            optimizer, inner_strategy = _loop_optimizer(strat)
            loop = ImprovementLoop(
                self.app, registry=self.registry, gates=self.policy.gates,
                prompt_name=self.prompt_name, optimizer=optimizer, strategy=inner_strategy,
            )
            res = await loop.arun(
                dataset=minibatch, max_variants=2, subset_size=2, dry_run=True
            )
            self._budget_spent += 2.0
            opt = res.optimization
            if opt and opt.best and opt.best.full_fitness is not None:
                return opt.best.full_fitness
            return opt.baseline_fitness if opt else 0.0

        (strategy, budget), _history = await successive_halving(
            configs, score, rounds=meta.halving_rounds
        )
        weights: FitnessWeights | None = None
        if meta.learn_weights:
            base_report = await _evaluate_spec(
                self.app, self.app.prompt_spec, self.app.prompt_compiler.options,
                minibatch, self.policy.metrics,
            )
            weights = learn_fitness_weights(base_report)
        return strategy, budget, weights

    def _emit(self, event: SelfImprovementEvent) -> SelfImprovementEvent:
        event.budget_spent = round(self._budget_spent, 4)
        event.budget_remaining = round(max(0.0, self.policy.eval_budget - self._budget_spent), 4)
        self.events.append(event)
        self.app.audit.record(
            "self_improvement",
            decision="allow" if event.action not in ("none", "skipped", "refused") else "skip",
            resource=self.prompt_name,
            details={
                "phase": event.phase, "action": event.action, "metric": event.metric,
                "reason": event.reason, "promoted_ref": event.promoted_ref,
                "rolled_back_to": event.rolled_back_to, "budget_spent": event.budget_spent,
            },
        )
        self.app.events.emit(f"self_improvement.{event.phase}", event.model_dump())
        return event
