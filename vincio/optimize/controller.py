"""The continual loop: an online improvement controller (1.10).

Vincio already *observes* online — the :class:`~vincio.evals.online.OnlineEvaluator`
scores sampled runs and the :class:`~vincio.evals.drift.DriftMonitor` raises
``drift.detected`` — but until now nothing acted on it. The
:class:`ContinuousImprovementController` closes that gap: it subscribes to
``drift.detected`` and ``eval.online`` on the app's event bus, streams online
scores into the monitor's CUSUM changepoint detector, and turns a *sustained*
signal into exactly one of three **gated** actions:

1. **targeted re-eval** — replay a held-out golden set to confirm the regression
   is real before spending anything bigger (drift can be noise);
2. **re-optimization** — a fresh, gated :class:`~vincio.optimize.loop.ImprovementLoop`
   run that only promotes a winner if it clears the same significance + safety
   gates as any other promotion;
3. **rollback** — revert the prompt registry to the last known-good version when
   the regression is severe (a safety metric, or a confirmed drop that
   re-optimization can't recover).

Every trigger is debounced (a per-metric cooldown) and bounded (a global eval
budget), and every trigger, debounce, decision, and rollback lands on the
hash-chained audit log and one trace. State (the budget already spent, per-metric
cooldown timers, sustain counts) persists to the shared store so the controller
is restart-safe. It is pure composition of organs Vincio already ships — it never
reaches inside the optimizer.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..core.utils import utcnow
from ..evals.datasets import Dataset
from ..evals.drift import DriftMonitor

if TYPE_CHECKING:
    from ..core.app import ContextApp
    from ..core.events import Event
    from ..prompts.registry import PromptRegistry

__all__ = ["ControllerDecision", "ContinuousImprovementController"]

# Metrics where *lower is better*: a regression is an *upward* changepoint, and
# the quality floor is an upper bound rather than a lower bound.
_LOWER_IS_BETTER = {"cost", "latency", "toxicity", "bias", "hallucination", "retry_rate"}
# Metrics whose regression is severe enough to roll back without re-optimizing.
_SAFETY_METRICS = {"safety", "toxicity", "bias", "hallucination"}


ActionName = Literal[
    "observing", "debounced", "budget_exhausted", "reeval_clear",
    "reoptimized", "reoptimize_failed", "rolled_back", "no_action",
]


class ControllerDecision(BaseModel):
    """The record of one controller evaluation — stamped on the audit chain."""

    trigger: str = "drift"  # drift | online_eval | manual
    metric: str = ""
    method: str = ""  # the drift method that fired (score | cusum | ks | …)
    action: ActionName = "observing"
    reason: str = ""
    confirmed: bool | None = None  # re-eval verdict, when one ran
    promoted_ref: str | None = None  # registry ref promoted by re-optimization
    rolled_back_to: str | None = None  # registry ref restored by a rollback
    sustain_count: int = 0
    budget_spent: float = 0.0
    budget_remaining: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class ContinuousImprovementController:
    """Drive gated re-optimization / re-eval / rollback from live signals."""

    def __init__(
        self,
        app: ContextApp,
        *,
        metrics: list[str] | None = None,
        golden: Dataset | None = None,
        registry: PromptRegistry | None = None,
        prompt_name: str | None = None,
        monitor: DriftMonitor | None = None,
        sustain: int = 2,
        cooldown_s: float = 300.0,
        eval_budget: float = 48.0,
        quality_floor: dict[str, float] | None = None,
        reoptimize: bool = True,
        gates: dict[str, str] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.app = app
        self.metrics = metrics or [e.name for e in app.online_evaluators] or ["semantic_similarity"]
        self.golden = golden
        self.prompt_name = prompt_name or app.prompt_spec.name
        self.sustain = max(1, sustain)
        self.cooldown_s = cooldown_s
        self.eval_budget = eval_budget
        self.quality_floor = quality_floor or {}
        self.reoptimize = reoptimize
        self.gates = gates
        self._clock = clock or time.time
        if registry is not None:
            self.registry = registry
        else:
            from ..prompts.registry import PromptRegistry

            self.registry = PromptRegistry()
        self.monitor = monitor or DriftMonitor(bus=app.events, store=app.store, app_name=app.name)
        # Per-metric runtime state (restored from the store on construction).
        self._sustain: dict[str, int] = {}
        self._last_action: dict[str, float] = {}
        self._budget_spent = 0.0
        self.decisions: list[ControllerDecision] = []
        self._unsubscribe: list[Callable[[], None]] = []
        self._state_id = f"{app.name}:continuous_improvement"
        self.load_state()

    # -- baselines -----------------------------------------------------------

    def set_baseline(self, metric: str, values: list[float]) -> ContinuousImprovementController:
        """Anchor a metric's drift baseline (mean + CUSUM target) from samples."""
        self.monitor.set_score_baseline(metric, values)
        return self

    def seed_from_online(self, *, window: int = 50) -> ContinuousImprovementController:
        """Set each watched metric's baseline from its recent online series."""
        for evaluator in self.app.online_evaluators:
            if evaluator.name not in self.metrics:
                continue
            series = evaluator.series(limit=window)
            values = [float(r.get("metric_value", 0.0)) for r in series]
            if len(values) >= 2:
                self.set_baseline(evaluator.name, values)
        return self

    # -- lifecycle -----------------------------------------------------------

    def attach(self) -> ContinuousImprovementController:
        """Subscribe to the event bus and start acting on live signals."""
        bus = self.app.events
        self._unsubscribe.append(bus.subscribe("eval.online", self._on_online_eval))
        self._unsubscribe.append(bus.subscribe("drift.detected", self._on_drift))
        return self

    def detach(self) -> None:
        for unsub in self._unsubscribe:
            unsub()
        self._unsubscribe.clear()

    # -- event handlers ------------------------------------------------------

    def _on_online_eval(self, event: Event) -> None:
        """Stream an online score into the monitor's CUSUM detector.

        The monitor raises ``drift.detected`` itself on a changepoint, so the
        action path stays single-sourced through :meth:`_on_drift`.
        """
        metric = str(event.payload.get("metric", ""))
        if metric not in self.metrics:
            return
        value = event.payload.get("value")
        if value is None:
            return
        self.monitor.observe_score(metric, float(value))

    def _on_drift(self, event: Event) -> None:
        report = event.payload
        metric = str(report.get("metric", ""))
        if metric and metric not in self.metrics:
            return
        self.evaluate(metric or (self.metrics[0] if self.metrics else ""), report)

    # -- the gated decision --------------------------------------------------

    def evaluate(self, metric: str, report: dict[str, Any] | None = None) -> ControllerDecision:
        """Process one drift signal for *metric*; debounce, gate, and act.

        Public and synchronous so the loop is unit-testable without a live bus.
        """
        report = report or {}
        method = str(report.get("method", "score"))
        self._sustain[metric] = self._sustain.get(metric, 0) + 1
        sustain = self._sustain[metric]
        decision = ControllerDecision(
            trigger="drift", metric=metric, method=method, sustain_count=sustain,
            budget_spent=self._budget_spent,
            budget_remaining=max(0.0, self.eval_budget - self._budget_spent),
        )

        if sustain < self.sustain:
            decision.action = "observing"
            decision.reason = f"{sustain}/{self.sustain} sustained signals; observing"
            return self._record(decision)

        now = self._clock()
        last = self._last_action.get(metric)
        if last is not None and (now - last) < self.cooldown_s:
            decision.action = "debounced"
            decision.reason = (
                f"within {self.cooldown_s:.0f}s cooldown ({now - last:.0f}s since last action)"
            )
            return self._record(decision)

        if self._budget_spent >= self.eval_budget:
            decision.action = "budget_exhausted"
            decision.reason = f"global eval budget {self.eval_budget:g} spent; refusing to act"
            return self._record(decision)

        # Sustained, off-cooldown, in-budget: act once, then reset + cooldown.
        self._last_action[metric] = now
        self._sustain[metric] = 0
        self.app.events.emit("improvement.triggered", {"metric": metric, "method": method})
        self._act(metric, report, decision)
        return self._record(decision)

    def _act(self, metric: str, report: dict[str, Any], decision: ControllerDecision) -> None:
        severe = metric in _SAFETY_METRICS

        # 1. Confirm with a targeted re-eval on the held-out golden set (cheap).
        if self.golden is not None and len(self.golden) > 0 and not severe:
            confirmed = self._confirm_regression(metric)
            decision.confirmed = confirmed
            if not confirmed:
                decision.action = "reeval_clear"
                decision.reason = f"re-eval of {len(self.golden)} cases cleared {metric}; false alarm"
                return
        elif severe:
            decision.confirmed = True

        # 2. Re-optimize (gated), unless the regression is a safety metric — those
        #    roll back immediately rather than waiting on an optimizer round.
        if self.reoptimize and not severe and self.golden is not None:
            promoted = self._reoptimize(metric, decision)
            if promoted:
                decision.action = "reoptimized"
                return
            decision.action = "reoptimize_failed"
            # fall through to rollback when re-optimization couldn't recover

        # 3. Rollback to the last known-good registry version.
        self._rollback(metric, decision, severe=severe)

    # -- the three actions ---------------------------------------------------

    def _confirm_regression(self, metric: str) -> bool:
        """Replay the golden set; True when *metric* is below its quality floor."""
        if self.golden is None:
            return True
        cost = float(len(self.golden))
        self._budget_spent += cost
        report = self.app.evaluate(self.golden, metrics=[metric])
        summary = report.summary().get(metric, {})
        value = float(summary.get("mean", 0.0))
        floor = self.quality_floor.get(metric)
        if floor is None:
            # No explicit floor: treat the drift signal as authoritative.
            return True
        if metric in _LOWER_IS_BETTER:
            return value > floor
        return value < floor

    def _reoptimize(self, metric: str, decision: ControllerDecision) -> bool:
        from .loop import ImprovementLoop

        budget = self._remaining_rollouts()
        if budget < 2:
            decision.details["reoptimize_skipped"] = "insufficient budget"
            return False
        loop = ImprovementLoop(
            self.app, registry=self.registry, gates=self.gates, prompt_name=self.prompt_name
        )
        self._budget_spent += float(budget)
        result = loop.run(dataset=self.golden, max_variants=min(budget, 8), subset_size=4)
        decision.details["reoptimize_reason"] = result.reason
        if result.promoted and result.promoted_ref:
            decision.promoted_ref = result.promoted_ref
            decision.reason = f"re-optimized {self.prompt_name}: {result.reason}"
            self._reset_baseline(metric)
            return True
        return False

    def _rollback(self, metric: str, decision: ControllerDecision, *, severe: bool) -> None:
        try:
            known_good = self._last_known_good()
        except Exception as exc:  # noqa: BLE001 - missing history is a no-op, not a crash
            decision.action = "no_action"
            decision.reason = f"no registry history to roll back to ({exc})"
            return
        if known_good is None:
            decision.action = "no_action"
            decision.reason = "no earlier known-good version to roll back to"
            return
        new_head = self.registry.rollback(self.prompt_name, to_version=known_good.version)
        self.app.prompt_spec = new_head.spec
        decision.action = "rolled_back"
        decision.rolled_back_to = known_good.ref
        decision.reason = (
            f"{'safety ' if severe else ''}regression in {metric}; "
            f"rolled back to {known_good.ref} (new head {new_head.ref})"
        )
        self._reset_baseline(metric)
        self.app.events.emit(
            "improvement.rolled_back",
            {"prompt": new_head.ref, "restored": known_good.ref, "metric": metric},
        )

    def _last_known_good(self) -> Any:
        """The version tagged ``production`` below the head, else the prior version."""
        versions = self.registry.versions(self.prompt_name)
        if len(versions) < 2:
            return None
        head = versions[-1]
        for version in reversed(versions[:-1]):
            if "production" in version.tags:
                return version
        # No production tag: fall back to the immediately preceding version.
        return versions[-2] if head is not None else None

    # -- helpers -------------------------------------------------------------

    def _remaining_rollouts(self) -> int:
        return int(max(0.0, self.eval_budget - self._budget_spent))

    def _reset_baseline(self, metric: str) -> None:
        """After an action the prompt regime changed; re-anchor the detector."""
        self.monitor.reset_cusum(metric)

    def _record(self, decision: ControllerDecision) -> ControllerDecision:
        decision.budget_spent = self._budget_spent
        decision.budget_remaining = max(0.0, self.eval_budget - self._budget_spent)
        self.decisions.append(decision)
        self.app.audit.record(
            "continuous_improvement",
            decision="allow" if decision.action != "no_action" else "skip",
            details={
                "metric": decision.metric,
                "method": decision.method,
                "action": decision.action,
                "reason": decision.reason,
                "promoted_ref": decision.promoted_ref,
                "rolled_back_to": decision.rolled_back_to,
                "budget_spent": decision.budget_spent,
            },
        )
        self.app.events.emit("improvement.decision", decision.model_dump())
        self.save_state()
        return decision

    # -- persisted state -----------------------------------------------------

    def save_state(self) -> None:
        if self.app.store is None:
            return
        self.app.store.save(
            "controller_state",
            {
                "id": self._state_id,
                "app_id": self.app.name,
                "budget_spent": self._budget_spent,
                "sustain": dict(self._sustain),
                "last_action": dict(self._last_action),
                "decisions": len(self.decisions),
                "updated_at": utcnow().isoformat(),
            },
        )

    def load_state(self) -> None:
        if self.app.store is None:
            return
        row = self.app.store.get("controller_state", self._state_id)
        if row is None:
            return
        self._budget_spent = float(row.get("budget_spent", 0.0))
        self._sustain = {k: int(v) for k, v in (row.get("sustain") or {}).items()}
        self._last_action = {k: float(v) for k, v in (row.get("last_action") or {}).items()}
