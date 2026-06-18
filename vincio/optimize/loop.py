"""The closed improvement loop: trace → dataset → eval → optimize → promote.

One continuous, reproducible cycle, entirely in the library: capture the
traces production runs already write, curate them into an eval dataset
(feedback-filtered), evaluate the current prompt as the baseline, run the
gated prompt optimizer, and promote the winner into the prompt registry —
tagged, eval-linked, applied to the app, and audited. Every stage records
what it did, so a loop run can be replayed and its promotion decision
re-derived from the same dataset and reports.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field

from ..evals.datasets import Dataset, dataset_from_traces
from ..evals.experiments import ExperimentTracker
from ..evals.runners import EvalRunner
from ..prompts.registry import PromptRegistry
from ..providers.base import run_sync
from .prompt_search import PromptOptimizer
from .search import FitnessWeights, OptimizationResult

if TYPE_CHECKING:
    from ..core.app import ContextApp
    from ..evals.datasets import GoldenRegressionSuite
    from ..observability.spans import Trace

__all__ = [
    "LoopResult",
    "ImprovementLoop",
    "DEFAULT_LOOP_METRICS",
    "ExperimentProposal",
    "ExperimentProposer",
]

DEFAULT_LOOP_METRICS = [
    "lexical_overlap",
    "groundedness",
    "schema_validity",
    "cost",
    "latency",
]


class LoopResult(BaseModel):
    """Outcome of one improvement-loop cycle, with full provenance."""

    experiment: str
    dataset_name: str = ""
    dataset_size: int = 0
    dataset_fingerprint: str = ""  # stable hash of the curated case ids
    optimization: OptimizationResult | None = None
    promoted: bool = False
    promoted_ref: str | None = None  # prompt-registry ref of the promoted version
    reason: str = ""
    steps: list[dict[str, Any]] = Field(default_factory=list)


class ImprovementLoop:
    """Runs the trace → dataset → eval → optimize → promote cycle on an app.

    The loop reuses the pieces that already exist — the tracer's exporter,
    ``dataset_from_traces``, the eval runner, the gated evolution loop, the
    prompt registry, and the experiment tracker — and wires them into one
    call. Promotion is conservative by construction: it inherits the
    optimizer's safety rules (no safety/schema regression, cost ceilings,
    eval gates) and does nothing on ``dry_run``.
    """

    def __init__(
        self,
        app: ContextApp,
        *,
        registry: PromptRegistry | None = None,
        tracker: ExperimentTracker | None = None,
        metrics: list[str] | None = None,
        weights: FitnessWeights | None = None,
        gates: dict[str, str] | None = None,
        max_cost_per_case: float | None = None,
        experiment: str = "improvement_loop",
        prompt_name: str | None = None,
        concurrency: int = 4,
        optimizer: str = "evolution",
        strategy: str = "reflective",
        reflector: str = "heuristic",
        golden_suite: GoldenRegressionSuite | None = None,
    ) -> None:
        self.app = app
        self.registry = registry or PromptRegistry()
        self.tracker = tracker or ExperimentTracker(app.store)
        self.metrics = metrics or (app.evaluators or DEFAULT_LOOP_METRICS)
        self.weights = weights
        self.gates = gates
        self.max_cost_per_case = max_cost_per_case
        self.experiment = experiment
        self.prompt_name = prompt_name or app.prompt_spec.name
        self.concurrency = concurrency
        # "evolution" (blind variant search) or "reflective" (GEPA-style
        # failure-driven search). The reflective path proposes targeted edits
        # from the eval report's failures and evolves a Pareto frontier; both
        # promote through the identical gated path below.
        self.optimizer = optimizer
        self.strategy = strategy
        # "heuristic" (deterministic floor) or "llm" (provider-backed GEPA
        # reflector wired to the app's own provider, with a heuristic fallback).
        self.reflector = reflector
        # The held-out, growing non-regression guard: promotions are
        # replayed against it, and grown by it on success.
        self.golden_suite = golden_suite

    # -- stage 1: capture --------------------------------------------------------

    def capture(self, *, limit: int = 500) -> list[Trace]:
        """Load captured traces from the app's exporter."""
        exporter = self.app.tracer.exporter
        if hasattr(exporter, "load_all"):
            return exporter.load_all(limit=limit)
        if hasattr(exporter, "traces"):
            return list(exporter.traces)[-limit:]
        return []

    # -- stage 2: curate ---------------------------------------------------------

    def curate(
        self,
        traces: list[Trace],
        *,
        min_feedback_score: float | None = None,
        only_ok: bool = True,
        name: str | None = None,
    ) -> Dataset:
        return dataset_from_traces(
            traces,
            name=name or f"{self.experiment}_dataset",
            min_feedback_score=min_feedback_score,
            only_ok=only_ok,
        )

    # -- stages 3-5: eval, optimize, promote ---------------------------------------

    async def arun(
        self,
        *,
        dataset: Dataset | None = None,
        traces: list[Trace] | None = None,
        min_feedback_score: float | None = None,
        only_ok: bool = True,
        max_variants: int = 8,
        subset_size: int = 8,
        top_n: int = 3,
        promote_tag: str = "production",
        dry_run: bool = False,
    ) -> LoopResult:
        app = self.app
        result = LoopResult(experiment=self.experiment)

        # 1-2. capture + curate (skipped when a dataset is supplied).
        if dataset is None:
            traces = traces if traces is not None else self.capture()
            result.steps.append({"stage": "capture", "traces": len(traces)})
            dataset = self.curate(
                traces, min_feedback_score=min_feedback_score, only_ok=only_ok
            )
            result.steps.append(
                {
                    "stage": "curate",
                    "cases": len(dataset),
                    "min_feedback_score": min_feedback_score,
                }
            )
        result.dataset_name = dataset.name
        result.dataset_size = len(dataset)
        case_ids = ",".join(sorted(case.id for case in dataset.cases))
        result.dataset_fingerprint = hashlib.sha256(case_ids.encode("utf-8")).hexdigest()[:16]

        # 3-4. evaluate baseline + candidates through the gated optimizer.
        # Candidate evaluations are memory-write-free: an eval run must never
        # pollute user memory or hand later candidates different recall state
        # than earlier ones saw — comparisons stay apples to apples.
        async def evaluate_variant(variant, ds):
            original_spec = app.prompt_spec
            original_options = app.prompt_compiler.options
            original_write_back = app.config.memory.write_back
            app.prompt_spec = variant.spec
            app.prompt_compiler.options = variant.compiler_options
            app.config.memory.write_back = []
            try:
                runner = EvalRunner(app, metrics=self.metrics, concurrency=self.concurrency)
                return await runner.arun(ds, name=variant.name)
            finally:
                app.prompt_spec = original_spec
                app.prompt_compiler.options = original_options
                app.config.memory.write_back = original_write_back

        if self.optimizer == "reflective":
            from .reflective import LLMReflector, ReflectiveOptimizer

            reflector_impl = None
            if self.reflector == "llm":
                reflector_impl = LLMReflector(app._base_provider(), app.model)
            reflective = ReflectiveOptimizer(
                evaluate_variant,
                weights=self.weights,
                gates=self.gates,
                max_cost_per_case=self.max_cost_per_case,
                reflector=reflector_impl,
            )
            optimization: OptimizationResult = await reflective.optimize(
                app.prompt_spec,
                dataset,
                strategy=cast('Literal["reflective", "mipro"]', self.strategy),
                budget=max_variants,
                minibatch_size=subset_size,
            )
        else:
            optimizer = PromptOptimizer(
                evaluate_variant,
                weights=self.weights,
                gates=self.gates,
                max_cost_per_case=self.max_cost_per_case,
            )
            optimization = await optimizer.optimize(
                app.prompt_spec,
                dataset,
                max_variants=max_variants,
                subset_size=subset_size,
                top_n=top_n,
            )
        result.optimization = optimization
        result.steps.append(
            {
                "stage": "optimize",
                "baseline_fitness": optimization.baseline_fitness,
                "candidates": len(optimization.candidates),
                "promoted": optimization.promoted,
                "reason": optimization.reason,
            }
        )
        if optimization.reason.startswith("dataset too small"):
            result.reason = optimization.reason
            return result

        # Log baseline and best to the experiment tracker (same store as runs).
        if optimization.baseline is not None and optimization.baseline.full_report is not None:
            self.tracker.log(
                self.experiment,
                optimization.baseline.full_report,
                variant="baseline",
                params=optimization.baseline.params,
            )
        best = optimization.best
        if best is not None and best.full_report is not None:
            self.tracker.log(
                self.experiment, best.full_report, variant=best.name, params=best.params
            )

        if not optimization.promoted or best is None:
            result.reason = optimization.reason or "no candidate promoted"
            return result

        # 5. promote: registry push + tag + eval link, apply, audit, event.
        winner = best.payload  # PromptVariant

        # Non-regression guard: replay the winner against the held-out,
        # growing golden suite. A sequential auto-promotion can never silently
        # undo a prior fix — regressing any recorded case blocks the promotion.
        if self.golden_suite is not None and len(self.golden_suite) > 0:
            suite_report = await evaluate_variant(winner, self.golden_suite.as_dataset())
            gate = self.golden_suite.gate(suite_report)
            result.steps.append(
                {"stage": "golden_gate", "passed": gate.passed,
                 "regressed": gate.regressed, "missing": gate.missing}
            )
            if not gate.passed:
                result.promoted = False
                result.reason = (
                    f"blocked by golden regression suite: {len(gate.regressed)} regressed, "
                    f"{len(gate.missing)} unverified ({optimization.reason})"
                )
                app.audit.record(
                    "loop_promotion_blocked",
                    decision="deny",
                    details={"experiment": self.experiment, "regressed": gate.regressed,
                             "missing": gate.missing, "candidate": best.name},
                )
                return result

        if dry_run:
            result.promoted = False
            result.reason = f"dry run: would promote {best.name} ({optimization.reason})"
            result.steps.append({"stage": "promote", "dry_run": True, "winner": best.name})
            return result

        self.registry.push(app.prompt_spec, name=self.prompt_name, message="loop baseline")
        version = self.registry.push(
            winner.spec,
            name=self.prompt_name,
            tags=[promote_tag],
            message=f"promoted by improvement loop: {optimization.reason}",
        )
        if best.full_report is not None:
            self.registry.link_eval(
                self.prompt_name, version.version, best.full_report, dataset=dataset.name
            )
        app.prompt_spec = winner.spec
        app.prompt_compiler.options = winner.compiler_options
        app.audit.record(
            "loop_promotion",
            decision="allow",
            details={
                "experiment": self.experiment,
                "prompt": version.ref,
                "tag": promote_tag,
                "dataset": dataset.name,
                "dataset_fingerprint": result.dataset_fingerprint,
                "baseline_fitness": optimization.baseline_fitness,
                "fitness": best.full_fitness,
                "params": best.params,
                # Statistical backing: the promotion is defensible at a
                # confidence level, not a point estimate.
                "significance": optimization.significance,
                "warnings": optimization.warnings,
            },
        )
        app.events.emit(
            "loop.promoted",
            {"prompt": version.ref, "experiment": self.experiment, "tag": promote_tag},
        )
        result.promoted = True
        result.promoted_ref = version.ref
        result.reason = optimization.reason
        result.steps.append(
            {"stage": "promote", "ref": version.ref, "tag": promote_tag, "dry_run": False}
        )
        # Grow the held-out suite with the cases this promotion newly passes, so
        # every future promotion must keep clearing them.
        if self.golden_suite is not None and best.full_report is not None:
            metric = self.weights.accuracy_metric if self.weights else "lexical_overlap"
            self.golden_suite.add_from_report(
                best.full_report, dataset, fixed_by=version.ref, guard_metric=metric
            )
        return result

    def run(self, **kwargs: Any) -> LoopResult:
        return run_sync(self.arun(**kwargs))


# ---------------------------------------------------------------------------
# Autonomous experiment proposer
# ---------------------------------------------------------------------------

# Higher-is-better quality targets a healthy app should clear. Cost/latency are
# ceilings (lower is better) and only ranked when supplied in ``targets``.
_DEFAULT_TARGETS = {
    "lexical_overlap": 0.8,
    "groundedness": 0.85,
    "faithfulness": 0.85,
    "answer_relevance": 0.8,
    "schema_validity": 0.98,
    "safety": 0.99,
}
_LOWER_IS_BETTER = {"cost", "latency", "toxicity", "bias", "hallucination", "retry_rate"}

# Which experiment kinds can move which weakness, cheapest-leverage first.
_METRIC_EXPERIMENTS: dict[str, list[str]] = {
    "groundedness": ["retrieval", "prompt"],
    "faithfulness": ["retrieval", "prompt"],
    "answer_relevance": ["prompt", "retrieval"],
    "lexical_overlap": ["prompt"],
    "schema_validity": ["prompt"],
    "safety": ["prompt"],
    "cost": ["routing", "distillation", "budget"],
    "latency": ["routing", "budget"],
}


class ExperimentProposal(BaseModel):
    """One ranked, budgeted self-improvement experiment the system could run."""

    kind: Literal["prompt", "retrieval", "budget", "routing", "distillation"]
    target_metric: str
    weakness: float  # how far below target (the ROI proxy), higher = weaker
    current: float
    target: float
    rationale: str = ""
    eval_budget: int = 0
    drift: bool = False  # whether a live drift signal reinforces this weakness


class ExperimentProposer:
    """Rank where the system is weakest and schedule the highest-ROI experiment.

    Reads the app's online eval series and recorded drift, scores each metric's
    *weakness* against a target, maps the weakest metrics to the experiment kinds
    that can move them (prompt vs. retrieval vs. budget vs. routing vs.
    distillation), and allocates a global eval budget across the ranked
    candidates. :meth:`run_next` executes the top proposal — the prompt
    experiment runs the gated :class:`ImprovementLoop` end to end; the others are
    scheduled and recorded against the organ that owns them — with every decision
    on the audit chain.
    """

    def __init__(
        self,
        app: ContextApp,
        *,
        targets: dict[str, float] | None = None,
        eval_budget: int = 24,
        golden_suite: Any | None = None,
        gates: dict[str, str] | None = None,
    ) -> None:
        self.app = app
        self.targets = {**_DEFAULT_TARGETS, **(targets or {})}
        self.eval_budget = eval_budget
        self.golden_suite = golden_suite
        self.gates = gates

    def signals(self, *, window: int = 100) -> dict[str, float]:
        """Mean of each online metric series the app records."""
        out: dict[str, float] = {}
        for evaluator in self.app.online_evaluators:
            series = evaluator.series(limit=window)
            values = [float(r.get("metric_value", 0.0)) for r in series]
            if values:
                out[evaluator.name] = sum(values) / len(values)
        return out

    def drifted_metrics(self) -> set[str]:
        """Metrics with a recorded drift baseline (a live weakness signal)."""
        store = self.app.store
        if store is None:
            return set()
        rows = store.query("drift_baselines", where={"app_id": self.app.name})
        return {str(r.get("metric")) for r in rows if r.get("metric")}

    def rank(
        self, signals: dict[str, float], *, drift: set[str] | None = None
    ) -> list[ExperimentProposal]:
        drift = drift or set()
        scored: list[ExperimentProposal] = []
        for metric, value in signals.items():
            target = self.targets.get(metric)
            if target is None:
                continue
            if metric in _LOWER_IS_BETTER:
                weakness = max(0.0, value - target)
            else:
                weakness = max(0.0, target - value)
            if weakness <= 0.0:
                continue
            # A live drift signal on the metric boosts its priority.
            boosted = weakness * (1.5 if metric in drift else 1.0)
            kind = cast(
                'Literal["prompt", "retrieval", "budget", "routing", "distillation"]',
                _METRIC_EXPERIMENTS.get(metric, ["prompt"])[0],
            )
            scored.append(
                ExperimentProposal(
                    kind=kind, target_metric=metric, weakness=round(boosted, 4),
                    current=round(value, 4), target=target, drift=metric in drift,
                    rationale=f"{metric} {value:.3f} below target {target:.3f}"
                    + (" (drift detected)" if metric in drift else ""),
                )
            )
        scored.sort(key=lambda p: p.weakness, reverse=True)
        # Allocate the global eval budget proportionally to weakness.
        total = sum(p.weakness for p in scored) or 1.0
        for proposal in scored:
            proposal.eval_budget = max(2, int(round(self.eval_budget * proposal.weakness / total)))
        return scored

    def propose(self) -> list[ExperimentProposal]:
        return self.rank(self.signals(), drift=self.drifted_metrics())

    def run_next(self, dataset: Dataset | None = None, *, dry_run: bool = False) -> dict[str, Any]:
        """Execute (or schedule) the highest-ROI proposal; record the decision."""
        proposals = self.propose()
        if not proposals:
            self.app.audit.record(
                "experiment_proposer", decision="skip",
                details={"reason": "no weakness above target"},
            )
            return {"proposal": None, "executed": False, "reason": "no weakness above target"}
        top = proposals[0]
        record: dict[str, Any] = {"proposal": top.model_dump(), "executed": False}
        if top.kind == "prompt" and not dry_run:
            loop = ImprovementLoop(
                self.app, optimizer="reflective", gates=self.gates,
                golden_suite=self.golden_suite,
            )
            result = loop.run(dataset=dataset, max_variants=min(top.eval_budget, 8), subset_size=4)
            record.update(executed=True, promoted=result.promoted, reason=result.reason,
                          promoted_ref=result.promoted_ref)
            decision = "allow" if result.promoted else "skip"
        else:
            # Non-prompt experiments are scheduled against the organ that owns them
            # (retrieval feedback, budget learner, routing optimizer, distillation).
            record.update(executed=False, scheduled=True,
                          reason=f"scheduled {top.kind} experiment for {top.target_metric}")
            decision = "schedule"
        self.app.audit.record(
            "experiment_proposer", decision=decision,
            details={"kind": top.kind, "metric": top.target_metric,
                     "weakness": top.weakness, "eval_budget": top.eval_budget,
                     "executed": record["executed"]},
        )
        self.app.events.emit("experiment.proposed", top.model_dump())
        return record
