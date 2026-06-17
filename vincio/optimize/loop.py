"""The closed improvement loop (0.8): trace → dataset → eval → optimize → promote.

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
    from ..observability.spans import Trace

__all__ = ["LoopResult", "ImprovementLoop", "DEFAULT_LOOP_METRICS"]

DEFAULT_LOOP_METRICS = [
    "semantic_similarity",
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
        # "evolution" (0.8 blind variant search) or "reflective" (1.4 GEPA-style
        # failure-driven search). The reflective path proposes targeted edits
        # from the eval report's failures and evolves a Pareto frontier; both
        # promote through the identical gated path below.
        self.optimizer = optimizer
        self.strategy = strategy

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
            from .reflective import ReflectiveOptimizer

            reflective = ReflectiveOptimizer(
                evaluate_variant,
                weights=self.weights,
                gates=self.gates,
                max_cost_per_case=self.max_cost_per_case,
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
                # Statistical backing (1.7): the promotion is defensible at a
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
        return result

    def run(self, **kwargs: Any) -> LoopResult:
        return run_sync(self.arun(**kwargs))
