"""Eval runner: run a target over a dataset with concurrency,
metrics, judges, gates, and baseline comparison."""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from ..core.errors import EvalError
from ..providers.base import run_sync
from .datasets import Dataset, EvalCase
from .judges import Judge
from .metrics import METRICS, Metric, MetricResult, RunOutput
from .reports import CaseResult, EvalReport, evaluate_gates

__all__ = ["EvalRunner", "EvalTarget"]

# A target maps an eval case to a RunOutput. ContextApp provides one; any
# async callable works (custom pipelines, baselines, ablations).
EvalTarget = Callable[[EvalCase], Awaitable[RunOutput]]

# Operational metrics vary run-to-run by nature (cost, latency); they are never
# treated as flake signals — only quality metrics quarantine a case.
_OPERATIONAL_METRICS = frozenset(
    {"cost", "latency", "input_tokens", "output_tokens", "cached_input_tokens", "retries"}
)

_AGGREGATORS: dict[str, Callable[[list[float]], float]] = {
    "mean": lambda xs: sum(xs) / len(xs),
    "median": statistics.median,
    "min": min,
    "max": max,
}


class EvalRunner:
    def __init__(
        self,
        target: Any,
        *,
        metrics: Sequence[str | Metric] | None = None,
        judges: list[Judge] | None = None,
        concurrency: int = 8,
        gates: dict[str, str] | None = None,
        repeats: int = 1,
        repeat_aggregate: str = "mean",
        flake_quarantine: bool = False,
        flake_threshold: float = 0.15,
    ) -> None:
        self.target = self._coerce_target(target)
        self.metric_fns = self._resolve_metrics(metrics or ["schema_validity", "semantic_similarity", "cost", "latency"])
        self.judges = judges or []
        self.concurrency = max(1, concurrency)
        self.gates = gates or {}
        # Repeats + flake control (1.8): run each case ``repeats`` times, report
        # per-case mean/stdev, aggregate by ``repeat_aggregate``, and (when
        # ``flake_quarantine``) tag cases whose quality metrics vary beyond
        # ``flake_threshold`` so non-mock provider noise never makes a swap gate
        # fire on a single noisy run.
        self.repeats = max(1, repeats)
        if repeat_aggregate not in _AGGREGATORS:
            raise EvalError(
                f"unknown repeat_aggregate {repeat_aggregate!r}; "
                f"known: {sorted(_AGGREGATORS)}"
            )
        self.repeat_aggregate = repeat_aggregate
        self.flake_quarantine = flake_quarantine
        self.flake_threshold = flake_threshold

    @staticmethod
    def _coerce_target(target: Any) -> EvalTarget:
        # ContextApp instances expose eval_target(); callables pass through.
        eval_target = getattr(target, "eval_target", None)
        if callable(eval_target):
            return eval_target
        if callable(target):
            return target
        raise EvalError(f"cannot evaluate target of type {type(target).__name__}")

    @staticmethod
    def _resolve_metrics(metrics: Sequence[str | Metric]) -> list[Metric]:
        resolved: list[Metric] = []
        for metric in metrics:
            if callable(metric):
                resolved.append(metric)
            elif metric in METRICS:
                resolved.append(METRICS[metric])
            else:
                raise EvalError(f"unknown metric {metric!r}; known: {sorted(METRICS)}")
        return resolved

    async def _score_output(
        self, case: EvalCase, output: RunOutput
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Score one run's output with every metric and judge."""
        metrics: dict[str, float] = {}
        details: dict[str, Any] = {}
        for metric_fn in self.metric_fns:
            try:
                metric_result: MetricResult = metric_fn(case, output)
            except Exception as exc:  # noqa: BLE001
                details[getattr(metric_fn, "__name__", "metric")] = f"metric error: {exc}"
                continue
            metrics[metric_result.name] = metric_result.value
            if metric_result.details:
                details[metric_result.name] = metric_result.details
        for judge in self.judges:
            try:
                judge_result = await judge.score(case, output)
            except Exception as exc:  # noqa: BLE001
                details[judge.name] = f"judge error: {exc}"
                continue
            metrics[judge_result.name] = judge_result.value
            if judge_result.details:
                details[judge_result.name] = judge_result.details
        return metrics, details

    async def _run_case(self, case: EvalCase, semaphore: asyncio.Semaphore) -> CaseResult:
        async with semaphore:
            runs: list[tuple[dict[str, float], dict[str, Any], RunOutput]] = []
            for _ in range(self.repeats):
                try:
                    output = await self.target(case)
                except Exception as exc:  # noqa: BLE001 - errored case, not errored eval
                    return CaseResult(
                        case_id=case.id, error=f"{type(exc).__name__}: {exc}", tags=list(case.tags)
                    )
                metrics, details = await self._score_output(case, output)
                runs.append((metrics, details, output))

            representative = runs[-1][2]
            aggregate = _AGGREGATORS[self.repeat_aggregate]
            # Aggregate the headline operational fields with the same aggregator as
            # the metrics so report.total_cost_usd matches the cost metric summary.
            result = CaseResult(
                case_id=case.id,
                output_text=representative.output_text[:2000],
                trace_id=representative.trace_id,
                latency_ms=int(aggregate([float(o.latency_ms) for _, _, o in runs])),
                cost_usd=aggregate([o.cost_usd for _, _, o in runs]),
                tags=list(case.tags),
                error=representative.error,
            )
            # Per-metric aggregation across repeats (last run's metric `details`
            # are kept as representative; values are aggregated).
            metric_names: set[str] = set()
            for metrics, _, _ in runs:
                metric_names.update(metrics)
            stdevs: dict[str, float] = {}
            for name in metric_names:
                values = [metrics[name] for metrics, _, _ in runs if name in metrics]
                if not values:
                    continue
                result.metrics[name] = round(aggregate(values), 6)
                if len(values) > 1:
                    stdevs[name] = round(statistics.stdev(values), 6)
            for _, details, _ in runs:
                for key, value in details.items():
                    result.details.setdefault(key, value)

            if self.repeats > 1:
                result.details["repeats"] = {
                    "n": self.repeats,
                    "aggregate": self.repeat_aggregate,
                    "stdev": stdevs,
                }
                if self.flake_quarantine:
                    flaky = {
                        name: sd
                        for name, sd in stdevs.items()
                        if name not in _OPERATIONAL_METRICS and sd > self.flake_threshold
                    }
                    if flaky:
                        result.tags.append("flaky")
                        result.details["flaky"] = {
                            "metrics": flaky,
                            "threshold": self.flake_threshold,
                        }
            return result

    async def arun(
        self,
        dataset: Dataset | str,
        *,
        baseline: EvalReport | None = None,
        raise_on_gate_failure: bool = False,
        name: str | None = None,
    ) -> EvalReport:
        if isinstance(dataset, str):
            dataset = Dataset.load(dataset)
        semaphore = asyncio.Semaphore(self.concurrency)
        case_results = await asyncio.gather(
            *(self._run_case(case, semaphore) for case in dataset.cases)
        )
        report = EvalReport(
            name=name or f"eval_{dataset.name}",
            dataset=dataset.name,
            cases=list(case_results),
        )
        if self.gates:
            # Quarantined-flaky cases are excluded from gate aggregation so
            # provider noise can't flip a gate on a single run; they remain in
            # the report (tagged "flaky") for inspection.
            gate_report = report
            if self.flake_quarantine and any("flaky" in c.tags for c in report.cases):
                gate_report = report.slice(lambda c: "flaky" not in c.tags)
                report.metadata["flaky_excluded_from_gates"] = sum(
                    1 for c in report.cases if "flaky" in c.tags
                )
            evaluate_gates(gate_report, self.gates, raise_on_failure=raise_on_gate_failure)
            report.gates = gate_report.gates
        if baseline is not None:
            report.metadata["baseline_diff"] = report.diff(baseline)
        return report

    def run(self, dataset: Dataset | str, **kwargs: Any) -> EvalReport:
        return run_sync(self.arun(dataset, **kwargs))
