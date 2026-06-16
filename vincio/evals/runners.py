"""Eval runner: run a target over a dataset with concurrency,
metrics, judges, gates, and baseline comparison."""

from __future__ import annotations

import asyncio
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


class EvalRunner:
    def __init__(
        self,
        target: Any,
        *,
        metrics: Sequence[str | Metric] | None = None,
        judges: list[Judge] | None = None,
        concurrency: int = 8,
        gates: dict[str, str] | None = None,
    ) -> None:
        self.target = self._coerce_target(target)
        self.metric_fns = self._resolve_metrics(metrics or ["schema_validity", "semantic_similarity", "cost", "latency"])
        self.judges = judges or []
        self.concurrency = max(1, concurrency)
        self.gates = gates or {}

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

    async def _run_case(self, case: EvalCase, semaphore: asyncio.Semaphore) -> CaseResult:
        async with semaphore:
            try:
                output = await self.target(case)
            except Exception as exc:  # noqa: BLE001 - errored case, not errored eval
                return CaseResult(case_id=case.id, error=f"{type(exc).__name__}: {exc}", tags=case.tags)
            result = CaseResult(
                case_id=case.id,
                output_text=output.output_text[:2000],
                trace_id=output.trace_id,
                latency_ms=output.latency_ms,
                cost_usd=output.cost_usd,
                tags=case.tags,
                error=output.error,
            )
            for metric_fn in self.metric_fns:
                try:
                    metric_result: MetricResult = metric_fn(case, output)
                except Exception as exc:  # noqa: BLE001
                    result.details[getattr(metric_fn, "__name__", "metric")] = f"metric error: {exc}"
                    continue
                result.metrics[metric_result.name] = metric_result.value
                if metric_result.details:
                    result.details[metric_result.name] = metric_result.details
            for judge in self.judges:
                try:
                    judge_result = await judge.score(case, output)
                except Exception as exc:  # noqa: BLE001
                    result.details[judge.name] = f"judge error: {exc}"
                    continue
                result.metrics[judge_result.name] = judge_result.value
                if judge_result.details:
                    result.details[judge_result.name] = judge_result.details
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
            evaluate_gates(report, self.gates, raise_on_failure=raise_on_gate_failure)
        if baseline is not None:
            report.metadata["baseline_diff"] = report.diff(baseline)
        return report

    def run(self, dataset: Dataset | str, **kwargs: Any) -> EvalReport:
        return run_sync(self.arun(dataset, **kwargs))
