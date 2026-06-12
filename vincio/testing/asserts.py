"""Eval assertions: unit-test-style checks over run outputs.

Every assertion accepts a ``RunResult``, ``RunOutput``, or plain string and
raises ``AssertionError`` with the metric breakdown on failure, so failures
read like test failures, not stack traces. Thresholds are explicit at the
call site — the CI-friendly contract is "this metric must clear this bar".
"""

from __future__ import annotations

from typing import Any

from ..core.types import EvidenceItem
from ..evals.datasets import EvalCase
from ..evals.metrics import METRICS, MetricResult, RunOutput

__all__ = ["assert_eval", "assert_metric", "assert_grounded", "assert_safe"]


def _to_run_output(output: Any, evidence: list[EvidenceItem] | None = None) -> RunOutput:
    if isinstance(output, RunOutput):
        return output
    if hasattr(output, "raw_text") and hasattr(output, "evidence"):  # RunResult
        return RunOutput(
            output=output.output,
            raw_text=output.raw_text,
            evidence=list(output.evidence),
            citations=list(getattr(output, "citations", [])),
            error=getattr(output, "error", None),
        )
    return RunOutput(output=str(output), evidence=evidence or [])


def _to_case(case: Any, *, expected: Any = None, context: dict[str, Any] | None = None) -> EvalCase:
    if isinstance(case, EvalCase):
        return case
    return EvalCase(id="assert", input=str(case), expected=expected, context=context or {})


def _run_metric(name: str, case: EvalCase, run: RunOutput) -> MetricResult:
    metric = METRICS.get(name)
    if metric is None:
        raise AssertionError(
            f"unknown metric {name!r}; registered metrics: {', '.join(sorted(METRICS))}"
        )
    return metric(case, run)


_LOWER_IS_BETTER = {"hallucination", "toxicity", "bias", "unsupported_claim_rate",
                    "cost", "latency", "retries"}


def assert_metric(
    output: Any,
    input: Any = "",
    *,
    metric: str,
    threshold: float,
    expected: Any = None,
    context: dict[str, Any] | None = None,
    evidence: list[EvidenceItem] | None = None,
) -> MetricResult:
    """Assert one metric clears its threshold; returns the result.

    Quality metrics must be ``>= threshold``; rate/cost-style metrics
    (hallucination, toxicity, bias, latency, ...) must be ``<= threshold``.
    """
    run = _to_run_output(output, evidence)
    case = _to_case(input, expected=expected, context=context)
    result = _run_metric(metric, case, run)
    ok = result.value <= threshold if metric in _LOWER_IS_BETTER else result.value >= threshold
    if not ok:
        comparator = "<=" if metric in _LOWER_IS_BETTER else ">="
        raise AssertionError(
            f"{metric}={result.value} failed {comparator} {threshold}\n"
            f"  details: {result.details}\n"
            f"  output: {run.output_text[:300]!r}"
        )
    return result


def assert_eval(
    output: Any,
    input: Any = "",
    *,
    metrics: dict[str, float],
    expected: Any = None,
    context: dict[str, Any] | None = None,
    evidence: list[EvidenceItem] | None = None,
) -> dict[str, MetricResult]:
    """Assert several metric thresholds at once; reports every failure.

    ``metrics`` maps metric name to threshold, e.g.
    ``{"answer_relevance": 0.5, "hallucination": 0.0}``.
    """
    run = _to_run_output(output, evidence)
    case = _to_case(input, expected=expected, context=context)
    results: dict[str, MetricResult] = {}
    failures: list[str] = []
    for name, threshold in metrics.items():
        result = _run_metric(name, case, run)
        results[name] = result
        ok = result.value <= threshold if name in _LOWER_IS_BETTER else result.value >= threshold
        if not ok:
            comparator = "<=" if name in _LOWER_IS_BETTER else ">="
            failures.append(f"  {name}={result.value} (wanted {comparator} {threshold}) {result.details}")
    if failures:
        raise AssertionError(
            "eval assertions failed:\n" + "\n".join(failures)
            + f"\n  output: {run.output_text[:300]!r}"
        )
    return results


def assert_grounded(
    output: Any,
    *,
    evidence: list[EvidenceItem] | None = None,
    threshold: float = 0.8,
) -> MetricResult:
    """Assert the output is grounded in its evidence (groundedness >= bar).

    Pass a ``RunResult`` (evidence rides along) or text plus ``evidence``.
    """
    run = _to_run_output(output, evidence)
    if evidence is not None and not run.evidence:
        run.evidence = evidence
    if not run.evidence:
        raise AssertionError("assert_grounded: no evidence on the output and none provided")
    case = _to_case("")
    result = _run_metric("groundedness", case, run)
    if result.value < threshold:
        raise AssertionError(
            f"groundedness={result.value} below threshold {threshold}\n"
            f"  details: {result.details}\n"
            f"  output: {run.output_text[:300]!r}"
        )
    return result


def assert_safe(output: Any, *, max_toxicity: float = 0.0, max_bias: float = 0.0) -> None:
    """Assert the output is free of toxic and biased language."""
    assert_metric(output, metric="toxicity", threshold=max_toxicity)
    assert_metric(output, metric="bias", threshold=max_bias)
