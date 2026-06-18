"""Metric-as-guardrail adapter.

The interconnection promise: *every metric is the same object usable as a
runtime guardrail*. :func:`metric_guardrail` wraps an eval :data:`Metric` as a
deterministic rail predicate ``(text, params) -> message | None``, so the metric
you gate releases with offline (e.g. ``answer_relevance``, ``toxicity``,
``conversation_outcome``) becomes an input/output guardrail at run time — no
second implementation, one source of truth for direction.

Text-level metrics work directly. Metrics that need evidence/expected can be fed
them through the rail's ``params`` (``evidence``, ``expected``, ``input``).
Trajectory metrics, which need a full agent trajectory, are gated offline rather
than at the text boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.types import EvidenceItem
from .datasets import EvalCase
from .metrics import LOWER_IS_BETTER, METRICS, Metric, RunOutput

__all__ = ["metric_guardrail"]

RailPredicate = Callable[[str, dict[str, Any]], str | None]


def metric_guardrail(
    metric: str | Metric,
    *,
    threshold: float,
    name: str | None = None,
    lower_is_better: bool | None = None,
) -> RailPredicate:
    """Return a rail predicate that fires when ``metric`` crosses ``threshold``.

    Direction defaults to the metric's entry in :data:`LOWER_IS_BETTER` (lower-is-
    better metrics fire when the value exceeds the threshold; higher-is-better
    metrics fire when it falls below). Override with ``lower_is_better``. The
    predicate returns a violation message (truthy) or ``None`` (pass).
    """
    if isinstance(metric, str):
        if metric not in METRICS:
            raise KeyError(f"unknown metric {metric!r}; known: {sorted(METRICS)}")
        metric_name = metric
        metric_fn: Metric = METRICS[metric]
    else:
        metric_name = getattr(metric, "__name__", "metric")
        metric_fn = metric
    fires_high = metric_name in LOWER_IS_BETTER if lower_is_better is None else lower_is_better

    def predicate(text: str, params: dict[str, Any]) -> str | None:
        params = params or {}
        limit = float(params.get("threshold", threshold))
        evidence_in = params.get("evidence") or []
        evidence = [
            e if isinstance(e, EvidenceItem) else EvidenceItem(id=f"e{i}", source_id="rail", text=str(e))
            for i, e in enumerate(evidence_in)
        ]
        case = EvalCase(id="guardrail", input=params.get("input", ""), expected=params.get("expected"))
        run = RunOutput(output=text, raw_text=text, evidence=evidence)
        result = metric_fn(case, run)
        breached = result.value > limit if fires_high else result.value < limit
        if breached:
            comparator = ">" if fires_high else "<"
            return f"{metric_name}={result.value} {comparator} threshold {limit}"
        return None

    predicate.__name__ = name or f"{metric_name}_guard"
    return predicate
