"""Online / continuous evaluation.

An :class:`OnlineEvaluator` scores a *sample* of live runs with the same metric
objects used offline, and writes each score as a time-series record on the
metadata store (kind ``eval_results``) — no traffic mirrored to any external
service. It runs after the response is finalized (the app schedules it off the
hot path), and sampling bounds the overhead. The synchronous :meth:`observe`
core makes online scoring deterministic and unit-testable.
"""

from __future__ import annotations

from typing import Any

from ..core.utils import new_id, utcnow
from .datasets import EvalCase
from .metrics import METRICS, Metric, MetricResult, RunOutput

__all__ = ["OnlineEvaluator"]


class OnlineEvaluator:
    """Score a sampled fraction of live runs and persist a score time series."""

    def __init__(
        self,
        metric: str | Metric,
        *,
        name: str | None = None,
        sample_rate: float = 1.0,
        store: Any = None,
        app_name: str = "",
    ) -> None:
        if isinstance(metric, str):
            if metric not in METRICS:
                raise KeyError(f"unknown metric {metric!r}; known: {sorted(METRICS)}")
            self.metric: Metric = METRICS[metric]
            self.name = name or metric
        else:
            self.metric = metric
            self.name = name or str(getattr(metric, "__name__", "online_metric"))
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.store = store
        self.app_name = app_name
        self._counter = 0

    def _should_sample(self) -> bool:
        """Deterministic 1-in-N sampling (matches the tracer), so online eval is
        reproducible in tests and evenly spread in production."""
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        self._counter += 1
        period = max(1, round(1.0 / self.sample_rate))
        return self._counter % period == 0

    def observe(
        self, run: RunOutput, *, case: EvalCase | None = None, run_id: str = ""
    ) -> MetricResult | None:
        """Score one completed run if it is sampled, persisting the score as a
        time-series record. Returns the MetricResult, or None when not sampled."""
        if not self._should_sample():
            return None
        case = case or EvalCase(id=run_id or "online", input=run.metadata.get("input", ""))
        result = self.metric(case, run)
        if self.store is not None:
            self.store.save(
                "eval_results",
                {
                    "id": new_id("online"),
                    "app_id": self.app_name,
                    "dataset_id": "online",
                    "run_id": run_id,
                    "metric_name": self.name,
                    "metric_value": result.value,
                    "created_at": utcnow().isoformat(),
                    "details": dict(result.details),
                },
            )
        return result

    def series(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        """The recorded score time series for this metric, oldest first."""
        if self.store is None:
            return []
        rows = self.store.query(
            "eval_results",
            where={"metric_name": self.name, "dataset_id": "online", "app_id": self.app_name},
            limit=limit,
        )
        return sorted(rows, key=lambda r: r.get("created_at", ""))
