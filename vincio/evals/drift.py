"""Drift detection.

:class:`DriftMonitor` watches two kinds of drift against a fixed baseline:

- **score drift** — a rolling metric's mean moving away from its baseline mean
  (a regression in quality, latency, cost, …);
- **embedding-distribution drift** — live inputs drifting away from the golden
  set's embedding distribution (the population the app was evaluated on).

When a baseline shifts past threshold it raises a ``drift.detected`` event on the
event bus and persists the baseline to the metadata store (kind
``drift_baselines``), so the same store holds runs, packets, and drift state.
Everything is computed in-process and offline; ``vincio eval drift`` reports it.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

from ..core.utils import utcnow

__all__ = ["DriftReport", "DriftMonitor"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - dot / (na * nb)


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


class DriftReport(BaseModel):
    metric: str = ""
    method: str = "score"  # score | embedding
    baseline: float = 0.0
    current: float = 0.0
    delta: float = 0.0
    z_score: float | None = None
    threshold: float = 0.0
    drifted: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class DriftMonitor:
    """Track score drift and embedding-distribution drift against a baseline."""

    def __init__(
        self,
        *,
        bus: Any = None,
        store: Any = None,
        app_name: str = "",
        score_threshold: float = 0.1,
        z_threshold: float = 3.0,
        embedding_threshold: float = 0.15,
    ) -> None:
        self.bus = bus
        self.store = store
        self.app_name = app_name
        self.score_threshold = score_threshold
        self.z_threshold = z_threshold
        self.embedding_threshold = embedding_threshold
        self._score_baselines: dict[str, tuple[float, float, int]] = {}  # metric -> (mean, std, n)
        self._embedding_baseline: tuple[list[float], float, int] | None = None  # centroid, spread, n

    # -- score drift ---------------------------------------------------------

    def set_score_baseline(self, metric: str, values: list[float]) -> None:
        mean = _mean(values)
        self._score_baselines[metric] = (mean, _stdev(values, mean), len(values))
        self._persist_baseline(
            f"score:{metric}", {"metric": metric, "method": "score", "mean": mean, "n": len(values)}
        )

    def check_scores(self, metric: str, values: list[float]) -> DriftReport:
        """Compare a recent window of metric values to the baseline. Drift fires
        when the absolute mean shift exceeds ``score_threshold`` or the z-score of
        the shift exceeds ``z_threshold``."""
        if metric not in self._score_baselines:
            self.set_score_baseline(metric, values)
            return DriftReport(metric=metric, method="score", baseline=_mean(values),
                               current=_mean(values), threshold=self.score_threshold,
                               details={"baseline_set": True})
        base_mean, base_std, base_n = self._score_baselines[metric]
        current = _mean(values)
        delta = current - base_mean
        z = None
        if base_std > 0 and values:
            z = abs(delta) / (base_std / math.sqrt(max(1, len(values))))
        drifted = abs(delta) > self.score_threshold or (z is not None and z > self.z_threshold)
        report = DriftReport(
            metric=metric, method="score", baseline=round(base_mean, 6), current=round(current, 6),
            delta=round(delta, 6), z_score=round(z, 4) if z is not None else None,
            threshold=self.score_threshold, drifted=drifted,
            details={"baseline_n": base_n, "window_n": len(values)},
        )
        if drifted:
            self._raise(report)
        return report

    # -- embedding-distribution drift ----------------------------------------

    def set_embedding_baseline(self, vectors: list[list[float]]) -> None:
        centroid = _centroid(vectors)
        spread = _mean([_cosine_distance(v, centroid) for v in vectors]) if centroid else 0.0
        self._embedding_baseline = (centroid, spread, len(vectors))
        self._persist_baseline(
            "embedding:inputs",
            {"method": "embedding", "spread": spread, "n": len(vectors), "dim": len(centroid)},
        )

    def check_embeddings(self, vectors: list[list[float]]) -> DriftReport:
        """Mean cosine distance of live input embeddings to the golden-set
        centroid, vs the golden set's own spread. Drift fires when the excess
        distance exceeds ``embedding_threshold``."""
        if self._embedding_baseline is None:
            self.set_embedding_baseline(vectors)
            return DriftReport(method="embedding", details={"baseline_set": True},
                               threshold=self.embedding_threshold)
        centroid, base_spread, base_n = self._embedding_baseline
        current = _mean([_cosine_distance(v, centroid) for v in vectors]) if centroid else 0.0
        delta = current - base_spread
        drifted = delta > self.embedding_threshold
        report = DriftReport(
            metric="input_embeddings", method="embedding", baseline=round(base_spread, 6),
            current=round(current, 6), delta=round(delta, 6), threshold=self.embedding_threshold,
            drifted=drifted, details={"baseline_n": base_n, "window_n": len(vectors)},
        )
        if drifted:
            self._raise(report)
        return report

    # -- internals -----------------------------------------------------------

    def _raise(self, report: DriftReport) -> None:
        if self.bus is not None:
            self.bus.emit("drift.detected", report.model_dump())

    def _persist_baseline(self, key: str, payload: dict[str, Any]) -> None:
        if self.store is None:
            return
        self.store.save(
            "drift_baselines",
            {"id": f"{self.app_name}:{key}", "app_id": self.app_name,
             "created_at": utcnow().isoformat(), **payload},
        )
