"""Drift detection.

:class:`DriftMonitor` watches three kinds of drift against a fixed baseline:

- **score drift** — a rolling metric's mean moving away from its baseline mean
  (a regression in quality, latency, cost, …), as a mean-shift / z-score test
  *and* as a streaming :class:`CUSUMDetector` changepoint over individual
  online scores;
- **distributional drift** — a window of values diverging from the baseline
  *distribution* via the two-sample **Kolmogorov–Smirnov** statistic, the
  **Population Stability Index** (PSI), or **Maximum Mean Discrepancy** (MMD²);
- **embedding-distribution drift** — live inputs drifting away from the golden
  set's embedding distribution (the population the app was evaluated on).

When a baseline shifts past threshold it raises a ``drift.detected`` event on the
event bus and persists the baseline to the metadata store (kind
``drift_baselines``), so the same store holds runs, packets, and drift state. The
CUSUM accumulators and changepoint counts persist to ``drift_state`` so the
detector is restart-safe and aggregatable across workers. Everything is computed
in-process and offline; ``vincio eval drift`` reports it.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

from ..core.utils import utcnow

__all__ = [
    "DriftReport",
    "DriftMonitor",
    "CUSUMDetector",
    "ks_statistic",
    "ks_drift",
    "psi",
    "rbf_mmd2",
]


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


# ---------------------------------------------------------------------------
# Distributional drift statistics (pure-Python, deterministic)
# ---------------------------------------------------------------------------


def ks_statistic(baseline: list[float], current: list[float]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic D = max|F_a(x) − F_b(x)|.

    The supremum distance between the two empirical CDFs, in [0, 1]; 0 means the
    samples are indistinguishable, 1 means fully separated. O((n+m) log(n+m)).
    """
    if not baseline or not current:
        return 0.0
    a = sorted(baseline)
    b = sorted(current)
    na, nb = len(a), len(b)
    i = j = 0
    d = 0.0
    # Merge-walk the two sorted samples over their union of distinct values,
    # advancing *both* pointers past ties so equal samples score exactly 0.
    while i < na or j < nb:
        if j >= nb or (i < na and a[i] <= b[j]):
            x = a[i]
        else:
            x = b[j]
        while i < na and a[i] == x:
            i += 1
        while j < nb and b[j] == x:
            j += 1
        d = max(d, abs(i / na - j / nb))
    return d


def _ks_pvalue(d: float, na: int, nb: int) -> float:
    """Asymptotic two-sided KS p-value (Kolmogorov distribution)."""
    if na == 0 or nb == 0 or d <= 0.0:
        return 1.0
    n_e = na * nb / (na + nb)
    lam = (math.sqrt(n_e) + 0.12 + 0.11 / math.sqrt(n_e)) * d
    # p = 2·Σ_{k≥1} (−1)^{k−1} e^{−2 k² λ²}
    total = 0.0
    for k in range(1, 101):
        term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * k * k * lam * lam)
        total += term
        if abs(term) < 1e-9:
            break
    return max(0.0, min(1.0, total))


def ks_drift(baseline: list[float], current: list[float], *, alpha: float = 0.05) -> tuple[float, float, bool]:
    """KS statistic, asymptotic p-value, and a drift verdict at ``alpha``."""
    d = ks_statistic(baseline, current)
    p = _ks_pvalue(d, len(baseline), len(current))
    return d, p, (p < alpha and d > 0.0)


def psi(baseline: list[float], current: list[float], *, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index between a baseline and a current sample.

    The baseline is split into ``bins`` quantile buckets; PSI sums
    ``(cur% − base%)·ln(cur%/base%)`` across buckets. Convention:
    < 0.1 stable, 0.1–0.25 moderate shift, > 0.25 significant shift.
    """
    if len(baseline) < 2 or not current:
        return 0.0
    ordered = sorted(baseline)
    n = len(ordered)
    bins = max(1, min(bins, n))
    # Quantile cut points from the baseline; collapse duplicate edges.
    edges = [ordered[min(n - 1, round(q * n / bins))] for q in range(1, bins)]
    edges = sorted(set(edges))
    bounds = [-math.inf, *edges, math.inf]

    def histogram(values: list[float]) -> list[float]:
        counts = [0] * (len(bounds) - 1)
        for v in values:
            for b in range(len(bounds) - 1):
                if bounds[b] < v <= bounds[b + 1] or (b == 0 and v <= bounds[1]):
                    counts[b] += 1
                    break
        total = sum(counts) or 1
        return [(c / total) for c in counts]

    base_p = histogram(ordered)
    cur_p = histogram(current)
    score = 0.0
    for bp, cp in zip(base_p, cur_p, strict=False):
        bp_a = max(bp, eps)
        cp_a = max(cp, eps)
        score += (cp_a - bp_a) * math.log(cp_a / bp_a)
    return score


def _as_vectors(sample: list[Any]) -> list[list[float]]:
    out: list[list[float]] = []
    for item in sample:
        if isinstance(item, list | tuple):
            out.append([float(x) for x in item])
        else:
            out.append([float(item)])
    return out


def rbf_mmd2(baseline: list[Any], current: list[Any], *, bandwidth: float | None = None) -> float:
    """Biased MMD² between two samples under an RBF kernel.

    Accepts scalars or equal-length vectors. The kernel bandwidth defaults to
    the median pairwise squared distance (the standard median heuristic), making
    the statistic scale-free. Returns a non-negative discrepancy; 0 means the
    samples are drawn from the same distribution.
    """
    xs = _as_vectors(baseline)
    ys = _as_vectors(current)
    if not xs or not ys:
        return 0.0

    def sq_dist(a: list[float], b: list[float]) -> float:
        return sum((p - q) ** 2 for p, q in zip(a, b, strict=False))

    if bandwidth is None:
        pooled = xs + ys
        dists = [
            sq_dist(pooled[i], pooled[j])
            for i in range(len(pooled))
            for j in range(i + 1, len(pooled))
        ]
        dists = [d for d in dists if d > 0.0]
        bandwidth = (sorted(dists)[len(dists) // 2]) if dists else 1.0
    gamma = 1.0 / (2.0 * max(bandwidth, 1e-9))

    def k(a: list[float], b: list[float]) -> float:
        return math.exp(-gamma * sq_dist(a, b))

    def mean_kernel(p: list[list[float]], q: list[list[float]], *, same: bool) -> float:
        total = 0.0
        count = 0
        for i, a in enumerate(p):
            for j, b in enumerate(q):
                if same and i == j:
                    continue
                total += k(a, b)
                count += 1
        return total / count if count else 0.0

    kxx = mean_kernel(xs, xs, same=True)
    kyy = mean_kernel(ys, ys, same=True)
    kxy = mean_kernel(xs, ys, same=False)
    return max(0.0, kxx + kyy - 2.0 * kxy)


class CUSUMDetector:
    """Two-sided cumulative-sum changepoint detector over a score stream.

    Tracks upward (``s_hi``) and downward (``s_lo``) cumulative deviations from a
    target mean. A point at index *t* raises an alarm when either accumulator
    crosses ``threshold·sigma``; the accumulators reset on alarm so the detector
    keeps watching for the *next* changepoint rather than latching. ``slack``
    (the reference value *k*) is the per-step tolerance in units of sigma — it
    is what makes CUSUM detect a *sustained* shift, not a single noisy sample.

    State is plain floats/ints, so :meth:`state` / :meth:`load` make it
    restart-safe and aggregatable across workers via the metadata store.
    """

    def __init__(
        self,
        *,
        target: float = 0.0,
        sigma: float = 1.0,
        slack: float = 0.5,
        threshold: float = 5.0,
    ) -> None:
        self.target = target
        self.sigma = max(sigma, 1e-9)
        self.slack = slack
        self.threshold = threshold
        self.s_hi = 0.0
        self.s_lo = 0.0
        self.n = 0
        self.changepoints = 0
        self.last_direction = ""  # "up" | "down" on the most recent alarm

    def observe(self, value: float) -> bool:
        """Feed one score; return True on a changepoint (and reset accumulators)."""
        self.n += 1
        z = (value - self.target) / self.sigma
        self.s_hi = max(0.0, self.s_hi + z - self.slack)
        self.s_lo = max(0.0, self.s_lo - z - self.slack)
        if self.s_hi > self.threshold:
            self.changepoints += 1
            self.last_direction = "up"
            self.s_hi = self.s_lo = 0.0
            return True
        if self.s_lo > self.threshold:
            self.changepoints += 1
            self.last_direction = "down"
            self.s_hi = self.s_lo = 0.0
            return True
        return False

    def reset(self) -> None:
        self.s_hi = self.s_lo = 0.0

    def state(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "sigma": self.sigma,
            "slack": self.slack,
            "threshold": self.threshold,
            "s_hi": self.s_hi,
            "s_lo": self.s_lo,
            "n": self.n,
            "changepoints": self.changepoints,
            "last_direction": self.last_direction,
        }

    def load(self, state: dict[str, Any]) -> None:
        self.s_hi = float(state.get("s_hi", 0.0))
        self.s_lo = float(state.get("s_lo", 0.0))
        self.n = int(state.get("n", 0))
        self.changepoints = int(state.get("changepoints", 0))
        self.last_direction = str(state.get("last_direction", ""))
        if "target" in state:
            self.target = float(state["target"])
        if "sigma" in state:
            self.sigma = max(float(state["sigma"]), 1e-9)


class DriftReport(BaseModel):
    metric: str = ""
    method: str = "score"  # score | cusum | ks | psi | mmd | embedding
    baseline: float = 0.0
    current: float = 0.0
    delta: float = 0.0
    z_score: float | None = None
    statistic: float | None = None  # KS D / PSI / MMD² / CUSUM accumulator
    p_value: float | None = None
    threshold: float = 0.0
    drifted: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class DriftMonitor:
    """Track score, distributional, and embedding-distribution drift."""

    def __init__(
        self,
        *,
        bus: Any = None,
        store: Any = None,
        app_name: str = "",
        score_threshold: float = 0.1,
        z_threshold: float = 3.0,
        embedding_threshold: float = 0.15,
        psi_threshold: float = 0.25,
        ks_alpha: float = 0.05,
        mmd_threshold: float = 0.1,
        cusum_slack: float = 0.5,
        cusum_threshold: float = 5.0,
    ) -> None:
        self.bus = bus
        self.store = store
        self.app_name = app_name
        self.score_threshold = score_threshold
        self.z_threshold = z_threshold
        self.embedding_threshold = embedding_threshold
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.mmd_threshold = mmd_threshold
        self.cusum_slack = cusum_slack
        self.cusum_threshold = cusum_threshold
        self._score_baselines: dict[str, tuple[float, float, int]] = {}  # metric -> (mean, std, n)
        self._distribution_baselines: dict[str, list[float]] = {}  # key -> baseline sample
        self._embedding_baseline: tuple[list[float], float, int] | None = None  # centroid, spread, n
        self._cusum: dict[str, CUSUMDetector] = {}

    # -- score drift ---------------------------------------------------------

    def set_score_baseline(self, metric: str, values: list[float]) -> None:
        mean = _mean(values)
        std = _stdev(values, mean)
        self._score_baselines[metric] = (mean, std, len(values))
        # A CUSUM detector is anchored on the same baseline so a stream of online
        # scores can be watched for a sustained changepoint, not just a window mean.
        self._cusum[metric] = CUSUMDetector(
            target=mean,
            sigma=std or max(abs(mean), 1.0) * 0.1 or 1.0,
            slack=self.cusum_slack,
            threshold=self.cusum_threshold,
        )
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

    def observe_score(self, metric: str, value: float) -> DriftReport | None:
        """Feed one online score into the metric's CUSUM changepoint detector.

        This is the streaming counterpart to :meth:`check_scores`: it turns the
        sequence of online eval scores into a *sustained-shift* alarm rather than
        a window comparison, so the monitor can fire ``drift.detected`` the moment
        a regression accumulates past the CUSUM threshold. Returns a
        ``method="cusum"`` :class:`DriftReport` on a changepoint, else ``None``.
        """
        if metric not in self._cusum:
            # No baseline yet: nothing to detect a changepoint against.
            return None
        detector = self._cusum[metric]
        fired = detector.observe(value)
        self._persist_state(metric, detector)
        if not fired:
            return None
        base_mean = self._score_baselines.get(metric, (detector.target, 0.0, 0))[0]
        report = DriftReport(
            metric=metric,
            method="cusum",
            baseline=round(base_mean, 6),
            current=round(value, 6),
            delta=round(value - base_mean, 6),
            statistic=round(max(detector.s_hi, detector.s_lo), 4),
            threshold=detector.threshold,
            drifted=True,
            details={
                "direction": detector.last_direction,
                "changepoints": detector.changepoints,
                "observations": detector.n,
            },
        )
        self._raise(report)
        return report

    # -- distributional drift (KS / PSI / MMD) -------------------------------

    def set_distribution_baseline(self, key: str, values: list[float]) -> None:
        self._distribution_baselines[key] = list(values)
        self._persist_baseline(
            f"dist:{key}", {"metric": key, "method": "distribution", "n": len(values)}
        )

    def check_distribution(
        self, key: str, values: list[Any], *, method: str = "ks"
    ) -> DriftReport:
        """Compare a window of values to a baseline distribution.

        ``method`` selects the test: ``"ks"`` (Kolmogorov–Smirnov + p-value),
        ``"psi"`` (Population Stability Index), or ``"mmd"`` (RBF MMD², which also
        accepts vectors). The first call with a given ``key`` sets the baseline.
        """
        if key not in self._distribution_baselines:
            self.set_distribution_baseline(key, [float(_scalar(v)) for v in values])
            return DriftReport(metric=key, method=method, details={"baseline_set": True})
        baseline = self._distribution_baselines[key]
        if method == "psi":
            cur = [float(_scalar(v)) for v in values]
            score = psi(baseline, cur)
            drifted = score > self.psi_threshold
            report = DriftReport(
                metric=key, method="psi", statistic=round(score, 6),
                threshold=self.psi_threshold, drifted=drifted,
                details={"baseline_n": len(baseline), "window_n": len(cur)},
            )
        elif method == "mmd":
            score = rbf_mmd2(baseline, list(values))
            drifted = score > self.mmd_threshold
            report = DriftReport(
                metric=key, method="mmd", statistic=round(score, 6),
                threshold=self.mmd_threshold, drifted=drifted,
                details={"baseline_n": len(baseline), "window_n": len(values)},
            )
        else:  # ks
            cur = [float(_scalar(v)) for v in values]
            d, p, drifted = ks_drift(baseline, cur, alpha=self.ks_alpha)
            report = DriftReport(
                metric=key, method="ks", statistic=round(d, 6), p_value=round(p, 6),
                threshold=self.ks_alpha, drifted=drifted,
                details={"baseline_n": len(baseline), "window_n": len(cur)},
            )
        if report.drifted:
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

    def _persist_state(self, metric: str, detector: CUSUMDetector) -> None:
        if self.store is None:
            return
        self.store.save(
            "drift_state",
            {"id": f"{self.app_name}:cusum:{metric}", "app_id": self.app_name,
             "metric": metric, "created_at": utcnow().isoformat(), **detector.state()},
        )

    def reset_cusum(self, metric: str) -> None:
        """Reset a metric's CUSUM accumulators (e.g. after a prompt regime change)."""
        detector = self._cusum.get(metric)
        if detector is not None:
            detector.reset()

    def load_state(self) -> int:
        """Restore persisted CUSUM accumulators from the store (restart-safe).

        Returns the number of detectors restored. Baselines must already be set
        (e.g. via :meth:`set_score_baseline`) so the detectors exist; this only
        rehydrates their running accumulators and changepoint counts.
        """
        if self.store is None:
            return 0
        restored = 0
        rows = self.store.query("drift_state", where={"app_id": self.app_name})
        for row in rows:
            metric = row.get("metric")
            if metric in self._cusum:
                self._cusum[metric].load(row)
                restored += 1
        return restored


def _scalar(value: Any) -> float:
    if isinstance(value, list | tuple):
        return float(value[0]) if value else 0.0
    return float(value)
