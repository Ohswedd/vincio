"""Experiment tracking: local run store, comparisons, ablations, A/B tests.

Experiments live in the same metadata store as everything else (SQLite by
default) — no hosted tracker. Log eval reports under an experiment/variant,
compare variants side by side, run ablations against a baseline, and test
prompt/retriever A/Bs for statistical significance (Welch's or paired t-test,
pure Python).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import EvalError
from ..core.utils import new_id, utcnow
from ..storage.base import MetadataStore
from .metrics import LOWER_IS_BETTER
from .reports import EvalReport

__all__ = ["ExperimentRun", "ExperimentTracker", "Experiment", "ab_test"]


class ExperimentRun(BaseModel):
    """One logged eval run inside an experiment."""

    id: str = Field(default_factory=lambda: new_id("exp"))
    experiment: str
    variant: str = "baseline"
    params: dict[str, Any] = Field(default_factory=dict)
    report: EvalReport
    created_at: Any = Field(default_factory=utcnow)

    def metric_mean(self, metric: str) -> float | None:
        values = self.report.metric_values(metric)
        return sum(values) / len(values) if values else None


class ExperimentTracker:
    """Append-only experiment log on a :class:`MetadataStore`.

    Accepts an existing store (e.g. ``app.store``) or a path, in which case a
    local SQLite store is created. Runs are stored under kind
    ``"experiments"`` and queried back as full reports, so comparisons and
    significance tests always work from per-case values.
    """

    KIND = "experiments"

    def __init__(self, store: MetadataStore | str | Path = ".vincio/experiments.db") -> None:
        if isinstance(store, (str, Path)):
            from ..storage.sqlite import SQLiteMetadataStore

            store = SQLiteMetadataStore(store)
        self.store = store

    # -- logging ---------------------------------------------------------------

    def log(
        self,
        experiment: str,
        report: EvalReport,
        *,
        variant: str = "baseline",
        params: dict[str, Any] | None = None,
    ) -> ExperimentRun:
        run = ExperimentRun(
            experiment=experiment, variant=variant, params=params or {}, report=report
        )
        record = run.model_dump(mode="json")
        record["id"] = run.id
        self.store.save(self.KIND, record)
        return run

    def runs(self, experiment: str, *, variant: str | None = None) -> list[ExperimentRun]:
        where: dict[str, Any] = {"experiment": experiment}
        if variant is not None:
            where["variant"] = variant
        records = self.store.query(self.KIND, where=where, limit=10_000)
        runs = [ExperimentRun.model_validate(record) for record in records]
        runs.sort(key=lambda run: str(run.created_at))
        return runs

    def experiments(self) -> list[str]:
        records = self.store.query(self.KIND, limit=10_000)
        return sorted({str(record.get("experiment")) for record in records})

    def latest(self, experiment: str, variant: str) -> ExperimentRun:
        runs = self.runs(experiment, variant=variant)
        if not runs:
            raise EvalError(f"experiment {experiment!r} has no runs for variant {variant!r}")
        return runs[-1]

    # -- analysis ---------------------------------------------------------------

    def _latest_per_variant(self, experiment: str) -> dict[str, ExperimentRun]:
        runs = self.runs(experiment)
        if not runs:
            raise EvalError(f"experiment {experiment!r} has no runs")
        latest: dict[str, ExperimentRun] = {}
        for run in runs:
            latest[run.variant] = run
        return latest

    def compare(
        self,
        experiment: str,
        *,
        metrics: list[str] | None = None,
        latest: dict[str, ExperimentRun] | None = None,
    ) -> dict[str, Any]:
        """Side-by-side comparison of each variant's latest run.

        Returns per-metric means by variant plus the best variant per metric
        (max for quality metrics; min for cost/latency/error-style metrics).
        """
        latest = latest or self._latest_per_variant(experiment)
        metric_names = set(metrics or [])
        if not metric_names:
            for run in latest.values():
                metric_names.update(run.report.summary().keys())
        table: dict[str, dict[str, float]] = {}
        best: dict[str, str] = {}
        for metric in sorted(metric_names):
            row = {
                variant: round(mean, 4)
                for variant, run in latest.items()
                if (mean := run.metric_mean(metric)) is not None
            }
            if not row:
                continue
            table[metric] = row
            chooser = min if metric in LOWER_IS_BETTER else max
            best[metric] = chooser(row, key=row.__getitem__)
        return {
            "experiment": experiment,
            "variants": {variant: run.id for variant, run in latest.items()},
            "metrics": table,
            "best": best,
        }

    def ablation(
        self, experiment: str, *, baseline: str = "baseline", metrics: list[str] | None = None
    ) -> dict[str, Any]:
        """Delta of every variant against the baseline variant, with
        significance per metric (latest run per variant, case-level test)."""
        latest = self._latest_per_variant(experiment)
        comparison = self.compare(experiment, metrics=metrics, latest=latest)
        if baseline not in latest:
            raise EvalError(f"experiment {experiment!r} has no baseline variant {baseline!r}")
        base_run = latest[baseline]
        deltas: dict[str, dict[str, Any]] = {}
        for variant, run in latest.items():
            if variant == baseline:
                continue
            entry: dict[str, Any] = {}
            for metric, row in comparison["metrics"].items():
                if baseline not in row or variant not in row:
                    continue
                test = ab_test(base_run.report, run.report, metric)
                entry[metric] = {
                    "baseline": row[baseline],
                    "variant": row[variant],
                    "delta": round(row[variant] - row[baseline], 4),
                    "p_value": test["p_value"],
                    "significant": test["significant"],
                }
            deltas[variant] = entry
        return {"experiment": experiment, "baseline": baseline, "ablation": deltas}


class Experiment:
    """A production-style A/B over prompt/model/config variants of one app.

    Each variant is evaluated over the same dataset; results are logged to an
    :class:`ExperimentTracker` (in the app's own store) so variants are compared
    on eval metrics **and** cost, with the significance tests this module already
    ships. Returned by :meth:`ContextApp.experiment`.
    """

    def __init__(
        self,
        app: Any,
        name: str,
        *,
        tracker: ExperimentTracker | None = None,
        metrics: list[str] | None = None,
    ) -> None:
        self.app = app
        self.name = name
        self.tracker = tracker or ExperimentTracker(app.store)
        self.metrics = metrics

    async def arun_variant(
        self,
        variant: str,
        dataset: Any,
        *,
        model: str | None = None,
        prompt: Any = None,
        apply: Any = None,
        params: dict[str, Any] | None = None,
        repeats: int = 1,
        repeat_aggregate: str = "mean",
        flake_quarantine: bool = False,
        flake_threshold: float = 0.15,
    ) -> EvalReport:
        from .runners import EvalRunner

        app = self.app
        saved_model = app.model
        had_prompt = hasattr(app, "prompt_spec")
        saved_prompt = getattr(app, "prompt_spec", None)
        try:
            if model is not None:
                app.model = model
            if prompt is not None:
                app.prompt_spec = prompt
            if apply is not None:
                apply(app)
            runner = EvalRunner(
                app,
                metrics=self.metrics,
                repeats=repeats,
                repeat_aggregate=repeat_aggregate,
                flake_quarantine=flake_quarantine,
                flake_threshold=flake_threshold,
            )
            report = await runner.arun(dataset, name=f"{self.name}:{variant}")
        finally:
            app.model = saved_model
            if had_prompt:
                app.prompt_spec = saved_prompt
        merged = dict(params or {})
        if model is not None:
            merged.setdefault("model", model)
        self.tracker.log(self.name, report, variant=variant, params=merged)
        return report

    def run_variant(self, variant: str, dataset: Any, **kwargs: Any) -> EvalReport:
        from ..providers.base import run_sync

        return run_sync(self.arun_variant(variant, dataset, **kwargs))

    def compare(self, *, metrics: list[str] | None = None) -> dict[str, Any]:
        return self.tracker.compare(self.name, metrics=metrics or self.metrics)

    def significance(self, metric: str, *, baseline: str = "baseline") -> dict[str, Any]:
        """Per-variant significance vs the baseline variant on one metric."""
        latest = self.tracker._latest_per_variant(self.name)
        if baseline not in latest:
            raise EvalError(f"experiment {self.name!r} has no baseline variant {baseline!r}")
        base = latest[baseline]
        out: dict[str, Any] = {}
        for variant, run in latest.items():
            if variant == baseline:
                continue
            out[variant] = ab_test(base.report, run.report, metric)
        return out

    def cost(self) -> dict[str, float]:
        """Total cost (USD) per variant's latest run."""
        latest = self.tracker._latest_per_variant(self.name)
        return {variant: round(run.report.total_cost_usd, 6) for variant, run in latest.items()}


# -- statistical significance (pure Python) -----------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    max_iterations, eps, fpmin = 300, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_two_sided_p(t_stat: float, df: float) -> float:
    """Two-sided p-value for Student's t."""
    if df <= 0:
        return 1.0
    x = df / (df + t_stat * t_stat)
    return max(0.0, min(1.0, _betainc(df / 2.0, 0.5, x)))


def _t_critical(df: float, alpha: float) -> float:
    """Two-sided critical t value for ``alpha`` — the inverse of
    :func:`_t_two_sided_p`, found by bisection (it is monotone in ``t``).
    Used to build the confidence interval for the mean difference."""
    if df <= 0:
        return float("inf")
    lo, hi = 0.0, 1000.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _case_values(report: EvalReport, metric: str) -> dict[str, float]:
    return {
        case.case_id: case.metrics[metric]
        for case in report.cases
        if metric in case.metrics
    }


def ab_test(
    report_a: EvalReport,
    report_b: EvalReport,
    metric: str,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Significance test between two eval reports on one metric.

    Uses a paired t-test when the reports share case ids (the usual A/B on
    one dataset), otherwise Welch's unequal-variance t-test. Returns means,
    delta, the test used, t statistic, degrees of freedom, two-sided p-value,
    and whether the difference is significant at ``alpha``.
    """
    values_a = _case_values(report_a, metric)
    values_b = _case_values(report_b, metric)
    if not values_a or not values_b:
        raise EvalError(f"metric {metric!r} missing from one or both reports")
    shared = sorted(set(values_a) & set(values_b))
    paired = len(shared) >= 2 and len(shared) == len(values_a) == len(values_b)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    def variance(xs: list[float], mu: float) -> float:
        return sum((x - mu) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0

    a_list, b_list = list(values_a.values()), list(values_b.values())
    mean_a, mean_b = mean(a_list), mean(b_list)
    delta = mean_b - mean_a

    if paired:
        diffs = [values_b[case_id] - values_a[case_id] for case_id in shared]
        mu = mean(diffs)
        var = variance(diffs, mu)
        n = len(diffs)
        df = float(n - 1)
        std_error = math.sqrt(var / n) if var > 0 else 0.0
        sd = math.sqrt(var)
        effect_size = (mu / sd) if sd > 0 else 0.0  # Cohen's d_z (paired)
        if var == 0.0:
            p_value = 1.0 if mu == 0.0 else 0.0
            t_stat = 0.0 if mu == 0.0 else math.inf
        else:
            t_stat = mu / std_error
            p_value = _t_two_sided_p(abs(t_stat), df)
        test = "paired_t"
    else:
        var_a, var_b = variance(a_list, mean_a), variance(b_list, mean_b)
        n_a, n_b = len(a_list), len(b_list)
        se_sq = var_a / n_a + var_b / n_b
        std_error = math.sqrt(se_sq) if se_sq > 0 else 0.0
        pooled_sd = math.sqrt((var_a + var_b) / 2.0)
        effect_size = (delta / pooled_sd) if pooled_sd > 0 else 0.0  # Cohen's d
        if se_sq == 0.0:
            p_value = 1.0 if delta == 0.0 else 0.0
            t_stat = 0.0 if delta == 0.0 else math.inf
            df = float(n_a + n_b - 2)
        else:
            t_stat = delta / std_error
            df = se_sq**2 / (
                (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
            ) if n_a > 1 and n_b > 1 else float(n_a + n_b - 2)
            p_value = _t_two_sided_p(abs(t_stat), df)
        test = "welch_t"

    # Confidence interval for the mean difference at (1 - alpha).
    half_width = _t_critical(df, alpha) * std_error
    ci_low, ci_high = delta - half_width, delta + half_width

    return {
        "metric": metric,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "delta": round(delta, 4),
        "test": test,
        "t": round(t_stat, 4) if math.isfinite(t_stat) else t_stat,
        "df": round(df, 2),
        "p_value": round(p_value, 6),
        "alpha": alpha,
        "significant": p_value < alpha,
        "effect_size": round(effect_size, 4) if math.isfinite(effect_size) else effect_size,
        "std_error": round(std_error, 6),
        "ci_low": round(ci_low, 4) if math.isfinite(ci_low) else ci_low,
        "ci_high": round(ci_high, 4) if math.isfinite(ci_high) else ci_high,
        "confidence": round(1.0 - alpha, 4),
        "n_a": len(a_list),
        "n_b": len(b_list),
    }
