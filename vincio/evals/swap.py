"""Model-swap regression & the swap gate (1.8).

A model swap is the most common and the riskiest change in production. This
module turns "is the new model safe?" into a gated, statistically-backed answer,
composed entirely from organs Vincio already ships:

* :func:`model_swap_regression` holds prompt / data / config fixed, swaps *only*
  the model, and reports per-metric significance (:func:`~vincio.evals.ab_test`),
  per-case deltas, the cost / latency trade, and the **worst-regressed slices** —
  the body of ``vincio eval regress``.
* :class:`SwapGate` wraps that with the 1.7 :class:`~vincio.evals.ReplayRunner`
  over golden traces, a :class:`~vincio.evals.DriftMonitor` check,
  :func:`~vincio.evals.evaluate_gates`, and **behavioral shape diffs**
  (tool-call rate, refusal rate, output-length distribution), and emits a
  PASS / FAIL migration verdict. A model is promoted into the live path only if
  it clears the gate — the swap is gated, not merely compared.

Everything runs offline against the deterministic mock and recorded provider
cassettes; the flake controls on :class:`~vincio.evals.runners.EvalRunner`
(``repeats`` + quarantine) keep non-mock variance from making the gate noisy.
``@experimental`` on the frozen 1.0 API.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

from pydantic import BaseModel, Field

from ..stability import experimental
from .experiments import Experiment, ab_test
from .metrics import LOWER_IS_BETTER, MetricResult, RunOutput
from .reports import EvalReport, evaluate_gates

__all__ = [
    "SwapRegressionReport",
    "SwapVerdict",
    "SwapGate",
    "model_swap_regression",
    "behavioral_shapes",
    "SHAPE_METRICS",
]

_DEFAULT_METRICS = ["semantic_similarity", "schema_validity", "cost", "latency"]

_REFUSAL_RE = re.compile(
    r"(?i)\b(i (?:can('?|no)t|cannot|am unable|'m unable|won'?t)|i am sorry|i'm sorry|"
    r"as an ai|unable to (?:help|assist|comply)|cannot (?:help|assist|comply)|"
    r"i (?:do not|don'?t) (?:have|provide)|not able to)\b"
)


# ---------------------------------------------------------------------------
# Behavioral shape metrics — descriptive signals that ride the metric pipeline
# so a swap's *shape* (does it call tools? refuse? get longer?) can be diffed.
# ---------------------------------------------------------------------------


def _shape_tool_use(case: Any, run: RunOutput) -> MetricResult:
    calls = 0
    if run.trajectory is not None:
        calls = int(run.trajectory.usage.get("tool_calls", 0) or 0)
    if not calls:
        calls = len([s for s in (run.trajectory.steps if run.trajectory else []) if s.type == "tool"])
    return MetricResult(name="tool_call_rate", value=1.0 if calls else 0.0,
                        details={"tool_calls": calls})


def _shape_refusal(case: Any, run: RunOutput) -> MetricResult:
    refused = bool(_REFUSAL_RE.search(run.output_text))
    return MetricResult(name="refusal_rate", value=1.0 if refused else 0.0)


def _shape_output_length(case: Any, run: RunOutput) -> MetricResult:
    return MetricResult(name="output_length", value=float(len(run.output_text)))


_shape_tool_use.__name__ = "tool_call_rate"
_shape_refusal.__name__ = "refusal_rate"
_shape_output_length.__name__ = "output_length"

SHAPE_METRICS = [_shape_tool_use, _shape_refusal, _shape_output_length]
_SHAPE_NAMES = {"tool_call_rate", "refusal_rate", "output_length"}


def behavioral_shapes(report: EvalReport) -> dict[str, float]:
    """Behavioral shape of a report: tool-call rate, refusal rate, and the
    output-length distribution — the dimensions a model swap most often shifts
    without moving the headline quality metric."""
    lengths = report.metric_values("output_length")
    return {
        "tool_call_rate": round(_mean(report.metric_values("tool_call_rate")), 4),
        "refusal_rate": round(_mean(report.metric_values("refusal_rate")), 4),
        "output_length_mean": round(_mean(lengths), 2),
        "output_length_p50": round(statistics.median(lengths), 2) if lengths else 0.0,
        "n": len(report.cases),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _is_worse(metric: str, delta: float) -> bool:
    """Whether a candidate-minus-baseline ``delta`` is a regression for ``metric``."""
    if metric in LOWER_IS_BETTER:
        return delta > 0
    return delta < 0


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


class SwapRegressionReport(BaseModel):
    """The result of swapping only the model on a fixed dataset."""

    baseline_model: str
    candidate_model: str
    n_cases: int = 0
    metric_tests: dict[str, dict[str, Any]] = Field(default_factory=dict)
    regressions: list[str] = Field(default_factory=list)  # metrics significantly worse
    cost: dict[str, float] = Field(default_factory=dict)  # baseline / candidate / delta / ratio
    latency: dict[str, float] = Field(default_factory=dict)
    behavioral: dict[str, Any] = Field(default_factory=dict)  # baseline / candidate shapes + deltas
    worst_slices: list[dict[str, Any]] = Field(default_factory=list)
    worst_cases: list[dict[str, Any]] = Field(default_factory=list)
    flaky_excluded: int = 0

    @property
    def regressed(self) -> bool:
        return bool(self.regressions)

    def summary(self) -> dict[str, Any]:
        return {
            "baseline_model": self.baseline_model,
            "candidate_model": self.candidate_model,
            "n_cases": self.n_cases,
            "regressed": self.regressed,
            "regressions": self.regressions,
            "cost": self.cost,
            "latency": self.latency,
            "flaky_excluded": self.flaky_excluded,
        }


class SwapVerdict(BaseModel):
    """PASS / FAIL verdict for promoting a model into the live path."""

    passed: bool
    baseline_model: str
    candidate_model: str
    reason: str = ""
    regression: SwapRegressionReport | None = None
    drift: dict[str, Any] = Field(default_factory=dict)
    gates: dict[str, Any] = Field(default_factory=dict)
    replay: dict[str, Any] | None = None
    behavioral_regressions: list[str] = Field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "passed": self.passed,
            "baseline_model": self.baseline_model,
            "candidate_model": self.candidate_model,
            "reason": self.reason,
        }
        if self.regression is not None:
            out["regression"] = self.regression.summary()
        if self.replay is not None:
            out["replay"] = self.replay
        if self.gates:
            out["gates_failed"] = [k for k, v in self.gates.items() if not v.get("passed")]
        return out


# ---------------------------------------------------------------------------
# Variant running (shared by the regression flow and the gate)
# ---------------------------------------------------------------------------


def _combined_metrics(metrics: list[Any] | None) -> list[Any]:
    base = list(metrics) if metrics else list(_DEFAULT_METRICS)
    have = {m if isinstance(m, str) else getattr(m, "__name__", "") for m in base}
    return base + [m for m in SHAPE_METRICS if m.__name__ not in have]


async def _dual_variant_reports(
    app: Any,
    dataset: Any,
    *,
    baseline_model: str,
    candidate_model: str,
    metrics: list[Any] | None = None,
    repeats: int = 1,
    flake_quarantine: bool = False,
    flake_threshold: float = 0.15,
    candidate_apply: Any = None,
    experiment_name: str = "model_swap",
) -> tuple[EvalReport, EvalReport]:
    """Run the same dataset on baseline vs candidate model, prompt/config fixed."""
    experiment = Experiment(app, experiment_name, metrics=_combined_metrics(metrics))
    base_report = await experiment.arun_variant(
        "baseline", dataset, model=baseline_model,
        repeats=repeats, flake_quarantine=flake_quarantine, flake_threshold=flake_threshold,
    )
    cand_report = await experiment.arun_variant(
        "candidate", dataset, model=candidate_model, apply=candidate_apply,
        repeats=repeats, flake_quarantine=flake_quarantine, flake_threshold=flake_threshold,
    )
    return base_report, cand_report


def _regression_from_reports(
    base_report: EvalReport,
    cand_report: EvalReport,
    *,
    baseline_model: str,
    candidate_model: str,
    metrics: list[str],
    alpha: float,
    slice_prefix: str,
    quality_metric: str,
) -> SwapRegressionReport:
    metric_tests: dict[str, dict[str, Any]] = {}
    regressions: list[str] = []
    base_summary = base_report.summary()
    cand_summary = cand_report.summary()
    for metric in metrics:
        if metric in _SHAPE_NAMES or metric in ("cost", "latency"):
            continue
        if metric not in base_summary or metric not in cand_summary:
            continue
        try:
            test = ab_test(base_report, cand_report, metric, alpha=alpha)
        except Exception:  # noqa: BLE001 - a missing/var-free metric is not fatal
            continue
        metric_tests[metric] = test
        if test["significant"] and _is_worse(metric, test["delta"]):
            regressions.append(metric)

    base_cost, cand_cost = _mean(base_report.metric_values("cost")), _mean(cand_report.metric_values("cost"))
    base_lat, cand_lat = _mean(base_report.metric_values("latency")), _mean(cand_report.metric_values("latency"))
    cost = {
        "baseline": round(base_cost, 8), "candidate": round(cand_cost, 8),
        "delta": round(cand_cost - base_cost, 8),
        "ratio": round(cand_cost / base_cost, 4) if base_cost > 0 else 0.0,
    }
    latency = {
        "baseline": round(base_lat, 2), "candidate": round(cand_lat, 2),
        "delta": round(cand_lat - base_lat, 2),
    }

    base_shapes, cand_shapes = behavioral_shapes(base_report), behavioral_shapes(cand_report)
    behavioral = {
        "baseline": base_shapes, "candidate": cand_shapes,
        "delta": {
            k: round(cand_shapes[k] - base_shapes[k], 4)
            for k in ("tool_call_rate", "refusal_rate", "output_length_mean")
        },
    }

    worst_slices = _worst_slices(base_report, cand_report, quality_metric, slice_prefix)
    worst_cases = _worst_cases(base_report, cand_report, quality_metric)

    return SwapRegressionReport(
        baseline_model=baseline_model,
        candidate_model=candidate_model,
        n_cases=len(cand_report.cases),
        metric_tests=metric_tests,
        regressions=regressions,
        cost=cost,
        latency=latency,
        behavioral=behavioral,
        worst_slices=worst_slices,
        worst_cases=worst_cases,
        flaky_excluded=int(cand_report.metadata.get("flaky_excluded_from_gates", 0)),
    )


def _worst_slices(
    base: EvalReport, cand: EvalReport, metric: str, prefix: str
) -> list[dict[str, Any]]:
    base_slices, cand_slices = base.slice_by_tag(prefix), cand.slice_by_tag(prefix)
    rows: list[dict[str, Any]] = []
    for value in sorted(set(base_slices) & set(cand_slices)):
        b = _mean(base_slices[value].metric_values(metric))
        c = _mean(cand_slices[value].metric_values(metric))
        rows.append({"slice": value, "baseline": round(b, 4), "candidate": round(c, 4),
                     "delta": round(c - b, 4)})
    worse_first = metric in LOWER_IS_BETTER
    rows.sort(key=lambda r: r["delta"], reverse=worse_first)
    return [r for r in rows if _is_worse(metric, r["delta"])][:5]


def _worst_cases(base: EvalReport, cand: EvalReport, metric: str) -> list[dict[str, Any]]:
    base_by_id = {c.case_id: c for c in base.cases}
    rows: list[dict[str, Any]] = []
    for case in cand.cases:
        b = base_by_id.get(case.case_id)
        if b is None or metric not in case.metrics or metric not in b.metrics:
            continue
        delta = case.metrics[metric] - b.metrics[metric]
        if _is_worse(metric, delta):
            rows.append({"case_id": case.case_id, "baseline": round(b.metrics[metric], 4),
                         "candidate": round(case.metrics[metric], 4), "delta": round(delta, 4)})
    rows.sort(key=lambda r: r["delta"], reverse=metric in LOWER_IS_BETTER)
    return rows[:10]


@experimental(since="1.8")
async def model_swap_regression(
    app: Any,
    dataset: Any,
    *,
    baseline_model: str | None = None,
    candidate_model: str,
    metrics: list[str] | None = None,
    quality_metric: str = "semantic_similarity",
    alpha: float = 0.05,
    repeats: int = 1,
    flake_quarantine: bool = True,
    flake_threshold: float = 0.15,
    slice_prefix: str = "lang:",
) -> SwapRegressionReport:
    """Swap only the model on a fixed dataset and report a statistically grounded
    regression analysis (the body of ``vincio eval regress``)."""
    baseline_model = baseline_model or app.model
    base_report, cand_report = await _dual_variant_reports(
        app, dataset, baseline_model=baseline_model, candidate_model=candidate_model,
        metrics=metrics, repeats=repeats, flake_quarantine=flake_quarantine,
        flake_threshold=flake_threshold,
    )
    measured = sorted(set(base_report.summary()) | set(cand_report.summary()))
    return _regression_from_reports(
        base_report, cand_report, baseline_model=baseline_model, candidate_model=candidate_model,
        metrics=metrics or measured, alpha=alpha, slice_prefix=slice_prefix,
        quality_metric=quality_metric,
    )


@experimental(since="1.8")
class SwapGate:
    """Gate a model/provider change on replayed golden traces + an eval/cost/
    latency/behavioral diff with statistical backing.

    Assemble it from a :class:`~vincio.core.app.ContextApp`, then call
    :meth:`evaluate` with a dataset (significance + cost/latency + behavioral
    shape diff) and/or golden traces (behavioral replay). The verdict PASSes only
    when no quality metric significantly regresses, no configured gate fails, no
    quality drift is detected, and no behavioral shape (refusal rate) regresses
    past ``behavior_threshold`` — so a model is promoted into the live path only
    if it clears the gate.
    """

    def __init__(
        self,
        app: Any,
        *,
        metrics: list[str] | None = None,
        quality_metric: str = "semantic_similarity",
        gates: dict[str, str] | None = None,
        alpha: float = 0.05,
        drift_threshold: float = 0.1,
        behavior_threshold: float = 0.2,
        repeats: int = 1,
        flake_quarantine: bool = True,
    ) -> None:
        self.app = app
        self.metrics = metrics
        self.quality_metric = quality_metric
        self.gates = gates or {}
        self.alpha = alpha
        self.drift_threshold = drift_threshold
        self.behavior_threshold = behavior_threshold
        self.repeats = repeats
        self.flake_quarantine = flake_quarantine

    async def evaluate(
        self,
        *,
        candidate_model: str,
        baseline_model: str | None = None,
        dataset: Any = None,
        traces: list[Any] | None = None,
        pin_tools: bool = True,
        candidate_apply: Any = None,
    ) -> SwapVerdict:
        if dataset is None and not traces:
            raise ValueError("SwapGate.evaluate requires a dataset and/or golden traces")
        baseline_model = baseline_model or self.app.model

        regression: SwapRegressionReport | None = None
        drift: dict[str, Any] = {}
        gates: dict[str, Any] = {}
        behavioral_regressions: list[str] = []
        reasons: list[str] = []

        if dataset is not None:
            base_report, cand_report = await _dual_variant_reports(
                self.app, dataset, baseline_model=baseline_model, candidate_model=candidate_model,
                metrics=self.metrics, repeats=self.repeats, flake_quarantine=self.flake_quarantine,
                candidate_apply=candidate_apply, experiment_name="swap_gate",
            )
            measured = sorted(set(base_report.summary()) | set(cand_report.summary()))
            regression = _regression_from_reports(
                base_report, cand_report, baseline_model=baseline_model,
                candidate_model=candidate_model, metrics=self.metrics or measured,
                alpha=self.alpha, slice_prefix="lang:", quality_metric=self.quality_metric,
            )
            if regression.regressions:
                reasons.append("quality regression: " + ", ".join(regression.regressions))

            # Quality drift on the headline metric.
            if self.quality_metric in cand_report.summary():
                from .drift import DriftMonitor

                monitor = DriftMonitor(score_threshold=self.drift_threshold)
                monitor.set_score_baseline(
                    self.quality_metric, base_report.metric_values(self.quality_metric)
                )
                drift_report = monitor.check_scores(
                    self.quality_metric, cand_report.metric_values(self.quality_metric)
                )
                drift = drift_report.model_dump()
                if drift_report.drifted and _is_worse(self.quality_metric, drift_report.delta):
                    reasons.append(
                        f"quality drift: {self.quality_metric} {drift_report.delta:+.4f}"
                    )

            # Configured gates run over the candidate report.
            if self.gates:
                gates = evaluate_gates(cand_report, self.gates)
                failed = [k for k, v in gates.items() if not v.get("passed")]
                if failed:
                    reasons.append("gate failure: " + ", ".join(failed))

            # Behavioral shape regression (refusal rate climbing materially).
            refusal_delta = regression.behavioral.get("delta", {}).get("refusal_rate", 0.0)
            if refusal_delta > self.behavior_threshold:
                behavioral_regressions.append(f"refusal_rate +{refusal_delta:.2f}")
                reasons.append(f"behavioral regression: refusal_rate +{refusal_delta:.2f}")

        replay: dict[str, Any] | None = None
        if traces:
            from .replay import ReplayRunner

            app = self.app
            saved_model = app.model
            try:
                app.model = candidate_model
                replay_result = await ReplayRunner(app).replay(traces, pin_tools=pin_tools)
            finally:
                app.model = saved_model
            replay = replay_result.summary()
            # A status regression (candidate errors where the baseline succeeded)
            # is a hard fail; pure text divergence between models is expected.
            status_regressions = sum(1 for c in replay_result.cases if not c.status_match)
            if status_regressions:
                reasons.append(f"replay status regressions: {status_regressions}")
            replay["status_regressions"] = status_regressions

        passed = not reasons
        return SwapVerdict(
            passed=passed,
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            reason="; ".join(reasons) if reasons else "no regression detected — safe to promote",
            regression=regression,
            drift=drift,
            gates=gates,
            replay=replay,
            behavioral_regressions=behavioral_regressions,
        )
