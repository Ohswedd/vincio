"""Deterministic statistical verification kernels.

The :mod:`~vincio.verify.kernels` set certifies *arithmetic*-class claims — a sum,
a unit conversion, a date arithmetic, a constraint, a schema, a citation. A data
answer makes a different class of claim: a **statistical** conclusion drawn from a
series of cited cells — a trend, a correlation, a confidence interval, a forecast.
This module gives each of those a checkable certificate the way an arithmetic one
already has, on the same :class:`~vincio.verify.certificates.Certificate` surface.

Four kernels, each a :class:`~vincio.verify.certificates.ReasoningVerifier` that
**recomputes** the stated statistic from the cited series and refutes a mismatch:

* :class:`TrendVerifier` — recomputes an ordinary-least-squares slope / intercept
  and its goodness-of-fit (``R²``) from the cited cells; a stated trend that the
  data does not bear out is refuted.
* :class:`CorrelationVerifier` — recomputes a Pearson correlation and, for a claim
  that asserts **causation**, refuses a *correlation-stated-as-causation* claim
  that declares no controls or randomization, and refutes a controlled claim whose
  association **vanishes once the declared confounders are partialled out** (the
  partial correlation collapses) — the spurious-causation refutation.
* :class:`IntervalVerifier` — recomputes a stated confidence interval for a mean,
  or a regression prediction interval at a stated point, from the cited series.
* :class:`ForecastVerifier` — re-runs a declared **deterministic** forecast model
  (naïve, mean, drift, linear-trend, moving-average, or simple exponential
  smoothing) over the cited series and checks the projection.

Every statistic is recomputed by a pure, offline, dependency-free routine — the
ordinary-least-squares fit, the Pearson and partial correlations, and the
Student-t quantiles (via the regularized incomplete beta function) are all
implemented here over the standard library. A statistic is **bound to the cited
cells**: a :class:`CitedSeries` carries the source-cell references its values came
from, and a kernel refutes a series whose values do not match the cells it claims
to cite, so a smuggled number cannot pass. The optional CAS backend behind
``vincio[verify]`` (:class:`~vincio.verify.smt.CasTrendVerifier`) re-discharges the
trend fit with exact rational arithmetic for the cases that warrant it; the
kernels here are the default and need no extra.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field

from .certificates import Check, VerificationContext

__all__ = [
    "CellRef",
    "CitedSeries",
    "StatisticalClaim",
    "TrendClaim",
    "CorrelationClaim",
    "IntervalClaim",
    "ForecastClaim",
    "TrendVerifier",
    "CorrelationVerifier",
    "IntervalVerifier",
    "ForecastVerifier",
    "statistical_verifiers",
    "ols_fit",
    "pearson_r",
    "partial_correlation",
    "mean_confidence_interval",
    "prediction_interval",
    "forecast",
    "student_t_ppf",
    "OlsFit",
    "ForecastModel",
]

# Default comparison tolerance for a recomputed-vs-claimed statistic. A stated
# statistic is almost always rounded ("R²=0.98", "slope ≈ 12.5"), so the default
# accepts a reasonably-rounded claim while a wrong one is off by far more.
_REL_TOL = 1e-2
_ABS_TOL = 1e-3


def _approx(computed: float, claimed: float, rel_tol: float, abs_tol: float) -> bool:
    """True when ``computed`` and ``claimed`` agree within a relative+absolute band."""
    if math.isnan(computed) or math.isnan(claimed):
        return False
    return abs(computed - claimed) <= abs_tol + rel_tol * abs(claimed)


# --------------------------------------------------------------------------- #
# Deterministic statistics core (pure, offline, dependency-free)               #
# --------------------------------------------------------------------------- #

ForecastModel = Literal["naive", "mean", "drift", "linear", "moving_average", "ses"]


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    """Sample standard deviation (Bessel-corrected, ``ddof=1``)."""
    n = len(values)
    if n < 2:
        raise ValueError("standard deviation needs at least two points")
    mu = _mean(values)
    return math.sqrt(math.fsum((v - mu) ** 2 for v in values) / (n - 1))


class OlsFit(BaseModel):
    """The result of an ordinary-least-squares fit ``y = slope·x + intercept``.

    Carries the fitted ``slope`` and ``intercept``, the coefficient of
    determination ``r_squared`` (goodness-of-fit), the residual sum of squares
    ``sse``, the count ``n``, and the predictor mean and spread (``x_mean`` /
    ``sxx``) a prediction interval needs — every quantity recomputed deterministically.
    """

    slope: float
    intercept: float
    r_squared: float
    sse: float
    n: int
    x_mean: float
    sxx: float

    def predict(self, x: float) -> float:
        """The fitted value ``slope·x + intercept`` at a predictor value."""
        return self.slope * x + self.intercept


def ols_fit(xs: list[float], ys: list[float]) -> OlsFit:
    """Fit ``y = slope·x + intercept`` by ordinary least squares, deterministically.

    Returns an :class:`OlsFit` with the slope, intercept, ``R²`` goodness-of-fit,
    and the residual / spread quantities an interval needs. Raises
    :class:`ValueError` when there are fewer than two points or the predictor has
    no spread (a vertical fit is undefined).
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError("xs and ys must be the same length")
    if n < 2:
        raise ValueError("a trend needs at least two points")
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    sxx = math.fsum((x - x_mean) ** 2 for x in xs)
    if sxx <= 0.0:
        raise ValueError("predictor has no spread; slope is undefined")
    sxy = math.fsum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    syy = math.fsum((y - y_mean) ** 2 for y in ys)
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    sse = math.fsum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    # R² = 1 - SSE/SST; a flat response (syy == 0) is perfectly fit by a constant.
    r_squared = 1.0 if syy <= 0.0 else max(0.0, 1.0 - sse / syy)
    return OlsFit(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        sse=sse,
        n=n,
        x_mean=x_mean,
        sxx=sxx,
    )


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """The Pearson product-moment correlation coefficient of two series.

    Raises :class:`ValueError` when the series differ in length, have fewer than
    two points, or either has no variance (the correlation is then undefined).
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError("series must be the same length")
    if n < 2:
        raise ValueError("correlation needs at least two points")
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    sxx = math.fsum((x - x_mean) ** 2 for x in xs)
    syy = math.fsum((y - y_mean) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        raise ValueError("a series has no variance; correlation is undefined")
    sxy = math.fsum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    return sxy / math.sqrt(sxx * syy)


def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve ``A·b = rhs`` by Gaussian elimination with partial pivoting."""
    n = len(matrix)
    aug = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular system; cannot solve the normal equations")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / pivot_val
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def _ss_centered(values: list[float]) -> float:
    """Total sum of squares about the mean."""
    mu = _mean(values)
    return math.fsum((v - mu) ** 2 for v in values)


def _residuals_on(target: list[float], controls: list[list[float]]) -> list[float]:
    """Residuals of ``target`` after regressing it on the control series (+ intercept)."""
    n = len(target)
    design = [[1.0] + [controls[k][i] for k in range(len(controls))] for i in range(n)]
    p = len(design[0])
    # Normal equations: (Xᵀ X) β = Xᵀ y.
    xtx = [[math.fsum(design[i][a] * design[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    xty = [math.fsum(design[i][a] * target[i] for i in range(n)) for a in range(p)]
    beta = _solve_linear(xtx, xty)
    return [target[i] - math.fsum(beta[a] * design[i][a] for a in range(p)) for i in range(n)]


def partial_correlation(
    xs: list[float], ys: list[float], controls: list[list[float]]
) -> float:
    """The partial correlation of ``xs`` and ``ys`` controlling for the controls.

    Computes the correlation of the residuals of ``xs`` and ``ys`` after each is
    regressed on the control series — the association that **survives** removing
    the confounders' linear influence. A magnitude near zero means the raw
    correlation is explained by the controls, which is the confounding signal the
    causal kernel reads. With no controls it is the ordinary Pearson correlation.
    """
    if not controls:
        return pearson_r(xs, ys)
    rx = _residuals_on(xs, controls)
    ry = _residuals_on(ys, controls)
    # When the controls explain essentially all of either series' variance, the
    # residual is numerical noise (a perfectly-collinear confounder); no meaningful
    # association survives, so the partial correlation is conventionally zero rather
    # than a spurious correlation of two near-zero residual vectors.
    sxx, syy = _ss_centered(xs), _ss_centered(ys)
    if (sxx <= 0.0 or _ss_centered(rx) <= 1e-9 * sxx) or (
        syy <= 0.0 or _ss_centered(ry) <= 1e-9 * syy
    ):
        return 0.0
    return pearson_r(rx, ry)


# -- Student-t quantiles via the regularized incomplete beta function --------- #


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta (Lentz's method)."""
    max_iter = 300
    eps = 1e-14
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
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


def _betai(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function ``Iₓ(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """The cumulative distribution function of Student's t with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)  # P(|T| > |t|) / 2
    return 1.0 - tail if t > 0.0 else tail


def student_t_ppf(p: float, df: float) -> float:
    """The inverse CDF (quantile) of Student's t — solved by bisection on the CDF.

    ``student_t_ppf(0.975, df)`` is the two-sided 95 % critical value; it tends to
    the normal ``1.959964`` as ``df`` grows. Raises :class:`ValueError` for a
    probability outside ``(0, 1)``.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be in (0, 1)")
    if p == 0.5:
        return 0.0
    lo, hi = -1.0e6, 1.0e6
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t_critical(df: float, confidence: float) -> float:
    """The two-sided Student-t critical value for a confidence level."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return student_t_ppf(0.5 * (1.0 + confidence), df)


def mean_confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """The Student-t confidence interval for the mean of a sample.

    ``x̄ ± t · s/√n`` with ``t`` the two-sided critical value at ``n − 1`` degrees
    of freedom. Raises :class:`ValueError` for fewer than two points.
    """
    n = len(values)
    if n < 2:
        raise ValueError("a confidence interval needs at least two points")
    x_bar = _mean(values)
    half = t_critical(n - 1, confidence) * _sample_std(values) / math.sqrt(n)
    return (x_bar - half, x_bar + half)


def prediction_interval(
    xs: list[float], ys: list[float], at: float, confidence: float = 0.95
) -> tuple[float, float, float]:
    """The regression prediction interval for a new observation at ``at``.

    Fits ``y = slope·x + intercept`` and returns ``(lower, upper, ŷ)`` where the
    half-width is ``t · s · √(1 + 1/n + (x₀ − x̄)² / Sxx)`` with ``s²`` the residual
    variance at ``n − 2`` degrees of freedom. Raises :class:`ValueError` for fewer
    than three points (the residual variance is then undefined).
    """
    n = len(xs)
    if n < 3:
        raise ValueError("a prediction interval needs at least three points")
    fit = ols_fit(xs, ys)
    y_hat = fit.predict(at)
    s = math.sqrt(fit.sse / (n - 2))
    se = s * math.sqrt(1.0 + 1.0 / n + (at - fit.x_mean) ** 2 / fit.sxx)
    half = t_critical(n - 2, confidence) * se
    return (y_hat - half, y_hat + half, y_hat)


def forecast(
    model: ForecastModel,
    ys: list[float],
    *,
    horizon: int,
    xs: list[float] | None = None,
    params: dict[str, Any] | None = None,
) -> list[float]:
    """Re-run a declared deterministic forecast model over a series.

    Produces ``horizon`` projected values for one of the classic deterministic
    benchmark forecasters — ``naive`` (last value carried forward), ``mean`` (the
    series mean), ``drift`` (last value plus the average per-step change),
    ``linear`` (the OLS trend extrapolated along the index), ``moving_average``
    (the mean of the last ``window`` points), or ``ses`` (simple exponential
    smoothing with smoothing factor ``alpha``). Deterministic given the inputs.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one step")
    n = len(ys)
    if n < 1:
        raise ValueError("a forecast needs at least one observation")
    opts = params or {}
    if model == "naive":
        return [ys[-1]] * horizon
    if model == "mean":
        return [_mean(ys)] * horizon
    if model == "drift":
        if n < 2:
            raise ValueError("the drift model needs at least two observations")
        step = (ys[-1] - ys[0]) / (n - 1)
        return [ys[-1] + step * (h + 1) for h in range(horizon)]
    if model == "moving_average":
        window = int(opts.get("window", min(n, 3)))
        if window < 1 or window > n:
            raise ValueError("moving-average window must be between 1 and the series length")
        return [_mean(ys[-window:])] * horizon
    if model == "ses":
        alpha = float(opts.get("alpha", 0.5))
        if not 0.0 < alpha <= 1.0:
            raise ValueError("the smoothing factor alpha must be in (0, 1]")
        level = ys[0]
        for value in ys[1:]:
            level = alpha * value + (1.0 - alpha) * level
        return [level] * horizon
    if model == "linear":
        index = xs if xs is not None else [float(i) for i in range(n)]
        if len(index) != n:
            raise ValueError("xs must match the series length")
        fit = ols_fit(index, ys)
        step = (index[-1] - index[0]) / (n - 1) if n >= 2 else 1.0
        return [fit.predict(index[-1] + step * (h + 1)) for h in range(horizon)]
    raise ValueError(f"unknown forecast model {model!r}")


# --------------------------------------------------------------------------- #
# Typed cited series & claims                                                  #
# --------------------------------------------------------------------------- #


class CellRef(BaseModel):
    """A reference to one source cell a series value came from.

    ``ref`` is a stable, opaque cell locator (e.g. the data plane's
    ``table#r<row>!<column>``); ``value`` binds the cell's value, so a kernel can
    confirm a series is really the cited cells and refuse a smuggled number. It is
    the verify plane's data-plane-agnostic mirror of
    :class:`~vincio.data.provenance.CellCitation`, so any object exposing ``.ref``
    and ``.value`` slots straight in via :meth:`CitedSeries.from_cells`.
    """

    ref: str
    value: float


class CitedSeries(BaseModel):
    """A named numeric series bound to the source cells it was read from.

    ``values`` are the observations; ``index`` is the optional predictor (defaults
    to ``0, 1, 2, …``); ``citations`` are the :class:`CellRef`\\ s the values came
    from. A statistic recomputed from this series is **bound to the cited cells**:
    :meth:`is_bound` confirms each value matches the cell it claims to cite, and a
    kernel refutes an unbound series so a value swapped after citation cannot pass.
    """

    name: str = "series"
    values: list[float] = Field(default_factory=list)
    index: list[float] | None = None
    citations: list[CellRef] = Field(default_factory=list)

    @property
    def n(self) -> int:
        """The number of observations."""
        return len(self.values)

    def xs(self) -> list[float]:
        """The predictor values (the explicit index, or ``0 … n−1``)."""
        if self.index is not None:
            return list(self.index)
        return [float(i) for i in range(len(self.values))]

    def ys(self) -> list[float]:
        """The observed values."""
        return list(self.values)

    def is_bound(self, *, abs_tol: float = 1e-6) -> bool:
        """True when the series is bound to its cited cells (or cites none).

        With no citations the binding is not asserted (a bare series is still
        checkable); with citations, every value must match the cell it cites, so a
        value edited after it was cited makes the series unbound.
        """
        if not self.citations:
            return True
        if len(self.citations) != len(self.values):
            return False
        return all(
            abs(c.value - v) <= abs_tol
            for c, v in zip(self.citations, self.values, strict=True)
        )

    @classmethod
    def from_pairs(
        cls, pairs: list[tuple[float, float]], *, name: str = "series"
    ) -> CitedSeries:
        """Build a series from ``(x, y)`` pairs (no cell binding)."""
        return cls(name=name, index=[x for x, _ in pairs], values=[y for _, y in pairs])

    @classmethod
    def from_cells(
        cls, cells: list[Any], *, name: str = "series", index: list[float] | None = None
    ) -> CitedSeries:
        """Build a cited series from cell objects exposing ``.ref`` and ``.value``.

        Accepts the data plane's :class:`~vincio.data.provenance.CellCitation`
        (or any duck-typed equivalent), so a cell-cited query column becomes a
        verifiable series in one call. The binding is carried, so a later edit to a
        cited cell is caught.
        """
        refs = [CellRef(ref=str(getattr(c, "ref", "")), value=float(c.value)) for c in cells]
        return cls(name=name, index=index, values=[r.value for r in refs], citations=refs)


class StatisticalClaim(BaseModel):
    """Base of the analytical-claim family the statistical kernels certify.

    ``label`` names the claim for the certificate, and ``rel_tol`` / ``abs_tol``
    set the band a recomputed statistic must fall in to confirm the claim — wide
    enough to accept a reasonably-rounded figure, narrow enough that a wrong one is
    refuted. The concrete subclasses carry the series and the stated statistic.
    """

    label: str = ""
    rel_tol: float = _REL_TOL
    abs_tol: float = _ABS_TOL


class TrendClaim(StatisticalClaim):
    """A stated linear trend over a cited series.

    Recomputed by :class:`TrendVerifier`: the OLS ``slope`` (and optional
    ``intercept`` and ``r_squared`` goodness-of-fit) must match the fit, and the
    optional qualitative ``direction`` (``"increasing"`` / ``"decreasing"`` /
    ``"flat"``) must match the fitted slope's sign.
    """

    series: CitedSeries
    slope: float | None = None
    intercept: float | None = None
    r_squared: float | None = None
    direction: Literal["increasing", "decreasing", "flat"] | None = None


class CorrelationClaim(StatisticalClaim):
    """A stated correlation between two cited series, optionally asserting causation.

    Recomputed by :class:`CorrelationVerifier`: the Pearson correlation of ``x``
    and ``y`` must match ``r``. When ``causal`` is set, the claim asserts ``x``
    *causes* ``y``; the kernel then demands a warrant — either a randomized
    ``design="experiment"``, or declared ``controls`` **with** their ``control_series``
    so the partial correlation can be recomputed. A causal claim with no warrant is
    refuted as correlation-stated-as-causation, and a controlled claim whose
    partial correlation collapses below ``confound_threshold`` is refuted as
    confounded.
    """

    x: CitedSeries
    y: CitedSeries
    r: float | None = None
    causal: bool = False
    design: Literal["observational", "experiment"] = "observational"
    controls: list[str] = Field(default_factory=list)
    control_series: list[CitedSeries] = Field(default_factory=list)
    confound_threshold: float = 0.2


class IntervalClaim(StatisticalClaim):
    """A stated interval over a cited series.

    Recomputed by :class:`IntervalVerifier`. With ``kind="mean"`` it is the
    Student-t confidence interval for the series mean; with ``kind="prediction"``
    it is the OLS prediction interval for a new observation at ``at``. ``lower`` and
    ``upper`` are the stated bounds and ``confidence`` the stated level.
    """

    series: CitedSeries
    lower: float
    upper: float
    confidence: float = 0.95
    kind: Literal["mean", "prediction"] = "mean"
    at: float | None = None


class ForecastClaim(StatisticalClaim):
    """A stated projection from a declared deterministic forecast model.

    Recomputed by :class:`ForecastVerifier`: re-running ``model`` (with optional
    ``params``) over the cited series for ``len(predictions)`` steps must reproduce
    the stated ``predictions`` (element-wise, within tolerance).
    """

    series: CitedSeries
    model: ForecastModel
    predictions: list[float]
    params: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# The four kernels                                                             #
# --------------------------------------------------------------------------- #


def _claims_of(
    explicit: list[Any] | None, context: VerificationContext, claim_type: type
) -> list[Any]:
    """The claims of one type, from the constructor list or the context."""
    if explicit:
        return [c for c in explicit if isinstance(c, claim_type)]
    return [c for c in context.statistical_claims if isinstance(c, claim_type)]


def _binding_check(series: CitedSeries, kind: str) -> Check | None:
    """A refutation when a series is not bound to its cited cells, else ``None``."""
    if series.is_bound():
        return None
    return Check(
        name="series_binding",
        kind=kind,
        status="refuted",
        detail=f"series {series.name!r} values do not match the cells they cite",
        evidence={"series": series.name},
    )


class TrendVerifier:
    """Recomputes a stated linear trend and its goodness-of-fit from cited cells.

    For each :class:`TrendClaim`, fits ``y = slope·x + intercept`` by ordinary
    least squares over the cited series and confirms the stated slope, intercept,
    ``R²``, and qualitative direction — a stated trend the data does not bear out is
    refuted. Reads claims passed at construction or from
    :attr:`VerificationContext.statistical_claims`; with none, the check is
    inapplicable.
    """

    kind = "trend"

    def __init__(self, claims: list[TrendClaim] | None = None) -> None:
        self._claims = list(claims) if claims is not None else None

    def check(self, answer: Any, context: VerificationContext) -> list[Check]:
        claims = _claims_of(self._claims, context, TrendClaim)
        if not claims:
            return [Check(name="trend", kind=self.kind, status="inapplicable",
                          detail="no trend claim supplied")]
        checks: list[Check] = []
        for claim in claims:
            binding = _binding_check(claim.series, self.kind)
            if binding is not None:
                checks.append(binding)
                continue
            label = claim.label or "trend"
            try:
                fit = ols_fit(claim.series.xs(), claim.series.ys())
            except ValueError as exc:
                checks.append(Check(name=label, kind=self.kind, status="refuted",
                                    detail=f"cannot recompute the stated trend: {exc}"))
                continue
            checks.extend(self._check_fit(claim, fit, label))
        return checks

    def _check_fit(self, claim: TrendClaim, fit: OlsFit, label: str) -> list[Check]:
        out: list[Check] = []
        if claim.slope is not None:
            ok = _approx(fit.slope, claim.slope, claim.rel_tol, claim.abs_tol)
            out.append(Check(
                name=f"{label}:slope", kind=self.kind,
                status="verified" if ok else "refuted",
                detail=f"slope = {fit.slope:g}, claimed {claim.slope:g}",
                evidence={"computed": fit.slope, "claimed": claim.slope},
            ))
        if claim.intercept is not None:
            ok = _approx(fit.intercept, claim.intercept, claim.rel_tol, claim.abs_tol)
            out.append(Check(
                name=f"{label}:intercept", kind=self.kind,
                status="verified" if ok else "refuted",
                detail=f"intercept = {fit.intercept:g}, claimed {claim.intercept:g}",
                evidence={"computed": fit.intercept, "claimed": claim.intercept},
            ))
        if claim.r_squared is not None:
            ok = _approx(fit.r_squared, claim.r_squared, claim.rel_tol, claim.abs_tol)
            out.append(Check(
                name=f"{label}:r_squared", kind=self.kind,
                status="verified" if ok else "refuted",
                detail=f"R² = {fit.r_squared:g}, claimed {claim.r_squared:g}",
                evidence={"computed": fit.r_squared, "claimed": claim.r_squared},
            ))
        if claim.direction is not None:
            slope_eps = claim.abs_tol
            actual = ("increasing" if fit.slope > slope_eps
                      else "decreasing" if fit.slope < -slope_eps else "flat")
            ok = actual == claim.direction
            out.append(Check(
                name=f"{label}:direction", kind=self.kind,
                status="verified" if ok else "refuted",
                detail=f"trend is {actual} (slope {fit.slope:g}), claimed {claim.direction}",
                evidence={"computed": actual, "claimed": claim.direction},
            ))
        if not out:
            out.append(Check(name=label, kind=self.kind, status="inapplicable",
                             detail="trend claim stated no slope, intercept, R², or direction"))
        return out


class CorrelationVerifier:
    """Recomputes a correlation and refutes correlation-stated-as-causation.

    For each :class:`CorrelationClaim`, recomputes the Pearson correlation of the
    two cited series and confirms the stated ``r``. When the claim asserts
    causation it demands a warrant: a randomized design, or declared controls with
    their series so the **partial correlation** can be recomputed. A causal claim
    with no warrant is refuted, and a controlled claim whose association collapses
    once the confounders are partialled out (``|partial r| < confound_threshold``)
    is refuted as confounded — the spurious-causation refutation. Reads claims
    passed at construction or from the context; with none, inapplicable.
    """

    kind = "correlation"

    def __init__(self, claims: list[CorrelationClaim] | None = None) -> None:
        self._claims = list(claims) if claims is not None else None

    def check(self, answer: Any, context: VerificationContext) -> list[Check]:
        claims = _claims_of(self._claims, context, CorrelationClaim)
        if not claims:
            return [Check(name="correlation", kind=self.kind, status="inapplicable",
                          detail="no correlation claim supplied")]
        checks: list[Check] = []
        for claim in claims:
            label = claim.label or "correlation"
            bad_binding = False
            for series in (claim.x, claim.y, *claim.control_series):
                binding = _binding_check(series, self.kind)
                if binding is not None:
                    checks.append(binding)
                    bad_binding = True
            if bad_binding:
                continue
            if claim.x.n != claim.y.n:
                checks.append(Check(name=label, kind=self.kind, status="refuted",
                                    detail="the two series differ in length"))
                continue
            try:
                r = pearson_r(claim.x.ys(), claim.y.ys())
            except ValueError as exc:
                checks.append(Check(name=label, kind=self.kind, status="refuted",
                                    detail=f"cannot recompute the correlation: {exc}"))
                continue
            if claim.r is not None:
                ok = _approx(r, claim.r, claim.rel_tol, claim.abs_tol)
                checks.append(Check(
                    name=f"{label}:r", kind=self.kind,
                    status="verified" if ok else "refuted",
                    detail=f"r = {r:g}, claimed {claim.r:g}",
                    evidence={"computed": r, "claimed": claim.r},
                ))
            if claim.causal:
                checks.append(self._check_causation(claim, r, label))
        return checks

    def _check_causation(self, claim: CorrelationClaim, r: float, label: str) -> Check:
        name = f"{label}:causation"
        if claim.design == "experiment":
            return Check(name=name, kind=self.kind, status="verified",
                         detail="causal claim warranted by a declared randomized design")
        if not claim.controls:
            return Check(
                name=name, kind=self.kind, status="refuted",
                detail=("correlation does not imply causation: the causal claim declares "
                        "no controls and no randomized design"),
                evidence={"r": r},
            )
        if len(claim.control_series) != len(claim.controls) or not claim.control_series:
            return Check(
                name=name, kind=self.kind, status="refuted",
                detail=(f"asserted controls {claim.controls} were not supplied "
                        "as series for verification"),
                evidence={"controls": claim.controls},
            )
        if any(cs.n != claim.x.n for cs in claim.control_series):
            return Check(name=name, kind=self.kind, status="refuted",
                         detail="a control series differs in length from the data")
        try:
            partial = partial_correlation(
                claim.x.ys(), claim.y.ys(), [cs.ys() for cs in claim.control_series]
            )
        except ValueError as exc:
            return Check(name=name, kind=self.kind, status="refuted",
                         detail=f"cannot recompute the partial correlation: {exc}")
        if abs(partial) < claim.confound_threshold:
            return Check(
                name=name, kind=self.kind, status="refuted",
                detail=(f"association explained by controls {claim.controls}: partial "
                        f"r = {partial:g} collapses below {claim.confound_threshold:g}"),
                evidence={"partial_r": partial, "raw_r": r, "controls": claim.controls},
            )
        return Check(
            name=name, kind=self.kind, status="verified",
            detail=(f"association survives controls {claim.controls}: partial r = {partial:g} "
                    f"(raw r = {r:g})"),
            evidence={"partial_r": partial, "raw_r": r, "controls": claim.controls},
        )


class IntervalVerifier:
    """Recomputes a stated confidence or prediction interval from the cited series.

    For each :class:`IntervalClaim`, recomputes a Student-t confidence interval for
    the series mean (``kind="mean"``) or an OLS prediction interval at a stated
    point (``kind="prediction"``) and confirms the stated bounds. A stated interval
    that is too tight or too wide is refuted. Reads claims passed at construction or
    from the context; with none, inapplicable.
    """

    kind = "interval"

    def __init__(self, claims: list[IntervalClaim] | None = None) -> None:
        self._claims = list(claims) if claims is not None else None

    def check(self, answer: Any, context: VerificationContext) -> list[Check]:
        claims = _claims_of(self._claims, context, IntervalClaim)
        if not claims:
            return [Check(name="interval", kind=self.kind, status="inapplicable",
                          detail="no interval claim supplied")]
        checks: list[Check] = []
        for claim in claims:
            binding = _binding_check(claim.series, self.kind)
            if binding is not None:
                checks.append(binding)
                continue
            checks.append(self._check_interval(claim))
        return checks

    def _check_interval(self, claim: IntervalClaim) -> Check:
        label = claim.label or f"{claim.kind}_interval"
        try:
            if claim.kind == "mean":
                lo, hi = mean_confidence_interval(claim.series.ys(), claim.confidence)
            else:
                if claim.at is None:
                    return Check(name=label, kind=self.kind, status="refuted",
                                 detail="a prediction interval needs the point 'at'")
                lo, hi, _ = prediction_interval(
                    claim.series.xs(), claim.series.ys(), claim.at, claim.confidence
                )
        except ValueError as exc:
            return Check(name=label, kind=self.kind, status="refuted",
                         detail=f"cannot recompute the interval: {exc}")
        ok = (_approx(lo, claim.lower, claim.rel_tol, claim.abs_tol)
              and _approx(hi, claim.upper, claim.rel_tol, claim.abs_tol))
        return Check(
            name=label, kind=self.kind, status="verified" if ok else "refuted",
            detail=(f"{int(claim.confidence * 100)}% interval = [{lo:g}, {hi:g}], "
                    f"claimed [{claim.lower:g}, {claim.upper:g}]"),
            evidence={"computed": [lo, hi], "claimed": [claim.lower, claim.upper]},
        )


class ForecastVerifier:
    """Re-runs a declared deterministic forecast over the cited series and checks it.

    For each :class:`ForecastClaim`, re-runs the declared model over the cited
    series for as many steps as the claim projects and confirms the stated values
    element-wise — a projection the model does not produce is refuted. Reads claims
    passed at construction or from the context; with none, inapplicable.
    """

    kind = "forecast"

    def __init__(self, claims: list[ForecastClaim] | None = None) -> None:
        self._claims = list(claims) if claims is not None else None

    def check(self, answer: Any, context: VerificationContext) -> list[Check]:
        claims = _claims_of(self._claims, context, ForecastClaim)
        if not claims:
            return [Check(name="forecast", kind=self.kind, status="inapplicable",
                          detail="no forecast claim supplied")]
        checks: list[Check] = []
        for claim in claims:
            binding = _binding_check(claim.series, self.kind)
            if binding is not None:
                checks.append(binding)
                continue
            checks.append(self._check_forecast(claim))
        return checks

    def _check_forecast(self, claim: ForecastClaim) -> Check:
        label = claim.label or f"{claim.model}_forecast"
        if not claim.predictions:
            return Check(name=label, kind=self.kind, status="inapplicable",
                         detail="forecast claim stated no predictions")
        try:
            projected = forecast(
                claim.model, claim.series.ys(),
                horizon=len(claim.predictions), xs=claim.series.xs(), params=claim.params,
            )
        except ValueError as exc:
            return Check(name=label, kind=self.kind, status="refuted",
                         detail=f"cannot recompute the forecast: {exc}")
        ok = all(
            _approx(p, c, claim.rel_tol, claim.abs_tol)
            for p, c in zip(projected, claim.predictions, strict=True)
        )
        return Check(
            name=label, kind=self.kind, status="verified" if ok else "refuted",
            detail=(f"{claim.model} projection = {[round(p, 4) for p in projected]}, "
                    f"claimed {[round(c, 4) for c in claim.predictions]}"),
            evidence={"computed": projected, "claimed": claim.predictions, "model": claim.model},
        )


def statistical_verifiers() -> list[Any]:
    """The four statistical kernels — trend, correlation, interval, forecast.

    Each is deterministic, offline, and dependency-free, returning ``inapplicable``
    when no claim of its kind is present, so the set is safe to add to any
    ``app.verify_reasoning`` call. ``app.verify_reasoning`` adds them automatically
    when ``statistical_claims`` are supplied.
    """
    return [TrendVerifier(), CorrelationVerifier(), IntervalVerifier(), ForecastVerifier()]
