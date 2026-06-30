"""Forecasting & causal-inference verifier kernels (5.8).

Covers the deterministic statistics core (OLS, Pearson and partial correlation,
Student-t quantiles, mean / prediction intervals, the forecast models) and the
four kernels that certify a data answer's analytical claims on the
:class:`~vincio.verify.Certificate` surface — trend, correlation (including the
refutation of correlation-stated-as-causation), interval, and forecast — plus the
cited-cell binding, the app integration, and the optional exact-arithmetic CAS
backend. Everything is deterministic and fully offline.
"""

from __future__ import annotations

import pytest

from vincio import (
    CellRef,
    CitedSeries,
    ContextApp,
    CorrelationClaim,
    CorrelationVerifier,
    ForecastClaim,
    ForecastVerifier,
    IntervalClaim,
    IntervalVerifier,
    StatisticalClaim,
    TrendClaim,
    TrendVerifier,
    VincioConfig,
    statistical_verifiers,
)
from vincio.providers import MockProvider
from vincio.verify import CompositeVerifier, VerificationContext
from vincio.verify.statistical import (
    forecast,
    mean_confidence_interval,
    ols_fit,
    partial_correlation,
    pearson_r,
    prediction_interval,
    student_t_ppf,
)


def _app() -> ContextApp:
    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    return ContextApp(name="stats", provider=MockProvider(default_text="ok"), config=cfg)


# --------------------------------------------------------------------------- #
# Deterministic statistics core                                               #
# --------------------------------------------------------------------------- #


def test_ols_fit_recovers_exact_line():
    fit = ols_fit([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])  # y = 2x + 1
    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(1.0)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.predict(10) == pytest.approx(21.0)


def test_ols_fit_goodness_of_fit_below_one_for_noisy_data():
    fit = ols_fit([0, 1, 2, 3], [0.0, 1.1, 1.9, 3.2])
    assert 0.9 < fit.r_squared < 1.0


def test_ols_fit_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        ols_fit([1.0], [1.0])
    with pytest.raises(ValueError):
        ols_fit([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])  # no predictor spread


def test_pearson_matches_known_values():
    assert pearson_r([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert pearson_r([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        pearson_r([1, 1, 1], [1, 2, 3])  # no variance


def test_student_t_quantiles_match_table():
    # Classic two-sided 95% critical values.
    assert student_t_ppf(0.975, 1) == pytest.approx(12.7062, abs=2e-3)
    assert student_t_ppf(0.975, 2) == pytest.approx(4.30265, abs=2e-3)
    assert student_t_ppf(0.975, 10) == pytest.approx(2.22814, abs=2e-3)
    assert student_t_ppf(0.975, 30) == pytest.approx(2.04227, abs=2e-3)
    # Tends to the normal as df grows.
    assert student_t_ppf(0.975, 1_000_000) == pytest.approx(1.95996, abs=2e-3)
    assert student_t_ppf(0.5, 5) == pytest.approx(0.0)


def test_mean_confidence_interval_brackets_the_mean():
    values = [10.0, 12.0, 11.0, 13.0, 9.0, 10.0, 14.0, 12.0]
    lo, hi = mean_confidence_interval(values, 0.95)
    mean = sum(values) / len(values)
    assert lo < mean < hi
    # Known: n=8, mean=11.375, s≈1.685, t_.975(7)=2.3646 -> half≈1.409.
    assert hi - lo == pytest.approx(2.817, abs=1e-2)


def test_prediction_interval_widens_away_from_center():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [1.0, 2.9, 5.2, 6.8, 9.1, 11.0]
    lo_c, hi_c, yhat_c = prediction_interval(xs, ys, 2.5, 0.95)
    lo_f, hi_f, _ = prediction_interval(xs, ys, 10.0, 0.95)
    assert lo_c < yhat_c < hi_c
    assert (hi_f - lo_f) > (hi_c - lo_c)  # the interval widens away from the center
    with pytest.raises(ValueError):
        prediction_interval([0.0, 1.0], [1.0, 2.0], 3.0)  # < 3 points


@pytest.mark.parametrize(
    "model,ys,expected,params",
    [
        ("naive", [3.0, 5.0, 8.0], [8.0, 8.0], None),
        ("mean", [2.0, 4.0, 6.0], [4.0, 4.0], None),
        ("drift", [10.0, 12.0, 14.0, 16.0, 18.0], [20.0, 22.0, 24.0], None),
        ("moving_average", [1.0, 2.0, 9.0, 10.0, 11.0], [10.0, 10.0], {"window": 3}),
        ("ses", [10.0, 10.0, 10.0], [10.0], {"alpha": 0.4}),
    ],
)
def test_forecast_models(model, ys, expected, params):
    got = forecast(model, ys, horizon=len(expected), params=params)
    assert got == pytest.approx(expected)


def test_forecast_linear_extrapolates_trend():
    ys = [1.0, 3.0, 5.0, 7.0, 9.0]  # slope 2, next index 5 -> 11, 13
    assert forecast("linear", ys, horizon=2) == pytest.approx([11.0, 13.0])


def test_forecast_rejects_unknown_model_and_bad_params():
    with pytest.raises(ValueError):
        forecast("quantum", [1.0, 2.0], horizon=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        forecast("ses", [1.0, 2.0], horizon=1, params={"alpha": 2.0})


def test_partial_correlation_collapses_for_a_confounder():
    # Two series each driven by a common cause with orthogonal idiosyncrasies.
    temp = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
    n1 = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    n2 = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
    a = [2.0 * t + 3 * i for t, i in zip(temp, n1, strict=True)]
    b = [0.5 * t + 3 * j for t, j in zip(temp, n2, strict=True)]
    raw = pearson_r(a, b)
    partial = partial_correlation(a, b, [[float(t) for t in temp]])
    assert raw > 0.8  # strongly correlated on the surface
    assert abs(partial) < 0.2  # the association is the confounder's


def test_partial_correlation_zero_for_perfectly_collinear_confounder():
    temp = list(range(10))
    a = [2.0 * t for t in temp]
    b = [0.5 * t for t in temp]
    assert partial_correlation(a, b, [[float(t) for t in temp]]) == 0.0


# --------------------------------------------------------------------------- #
# CitedSeries binding                                                         #
# --------------------------------------------------------------------------- #


def test_cited_series_constructors_and_binding():
    pairs = CitedSeries.from_pairs([(0.0, 1.0), (1.0, 3.0)], name="p")
    assert pairs.xs() == [0.0, 1.0]
    assert pairs.ys() == [1.0, 3.0]
    assert pairs.is_bound()  # no citations -> binding not asserted

    cells = [CellRef(ref="t#r0!v", value=1.0), CellRef(ref="t#r1!v", value=3.0)]
    bound = CitedSeries.from_cells(cells, name="c")
    assert bound.values == [1.0, 3.0]
    assert bound.is_bound()
    bound.values[1] = 99.0  # edit a value after it was cited
    assert not bound.is_bound()


def test_from_cells_accepts_data_plane_cell_citation():
    from vincio.data.provenance import CellCitation

    cells = [
        CellCitation(table="sales", row=0, column="rev", value=10.0),
        CellCitation(table="sales", row=1, column="rev", value=20.0),
    ]
    series = CitedSeries.from_cells(cells, name="rev")
    assert series.values == [10.0, 20.0]
    assert series.citations[0].ref == "sales#r0!rev"
    assert series.is_bound()


# --------------------------------------------------------------------------- #
# TrendVerifier                                                               #
# --------------------------------------------------------------------------- #


def test_trend_verifier_confirms_and_refutes():
    series = CitedSeries(name="rev", values=[1.0, 3.0, 5.0, 7.0, 9.0])
    good = CompositeVerifier([TrendVerifier()]).certify(
        "trend",
        VerificationContext(statistical_claims=[
            TrendClaim(label="rev", series=series, slope=2.0, intercept=1.0,
                       r_squared=1.0, direction="increasing")
        ]),
    )
    assert good.status == "verified"
    bad = CompositeVerifier([TrendVerifier()]).certify(
        "trend",
        VerificationContext(statistical_claims=[TrendClaim(series=series, slope=5.0)]),
    )
    assert bad.status == "refuted"


def test_trend_verifier_refutes_wrong_direction_and_unbound_series():
    falling = CitedSeries(name="x", values=[9.0, 7.0, 5.0, 3.0])
    cert = CompositeVerifier([TrendVerifier()]).certify(
        "t", VerificationContext(statistical_claims=[
            TrendClaim(series=falling, direction="increasing")]))
    assert cert.status == "refuted"

    smuggled = CitedSeries(
        name="s", values=[1.0, 2.0, 3.0],
        citations=[CellRef(ref="t#r0!a", value=1.0), CellRef(ref="t#r1!a", value=9.0),
                   CellRef(ref="t#r2!a", value=3.0)],
    )
    cert2 = CompositeVerifier([TrendVerifier()]).certify(
        "t", VerificationContext(statistical_claims=[TrendClaim(series=smuggled, slope=1.0)]))
    assert cert2.status == "refuted"
    assert any("cite" in c.detail for c in cert2.refutations)


def test_trend_verifier_inapplicable_without_a_claim():
    cert = CompositeVerifier([TrendVerifier()]).certify("no trend here")
    assert cert.status == "inapplicable"


# --------------------------------------------------------------------------- #
# CorrelationVerifier — correlation & spurious causation                      #
# --------------------------------------------------------------------------- #


def _confounded_claim(**overrides):
    temp = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
    n1 = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    n2 = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
    a = [2.0 * t + 3 * i for t, i in zip(temp, n1, strict=True)]
    b = [0.5 * t + 3 * j for t, j in zip(temp, n2, strict=True)]
    x = CitedSeries(name="ice", values=a)
    y = CitedSeries(name="drown", values=b)
    z = CitedSeries(name="temp", values=[float(t) for t in temp])
    fields = dict(label="link", x=x, y=y, r=round(pearson_r(a, b), 2), causal=True)
    fields.update(overrides)
    return CorrelationClaim(**fields), z


def test_correlation_value_verified_and_refuted():
    x = CitedSeries(name="x", values=[1.0, 2.0, 3.0, 4.0])
    y = CitedSeries(name="y", values=[2.0, 4.0, 6.0, 8.0])
    ok = CompositeVerifier([CorrelationVerifier()]).certify(
        "c", VerificationContext(statistical_claims=[CorrelationClaim(x=x, y=y, r=1.0)]))
    assert ok.status == "verified"
    bad = CompositeVerifier([CorrelationVerifier()]).certify(
        "c", VerificationContext(statistical_claims=[CorrelationClaim(x=x, y=y, r=0.1)]))
    assert bad.status == "refuted"


def test_causal_claim_without_controls_is_refuted():
    claim, _ = _confounded_claim()
    cert = CompositeVerifier([CorrelationVerifier()]).certify(
        "ice cream sales cause drownings",
        VerificationContext(statistical_claims=[claim]))
    assert cert.status == "refuted"
    assert any("does not imply causation" in c.detail for c in cert.refutations)


def test_causal_claim_collapses_under_declared_control():
    claim, z = _confounded_claim(controls=["temperature"], control_series=[])
    # asserted a control but did not supply the series
    cert = CompositeVerifier([CorrelationVerifier()]).certify(
        "x", VerificationContext(statistical_claims=[claim]))
    assert cert.status == "refuted"
    assert any("not supplied" in c.detail for c in cert.refutations)

    claim2, z2 = _confounded_claim(controls=["temperature"], control_series=[z])
    cert2 = CompositeVerifier([CorrelationVerifier()]).certify(
        "controlling for temperature, ice cream causes drownings",
        VerificationContext(statistical_claims=[claim2]))
    assert cert2.status == "refuted"
    assert any("explained by controls" in c.detail for c in cert2.refutations)


def test_genuine_controlled_association_survives():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    z = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]
    ys = [2.0 * xv + 0.1 * zv for xv, zv in zip(xs, z, strict=True)]
    x = CitedSeries(name="x", values=[float(v) for v in xs])
    y = CitedSeries(name="y", values=ys)
    zc = CitedSeries(name="z", values=[float(v) for v in z])
    claim = CorrelationClaim(
        label="real", x=x, y=y, r=round(pearson_r([float(v) for v in xs], ys), 2),
        causal=True, controls=["z"], control_series=[zc])
    cert = CompositeVerifier([CorrelationVerifier()]).certify(
        "x drives y", VerificationContext(statistical_claims=[claim]))
    assert cert.status == "verified"


def test_experimental_design_warrants_causation():
    x = CitedSeries(name="dose", values=[0.0, 1.0, 2.0, 3.0, 4.0])
    y = CitedSeries(name="effect", values=[1.0, 2.1, 2.9, 4.2, 5.0])
    claim = CorrelationClaim(
        x=x, y=y, r=round(pearson_r(x.ys(), y.ys()), 2),
        causal=True, design="experiment")
    cert = CompositeVerifier([CorrelationVerifier()]).certify(
        "the treatment causes the effect (RCT)",
        VerificationContext(statistical_claims=[claim]))
    assert cert.status == "verified"


# --------------------------------------------------------------------------- #
# IntervalVerifier                                                            #
# --------------------------------------------------------------------------- #


def test_interval_verifier_mean_and_prediction():
    vals = [10.0, 12.0, 11.0, 13.0, 9.0, 10.0, 14.0, 12.0]
    lo, hi = mean_confidence_interval(vals, 0.95)
    series = CitedSeries(name="m", values=vals)
    ok = CompositeVerifier([IntervalVerifier()]).certify(
        "i", VerificationContext(statistical_claims=[
            IntervalClaim(series=series, lower=round(lo, 3), upper=round(hi, 3), kind="mean")]))
    assert ok.status == "verified"
    too_tight = CompositeVerifier([IntervalVerifier()]).certify(
        "i", VerificationContext(statistical_claims=[
            IntervalClaim(series=series, lower=11.0, upper=11.5, kind="mean")]))
    assert too_tight.status == "refuted"


def test_interval_verifier_prediction_kind():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [1.0, 2.9, 5.2, 6.8, 9.1, 11.0]
    lo, hi, _ = prediction_interval(xs, ys, 6.0, 0.95)
    series = CitedSeries(name="p", index=xs, values=ys)
    cert = CompositeVerifier([IntervalVerifier()]).certify(
        "i", VerificationContext(statistical_claims=[
            IntervalClaim(series=series, lower=round(lo, 2), upper=round(hi, 2),
                          kind="prediction", at=6.0)]))
    assert cert.status == "verified"
    missing_at = CompositeVerifier([IntervalVerifier()]).certify(
        "i", VerificationContext(statistical_claims=[
            IntervalClaim(series=series, lower=0.0, upper=1.0, kind="prediction")]))
    assert missing_at.status == "refuted"


# --------------------------------------------------------------------------- #
# ForecastVerifier                                                            #
# --------------------------------------------------------------------------- #


def test_forecast_verifier_confirms_and_refutes():
    series = CitedSeries(name="f", values=[10.0, 12.0, 14.0, 16.0, 18.0])
    projected = forecast("drift", series.ys(), horizon=3)
    ok = CompositeVerifier([ForecastVerifier()]).certify(
        "f", VerificationContext(statistical_claims=[
            ForecastClaim(series=series, model="drift", predictions=projected)]))
    assert ok.status == "verified"
    bad = CompositeVerifier([ForecastVerifier()]).certify(
        "f", VerificationContext(statistical_claims=[
            ForecastClaim(series=series, model="drift", predictions=[20.0, 20.0, 20.0])]))
    assert bad.status == "refuted"


def test_forecast_verifier_uses_declared_params():
    series = CitedSeries(name="f", values=[1.0, 2.0, 9.0, 10.0, 11.0])
    projected = forecast("moving_average", series.ys(), horizon=2, params={"window": 3})
    cert = CompositeVerifier([ForecastVerifier()]).certify(
        "f", VerificationContext(statistical_claims=[
            ForecastClaim(series=series, model="moving_average",
                          predictions=projected, params={"window": 3})]))
    assert cert.status == "verified"


# --------------------------------------------------------------------------- #
# App integration & certificate content-binding                              #
# --------------------------------------------------------------------------- #


def test_verify_reasoning_default_path_unchanged_without_claims():
    app = _app()
    cert = app.verify_reasoning("2 + 2 = 4").certificate
    # The statistical kinds are NOT added when no statistical claim is supplied.
    assert "trend" not in cert.kinds
    assert "arithmetic" in cert.kinds


def test_verify_reasoning_adds_statistical_kernels_when_claims_supplied():
    app = _app()
    series = CitedSeries(name="rev", values=[1.0, 3.0, 5.0, 7.0, 9.0])
    verified = app.verify_reasoning(
        "Revenue is trending up by 2 per period.",
        statistical_claims=[TrendClaim(series=series, slope=2.0, direction="increasing")],
    )
    assert verified.holds
    assert "trend" in verified.certificate.kinds
    assert any(e.action == "reasoning_verification" for e in app.audit.entries)


def test_verify_reasoning_refuses_spurious_causation():
    app = _app()
    claim, _ = _confounded_claim()
    out = app.verify_reasoning(
        "Ice cream sales cause drownings.", statistical_claims=[claim])
    assert out.refused and not out.holds


def test_verify_reasoning_repairs_a_bad_forecast():
    app = _app()
    series = CitedSeries(name="f", values=[10.0, 12.0, 14.0, 16.0, 18.0])
    correct = forecast("drift", series.ys(), horizon=2)

    # A repair that re-states the corrected claim re-grounds the context and verifies.
    repaired = app.verify_reasoning(
        "Next two periods will both exceed 99.",
        statistical_claims=[ForecastClaim(series=series, model="drift", predictions=[99.0, 99.0])],
        regenerate=lambda answer, critique: ForecastClaim(
            series=series, model="drift", predictions=correct),
    )
    assert repaired.holds
    assert repaired.attempts == 2

    # A no-op repair leaves the wrong claim in place, so it stays refused.
    refused = app.verify_reasoning(
        "still wrong",
        statistical_claims=[ForecastClaim(series=series, model="drift", predictions=[99.0, 99.0])],
        regenerate=lambda a, c: a,
    )
    assert refused.refused and not refused.holds


def test_statistical_certificate_catches_a_flipped_verdict():
    series = CitedSeries(name="rev", values=[1.0, 3.0, 5.0, 7.0, 9.0])
    cert = CompositeVerifier(statistical_verifiers()).certify(
        "t", VerificationContext(statistical_claims=[TrendClaim(series=series, slope=99.0)]))
    assert cert.status == "refuted"
    assert cert.verify()
    cert.checks[0].status = "verified"
    cert.status = "verified"
    assert not cert.verify()  # re-derivation from the bytes catches the tamper


def test_statistical_verifiers_are_inapplicable_over_a_plain_answer():
    cert = CompositeVerifier(statistical_verifiers()).certify("just some prose")
    assert cert.status == "inapplicable"


def test_statistical_claim_base_is_exported_and_typed():
    # The base carries the tolerance band every concrete claim shares.
    base = StatisticalClaim(label="x", rel_tol=0.05, abs_tol=0.01)
    assert base.rel_tol == 0.05
    assert issubclass(TrendClaim, StatisticalClaim)


# --------------------------------------------------------------------------- #
# Defensive branches — degenerate inputs degrade, never crash                 #
# --------------------------------------------------------------------------- #


def test_math_core_rejects_degenerate_inputs():
    from vincio.verify.statistical import _sample_std, _solve_linear, student_t_cdf, t_critical

    with pytest.raises(ValueError):
        _sample_std([1.0])
    with pytest.raises(ValueError):
        ols_fit([0.0, 1.0], [1.0])  # length mismatch
    with pytest.raises(ValueError):
        pearson_r([1.0, 2.0], [1.0])  # length mismatch
    with pytest.raises(ValueError):
        pearson_r([1.0], [1.0])  # < 2 points
    with pytest.raises(ValueError):
        _solve_linear([[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0])  # singular
    with pytest.raises(ValueError):
        student_t_cdf(1.0, 0.0)  # df must be positive
    with pytest.raises(ValueError):
        t_critical(5, 1.5)  # confidence out of (0, 1)
    with pytest.raises(ValueError):
        student_t_ppf(0.0, 5)  # probability out of (0, 1)
    with pytest.raises(ValueError):
        mean_confidence_interval([1.0])  # < 2 points


def test_forecast_guards():
    with pytest.raises(ValueError):
        forecast("naive", [1.0, 2.0], horizon=0)  # horizon < 1
    with pytest.raises(ValueError):
        forecast("naive", [], horizon=1)  # empty series
    with pytest.raises(ValueError):
        forecast("drift", [1.0], horizon=1)  # drift needs >= 2
    with pytest.raises(ValueError):
        forecast("moving_average", [1.0, 2.0], horizon=1, params={"window": 9})
    with pytest.raises(ValueError):
        forecast("ses", [1.0, 2.0], horizon=1, params={"alpha": 0.0})
    with pytest.raises(ValueError):
        forecast("linear", [1.0, 2.0, 3.0], horizon=1, xs=[0.0, 1.0])  # xs length mismatch


def test_cited_series_unbound_when_citation_count_mismatches():
    s = CitedSeries(
        name="s", values=[1.0, 2.0, 3.0],
        citations=[CellRef(ref="t#r0!a", value=1.0)],  # one citation, three values
    )
    assert not s.is_bound()


def test_trend_claim_with_no_fields_is_inapplicable():
    series = CitedSeries(name="x", values=[1.0, 2.0, 3.0])
    cert = CompositeVerifier([TrendVerifier()]).certify(
        "t", VerificationContext(statistical_claims=[TrendClaim(series=series)]))
    assert cert.status == "inapplicable"


def test_trend_verifier_refutes_a_series_too_short_to_fit():
    series = CitedSeries(name="x", values=[5.0])
    cert = CompositeVerifier([TrendVerifier()]).certify(
        "t", VerificationContext(statistical_claims=[TrendClaim(series=series, slope=1.0)]))
    assert cert.status == "refuted"


def test_correlation_refutes_length_mismatch_and_unverifiable_controls():
    x = CitedSeries(name="x", values=[1.0, 2.0, 3.0, 4.0])
    short_y = CitedSeries(name="y", values=[1.0, 2.0, 3.0])
    mismatch = CompositeVerifier([CorrelationVerifier()]).certify(
        "c", VerificationContext(statistical_claims=[CorrelationClaim(x=x, y=short_y, r=0.5)]))
    assert mismatch.status == "refuted"

    y = CitedSeries(name="y", values=[2.0, 4.0, 6.0, 8.0])
    bad_control = CitedSeries(name="z", values=[1.0, 2.0])  # wrong length control
    cert = CompositeVerifier([CorrelationVerifier()]).certify(
        "c", VerificationContext(statistical_claims=[CorrelationClaim(
            x=x, y=y, r=1.0, causal=True, controls=["z"], control_series=[bad_control])]))
    assert cert.status == "refuted"


def test_forecast_claim_with_no_predictions_is_inapplicable():
    series = CitedSeries(name="f", values=[1.0, 2.0, 3.0])
    cert = CompositeVerifier([ForecastVerifier()]).certify(
        "f", VerificationContext(statistical_claims=[
            ForecastClaim(series=series, model="naive", predictions=[])]))
    assert cert.status == "inapplicable"


# --------------------------------------------------------------------------- #
# Optional CAS backend (vincio[verify])                                       #
# --------------------------------------------------------------------------- #


def test_cas_trend_verifier_discharges_exact_fit():
    pytest.importorskip("sympy")
    from vincio.verify.smt import CasTrendVerifier

    series = CitedSeries(name="rev", values=[1.0, 3.0, 5.0, 7.0, 9.0])
    cert = CompositeVerifier([CasTrendVerifier()]).certify(
        "t", VerificationContext(statistical_claims=[
            TrendClaim(series=series, slope=2.0, intercept=1.0)]))
    assert cert.status == "verified"
    bad = CompositeVerifier([CasTrendVerifier()]).certify(
        "t", VerificationContext(statistical_claims=[TrendClaim(series=series, slope=3.0)]))
    assert bad.status == "refuted"
