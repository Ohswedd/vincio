"""Forecasting & causal-inference verifier kernels — certified statistical claims.

A data answer does not just *retrieve* numbers; it *concludes* from them — "revenue
is trending up ~320/month", "the 95% interval for the mean is [.., ..]", "next
quarter projects to ..", "ice cream sales cause drownings". The verified-reasoning
plane already gives an *arithmetic* claim a checkable certificate (example 09); this
program gives a **statistical** claim the same, on the same `Certificate` surface,
fully offline:

  * `TrendVerifier` — recomputes an ordinary-least-squares slope / intercept and its
    R² goodness-of-fit from the **cited cells** of a real query result;
  * `CorrelationVerifier` — recomputes a Pearson correlation and, crucially,
    **refuses a correlation stated as causation** that declares no controls, and
    **refutes** a controlled claim whose association vanishes once the confounder is
    partialled out (the partial correlation collapses) — the spurious-causation guard;
  * `IntervalVerifier` — recomputes a stated confidence / prediction interval;
  * `ForecastVerifier` — re-runs a declared deterministic forecast model and checks
    the projection, driving refuse-or-repair self-correction on a wrong one.

Every statistic is **bound to the cited cells**: a value swapped after it was cited
makes the series unbound and the kernel refuses. Nothing here calls a model to judge
correctness — the verifier *recomputes* and refuses-or-repairs, the way the
arithmetic kernel does. Everything is deterministic and runs without a network.
"""

from __future__ import annotations

from _shared import example_provider

from vincio import (
    CitedSeries,
    ContextApp,
    CorrelationClaim,
    ForecastClaim,
    IntervalClaim,
    TrendClaim,
)
from vincio.verify.statistical import forecast, mean_confidence_interval, ols_fit, pearson_r

# A quarter of monthly revenue (USD), the kind of series an analyst trends and projects.
REVENUE = [
    {"month": 1, "revenue": 12_000.0},
    {"month": 2, "revenue": 12_300.0},
    {"month": 3, "revenue": 12_650.0},
    {"month": 4, "revenue": 12_900.0},
    {"month": 5, "revenue": 13_300.0},
    {"month": 6, "revenue": 13_550.0},
]


def banner(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


def cited_revenue_series(app: ContextApp) -> CitedSeries:
    """Run a cell-cited query and lift its revenue column into a verifiable series."""
    result = app.query_data("select month, revenue from revenue order by month", table="revenue")
    # Each revenue value carries the exact source cell it came from; `from_cells`
    # keeps that binding so a later edit to a cited cell is caught by the kernel.
    cells = [result.citations(i, "revenue")[0] for i in range(result.row_count)]
    months = [float(result.value(i, "month")) for i in range(result.row_count)]
    return CitedSeries.from_cells(cells, name="revenue", index=months)


def main() -> None:
    provider, _ = example_provider()
    app = ContextApp(name="analytics", provider=provider)
    app.register_dataset(REVENUE, name="revenue")

    # ---- 1. A trend recomputed from the cited cells ----------------------
    banner("1. TrendVerifier — slope & goodness-of-fit from the cited cells")
    series = cited_revenue_series(app)
    fit = ols_fit(series.xs(), series.ys())
    print(f"  cited series: {series.n} cells, e.g. {series.citations[0].ref} = {series.values[0]:,.0f}")
    print(f"  recomputed trend: slope ≈ {fit.slope:,.1f}/month, R² ≈ {fit.r_squared:.3f}")

    truthful = app.verify_reasoning(
        f"Revenue is trending up about {fit.slope:,.0f}/month (R²≈{fit.r_squared:.2f}).",
        statistical_claims=[
            TrendClaim(series=series, slope=round(fit.slope, 1),
                       r_squared=round(fit.r_squared, 3), direction="increasing")
        ],
    )
    print(f"  truthful trend → certificate {truthful.certificate.status}, holds={truthful.holds}")

    inflated = app.verify_reasoning(
        "Revenue is exploding at 5,000/month.",
        statistical_claims=[TrendClaim(series=series, slope=5_000.0)],
    )
    print(f"  inflated trend → certificate {inflated.certificate.status}, refused={inflated.refused}")
    for ref in inflated.certificate.refutations:
        print(f"    ✗ {ref.detail}")

    # ---- 2. Correlation is not causation ---------------------------------
    banner("2. CorrelationVerifier — refuses correlation stated as causation")
    # Two series both driven by temperature (the confounder), with independent
    # idiosyncrasies — they correlate strongly, but neither causes the other.
    temp = [60, 65, 70, 75, 80, 85, 90, 78, 72, 83, 88, 68]
    quirk_a = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    quirk_b = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
    ice_cream = [40.0 * t + 30 * a for t, a in zip(temp, quirk_a, strict=True)]
    drownings = [0.30 * t + 0.6 * b for t, b in zip(temp, quirk_b, strict=True)]
    ice = CitedSeries(name="ice_cream_sales", values=ice_cream)
    drown = CitedSeries(name="drownings", values=drownings)
    temperature = CitedSeries(name="temperature", values=[float(t) for t in temp])
    r = pearson_r(ice_cream, drownings)
    print(f"  corr(ice cream sales, drownings) = {r:.2f}  — strong and real")

    uncontrolled = app.verify_reasoning(
        "Ice cream sales cause drownings.",
        statistical_claims=[CorrelationClaim(x=ice, y=drown, r=round(r, 2), causal=True)],
    )
    print(f"  'ice cream causes drownings' (no controls) → {uncontrolled.certificate.status}")
    for ref in uncontrolled.certificate.refutations:
        print(f"    ✗ {ref.detail}")

    controlled = app.verify_reasoning(
        "Controlling for temperature, ice cream sales cause drownings.",
        statistical_claims=[CorrelationClaim(
            x=ice, y=drown, r=round(r, 2), causal=True,
            controls=["temperature"], control_series=[temperature])],
    )
    print(f"  '...controlling for temperature' → {controlled.certificate.status}")
    for ref in controlled.certificate.refutations:
        print(f"    ✗ {ref.detail}")

    # A genuine driver whose association survives the control is verified.
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    other = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]
    spend = [3.0 * x + 0.2 * o for x, o in zip(xs, other, strict=True)]
    real = app.verify_reasoning(
        "More ad spend drives more signups, even after controlling for seasonality.",
        statistical_claims=[CorrelationClaim(
            x=CitedSeries(name="ad_spend", values=[float(v) for v in xs]),
            y=CitedSeries(name="signups", values=spend),
            r=round(pearson_r([float(v) for v in xs], spend), 2), causal=True,
            controls=["seasonality"],
            control_series=[CitedSeries(name="seasonality", values=[float(v) for v in other])])],
    )
    print(f"  genuine controlled driver (association survives) → {real.certificate.status}")

    # ---- 3. A confidence interval recomputed -----------------------------
    banner("3. IntervalVerifier — a stated confidence interval, recomputed")
    lo, hi = mean_confidence_interval(series.ys(), 0.95)
    print(f"  95% CI for mean monthly revenue = [{lo:,.0f}, {hi:,.0f}]")
    interval = app.verify_reasoning(
        "The 95% confidence interval for mean monthly revenue is the stated band.",
        statistical_claims=[IntervalClaim(series=series, lower=round(lo, 1),
                                          upper=round(hi, 1), kind="mean")],
    )
    print(f"  truthful interval → {interval.certificate.status}")
    too_tight = app.verify_reasoning(
        "The 95% interval is a razor-thin ±50 around the mean.",
        statistical_claims=[IntervalClaim(series=series, lower=12_780.0, upper=12_880.0, kind="mean")],
    )
    print(f"  over-tight interval → {too_tight.certificate.status}, refused={too_tight.refused}")

    # ---- 4. A forecast, with refuse-or-repair self-correction ------------
    banner("4. ForecastVerifier — re-runs the model & repairs a wrong projection")
    projected = forecast("linear", series.ys(), horizon=2, xs=series.xs())
    print(f"  linear-trend projection for the next 2 months ≈ "
          f"[{projected[0]:,.0f}, {projected[1]:,.0f}]")

    # A first answer overshoots; the deterministic refutation drives a repair to the
    # model's actual output, and the repaired answer carries a verified certificate.
    def repair(_answer: str, _critique: str) -> str:
        return f"Next two months project to {projected[0]:,.0f} then {projected[1]:,.0f}."

    repaired = app.verify_reasoning(
        "Next two months will both blow past 20,000.",
        statistical_claims=[ForecastClaim(series=series, model="linear", predictions=[20_000.0, 20_000.0])],
        regenerate=lambda a, c: ForecastClaim(  # repaired claim re-states the model's output
            series=series, model="linear", predictions=projected),
    )
    print(f"  refuse-or-repair → {repaired.certificate.status} after {repaired.attempts} attempt(s)")

    # ---- 5. The certificate re-derives from the bytes --------------------
    banner("5. The statistical certificate is tamper-evident")
    cert = truthful.certificate
    print(f"  verify() before tamper: {cert.verify()}")
    cert.checks[0].status = "refuted"  # flip a recorded verdict
    print(f"  verify() after a flipped check: {cert.verify()}  (caught from the bytes)")

    print("\nEvery statistical conclusion now carries a proof you can re-check offline —")
    print("a refuted one refuses to emit, and a correlation stated as causation must")
    print("earn its warrant or it does not pass.")


if __name__ == "__main__":
    main()
