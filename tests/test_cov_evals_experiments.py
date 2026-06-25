"""Real-behavior coverage tests for vincio.evals.experiments.

Exercises the experiment tracker store round-trips, comparison/ablation logic,
the Experiment A/B harness over a real MockProvider-backed app, and the
pure-Python significance machinery (paired/Welch t-tests, incomplete beta,
confidence intervals) through their real edge cases.
"""

from __future__ import annotations

import math

import pytest

from vincio.core.errors import EvalError
from vincio.evals.experiments import (
    Experiment,
    ExperimentRun,
    ExperimentTracker,
    _betacf,
    _betainc,
    _t_critical,
    _t_two_sided_p,
    ab_test,
)
from vincio.evals.reports import CaseResult, EvalReport


def _report(values, *, metric="quality", name="r", cost=0.0, prefix="c"):
    """Build an EvalReport with one case per value under a single metric."""
    cases = [
        CaseResult(case_id=f"{prefix}{i}", metrics={metric: v}, cost_usd=cost)
        for i, v in enumerate(values)
    ]
    return EvalReport(name=name, cases=cases)


def _shared_report(pairs, *, metric="quality", name="r"):
    """Report where case ids are explicitly given (for pairing control)."""
    cases = [CaseResult(case_id=cid, metrics={metric: v}) for cid, v in pairs]
    return EvalReport(name=name, cases=cases)


# -- ExperimentTracker store round-trips --------------------------------------


def test_tracker_from_path_creates_sqlite_store(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "exp.db"))
    from vincio.storage.sqlite import SQLiteMetadataStore

    assert isinstance(tracker.store, SQLiteMetadataStore)


def test_log_and_runs_roundtrip_with_variant_filter(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "exp.db"))
    tracker.log("exp1", _report([0.9, 0.8]), variant="baseline")
    tracker.log("exp1", _report([0.95, 0.85]), variant="candidate", params={"k": 1})

    # Unfiltered returns both variants (line covering where without variant).
    all_runs = tracker.runs("exp1")
    assert {r.variant for r in all_runs} == {"baseline", "candidate"}

    # variant=... adds the where clause (line 81).
    only_cand = tracker.runs("exp1", variant="candidate")
    assert [r.variant for r in only_cand] == ["candidate"]
    assert only_cand[0].params == {"k": 1}


def test_experiments_lists_distinct_sorted_names(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "exp.db"))
    tracker.log("zeta", _report([1.0]))
    tracker.log("alpha", _report([1.0]))
    tracker.log("alpha", _report([0.5]), variant="v2")
    assert tracker.experiments() == ["alpha", "zeta"]


def test_metric_mean_none_when_metric_absent():
    run = ExperimentRun(experiment="e", report=_report([0.4, 0.6]))
    assert run.metric_mean("quality") == pytest.approx(0.5)
    assert run.metric_mean("missing") is None


def test_latest_returns_most_recent_run(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "exp.db"))
    first = tracker.log("e", _report([0.1]), variant="b")
    second = tracker.log("e", _report([0.2]), variant="b")
    latest = tracker.latest("e", "b")
    assert latest.id == second.id
    assert latest.id != first.id


def test_latest_raises_for_unknown_variant(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "exp.db"))
    tracker.log("e", _report([0.1]), variant="baseline")
    with pytest.raises(EvalError, match="has no runs for variant 'ghost'"):
        tracker.latest("e", "ghost")


def test_latest_per_variant_raises_when_experiment_empty(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "exp.db"))
    with pytest.raises(EvalError, match="experiment 'nope' has no runs"):
        tracker._latest_per_variant("nope")


# -- compare ------------------------------------------------------------------


def test_compare_picks_best_per_metric_direction(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "exp.db"))
    # quality: higher is better; cost: lower is better (in LOWER_IS_BETTER).
    base = EvalReport(
        name="b",
        cases=[CaseResult(case_id="c0", metrics={"quality": 0.7, "cost": 200.0})],
    )
    cand = EvalReport(
        name="c",
        cases=[CaseResult(case_id="c0", metrics={"quality": 0.9, "cost": 120.0})],
    )
    tracker.log("e", base, variant="baseline")
    tracker.log("e", cand, variant="candidate")

    result = tracker.compare("e")
    assert result["best"]["quality"] == "candidate"  # max
    assert result["best"]["cost"] == "candidate"  # min (lower is better)
    assert result["metrics"]["quality"]["baseline"] == 0.7
    assert set(result["variants"]) == {"baseline", "candidate"}


def test_compare_skips_metric_absent_from_all_variants(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "exp.db"))
    tracker.log("e", _report([0.5]), variant="baseline")
    # Explicitly request a metric no run has -> row empty -> continue (line 134).
    result = tracker.compare("e", metrics=["does_not_exist"])
    assert result["metrics"] == {}
    assert result["best"] == {}


# -- ablation -----------------------------------------------------------------


def test_ablation_computes_signed_deltas_and_significance(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "exp.db"))
    base = _shared_report([("c0", 0.5), ("c1", 0.5), ("c2", 0.5)])
    cand = _shared_report([("c0", 0.8), ("c1", 0.8), ("c2", 0.8)])
    tracker.log("e", base, variant="baseline")
    tracker.log("e", cand, variant="candidate")

    out = tracker.ablation("e")
    entry = out["ablation"]["candidate"]["quality"]
    assert entry["baseline"] == 0.5
    assert entry["variant"] == 0.8
    assert entry["delta"] == pytest.approx(0.3)
    assert out["baseline"] == "baseline"


def test_ablation_raises_when_baseline_missing(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "exp.db"))
    tracker.log("e", _report([0.5]), variant="treatment")
    with pytest.raises(EvalError, match="has no baseline variant 'baseline'"):
        tracker.ablation("e")


def test_ablation_skips_metric_missing_for_a_variant(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "exp.db"))
    # baseline has metric "shared" + "only_base"; candidate has only "shared".
    base = EvalReport(
        name="b",
        cases=[CaseResult(case_id="c0", metrics={"shared": 0.5, "only_base": 0.1})],
    )
    cand = EvalReport(
        name="c", cases=[CaseResult(case_id="c0", metrics={"shared": 0.9})]
    )
    tracker.log("e", base, variant="baseline")
    tracker.log("e", cand, variant="candidate")

    out = tracker.ablation("e")
    cand_entry = out["ablation"]["candidate"]
    # only_base appears in the compare table but not in candidate's row -> skipped.
    assert "only_base" not in cand_entry
    assert "shared" in cand_entry


# -- ab_test: errors and test selection ---------------------------------------


def test_ab_test_raises_when_metric_missing_from_a_report():
    a = _report([0.5, 0.6], metric="quality")
    b = _report([0.5, 0.6], metric="other")
    with pytest.raises(EvalError, match="metric 'quality' missing from one or both"):
        ab_test(a, b, "quality")


def test_ab_test_paired_detects_significant_uniform_shift():
    base = _shared_report([("c0", 0.5), ("c1", 0.5), ("c2", 0.5), ("c3", 0.5)])
    cand = _shared_report([("c0", 0.8), ("c1", 0.8), ("c2", 0.8), ("c3", 0.8)])
    res = ab_test(base, cand, "quality")
    assert res["test"] == "paired_t"
    # Zero within-pair variance + nonzero mean diff -> p=0, t=inf.
    assert res["delta"] == pytest.approx(0.3)
    assert res["p_value"] == 0.0
    assert res["t"] == math.inf
    assert res["significant"] is True


def test_ab_test_paired_identical_reports_not_significant():
    base = _shared_report([("c0", 0.5), ("c1", 0.7), ("c2", 0.9)])
    cand = _shared_report([("c0", 0.5), ("c1", 0.7), ("c2", 0.9)])
    res = ab_test(base, cand, "quality")
    assert res["test"] == "paired_t"
    assert res["delta"] == 0.0
    assert res["t"] == 0.0
    assert res["p_value"] == 1.0
    assert res["significant"] is False
    assert res["effect_size"] == 0.0


def test_ab_test_paired_real_t_statistic_with_variance():
    # Differences vary (not all equal) -> goes through the finite-t branch.
    base = _shared_report([("c0", 0.5), ("c1", 0.5), ("c2", 0.5), ("c3", 0.5)])
    cand = _shared_report([("c0", 0.6), ("c1", 0.7), ("c2", 0.55), ("c3", 0.9)])
    res = ab_test(base, cand, "quality")
    assert res["test"] == "paired_t"
    assert math.isfinite(res["t"])
    assert res["df"] == 3.0  # n-1
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["std_error"] > 0


def test_ab_test_welch_when_case_ids_disjoint():
    # No shared ids -> not paired -> Welch.
    a = _shared_report([("a0", 0.4), ("a1", 0.5), ("a2", 0.6)])
    b = _shared_report([("b0", 0.7), ("b1", 0.8), ("b2", 0.9)])
    res = ab_test(a, b, "quality")
    assert res["test"] == "welch_t"
    assert res["n_a"] == 3
    assert res["n_b"] == 3
    assert math.isfinite(res["t"])
    assert res["delta"] == pytest.approx(0.3)


def test_ab_test_welch_zero_variance_both_sides_separates():
    # Disjoint ids (Welch), each side constant but different -> se_sq==0 branch.
    a = _shared_report([("a0", 0.5), ("a1", 0.5)])
    b = _shared_report([("b0", 0.8), ("b1", 0.8)])
    res = ab_test(a, b, "quality")
    assert res["test"] == "welch_t"
    assert res["t"] == math.inf
    assert res["p_value"] == 0.0
    assert res["df"] == 2.0  # n_a + n_b - 2
    assert res["significant"] is True


def test_ab_test_welch_zero_variance_equal_means_not_significant():
    a = _shared_report([("a0", 0.5), ("a1", 0.5)])
    b = _shared_report([("b0", 0.5), ("b1", 0.5)])
    res = ab_test(a, b, "quality")
    assert res["test"] == "welch_t"
    assert res["delta"] == 0.0
    assert res["t"] == 0.0
    assert res["p_value"] == 1.0
    assert res["significant"] is False


def test_ab_test_welch_singleton_per_side_falls_back_df():
    # n_a == n_b == 1 with variance==0 forces df = n_a + n_b - 2 fallback.
    a = _shared_report([("a0", 0.2)])
    b = _shared_report([("b0", 0.9)])
    res = ab_test(a, b, "quality")
    assert res["test"] == "welch_t"
    assert res["df"] == 0.0
    # df<=0 -> p_value path returns 1.0 via _t_two_sided_p guard.
    assert res["p_value"] == 0.0  # se_sq==0 branch sets p before df is used


def test_ab_test_single_shared_id_is_welch_not_paired():
    # Exactly one shared id -> paired requires >=2 shared -> Welch.
    a = _shared_report([("c0", 0.4), ("a1", 0.5)])
    b = _shared_report([("c0", 0.7), ("b1", 0.8)])
    res = ab_test(a, b, "quality")
    assert res["test"] == "welch_t"


def test_ab_test_confidence_interval_brackets_delta():
    a = _shared_report([("a0", 0.40), ("a1", 0.42), ("a2", 0.44), ("a3", 0.46)])
    b = _shared_report([("b0", 0.70), ("b1", 0.72), ("b2", 0.74), ("b3", 0.76)])
    res = ab_test(a, b, "quality", alpha=0.05)
    assert res["ci_low"] < res["delta"] < res["ci_high"]
    assert res["confidence"] == 0.95
    assert res["alpha"] == 0.05


# -- low-level statistical helpers --------------------------------------------


def test_betacf_handles_underflowing_initial_denominator():
    # With a=b=1 the seed d = 1 - 2x/2 = 1 - x; at x ~= 1 it underflows below
    # fpmin and the guard on line 280 substitutes fpmin so we still get a
    # finite continued-fraction value instead of dividing by zero.
    value = _betacf(1.0, 1.0, 1.0 - 1e-301)
    assert math.isfinite(value)
    assert value > 0.0
    # Sanity: away from the boundary the fraction is well-behaved and finite.
    assert math.isfinite(_betacf(2.0, 3.0, 0.4))


def test_betainc_boundary_values():
    assert _betainc(2.0, 3.0, 0.0) == 0.0  # x <= 0
    assert _betainc(2.0, 3.0, -0.5) == 0.0  # negative x
    assert _betainc(2.0, 3.0, 1.0) == 1.0  # x >= 1
    assert _betainc(2.0, 3.0, 1.5) == 1.0  # above 1


def test_betainc_symmetry_identity():
    # I_x(a,b) == 1 - I_{1-x}(b,a)
    x = 0.37
    left = _betainc(2.0, 5.0, x)
    right = 1.0 - _betainc(5.0, 2.0, 1.0 - x)
    assert left == pytest.approx(right, abs=1e-9)
    assert 0.0 < left < 1.0


def test_betainc_uses_both_continued_fraction_branches():
    # x < (a+1)/(a+b+2) takes one branch; x above it takes the other.
    small = _betainc(2.0, 8.0, 0.05)
    large = _betainc(2.0, 8.0, 0.95)
    assert 0.0 < small < large <= 1.0


def test_t_two_sided_p_known_value_and_guards():
    # df<=0 guard returns 1.0.
    assert _t_two_sided_p(2.0, 0.0) == 1.0
    assert _t_two_sided_p(2.0, -3.0) == 1.0
    # t=0 -> p=1 (no evidence against null).
    assert _t_two_sided_p(0.0, 10.0) == pytest.approx(1.0, abs=1e-9)
    # Large t with df=1: two-sided p for t=12.71 is ~0.05.
    p = _t_two_sided_p(12.706, 1.0)
    assert p == pytest.approx(0.05, abs=2e-3)


def test_t_critical_inverts_two_sided_p():
    df, alpha = 8.0, 0.05
    tc = _t_critical(df, alpha)
    # By construction p(tc, df) ~= alpha.
    assert _t_two_sided_p(tc, df) == pytest.approx(alpha, abs=1e-3)
    # df<=0 -> infinite critical value.
    assert _t_critical(0.0, 0.05) == float("inf")
    assert _t_critical(-1.0, 0.05) == float("inf")


# -- Experiment harness over a real app ---------------------------------------


@pytest.fixture()
def stat_app():
    from vincio import ContextApp
    from vincio.core.config import StorageConfig, VincioConfig
    from vincio.providers import MockProvider

    return ContextApp(
        name="statapp",
        config=VincioConfig(storage=StorageConfig(metadata="memory://")),
        provider=MockProvider(default_text="Refunds take 30 days."),
        model="mock-1",
    )


def _dataset():
    from vincio.evals import Dataset, EvalCase

    return Dataset(
        name="g",
        cases=[
            EvalCase(id="c1", input="refund window?", expected="30 days"),
            EvalCase(id="c2", input="renewal?", expected="60 days"),
        ],
    )


def test_experiment_run_variant_logs_and_compares(stat_app):
    exp = Experiment(stat_app, "ab", metrics=["lexical_overlap"])
    exp.run_variant("baseline", _dataset())
    exp.run_variant("fast", _dataset(), model="mock-fast")

    # Both variants logged under the tracker.
    variants = {r.variant for r in exp.tracker.runs("ab")}
    assert variants == {"baseline", "fast"}

    # model kwarg threaded into params via setdefault.
    fast_run = exp.tracker.latest("ab", "fast")
    assert fast_run.params["model"] == "mock-fast"

    comparison = exp.compare()
    assert set(comparison["variants"]) == {"baseline", "fast"}


def test_experiment_run_variant_restores_model_after_run(stat_app):
    original = stat_app.model
    exp = Experiment(stat_app, "restore", metrics=["lexical_overlap"])
    exp.run_variant("v1", _dataset(), model="some-other-model")
    assert stat_app.model == original


def test_experiment_run_variant_apply_hook_runs(stat_app):
    seen = {}

    def apply(app):
        seen["called"] = app is stat_app

    exp = Experiment(stat_app, "applyexp", metrics=["lexical_overlap"])
    exp.run_variant("v1", _dataset(), apply=apply)
    assert seen == {"called": True}


def test_experiment_run_variant_swaps_and_restores_prompt(stat_app):
    from vincio import PromptSpec

    original_prompt = stat_app.prompt_spec
    variant_prompt = PromptSpec(role="be terse", objective="answer briefly")
    assert variant_prompt is not original_prompt

    exp = Experiment(stat_app, "promptexp", metrics=["lexical_overlap"])
    exp.run_variant("terse", _dataset(), prompt=variant_prompt)

    # After the run the original prompt_spec is restored (had_prompt branch).
    assert stat_app.prompt_spec is original_prompt


def test_experiment_run_variant_explicit_params_preserved(stat_app):
    exp = Experiment(stat_app, "paramexp", metrics=["lexical_overlap"])
    # Caller-supplied params win; model only setdefault'd.
    exp.run_variant("v1", _dataset(), model="m2", params={"model": "explicit", "tag": "x"})
    run = exp.tracker.latest("paramexp", "v1")
    assert run.params == {"model": "explicit", "tag": "x"}


def test_experiment_cost_and_significance(stat_app):
    exp = Experiment(stat_app, "costexp", metrics=["lexical_overlap", "cost"])
    exp.run_variant("baseline", _dataset())
    exp.run_variant("cand", _dataset())

    costs = exp.cost()
    assert set(costs) == {"baseline", "cand"}
    assert all(isinstance(v, float) for v in costs.values())

    sig = exp.significance("lexical_overlap")
    # significance excludes the baseline itself, keyed by other variants.
    assert set(sig) == {"cand"}
    assert sig["cand"]["metric"] == "lexical_overlap"


def test_experiment_significance_raises_without_baseline(stat_app):
    exp = Experiment(stat_app, "nobase", metrics=["lexical_overlap"])
    exp.run_variant("only", _dataset(), params={"x": 1})
    with pytest.raises(EvalError, match="has no baseline variant 'baseline'"):
        exp.significance("lexical_overlap")


def test_experiment_default_tracker_uses_app_store(stat_app):
    exp = Experiment(stat_app, "deftracker")
    assert exp.tracker.store is stat_app.store
