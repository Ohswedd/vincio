"""The unified three-track benchmark platform: Model / Uplift / Feature.

Track 1 (Model) is covered by the plane's own tests; here we lock the two new
tracks and the shared track dimension end to end — honest tiers, real measured
deltas, custom registration, tier enforcement, and the report renderers.
"""

from __future__ import annotations

import pytest

from vincio.core.errors import EvalSuiteError
from vincio.evals.suite import (
    BenchmarkTrack,
    Contender,
    FeatureContest,
    FeatureMeasurement,
    FeatureSuite,
    ProvenanceTier,
    UpliftBenchmark,
    UpliftSuite,
    available_feature_contests,
    available_uplift_benchmarks,
    default_feature_registry,
    default_uplift_registry,
    register_feature_contest,
    register_uplift_benchmark,
    render_feature_report,
    render_uplift_report,
)


@pytest.fixture
def clean_default_registries():
    """Snapshot and restore the process-wide registries, so a test that exercises the
    public `register_*` extension points never leaks into other tests."""
    fr = default_feature_registry()
    ur = default_uplift_registry()
    f_snap, u_snap = dict(fr._contests), dict(ur._benchmarks)
    yield
    fr._contests.clear()
    fr._contests.update(f_snap)
    ur._benchmarks.clear()
    ur._benchmarks.update(u_snap)


# --------------------------------------------------------------------------- #
# The track dimension.
# --------------------------------------------------------------------------- #


def test_track_parse_and_labels():
    assert BenchmarkTrack.parse("feature") is BenchmarkTrack.FEATURE
    assert BenchmarkTrack.parse(BenchmarkTrack.UPLIFT) is BenchmarkTrack.UPLIFT
    assert BenchmarkTrack.MODEL.label == "Model"
    assert "public benchmarks" in BenchmarkTrack.MODEL.question
    with pytest.raises(EvalSuiteError):
        BenchmarkTrack.parse("nonsense")


# --------------------------------------------------------------------------- #
# Track 3 — Feature.
# --------------------------------------------------------------------------- #


def test_feature_suite_runs_every_builtin_contest():
    run = FeatureSuite().run("all")
    ids = {r.contest_id for r in run.runs}
    # The user-named capabilities are present.
    assert {"memory.recall", "retrieval.bm25"} <= ids
    assert len(run.runs) == len(available_feature_contests())
    for r in run.runs:
        assert r.winner  # every contest that ran picks a winner
        assert r.determinism_digest  # deterministic quality hash exists


def test_feature_memory_supersede_beats_naive_store():
    """Vincio's layered memory supersedes a contradicted fact — a real precision win."""
    run = FeatureSuite().run("memory.recall")
    r = run.runs[0]
    vincio = next(m for m in r.measurements if m.contender == "vincio")
    naive = next(m for m in r.measurements if m.contender == "naive_keyword_store")
    assert vincio.primary == 1.0          # returns only the current fact
    assert naive.primary < vincio.primary  # naive store also serves the stale one
    assert r.winner == "vincio"


def test_feature_tier_is_live_only_when_competitor_runs():
    """retrieval.bm25 has rank_bm25 installed here → a real head-to-head → Live."""
    r = FeatureSuite().run("retrieval.bm25").runs[0]
    assert r.ran_live is True
    assert r.tier is ProvenanceTier.LIVE
    # A vincio-vs-baseline-only contest never claims Live (no competitor ran).
    assembly = FeatureSuite().run("context.assembly").runs[0]
    assert assembly.ran_live is False
    assert assembly.tier is ProvenanceTier.STATIC


def test_feature_missing_competitor_is_skipped_not_fabricated():
    from vincio.evals.suite import FeatureRegistry

    def runner() -> list[Contender]:
        return [
            Contender("vincio", lambda: FeatureMeasurement(primary=1.0), kind="vincio"),
            Contender("ghostlib", lambda: FeatureMeasurement(primary=9.9),
                      kind="competitor", requires=("definitely_not_installed_xyz",)),
        ]

    contest = FeatureContest(id="custom.skip_demo", title="Skip demo", capability="custom",
                             primary_metric="score", runner=runner)
    # An isolated registry so the process-wide catalog (and the manifest) is untouched.
    reg = FeatureRegistry(with_builtins=False)
    reg.register(contest)
    r = FeatureSuite(registry=reg).run("custom.skip_demo").runs[0]
    ghost = next(m for m in r.measurements if m.contender == "ghostlib")
    assert ghost.available is False and "skipped" in ghost.note
    assert r.tier is ProvenanceTier.STATIC     # the head-to-head did not happen
    assert r.winner == "vincio"                # winner is the one that actually ran
    assert "custom.skip_demo" in reg.ids()


def test_register_feature_contest_extends_default_registry(clean_default_registries):
    """The public extension point adds to the default catalog (on a throwaway id)."""
    contest = FeatureContest(
        id="custom.ext_point", title="Ext", capability="custom", primary_metric="score",
        runner=lambda: [Contender("vincio", lambda: FeatureMeasurement(primary=1.0), kind="vincio")],
    )
    register_feature_contest(contest, replace=True)
    assert "custom.ext_point" in available_feature_contests()


def test_feature_contender_that_errors_at_runtime_is_skipped_not_crashed():
    """A competitor that is installed but throws at run time must degrade to a skip
    (and drop the contest to Static), never crash the suite."""
    from vincio.evals.suite import FeatureRegistry

    def boom() -> FeatureMeasurement:
        raise RuntimeError("competitor blew up at runtime")

    def runner() -> list[Contender]:
        return [
            Contender("vincio", lambda: FeatureMeasurement(primary=1.0), kind="vincio"),
            Contender("crashy", boom, kind="competitor"),  # importable, but run() raises
        ]

    reg = FeatureRegistry(with_builtins=False)
    reg.register(FeatureContest(id="custom.boom", title="Boom", capability="custom",
                                primary_metric="score", runner=runner))
    r = FeatureSuite(registry=reg).run("custom.boom").runs[0]
    crashy = next(m for m in r.measurements if m.contender == "crashy")
    assert crashy.available is False and "errored" in crashy.note
    assert r.tier is ProvenanceTier.STATIC   # the head-to-head did not complete
    assert r.winner == "vincio"              # the winner is what actually ran


def test_feature_report_renders_with_tier():
    md = render_feature_report(FeatureSuite().run(["retrieval.bm25", "memory.recall"]))
    assert "# Feature report" in md
    assert "retrieval.bm25" in md and "memory.recall" in md
    assert "Tiers —" in md  # the tier legend is present


# --------------------------------------------------------------------------- #
# Track 2 — Uplift.
# --------------------------------------------------------------------------- #


def test_uplift_suite_measures_direct_vs_vincio():
    run = UpliftSuite().run("all", tier="static")
    assert run.tier is ProvenanceTier.STATIC
    assert run.results
    # Every built-in uplift is an improvement in the mockup, and the overall is > 0.
    assert run.overall_delta() > 0
    for r in run.results:
        assert r.vincio >= r.direct
        assert r.delta == pytest.approx(round(r.vincio - r.direct, 4))


def test_uplift_grounding_and_containment_deltas():
    run = UpliftSuite().run(["rag.grounded", "safety.injection"], tier="static")
    by_id = {r.benchmark_id: r for r in run.results}
    assert by_id["rag.grounded"].direct < 1.0 and by_id["rag.grounded"].vincio == 1.0
    assert by_id["safety.injection"].direct == 0.0 and by_id["safety.injection"].vincio == 1.0
    assert by_id["safety.injection"].improved is True


def test_uplift_live_requires_both_targets():
    with pytest.raises(EvalSuiteError):
        UpliftSuite().run("all", tier="live")  # no direct/vincio targets supplied


def test_uplift_custom_benchmark_registration(clean_default_registries):
    from vincio.evals.suite import UpliftRegistry
    from vincio.evals.suite.uplift_builtin import SchemaValidAdapter

    bench = UpliftBenchmark(
        id="custom.uplift", title="Custom uplift", adapter=SchemaValidAdapter,
        primary_metric="valid_rate",
        tasks=[{"id": "t1", "gold": ["k"], "recorded": "{bad", "recorded_vincio": '{"k": 1}'}],
    )
    reg = UpliftRegistry(with_builtins=False)
    reg.register(bench)
    r = UpliftSuite(registry=reg).run("custom.uplift", tier="static").results[0]
    assert r.direct == 0.0 and r.vincio == 1.0
    assert "custom.uplift" in reg.ids()
    # The public extension point reaches the default registry too.
    register_uplift_benchmark(bench, replace=True)
    assert "custom.uplift" in available_uplift_benchmarks()


def test_uplift_report_renders_with_deltas():
    md = render_uplift_report(UpliftSuite().run("all", tier="static"))
    assert "# Uplift report" in md
    assert "Through Vincio" in md and "Δ" in md


# --------------------------------------------------------------------------- #
# Determinism — the mockup tiers are byte-identical across runs.
# --------------------------------------------------------------------------- #


def test_mockup_tracks_are_deterministic():
    a = UpliftSuite().run("all", tier="static").determinism_digest
    b = UpliftSuite().run("all", tier="static").determinism_digest
    assert a == b
    # Feature quality digest excludes latency, so it is stable too.
    fa = FeatureSuite().run(["memory.recall", "output.json_repair"]).runs
    fb = FeatureSuite().run(["memory.recall", "output.json_repair"]).runs
    assert [r.determinism_digest for r in fa] == [r.determinism_digest for r in fb]


def test_registries_are_populated():
    assert default_feature_registry().ids()
    assert default_uplift_registry().ids()


# --------------------------------------------------------------------------- #
# The unified `vincio bench` CLI.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("argv", [
    ["bench", "list"],
    ["bench", "feature", "retrieval.bm25"],
    ["bench", "uplift", "rag.grounded", "--tier", "static"],
    ["bench", "feature", "memory.recall", "--format", "markdown"],
])
def test_bench_cli_smoke(argv, capsys):
    from vincio.cli.main import main

    assert main(argv) == 0
    out = capsys.readouterr().out
    assert out.strip()  # produced output


def test_bench_cli_json(capsys):
    from vincio.cli.main import main

    assert main(["bench", "list", "--json"]) == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"model", "uplift", "feature"}
    assert payload["feature"] and payload["uplift"]


# --------------------------------------------------------------------------- #
# Honesty contract, tier consistency, and error paths (audit regressions).
# --------------------------------------------------------------------------- #


def test_uplift_refuses_recorded_tier():
    """A fabricated two-arm mockup may not print a Recorded label."""
    from vincio.core.errors import TierViolationError

    with pytest.raises(TierViolationError):
        UpliftSuite().run("all", tier="recorded")


def test_uplift_run_tier_never_contradicts_result_tiers():
    run = UpliftSuite().run("all", tier="static")
    assert all(r.tier == run.tier for r in run.results)
    assert run.tier is ProvenanceTier.STATIC
    assert run.run_id.startswith("uplift_")


def test_uplift_respects_lower_is_better_direction():
    """For a lower-is-better metric, a *drop* is an improvement — not the raw sign."""
    from vincio.evals.suite import UpliftRegistry
    from vincio.evals.suite.uplift_builtin import SchemaValidAdapter

    # A benchmark where the vincio arm scores 0 and direct scores 1 (valid), but
    # lower-is-better means the vincio arm improved.
    bench = UpliftBenchmark(
        id="c.lower", title="Lower", adapter=SchemaValidAdapter, primary_metric="errors",
        higher_is_better=False,
        tasks=[{"id": "t", "gold": ["k"], "recorded": '{"k": 1}', "recorded_vincio": "{bad"}],
    )
    reg = UpliftRegistry(with_builtins=False)
    reg.register(bench)
    r = UpliftSuite(registry=reg).run("c.lower", tier="static").results[0]
    assert r.direct == 1.0 and r.vincio == 0.0
    assert r.delta == -1.0
    assert r.improved is True and r.regressed is False  # lower is better → a drop improves


def test_feature_lower_is_better_winner_is_smallest():
    """encoding.tabular is lower-is-better (tokens); the winner is the smallest count."""
    r = FeatureSuite().run("encoding.tabular").runs[0]
    assert r.higher_is_better is False
    ran = [m for m in r.measurements if m.available]
    assert r.winner == min(ran, key=lambda m: m.primary).contender
    assert r.winner == "vincio"


def test_feature_suite_aggregate_tier_reflects_all_contests():
    from vincio.evals.suite import FeatureRegistry

    reg = FeatureRegistry(with_builtins=False)
    reg.register(FeatureContest(
        id="c.live", title="Live", capability="custom", primary_metric="score",
        runner=lambda: [Contender("vincio", lambda: FeatureMeasurement(primary=1.0), kind="vincio"),
                        Contender("json", lambda: FeatureMeasurement(primary=0.5), kind="competitor")],
    ))
    reg.register(FeatureContest(
        id="c.static", title="Static", capability="custom", primary_metric="score",
        runner=lambda: [Contender("vincio", lambda: FeatureMeasurement(primary=1.0), kind="vincio")],
    ))
    suite = FeatureSuite(registry=reg)
    assert suite.run("c.live").tier is ProvenanceTier.LIVE       # its only competitor ran
    assert suite.run("all").tier is ProvenanceTier.STATIC        # c.static has no competitor


def test_contender_rejects_unknown_kind():
    with pytest.raises(EvalSuiteError):
        Contender("x", lambda: FeatureMeasurement(), kind="bogus")


def test_registries_reject_duplicate_and_unknown():
    from vincio.evals.suite import FeatureRegistry, UpliftRegistry

    fr = FeatureRegistry(with_builtins=False)
    c = FeatureContest(id="c.dup", title="Dup", capability="custom", primary_metric="score",
                       runner=lambda: [Contender("vincio", lambda: FeatureMeasurement(), kind="vincio")])
    fr.register(c)
    with pytest.raises(EvalSuiteError):
        fr.register(c)  # duplicate id
    with pytest.raises(EvalSuiteError):
        FeatureSuite(registry=fr).run("does.not.exist")
    with pytest.raises(EvalSuiteError):
        UpliftRegistry(with_builtins=False).get("nope")


def test_uplift_build_adapter_accepts_a_factory_function():
    """The adapter may be a zero-arg factory, not only a class."""
    from vincio.evals.suite import UpliftRegistry
    from vincio.evals.suite.uplift_builtin import SchemaValidAdapter

    bench = UpliftBenchmark(
        id="c.factory", title="Factory", adapter=lambda: SchemaValidAdapter([]),
        primary_metric="valid_rate",
        tasks=[{"id": "t", "gold": ["k"], "recorded": "{bad", "recorded_vincio": '{"k": 1}'}],
    )
    reg = UpliftRegistry(with_builtins=False)
    reg.register(bench)
    r = UpliftSuite(registry=reg).run("c.factory", tier="static").results[0]
    assert r.direct == 0.0 and r.vincio == 1.0


def test_reports_render_skipped_and_regressed_branches():
    from vincio.evals.suite import FeatureRegistry

    # A feature report with a skipped competitor exercises the skipped row.
    reg = FeatureRegistry(with_builtins=False)
    reg.register(FeatureContest(
        id="c.skip", title="Skip", capability="custom", primary_metric="score",
        runner=lambda: [Contender("vincio", lambda: FeatureMeasurement(primary=1.0), kind="vincio"),
                        Contender("ghost", lambda: FeatureMeasurement(),
                                  kind="competitor", requires=("nope_xyz",))],
    ))
    md = render_feature_report(FeatureSuite(registry=reg).run("all"))
    assert "skipped" in md and "ghost" in md

    # An uplift report with a regression exercises the ▼ branch.
    from vincio.evals.suite import UpliftRegistry
    from vincio.evals.suite.uplift_builtin import SchemaValidAdapter

    ureg = UpliftRegistry(with_builtins=False)
    ureg.register(UpliftBenchmark(
        id="c.regress", title="Regress", adapter=SchemaValidAdapter, primary_metric="valid_rate",
        tasks=[{"id": "t", "gold": ["k"], "recorded": '{"k": 1}', "recorded_vincio": "{bad"}],
    ))
    umd = render_uplift_report(UpliftSuite(registry=ureg).run("all", tier="static"))
    assert "▼" in umd
