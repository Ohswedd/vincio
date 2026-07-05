"""The open evaluation plane — the standard public benchmarks, one harness.

One pluggable plane for running the standard public model benchmarks (MMLU, GPQA,
GSM8K, HumanEval, IFEval, RULER, …), grouped by niche, scored by reusable metrics,
reported identically for every model. It is in-process and offline-reproducible.

The load-bearing idea is the **provenance tier** stamped on every number — an
enforced contract, not a convention:

  * **S — Static**   a bundled *fabricated* fixture exercising adapter + metric (gates CI).
  * **R — Recorded** a hash-pinned real-dataset slice replayed vs recorded outputs (gates CI).
  * **L — Live**     the full public dataset vs a live model (reported, never gated).

The engine refuses to let a lower tier print a higher tier's label, so a fabricated
fixture can never masquerade as a live score. Runs fully offline on the Tier-S fixtures.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _shared import example_provider

from vincio import (
    BenchmarkDataset,
    BenchmarkSpec,
    BenchmarkSuite,
    ContextApp,
    Leaderboard,
    ProvenanceTier,
    RunStore,
    SuiteReport,
    register_benchmark,
)
from vincio.core.errors import TierViolationError
from vincio.evals.suite import default_benchmark_registry, leaderboard_chart, radar_chart
from vincio.evals.suite.adapters import MMLUAdapter


def main() -> None:
    provider, model = example_provider()
    app = ContextApp(name="open_eval", provider=provider, model=model)

    # 1. Run a niche from the app. Every run carries its tier; a Tier-S run gates CI
    #    ('static' = fabricated fixtures), so this is the number you can block a merge on.
    run = app.benchmark_suite("knowledge", tier="static")
    print(f"1. Knowledge niche · {len(run.runs)} benchmarks · overall {run.overall():.3f} "
          f"· tier {run.tier.code} ({'gates CI' if run.gated else 'reported only'})")
    for r in sorted(run.runs, key=lambda r: r.benchmark_id):
        print(f"   {r.benchmark_id:22s} {r.primary_metric:10s} {r.primary:.3f}")

    # 2. The catalog: every benchmark sits behind one BenchmarkAdapter contract, so
    #    adding a niche never touches the harness. Group by niche to see the surface.
    registry = default_benchmark_registry()
    print("\n2. Catalog — niches behind one adapter contract")
    for niche, specs in registry.niches().items():
        print(f"   {niche:14s} {', '.join(s.name for s in specs)}")

    # 3. Determinism: a Tier-S run is byte-identical across processes (the digest is
    #    the reproducibility receipt) — CI can assert the digest, not just the score.
    suite = BenchmarkSuite()
    a, b = suite.run("all", tier="static"), suite.run("all", tier="static")
    print(f"\n3. Determinism — digest stable: {a.determinism_digest == b.determinism_digest}")

    # 4. Tier integrity: asking a fabricated fixture to report as 'live' is refused —
    #    this is the enforcement that keeps the tier labels honest.
    try:
        suite.run("knowledge.mmlu", tier="live")
    except TierViolationError as exc:
        print(f"4. Tier integrity — refused: {str(exc).split('—')[0].strip()}")

    # 5. Long-context uplift: RULER is run with and without the governor, so the
    #    benchmark measures *Vincio's* contribution, not just the base model.
    ruler = suite.run("long_context.ruler", tier="static").runs[0]
    g = ruler.governed
    print(f"5. RULER uplift — base {g['base']:.3f} → governed {g['governed']:.3f} ({g['uplift']:+.3f})")

    # 6. Reporting: one run renders to every format with the tier on every number.
    report = SuiteReport(a)
    tier_tagged = '"tier": "S"' in report.to_json()
    print(f"6. Reporting — markdown/csv/json render; json carries the tier: {tier_tagged}")

    # 7. Leaderboard & charts: rank models and emit deterministic Vega-Lite (no
    #    plotting dependency) so results drop straight into a report or the docs.
    run_a = suite.run("knowledge", tier="static", model="A")
    run_b = suite.run("knowledge", tier="static", model="B")
    board = Leaderboard.from_runs([run_a, run_b])
    print(f"7. Leaderboard — ranked {len(board.rows)} models, #1={board.rows[0].model}; "
          f"{leaderboard_chart(board).kind}+{radar_chart(run_a).kind} charts (Vega-Lite JSON)")

    # 8. The run store persists runs so you can diff a model across versions —
    #    the regression signal you actually gate releases on.
    with tempfile.TemporaryDirectory() as tmp:
        store = RunStore(Path(tmp) / "runs.db")
        store.save(suite.run("knowledge", tier="static", model="A"), version="1.0")
        store.save(suite.run(["knowledge", "reasoning"], tier="static", model="A"), version="2.0")
        diff = store.model_version_diff("A")
        print(f"8. Run store — v1→v2 overall {diff['overall_from']:.3f} → {diff['overall_to']:.3f}")
        store.close()

    # 9. Extend it: register your own benchmark (reusing a stock adapter/metric)
    #    with no change to the core — the whole plane is open at the edges.
    register_benchmark(
        BenchmarkSpec(
            id="custom.house_style", niche="custom", title="House style",
            adapter=MMLUAdapter, primary_metric="accuracy",
            static_tasks=[{"id": "h1", "prompt": "Pick A.", "gold": "A",
                           "inputs": {"options": ["yes", "no", "maybe", "later"]},
                           "recorded": "The answer is (A)."}],
        ),
        replace=True,
    )
    custom = BenchmarkSuite().run("custom.house_style", tier="static")
    print(f"9. Custom benchmark scored {custom.runs[0].primary:.3f}")

    # Point a Recorded/Live run at a real dataset with BenchmarkDataset (and, for
    # live, VINCIO_PROVIDER); see docs/guides/run-benchmark-suite.md.
    _ = BenchmarkDataset, ProvenanceTier  # referenced for the reachability gate
    print("\nDone — every number above carried its provenance tier.")


if __name__ == "__main__":
    main()
