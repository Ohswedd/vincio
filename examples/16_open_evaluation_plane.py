"""The open evaluation plane — the standard public benchmarks, one harness.

Vincio already ships an evaluation subsystem (golden datasets, 30+ metrics,
calibrated judges) and a three-tier internal benchmark suite. This tour is the
*other* half: one coherent, pluggable plane for running the **standard public
model benchmarks** — MMLU, GPQA, GSM8K, HumanEval, IFEval, TruthfulQA, RULER, … —
grouped by niche, scored by reusable metrics, and reported the same way for every
model and every model version. It is in-process and offline-reproducible; it never
becomes a hosted leaderboard.

The organizing idea is the **provenance tier** on every number, an enforced
contract, not a convention:

  * **S — Static**   a small, bundled, *fabricated* fixture that exercises the
                     adapter + metric end to end (reproducible, gates CI).
  * **R — Recorded** a hash-pinned slice of the *real* dataset replayed against
                     recorded model outputs (reproducible, gates CI).
  * **L — Live**     the full public dataset against a live model (reported,
                     never gated).

The engine *refuses* to let a lower tier print a higher tier's label, so a
fabricated fixture can never masquerade as a live score. Everything below runs
fully offline on the bundled Tier-S fixtures.

Sections:
  1.  Run a niche and read the tiered results.
  2.  The catalog — eleven niches behind one BenchmarkAdapter contract.
  3.  Determinism — a Tier-S run is byte-identical.
  4.  Tier integrity — a static fixture cannot be reported live.
  5.  Long-context uplift — RULER is run twice, with and without the governor.
  6.  Reporting — one run rendered to Markdown / HTML / JSON / CSV.
  7.  Leaderboard & visualization — rank models, chart the breakdown.
  8.  The run store — persist runs and diff a model version.
  9.  Extend it — register your own benchmark without touching the core.
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

    # 1. Run a niche and read the tiered results -----------------------------
    print("1. Run the Knowledge niche (Tier-S, fully offline)")
    run = app.benchmark_suite("knowledge", tier="static")
    print(f"   {len(run.runs)} benchmarks · overall primary {run.overall():.3f} "
          f"· tier {run.tier.code} ({'gates CI' if run.gated else 'reported only'})")
    for r in sorted(run.runs, key=lambda r: r.benchmark_id):
        print(f"     {r.benchmark_id:24s} {r.primary_metric:10s} {r.primary:.3f}  [tier {r.tier.code}]")

    # 2. The catalog ---------------------------------------------------------
    print("\n2. The catalog — eleven niches, one contract")
    registry = default_benchmark_registry()
    for niche, specs in registry.niches().items():
        print(f"   {niche:14s} {', '.join(s.name for s in specs)}")

    # 3. Determinism ---------------------------------------------------------
    print("\n3. Determinism — a Tier-S run is byte-identical")
    suite = BenchmarkSuite()
    a = suite.run("all", tier="static")
    b = suite.run("all", tier="static")
    print(f"   digest A == digest B: {a.determinism_digest == b.determinism_digest} "
          f"({a.determinism_digest})")

    # 4. Tier integrity ------------------------------------------------------
    print("\n4. Tier integrity — a fabricated fixture cannot be reported live")
    try:
        suite.run("knowledge.mmlu", tier="live")
    except TierViolationError as exc:
        print(f"   refused: {str(exc).split('—')[0].strip()}")

    # 5. Long-context uplift -------------------------------------------------
    print("\n5. Long-context uplift — RULER run twice (with & without the governor)")
    ruler = suite.run("long_context.ruler", tier="static").runs[0]
    assert ruler.governed is not None
    print(f"   base {ruler.governed['base']:.3f} → governed {ruler.governed['governed']:.3f} "
          f"(uplift {ruler.governed['uplift']:+.3f})")

    # 6. Reporting -----------------------------------------------------------
    print("\n6. Reporting — one run, every format, every number tiered")
    report = SuiteReport(a)
    print("   markdown head:", report.to_markdown().splitlines()[0])
    print("   csv header:   ", report.to_csv().splitlines()[0])
    print("   json tier:    ", '"tier": "S"' in report.to_json())

    # 7. Leaderboard & visualization -----------------------------------------
    print("\n7. Leaderboard & visualization")
    run_a = suite.run("knowledge", tier="static", model="model-A")
    run_b = suite.run("knowledge", tier="static", model="model-B")
    board = Leaderboard.from_runs([run_a, run_b])
    print(f"   ranked {len(board.rows)} models; #1 = {board.rows[0].model}")
    chart = leaderboard_chart(board)
    radar = radar_chart(run_a)
    print(f"   {chart.kind} + {radar.kind} charts → deterministic Vega-Lite JSON "
          f"({len(chart.to_json())} bytes)")

    # 8. The run store -------------------------------------------------------
    print("\n8. The run store — persist & diff a model version")
    with tempfile.TemporaryDirectory() as tmp:
        store = RunStore(Path(tmp) / "runs.db")
        store.save(run_a, version="1.0")
        store.save(suite.run(["knowledge", "reasoning"], tier="static", model="model-A"),
                   version="2.0")
        diff = store.model_version_diff("model-A")
        print(f"   versions diffed: overall {diff['overall_from']:.3f} → {diff['overall_to']:.3f}")
        store.close()

    # 9. Extend it -----------------------------------------------------------
    print("\n9. Extend it — register your own benchmark, no core change")
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
    print(f"   custom.house_style scored {custom.runs[0].primary:.3f}")

    # The plane never touches a database or a network on the Tier-S path. Point
    # a Recorded/Live run at a real dataset with BenchmarkDataset (and, for live,
    # a real model via VINCIO_PROVIDER); see docs/guides/run-benchmark-suite.md.
    _ = BenchmarkDataset, ProvenanceTier  # referenced for the reachability gate
    print("\nDone — every number above carried its provenance tier.")


if __name__ == "__main__":
    main()
