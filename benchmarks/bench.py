"""The Vincio benchmark platform driver — one entry point, three tracks.

This is the thin driver over the library's benchmark system
(:mod:`vincio.evals.suite`). The logic lives in the library, tested and CI-gated;
this script just runs a track and saves the report to ``benchmarks/results/``. It
mirrors the ``vincio bench`` CLI.

    python benchmarks/bench.py feature            # track 3: Vincio features vs competitors (LIVE)
    python benchmarks/bench.py uplift             # track 2: the same model through Vincio vs direct
    python benchmarks/bench.py model              # track 1: a model on the public benchmarks (Tier-S)
    python benchmarks/bench.py all                # all three

Track 3 (feature) runs **genuinely live** against whatever competitor libraries are
installed (tiktoken, rank_bm25, pandas, …); a missing one is reported as skipped,
never fabricated. Tracks 1 and 2 run offline (Tier-S mockup) here; point them at a
live model with ``benchmarks/eval_live.py`` (track 1) or the ``UpliftSuite`` live
path (track 2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vincio.evals.suite import (
    BenchmarkSuite,
    FeatureSuite,
    SuiteReport,
    UpliftSuite,
    render_feature_report,
    render_uplift_report,
)

RESULTS = Path(__file__).parent / "results"


def run_feature() -> dict:
    run = FeatureSuite().run("all")
    print(render_feature_report(run))
    _save("feature_track", run.model_dump(mode="json"))
    live = sum(1 for r in run.runs if r.ran_live)
    print(f"feature track: {len(run.runs)} contests, {live} ran live vs a real competitor "
          f"(suite tier {run.tier.code})")
    return run.model_dump(mode="json")


def run_uplift() -> dict:
    run = UpliftSuite().run("all", tier="static")
    print(render_uplift_report(run))
    _save("uplift_track", run.model_dump(mode="json"))
    print(f"uplift track: overall {run.overall_direct():.3f} -> {run.overall_vincio():.3f} "
          f"({run.overall_delta():+.3f}), tier {run.tier.code}")
    return run.model_dump(mode="json")


def run_model() -> dict:
    run = BenchmarkSuite().run("all", tier="static")
    print(SuiteReport(run).to_markdown())
    _save("model_track", run.model_dump(mode="json", exclude={"created_at"}))
    print(f"model track: {len(run.runs)} benchmarks, overall {run.overall():.3f}, tier {run.tier.code}")
    return run.model_dump(mode="json", exclude={"created_at"})


def _save(name: str, payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}_latest.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    tracks = args or ["all"]
    runners = {"feature": run_feature, "uplift": run_uplift, "model": run_model}
    selected = list(runners) if "all" in tracks else [t for t in tracks if t in runners]
    if not selected:
        print(f"usage: python benchmarks/bench.py [{' | '.join(runners)} | all]", file=sys.stderr)
        return 2
    for track in selected:
        runners[track]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
