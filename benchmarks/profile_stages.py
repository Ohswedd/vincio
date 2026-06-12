"""Per-stage pipeline profiler (0.2).

Runs a representative offline pipeline (ingest → retrieve → compile →
model → validate) and reports where the time goes, two ways:

1. **Stage breakdown** — wall-clock per pipeline stage, taken from the run's
   trace spans (input, retrieval, context_compile, prompt_render,
   model_call, output_validation, ...), aggregated over N runs.
2. **CPU profile** — an optional cProfile capture (``--cprofile out.prof``)
   for flamegraph rendering: open with ``snakeviz out.prof`` or render with
   ``flameprof out.prof > flame.svg``.

Usage::

    python benchmarks/profile_stages.py                # stage breakdown
    python benchmarks/profile_stages.py --runs 50
    python benchmarks/profile_stages.py --cprofile vincio.prof
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vinciobench import CORPUS, QA_CASES  # noqa: E402

from vincio import ContextApp, VincioConfig  # noqa: E402
from vincio.core.types import Document  # noqa: E402
from vincio.providers import MockProvider  # noqa: E402


def build_app() -> ContextApp:
    config = VincioConfig()
    config.storage.metadata = "memory://"
    config.observability.exporter = "memory"
    config.security.audit_log = False
    app = ContextApp(name="profile", provider=MockProvider(), model="mock-1", config=config)
    app.add_source(
        "corpus",
        documents=[Document(id=f"doc_{name}", title=name, text=text) for name, text in CORPUS],
    )
    return app


async def run_pipeline(app: ContextApp, runs: int) -> None:
    for index in range(runs):
        question = QA_CASES[index % len(QA_CASES)][0]
        await app.arun(question)


def stage_breakdown(app: ContextApp) -> dict[str, dict[str, float]]:
    durations: dict[str, list[float]] = defaultdict(list)
    totals: dict[str, float] = defaultdict(float)
    for trace in app.tracer.exporter.traces:
        for span in trace.spans:
            if span.end_time is None:
                continue
            # Microsecond-precise duration; span.duration_ms truncates to int.
            elapsed_ms = (span.end_time - span.start_time).total_seconds() * 1000
            durations[f"{span.type}:{span.name}"].append(elapsed_ms)
            totals[f"{span.type}:{span.name}"] += elapsed_ms
    grand_total = sum(totals.values()) or 1.0
    report = {}
    for stage, values in sorted(durations.items(), key=lambda kv: -totals[kv[0]]):
        report[stage] = {
            "calls": len(values),
            "total_ms": round(totals[stage], 2),
            "p50_ms": round(statistics.median(values), 3),
            "max_ms": round(max(values), 3),
            "share": round(totals[stage] / grand_total, 3),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile the Vincio pipeline per stage.")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--cprofile", metavar="OUT.prof", help="also capture a cProfile file")
    parser.add_argument("--json", action="store_true", help="emit the breakdown as JSON")
    args = parser.parse_args()

    app = build_app()
    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        asyncio.run(run_pipeline(app, args.runs))
        profiler.disable()
        profiler.dump_stats(args.cprofile)
        print(
            f"cProfile written to {args.cprofile} — render a flamegraph with "
            f"'snakeviz {args.cprofile}' or 'flameprof {args.cprofile} > flame.svg'",
            file=sys.stderr,
        )
    else:
        asyncio.run(run_pipeline(app, args.runs))

    report = stage_breakdown(app)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"\nPer-stage breakdown over {args.runs} runs (sorted by total time):\n")
    header = f"{'stage':<42} {'calls':>5} {'total_ms':>9} {'p50_ms':>8} {'max_ms':>8} {'share':>6}"
    print(header)
    print("-" * len(header))
    for stage, row in report.items():
        print(
            f"{stage:<42} {row['calls']:>5} {row['total_ms']:>9} {row['p50_ms']:>8} "
            f"{row['max_ms']:>8} {row['share']:>6.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
