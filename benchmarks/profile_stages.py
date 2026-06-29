"""Per-stage pipeline profiler.

Runs a representative offline pipeline (ingest → retrieve → compile →
model → validate) and reports where the time goes, three ways:

1. **Stage breakdown** — wall-clock per pipeline stage, taken from the run's
   trace spans (input, retrieval, context_compile, prompt_render,
   model_call, output_validation, ...), aggregated over N runs.
2. **Before/after compile comparison** (``--compare``) — the single-pass
   feature arena's win, measured directly on the context compiler over a large,
   mostly-distinct candidate pool (the 10k-scale regime where the bounded global
   feature cache thrashes): the per-compile median latency with the optimization
   off then on, and the speedup the PerfBench `compile_speedup` floor gates.
3. **CPU profile** — an optional cProfile capture (``--cprofile out.prof``)
   for flamegraph rendering: open with ``snakeviz out.prof`` or render with
   ``flameprof out.prof > flame.svg``.

Usage::

    python benchmarks/profile_stages.py                # stage breakdown
    python benchmarks/profile_stages.py --runs 50
    python benchmarks/profile_stages.py --compare      # single-pass before/after
    python benchmarks/profile_stages.py --cprofile vincio.prof
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vinciobench import CORPUS, QA_CASES  # noqa: E402

from vincio import ContextApp, VincioConfig  # noqa: E402
from vincio.context.compiler import ContextCompiler, ContextCompilerOptions  # noqa: E402
from vincio.core.types import (  # noqa: E402
    Budget,
    Document,
    EvidenceItem,
    Objective,
    TaskType,
    UserInput,
)
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


def _large_pool(size: int) -> list[EvidenceItem]:
    """A large, mostly-distinct evidence pool — the regime where the bounded
    global term/shingle cache thrashes and the per-compile arena pays each
    derivation once."""
    base = [text for _name, text in CORPUS]
    return [
        EvidenceItem(
            id=f"sp{i}",
            source_id=f"sp_doc_{i}",
            text=f"{base[i % len(base)]} (clause {i}) renewal, refund, and termination terms {i}.",
            relevance=0.2,
        )
        for i in range(size)
    ]


def compile_comparison(pool_size: int, runs: int) -> dict[str, float]:
    """Median per-compile latency with the single-pass feature arena off then on,
    over a large pool, interleaved so a transient hiccup hits both equally."""
    objective = Objective("Answer policy questions", task_type=TaskType.DOCUMENT_QA)
    pool = _large_pool(pool_size)
    kwargs = dict(
        objective=objective,
        user_input=UserInput(text="What are the refund and renewal-notice terms?"),
        evidence=pool,
        budget=Budget(max_input_tokens=3000),
    )

    async def one(flag: bool) -> None:
        await ContextCompiler(
            ContextCompilerOptions(single_pass_selection=flag)
        ).compile(**kwargs)

    asyncio.run(one(True))  # warm imports/caches
    asyncio.run(one(False))
    on_t: list[float] = []
    off_t: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        asyncio.run(one(False))
        off_t.append(time.perf_counter() - started)
        started = time.perf_counter()
        asyncio.run(one(True))
        on_t.append(time.perf_counter() - started)
    off_ms = statistics.median(off_t) * 1000
    on_ms = statistics.median(on_t) * 1000
    return {
        "pool_size": pool_size,
        "off_p50_ms": round(off_ms, 2),
        "on_p50_ms": round(on_ms, 2),
        "speedup": round(off_ms / max(1e-9, on_ms), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile the Vincio pipeline per stage.")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--cprofile", metavar="OUT.prof", help="also capture a cProfile file")
    parser.add_argument("--json", action="store_true", help="emit the breakdown as JSON")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="single-pass feature arena before/after compile comparison on a large pool",
    )
    parser.add_argument(
        "--pool-size", type=int, default=6000, help="candidate pool size for --compare"
    )
    args = parser.parse_args()

    if args.compare:
        result = compile_comparison(args.pool_size, args.runs)
        if args.json:
            print(json.dumps(result, indent=2))
            return 0
        print(
            f"\nSingle-pass feature arena — context compile over {result['pool_size']} "
            f"candidates (median of {args.runs} interleaved runs):\n"
        )
        print(f"  off (per-pass derivation) p50 : {result['off_p50_ms']:>8.2f} ms")
        print(f"  on  (single-pass arena)   p50 : {result['on_p50_ms']:>8.2f} ms")
        print(f"  speedup                       : {result['speedup']:>8.2f}x")
        print(
            "\nSelection is byte-identical either way (PerfBench selection_byte_identical); "
            "the ratio floor gates the win.\n"
        )
        return 0

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
