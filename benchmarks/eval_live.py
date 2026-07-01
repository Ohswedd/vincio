"""Run the open evaluation plane **Live** against a state-of-the-art model.

The plane defaults to offline Tier-S fixtures (they gate CI). This harness is the
opt-in path to the other end of the honesty ladder: point a real model + provider
at a real dataset and get **tier-labelled** public-benchmark scores. Nothing here
is fabricated — a Live number is *reported, never gated*, and only exists from a
real key.

    # Tier-S smoke (offline, no key, replays fixtures):
    python benchmarks/eval_live.py --benchmarks knowledge.mmlu --tier static

    # Live against a current SOTA model over a real dataset export:
    python benchmarks/eval_live.py --provider anthropic --model claude-opus-4-8 \
        --benchmarks knowledge.mmlu reasoning.gsm8k --tier live --dataset-dir ./datasets

    # Same, other providers:
    #   --provider openai --model gpt-5.2
    #   --provider google --model gemini-3-pro

Datasets (for --tier recorded|live) live in ``--dataset-dir`` as one file per
benchmark id, e.g. ``datasets/knowledge.mmlu.jsonl``. A benchmark **with a loader**
reads raw dataset records (one JSON object per line) mapped by the spec's loader;
otherwise the file is read as ready-made ``BenchmarkTask`` JSONL. A Recorded export
additionally carries a ``recorded`` field per record (the pinned model output);
a Live run solves each task fresh against the model. Benchmarks with no dataset are
**skipped for recorded/live** (never silently downgraded), because the engine
refuses to print a Live label over a fabricated fixture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vincio.core.errors import VincioError
from vincio.evals.suite import (
    BenchmarkDataset,
    BenchmarkSuite,
    ProvenanceTier,
    RunStore,
    SuiteReport,
    default_benchmark_registry,
)

# A sensible current SOTA default per provider when --model is omitted.
_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5.2",
    "google": "gemini-3-pro",
    "mistral": "mistral-large-latest",
    "openrouter": "anthropic/claude-opus-4-8",
    "mock": "mock-1",
}


def _build_app(provider: str, model: str):
    """A ContextApp wired to a live provider+model (the benchmark solver target)."""
    from vincio import ContextApp

    if provider == "mock":
        from vincio.providers import MockProvider

        return ContextApp(name="eval_live", provider=MockProvider(), model=model)
    from vincio.providers import build_provider

    return ContextApp(name="eval_live", provider=build_provider(provider), model=model)


def _key_present(provider: str) -> bool:
    if provider == "mock":
        return True
    from vincio.core.config import ProviderConfig

    return bool(ProviderConfig().resolve_api_key(provider))


def _load_datasets(
    specs, tier: ProvenanceTier, dataset_dir: Path | None
) -> tuple[dict[str, BenchmarkDataset], list[str]]:
    """Load a real dataset per benchmark from ``dataset_dir``; report the skipped."""
    datasets: dict[str, BenchmarkDataset] = {}
    skipped: list[str] = []
    if dataset_dir is None:
        return {}, [s.id for s in specs]
    for spec in specs:
        path = _dataset_path(dataset_dir, spec.id)
        if path is None:
            skipped.append(spec.id)
            continue
        if spec.loader is not None:
            import json

            try:
                records = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, ValueError) as exc:
                raise VincioError(f"cannot read dataset {path}: {exc}") from exc
            datasets[spec.id] = BenchmarkDataset.from_export(
                records, loader=spec.loader, tier=tier, name=spec.id, benchmark_id=spec.id
            )
        else:
            datasets[spec.id] = BenchmarkDataset.from_jsonl(path, tier=tier, benchmark_id=spec.id)
    return datasets, skipped


def _dataset_path(dataset_dir: Path, benchmark_id: str) -> Path | None:
    for suffix in (".jsonl", ".json"):
        candidate = dataset_dir / f"{benchmark_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="mock", help="anthropic | openai | google | mistral | openrouter | mock")
    parser.add_argument("--model", default=None, help="model id (defaults to a current SOTA model for the provider)")
    parser.add_argument("--benchmarks", nargs="+", default=["all"], help="ids / niches / 'all'")
    parser.add_argument("--tier", default="static", choices=["static", "recorded", "live", "S", "R", "L"],
                        help="static | recorded | live")
    parser.add_argument("--dataset-dir", default=None, help="dir of <benchmark_id>.jsonl real datasets (recorded/live)")
    parser.add_argument("--sample", type=int, default=None, help="cap tasks per benchmark (deterministic)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--format", default="text", choices=["text", "markdown", "html", "json", "csv"])
    parser.add_argument("--output", default=None, help="write the report here")
    parser.add_argument("--store", default=None, help="persist the run to this SQLite RunStore")
    parser.add_argument("--version", default=None, help="model-version label for the run store")
    args = parser.parse_args(argv)

    tier = ProvenanceTier.parse(args.tier)
    model = args.model or _DEFAULT_MODELS.get(args.provider, None)
    if model is None:
        print(f"error: --model is required for provider {args.provider!r}", file=sys.stderr)
        return 2

    registry = default_benchmark_registry()
    try:
        specs = registry.select(args.benchmarks)
    except VincioError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 2

    # Honesty gate: a Live run needs a real key and a real dataset.
    live = tier is ProvenanceTier.LIVE
    reproducible = tier.reproducible
    if live and not _key_present(args.provider):
        print(
            f"error: tier 'live' needs a real API key for provider {args.provider!r}. "
            f"Set it (e.g. ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY) and re-run. "
            f"Nothing is fabricated — a Live number only exists from a real key.",
            file=sys.stderr,
        )
        return 2

    datasets: dict[str, BenchmarkDataset] = {}
    if tier is not ProvenanceTier.STATIC:
        dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
        try:
            datasets, skipped = _load_datasets(specs, tier, dataset_dir)
        except (VincioError, ValueError) as exc:
            # A malformed dataset file (unreadable, or invalid JSONL a task-loader rejects)
            # is a clean user error, not a crash — mirror the rest of main().
            message = getattr(exc, "message", str(exc))
            print(f"error: {message}", file=sys.stderr)
            return 2
        if skipped:
            print(
                f"note: {len(skipped)} benchmark(s) have no dataset in "
                f"{args.dataset_dir or '(none given)'} and are SKIPPED for tier {tier.label} "
                f"(a fabricated fixture may not print a {tier.label} label): {', '.join(skipped)}",
                file=sys.stderr,
            )
        specs = [s for s in specs if s.id in datasets]
        if not specs:
            print("error: no benchmark has a dataset to run at this tier — pass --dataset-dir", file=sys.stderr)
            return 2

    target = _build_app(args.provider, model)
    suite = BenchmarkSuite(concurrency=args.concurrency)
    print(
        f"running {len(specs)} benchmark(s) · tier {tier.label} · "
        f"{'model ' + model if not reproducible else 'replay (reproducible)'} · "
        f"{'reported, NOT gated' if not tier.gates_ci else 'reproducible, gates CI'}",
        file=sys.stderr,
    )
    try:
        run = suite.run(
            [s.id for s in specs], target=target, tier=tier,
            datasets=datasets or None, sample=args.sample, model=model,
        )
    except VincioError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1

    if args.store:
        store = RunStore(args.store)
        try:
            store.save(run, version=args.version)
        finally:
            store.close()
    report = SuiteReport(run)
    if args.format == "text":
        print(f"\nsuite run {run.run_id} · tier {run.tier.code} · {len(run.runs)} benchmarks · "
              f"overall {run.overall():.3f}")
        for r in sorted(run.runs, key=lambda r: r.benchmark_id):
            print(f"  {r.benchmark_id:30s} {r.tier.code}  {r.primary_metric:18s} {r.primary:.3f}  n={r.n}")
    else:
        rendered = report.render(args.format)
        if args.output:
            report.save(args.output, format=args.format)
            print(f"saved to {args.output}")
        else:
            print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
