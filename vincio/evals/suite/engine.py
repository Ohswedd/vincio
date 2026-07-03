"""The core engine — deterministic, concurrent, resumable benchmark execution.

:class:`BenchmarkSuite` runs one or many benchmarks over a model (or a
Vincio-wrapped app) at a chosen provenance tier and folds the results into a
:class:`~vincio.evals.suite.results.SuiteRun`. It composes the shipped
:class:`~vincio.evals.benchmarks.BenchmarkAdapter` ``replay`` / ``run`` paths, the
bounded fan-out helper, and durable-graph-style checkpointing:

* **Tier enforcement.** The engine resolves the tier a run may honestly claim from
  the dataset's provenance and whether the solver is live, and **refuses** a lower
  tier trying to print a higher tier's label.
* **Deterministic & reproducible.** Tier-S / Tier-R runs replay recorded outputs
  through the real scorer, with seeded sampling, so a re-run is byte-identical
  (the :attr:`~vincio.evals.suite.results.SuiteRun.determinism_digest` is stable).
* **Resumable.** Per-benchmark results are checkpointed; a resumed run skips
  benchmarks already completed against the same task-set pin.
* **Long-context uplift, measured.** On the reproducible tiers a ``long_context``
  benchmark is run **twice — with and without the context governor** (the governed
  variant replays the recorded with-governor transcript) — and the uplift recorded,
  never assumed. A live governor toggle is a two-run comparison the caller drives.
"""

from __future__ import annotations

import asyncio
import json
import platform as _platform
import sys
import time
from pathlib import Path
from typing import Any

from ...core.concurrency import gather_bounded
from ...core.errors import EvalSuiteError
from ...core.utils import compact_hash
from ...providers.base import run_sync
from ..benchmarks import BenchmarkReport, BenchmarkTask, build_agent_solver
from .datasets import BenchmarkDataset
from .metrics import summarize_results
from .registry import BenchmarkRegistry, BenchmarkSpec, default_benchmark_registry
from .results import BenchmarkRun, ItemResult, SuiteRun
from .tiers import ProvenanceTier, resolve_tier

__all__ = ["BenchmarkSuite"]


def _suite_environment() -> dict[str, Any]:
    """Reproducibility metadata — wall-clock excluded so the report is diff-stable."""
    from ... import __version__

    return {
        "schema_version": "1.0",
        "vincio_version": __version__,
        "python_version": _platform.python_version(),
        "platform": sys.platform,
        "deterministic": True,
    }


class BenchmarkSuite:
    """Run benchmarks over a model or app, deterministically and resumably.

    ``registry`` defaults to the process-wide catalog. ``concurrency`` bounds the
    per-benchmark fan-out. ``checkpoint_dir`` (when set) persists per-benchmark
    results so a crashed run resumes; ``seed`` makes sampling reproducible.
    """

    def __init__(
        self,
        registry: BenchmarkRegistry | None = None,
        *,
        concurrency: int = 8,
        seed: int = 42,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.registry = registry or default_benchmark_registry()
        self.concurrency = max(1, concurrency)
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._checkpoint_lock = asyncio.Lock()

    # -- public API -----------------------------------------------------------

    async def arun(
        self,
        benchmarks: str | list[str],
        *,
        target: Any | None = None,
        model: str | None = None,
        tier: ProvenanceTier | str = ProvenanceTier.STATIC,
        sample: int | None = None,
        datasets: dict[str, BenchmarkDataset] | None = None,
        solver_mode: str | None = None,
        resume: bool = False,
    ) -> SuiteRun:
        """Run the selected benchmarks and return a :class:`SuiteRun`.

        ``benchmarks`` is an id (``"knowledge.mmlu"``), a niche (``"knowledge"``),
        ``"all"``, or a list of these. ``tier`` selects the provenance tier: the
        default :attr:`~ProvenanceTier.STATIC` replays the bundled fixtures fully
        offline; ``"recorded"`` / ``"live"`` require a per-benchmark
        :class:`~vincio.evals.suite.datasets.BenchmarkDataset` in ``datasets`` (and,
        for live, a ``target`` model/app). ``sample`` caps tasks per benchmark
        deterministically; ``resume`` reuses checkpointed benchmark results.
        ``solver_mode`` (``"text"`` / ``"calls"``) overrides each benchmark's own
        grading mode for a live run — leave it ``None`` to grade every benchmark the
        way its spec declares (so a mixed suite grades call-based and text-based
        benchmarks each correctly).
        """
        names = [benchmarks] if isinstance(benchmarks, str) else list(benchmarks)
        specs = self.registry.select(names)
        if not specs:
            raise EvalSuiteError(f"no benchmarks matched {names!r}")
        requested = ProvenanceTier.parse(tier)
        model_label = model or self._model_label(target, requested)
        run_id = self._run_id(specs, requested, model_label, sample)
        checkpoint = self._load_checkpoint(run_id) if resume else {}

        coros = [
            self._run_one(
                spec, requested, target=target, model_label=model_label,
                datasets=datasets or {}, sample=sample, solver_mode=solver_mode,
                checkpoint=checkpoint, run_id=run_id,
            )
            for spec in specs
        ]
        runs = await gather_bounded(coros, limit=self.concurrency)
        suite = SuiteRun(
            run_id=run_id, model=model_label,
            provider=str(getattr(target, "provider_name", "") or ("replay" if requested.reproducible else "")),
            tier=requested, environment=_suite_environment(),
            runs=[r for r in runs if r is not None],
            metadata={"seed": self.seed, "sample": sample, "requested_tier": requested.value},
        )
        return suite

    def run(self, benchmarks: str | list[str], **kwargs: Any) -> SuiteRun:
        """Synchronous wrapper over :meth:`arun`."""
        return run_sync(self.arun(benchmarks, **kwargs))

    # -- per-benchmark --------------------------------------------------------

    async def _run_one(
        self,
        spec: BenchmarkSpec,
        requested: ProvenanceTier,
        *,
        target: Any | None,
        model_label: str,
        datasets: dict[str, BenchmarkDataset],
        sample: int | None,
        solver_mode: str | None,
        checkpoint: dict[str, dict[str, Any]],
        run_id: str,
    ) -> BenchmarkRun:
        dataset = self._dataset_for(spec, requested, datasets)
        if sample is not None and sample > 0:
            dataset = dataset.sample(sample, seed=self.seed)

        # A resumed benchmark with the same task-set pin is reused verbatim.
        cached = checkpoint.get(spec.id)
        if cached is not None and cached.get("task_set_hash") == dataset.task_set_hash:
            return BenchmarkRun.model_validate(cached)

        solver_live = requested is ProvenanceTier.LIVE and target is not None
        effective = resolve_tier(requested, dataset_ceiling=dataset.tier, solver_live=solver_live)
        mode = solver_mode if solver_mode is not None else spec.solver_mode

        started = time.perf_counter()
        base_report = await self._score(spec, dataset, effective, target, mode, governed=False)
        governed_summary: dict[str, float] | None = None
        # The with/without-governor uplift is measured on the reproducible tiers,
        # where the governed variant replays the recorded with-governor transcript
        # (``metadata['recorded_governed']``). A live governor toggle is a two-run
        # comparison the caller drives (two targets → RunStore.compare_runs).
        if spec.long_context and not solver_live:
            governed_report = await self._score(
                spec, dataset, effective, target, mode, governed=True
            )
            base_primary = summarize_results(base_report.results, primary_metric=spec.primary_metric)["primary"]
            gov_primary = summarize_results(governed_report.results, primary_metric=spec.primary_metric)["primary"]
            governed_summary = {
                "base": float(base_primary),
                "governed": float(gov_primary),
                "uplift": round(float(gov_primary) - float(base_primary), 4),
            }

        summary = summarize_results(base_report.results, primary_metric=spec.primary_metric)
        run = BenchmarkRun(
            benchmark_id=spec.id, niche=spec.niche, title=spec.title,
            tier=effective, requested_tier=requested,
            primary_metric=spec.primary_metric, primary=float(summary["primary"]),
            success_rate=float(summary["success_rate"]), mean_score=float(summary["mean_score"]),
            n=int(summary["n"]), task_set_hash=dataset.task_set_hash,
            replayed=not solver_live, source=dataset.source,
            items=[
                ItemResult(task_id=r.task_id, success=r.success, score=r.score,
                           tier=effective, details=r.details)
                for r in base_report.results
            ],
            governed=governed_summary,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        await self._checkpoint(run_id, run)
        return run

    async def _score(
        self,
        spec: BenchmarkSpec,
        dataset: BenchmarkDataset,
        effective: ProvenanceTier,
        target: Any | None,
        solver_mode: str,
        *,
        governed: bool,
    ) -> BenchmarkReport:
        """Score a benchmark at the effective tier: replay (S/R) or run live (L)."""
        adapter = spec.build_adapter(self._tasks_for(dataset, governed=governed))
        if effective is ProvenanceTier.LIVE:
            if target is None:
                raise EvalSuiteError(f"benchmark {spec.id!r}: tier 'live' needs a target model/app")
            solver = build_agent_solver(target, mode=solver_mode)
            return await adapter.run(solver)
        return await adapter.replay()

    @staticmethod
    def _tasks_for(dataset: BenchmarkDataset, *, governed: bool) -> list[BenchmarkTask]:
        """Tasks for a variant. The governed long-context variant swaps in each
        task's ``metadata['recorded_governed']`` (the with-governor recorded
        output) when present, so the uplift is a real, measured delta."""
        if not governed:
            return list(dataset.tasks)
        swapped: list[BenchmarkTask] = []
        for task in dataset.tasks:
            alt = task.metadata.get("recorded_governed")
            swapped.append(task.model_copy(update={"recorded": alt}) if alt is not None else task)
        return swapped

    # -- dataset selection ----------------------------------------------------

    def _dataset_for(
        self,
        spec: BenchmarkSpec,
        requested: ProvenanceTier,
        datasets: dict[str, BenchmarkDataset],
    ) -> BenchmarkDataset:
        provided = datasets.get(spec.id)
        if requested is ProvenanceTier.STATIC or provided is None:
            # Tier S always uses the bundled fabricated fixture. For a higher
            # requested tier with no dataset supplied we also fall back to the
            # fixture — whose Tier-S ceiling makes resolve_tier refuse the run, so
            # a fabricated fixture can never be relabeled Recorded or Live.
            return BenchmarkDataset.from_spec(spec)
        return provided

    # -- run identity & checkpointing -----------------------------------------

    def _model_label(self, target: Any | None, requested: ProvenanceTier) -> str:
        if requested.reproducible:
            return "recorded"
        return str(getattr(target, "model", None) or getattr(target, "name", None) or "live")

    def _run_id(
        self, specs: list[BenchmarkSpec], tier: ProvenanceTier, model: str, sample: int | None
    ) -> str:
        payload = {
            "benchmarks": sorted(s.id for s in specs),
            "tier": tier.value, "model": model, "sample": sample, "seed": self.seed,
        }
        return "run_" + compact_hash(payload)

    def _checkpoint_path(self, run_id: str) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / f"{run_id}.json"

    def _load_checkpoint(self, run_id: str) -> dict[str, dict[str, Any]]:
        path = self._checkpoint_path(run_id)
        if path is None or not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {r["benchmark_id"]: r for r in data.get("runs", [])}
        except (OSError, ValueError, KeyError):
            return {}

    async def _checkpoint(self, run_id: str, run: BenchmarkRun) -> None:
        path = self._checkpoint_path(run_id)
        if path is None:
            return
        async with self._checkpoint_lock:
            existing: dict[str, dict[str, Any]] = self._load_checkpoint(run_id)
            existing[run.benchmark_id] = run.model_dump(mode="json")
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"run_id": run_id, "runs": list(existing.values())}
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
