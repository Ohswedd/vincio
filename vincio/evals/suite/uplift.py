"""Track 2 — the **Uplift arena**: the same model, *through Vincio* vs *direct*.

Track 1 asks how good a model is; this track asks what routing that model through
Vincio's infrastructure — grounding, rails, the context governor, structured-output
repair — *adds or removes*, benchmark by benchmark. Each benchmark is scored **twice
by the identical scorer**: once on the model's **direct** answer and once on its
**Vincio-routed** answer, and the per-benchmark delta is the measured uplift (or
regression), never assumed.

* **Mockup (offline, gates CI).** Each task carries two recorded outputs — the
  ``recorded`` direct answer and a ``recorded_vincio`` routed answer — and both are
  replayed through the real scorer, so the deterministic uplift is byte-identical.
* **Live.** The direct arm calls the model plainly; the Vincio arm calls it through
  a governed :class:`~vincio.core.app.ContextApp`. Same model, same scorer, two arms.

This generalizes the plane engine's with/without-governor long-context measurement
to every benchmark, and puts the honest, mechanism-level uplift that ``benchmarks/
quality_uplift.py`` reports onto the same tiered, reportable footing as the model
track.
"""

from __future__ import annotations

import hashlib
import json
import platform as _platform
import sys
from typing import Any

from pydantic import BaseModel, Field

from ...core.errors import EvalSuiteError, TierViolationError
from ...providers.base import run_sync
from ..benchmarks import BenchmarkAdapter, BenchmarkTask, make_agent_solver
from .metrics import summarize_results
from .tiers import ProvenanceTier
from .tracks import BenchmarkTrack

__all__ = [
    "UpliftBenchmark",
    "UpliftResult",
    "UpliftRun",
    "UpliftRegistry",
    "UpliftSuite",
    "default_uplift_registry",
    "register_uplift_benchmark",
    "available_uplift_benchmarks",
]


class UpliftBenchmark(BaseModel):
    """One two-armed benchmark: an adapter, and tasks carrying both recorded arms.

    Each task dict carries the usual ``id`` / ``prompt`` / ``inputs`` / ``gold`` plus
    **two** recorded outputs: ``recorded`` (the direct-model arm) and
    ``recorded_vincio`` (the Vincio-routed arm). The same adapter scores both.
    """

    model_config = {"arbitrary_types_allowed": True, "frozen": True}

    id: str
    title: str
    capability: str = ""
    adapter: Any
    primary_metric: str = "accuracy"
    higher_is_better: bool = True
    solver_mode: str = "text"  # how a live solver is graded: "text" | "calls"
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""

    def _tasks(self, *, arm: str) -> list[BenchmarkTask]:
        """Build the tasks for one arm, swapping in that arm's recorded output."""
        key = "recorded" if arm == "direct" else "recorded_vincio"
        out: list[BenchmarkTask] = []
        for t in self.tasks:
            data = {k: v for k, v in t.items() if k != "recorded_vincio"}
            data["recorded"] = t.get(key, t.get("recorded"))
            out.append(BenchmarkTask.model_validate(data))
        return out

    def build_adapter(self, tasks: list[BenchmarkTask]) -> BenchmarkAdapter:
        factory = self.adapter
        if isinstance(factory, type) and issubclass(factory, BenchmarkAdapter):
            return factory(tasks)
        adapter = factory()
        adapter._tasks = list(tasks)  # type: ignore[attr-defined]
        return adapter


class UpliftResult(BaseModel):
    """One benchmark's two-arm outcome: direct vs Vincio, and the delta."""

    benchmark_id: str
    title: str = ""
    capability: str = ""
    primary_metric: str = "accuracy"
    direct: float = 0.0
    vincio: float = 0.0
    delta: float = 0.0
    improved: bool = False
    regressed: bool = False
    tier: ProvenanceTier = ProvenanceTier.STATIC
    n: int = 0


class UpliftRun(BaseModel):
    """A whole uplift-track run: one model, direct vs Vincio, over several benchmarks."""

    run_id: str = ""
    model: str = "recorded"
    track: BenchmarkTrack = BenchmarkTrack.UPLIFT
    tier: ProvenanceTier = ProvenanceTier.STATIC
    environment: dict[str, Any] = Field(default_factory=dict)
    results: list[UpliftResult] = Field(default_factory=list)

    def overall_direct(self) -> float:
        return round(sum(r.direct for r in self.results) / len(self.results), 4) if self.results else 0.0

    def overall_vincio(self) -> float:
        return round(sum(r.vincio for r in self.results) / len(self.results), 4) if self.results else 0.0

    def overall_delta(self) -> float:
        return round(self.overall_vincio() - self.overall_direct(), 4)

    @property
    def determinism_digest(self) -> str:
        """A content hash over the sorted per-benchmark direct/vincio scores — the same
        determinism pin the model track's :class:`SuiteRun` exposes under this name."""
        return _results_digest(self.results)

    @property
    def gated(self) -> bool:
        return self.tier.gates_ci


class UpliftRegistry:
    """A catalog of :class:`UpliftBenchmark`s."""

    def __init__(self, *, with_builtins: bool = True) -> None:
        self._benchmarks: dict[str, UpliftBenchmark] = {}
        if with_builtins:
            from . import uplift_builtin

            uplift_builtin.register_builtins(self)

    def register(self, benchmark: UpliftBenchmark, *, replace: bool = False) -> UpliftBenchmark:
        if benchmark.id in self._benchmarks and not replace:
            raise EvalSuiteError(f"uplift benchmark {benchmark.id!r} already registered")
        self._benchmarks[benchmark.id] = benchmark
        return benchmark

    def get(self, benchmark_id: str) -> UpliftBenchmark:
        b = self._benchmarks.get(benchmark_id)
        if b is None:
            ids = self.ids()
            raise EvalSuiteError(
                f"unknown uplift benchmark {benchmark_id!r}; known: {ids[:20]}"
                f"{' …' if len(ids) > 20 else ''}"
            )
        return b

    def ids(self) -> list[str]:
        return sorted(self._benchmarks)

    def all(self) -> list[UpliftBenchmark]:
        return [self._benchmarks[i] for i in self.ids()]

    def by_capability(self) -> dict[str, list[UpliftBenchmark]]:
        grouped: dict[str, list[UpliftBenchmark]] = {}
        for b in self.all():
            grouped.setdefault(b.capability, []).append(b)
        return grouped

    def select(self, patterns: list[str]) -> list[UpliftBenchmark]:
        """Resolve a list of ids / capability names / ``"all"`` to concrete benchmarks."""
        caps = self.by_capability()
        out: list[UpliftBenchmark] = []
        seen: set[str] = set()
        for pattern in patterns:
            text = pattern.strip()
            if text in ("", "all"):
                matched = self.all()
            elif text in caps and text not in self._benchmarks:
                matched = caps[text]
            else:
                matched = [self.get(text)]
            for b in matched:
                if b.id not in seen:
                    seen.add(b.id)
                    out.append(b)
        return out


_DEFAULT_REGISTRY: UpliftRegistry | None = None


def default_uplift_registry() -> UpliftRegistry:
    """The process-wide uplift-benchmark catalog, populated on first use."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = UpliftRegistry(with_builtins=True)
    return _DEFAULT_REGISTRY


def register_uplift_benchmark(benchmark: UpliftBenchmark, *, replace: bool = False) -> UpliftBenchmark:
    """Register a custom uplift benchmark on the default registry — the extension point."""
    return default_uplift_registry().register(benchmark, replace=replace)


def available_uplift_benchmarks() -> list[str]:
    """The ids of every registered uplift benchmark."""
    return default_uplift_registry().ids()


class UpliftSuite:
    """Run uplift benchmarks (direct vs Vincio) and fold them into an :class:`UpliftRun`.

    Mockup replays each task's two recorded arms through the identical scorer; live
    runs solve each arm against ``direct`` (a plain model) and ``vincio`` (a governed
    app) targets the caller supplies.
    """

    def __init__(self, registry: UpliftRegistry | None = None) -> None:
        self.registry = registry or default_uplift_registry()

    def run(
        self,
        benchmarks: str | list[str] = "all",
        *,
        tier: ProvenanceTier | str = ProvenanceTier.STATIC,
        direct: Any | None = None,
        vincio: Any | None = None,
        model: str | None = None,
    ) -> UpliftRun:
        names = [benchmarks] if isinstance(benchmarks, str) else list(benchmarks)
        specs = self.registry.select(names)
        if not specs:
            raise EvalSuiteError(f"no uplift benchmark matched {names!r}")
        requested = ProvenanceTier.parse(tier)
        # Honesty gate — the uplift track's built-in arms are a *fabricated* mockup, so
        # it supports Static (mockup, gates CI) or Live (a real model), never Recorded:
        # a fabricated run may not print a Recorded label. This mirrors the model
        # track's resolve_tier and is the invariant the platform promises to enforce.
        if requested is ProvenanceTier.RECORDED:
            raise TierViolationError(
                "the uplift track compares a model direct vs through Vincio; its benchmarks are a "
                "fabricated two-arm mockup, so it supports tier 'static' (mockup, gates CI) or 'live' "
                "(a real model), not 'recorded'. A fabricated run may not print a Recorded label."
            )
        live = requested is ProvenanceTier.LIVE
        if live and (direct is None or vincio is None):
            raise EvalSuiteError("tier 'live' needs both a `direct` and a `vincio` target")
        # The effective tier is resolved once and applied everywhere, so the run header
        # tier can never contradict the per-result tiers.
        effective = ProvenanceTier.LIVE if live else ProvenanceTier.STATIC
        results = [self._run_one(spec, effective, direct=direct, vincio=vincio) for spec in specs]
        return UpliftRun(
            run_id="uplift_" + _results_digest(results),
            model=model or ("live" if live else "recorded-arms"),
            tier=effective, environment=_environment(), results=results,
        )

    def _run_one(
        self, spec: UpliftBenchmark, effective: ProvenanceTier, *, direct: Any | None, vincio: Any | None
    ) -> UpliftResult:
        if effective is ProvenanceTier.LIVE:
            direct_primary = self._score_live(spec, direct, arm="direct")
            vincio_primary = self._score_live(spec, vincio, arm="vincio")
            n = len(spec.tasks)
        else:
            direct_primary, n = self._score_replay(spec, arm="direct")
            vincio_primary, _ = self._score_replay(spec, arm="vincio")
        delta = round(vincio_primary - direct_primary, 4)
        # "Improved" respects the metric's direction (a lower-is-better metric improves
        # when the delta is negative), not just the raw sign.
        better = (delta > 0) if spec.higher_is_better else (delta < 0)
        significant = abs(delta) > 1e-9
        return UpliftResult(
            benchmark_id=spec.id, title=spec.title, capability=spec.capability,
            primary_metric=spec.primary_metric, direct=round(direct_primary, 4),
            vincio=round(vincio_primary, 4), delta=delta,
            improved=better and significant, regressed=(not better) and significant,
            tier=effective, n=n,
        )

    def _score_replay(self, spec: UpliftBenchmark, *, arm: str) -> tuple[float, int]:
        tasks = spec._tasks(arm=arm)
        adapter = spec.build_adapter(tasks)
        report = run_sync(adapter.replay())
        summary = summarize_results(report.results, primary_metric=spec.primary_metric)
        return float(summary["primary"]), int(summary["n"])

    def _score_live(self, spec: UpliftBenchmark, target: Any, *, arm: str) -> float:
        tasks = spec._tasks(arm=arm)
        adapter = spec.build_adapter(tasks)
        report = run_sync(adapter.run(make_agent_solver(target, mode=spec.solver_mode)))
        return float(summarize_results(report.results, primary_metric=spec.primary_metric)["primary"])


def _results_digest(results: list[UpliftResult]) -> str:
    """The single canonical content hash over the per-benchmark scores, used for both
    the run id and :attr:`UpliftRun.determinism_digest` (computed one way, not two)."""
    canonical = sorted([r.benchmark_id, round(r.direct, 6), round(r.vincio, 6)] for r in results)
    blob = json.dumps({"track": BenchmarkTrack.UPLIFT.value, "results": canonical},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _environment() -> dict[str, Any]:
    from ... import __version__

    return {
        "schema_version": "1.0",
        "vincio_version": __version__,
        "python_version": _platform.python_version(),
        "platform": sys.platform,
        "track": BenchmarkTrack.UPLIFT.value,
    }
