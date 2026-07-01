"""Track 3 — the **Feature arena**: a Vincio feature vs the same feature in a
competitor library (and, where it clarifies, a naive baseline).

Where the model track (:mod:`~vincio.evals.suite.engine`) asks *how good is a
model* and the uplift track asks *what does Vincio add to a model*, this track
asks *how good is a Vincio feature — memory, retrieval, output repair, context
assembly, tabular encoding — against the real alternative a team would otherwise
reach for*. Every number is **measured on this machine from a live run of both
sides**: a :class:`Contender` runs, is timed, and its quality is scored by the
contest's own deterministic metric. A competitor library that is not installed is
reported as **skipped**, never assumed and never fabricated.

The provenance tier is honest about the comparison that actually happened: a
contest is **Live** only when every declared competitor executed; if a competitor
is missing it drops to **Static** (the Vincio and baseline sides still ran, but the
head-to-head did not). The deterministic quality metric — not wall-clock — is what
gates CI, so a re-run is byte-identical on the part that matters.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ...core.errors import EvalSuiteError
from .tiers import ProvenanceTier
from .tracks import BenchmarkTrack

__all__ = [
    "Contender",
    "FeatureMeasurement",
    "FeatureRun",
    "FeatureSuiteRun",
    "FeatureContest",
    "FeatureRegistry",
    "FeatureSuite",
    "default_feature_registry",
    "register_feature_contest",
    "available_feature_contests",
    "median_ms",
    "have_module",
]

_KINDS = ("vincio", "competitor", "baseline")


# --------------------------------------------------------------------------- #
# Measurement helpers (shared by every built-in contest).
# --------------------------------------------------------------------------- #


def have_module(module: str) -> bool:
    """Whether an optional competitor module is importable (no hard dependency)."""
    try:
        __import__(module)
        return True
    except Exception:  # noqa: BLE001 - a missing/broken competitor is a skip, not a crash
        from ...core.diagnostics import note_suppressed

        note_suppressed("feature_bench.competitor_import")
        return False


def median_ms(fn: Callable[[], Any], *, iterations: int = 10, warmup: int = 2) -> float:
    """Median wall-clock milliseconds per call over ``iterations`` runs after warmup."""
    for _ in range(max(0, warmup)):
        fn()
    samples: list[float] = []
    for _ in range(max(1, iterations)):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return round(statistics.median(samples), 4)


# --------------------------------------------------------------------------- #
# Contenders — one runnable implementation of a capability.
# --------------------------------------------------------------------------- #


class Contender:
    """One implementation of a capability, timed and scored by a contest.

    ``kind`` is ``"vincio"`` (the feature under test), ``"competitor"`` (a real
    third-party library), or ``"baseline"`` (a naive in-house reference).
    ``requires`` lists the modules a competitor needs; when any is missing the
    contender is reported **skipped** rather than run. ``run`` is called by the
    contest to produce a :class:`FeatureMeasurement`.
    """

    def __init__(
        self,
        name: str,
        run: Callable[[], FeatureMeasurement],
        *,
        kind: str = "competitor",
        requires: tuple[str, ...] = (),
    ) -> None:
        if kind not in _KINDS:
            raise EvalSuiteError(f"contender {name!r}: kind must be one of {_KINDS}")
        self.name = name
        self.kind = kind
        self.requires = requires
        self._run = run

    @property
    def available(self) -> bool:
        return all(have_module(m) for m in self.requires)

    def measure(self) -> FeatureMeasurement:
        if not self.available:
            missing = ", ".join(m for m in self.requires if not have_module(m))
            return FeatureMeasurement(
                contender=self.name, kind=self.kind, available=False,
                note=f"skipped (pip install {missing})",
            )
        try:
            m = self._run()
        except Exception as exc:  # noqa: BLE001 - a contender that errors at runtime is a skip, not a crash
            from ...core.diagnostics import note_suppressed

            note_suppressed("feature_bench.contender_run")
            return FeatureMeasurement(
                contender=self.name, kind=self.kind, available=False,
                note=f"errored: {type(exc).__name__}: {exc}"[:160],
            )
        # Stamp identity from the contender so a runner need not repeat it.
        return m.model_copy(update={"contender": self.name, "kind": self.kind, "available": True})


# --------------------------------------------------------------------------- #
# Result models.
# --------------------------------------------------------------------------- #


class FeatureMeasurement(BaseModel):
    """One contender's measured result: a deterministic quality ``primary`` plus an
    informational ``latency_ms`` and any extra ``metrics``."""

    contender: str = ""
    kind: str = "competitor"
    available: bool = True
    primary: float = 0.0
    latency_ms: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    note: str = ""


class FeatureRun(BaseModel):
    """One contest's result: every contender's measurement, the winner, a verdict."""

    contest_id: str
    title: str = ""
    capability: str = ""
    primary_metric: str = "score"
    higher_is_better: bool = True
    unit: str = ""
    tier: ProvenanceTier = ProvenanceTier.STATIC
    track: BenchmarkTrack = BenchmarkTrack.FEATURE
    measurements: list[FeatureMeasurement] = Field(default_factory=list)
    winner: str = ""
    verdict: str = ""
    duration_ms: int = 0

    @property
    def vincio(self) -> FeatureMeasurement | None:
        return next((m for m in self.measurements if m.kind == "vincio"), None)

    @property
    def competitors(self) -> list[FeatureMeasurement]:
        return [m for m in self.measurements if m.kind == "competitor"]

    @property
    def ran_live(self) -> bool:
        """Whether a real competitor was actually measured (the head-to-head happened)."""
        return any(m.kind == "competitor" and m.available for m in self.measurements)

    @property
    def skipped(self) -> list[str]:
        return [m.contender for m in self.measurements if not m.available]

    @property
    def determinism_digest(self) -> str:
        """Content hash over the deterministic quality metrics — latency excluded, so
        two runs on different machines agree on the part CI gates. Named to match the
        model track's :attr:`SuiteRun.determinism_digest`."""
        import hashlib
        import json

        canonical = [
            [m.contender, round(m.primary, 6), sorted((k, round(v, 6)) for k, v in m.metrics.items())]
            for m in sorted(self.measurements, key=lambda x: x.contender)
            if m.available
        ]
        blob = json.dumps({"contest": self.contest_id, "metric": self.primary_metric,
                           "measurements": canonical}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class FeatureSuiteRun(BaseModel):
    """A whole feature-track run over a set of contests."""

    run_id: str = ""
    track: BenchmarkTrack = BenchmarkTrack.FEATURE
    tier: ProvenanceTier = ProvenanceTier.STATIC
    environment: dict[str, Any] = Field(default_factory=dict)
    runs: list[FeatureRun] = Field(default_factory=list)

    @property
    def determinism_digest(self) -> str:
        """A content hash over every contest's digest — the suite-level determinism pin,
        over *all* contests (not just the first)."""
        import hashlib
        import json

        parts = sorted(f"{r.contest_id}:{r.determinism_digest}" for r in self.runs)
        blob = json.dumps({"track": self.track.value, "runs": parts}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @property
    def gated(self) -> bool:
        return self.tier.gates_ci


# --------------------------------------------------------------------------- #
# Contest spec + registry.
# --------------------------------------------------------------------------- #


class FeatureContest(BaseModel):
    """One capability workload: its contenders and how to score them.

    ``runner`` returns the ordered contenders for the contest; the suite calls each
    contender's :meth:`Contender.measure`, picks the winner by ``primary_metric``
    (respecting ``higher_is_better``), and resolves the tier from whether every
    declared competitor ran.
    """

    model_config = {"arbitrary_types_allowed": True, "frozen": True}

    id: str
    title: str
    capability: str
    primary_metric: str = "score"
    higher_is_better: bool = True
    unit: str = ""
    summary: str = ""
    runner: Callable[[], list[Contender]]
    verdict: Callable[[list[FeatureMeasurement]], str] | None = None
    # Optional post-processing once every contender is measured — e.g. to compute a
    # metric that is relative to a reference contender (token-counting accuracy vs
    # the exact counter). Receives and returns the full measurement list.
    finalize: Callable[[list[FeatureMeasurement]], list[FeatureMeasurement]] | None = None


class FeatureRegistry:
    """A catalog of :class:`FeatureContest`s, grouped by capability."""

    def __init__(self, *, with_builtins: bool = True) -> None:
        self._contests: dict[str, FeatureContest] = {}
        if with_builtins:
            from . import feature_builtin

            feature_builtin.register_builtins(self)

    def register(self, contest: FeatureContest, *, replace: bool = False) -> FeatureContest:
        if contest.id in self._contests and not replace:
            raise EvalSuiteError(f"feature contest {contest.id!r} already registered")
        self._contests[contest.id] = contest
        return contest

    def get(self, contest_id: str) -> FeatureContest:
        contest = self._contests.get(contest_id)
        if contest is None:
            raise EvalSuiteError(
                f"unknown feature contest {contest_id!r}; known: {self.ids()[:20]}"
            )
        return contest

    def __contains__(self, contest_id: object) -> bool:
        return contest_id in self._contests

    def ids(self) -> list[str]:
        return sorted(self._contests)

    def all(self) -> list[FeatureContest]:
        return [self._contests[i] for i in self.ids()]

    def by_capability(self) -> dict[str, list[FeatureContest]]:
        grouped: dict[str, list[FeatureContest]] = {}
        for contest in self.all():
            grouped.setdefault(contest.capability, []).append(contest)
        return grouped

    def select(self, patterns: list[str]) -> list[FeatureContest]:
        """Resolve ids / capability names / ``"all"`` to concrete contests."""
        caps = self.by_capability()
        out: list[FeatureContest] = []
        seen: set[str] = set()
        for pattern in patterns:
            text = pattern.strip()
            if text in ("", "all"):
                matched = self.all()
            elif text in caps and text not in self._contests:
                matched = caps[text]
            else:
                matched = [self.get(text)]
            for contest in matched:
                if contest.id not in seen:
                    seen.add(contest.id)
                    out.append(contest)
        return out


_DEFAULT_REGISTRY: FeatureRegistry | None = None


def default_feature_registry() -> FeatureRegistry:
    """The process-wide feature-contest catalog, populated on first use."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = FeatureRegistry(with_builtins=True)
    return _DEFAULT_REGISTRY


def register_feature_contest(contest: FeatureContest, *, replace: bool = False) -> FeatureContest:
    """Register a custom feature contest on the default registry — the extension point."""
    return default_feature_registry().register(contest, replace=replace)


def available_feature_contests() -> list[str]:
    """The ids of every registered feature contest."""
    return default_feature_registry().ids()


# --------------------------------------------------------------------------- #
# The suite runner.
# --------------------------------------------------------------------------- #


class FeatureSuite:
    """Run feature contests and fold them into a :class:`FeatureSuiteRun`.

    ``registry`` defaults to the process-wide catalog. Each contest runs every
    contender (a missing competitor is skipped, not fabricated), the winner is the
    best available primary, and the run's tier is **Live** iff every declared
    competitor executed.
    """

    def __init__(self, registry: FeatureRegistry | None = None) -> None:
        self.registry = registry or default_feature_registry()

    def run(self, contests: str | list[str] = "all") -> FeatureSuiteRun:
        names = [contests] if isinstance(contests, str) else list(contests)
        specs = self.registry.select(names)
        if not specs:
            raise EvalSuiteError(f"no feature contest matched {names!r}")
        runs = [self._run_one(spec) for spec in specs]
        suite_tier = ProvenanceTier.LIVE if runs and all(r.ran_live for r in runs) else ProvenanceTier.STATIC
        suite = FeatureSuiteRun(tier=suite_tier, environment=_environment(), runs=runs)
        return FeatureSuiteRun(
            run_id="feat_" + (suite.determinism_digest if runs else "empty"),
            tier=suite_tier, environment=_environment(), runs=runs,
        )

    def _run_one(self, spec: FeatureContest) -> FeatureRun:
        started = time.perf_counter()
        contenders = spec.runner()
        measurements = [c.measure() for c in contenders]
        if spec.finalize is not None:
            measurements = spec.finalize(measurements)
        # The tier reflects what actually *ran*, not what was merely importable: a
        # competitor that is missing or errors at runtime leaves the head-to-head
        # incomplete, so the contest drops to Static.
        competitor_ms = [m for m in measurements if m.kind == "competitor"]
        ran_all_competitors = bool(competitor_ms) and all(m.available for m in competitor_ms)
        tier = ProvenanceTier.LIVE if ran_all_competitors else ProvenanceTier.STATIC
        winner = _pick_winner(measurements, higher_is_better=spec.higher_is_better)
        verdict = spec.verdict(measurements) if spec.verdict else _default_verdict(spec, measurements, winner)
        return FeatureRun(
            contest_id=spec.id, title=spec.title, capability=spec.capability,
            primary_metric=spec.primary_metric, higher_is_better=spec.higher_is_better,
            unit=spec.unit, tier=tier, measurements=measurements, winner=winner, verdict=verdict,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


def _pick_winner(measurements: list[FeatureMeasurement], *, higher_is_better: bool) -> str:
    ran = [m for m in measurements if m.available]
    if not ran:
        return ""
    best = (max if higher_is_better else min)(ran, key=lambda m: m.primary)
    return best.contender


def _default_verdict(spec: FeatureContest, measurements: list[FeatureMeasurement], winner: str) -> str:
    ran = [m for m in measurements if m.available]
    skipped = [m.contender for m in measurements if not m.available]
    unit = f" {spec.unit}" if spec.unit else ""
    parts = [f"{m.contender} {m.primary:g}{unit}" for m in ran]
    line = f"{spec.primary_metric}: " + ", ".join(parts) + f" — winner: {winner or 'n/a'}"
    if skipped:
        line += f" (skipped: {', '.join(skipped)})"
    return line


def _environment() -> dict[str, Any]:
    import platform as _platform
    import sys

    from ... import __version__

    return {
        "schema_version": "1.0",
        "vincio_version": __version__,
        "python_version": _platform.python_version(),
        "platform": sys.platform,
        "track": BenchmarkTrack.FEATURE.value,
    }
