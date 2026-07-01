"""The benchmark registry — a niche-grouped, pluggable catalog.

Every benchmark the open evaluation plane can run is a :class:`BenchmarkSpec`
declaring its niche, the :class:`~vincio.evals.benchmarks.BenchmarkAdapter` that
scores it, its primary metric, its provenance, and a small bundled **Tier-S**
fabricated fixture that exercises the adapter + metric end to end. Specs live in a
:class:`BenchmarkRegistry`, grouped by niche; built-ins register at import (see
:mod:`vincio.evals.suite.builtin`), installed third-party benchmarks register
through the ``vincio.benchmarks`` entry-point group, and a user registers one with
:func:`register_benchmark` — *without touching the core*.

The registry is the spine of the **registry-completeness** gate: every catalog
entry must resolve to a live adapter, a dataset, a metric, and a report.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel, Field

from ...core.errors import EvalSuiteError
from ..benchmarks import BenchmarkAdapter, BenchmarkTask

__all__ = [
    "NICHES",
    "BenchmarkSpec",
    "BenchmarkRegistry",
    "register_benchmark",
    "default_benchmark_registry",
    "available_suite_benchmarks",
]


# The eleven niches — the catalog's top-level grouping. Order is display order.
NICHES: dict[str, str] = {
    "knowledge": "Knowledge",
    "reasoning": "Reasoning",
    "math": "Math",
    "coding": "Coding",
    "instruction": "Instruction",
    "truthfulness": "Truthfulness",
    "safety": "Safety",
    "rag": "RAG",
    "agent": "Agent",
    "long_context": "Long Context",
    "custom": "Custom",
}


class BenchmarkSpec(BaseModel):
    """One catalog entry: a benchmark, the adapter that scores it, and its
    provenance.

    ``id`` is the dotted ``niche.name`` a caller runs (``"knowledge.mmlu"``).
    ``adapter`` is the :class:`~vincio.evals.benchmarks.BenchmarkAdapter` subclass
    (or a zero-arg factory returning an instance). ``static_tasks`` is the inline,
    fabricated **Tier-S** fixture — each a :class:`~vincio.evals.benchmarks.
    BenchmarkTask` dict carrying a ``recorded`` output so the adapter + metric run
    end to end, byte-identically, offline. An inline fixture is *always* Tier-S by
    construction (see :meth:`~vincio.evals.suite.datasets.BenchmarkDataset.from_spec`),
    so a spec author cannot label a fabricated fixture Recorded or Live; the higher
    tiers come only from a real dataset supplied at run time. ``loader`` maps an
    official dataset export onto tasks for the Recorded / Live tiers.
    ``solver_mode`` is how a live solver is graded (``"text"`` for an answer string,
    ``"calls"`` for tool calls). ``long_context`` marks a benchmark the engine runs
    twice on the reproducible tiers — with and without the context governor — so the
    uplift is measured.
    """

    model_config = {"arbitrary_types_allowed": True, "frozen": True}

    id: str
    niche: str
    title: str
    summary: str = ""
    adapter: Any  # type[BenchmarkAdapter] | Callable[[], BenchmarkAdapter]
    primary_metric: str = "accuracy"
    static_tasks: list[dict[str, Any]] = Field(default_factory=list)
    loader: Callable[[list[dict[str, Any]]], list[BenchmarkTask]] | None = None
    solver_mode: str = "text"  # how a live solver is graded: "text" | "calls"
    long_context: bool = False
    provenance: str = "fabricated"  # human note on where the Tier-S fixture came from
    tags: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """The benchmark's short name (the part after the niche prefix)."""
        return self.id.split(".", 1)[-1]

    def build_adapter(self, tasks: list[BenchmarkTask] | None = None) -> BenchmarkAdapter:
        """Construct the adapter, optionally over an explicit task set.

        With ``tasks=None`` the adapter is built over the inline Tier-S fixture.
        """
        factory = self.adapter
        task_list = tasks if tasks is not None else [BenchmarkTask.model_validate(t) for t in self.static_tasks]
        if isinstance(factory, type) and issubclass(factory, BenchmarkAdapter):
            return factory(task_list)
        if callable(factory):
            adapter = factory()
            if not isinstance(adapter, BenchmarkAdapter):
                raise EvalSuiteError(f"benchmark {self.id!r}: factory did not return a BenchmarkAdapter")
            adapter._tasks = list(task_list)  # type: ignore[attr-defined]
            return adapter
        raise EvalSuiteError(f"benchmark {self.id!r}: adapter must be a BenchmarkAdapter subclass or factory")


class BenchmarkRegistry:
    """A niche-grouped catalog of :class:`BenchmarkSpec`s.

    Built-ins are registered at construction (when ``with_builtins`` is set);
    installed ``vincio.benchmarks`` plugins are loaded lazily on a name miss; a
    user adds one with :meth:`register`. Resolution is by the dotted ``id``.
    """

    def __init__(self, *, with_builtins: bool = True) -> None:
        self._specs: dict[str, BenchmarkSpec] = {}
        self._plugins_loaded = False
        if with_builtins:
            from . import builtin

            builtin.register_builtins(self)

    # -- registration ---------------------------------------------------------

    def register(self, spec: BenchmarkSpec, *, replace: bool = False) -> BenchmarkSpec:
        """Add a spec to the catalog. Rejects a duplicate id unless ``replace``."""
        if spec.niche not in NICHES:
            raise EvalSuiteError(
                f"benchmark {spec.id!r}: unknown niche {spec.niche!r}; known: {sorted(NICHES)}"
            )
        if "." not in spec.id:
            raise EvalSuiteError(f"benchmark id {spec.id!r} must be dotted 'niche.name'")
        if spec.id in self._specs and not replace:
            raise EvalSuiteError(f"benchmark {spec.id!r} already registered")
        self._specs[spec.id] = spec
        return spec

    def _ensure_plugins(self) -> None:
        if self._plugins_loaded:
            return
        self._plugins_loaded = True
        try:
            from ...plugins import load_plugins

            # Consuming the loader imports each installed ``vincio.benchmarks`` entry
            # point, which self-registers its spec. Discovery failures must never break
            # resolution (handled below).
            for _info in load_plugins(groups=["vincio.benchmarks"]):
                pass
        except Exception:  # noqa: BLE001 - plugin discovery must never break resolution
            from ...core.diagnostics import note_suppressed

            note_suppressed("benchmarks.load_plugins")

    # -- lookup ---------------------------------------------------------------

    def get(self, benchmark_id: str) -> BenchmarkSpec:
        """Resolve a spec by dotted id, loading plugins once on a miss."""
        if benchmark_id not in self._specs:
            self._ensure_plugins()
        spec = self._specs.get(benchmark_id)
        if spec is None:
            raise EvalSuiteError(
                f"unknown benchmark {benchmark_id!r}; known: {self.ids()[:20]}"
                f"{' …' if len(self._specs) > 20 else ''}"
            )
        return spec

    def __contains__(self, benchmark_id: object) -> bool:
        return benchmark_id in self._specs

    def ids(self) -> list[str]:
        """Every registered benchmark id, sorted."""
        return sorted(self._specs)

    def all(self) -> list[BenchmarkSpec]:
        """Every registered spec, sorted by id."""
        return [self._specs[i] for i in self.ids()]

    def niches(self) -> dict[str, list[BenchmarkSpec]]:
        """Specs grouped by niche, in catalog order."""
        grouped: dict[str, list[BenchmarkSpec]] = {key: [] for key in NICHES}
        for spec in self.all():
            grouped.setdefault(spec.niche, []).append(spec)
        return {key: specs for key, specs in grouped.items() if specs}

    def resolve(
        self, benchmark_id: str, *, tasks: list[BenchmarkTask] | None = None
    ) -> BenchmarkAdapter:
        """Construct the adapter for ``benchmark_id`` (over its Tier-S fixture by
        default, or an explicit ``tasks`` set)."""
        return self.get(benchmark_id).build_adapter(tasks)

    def select(self, patterns: Iterable[str]) -> list[BenchmarkSpec]:
        """Resolve a list of ids / niche names / ``"all"`` to concrete specs.

        ``"knowledge"`` expands to every benchmark in the Knowledge niche;
        ``"knowledge.mmlu"`` selects one; ``"all"`` selects the whole catalog.
        """
        out: list[BenchmarkSpec] = []
        seen: set[str] = set()
        for pattern in patterns:
            text = pattern.strip()
            if text in ("", "all"):
                matched = self.all()
            elif text in NICHES and text not in self._specs:
                matched = self.niches().get(text, [])
            else:
                matched = [self.get(text)]  # raises EvalSuiteError with a helpful list on a miss
            for spec in matched:
                if spec.id not in seen:
                    seen.add(spec.id)
                    out.append(spec)
        return out


_DEFAULT_REGISTRY: BenchmarkRegistry | None = None


def default_benchmark_registry() -> BenchmarkRegistry:
    """The process-wide registry, populated with the built-in catalog on first use."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = BenchmarkRegistry(with_builtins=True)
    return _DEFAULT_REGISTRY


def register_benchmark(spec: BenchmarkSpec, *, replace: bool = False) -> BenchmarkSpec:
    """Register a benchmark on the default registry — the public extension point.

    A package may instead advertise a ``vincio.benchmarks`` entry point resolving
    to a :class:`BenchmarkSpec` (or a factory returning one), so installing it is
    all it takes; this function is the in-process equivalent::

        from vincio.evals.suite import BenchmarkSpec, register_benchmark

        register_benchmark(BenchmarkSpec(
            id="custom.my_eval", niche="custom", title="My eval",
            adapter=MyAdapter, primary_metric="accuracy",
            static_tasks=[{"id": "t1", "prompt": "...", "gold": "A", "recorded": "A"}],
        ))
    """
    return default_benchmark_registry().register(spec, replace=replace)


def available_suite_benchmarks() -> list[str]:
    """The dotted ids of every benchmark in the default catalog."""
    return default_benchmark_registry().ids()
