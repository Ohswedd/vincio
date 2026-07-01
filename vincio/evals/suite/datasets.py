"""The dataset layer — content-addressed, hash-pinned benchmark task sets.

A :class:`BenchmarkDataset` wraps a benchmark's tasks with its **provenance tier
ceiling** and a content hash over ``(id, gold)`` (the same pin
:meth:`~vincio.evals.benchmarks.BenchmarkAdapter.task_set_hash` uses), so a silent
task-set change is caught against a recorded pin. Tier-S fixtures are inline and
fabricated; Recorded / Live datasets come from a JSONL export, a user dataset, or
an optional Hugging Face fetch (behind ``vincio[eval-datasets]``).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from ...core.errors import EvalSuiteError
from ..benchmarks import BenchmarkTask, compute_task_set_hash, tasks_from_jsonl
from .registry import BenchmarkSpec
from .tiers import ProvenanceTier

__all__ = ["BenchmarkDataset"]


class BenchmarkDataset(BaseModel):
    """A pinned set of :class:`~vincio.evals.benchmarks.BenchmarkTask`s and its
    provenance tier ceiling.

    ``tier`` is the highest tier this dataset's provenance can honestly support —
    :attr:`~ProvenanceTier.STATIC` for a fabricated fixture, higher for a real
    slice. ``task_set_hash`` is computed on construction; if ``pinned_hash`` is set
    and differs, construction raises (drift caught). ``source`` records the origin
    for the report.
    """

    name: str = "dataset"
    benchmark_id: str = ""
    tier: ProvenanceTier = ProvenanceTier.STATIC
    tasks: list[BenchmarkTask] = Field(default_factory=list)
    task_set_hash: str = ""
    pinned_hash: str = ""
    source: str = "inline"

    @model_validator(mode="after")
    def _pin(self) -> BenchmarkDataset:
        computed = compute_task_set_hash(self.tasks)
        object.__setattr__(self, "task_set_hash", computed)
        if self.pinned_hash and self.pinned_hash != computed:
            raise EvalSuiteError(
                f"dataset {self.name!r}: task-set hash drift — pinned {self.pinned_hash!r}, "
                f"computed {computed!r}. The task set changed; re-pin deliberately."
            )
        return self

    def __len__(self) -> int:
        return len(self.tasks)

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_spec(cls, spec: BenchmarkSpec) -> BenchmarkDataset:
        """The inline, fabricated **Tier-S** fixture bundled with a built-in spec.

        The tier is hardcoded :attr:`~ProvenanceTier.STATIC`: an inline fixture is
        fabricated by construction, so a spec author can never have it reported as
        Recorded or Live. The higher tiers come only from a real dataset passed to
        the engine at run time.
        """
        tasks = [BenchmarkTask.model_validate(t) for t in spec.static_tasks]
        return cls(
            name=f"{spec.id}@static", benchmark_id=spec.id, tier=ProvenanceTier.STATIC,
            tasks=tasks, source="fabricated",
        )

    @classmethod
    def from_tasks(
        cls,
        tasks: list[BenchmarkTask],
        *,
        name: str = "dataset",
        tier: ProvenanceTier = ProvenanceTier.RECORDED,
        benchmark_id: str = "",
        pinned_hash: str = "",
        source: str = "tasks",
    ) -> BenchmarkDataset:
        """A dataset from an explicit task list (default tier Recorded)."""
        return cls(name=name, benchmark_id=benchmark_id, tier=tier, tasks=list(tasks),
                   pinned_hash=pinned_hash, source=source)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        tier: ProvenanceTier = ProvenanceTier.RECORDED,
        benchmark_id: str = "",
        pinned_hash: str = "",
    ) -> BenchmarkDataset:
        """A dataset from a JSONL export (one task object per line)."""
        tasks = tasks_from_jsonl(path)
        return cls(name=Path(path).stem, benchmark_id=benchmark_id, tier=tier, tasks=tasks,
                   pinned_hash=pinned_hash, source=str(path))

    @classmethod
    def from_export(
        cls,
        records: list[dict[str, Any]],
        *,
        loader: Callable[[list[dict[str, Any]]], list[BenchmarkTask]],
        tier: ProvenanceTier = ProvenanceTier.RECORDED,
        name: str = "export",
        benchmark_id: str = "",
        pinned_hash: str = "",
    ) -> BenchmarkDataset:
        """A dataset from an official benchmark export, mapped by a spec ``loader``."""
        return cls(name=name, benchmark_id=benchmark_id, tier=tier, tasks=loader(records),
                   pinned_hash=pinned_hash, source="export")

    @classmethod
    def from_huggingface(
        cls,
        path: str,
        *,
        loader: Callable[[list[dict[str, Any]]], list[BenchmarkTask]],
        split: str = "test",
        name: str | None = None,
        config: str | None = None,
        limit: int | None = None,
        tier: ProvenanceTier = ProvenanceTier.LIVE,
        benchmark_id: str = "",
        pinned_hash: str = "",
    ) -> BenchmarkDataset:
        """Fetch a public dataset from the Hugging Face Hub and map it to tasks.

        Requires ``vincio[eval-datasets]`` (the ``datasets`` package). The fetch is
        the only networked path in the plane; everything else runs offline. The
        result defaults to tier :attr:`~ProvenanceTier.LIVE` (the full real dataset)
        — pass ``tier=ProvenanceTier.RECORDED`` with a ``pinned_hash`` to treat a
        hash-pinned slice as reproducible.
        """
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise EvalSuiteError(
                "Hugging Face dataset fetch requires the 'datasets' package: "
                'pip install "vincio[eval-datasets]"'
            ) from exc
        ds = load_dataset(path, config, split=split)  # pragma: no cover - network
        records = [dict(row) for row in ds]  # pragma: no cover - network
        if limit is not None:  # pragma: no cover - network
            records = records[:limit]
        return cls(  # pragma: no cover - network
            name=name or path, benchmark_id=benchmark_id, tier=tier,
            tasks=loader(records), pinned_hash=pinned_hash, source=f"hf:{path}",
        )

    # -- views ----------------------------------------------------------------

    def sample(self, n: int, *, seed: int = 42) -> BenchmarkDataset:
        """A deterministic sub-sample of ``n`` tasks (the seed makes it reproducible)."""
        if n >= len(self.tasks):
            return self
        rng = random.Random(seed)
        chosen = rng.sample(self.tasks, n)
        chosen.sort(key=lambda t: t.id)
        return BenchmarkDataset(
            name=f"{self.name}@sample{n}", benchmark_id=self.benchmark_id,
            tier=self.tier, tasks=chosen, source=self.source,
        )

    def pin(self) -> str:
        """Return the current task-set hash to record as a pin elsewhere."""
        return self.task_set_hash
