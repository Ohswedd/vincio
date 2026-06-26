"""Fitting a dataset far larger than the window into bounded, faithful evidence.

A table of ten million rows cannot enter a prompt, and truncating it to the
first thousand throws away everything the rest of the table would have said.
:func:`fit_to_window` instead represents the whole faithfully under a *fixed
token budget*: a deterministic :class:`~vincio.data.DatasetProfile` (every
column's type, null rate, cardinality, range, distribution — computed over all
rows in bounded memory) plus a *representative sample* sized to whatever budget
the profile leaves. The profile's size depends on the number of columns, not the
number of rows, and the sample is capped by the budget — so the representation
stays inside the same window whether the table has ten thousand rows or ten
million.

:func:`fit_stream` does the same over a row iterator in a single bounded pass,
profiling and reservoir-sampling together without ever materializing the table —
the path for a source larger than memory.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from .core import ColumnSchema, Dataset
from .profile import (
    DatasetProfile,
    _build_profile,
    _make_accumulators,
    profile_dataset,
)
from .sampling import SampleMethod, sample_dataset

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.types import EvidenceItem

__all__ = ["WindowFit", "fit_to_window", "fit_stream"]

DEFAULT_SAMPLE_CAP = 4_096


class WindowFit(BaseModel):
    """A dataset fitted into a fixed token budget: a full-fidelity column
    :attr:`profile` plus a representative :attr:`sample` whose combined encoding
    stays within :attr:`budget_tokens`.

    Project both halves to evidence with :meth:`to_evidence_items` (the profile
    and the sample become two cited table evidence items), or read the totals to
    confirm the fit (:attr:`within_budget`).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile: DatasetProfile
    sample: Dataset
    original_row_count: int = 0
    sample_size: int = 0
    budget_tokens: int = 0
    profile_tokens: int = 0
    sample_tokens: int = 0
    token_cost: int = 0
    within_budget: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_evidence_items(self, *, source_id: str = "", model: str | None = None) -> list[EvidenceItem]:
        """The profile and the representative sample as two table evidence items
        the context compiler scores, budgets, orders, and cites."""
        base = source_id or self.sample.name or self.profile.name or "dataset"
        coverage = (
            f"{self.sample_size:,} of {self.original_row_count:,} rows"
            if self.original_row_count
            else f"{self.sample_size:,} rows"
        )
        items = [self.profile.to_evidence_item(source_id=f"{base}_profile")]
        if self.sample.row_count:
            items.append(
                self.sample.to_evidence_item(
                    source_id=f"{base}_sample",
                    caption=f"Representative sample ({coverage})",
                )
            )
        return items

    def summary(self) -> str:
        """A one-line human summary of the fit."""
        status = "within" if self.within_budget else "OVER"
        return (
            f"{self.profile.name or 'dataset'}: profile + {self.sample_size:,}-row sample "
            f"= {self.token_cost:,} tokens ({status} {self.budget_tokens:,} budget) "
            f"for {self.original_row_count:,} rows"
        )


def _select(dataset: Dataset, indices: Sequence[int]) -> Dataset:
    cells = [[column[i] for i in indices] for column in dataset.cells]
    return Dataset(name=dataset.name, columns=list(dataset.columns), cells=cells, source=dataset.source)


def _fit_order(candidate: Dataset, method: SampleMethod, by: Any, seed: int) -> list[int]:
    """A representative ordering of the candidate's rows such that *any prefix* is
    representative — a seeded shuffle for uniform methods, a round-robin interleave
    across strata for a stratified fit (so a prefix keeps the key's proportions)."""
    n = candidate.row_count
    if method is SampleMethod.STRATIFIED and by is not None:
        keys = [by] if isinstance(by, str) else list(by)
        col_indices = [candidate.column_names.index(k) for k in keys if k in candidate.column_names]
        groups: dict[tuple[Any, ...], list[int]] = {}
        for i in range(n):
            stratum = tuple(candidate.cells[j][i] for j in col_indices)
            groups.setdefault(stratum, []).append(i)
        ordered_keys = sorted(groups, key=lambda s: tuple(str(v) for v in s))
        order: list[int] = []
        cursors = {k: 0 for k in ordered_keys}
        remaining = n
        while remaining:
            for k in ordered_keys:
                if cursors[k] < len(groups[k]):
                    order.append(groups[k][cursors[k]])
                    cursors[k] += 1
                    remaining -= 1
        return order
    order = list(range(n))
    random.Random(seed + 1).shuffle(order)
    return order


def _fit_sample(
    candidate: Dataset,
    remaining: int,
    *,
    method: SampleMethod,
    by: Any,
    seed: int,
    model: str | None,
    original_row_count: int,
) -> Dataset:
    """Trim the candidate sample to the largest representative subset whose
    encoding fits ``remaining`` tokens (binary search over a representative
    ordering — token cost is monotonic in the prefix length)."""
    if candidate.row_count == 0 or remaining <= 0:
        return _finalize_sample(candidate, [], method, by, seed, remaining, original_row_count)
    if candidate.token_cost(model=model) <= remaining:
        return _finalize_sample(candidate, list(range(candidate.row_count)), method, by, seed, remaining, original_row_count)
    order = _fit_order(candidate, method, by, seed)
    lo, hi, best = 0, candidate.row_count, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        cost = _select(candidate, order[:mid]).token_cost(model=model) if mid else 0
        if cost <= remaining:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    chosen = sorted(order[:best])
    return _finalize_sample(candidate, chosen, method, by, seed, remaining, original_row_count)


def _finalize_sample(
    candidate: Dataset,
    indices: Sequence[int],
    method: SampleMethod,
    by: Any,
    seed: int,
    remaining: int,
    original_row_count: int,
) -> Dataset:
    sample = _select(candidate, indices)
    meta: dict[str, Any] = {
        "method": method.value,
        "size": sample.row_count,
        "of": original_row_count,
        "seed": seed,
        "fit_to_tokens": remaining,
    }
    if by is not None:
        meta["by"] = by
    sample.metadata = {**candidate.metadata, "sample": meta}
    return sample


def _assemble(
    profile: DatasetProfile,
    candidate: Dataset,
    *,
    original_row_count: int,
    max_tokens: int,
    method: SampleMethod,
    by: Any,
    seed: int,
    model: str | None,
) -> WindowFit:
    profile_tokens = profile.token_cost(model=model)
    remaining = max(0, max_tokens - profile_tokens)
    sample = _fit_sample(
        candidate,
        remaining,
        method=method,
        by=by,
        seed=seed,
        model=model,
        original_row_count=original_row_count,
    )
    sample_tokens = sample.token_cost(model=model) if sample.row_count else 0
    token_cost = profile_tokens + sample_tokens
    return WindowFit(
        profile=profile,
        sample=sample,
        original_row_count=original_row_count,
        sample_size=sample.row_count,
        budget_tokens=max_tokens,
        profile_tokens=profile_tokens,
        sample_tokens=sample_tokens,
        token_cost=token_cost,
        within_budget=token_cost <= max_tokens,
    )


def fit_to_window(
    data: Dataset | list[dict[str, Any]],
    *,
    max_tokens: int,
    method: SampleMethod | str = SampleMethod.RESERVOIR,
    by: str | Sequence[str] | None = None,
    seed: int = 0,
    sample_cap: int = DEFAULT_SAMPLE_CAP,
    model: str | None = None,
    profile_kwargs: dict[str, Any] | None = None,
) -> WindowFit:
    """Fit an in-memory dataset into ``max_tokens``: a full column profile plus a
    representative sample sized to the budget the profile leaves.

    ``method`` selects how the sample is drawn (``reservoir`` by default;
    ``stratified`` requires ``by`` and keeps that column's distribution in the
    sample). Returns a :class:`WindowFit` whose combined encoding is within
    budget; deterministic for a given ``seed``.
    """
    from ..core.errors import DataError

    if max_tokens <= 0:
        raise DataError(f"max_tokens must be positive, got {max_tokens}")
    method = SampleMethod(method)
    dataset = data if isinstance(data, Dataset) else Dataset.from_records(data)
    profile = profile_dataset(dataset, **(profile_kwargs or {}))
    candidate = (
        dataset
        if dataset.row_count <= sample_cap
        else sample_dataset(dataset, sample_cap, method=method, by=by, seed=seed)
    )
    return _assemble(
        profile,
        candidate,
        original_row_count=dataset.row_count,
        max_tokens=max_tokens,
        method=method,
        by=by,
        seed=seed,
        model=model,
    )


def fit_stream(
    rows: Iterable[Sequence[Any]],
    schema: Sequence[ColumnSchema],
    *,
    max_tokens: int,
    name: str = "",
    seed: int = 0,
    sample_cap: int = DEFAULT_SAMPLE_CAP,
    model: str | None = None,
    profile_kwargs: dict[str, Any] | None = None,
) -> WindowFit:
    """Fit a row iterator into ``max_tokens`` in a single bounded pass — profile
    and reservoir-sample together without materializing the table. The path for a
    source far larger than memory; uses uniform reservoir sampling (stratified
    sampling needs the whole dataset, so use :func:`fit_to_window` for that)."""
    from ..core.errors import DataError

    if max_tokens <= 0:
        raise DataError(f"max_tokens must be positive, got {max_tokens}")
    columns = list(schema)
    width = len(columns)
    pk = profile_kwargs or {}
    accumulators = _make_accumulators(
        columns,
        bins=pk.get("bins", 10),
        top_k=pk.get("top_k", 10),
        max_exemplars=pk.get("max_exemplars", 3),
        percentiles=pk.get("percentiles", (25, 50, 75, 95, 99)),
        distinct_cap=pk.get("distinct_cap", 10_000),
        reservoir_size=pk.get("reservoir_size", 2_048),
        seed=seed,
    )
    rng = random.Random(seed)
    reservoir: list[tuple[int, list[Any]]] = []
    row_count = 0
    for row in rows:
        materialized = [row[j] if j < len(row) else None for j in range(width)]
        for j in range(width):
            accumulators[j].add(materialized[j])
        if row_count < sample_cap:
            reservoir.append((row_count, materialized))
        else:
            k = rng.randint(0, row_count)
            if k < sample_cap:
                reservoir[k] = (row_count, materialized)
        row_count += 1
    profile = _build_profile(accumulators, name=name, row_count=row_count, sampled=False)
    reservoir.sort(key=lambda pair: pair[0])
    candidate = Dataset.from_rows([row for _, row in reservoir], list(columns), name=name)
    candidate.metadata = {"sample": {"method": SampleMethod.RESERVOIR.value, "size": candidate.row_count, "of": row_count, "seed": seed}}
    return _assemble(
        profile,
        candidate,
        original_row_count=row_count,
        max_tokens=max_tokens,
        method=SampleMethod.RESERVOIR,
        by=None,
        seed=seed,
        model=model,
    )
