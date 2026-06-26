"""Representative sampling for fitting a dataset larger than the window.

A hard ``head(1000)`` cutoff is not a sample — it is the first thousand rows,
biased by whatever order the source returned. The data plane instead stands a
*representative* sample in for the whole:

* :func:`reservoir_sample` draws a uniform sample of ``k`` items in a single
  pass over any iterable in ``O(k)`` memory — the row order is irrelevant and
  the source need never be materialized (so a connector can sample a result set
  far larger than memory).
* :func:`stratified_sample` preserves the distribution of a key column: each
  stratum keeps its proportional share of the sample, so a rare category is not
  washed out by a uniform draw.
* :func:`systematic_sample` takes evenly-spaced rows with no randomness — a
  deterministic, evenly-covering baseline.

Every sample is seeded, so the same input and seed yield the same rows, and a
:class:`~vincio.data.Dataset` sample records how it was drawn in its metadata so
a downstream reader knows it stands in for a larger whole.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Any, TypeVar

from ..core.errors import DataError
from .core import Dataset

__all__ = [
    "SampleMethod",
    "reservoir_sample",
    "stratified_sample",
    "systematic_sample",
    "sample_dataset",
]

T = TypeVar("T")


class SampleMethod(StrEnum):
    """How a sample stands in for the whole dataset.

    - ``head`` — the first ``k`` rows (fast, but order-biased; the legacy cutoff).
    - ``reservoir`` — a uniform random sample in a single bounded pass.
    - ``stratified`` — proportional across a key column, preserving its
      distribution (requires ``by``).
    - ``systematic`` — evenly-spaced rows, deterministic and randomness-free.
    """

    HEAD = "head"
    RESERVOIR = "reservoir"
    STRATIFIED = "stratified"
    SYSTEMATIC = "systematic"


def _indexed_reservoir(items: Iterable[T], k: int, *, seed: int) -> list[tuple[int, T]]:
    """Algorithm R over an iterable, returning ``(original_index, item)`` pairs
    sorted by original index so the sample preserves input order. Bounded to
    ``O(k)`` memory regardless of the input length."""
    rng = random.Random(seed)
    reservoir: list[tuple[int, T]] = []
    for index, item in enumerate(items):
        if index < k:
            reservoir.append((index, item))
            continue
        j = rng.randint(0, index)
        if j < k:
            reservoir[j] = (index, item)
    reservoir.sort(key=lambda pair: pair[0])
    return reservoir


def reservoir_sample(
    items: Iterable[T], k: int, *, seed: int = 0, preserve_order: bool = True
) -> list[T]:
    """Draw a uniform random sample of up to ``k`` items in a single pass.

    Uses reservoir sampling (Algorithm R): every item has an equal probability
    of being kept, the iterable is consumed once, and only ``k`` items are ever
    held — so an iterator far larger than memory can be sampled. The draw is
    deterministic for a given ``seed``. With ``preserve_order`` (the default) the
    sample is returned in the items' original order; otherwise in reservoir order.
    """
    if k < 0:
        raise DataError(f"sample size must be non-negative, got {k}")
    if k == 0:
        return []
    indexed = _indexed_reservoir(items, k, seed=seed)
    if not preserve_order:
        rng = random.Random(seed + 1)
        rng.shuffle(indexed)
    return [item for _, item in indexed]


def _coerce_dataset(data: Dataset | list[dict[str, Any]]) -> Dataset:
    if isinstance(data, Dataset):
        return data
    if isinstance(data, list):
        return Dataset.from_records(data)
    raise DataError(f"cannot sample {type(data).__name__}; pass a Dataset or list of records")


def _select_rows(dataset: Dataset, indices: Sequence[int], *, method: str, seed: int, by: Any = None) -> Dataset:
    """Build a new dataset from the chosen row indices, preserving the schema and
    recording how the sample was drawn."""
    cells = [[column[i] for i in indices] for column in dataset.cells]
    metadata = dict(dataset.metadata)
    sample_meta: dict[str, Any] = {
        "method": method,
        "size": len(indices),
        "of": dataset.row_count,
        "seed": seed,
    }
    if by is not None:
        sample_meta["by"] = by
    metadata["sample"] = sample_meta
    return Dataset(
        name=dataset.name,
        columns=list(dataset.columns),
        cells=cells,
        source=dataset.source,
        metadata=metadata,
    )


def systematic_sample(data: Dataset | list[dict[str, Any]], k: int) -> Dataset:
    """Take up to ``k`` evenly-spaced rows — a deterministic, randomness-free
    sample that covers the dataset uniformly from first row to last."""
    dataset = _coerce_dataset(data)
    n = dataset.row_count
    if k < 0:
        raise DataError(f"sample size must be non-negative, got {k}")
    if k == 0:
        return _select_rows(dataset, [], method=SampleMethod.SYSTEMATIC.value, seed=0)
    if k >= n:
        return _select_rows(dataset, list(range(n)), method=SampleMethod.SYSTEMATIC.value, seed=0)
    # Evenly spaced positions, de-duplicated while preserving order.
    seen: set[int] = set()
    indices: list[int] = []
    for i in range(k):
        pos = (i * n) // k
        if pos not in seen:
            seen.add(pos)
            indices.append(pos)
    return _select_rows(dataset, indices, method=SampleMethod.SYSTEMATIC.value, seed=0)


def stratified_sample(
    data: Dataset | list[dict[str, Any]],
    k: int,
    *,
    by: str | Sequence[str],
    seed: int = 0,
) -> Dataset:
    """Draw a sample of up to ``k`` rows that preserves the distribution of the
    ``by`` column(s): each distinct stratum keeps a share of the sample
    proportional to its share of the dataset (largest-remainder rounding), and
    rows within a stratum are reservoir-sampled. A rare category therefore keeps
    representation a uniform draw would wash out."""
    dataset = _coerce_dataset(data)
    n = dataset.row_count
    if k < 0:
        raise DataError(f"sample size must be non-negative, got {k}")
    keys = [by] if isinstance(by, str) else list(by)
    col_indices = []
    for key in keys:
        try:
            col_indices.append(dataset.column_names.index(key))
        except ValueError as exc:
            raise DataError(f"no column named {key!r} to stratify by") from exc
    target = min(k, n)
    if target == 0:
        return _select_rows(dataset, [], method=SampleMethod.STRATIFIED.value, seed=seed, by=by)
    if target == n:
        return _select_rows(dataset, list(range(n)), method=SampleMethod.STRATIFIED.value, seed=seed, by=by)

    # Group row indices by stratum, preserving first-seen order for determinism.
    strata: dict[tuple[Any, ...], list[int]] = {}
    for i in range(n):
        stratum = tuple(dataset.cells[j][i] for j in col_indices)
        strata.setdefault(stratum, []).append(i)
    ordered = sorted(strata, key=lambda s: tuple(str(v) for v in s))
    rank_of = {s: i for i, s in enumerate(ordered)}

    # Largest-remainder proportional allocation, capped at each stratum's size.
    quotas = {s: target * len(strata[s]) / n for s in ordered}
    alloc = {s: min(len(strata[s]), int(quotas[s])) for s in ordered}
    assigned = sum(alloc.values())
    while assigned < target:
        candidates = [s for s in ordered if alloc[s] < len(strata[s])]
        if not candidates:
            break
        # Give the next slot to the stratum furthest below its quota (deterministic).
        pick = max(candidates, key=lambda s: (quotas[s] - alloc[s], -rank_of[s]))
        alloc[pick] += 1
        assigned += 1

    chosen: list[int] = []
    for rank, stratum in enumerate(ordered):
        share = alloc[stratum]
        if share <= 0:
            continue
        chosen.extend(reservoir_sample(strata[stratum], share, seed=seed + rank))
    chosen.sort()
    return _select_rows(dataset, chosen, method=SampleMethod.STRATIFIED.value, seed=seed, by=by)


def sample_dataset(
    data: Dataset | list[dict[str, Any]],
    k: int,
    *,
    method: SampleMethod | str = SampleMethod.RESERVOIR,
    by: str | Sequence[str] | None = None,
    seed: int = 0,
) -> Dataset:
    """Sample up to ``k`` rows, returning a schema-preserving
    :class:`~vincio.data.Dataset` that records how it was drawn in
    ``metadata['sample']``.

    ``method`` selects the strategy (see :class:`SampleMethod`); ``stratified``
    requires ``by``. The result stands in for the whole dataset — encode it,
    profile it, or carry it as table evidence exactly like any other dataset.
    """
    method = SampleMethod(method)
    dataset = _coerce_dataset(data)
    if method is SampleMethod.STRATIFIED:
        if by is None:
            raise DataError("stratified sampling requires `by=` (a column or columns)")
        return stratified_sample(dataset, k, by=by, seed=seed)
    if method is SampleMethod.SYSTEMATIC:
        return systematic_sample(dataset, k)
    if method is SampleMethod.HEAD:
        head = dataset.head(k)
        head.metadata = {
            **dataset.metadata,
            "sample": {"method": method.value, "size": head.row_count, "of": dataset.row_count, "seed": seed},
        }
        return head
    indices = reservoir_sample(range(dataset.row_count), k, seed=seed)
    return _select_rows(dataset, indices, method=method.value, seed=seed)
