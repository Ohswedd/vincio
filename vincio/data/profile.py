"""Deterministic, bounded-memory dataset profiling.

A profile is the faithful *summary* a model can reason over when the rows
themselves will not fit: per column, its type, null rate, cardinality, range,
central tendency, a distribution histogram, and a handful of exemplars. The
profile is computed in a **single pass with bounded memory** — exact counts,
extrema, and moments accumulate in ``O(1)`` per column while percentiles and the
histogram are estimated from a fixed-size reservoir — so a table of any height
profiles inside the same footprint, and the profile itself is a fixed-size piece
of evidence the context compiler scores, budgets, orders, and cites exactly like
a table.

Everything here is deterministic (a seeded reservoir over a fixed input order)
and dependency-free. :func:`profile_dataset` profiles an in-memory
:class:`~vincio.data.Dataset`; :func:`profile_stream` profiles a row iterator
without materializing it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .core import ColumnSchema, Dataset, DataType

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.types import EvidenceItem
    from .evidence import TableEvidence

__all__ = [
    "HistogramBin",
    "ColumnProfile",
    "DatasetProfile",
    "profile_dataset",
    "profile_stream",
    "DEFAULT_PERCENTILES",
]

DEFAULT_PERCENTILES: tuple[int, ...] = (25, 50, 75, 95, 99)
_NUMERIC_TYPES = (DataType.INT, DataType.FLOAT)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class HistogramBin(BaseModel):
    """One bucket of a column's distribution. A numeric column's bins carry a
    half-open ``[lo, hi)`` range (the last bin is closed); a categorical column's
    bins carry a value ``label`` instead. ``count`` is the population estimate."""

    count: int = 0
    lo: float | None = None
    hi: float | None = None
    label: str | None = None


class ColumnProfile(BaseModel):
    """The deterministic summary of one column.

    Exact figures (``count``, ``null_count``, ``null_rate``, ``min``, ``max``,
    ``mean``, ``stddev``) are computed over every value; ``distinct`` is exact up
    to a cap and otherwise a lower bound (``distinct_is_lower_bound``);
    ``percentiles`` and ``histogram`` are estimated from a fixed-size reservoir
    when the column is larger than it (``estimated``).
    """

    name: str
    dtype: DataType = DataType.STR
    unit: str | None = None
    nullable: bool = False
    count: int = 0
    null_count: int = 0
    null_rate: float = 0.0
    distinct: int = 0
    distinct_is_lower_bound: bool = False
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    stddev: float | None = None
    percentiles: dict[str, float] = Field(default_factory=dict)
    histogram: list[HistogramBin] = Field(default_factory=list)
    top_values: list[tuple[Any, int]] = Field(default_factory=list)
    exemplars: list[Any] = Field(default_factory=list)
    estimated: bool = False

    @property
    def is_numeric(self) -> bool:
        return self.dtype in _NUMERIC_TYPES


class _ColumnAccumulator:
    """Single-pass, bounded-memory statistics for one column."""

    def __init__(
        self,
        schema: ColumnSchema,
        *,
        bins: int,
        top_k: int,
        max_exemplars: int,
        percentiles: Sequence[int],
        distinct_cap: int,
        reservoir_size: int,
        seed: int,
    ) -> None:
        self.schema = schema
        self.numeric = schema.dtype in _NUMERIC_TYPES
        self.bins = bins
        self.top_k = top_k
        self.max_exemplars = max_exemplars
        self.percentiles = list(percentiles)
        self.distinct_cap = distinct_cap
        self.reservoir_size = reservoir_size
        self.count = 0
        self.null_count = 0
        self.nonnull = 0
        self._distinct: set[Any] = set()
        self._distinct_overflow = False
        self._counts: dict[Any, int] = {}
        self._counts_overflow = False
        self._exemplars: list[Any] = []
        self.min: float | None = None
        self.max: float | None = None
        self._sum = 0.0
        self._sum_sq = 0.0
        self._n_num = 0
        self._reservoir: list[float] = []
        self._rng = random.Random(seed)

    def add(self, value: Any) -> None:
        self.count += 1
        if value is None:
            self.null_count += 1
            return
        self.nonnull += 1
        # Cardinality (exact up to the cap, then a lower bound).
        if not self._distinct_overflow:
            self._distinct.add(value)
            if len(self._distinct) > self.distinct_cap:
                self._distinct_overflow = True
        # Top values (a bounded value→count counter).
        if value in self._counts:
            self._counts[value] += 1
        elif not self._counts_overflow:
            self._counts[value] = 1
            if len(self._counts) > self.top_k * 8:
                self._counts_overflow = True
        if len(self._exemplars) < self.max_exemplars and value not in self._exemplars:
            self._exemplars.append(value)
        if self.numeric and _is_number(value):
            x = float(value)
            self._n_num += 1
            self._sum += x
            self._sum_sq += x * x
            self.min = x if self.min is None else min(self.min, x)
            self.max = x if self.max is None else max(self.max, x)
            self._reservoir_add(x)

    def _reservoir_add(self, x: float) -> None:
        if len(self._reservoir) < self.reservoir_size:
            self._reservoir.append(x)
            return
        j = self._rng.randint(0, self._n_num - 1)
        if j < self.reservoir_size:
            self._reservoir[j] = x

    def finalize(self) -> ColumnProfile:
        estimated = self.numeric and self._n_num > self.reservoir_size
        distinct = min(len(self._distinct), self.distinct_cap) if self._distinct_overflow else len(self._distinct)
        mean = stddev = None
        percentiles: dict[str, float] = {}
        histogram: list[HistogramBin] = []
        if self.numeric and self._n_num:
            mean = round(self._sum / self._n_num, 6)
            variance = max(0.0, self._sum_sq / self._n_num - (self._sum / self._n_num) ** 2)
            stddev = round(math.sqrt(variance), 6)
            ordered = sorted(self._reservoir)
            percentiles = {f"p{p}": round(_percentile(ordered, p), 6) for p in self.percentiles}
            histogram = _numeric_histogram(ordered, self.min, self.max, self.bins, self._n_num)
        elif not self.numeric:
            histogram = self._categorical_histogram()
        return ColumnProfile(
            name=self.schema.name,
            dtype=self.schema.dtype,
            unit=self.schema.unit,
            nullable=self.schema.nullable,
            count=self.nonnull,
            null_count=self.null_count,
            null_rate=round(self.null_count / self.count, 6) if self.count else 0.0,
            distinct=distinct,
            distinct_is_lower_bound=self._distinct_overflow,
            min=self.min,
            max=self.max,
            mean=mean,
            stddev=stddev,
            percentiles=percentiles,
            histogram=histogram,
            top_values=self._top_values(),
            exemplars=list(self._exemplars),
            estimated=estimated,
        )

    def _top_values(self) -> list[tuple[Any, int]]:
        ranked = sorted(self._counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
        return ranked[: self.top_k]

    def _categorical_histogram(self) -> list[HistogramBin]:
        return [HistogramBin(label=str(value), count=count) for value, count in self._top_values()]


def _percentile(ordered: list[float], pct: float) -> float:
    """Linear-interpolation percentile over a sorted list."""
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def _numeric_histogram(
    ordered: list[float], lo: float | None, hi: float | None, bins: int, total: int
) -> list[HistogramBin]:
    """An equi-width histogram over ``[lo, hi]``, with bin counts scaled from the
    reservoir to the column's full population so the shape reflects the whole."""
    if lo is None or hi is None or not ordered:
        return []
    if hi <= lo:
        return [HistogramBin(lo=lo, hi=hi, count=total)]
    width = (hi - lo) / bins
    counts = [0] * bins
    for x in ordered:
        idx = int((x - lo) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    scale = total / len(ordered)
    return [
        HistogramBin(lo=round(lo + i * width, 6), hi=round(lo + (i + 1) * width, 6), count=round(c * scale))
        for i, c in enumerate(counts)
    ]


class DatasetProfile(BaseModel):
    """A dataset's deterministic, fixed-size column profile.

    The profile is the same size whether it summarizes a thousand rows or ten
    million, so it stands in for a table that will never fit: project it to
    evidence with :meth:`to_evidence` (it renders as a compact stats table the
    compiler scores and cites) or read a column's summary with :meth:`column`.
    """

    name: str = ""
    row_count: int = 0
    column_count: int = 0
    columns: list[ColumnProfile] = Field(default_factory=list)
    sampled: bool = False
    estimated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def column(self, name: str) -> ColumnProfile:
        """The profile of the named column."""
        for col in self.columns:
            if col.name == name:
                return col
        from ..core.errors import DataError

        raise DataError(f"no profiled column named {name!r}")

    def to_dataset(self) -> Dataset:
        """Render the profile as a compact stats table — one row per column —
        suitable for encoding or carrying as evidence."""
        records: list[dict[str, Any]] = []
        for col in self.columns:
            records.append(
                {
                    "column": col.name,
                    "dtype": col.dtype.value,
                    "count": col.count,
                    "nulls": col.null_count,
                    "null_rate": col.null_rate,
                    "distinct": col.distinct,
                    "min": col.min,
                    "max": col.max,
                    "mean": col.mean,
                    "stddev": col.stddev,
                    "p50": col.percentiles.get("p50"),
                    "p95": col.percentiles.get("p95"),
                    "top": _format_top(col),
                }
            )
        name = f"{self.name}_profile" if self.name else "profile"
        return Dataset.from_records(records, name=name)

    def encode(self) -> str:
        """The compact encoding of the profile's stats table."""
        return self.to_dataset().encode()

    def token_cost(self, *, model: str | None = None) -> int:
        """The exact token cost of the encoded profile."""
        return self.to_dataset().token_cost(model=model)

    def to_evidence(self, *, source_id: str = "", caption: str = "", **kwargs: Any) -> TableEvidence:
        """Project the profile into first-class table evidence the compiler scores,
        budgets, orders, and cites."""
        rows = self.row_count
        default_caption = f"Column profile of {self.name or 'dataset'} ({rows:,} rows)"
        return self.to_dataset().to_evidence(
            source_id=source_id or (f"{self.name}_profile" if self.name else "profile"),
            caption=caption or default_caption,
            **kwargs,
        )

    def to_evidence_item(self, **kwargs: Any) -> EvidenceItem:
        """Project straight to a ``modality='table'``
        :class:`~vincio.core.types.EvidenceItem`."""
        return self.to_evidence(**kwargs).to_evidence_item()

    def summary(self) -> str:
        """A one-line human summary of the profile."""
        kind = "sampled " if self.sampled else ""
        approx = " (estimated)" if self.estimated else ""
        return (
            f"{self.name or 'dataset'}: {self.row_count:,} {kind}rows × "
            f"{self.column_count} columns{approx}"
        )


def _format_top(col: ColumnProfile) -> str | None:
    if not col.top_values:
        return None
    return ", ".join(f"{value}×{count}" for value, count in col.top_values[:3])


def _build_profile(
    accumulators: list[_ColumnAccumulator],
    *,
    name: str,
    row_count: int,
    sampled: bool,
) -> DatasetProfile:
    columns = [acc.finalize() for acc in accumulators]
    return DatasetProfile(
        name=name,
        row_count=row_count,
        column_count=len(columns),
        columns=columns,
        sampled=sampled,
        estimated=any(c.estimated or c.distinct_is_lower_bound for c in columns),
    )


def _make_accumulators(
    schema_columns: Sequence[ColumnSchema],
    *,
    bins: int,
    top_k: int,
    max_exemplars: int,
    percentiles: Sequence[int],
    distinct_cap: int,
    reservoir_size: int,
    seed: int,
) -> list[_ColumnAccumulator]:
    return [
        _ColumnAccumulator(
            col,
            bins=bins,
            top_k=top_k,
            max_exemplars=max_exemplars,
            percentiles=percentiles,
            distinct_cap=distinct_cap,
            reservoir_size=reservoir_size,
            seed=seed + i,
        )
        for i, col in enumerate(schema_columns)
    ]


def profile_dataset(
    data: Dataset | list[dict[str, Any]],
    *,
    bins: int = 10,
    top_k: int = 10,
    max_exemplars: int = 3,
    percentiles: Sequence[int] = DEFAULT_PERCENTILES,
    distinct_cap: int = 10_000,
    reservoir_size: int = 2_048,
    seed: int = 0,
) -> DatasetProfile:
    """Profile an in-memory dataset, column by column, in bounded memory.

    Returns a :class:`DatasetProfile`: per-column type, null rate, cardinality,
    extrema, mean/stddev, percentiles, a distribution histogram, and exemplars.
    Exact figures cover every value; percentiles and histograms are estimated
    from a fixed-size reservoir once a column exceeds ``reservoir_size`` (flagged
    on the column and the profile). Deterministic for a given ``seed``.
    """
    if isinstance(data, list):
        data = Dataset.from_records(data)
    accumulators = _make_accumulators(
        data.columns,
        bins=bins,
        top_k=top_k,
        max_exemplars=max_exemplars,
        percentiles=percentiles,
        distinct_cap=distinct_cap,
        reservoir_size=reservoir_size,
        seed=seed,
    )
    for j, acc in enumerate(accumulators):
        for value in data.cells[j]:
            acc.add(value)
    return _build_profile(
        accumulators,
        name=data.name,
        row_count=data.row_count,
        sampled=bool(data.metadata.get("sample")),
    )


def profile_stream(
    rows: Iterable[Sequence[Any]],
    schema: Sequence[ColumnSchema],
    *,
    name: str = "",
    bins: int = 10,
    top_k: int = 10,
    max_exemplars: int = 3,
    percentiles: Sequence[int] = DEFAULT_PERCENTILES,
    distinct_cap: int = 10_000,
    reservoir_size: int = 2_048,
    seed: int = 0,
) -> DatasetProfile:
    """Profile a row iterator without materializing it — the bounded-memory path
    for a table far larger than memory. ``schema`` declares the columns the rows
    are positionally aligned to; the iterator is consumed once."""
    columns = list(schema)
    width = len(columns)
    accumulators = _make_accumulators(
        columns,
        bins=bins,
        top_k=top_k,
        max_exemplars=max_exemplars,
        percentiles=percentiles,
        distinct_cap=distinct_cap,
        reservoir_size=reservoir_size,
        seed=seed,
    )
    row_count = 0
    for row in rows:
        row_count += 1
        for j in range(width):
            accumulators[j].add(row[j] if j < len(row) else None)
    return _build_profile(accumulators, name=name, row_count=row_count, sampled=False)
