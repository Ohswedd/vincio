"""Streaming and out-of-core processing for datasets larger than memory.

The profiling, sampling, and fit-in-window rungs already process a row *stream*
in a single bounded pass (:func:`~vincio.data.profile_stream`,
:func:`~vincio.data.fit_stream`). This module makes that the first-class way to
hold a dataset that will never fit in memory: a :class:`RowStream` is a lazy,
*re-iterable*, schema-bearing handle over a row source — a list, a generator
factory, or a CSV / JSON-Lines file read line by line — that the rest of the
data plane consumes without ever materializing the whole table.

* :class:`RowStream` — the out-of-core analogue of a :class:`~vincio.data.Dataset`.
  Build one from records, rows, a dataset, or a file (:meth:`RowStream.from_csv` /
  :meth:`RowStream.from_jsonl` / :meth:`RowStream.open`), then iterate it in
  bounded :meth:`~RowStream.chunks`, :meth:`~RowStream.profile` it,
  :meth:`~RowStream.fit` it into a token budget, :meth:`~RowStream.sample` it, or
  :meth:`~RowStream.aggregate` it — each a single bounded pass whose footprint is
  invariant to the row count.
* :func:`stream_aggregate` / :class:`StreamAggregation` — a deterministic,
  bounded-memory group-by over a stream: the working set tracks the number of
  *groups*, not the number of rows, so a billion-row table aggregates inside a
  fixed footprint and a group cardinality beyond the bound is refused rather than
  silently spilling.
* :func:`encode_stream` — render a stream to the compact, lossless encoding (and
  optionally gzip it) header-once, row-by-row, to an in-memory buffer or straight
  to a file sink, so a dataset larger than memory is compressed in one bounded
  pass.
* :func:`stream_map` / :class:`BulkMapResult` — run an analytical transform over
  a stream *at scale* by chunking it into the existing
  :class:`~vincio.providers.BatchRunner` (half-cost provider batch APIs, bounded
  concurrency), reconciled by chunk.

Everything here is deterministic (seeded reservoirs over a fixed input order) and
dependency-free; the CSV / JSON-Lines readers use only the standard library.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from ..core import tabular
from ..core.errors import StreamError
from .core import ColumnSchema, DataSchema, Dataset, DataType
from .profile import DatasetProfile, profile_stream
from .sampling import reservoir_sample
from .window import WindowFit, fit_stream

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.types import EvidenceItem, ModelRequest
    from ..providers.base import ModelProvider
    from ..providers.batch import BatchBackend, BatchRunner

__all__ = [
    "RowStream",
    "StreamAggregation",
    "stream_aggregate",
    "encode_stream",
    "BulkMapResult",
    "stream_map",
    "DEFAULT_CHUNK_ROWS",
    "DEFAULT_MAX_GROUPS",
]

Row = Sequence[Any]
RowFactory = Callable[[], Iterator[Row]]
_T = TypeVar("_T")

DEFAULT_CHUNK_ROWS = 4_096
DEFAULT_MAX_GROUPS = 1_000_000
# How many rows a file/record reader peeks to infer a schema when none is given.
_SCHEMA_PEEK_ROWS = 256
# Aggregations a measure column supports. Every group also carries its row
# ``count`` unconditionally, so it is not a per-measure choice.
_AGGREGATIONS = ("sum", "mean", "min", "max")


class _OneShot(Generic[_T]):
    """A factory wrapper for a single-use iterator: the first call hands back the
    underlying iterator; a second call refuses rather than silently yielding an
    empty pass."""

    def __init__(self, iterator: Iterator[_T]) -> None:
        self._iterator: Iterator[_T] | None = iterator

    def __call__(self) -> Iterator[_T]:
        if self._iterator is None:
            raise StreamError(
                "this RowStream is backed by a one-shot iterator and has already "
                "been consumed; pass a sequence, a re-iterable, or a zero-argument "
                "callable returning a fresh iterator (or use RowStream.from_csv / "
                ".from_jsonl) so it can be read in more than one pass"
            )
        iterator, self._iterator = self._iterator, None
        return iterator


def _as_factory(source: Iterable[_T] | Callable[[], Iterator[_T]]) -> Callable[[], Iterator[_T]]:
    """Resolve a row source into a re-iterable factory. A zero-argument callable
    is used as-is; a list/tuple/range/view is wrapped fresh each pass; a bare
    generator or iterator is allowed exactly one pass."""
    if callable(source) and not isinstance(source, (list, tuple)):
        return source
    if isinstance(source, (list, tuple)):
        items = list(source)
        return lambda: iter(items)
    iterator = iter(source)
    if iterator is source:  # a one-shot iterator/generator returns itself from iter()
        return _OneShot(iterator)
    reiterable = source
    return lambda: iter(reiterable)


class RowStream:
    """A lazy, re-iterable, schema-bearing handle over an out-of-core row source.

    A :class:`RowStream` never materializes its rows: it holds a *factory* that
    produces a fresh row iterator on demand, so the same stream can be profiled,
    fitted, sampled, and aggregated — each a single bounded pass. The schema is
    declared once and the rows are positionally aligned to it.

    Build one from in-memory data (:meth:`from_records` / :meth:`from_rows` /
    :meth:`from_dataset`), a re-iterable generator factory, or a file
    (:meth:`from_csv` / :meth:`from_jsonl` / :meth:`open`). A bare generator
    object is single-use; pass a sequence or a zero-argument callable for a
    source that must be read more than once.
    """

    def __init__(
        self,
        source: Iterable[Row] | RowFactory,
        schema: DataSchema | Sequence[ColumnSchema],
        *,
        name: str = "",
        source_id: str = "",
    ) -> None:
        columns = schema.columns if isinstance(schema, DataSchema) else list(schema)
        if not columns:
            raise StreamError("a RowStream needs a non-empty schema (declare its columns)")
        self.columns: list[ColumnSchema] = columns
        self.name = name
        self.source_id = source_id
        self._factory = _as_factory(source)

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Row] | RowFactory,
        schema: DataSchema | Sequence[ColumnSchema] | Sequence[str],
        *,
        name: str = "",
        source_id: str = "",
    ) -> RowStream:
        """Wrap a row source (a sequence, a re-iterable, or a zero-argument
        callable returning an iterator) and a schema. Bare column names are typed
        ``str`` — declare typed :class:`~vincio.data.ColumnSchema` columns (or
        let :meth:`from_csv` infer them) for numeric columns."""
        return cls(rows, _coerce_schema(schema), name=name, source_id=source_id)

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]] | Callable[[], Iterator[Mapping[str, Any]]],
        *,
        schema: DataSchema | Sequence[ColumnSchema] | Sequence[str] | None = None,
        name: str = "",
        source_id: str = "",
    ) -> RowStream:
        """Wrap a source of record mappings. When no schema is given the columns
        and their types are inferred from a bounded peek of the first records; an
        explicit schema fixes them without peeking."""
        record_factory = _as_factory(records)
        resolved = _resolve_record_schema(record_factory, schema)
        names = resolved.names

        def rows() -> Iterator[Row]:
            for record in record_factory():
                yield [record.get(col) for col in names]

        return cls(rows, resolved, name=name, source_id=source_id)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> RowStream:
        """Wrap an in-memory :class:`~vincio.data.Dataset` as a stream (so the
        streaming operators work uniformly over both)."""
        return cls(
            lambda: iter(dataset.rows()),
            DataSchema(columns=list(dataset.columns)),
            name=dataset.name,
            source_id=dataset.source,
        )

    @classmethod
    def from_csv(
        cls,
        source: str | Path | Iterable[str],
        *,
        schema: DataSchema | Sequence[ColumnSchema] | Sequence[str] | None = None,
        has_header: bool = True,
        delimiter: str = ",",
        name: str = "",
        encoding: str = "utf-8",
    ) -> RowStream:
        """Stream rows from a CSV file (a path) or an iterable of text lines,
        read lazily line by line so a file larger than memory is never loaded
        whole. When no schema is given the column names come from the header row
        (or are positional) and each column's type is inferred from a bounded
        peek and coerced losslessly (a value that does not round-trip stays
        text)."""
        return _csv_stream(
            source,
            schema=schema,
            has_header=has_header,
            delimiter=delimiter,
            name=name,
            encoding=encoding,
        )

    @classmethod
    def from_jsonl(
        cls,
        source: str | Path | Iterable[str],
        *,
        schema: DataSchema | Sequence[ColumnSchema] | Sequence[str] | None = None,
        name: str = "",
        encoding: str = "utf-8",
    ) -> RowStream:
        """Stream rows from a JSON-Lines / NDJSON file (a path) or an iterable of
        text lines — one JSON object (or array) per line, parsed lazily. Object
        values are already typed, so the schema is inferred from a bounded peek
        unless given; array lines require an explicit ``schema``."""
        return _jsonl_stream(source, schema=schema, name=name, encoding=encoding)

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        format: str | None = None,
        schema: DataSchema | Sequence[ColumnSchema] | Sequence[str] | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> RowStream:
        """Open a file as a stream, choosing the reader by ``format`` (``"csv"``
        or ``"jsonl"``) or by the path's extension (``.csv`` /
        ``.jsonl`` / ``.ndjson``)."""
        resolved = (format or Path(path).suffix.lstrip(".")).lower()
        stream_name = name or Path(path).stem
        if resolved in ("jsonl", "ndjson", "json"):
            return cls.from_jsonl(path, schema=schema, name=stream_name, **kwargs)
        if resolved in ("csv", "tsv", "txt"):
            kwargs.setdefault("delimiter", "\t" if resolved == "tsv" else ",")
            return cls.from_csv(path, schema=schema, name=stream_name, **kwargs)
        raise StreamError(
            f"cannot infer a reader for {resolved!r}; pass format='csv' or format='jsonl'"
        )

    # -- access ----------------------------------------------------------------

    @property
    def data_schema(self) -> DataSchema:
        """The stream's typed :class:`~vincio.data.DataSchema`."""
        return DataSchema(columns=self.columns)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def width(self) -> int:
        return len(self.columns)

    def rows(self) -> Iterator[Row]:
        """A fresh iterator over the stream's rows (one bounded pass)."""
        return self._factory()

    def __iter__(self) -> Iterator[Row]:
        return self.rows()

    def chunks(self, size: int = DEFAULT_CHUNK_ROWS) -> Iterator[Dataset]:
        """Iterate the stream as bounded :class:`~vincio.data.Dataset` chunks of
        up to ``size`` rows — the unit of out-of-core processing. At most ``size``
        rows are resident at once, whatever the stream's length."""
        if size <= 0:
            raise StreamError(f"chunk size must be positive, got {size}")
        schema = self.data_schema
        buffer: list[list[Any]] = []
        for row in self.rows():
            buffer.append(list(row))
            if len(buffer) >= size:
                yield Dataset.from_rows(buffer, schema, name=self.name, source=self.source_id)
                buffer = []
        if buffer:
            yield Dataset.from_rows(buffer, schema, name=self.name, source=self.source_id)

    # -- bounded-pass operators ------------------------------------------------

    def profile(self, **kwargs: Any) -> DatasetProfile:
        """A deterministic, bounded-memory column profile of the whole stream in
        a single pass (see :func:`~vincio.data.profile_stream`)."""
        return profile_stream(self.rows(), self.columns, name=self.name, **kwargs)

    def fit(self, *, max_tokens: int, **kwargs: Any) -> WindowFit:
        """Fit the stream into ``max_tokens`` — a full-fidelity profile plus a
        budget-sized representative sample — in one bounded pass (see
        :func:`~vincio.data.fit_stream`)."""
        return fit_stream(self.rows(), self.columns, max_tokens=max_tokens, name=self.name, **kwargs)

    def sample(self, k: int, *, seed: int = 0) -> Dataset:
        """Draw a uniform reservoir sample of up to ``k`` rows in a single bounded
        pass, returned as a schema-preserving dataset that records how it was
        drawn (the streaming counterpart of :func:`~vincio.data.sample_dataset`)."""
        from .sampling import SampleMethod

        rows = reservoir_sample(self.rows(), k, seed=seed)
        dataset = Dataset.from_rows(
            [list(row) for row in rows], self.data_schema, name=self.name, source=self.source_id
        )
        dataset.metadata = {
            "sample": {"method": SampleMethod.RESERVOIR.value, "size": dataset.row_count, "seed": seed}
        }
        return dataset

    def aggregate(
        self,
        *,
        group_by: str | Sequence[str],
        measures: Mapping[str, str | Sequence[str]] | None = None,
        max_groups: int = DEFAULT_MAX_GROUPS,
    ) -> StreamAggregation:
        """Group the stream by one or more columns and reduce measures over each
        group in a single bounded pass (see :func:`stream_aggregate`)."""
        return stream_aggregate(self, group_by=group_by, measures=measures, max_groups=max_groups)

    def encode(self, *, compress: bool = False, delimiter: str = ",") -> bytes:
        """Render the whole stream to its compact, lossless encoding (optionally
        gzip-compressed) header-once, row-by-row (see :func:`encode_stream`)."""
        return encode_stream(self, compress=compress, delimiter=delimiter)

    def materialize(self) -> Dataset:
        """Read the entire stream into an in-memory :class:`~vincio.data.Dataset`.
        The escape hatch out of out-of-core processing — loads every row, so use
        it only when the dataset is known to fit."""
        return Dataset.from_rows(
            [list(row) for row in self.rows()],
            self.data_schema,
            name=self.name,
            source=self.source_id,
        )


# --------------------------------------------------------------------------- #
# Schema resolution / file readers
# --------------------------------------------------------------------------- #


def _coerce_schema(
    schema: DataSchema | Sequence[ColumnSchema] | Sequence[str],
) -> DataSchema:
    if isinstance(schema, DataSchema):
        return schema
    items = list(schema)
    if items and isinstance(items[0], ColumnSchema):
        return DataSchema(columns=[c for c in items if isinstance(c, ColumnSchema)])
    return DataSchema.from_names([str(c) for c in items])


def _resolve_record_schema(
    record_factory: Callable[[], Iterator[Mapping[str, Any]]],
    schema: DataSchema | Sequence[ColumnSchema] | Sequence[str] | None,
) -> DataSchema:
    if schema is not None:
        return _coerce_schema(schema)
    names: list[str] = []
    seen: set[str] = set()
    peek: list[list[Any]] = []
    for index, record in enumerate(record_factory()):
        for key in record:
            text = str(key)
            if text not in seen:
                seen.add(text)
                names.append(text)
        if index < _SCHEMA_PEEK_ROWS:
            peek.append([record.get(n) for n in names])
        else:
            break
    # Realign the peek to the full column set discovered, then infer types.
    aligned = [[row[j] if j < len(row) else None for j in range(len(names))] for row in peek]
    columns = [
        ColumnSchema(
            name=names[j],
            dtype=DataType(tabular.infer_dtype([row[j] for row in aligned])),
            nullable=any(row[j] is None for row in aligned),
        )
        for j in range(len(names))
    ]
    return DataSchema(columns=columns)


def _ensure_reiterable_lines(source: str | Path | Iterable[str]) -> str | Path | Sequence[str]:
    """Normalize a line source so it can be read more than once (a peek to infer
    the schema, then the rows). A path is reopened lazily and a string is
    re-split each pass, so both stay out-of-core; a one-shot line iterator is
    materialized once (it is already an in-memory source — a file far larger than
    memory is read by passing its path, not an iterator)."""
    if isinstance(source, (str, Path)) or isinstance(source, Sequence):
        return source
    return list(source)


def _iter_text_lines(source: str | Path | Iterable[str], *, encoding: str) -> Iterator[str]:
    """Yield text lines from a path (opened lazily) or an iterable of lines."""
    if isinstance(source, (str, Path)):
        if _looks_like_path(source):
            with open(source, encoding=encoding, newline="") as handle:
                yield from handle
        elif isinstance(source, str):
            yield from source.splitlines()
        else:
            raise StreamError(f"no such file: {source}")
    else:
        yield from source


def _looks_like_path(source: str | Path) -> bool:
    if isinstance(source, Path):
        return True
    # A bare string is a path only when it has no newline and names a real file —
    # otherwise it is treated as inline content (its own lines).
    return "\n" not in source and Path(source).is_file()


def _split_csv_line(line: str, delimiter: str) -> list[str | None]:
    line = line.rstrip("\r\n")
    if line == "":
        return []
    return [None if field_text == "" else field_text for field_text in _csv_fields(line, delimiter)]


def _csv_fields(line: str, delimiter: str) -> list[str]:
    """RFC-4180 field split honoring double-quoted fields (with ``""`` escaping)."""
    fields: list[str] = []
    index = 0
    n = len(line)
    while True:
        if index < n and line[index] == '"':
            index += 1
            buf: list[str] = []
            while index < n:
                char = line[index]
                if char == '"':
                    if index + 1 < n and line[index + 1] == '"':
                        buf.append('"')
                        index += 2
                        continue
                    index += 1
                    break
                buf.append(char)
                index += 1
            fields.append("".join(buf))
        else:
            start = index
            while index < n and line[index] != delimiter:
                index += 1
            fields.append(line[start:index])
        if index < n and line[index] == delimiter:
            index += 1
            if index == n:
                fields.append("")
                break
            continue
        break
    return fields


def _csv_stream(
    source: str | Path | Iterable[str],
    *,
    schema: DataSchema | Sequence[ColumnSchema] | Sequence[str] | None,
    has_header: bool,
    delimiter: str,
    name: str,
    encoding: str,
) -> RowStream:
    source = _ensure_reiterable_lines(source)
    # Resolve column names and the (optional) typed schema up front from a peek;
    # the streaming pass then coerces each row by the resolved types.
    header_names: list[str] | None = None
    peek: list[list[str | None]] = []
    for line in _iter_text_lines(source, encoding=encoding):
        cells = _split_csv_line(line, delimiter)
        if not cells and header_names is None and not peek:
            continue
        if has_header and header_names is None:
            header_names = [str(c) if c is not None else f"c{j}" for j, c in enumerate(cells)]
            continue
        peek.append(cells)
        if len(peek) >= _SCHEMA_PEEK_ROWS:
            break

    resolved = _resolve_csv_schema(schema, header_names, peek)
    casters = [_caster(col.dtype) for col in resolved.columns]
    width = len(resolved.columns)
    skip_header = has_header

    def rows() -> Iterator[Row]:
        seen_header = not skip_header
        for line in _iter_text_lines(source, encoding=encoding):
            cells = _split_csv_line(line, delimiter)
            if not cells:
                continue
            if not seen_header:
                seen_header = True
                continue
            yield [casters[j](cells[j]) if j < len(cells) else None for j in range(width)]

    return RowStream(rows, resolved, name=name)


def _resolve_csv_schema(
    schema: DataSchema | Sequence[ColumnSchema] | Sequence[str] | None,
    header_names: list[str] | None,
    peek: list[list[str | None]],
) -> DataSchema:
    if schema is not None:
        resolved = _coerce_schema(schema)
        if not isinstance(schema, DataSchema) and not _is_typed(schema):
            # Bare names: infer types from the peek so numeric columns are typed.
            resolved = _infer_csv_types([c.name for c in resolved.columns], peek)
        return resolved
    width = max((len(row) for row in peek), default=len(header_names or []))
    names = header_names or [f"c{j}" for j in range(width)]
    return _infer_csv_types(names, peek)


def _is_typed(schema: Sequence[ColumnSchema] | Sequence[str]) -> bool:
    items = list(schema)
    return bool(items) and isinstance(items[0], ColumnSchema)


def _infer_csv_types(names: list[str], peek: list[list[str | None]]) -> DataSchema:
    columns: list[ColumnSchema] = []
    for j, col in enumerate(names):
        values = [row[j] if j < len(row) else None for row in peek]
        nonnull = [v for v in values if v is not None]
        dtype = _infer_string_dtype(nonnull)
        columns.append(
            ColumnSchema(name=col, dtype=dtype, nullable=any(v is None for v in values))
        )
    return DataSchema(columns=columns)


def _infer_string_dtype(values: list[str]) -> DataType:
    """Infer a column type from raw string cells, requiring an exact round-trip
    so a leading-zero id or a thousands-separated number stays text (lossless)."""
    if not values:
        return DataType.STR
    if all(_int_round_trips(v) for v in values):
        return DataType.INT
    if all(_float_round_trips(v) for v in values):
        return DataType.FLOAT
    if all(v.strip().lower() in ("true", "false") for v in values):
        return DataType.BOOL
    return DataType.STR


def _int_round_trips(value: str) -> bool:
    try:
        return str(int(value)) == value
    except ValueError:
        return False


def _float_round_trips(value: str) -> bool:
    try:
        return repr(float(value)) == value
    except ValueError:
        return False


def _caster(dtype: DataType) -> Callable[[str | None], Any]:
    """A per-cell coercion from raw CSV text to the declared type, falling back
    to the original text when a cell does not parse (so a stray value in an
    otherwise-numeric column is preserved rather than raising)."""
    if dtype is DataType.INT:
        return lambda v: _try(v, int)
    if dtype is DataType.FLOAT:
        return lambda v: _try(v, float)
    if dtype is DataType.BOOL:
        return lambda v: (None if v is None else (v.strip().lower() == "true" if v.strip().lower() in ("true", "false") else v))
    return lambda v: v


def _try(value: str | None, cast: Callable[[str], Any]) -> Any:
    if value is None:
        return None
    try:
        return cast(value)
    except (ValueError, TypeError):
        return value


def _jsonl_stream(
    source: str | Path | Iterable[str],
    *,
    schema: DataSchema | Sequence[ColumnSchema] | Sequence[str] | None,
    name: str,
    encoding: str,
) -> RowStream:
    source = _ensure_reiterable_lines(source)
    peek_records: list[Mapping[str, Any]] = []
    array_mode = False
    for line in _iter_text_lines(source, encoding=encoding):
        text = line.strip()
        if not text:
            continue
        parsed = json.loads(text)
        if isinstance(parsed, list):
            array_mode = True
            break
        if isinstance(parsed, dict):
            peek_records.append(parsed)
        if len(peek_records) >= _SCHEMA_PEEK_ROWS:
            break

    if array_mode:
        if schema is None:
            raise StreamError("JSON-Lines array rows require an explicit schema= to name the columns")
        resolved = _coerce_schema(schema)
        width = len(resolved.columns)

        def array_rows() -> Iterator[Row]:
            for raw in _iter_text_lines(source, encoding=encoding):
                text = raw.strip()
                if not text:
                    continue
                values = json.loads(text)
                yield [values[j] if j < len(values) else None for j in range(width)]

        return RowStream(array_rows, resolved, name=name)

    resolved = (
        _coerce_schema(schema)
        if schema is not None
        else _resolve_record_schema(lambda: iter(peek_records), None)
    )
    names = resolved.names

    def object_rows() -> Iterator[Row]:
        for raw in _iter_text_lines(source, encoding=encoding):
            text = raw.strip()
            if not text:
                continue
            record = json.loads(text)
            yield [record.get(col) for col in names]

    return RowStream(object_rows, resolved, name=name)


# --------------------------------------------------------------------------- #
# Streaming compact encoding / compression
# --------------------------------------------------------------------------- #


def encode_stream(
    stream: RowStream | Dataset,
    *,
    compress: bool = False,
    delimiter: str = ",",
    sink: IO[bytes] | None = None,
) -> bytes:
    """Render a stream to its compact, lossless encoding in one bounded pass.

    The schema header is written once and the rows follow, so the working set is
    a single row at a time rather than the whole table. With ``compress`` the
    output is gzip-compressed (the encoding is highly compressible — a column's
    values repeat). When a binary ``sink`` is given the bytes are streamed to it
    (the path for a dataset whose *encoding* is also larger than memory) and the
    return value is empty; otherwise the full encoding is returned.

    The header omits the row count (a stream does not know its length up front);
    :func:`~vincio.core.tabular.decode_table` reads rows to end-of-input, so the
    round-trip is exact."""
    rows = RowStream.from_dataset(stream) if isinstance(stream, Dataset) else stream
    options = tabular.EncodeOptions(delimiter=delimiter, include_count=False)
    header = tabular.encode_header(
        rows.column_names,
        types=[c.dtype.value for c in rows.columns],
        units=[c.unit for c in rows.columns],
        nullable=[c.nullable for c in rows.columns],
        name=rows.name,
        options=options,
    )
    width = rows.width

    buffer: io.BytesIO | None = None
    if sink is not None:
        target: IO[bytes] = sink
    else:
        buffer = io.BytesIO()
        target = buffer
    out = gzip.GzipFile(fileobj=target, mode="wb", mtime=0) if compress else target
    try:
        out.write(header.encode("utf-8"))
        for row in rows.rows():
            line = tabular.encode_row(list(row), width, options=options)
            out.write(b"\n")
            out.write(line.encode("utf-8"))
    finally:
        if compress:
            out.close()
    if buffer is not None:
        return buffer.getvalue()
    return b""


# --------------------------------------------------------------------------- #
# Bounded-memory streaming aggregation (out-of-core group-by)
# --------------------------------------------------------------------------- #


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass
class _MeasureAccumulator:
    """Single-pass numeric reducer for one (column, aggregations) within a group."""

    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    numeric_count: int = 0

    def add(self, value: Any) -> None:
        self.count += 1
        if _is_number(value):
            x = float(value)
            self.total += x
            self.numeric_count += 1
            self.minimum = x if self.minimum is None else min(self.minimum, x)
            self.maximum = x if self.maximum is None else max(self.maximum, x)

    def value(self, agg: str) -> Any:
        if self.numeric_count == 0:
            return None
        if agg == "sum":
            return round(self.total, 6)
        if agg == "mean":
            return round(self.total / self.numeric_count, 6)
        if agg == "min":
            return self.minimum
        if agg == "max":
            return self.maximum
        return None


class StreamAggregation(BaseModel):
    """The deterministic, bounded-memory result of a streaming group-by.

    :attr:`result` is a small :class:`~vincio.data.Dataset` — one row per group,
    with the group-by columns followed by each measure column (``revenue_sum``,
    ``revenue_mean``, …) and the group ``count``. Because the working set tracked
    only the distinct groups, :attr:`rows_processed` can be far larger than the
    table that fits in memory.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    result: Dataset
    group_by: list[str] = Field(default_factory=list)
    measure_columns: list[str] = Field(default_factory=list)
    rows_processed: int = 0
    group_count: int = 0
    bounded: bool = True
    max_groups: int = DEFAULT_MAX_GROUPS

    def to_dataset(self) -> Dataset:
        """The aggregated result as a dataset (encode it, profile it, or carry it
        as evidence like any other)."""
        return self.result

    def to_evidence_item(self, **kwargs: Any) -> EvidenceItem:
        """Project the aggregated result straight to table evidence the context
        compiler scores, budgets, orders, and cites."""
        return cast("EvidenceItem", self.result.to_evidence_item(**kwargs))

    def summary(self) -> str:
        """A one-line human summary of the aggregation."""
        return (
            f"{self.result.name or 'aggregation'}: {self.group_count:,} groups "
            f"over {self.rows_processed:,} rows by {', '.join(self.group_by)}"
        )


def stream_aggregate(
    data: RowStream | Dataset | list[dict[str, Any]],
    *,
    group_by: str | Sequence[str],
    measures: Mapping[str, str | Sequence[str]] | None = None,
    max_groups: int = DEFAULT_MAX_GROUPS,
) -> StreamAggregation:
    """Group a stream by one or more columns and reduce measures over each group
    in a single bounded-memory pass.

    ``measures`` maps a column to the aggregation(s) to compute over it —
    ``"sum"``, ``"mean"``, ``"min"``, or ``"max"`` (each group's row ``count`` is
    emitted unconditionally). The working set holds one accumulator per
    *distinct group*, never the rows, so a table far larger than memory
    aggregates inside a fixed footprint; a group cardinality beyond ``max_groups``
    raises :class:`~vincio.core.errors.StreamError` rather than growing without
    bound. Groups are emitted in a deterministic key order. Returns a
    :class:`StreamAggregation`."""
    stream = _coerce_stream(data)
    keys = [group_by] if isinstance(group_by, str) else list(group_by)
    name_to_index = {c.name: j for j, c in enumerate(stream.columns)}
    for key in keys:
        if key not in name_to_index:
            raise StreamError(f"no column named {key!r} to group by")
    key_indices = [name_to_index[k] for k in keys]

    spec = _normalize_measures(measures, name_to_index)
    # Output measure columns: the group keys, then "<col>_<agg>" per measure, then count.
    measure_columns = [f"{col}_{agg}" for col, agg, _ in spec]

    # Each group holds its row count and one accumulator per measure. The working
    # set is one entry per distinct group — never the rows themselves.
    counts: dict[tuple[Any, ...], int] = {}
    groups: dict[tuple[Any, ...], list[_MeasureAccumulator]] = {}
    rows_processed = 0
    for row in stream.rows():
        group_key = tuple(row[i] if i < len(row) else None for i in key_indices)
        accumulators = groups.get(group_key)
        if accumulators is None:
            if len(groups) >= max_groups:
                raise StreamError(
                    f"group cardinality exceeded the bound of {max_groups:,}; "
                    "group by a coarser key or raise max_groups"
                )
            accumulators = [_MeasureAccumulator() for _ in spec]
            groups[group_key] = accumulators
            counts[group_key] = 0
        counts[group_key] += 1
        for acc, (_, _, col_index) in zip(accumulators, spec, strict=True):
            acc.add(row[col_index] if col_index < len(row) else None)
        rows_processed += 1

    ordered_keys = sorted(groups, key=lambda k: tuple(str(v) for v in k))
    records: list[dict[str, Any]] = []
    for group_key in ordered_keys:
        accumulators = groups[group_key]
        record: dict[str, Any] = {keys[i]: group_key[i] for i in range(len(keys))}
        for (col, agg, _), acc in zip(spec, accumulators, strict=True):
            record[f"{col}_{agg}"] = acc.value(agg)
        record["count"] = counts[group_key]
        records.append(record)

    result_name = f"{stream.name}_by_{'_'.join(keys)}" if stream.name else "aggregation"
    result = Dataset.from_records(records, name=result_name) if records else _empty_aggregation(
        keys, measure_columns, result_name
    )
    return StreamAggregation(
        result=result,
        group_by=keys,
        measure_columns=measure_columns,
        rows_processed=rows_processed,
        group_count=len(groups),
        bounded=True,
        max_groups=max_groups,
    )


def _normalize_measures(
    measures: Mapping[str, str | Sequence[str]] | None,
    name_to_index: Mapping[str, int],
) -> list[tuple[str, str, int]]:
    """Normalize the measure spec into ``(column, aggregation, column_index)``
    triples, validating columns and aggregation names. May be empty — every group
    always carries its ``count`` regardless of whether any measure is declared."""
    spec: list[tuple[str, str, int]] = []
    for col, aggs in (measures or {}).items():
        if col not in name_to_index:
            raise StreamError(f"no column named {col!r} to aggregate")
        agg_list = [aggs] if isinstance(aggs, str) else list(aggs)
        for agg in agg_list:
            if agg not in _AGGREGATIONS:
                raise StreamError(
                    f"unknown aggregation {agg!r}; choose from {', '.join(_AGGREGATIONS)}"
                )
            spec.append((col, agg, name_to_index[col]))
    return spec


def _empty_aggregation(keys: list[str], measure_columns: list[str], name: str) -> Dataset:
    columns = [ColumnSchema(name=k) for k in keys]
    columns += [ColumnSchema(name=m, dtype=DataType.FLOAT, nullable=True) for m in measure_columns]
    columns.append(ColumnSchema(name="count", dtype=DataType.INT))
    return Dataset(name=name, columns=columns, cells=[[] for _ in columns])


def _coerce_stream(data: RowStream | Dataset | list[dict[str, Any]]) -> RowStream:
    if isinstance(data, RowStream):
        return data
    if isinstance(data, Dataset):
        return RowStream.from_dataset(data)
    if isinstance(data, list):
        return RowStream.from_records(data)
    raise StreamError(
        f"cannot stream {type(data).__name__}; pass a RowStream, a Dataset, or a list of records"
    )


# --------------------------------------------------------------------------- #
# Analytical pipelines at scale on the BatchRunner
# --------------------------------------------------------------------------- #


@dataclass
class BulkMapResult:
    """The outcome of a :func:`stream_map` run: one provider response per chunk,
    reconciled by chunk index, with the total (batch-discounted) cost."""

    chunk_count: int
    results: list[Any] = field(default_factory=list)  # list[BatchResult]
    cost_usd: float = 0.0

    @property
    def succeeded(self) -> list[Any]:
        return [r for r in self.results if getattr(r, "ok", False)]

    @property
    def failed(self) -> list[Any]:
        return [r for r in self.results if not getattr(r, "ok", False)]

    def by_chunk(self) -> dict[int, Any]:
        """The result for each chunk index (parsed from its ``chunk-<i>`` id)."""
        out: dict[int, Any] = {}
        for result in self.results:
            cid = getattr(result, "custom_id", "")
            if cid.startswith("chunk-"):
                try:
                    out[int(cid.split("-", 1)[1])] = result
                except ValueError:
                    continue
        return out


async def stream_map(
    stream: RowStream | Dataset | list[dict[str, Any]],
    build_request: Callable[[Dataset, int], ModelRequest],
    *,
    runner: BatchRunner | None = None,
    backend: BatchBackend | ModelProvider | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    timeout_s: float | None = None,
) -> BulkMapResult:
    """Run an analytical transform over a stream *at scale* by chunking it into
    the existing :class:`~vincio.providers.BatchRunner`.

    Each bounded chunk becomes one provider request via ``build_request(chunk,
    index)`` (typically a prompt over the chunk's compact encoding), the set is
    submitted to a provider Batch API (half-cost, bounded concurrency), and the
    responses are reconciled by chunk index — a missing or failed chunk surfaces
    as a failed result rather than being dropped. Pass an existing ``runner`` or
    a ``backend`` / provider to build one. Returns a :class:`BulkMapResult`."""
    from ..providers.batch import BatchRequest, BatchRunner

    if runner is None:
        if backend is None:
            raise StreamError("stream_map needs a runner= or a backend=/provider= to dispatch on")
        runner = BatchRunner(backend)

    stream = _coerce_stream(stream)
    requests: list[BatchRequest] = []
    for index, chunk in enumerate(stream.chunks(chunk_rows)):
        requests.append(BatchRequest(custom_id=f"chunk-{index}", request=build_request(chunk, index)))

    if not requests:
        return BulkMapResult(chunk_count=0)

    run = await runner.run(requests, timeout_s=timeout_s)
    return BulkMapResult(chunk_count=len(requests), results=list(run.results), cost_usd=run.cost_usd)
