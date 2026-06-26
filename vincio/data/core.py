"""First-class tabular data: a typed schema and columnar cells.

A :class:`Dataset` is schema-bearing, columnar evidence — never a row-flattened
``Document``. It carries a typed :class:`DataSchema` (per-column name, type, unit,
and nullability) and stores its cells **column-major**, so the schema is declared
once and the values are the model's payload. It renders to the compact,
token-oriented, lossless encoding of :mod:`vincio.core.tabular` and projects to
first-class context evidence via :class:`~vincio.data.TableEvidence`.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core import tabular
from ..core.errors import DataError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..documents.parsers import TableData
    from .encoders import DataEncoder
    from .evidence import TableEvidence

__all__ = [
    "DataType",
    "ColumnSchema",
    "DataSchema",
    "Dataset",
]


class DataType(StrEnum):
    """The closed column-type vocabulary a dataset's schema declares. Every type
    but ``NULL`` (an all-null column) round-trips exactly through the encoder."""

    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    NULL = "null"


class ColumnSchema(BaseModel):
    """One column's typed declaration: its name, data type, optional unit
    (e.g. ``USD``, ``ms``), whether it admits nulls, and an optional
    description — injected into the encoding header once, not per row."""

    name: str
    dtype: DataType = DataType.STR
    unit: str | None = None
    nullable: bool = False
    description: str = ""


class DataSchema(BaseModel):
    """The ordered list of column declarations for a dataset."""

    columns: list[ColumnSchema] = Field(default_factory=list)

    @classmethod
    def from_names(cls, names: list[str]) -> DataSchema:
        """A string-typed schema from bare column names (types default to
        ``str``; refine with :meth:`infer` or by passing typed columns)."""
        return cls(columns=[ColumnSchema(name=n) for n in names])

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def dtypes(self) -> list[str]:
        return [c.dtype.value for c in self.columns]

    @property
    def units(self) -> list[str | None]:
        return [c.unit for c in self.columns]

    @property
    def nullable(self) -> list[bool]:
        return [c.nullable for c in self.columns]

    def __len__(self) -> int:
        return len(self.columns)

    def iter_columns(self) -> Iterator[ColumnSchema]:
        """Iterate the column declarations."""
        return iter(self.columns)


def _coerce_schema(
    schema: DataSchema | list[ColumnSchema] | list[str] | None,
    *,
    names: list[str],
    cells: list[list[Any]],
) -> DataSchema:
    """Resolve a caller-supplied schema (a :class:`DataSchema`, a list of
    :class:`ColumnSchema`, a list of names, or ``None`` for full inference) into
    a concrete :class:`DataSchema` aligned to *names*/*cells*."""
    if isinstance(schema, DataSchema):
        resolved = schema
    elif schema is not None and schema and isinstance(schema[0], ColumnSchema):
        resolved = DataSchema(columns=[c for c in schema if isinstance(c, ColumnSchema)])
    elif schema is not None and schema and isinstance(schema[0], str):
        resolved = DataSchema(columns=[ColumnSchema(name=str(n)) for n in schema])
    else:
        resolved = DataSchema()
    if not resolved.columns:
        resolved = DataSchema(
            columns=[
                ColumnSchema(
                    name=names[j],
                    dtype=DataType(tabular.infer_dtype(cells[j])),
                    nullable=any(v is None for v in cells[j]),
                )
                for j in range(len(names))
            ]
        )
    if len(resolved.columns) != len(names):
        raise DataError(
            f"schema declares {len(resolved.columns)} columns but the data has {len(names)}"
        )
    return resolved


class Dataset(BaseModel):
    """Schema-bearing, columnar tabular data — the first-class evidence the data
    plane is built on.

    The schema is declared once (``columns``); the cells are stored column-major
    (``cells[j]`` is column *j*'s values). Build one from rows, records, columns,
    a legacy :class:`~vincio.documents.parsers.TableData`, or a compact encoding,
    then render it with :meth:`encode` or carry it as context evidence with
    :meth:`to_evidence`.
    """

    name: str = ""
    columns: list[ColumnSchema] = Field(default_factory=list)
    cells: list[list[Any]] = Field(default_factory=list)
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_rows(
        cls,
        rows: list[list[Any]],
        schema: DataSchema | list[ColumnSchema] | list[str],
        *,
        name: str = "",
        source: str = "",
    ) -> Dataset:
        """Build from row-major data and a schema (typed columns, or bare names
        with types inferred from the rows)."""
        names = (
            schema.names
            if isinstance(schema, DataSchema)
            else [c.name if isinstance(c, ColumnSchema) else str(c) for c in schema]
        )
        width = len(names)
        widest = max((len(row) for row in rows), default=0)
        if widest > width:
            raise DataError(
                f"schema declares {width} columns but a row has {widest} values"
            )
        cells: list[list[Any]] = [[row[j] if j < len(row) else None for row in rows] for j in range(width)]
        resolved = _coerce_schema(schema, names=names, cells=cells)
        return cls(name=name, columns=resolved.columns, cells=cells, source=source)

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]],
        *,
        schema: DataSchema | list[ColumnSchema] | list[str] | None = None,
        name: str = "",
        source: str = "",
    ) -> Dataset:
        """Build from a list of mappings. The columns are the union of keys in
        first-seen order; types are inferred unless a schema is given."""
        if schema is not None:
            names = (
                schema.names
                if isinstance(schema, DataSchema)
                else [c.name if isinstance(c, ColumnSchema) else str(c) for c in schema]
            )
        else:
            names = []
            seen: set[str] = set()
            for record in records:
                for key in record:
                    text = str(key)
                    if text not in seen:
                        seen.add(text)
                        names.append(text)
        cells = [[record.get(col) for record in records] for col in names]
        resolved = _coerce_schema(schema, names=names, cells=cells)
        return cls(name=name, columns=resolved.columns, cells=cells, source=source)

    @classmethod
    def from_columns(
        cls,
        columns: dict[str, list[Any]],
        *,
        schema: DataSchema | list[ColumnSchema] | None = None,
        name: str = "",
        source: str = "",
    ) -> Dataset:
        """Build directly from column-major data (``{name: values}``)."""
        names = list(columns)
        cells = [list(columns[name]) for name in names]
        resolved = _coerce_schema(schema, names=names, cells=cells)
        return cls(name=name, columns=resolved.columns, cells=cells, source=source)

    @classmethod
    def from_table_data(cls, table: TableData, *, name: str = "") -> Dataset:
        """Bridge a legacy string-celled :class:`~vincio.documents.parsers.TableData`
        into a typed dataset, coercing numeric and boolean columns to their
        Python values where unambiguous (so the encoding is typed) and keeping
        everything else as text (so it stays lossless)."""
        names = list(table.columns)
        width = len(names)
        raw: list[list[Any]] = [
            [row[j] if j < len(row) else None for row in table.rows] for j in range(width)
        ]
        schema_cols: list[ColumnSchema] = []
        cells: list[list[Any]] = []
        for j, col in enumerate(names):
            inferred = table.inferred_schema.get(col, "string")
            values, dtype = _coerce_string_column([("" if v is None else str(v)) for v in raw[j]], inferred)
            cells.append(values)
            schema_cols.append(
                ColumnSchema(
                    name=col,
                    dtype=dtype,
                    unit=table.units.get(col),
                    nullable=any(v is None for v in values),
                )
            )
        return cls(
            name=name or table.title,
            columns=schema_cols,
            cells=cells,
            source=table.source,
        )

    @classmethod
    def from_encoding(cls, text: str, *, name: str = "") -> Dataset:
        """Reconstruct a dataset from a compact encoding (the inverse of
        :meth:`encode`)."""
        decoded = tabular.decode_table(text)
        width = len(decoded.columns)
        typed = decoded.typed_rows()
        cells = [[row[j] if j < len(row) else None for row in typed] for j in range(width)]
        columns = [
            ColumnSchema(
                name=decoded.columns[j],
                dtype=DataType(decoded.types[j]) if decoded.types[j] in DataType._value2member_map_ else DataType.STR,
                unit=decoded.units[j] if j < len(decoded.units) else None,
                nullable=decoded.nullable[j] if j < len(decoded.nullable) else False,
            )
            for j in range(width)
        ]
        return cls(name=name or decoded.name, columns=columns, cells=cells)

    # -- access ----------------------------------------------------------------

    @property
    def data_schema(self) -> DataSchema:
        """The dataset's typed :class:`DataSchema` (named ``data_schema`` to avoid
        Pydantic's reserved ``schema`` attribute)."""
        return DataSchema(columns=self.columns)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def dtypes(self) -> list[str]:
        return [c.dtype.value for c in self.columns]

    @property
    def units(self) -> list[str | None]:
        return [c.unit for c in self.columns]

    @property
    def width(self) -> int:
        """The number of columns."""
        return len(self.columns)

    @property
    def row_count(self) -> int:
        """The number of rows."""
        return len(self.cells[0]) if self.cells else 0

    def column(self, name: str) -> list[Any]:
        """The values of the named column."""
        try:
            index = self.column_names.index(name)
        except ValueError as exc:
            raise DataError(f"no column named {name!r}") from exc
        return list(self.cells[index])

    def rows(self) -> list[list[Any]]:
        """The cells as row-major lists."""
        return [[self.cells[j][i] for j in range(self.width)] for i in range(self.row_count)]

    def records(self) -> list[dict[str, Any]]:
        """The rows as mappings keyed by column name."""
        names = self.column_names
        return [{names[j]: self.cells[j][i] for j in range(self.width)} for i in range(self.row_count)]

    def head(self, n: int) -> Dataset:
        """A new dataset with only the first *n* rows (schema preserved)."""
        return Dataset(
            name=self.name,
            columns=list(self.columns),
            cells=[col[:n] for col in self.cells],
            source=self.source,
            metadata=dict(self.metadata),
        )

    def exemplars(self, k: int = 2) -> dict[str, list[Any]]:
        """Up to *k* distinct non-null example values per column."""
        out: dict[str, list[Any]] = {}
        for j, col in enumerate(self.columns):
            seen: list[Any] = []
            for value in self.cells[j]:
                if value is None or value in seen:
                    continue
                seen.append(value)
                if len(seen) >= k:
                    break
            out[col.name] = seen
        return out

    # -- encoding / bridges ----------------------------------------------------

    def encode(self, encoder: DataEncoder | None = None, *, options: tabular.EncodeOptions | None = None) -> str:
        """Render the dataset to its compact, token-oriented encoding."""
        if encoder is not None:
            return encoder.encode(self)
        return tabular.encode_table(
            self.column_names,
            self.rows(),
            types=self.dtypes,
            units=self.units,
            nullable=[c.nullable for c in self.columns],
            name=self.name,
            options=options,
        )

    def token_cost(self, *, model: str | None = None, options: tabular.EncodeOptions | None = None) -> int:
        """The exact token cost of the dataset's encoding — the columnar-accurate
        replacement for a per-cell heuristic."""
        return tabular.table_token_cost(
            self.column_names,
            self.rows(),
            types=self.dtypes,
            units=self.units,
            nullable=[c.nullable for c in self.columns],
            name=self.name,
            model=model,
            options=options,
        )

    def to_table_data(self) -> TableData:
        """Project back to a legacy string-celled
        :class:`~vincio.documents.parsers.TableData` (for the chunking / loader
        paths that consume it)."""
        from ..documents.parsers import TableData

        rows = [[("" if v is None else str(v)) for v in row] for row in self.rows()]
        return TableData(
            id="T1",
            title=self.name,
            columns=self.column_names,
            rows=rows,
            source=self.source,
            units={c.name: c.unit for c in self.columns if c.unit},
            inferred_schema={c.name: c.dtype.value for c in self.columns},
        )

    def to_evidence(
        self,
        *,
        source_id: str = "",
        caption: str = "",
        encoder: DataEncoder | None = None,
        **kwargs: Any,
    ) -> TableEvidence:
        """Wrap the dataset as first-class :class:`~vincio.data.TableEvidence` the
        context compiler scores, budgets, orders, and cites."""
        from .evidence import TableEvidence

        return TableEvidence(
            dataset=self,
            source_id=source_id or self.name or "dataset",
            caption=caption,
            encoder=encoder,
            **kwargs,
        )

    def to_evidence_item(self, **kwargs: Any) -> Any:
        """Project straight to a ``modality="table"``
        :class:`~vincio.core.types.EvidenceItem` (so a bare dataset can be passed
        into the compiler's evidence list)."""
        return self.to_evidence(**kwargs).to_evidence_item()


def _coerce_string_column(values: list[str], inferred: str) -> tuple[list[Any], DataType]:
    """Coerce a string column to typed Python values **only when the coercion
    round-trips the exact string** — so a clean ``"5"`` becomes ``5`` while a
    leading-zero id (``"01234"``), a thousands-separated number, or a
    trailing-zero decimal (``"980.00"``) stays text and is preserved losslessly.
    An empty string in a coerced column becomes a null."""
    cleaned = [v for v in values if v != ""]
    if inferred == "number" and cleaned:
        if all(_int_round_trips(v) for v in cleaned):
            return ([None if v == "" else int(v) for v in values], DataType.INT)
        if all(_float_round_trips(v) for v in cleaned):
            return ([None if v == "" else float(v) for v in values], DataType.FLOAT)
    if inferred == "boolean" and cleaned and all(v.strip() in ("true", "false") for v in cleaned):
        return ([None if v == "" else v.strip() == "true" for v in values], DataType.BOOL)
    return (list(values), DataType.STR)


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
