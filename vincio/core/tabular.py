"""Compact, token-oriented table encoding kernel.

The canonical rendering of tabular data into a string that is *cheap in tokens*
and *lossless*: the schema, column types, units, and null-handling are declared
**once** in a one-line header and the cells follow as delimited rows, instead of
repeating the keys and structural punctuation on every row the way
``json.dumps`` does (and instead of the per-row separators a Markdown table
repeats). The encoding round-trips — :func:`decode_table` reconstructs the
columns, types, and cells from the bytes alone — so a table can be carried as
compact evidence without losing structure.

This is a low-level kernel: it depends only on :mod:`vincio.core.tokens` and the
standard library, so the typed :class:`~vincio.data.Dataset` container, the
:class:`~vincio.data.DataEncoder`, the evidence token accounting in
:mod:`vincio.core.types`, the context scorer, and the document parsers can all
share one encoding without importing each other. The user-facing surface lives
in :mod:`vincio.data`.

Format (default options)::

    sales{#3,id:int,region:str,revenue:float USD,units:int?}
    1,NA,1200.5,5
    2,EU,980.0,
    3,APAC,1500.25,8

* ``sales`` — the optional table name.
* ``{...}`` — the header, declared once. A leading ``#3`` token is the row count
  (so a reader can verify nothing was truncated); the rest is the schema —
  ``name:type`` per column, a trailing ``?`` marks a nullable column, and a
  space-separated token after the type is the column's unit (e.g. ``float USD``).
  The count is kept inside the braces (rather than a ``[3]`` prefix) so it can
  never be mistaken for an ``[E1]``-style citation marker.
* each following line is one row, comma-delimited, with RFC-4180 minimal
  quoting. A **null** cell is an empty field; an **empty string** is the quoted
  ``""`` — so the two are distinguishable and the round-trip is exact.

An optional ``# ...`` description line (one per encoding, emitted when
``exemplars`` is set) carries a few example values per column for the model;
:func:`decode_table` skips it, so it never affects the round-trip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

from .errors import DataError
from .tokens import count_tokens

__all__ = [
    "TABULAR_DTYPES",
    "EncodeOptions",
    "DecodedTable",
    "infer_dtype",
    "encode_table",
    "decode_table",
    "encode_records",
    "encode_value",
    "table_token_cost",
]

# The closed type vocabulary the header declares. ``null`` is the type of an
# all-null column (no value ever observed); every other name coerces on decode.
TABULAR_DTYPES: tuple[str, ...] = (
    "int",
    "float",
    "str",
    "bool",
    "date",
    "datetime",
    "time",
    "null",
)


@dataclass(frozen=True)
class EncodeOptions:
    """Deterministic knobs for :func:`encode_table` / :func:`encode_value`.

    Defaults render the lossless, schema-once form. ``max_rows`` truncates the
    body to a preview; the header's row count reflects the truncated body, so the
    encoding stays self-consistent (a caller wanting a representative sample
    should select rows before encoding — see :class:`~vincio.data.Dataset`).
    ``exemplars`` prepends a ``#`` description line carrying that many example
    values per column. :func:`decode_table` is self-describing: it reads the name,
    count, types, and units back from the bytes and ignores the structural
    ``include_*`` toggles — only ``delimiter`` must match on decode.
    """

    delimiter: str = ","
    include_name: bool = True
    include_count: bool = True
    include_types: bool = True
    include_units: bool = True
    exemplars: int = 0
    max_rows: int | None = None


@dataclass(frozen=True)
class DecodedTable:
    """The result of :func:`decode_table`: a self-describing table reconstructed
    from an encoding. ``rows`` holds the raw string cells (``None`` for a null);
    :meth:`typed_rows` coerces each cell to its declared column type."""

    name: str
    columns: list[str]
    types: list[str]
    units: list[str | None]
    nullable: list[bool]
    rows: list[list[str | None]]

    def typed_rows(self) -> list[list[Any]]:
        """The cells coerced to their declared column types (a string cell stays
        a string; an ``int`` column yields ``int``; a null stays ``None``)."""
        return [
            [_coerce(cell, self.types[j] if j < len(self.types) else "str") for j, cell in enumerate(row)]
            for row in self.rows
        ]


# --------------------------------------------------------------------------- #
# Value formatting / coercion
# --------------------------------------------------------------------------- #


def _format_value(value: Any) -> str | None:
    """Render a Python value to its canonical, round-trip-safe cell text, or
    ``None`` for a null cell (which the row codec writes as an empty field)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # repr() is the shortest string that round-trips a float exactly.
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return str(value)


def _coerce(text: str | None, dtype: str) -> Any:
    """Inverse of :func:`_format_value` for a declared *dtype*. A value that does
    not parse is kept as its string form rather than raising, so a hand-authored
    or lightly-malformed encoding still decodes."""
    if text is None:
        return None
    try:
        if dtype == "int":
            return int(text)
        if dtype == "float":
            return float(text)
        if dtype == "bool":
            return text == "true"
        if dtype == "datetime":
            return datetime.fromisoformat(text)
        if dtype == "date":
            return date.fromisoformat(text)
        if dtype == "time":
            return time.fromisoformat(text)
    except (ValueError, TypeError):
        return text
    return text


def infer_dtype(values: list[Any]) -> str:
    """Infer the column type of a sequence of Python values (nulls ignored). An
    all-null or empty column is ``"null"``; ``bool`` is kept distinct from
    ``int`` (a ``bool`` is an ``int`` subclass in Python)."""
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "null"
    if all(isinstance(v, bool) for v in non_null):
        return "bool"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in non_null):
        return "int"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
        return "float"
    if all(isinstance(v, datetime) for v in non_null):
        return "datetime"
    if all(isinstance(v, date) and not isinstance(v, datetime) for v in non_null):
        return "date"
    if all(isinstance(v, time) for v in non_null):
        return "time"
    return "str"


# --------------------------------------------------------------------------- #
# Field / token codec (RFC-4180 minimal quoting, null vs empty-string aware)
# --------------------------------------------------------------------------- #


def _escape_cell(text: str) -> str:
    """C-escape the line-breaking characters so every row stays on one line
    (newlines and carriage returns become ``\\n`` / ``\\r``); the backslash is
    escaped first so the transform is reversible."""
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def _unescape_cell(text: str) -> str:
    out: list[str] = []
    index = 0
    n = len(text)
    while index < n:
        char = text[index]
        if char == "\\" and index + 1 < n:
            nxt = text[index + 1]
            out.append({"\\": "\\", "n": "\n", "r": "\r"}.get(nxt, nxt))
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _needs_quote(text: str, delimiter: str) -> bool:
    return text == "" or delimiter in text or '"' in text or text != text.strip()


def _encode_field(text: str | None, delimiter: str) -> str:
    """A null is an empty field; an empty string is quoted (``""``) so it is
    distinguishable from a null; everything else is escaped (line breaks) and
    quoted only when it would otherwise be ambiguous."""
    if text is None:
        return ""
    escaped = _escape_cell(text)
    if _needs_quote(escaped, delimiter):
        return '"' + escaped.replace('"', '""') + '"'
    return escaped


def _read_token(text: str, index: int, stops: str) -> tuple[str, int, bool]:
    """Read one (optionally double-quoted) token from *text* starting at *index*,
    stopping at any character in *stops* when unquoted. Returns the decoded
    token, the new index, and whether the token was quoted (so a caller can tell
    a quoted empty string from a bare empty field)."""
    n = len(text)
    if index < n and text[index] == '"':
        index += 1
        buf: list[str] = []
        while index < n:
            char = text[index]
            if char == '"':
                if index + 1 < n and text[index + 1] == '"':
                    buf.append('"')
                    index += 2
                    continue
                index += 1
                break
            buf.append(char)
            index += 1
        return "".join(buf), index, True
    start = index
    while index < n and text[index] not in stops:
        index += 1
    return text[start:index], index, False


def _encode_row(cells: list[str | None], delimiter: str) -> str:
    return delimiter.join(_encode_field(cell, delimiter) for cell in cells)


def _decode_row(line: str, delimiter: str) -> list[str | None]:
    fields: list[str | None] = []
    index = 0
    n = len(line)
    while True:
        text, index, quoted = _read_token(line, index, delimiter)
        fields.append(None if (text == "" and not quoted) else _unescape_cell(text))
        if index < n and line[index] == delimiter:
            index += 1
            if index == n:  # a trailing delimiter is a final null field
                fields.append(None)
                break
            continue
        break
    return fields


# --------------------------------------------------------------------------- #
# Header (schema declared once)
# --------------------------------------------------------------------------- #

# Characters that force a header token (name / column / unit) to be quoted.
_HEADER_STOPS = ',:{}"? \t'


def _encode_header_token(text: str) -> str:
    """Render a header token (name / column / unit). Line breaks are escaped so
    the header stays one line; the token is quoted when it would otherwise be
    ambiguous, when it is empty, or when it begins with ``#`` (so a header line
    can never be mistaken for a ``#``-prefixed description line on decode)."""
    escaped = _escape_cell(text)
    if escaped == "" or escaped[:1] == "#" or any(c in escaped for c in _HEADER_STOPS):
        return '"' + escaped.replace('"', '""') + '"'
    return escaped


def _coldef(name: str, dtype: str | None, unit: str | None, nullable: bool, options: EncodeOptions) -> str:
    out = _encode_header_token(name)
    if options.include_types and dtype:
        spec = dtype
        if nullable:
            spec += "?"
        if options.include_units and unit:
            spec += " " + _encode_header_token(unit)
        out += ":" + spec
    return out


def _parse_coldef(text: str) -> tuple[str, str, str | None, bool]:
    """Parse one ``name[:type[?][ unit]]`` definition into
    ``(name, dtype, unit, nullable)``. ``dtype`` defaults to ``"str"``."""
    name, index, _ = _read_token(text, 0, ":")
    name = _unescape_cell(name)
    dtype = "str"
    unit: str | None = None
    nullable = False
    if index < len(text) and text[index] == ":":
        index += 1
        type_token, index, _ = _read_token(text, index, " ")
        if type_token.endswith("?"):
            nullable = True
            type_token = type_token[:-1]
        dtype = type_token or "str"
        if index < len(text) and text[index] == " ":
            unit_token, index, _ = _read_token(text, index + 1, "")
            unit = _unescape_cell(unit_token) or None
    return name, dtype, unit, nullable


def _split_coldefs(body: str) -> list[str]:
    """Split a ``{...}`` body into coldef spans on unquoted commas, respecting
    quoted names/units."""
    spans: list[str] = []
    index = 0
    n = len(body)
    start = 0
    in_quote = False
    while index < n:
        char = body[index]
        if char == '"':
            in_quote = not in_quote
            index += 1
            continue
        if char == "," and not in_quote:
            spans.append(body[start:index])
            index += 1
            start = index
            continue
        index += 1
    spans.append(body[start:index])
    if body == "":
        return []
    # A trailing unquoted delimiter yields a phantom empty span; drop it (a real
    # empty column name is the quoted ``""`` span, not the bare empty string).
    if len(spans) > 1 and spans[-1] == "":
        spans.pop()
    return spans


def _exemplar_line(columns: list[str], rows: list[list[Any]], count: int) -> str:
    parts: list[str] = []
    for j, col in enumerate(columns):
        seen: list[str] = []
        for row in rows:
            if j >= len(row):
                continue
            cell = row[j]
            if cell is None:
                continue
            # Escape line breaks so the description stays on one line (it is a
            # decode-skipped comment, so it is never parsed back).
            rendered = _escape_cell(_format_value(cell) or "")
            if rendered not in seen:
                seen.append(rendered)
            if len(seen) >= count:
                break
        if seen:
            parts.append(f"{_escape_cell(col)}=" + "|".join(seen))
    return "# " + "; ".join(parts)


# --------------------------------------------------------------------------- #
# Public encode / decode
# --------------------------------------------------------------------------- #


def encode_table(
    columns: list[str],
    rows: list[list[Any]],
    *,
    types: list[str] | None = None,
    units: list[str | None] | None = None,
    nullable: list[bool] | None = None,
    name: str = "",
    options: EncodeOptions | None = None,
) -> str:
    """Encode a table — column names, a parallel list of rows, and an optional
    parallel schema — into the compact, lossless, token-oriented form.

    ``types`` are drawn from :data:`TABULAR_DTYPES`; ``units`` and ``nullable``
    are parallel to ``columns``. When ``types`` is omitted the cell types are
    inferred from the data. The header (name, count, and schema) is emitted once;
    the rows follow. A row shorter than ``columns`` is padded with nulls; a row
    with *more* cells than ``columns`` raises :class:`~vincio.core.errors.DataError`
    (the extra cells could not be encoded losslessly)."""
    opt = options or EncodeOptions()
    width = len(columns)
    body = rows if opt.max_rows is None else rows[: opt.max_rows]
    widest = max((len(row) for row in body), default=0)
    if widest > width:
        raise DataError(f"schema declares {width} columns but a row has {widest} values")
    resolved_types = list(types) if types is not None else [
        infer_dtype([row[j] if j < len(row) else None for row in body]) for j in range(width)
    ]
    coldefs = []
    for j, col in enumerate(columns):
        dtype = resolved_types[j] if j < len(resolved_types) else "str"
        unit = units[j] if (units is not None and j < len(units)) else None
        is_null = nullable[j] if (nullable is not None and j < len(nullable)) else any(
            (row[j] if j < len(row) else None) is None for row in body
        )
        coldefs.append(_coldef(col, dtype, unit, is_null, opt))

    head = ""
    if opt.include_name and name:
        head += _encode_header_token(name)
    inner = [f"#{len(body)}", *coldefs] if opt.include_count else coldefs
    head += "{" + ",".join(inner) + "}"

    lines: list[str] = []
    if opt.exemplars > 0:
        lines.append(_exemplar_line(columns, body, opt.exemplars))
    lines.append(head)
    for row in body:
        cells = [_format_value(row[j] if j < len(row) else None) for j in range(width)]
        lines.append(_encode_row(cells, opt.delimiter))
    return "\n".join(lines)


def decode_table(text: str, *, options: EncodeOptions | None = None) -> DecodedTable:
    """Reconstruct a table from an :func:`encode_table` string. Any leading
    ``#`` description lines are skipped; the header is parsed for the name and
    schema, and each remaining line is one row."""
    opt = options or EncodeOptions()
    raw_lines = text.split("\n")
    cursor = 0
    while cursor < len(raw_lines) and raw_lines[cursor].startswith("#"):
        cursor += 1
    if cursor >= len(raw_lines):
        return DecodedTable("", [], [], [], [], [])
    header = raw_lines[cursor]
    cursor += 1

    name = ""
    index = 0
    if header and header[0] != "{":
        name, index, _ = _read_token(header, 0, "{")
        name = _unescape_cell(name)
    brace_open = header.find("{", index)
    brace_close = header.rfind("}")
    body = header[brace_open + 1 : brace_close] if (brace_open != -1 and brace_close != -1) else ""

    columns: list[str] = []
    types: list[str] = []
    units: list[str | None] = []
    nullable: list[bool] = []
    count: int | None = None
    spans = _split_coldefs(body)
    if spans:
        match = re.fullmatch(r"#(\d+)", spans[0])
        if match is not None:
            count = int(match.group(1))
            spans = spans[1:]
    for span in spans:
        col, dtype, unit, is_null = _parse_coldef(span)
        columns.append(col)
        types.append(dtype)
        units.append(unit)
        nullable.append(is_null)

    n_rows = count if count is not None else len(raw_lines) - cursor
    rows: list[list[str | None]] = []
    for line in raw_lines[cursor : cursor + n_rows]:
        rows.append(_decode_row(line, opt.delimiter))
    return DecodedTable(name, columns, types, units, nullable, rows)


def encode_records(
    records: list[dict[str, Any]], *, name: str = "", options: EncodeOptions | None = None
) -> str:
    """Encode a list of record mappings as a table, taking the column set from
    the union of keys (first-seen order) and inferring each column's type."""
    columns: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            text = str(key)
            if text not in seen:
                seen.add(text)
                columns.append(text)
    rows: list[list[Any]] = [[record.get(col) for col in columns] for record in records]
    return encode_table(columns, rows, name=name, options=options)


def encode_value(obj: Any, *, options: EncodeOptions | None = None, _indent: int = 0) -> str:
    """Render an arbitrary JSON-like value into a compact, token-oriented form,
    the token-efficient replacement for ``json.dumps(indent=2)``.

    A list of record mappings becomes a :func:`encode_table` block; a mapping
    becomes ``key: value`` lines; a list of scalars becomes an inline
    ``[a, b, c]``; scalars render directly. Every leaf value is preserved."""
    opt = options or EncodeOptions()
    pad = "  " * _indent
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, float, str)):
        rendered = _format_value(obj)
        return rendered if rendered is not None else ""
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines: list[str] = []
        for key, value in obj.items():
            if _is_inline(value):
                lines.append(f"{pad}{key}: {encode_value(value, options=opt, _indent=_indent + 1)}")
            else:
                lines.append(f"{pad}{key}:")
                lines.append(encode_value(value, options=opt, _indent=_indent + 1))
        return "\n".join(lines)
    if isinstance(obj, (list, tuple)):
        items = list(obj)
        if not items:
            return "[]"
        if all(isinstance(item, dict) for item in items):
            return encode_records(items, options=opt)
        if all(not isinstance(item, (dict, list, tuple)) for item in items):
            return "[" + ", ".join((_format_value(item) or "") for item in items) + "]"
        return "\n".join(
            f"{pad}- " + encode_value(item, options=opt, _indent=_indent + 1).lstrip()
            for item in items
        )
    rendered = _format_value(obj)
    return rendered if rendered is not None else ""


def _is_inline(obj: Any) -> bool:
    """A value renders inline (on the same line as its key) when it is a scalar
    or a flat list of scalars; dicts and nested lists render as an indented
    block on the following lines."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return True
    if isinstance(obj, (list, tuple)):
        return all(not isinstance(item, (dict, list, tuple)) for item in obj)
    return False


def table_token_cost(
    columns: list[str],
    rows: list[list[Any]],
    *,
    types: list[str] | None = None,
    units: list[str | None] | None = None,
    nullable: list[bool] | None = None,
    name: str = "",
    model: str | None = None,
    options: EncodeOptions | None = None,
) -> int:
    """The exact token cost of a table: the token count of its compact encoding.

    This is the columnar-accurate replacement for the ``3 * cells`` heuristic —
    it counts the tokens the model will actually receive (header declared once,
    rows once), rather than a flat per-cell estimate."""
    encoded = encode_table(
        columns, rows, types=types, units=units, nullable=nullable, name=name, options=options
    )
    return count_tokens(encoded, model)
