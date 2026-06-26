"""Document structuring: sections, tables, spreadsheets,
code symbols. Operates on raw text/bytes produced by the loaders."""

from __future__ import annotations

import ast as python_ast
import csv
import io
import re
from html.parser import HTMLParser
from typing import Any

from pydantic import BaseModel, Field

from ..core.tabular import encode_table, encode_value

# Map the parser's inferred column types onto the compact encoder's vocabulary.
# Numbers are refined to int / float per column when rendered; everything else
# carries through as a header hint while the string cells stay exact (lossless).
_DTYPE_HINT = {
    "currency": "str",
    "date": "date",
    "boolean": "bool",
    "string": "str",
    "empty": "null",
}

__all__ = [
    "Section",
    "TableData",
    "extract_markdown_sections",
    "extract_markdown_tables",
    "parse_csv_table",
    "infer_table_schema",
    "table_quality_checks",
    "extract_code_symbols",
    "strip_html",
    "parse_html",
    "structure_data",
]


class Section(BaseModel):
    title: str
    level: int
    path: list[str]
    text: str
    start_line: int


class TableData(BaseModel):
    id: str = ""
    title: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    source: str = ""
    footnotes: list[str] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    inferred_schema: dict[str, str] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)

    def to_text(self) -> str:
        """Render in the compact, token-oriented encoding: the schema, types, and
        units are declared once in the header and the cells follow as delimited
        rows, instead of repeating the ``|`` separators on every line. The string
        cells are preserved exactly (lossless); footnotes follow as ``#`` notes.
        An empty table renders as the empty string, so an empty document or chunk
        stays empty."""
        if not self.columns and not self.rows:
            return "\n".join(f"# note: {note}" for note in self.footnotes)
        types = self._dtype_hints()
        units = [self.units.get(col) for col in self.columns]
        text = encode_table(
            self.columns,
            self.rows,
            types=types,
            units=units,
            name=self.title,
        )
        if self.footnotes:
            text += "\n" + "\n".join(f"# note: {note}" for note in self.footnotes)
        return text

    def _dtype_hints(self) -> list[str]:
        """The per-column type hints for the encoding header, refining a numeric
        column to ``int`` or ``float`` while keeping the string cells intact."""
        hints: list[str] = []
        for index, col in enumerate(self.columns):
            inferred = self.inferred_schema.get(col, "string")
            if inferred == "number":
                values = [
                    row[index]
                    for row in self.rows
                    if index < len(row) and str(row[index]).strip()
                ]
                hints.append("int" if all("." not in str(v) for v in values) else "float")
            else:
                hints.append(_DTYPE_HINT.get(inferred, "str"))
        return hints


# -- markdown ------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def extract_markdown_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    current_title, current_level, current_start = "", 0, 0
    current_lines: list[str] = []
    stack: list[tuple[int, str]] = []

    def flush() -> None:
        if current_title or current_lines:
            path = [t for _, t in stack]
            if current_title and (not path or path[-1] != current_title):
                path = path + [current_title]
            sections.append(
                Section(
                    title=current_title,
                    level=current_level,
                    path=path,
                    text="\n".join(current_lines).strip(),
                    start_line=current_start,
                )
            )

    for line_number, line in enumerate(text.splitlines()):
        match = _HEADING_RE.match(line)
        if match:
            flush()
            current_level = len(match.group(1))
            current_title = match.group(2).strip()
            current_lines = []
            current_start = line_number
            while stack and stack[-1][0] >= current_level:
                stack.pop()
            stack.append((current_level, current_title))
        else:
            current_lines.append(line)
    flush()
    return [s for s in sections if s.text or s.title]


_MD_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_MD_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def extract_markdown_tables(text: str) -> list[TableData]:
    tables: list[TableData] = []
    lines = text.splitlines()
    index = 0
    table_counter = 0
    while index < len(lines):
        match = _MD_TABLE_ROW_RE.match(lines[index])
        if match and index + 1 < len(lines) and _MD_TABLE_SEP_RE.match(lines[index + 1]):
            columns = [cell.strip() for cell in match.group(1).split("|")]
            rows: list[list[str]] = []
            cursor = index + 2
            while cursor < len(lines):
                row_match = _MD_TABLE_ROW_RE.match(lines[cursor])
                if not row_match:
                    break
                rows.append([cell.strip() for cell in row_match.group(1).split("|")])
                cursor += 1
            table_counter += 1
            table = TableData(id=f"T{table_counter}", columns=columns, rows=rows)
            table.inferred_schema = infer_table_schema(table)
            table.quality = table_quality_checks(table)
            tables.append(table)
            index = cursor
        else:
            index += 1
    return tables


# -- CSV / spreadsheets -----------------------------------------------------------

def parse_csv_table(content: str, *, title: str = "", delimiter: str | None = None) -> TableData:
    if delimiter is None:
        try:
            dialect = csv.Sniffer().sniff(content[:4096], delimiters=",;\t|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
    reader = csv.reader(io.StringIO(content), delimiter=delimiter)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return TableData(title=title)
    table = TableData(
        id="T1",
        title=title,
        columns=[c.strip() for c in rows[0]],
        rows=[[c.strip() for c in row] for row in rows[1:]],
    )
    table.inferred_schema = infer_table_schema(table)
    table.quality = table_quality_checks(table)
    return table


_NUMERIC_RE = re.compile(r"^-?[\d,]+(?:\.\d+)?%?$")
_DATE_RE = re.compile(r"^(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}[/.]\d{2,4})$")
_CURRENCY_RE = re.compile(r"^[$€£¥]\s?-?[\d,]+(?:\.\d+)?$|^-?[\d,]+(?:\.\d+)?\s?(?:USD|EUR|GBP)$")


def infer_table_schema(table: TableData) -> dict[str, str]:
    """Infer per-column types (inferred schema)."""
    schema: dict[str, str] = {}
    for column_index, column in enumerate(table.columns):
        values = [
            row[column_index].strip()
            for row in table.rows
            if column_index < len(row) and row[column_index].strip()
        ]
        if not values:
            schema[column] = "empty"
            continue
        if all(_CURRENCY_RE.match(v) for v in values):
            schema[column] = "currency"
        elif all(_NUMERIC_RE.match(v) for v in values):
            schema[column] = "number"
        elif all(_DATE_RE.match(v) for v in values):
            schema[column] = "date"
        elif set(v.lower() for v in values) <= {"true", "false", "yes", "no", "y", "n"}:
            schema[column] = "boolean"
        else:
            schema[column] = "string"
    return schema


def table_quality_checks(table: TableData) -> dict[str, Any]:
    """Data quality checks."""
    expected_width = len(table.columns)
    ragged = sum(1 for row in table.rows if len(row) != expected_width)
    empty_cells = sum(
        1 for row in table.rows for cell in row[:expected_width] if not str(cell).strip()
    )
    total_cells = max(1, len(table.rows) * expected_width)
    duplicate_rows = len(table.rows) - len({tuple(row) for row in table.rows}) if table.rows else 0
    return {
        "row_count": len(table.rows),
        "column_count": expected_width,
        "ragged_rows": ragged,
        "empty_cell_ratio": round(empty_cells / total_cells, 4),
        "duplicate_rows": duplicate_rows,
    }


# -- HTML ---------------------------------------------------------------------------

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_MAP = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&nbsp;": " "}


def strip_html(html: str) -> str:
    text = _SCRIPT_STYLE_RE.sub(" ", html)
    # Preserve block structure as newlines.
    text = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|br)>|<br\s*/?>", "\n", text)
    text = _TAG_RE.sub(" ", text)
    for entity, char in _ENTITY_MAP.items():
        text = text.replace(entity, char)
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


class _HTMLStructureParser(HTMLParser):
    """Extract title, heading-delimited sections, and tables from HTML.

    Dependency-free (stdlib :mod:`html.parser`): a real structural path so HTML
    charts and tables become sections/:class:`TableData` instead of opaque text.
    """

    _SKIP = {"script", "style", "noscript"}
    _HEADINGS = {f"h{i}" for i in range(1, 7)}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.sections: list[Section] = []
        self.tables: list[TableData] = []
        self._buffer: list[str] = []
        self._cur_title = ""
        self._cur_level = 0
        self._stack: list[tuple[int, str]] = []
        self._skip_depth = 0
        self._in_title = False
        # table state
        self._in_table = 0
        self._cur_table_rows: list[list[str]] = []
        self._cur_row: list[str] | None = None
        self._cell: list[str] | None = None

    def _flush_section(self) -> None:
        text = " ".join(" ".join(self._buffer).split()).strip()
        if self._cur_title or text:
            path = [t for _, t in self._stack]
            if self._cur_title and (not path or path[-1] != self._cur_title):
                path = path + [self._cur_title]
            self.sections.append(
                Section(title=self._cur_title, level=self._cur_level, path=path, text=text, start_line=0)
            )
        self._buffer = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag in self._HEADINGS:
            self._flush_section()
            self._cur_level = int(tag[1])
            self._cur_title = ""
            while self._stack and self._stack[-1][0] >= self._cur_level:
                self._stack.pop()
        elif tag == "table":
            self._in_table += 1
            self._cur_table_rows = []
        elif tag == "tr" and self._in_table:
            self._cur_row = []
        elif tag in ("td", "th") and self._in_table:
            self._cell = []
        elif tag in ("br", "p", "div", "li"):
            self._buffer.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag in self._HEADINGS:
            self._cur_title = " ".join(self._cur_title.split()).strip()
            if self._cur_title:
                self._stack.append((self._cur_level, self._cur_title))
        elif tag == "table" and self._in_table:
            self._in_table -= 1
            rows = [r for r in self._cur_table_rows if any(c.strip() for c in r)]
            if rows:
                table = TableData(id=f"T{len(self.tables) + 1}", columns=rows[0], rows=rows[1:])
                table.inferred_schema = infer_table_schema(table)
                table.quality = table_quality_checks(table)
                self.tables.append(table)
        elif tag == "tr" and self._in_table and self._cur_row is not None:
            self._cur_table_rows.append(self._cur_row)
            self._cur_row = None
        elif tag in ("td", "th") and self._in_table and self._cell is not None:
            if self._cur_row is not None:
                self._cur_row.append(" ".join("".join(self._cell).split()).strip())
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
        elif self._cell is not None:
            self._cell.append(data)
        elif self._in_table:
            return  # text between cells is layout noise
        else:
            if self._cur_level and not self._cur_title.strip():
                # data right after a heading start belongs to the heading until
                # its end tag; collect both heading text and body.
                self._cur_title += data
            self._buffer.append(data)

    def close(self) -> None:  # type: ignore[override]
        super().close()
        self._flush_section()


def parse_html(html: str) -> tuple[str, str, list[Section], list[TableData]]:
    """Parse HTML into ``(title, text, sections, tables)`` — a real structural
    path, dependency-free. ``text`` is the reading-order plain text."""
    parser = _HTMLStructureParser()
    parser.feed(html)
    parser.close()
    title = " ".join(parser.title.split()).strip()
    text = strip_html(html)
    return title, text, parser.sections, parser.tables


def _flatten_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def structure_data(obj: Any, *, title: str = "") -> tuple[str, list[Section], list[TableData]]:
    """Structure parsed JSON/JSONL/YAML into sections and tables.

    A list of flat objects becomes a :class:`TableData`; a mapping becomes one
    section per top-level key (scalars inline, nested values compactly encoded
    token-oriented rather than ``json.dumps``-dumped); anything else falls back
    to a single text section. Returns ``(text, sections, tables)``.
    """
    sections: list[Section] = []
    tables: list[TableData] = []

    if isinstance(obj, list) and obj and all(isinstance(row, dict) for row in obj):
        columns = list(dict.fromkeys(k for row in obj for k in row))
        rows = [[_flatten_scalar(row.get(col)) for col in columns] for row in obj]
        table = TableData(id="T1", title=title, columns=columns, rows=rows)
        table.inferred_schema = infer_table_schema(table)
        table.quality = table_quality_checks(table)
        tables.append(table)
        return table.to_text(), sections, tables

    if isinstance(obj, dict):
        text_parts: list[str] = []
        for key, value in obj.items():
            if isinstance(value, list) and value and all(isinstance(r, dict) for r in value):
                cols = list(dict.fromkeys(k for r in value for k in r))
                rows = [[_flatten_scalar(r.get(c)) for c in cols] for r in value]
                table = TableData(id=f"T{len(tables) + 1}", title=str(key), columns=cols, rows=rows)
                table.inferred_schema = infer_table_schema(table)
                table.quality = table_quality_checks(table)
                tables.append(table)
                body = table.to_text()
            elif isinstance(value, (dict, list)):
                body = encode_value(value)
            else:
                body = _flatten_scalar(value)
            sections.append(Section(title=str(key), level=1, path=[str(key)], text=body, start_line=0))
            text_parts.append(f"{key}: {body}")
        return "\n\n".join(text_parts), sections, tables

    body = encode_value(obj)
    return body, sections, tables


# -- code ------------------------------------------------------------------

class CodeSymbol(BaseModel):
    name: str
    kind: str  # function | class | method | import
    signature: str = ""
    line: int = 0
    docstring: str | None = None


_GENERIC_FUNC_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|def\s+(\w+)|fn\s+(\w+)|func\s+(?:\([^)]*\)\s*)?(\w+)|(?:public|private|protected|static|\s)*\w[\w<>\[\]]*\s+(\w+)\s*\()",
)
_GENERIC_CLASS_RE = re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?(?:class|struct|interface|trait|impl)\s+(\w+)")


def extract_code_symbols(source: str, *, language: str = "python") -> list[CodeSymbol]:
    """Extract symbols: Python via stdlib AST, other languages via patterns."""
    symbols: list[CodeSymbol] = []
    if language == "python":
        try:
            tree = python_ast.parse(source)
        except SyntaxError:
            return _extract_symbols_generic(source)
        for node in python_ast.walk(tree):
            if isinstance(node, (python_ast.FunctionDef, python_ast.AsyncFunctionDef)):
                args = ", ".join(a.arg for a in node.args.args)
                symbols.append(
                    CodeSymbol(
                        name=node.name,
                        kind="function",
                        signature=f"def {node.name}({args})",
                        line=node.lineno,
                        docstring=python_ast.get_docstring(node),
                    )
                )
            elif isinstance(node, python_ast.ClassDef):
                bases = ", ".join(
                    b.id if isinstance(b, python_ast.Name) else getattr(b, "attr", "?")
                    for b in node.bases
                )
                symbols.append(
                    CodeSymbol(
                        name=node.name,
                        kind="class",
                        signature=f"class {node.name}({bases})" if bases else f"class {node.name}",
                        line=node.lineno,
                        docstring=python_ast.get_docstring(node),
                    )
                )
            elif isinstance(node, python_ast.Import):
                for alias in node.names:
                    symbols.append(CodeSymbol(name=alias.name, kind="import", line=node.lineno))
            elif isinstance(node, python_ast.ImportFrom):
                symbols.append(CodeSymbol(name=node.module or ".", kind="import", line=node.lineno))
        return symbols
    return _extract_symbols_generic(source)


def _extract_symbols_generic(source: str) -> list[CodeSymbol]:
    symbols: list[CodeSymbol] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        class_match = _GENERIC_CLASS_RE.match(line)
        if class_match:
            symbols.append(
                CodeSymbol(name=class_match.group(1), kind="class", signature=line.strip(), line=line_number)
            )
            continue
        func_match = _GENERIC_FUNC_RE.match(line)
        if func_match:
            name = next((g for g in func_match.groups() if g), None)
            if name and name not in ("if", "for", "while", "switch", "return"):
                symbols.append(
                    CodeSymbol(name=name, kind="function", signature=line.strip()[:120], line=line_number)
                )
    return symbols
