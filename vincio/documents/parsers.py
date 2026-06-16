"""Document structuring: sections, tables, spreadsheets,
code symbols. Operates on raw text/bytes produced by the loaders."""

from __future__ import annotations

import ast as python_ast
import csv
import io
import re
from typing import Any

from pydantic import BaseModel, Field

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
        """Render preserving headers — chunk-friendly."""
        lines = []
        if self.title:
            lines.append(f"Table: {self.title}")
        lines.append(" | ".join(self.columns))
        for row in self.rows:
            lines.append(" | ".join(str(cell) for cell in row))
        for note in self.footnotes:
            lines.append(f"Note: {note}")
        return "\n".join(lines)


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
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}|^\d{1,2}[/.]\d{1,2}[/.]\d{2,4}$")
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
