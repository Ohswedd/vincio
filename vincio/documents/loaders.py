"""Document loaders.

``load_document(path)`` dispatches by type and returns a structured
:class:`Document` with text, sections, tables, and metadata. Supported:
text, Markdown, HTML, CSV/TSV, JSON/JSONL/YAML, code files, email (.eml),
PDF (``pip install "vincio[pdf]"``), DOCX (``pip install "vincio[docx]"``),
XLSX (via openpyxl when installed), and code repositories (directories).
"""

from __future__ import annotations

import json
import re
from email import policy as email_policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from ..core.errors import LoaderError
from ..core.types import Document
from ..core.utils import new_id
from .parsers import (
    extract_code_symbols,
    extract_markdown_sections,
    extract_markdown_tables,
    parse_csv_table,
    strip_html,
)

__all__ = ["load_document", "load_directory", "load_pdf", "load_docx", "load_xlsx", "SUPPORTED_EXTENSIONS"]

_CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".sql": "sql",
    ".sh": "shell",
}

SUPPORTED_EXTENSIONS = (
    {".txt", ".md", ".rst", ".html", ".htm", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml",
     ".pdf", ".docx", ".xlsx", ".eml"}
    | set(_CODE_EXTENSIONS)
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def load_document(path: str | Path, *, tenant_id: str | None = None) -> Document:
    """Load a single file into a Document. Raises LoaderError on failure."""
    path = Path(path)
    if not path.is_file():
        raise LoaderError(f"file not found: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            document = load_pdf(path)
        elif suffix == ".docx":
            document = load_docx(path)
        elif suffix == ".xlsx":
            document = load_xlsx(path)
        elif suffix == ".eml":
            document = _load_email(path)
        elif suffix in (".html", ".htm"):
            html = _read_text(path)
            title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
            document = Document(
                text=strip_html(html),
                title=title_match.group(1).strip() if title_match else path.stem,
                media_type="text/html",
            )
        elif suffix in (".csv", ".tsv"):
            content = _read_text(path)
            table = parse_csv_table(content, title=path.stem, delimiter="\t" if suffix == ".tsv" else None)
            document = Document(
                text=table.to_text(),
                title=path.stem,
                media_type="text/csv",
                tables=[table.model_dump(mode="json")],
            )
        elif suffix in (".json", ".jsonl"):
            content = _read_text(path)
            document = Document(text=content, title=path.stem, media_type="application/json")
            try:
                if suffix == ".json":
                    document.metadata["parsed"] = True
                    json.loads(content)
            except json.JSONDecodeError:
                document.metadata["parsed"] = False
        elif suffix in (".yaml", ".yml"):
            document = Document(text=_read_text(path), title=path.stem, media_type="application/yaml")
        elif suffix in _CODE_EXTENSIONS:
            source = _read_text(path)
            language = _CODE_EXTENSIONS[suffix]
            symbols = extract_code_symbols(source, language=language)
            document = Document(
                text=source,
                title=path.name,
                media_type=f"text/x-{language}",
                metadata={
                    "language": language,
                    "symbols": [s.model_dump(mode="json") for s in symbols],
                    "imports": [s.name for s in symbols if s.kind == "import"],
                },
            )
        else:  # plain/markdown/unknown text
            text = _read_text(path)
            document = Document(
                text=text,
                title=path.stem,
                media_type="text/markdown" if suffix in (".md", ".rst") else "text/plain",
            )
            if suffix in (".md", ".rst"):
                document.sections = [
                    s.model_dump(mode="json") for s in extract_markdown_sections(text)
                ]
                document.tables = [t.model_dump(mode="json") for t in extract_markdown_tables(text)]
    except LoaderError:
        raise
    except Exception as exc:  # noqa: BLE001 - wrap any parser failure
        raise LoaderError(f"failed to load {path}: {exc}") from exc

    document.source_uri = str(path)
    document.tenant_id = tenant_id
    document.metadata.setdefault("filename", path.name)
    document.metadata.setdefault("size_bytes", path.stat().st_size)
    return document


def load_pdf(path: str | Path) -> Document:
    """PDF text extraction via pypdf (``pip install "vincio[pdf]"``)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise LoaderError(
            'PDF support requires pypdf: pip install "vincio[pdf]"'
        ) from exc
    path = Path(path)
    reader = PdfReader(str(path))
    pages: list[str] = []
    sections: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append(page_text)
        sections.append(
            {"title": f"page {page_number}", "level": 1, "path": [f"p{page_number}"],
             "text": page_text, "start_line": 0, "page": page_number}
        )
    metadata: dict[str, Any] = {"page_count": len(reader.pages)}
    if reader.metadata:
        for key in ("title", "author", "subject"):
            value = getattr(reader.metadata, key, None)
            if value:
                metadata[key] = str(value)
    return Document(
        text="\n\n".join(pages),
        title=metadata.get("title") or path.stem,
        media_type="application/pdf",
        sections=sections,
        metadata=metadata,
    )


def load_docx(path: str | Path) -> Document:
    """DOCX extraction via python-docx (``pip install "vincio[docx]"``)."""
    try:
        import docx  # python-docx
    except ImportError as exc:
        raise LoaderError(
            'DOCX support requires python-docx: pip install "vincio[docx]"'
        ) from exc
    path = Path(path)
    document = docx.Document(str(path))
    paragraphs: list[str] = []
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower() if paragraph.style else ""
        if style.startswith("heading"):
            level = int(re.sub(r"\D", "", style) or 1)
            current = {"title": text, "level": level, "path": [text], "text": "", "start_line": 0}
            sections.append(current)
        elif current is not None:
            current["text"] = (current["text"] + "\n" + text).strip()
        paragraphs.append(text)
    tables: list[dict[str, Any]] = []
    for index, table in enumerate(document.tables, start=1):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if rows:
            tables.append(
                {"id": f"T{index}", "title": "", "columns": rows[0], "rows": rows[1:],
                 "source": str(path), "footnotes": [], "units": {}, "inferred_schema": {}, "quality": {}}
            )
    return Document(
        text="\n".join(paragraphs),
        title=path.stem,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        sections=sections,
        tables=tables,
    )


def load_xlsx(path: str | Path) -> Document:
    """XLSX extraction via openpyxl, including formulas and sheet names."""
    try:
        import openpyxl
    except ImportError as exc:
        raise LoaderError("XLSX support requires openpyxl: pip install openpyxl") from exc
    path = Path(path)
    workbook = openpyxl.load_workbook(str(path), data_only=False)
    tables: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for sheet_index, sheet in enumerate(workbook.worksheets, start=1):
        rows = [[("" if c.value is None else str(c.value)) for c in row] for row in sheet.iter_rows()]
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        if not rows:
            continue
        from .parsers import TableData, infer_table_schema, table_quality_checks

        table = TableData(
            id=f"S{sheet_index}",
            title=sheet.title,
            columns=rows[0],
            rows=rows[1:],
            source=f"{path.name}:{sheet.title}",
        )
        table.inferred_schema = infer_table_schema(table)
        table.quality = table_quality_checks(table)
        formulas = [
            f"{cell.coordinate}={cell.value}"
            for row in sheet.iter_rows()
            for cell in row
            if isinstance(cell.value, str) and cell.value.startswith("=")
        ]
        table_dump = table.model_dump(mode="json")
        table_dump["formulas"] = formulas
        tables.append(table_dump)
        text_parts.append(table.to_text())
    return Document(
        text="\n\n".join(text_parts),
        title=path.stem,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        tables=tables,
        metadata={"sheets": [s.title for s in workbook.worksheets]},
    )


def _load_email(path: Path) -> Document:
    message = BytesParser(policy=email_policy.default).parse(path.open("rb"))
    body = message.get_body(preferencelist=("plain", "html"))
    text = ""
    if body is not None:
        content = body.get_content()
        text = strip_html(content) if body.get_content_subtype() == "html" else content
    headers = {
        "from": str(message.get("From", "")),
        "to": str(message.get("To", "")),
        "date": str(message.get("Date", "")),
        "subject": str(message.get("Subject", "")),
    }
    return Document(
        text=f"Subject: {headers['subject']}\nFrom: {headers['from']}\nTo: {headers['to']}\n\n{text}",
        title=headers["subject"] or path.stem,
        media_type="message/rfc822",
        metadata=headers,
    )


_DEFAULT_IGNORES = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}


def load_directory(
    path: str | Path,
    *,
    extensions: set[str] | None = None,
    ignore_dirs: set[str] | None = None,
    max_files: int = 5000,
    tenant_id: str | None = None,
) -> list[Document]:
    """Load a directory tree (code repositories / doc folders).

    Adds a repository summary document with the dependency/import graph when
    code files are present.
    """
    root = Path(path)
    if not root.is_dir():
        raise LoaderError(f"directory not found: {root}")
    ignore = _DEFAULT_IGNORES | (ignore_dirs or set())
    allowed = extensions or SUPPORTED_EXTENSIONS
    documents: list[Document] = []
    import_graph: dict[str, list[str]] = {}
    for file_path in sorted(root.rglob("*")):
        if len(documents) >= max_files:
            break
        if not file_path.is_file():
            continue
        if any(part in ignore for part in file_path.parts):
            continue
        if file_path.suffix.lower() not in allowed:
            continue
        try:
            document = load_document(file_path, tenant_id=tenant_id)
        except LoaderError:
            continue  # unreadable/optional-dep file inside a bulk load
        document.metadata["relative_path"] = str(file_path.relative_to(root))
        documents.append(document)
        if "imports" in document.metadata:
            import_graph[document.metadata["relative_path"]] = document.metadata["imports"]
    if import_graph:
        summary_lines = ["Repository import graph:"]
        for module, imports in sorted(import_graph.items()):
            if imports:
                summary_lines.append(f"{module} -> {', '.join(sorted(set(imports)))}")
        documents.append(
            Document(
                id=new_id("doc"),
                text="\n".join(summary_lines),
                title="repository_summary",
                media_type="text/plain",
                source_uri=str(root),
                tenant_id=tenant_id,
                metadata={"kind": "repo_summary", "file_count": len(documents)},
            )
        )
    return documents
