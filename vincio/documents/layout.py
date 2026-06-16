"""Layout-aware document extraction for complex PDFs.

The offline loaders (``load_pdf`` via pypdf) extract text in stream order,
which scrambles multi-column pages and loses tables and figures. This module
adds an advanced path that recovers **reading order**, **tables**, and
**figures** with bounding boxes — behind the existing document engine, with
the dependency-free loaders kept as the default.

The reading-order and assembly logic is pure and fully testable offline
(:func:`group_words_into_lines`, :func:`order_blocks`, :func:`assemble_layout`);
:func:`extract_pdf_layout` is a thin adapter over ``pdfplumber`` (lazy import,
``pip install "vincio[pdf-layout]"``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import LoaderError
from ..core.types import Document
from .parsers import TableData, infer_table_schema, table_quality_checks

__all__ = [
    "LayoutWord",
    "LayoutBlock",
    "LayoutFigure",
    "PageLayout",
    "group_words_into_lines",
    "order_blocks",
    "assemble_layout",
    "extract_pdf_layout",
]


class LayoutWord(BaseModel):
    text: str
    x0: float
    top: float
    x1: float
    bottom: float


class LayoutBlock(BaseModel):
    text: str
    x0: float
    top: float
    x1: float
    bottom: float


class LayoutFigure(BaseModel):
    page: int
    x0: float
    top: float
    x1: float
    bottom: float
    caption: str = ""


class PageLayout(BaseModel):
    page_number: int
    width: float = 0.0
    height: float = 0.0
    blocks: list[LayoutBlock] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)
    figures: list[LayoutFigure] = Field(default_factory=list)


def group_words_into_lines(words: list[LayoutWord], *, y_tol: float = 3.0) -> list[LayoutBlock]:
    """Group positioned words into text lines (one :class:`LayoutBlock` per
    line). Words whose vertical positions are within ``y_tol`` join a line;
    within a line they are ordered left-to-right."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(w.top, 1), w.x0))
    lines: list[list[LayoutWord]] = []
    current: list[LayoutWord] = [ordered[0]]
    line_top = ordered[0].top
    for word in ordered[1:]:
        if abs(word.top - line_top) <= y_tol:
            current.append(word)
        else:
            lines.append(current)
            current = [word]
            line_top = word.top
    lines.append(current)
    blocks: list[LayoutBlock] = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w.x0)
        blocks.append(
            LayoutBlock(
                text=" ".join(w.text for w in line_sorted),
                x0=min(w.x0 for w in line),
                top=min(w.top for w in line),
                x1=max(w.x1 for w in line),
                bottom=max(w.bottom for w in line),
            )
        )
    return blocks


def order_blocks(blocks: list[LayoutBlock], *, page_width: float) -> list[LayoutBlock]:
    """Order blocks into reading order, column-aware.

    Detects a two-column layout (blocks that sit clearly left vs. clearly
    right of the page midline, with none straddling it) and reads the left
    column top-to-bottom before the right; otherwise reads top-to-bottom,
    left-to-right."""
    if len(blocks) < 4 or page_width <= 0:
        return sorted(blocks, key=lambda b: (round(b.top, 1), b.x0))
    mid = page_width / 2.0
    left = [b for b in blocks if b.x1 <= mid]
    right = [b for b in blocks if b.x0 >= mid]
    straddling = [b for b in blocks if b.x0 < mid < b.x1]
    # Two real columns: a balanced split with nothing spanning the gutter.
    if not straddling and len(left) >= 2 and len(right) >= 2:
        left.sort(key=lambda b: (round(b.top, 1), b.x0))
        right.sort(key=lambda b: (round(b.top, 1), b.x0))
        return left + right
    return sorted(blocks, key=lambda b: (round(b.top, 1), b.x0))


def assemble_layout(
    pages: list[PageLayout],
    *,
    title: str = "",
    source_uri: str | None = None,
    tenant_id: str | None = None,
) -> Document:
    """Assemble extracted page layouts into a :class:`Document` with
    reading-order text, sections, tables, and figure provenance."""
    page_texts: list[str] = []
    sections: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    table_counter = 0
    for page in pages:
        ordered = order_blocks(page.blocks, page_width=page.width)
        body = "\n".join(b.text for b in ordered if b.text.strip())
        page_texts.append(body)
        sections.append(
            {
                "title": f"page {page.page_number}",
                "level": 1,
                "path": [f"p{page.page_number}"],
                "text": body,
                "start_line": 0,
                "page": page.page_number,
                "reading_order_blocks": len(ordered),
            }
        )
        for table in page.tables:
            table_counter += 1
            table.id = table.id or f"T{table_counter}"
            if not table.inferred_schema:
                table.inferred_schema = infer_table_schema(table)
            if not table.quality:
                table.quality = table_quality_checks(table)
            dump = table.model_dump(mode="json")
            dump["page"] = page.page_number
            tables.append(dump)
        for figure in page.figures:
            figures.append(
                {
                    "page": figure.page or page.page_number,
                    "bbox": [figure.x0, figure.top, figure.x1, figure.bottom],
                    "caption": figure.caption,
                }
            )
    return Document(
        text="\n\n".join(t for t in page_texts if t),
        title=title,
        media_type="application/pdf",
        source_uri=source_uri,
        tenant_id=tenant_id,
        sections=sections,
        tables=tables,
        metadata={
            "extractor": "layout",
            "page_count": len(pages),
            "table_count": len(tables),
            "figure_count": len(figures),
            "figures": figures,
        },
    )


def extract_pdf_layout(path: str | Path) -> Document:
    """Layout-aware PDF extraction via ``pdfplumber``
    (``pip install "vincio[pdf-layout]"``).

    Recovers reading order from positioned words, extracts tables with their
    bounding boxes, and records figure regions — then assembles them with
    :func:`assemble_layout`."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise LoaderError(
            'Layout-aware PDF extraction requires pdfplumber: '
            'pip install "vincio[pdf-layout]"'
        ) from exc
    path = Path(path)
    pages: list[PageLayout] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = [
                LayoutWord(
                    text=w["text"],
                    x0=float(w["x0"]),
                    top=float(w["top"]),
                    x1=float(w["x1"]),
                    bottom=float(w["bottom"]),
                )
                for w in page.extract_words()
            ]
            tables: list[TableData] = []
            for raw in page.extract_tables() or []:
                rows = [[("" if cell is None else str(cell)).strip() for cell in row] for row in raw]
                rows = [r for r in rows if any(c for c in r)]
                if not rows:
                    continue
                tables.append(TableData(columns=rows[0], rows=rows[1:]))
            figures = [
                LayoutFigure(
                    page=page_number,
                    x0=float(img["x0"]),
                    top=float(img["top"]),
                    x1=float(img["x1"]),
                    bottom=float(img["bottom"]),
                )
                for img in (page.images or [])
            ]
            pages.append(
                PageLayout(
                    page_number=page_number,
                    width=float(page.width or 0.0),
                    height=float(page.height or 0.0),
                    blocks=group_words_into_lines(words),
                    tables=tables,
                    figures=figures,
                )
            )
    return assemble_layout(pages, title=path.stem, source_uri=str(path))
