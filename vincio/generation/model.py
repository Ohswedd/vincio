"""The document intermediate representation (IR).

A :class:`DocumentModel` is a format-neutral tree of :class:`DocBlock`\\ s that
every renderer (Markdown/HTML/DOCX/PDF/PPTX) consumes, and that the
:class:`~vincio.generation.contracts.DocumentContract` validates. Building it
from a *validated* result is what makes a generated deliverable grounded by
construction: the renderers only lay out content that already passed the output
contract — they never invent it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..documents.parsers import TableData

__all__ = ["BlockKind", "DocBlock", "DocumentModel"]

BlockKind = Literal[
    "heading",
    "paragraph",
    "list",
    "table",
    "image",
    "code",
    "quote",
    "page_break",
]


class DocBlock(BaseModel):
    """One renderable block. Only the fields relevant to ``kind`` are used."""

    kind: BlockKind = "paragraph"
    text: str = ""  # heading / paragraph / quote / code / image caption
    level: int = 1  # heading level (1–6); also the section nesting for grouping
    items: list[str] = Field(default_factory=list)  # list blocks
    ordered: bool = False  # ordered vs. bulleted list
    table: TableData | None = None  # table blocks
    image_path: str | None = None  # image blocks (local path)
    image_alt: str = ""  # image alt text / accessibility label
    language: str = ""  # code blocks (syntax label)
    anchor: str = ""  # stable id for cross-references / section keys
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def section_key(self) -> str:
        """A normalized key for a heading block (for contract section matching)."""
        return (self.anchor or self.text).strip().lower()


class DocumentModel(BaseModel):
    """A format-neutral document: title, ordered blocks, resolved references."""

    model_config = {"arbitrary_types_allowed": True}

    title: str = ""
    subtitle: str = ""
    blocks: list[DocBlock] = Field(default_factory=list)
    # Resolved citation references (rendered as footnotes/endnotes) and the
    # generated source table, populated by the CitedReportBuilder.
    footnotes: list[str] = Field(default_factory=list)
    bibliography: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Evidence ids this document was grounded on (carried into the audit event).
    source_evidence_ids: list[str] = Field(default_factory=list)

    # -- construction helpers --------------------------------------------------

    def heading(self, text: str, *, level: int = 1, anchor: str = "") -> DocumentModel:
        self.blocks.append(DocBlock(kind="heading", text=text, level=level, anchor=anchor))
        return self

    def paragraph(self, text: str) -> DocumentModel:
        self.blocks.append(DocBlock(kind="paragraph", text=text))
        return self

    def bullet_list(self, items: list[str], *, ordered: bool = False) -> DocumentModel:
        self.blocks.append(DocBlock(kind="list", items=list(items), ordered=ordered))
        return self

    def add_table(self, table: TableData) -> DocumentModel:
        self.blocks.append(DocBlock(kind="table", table=table))
        return self

    def image(self, path: str, *, caption: str = "", alt: str = "") -> DocumentModel:
        self.blocks.append(DocBlock(kind="image", image_path=path, text=caption, image_alt=alt))
        return self

    # -- queries ---------------------------------------------------------------

    def headings(self) -> list[DocBlock]:
        return [b for b in self.blocks if b.kind == "heading"]

    def tables(self) -> list[DocBlock]:
        return [b for b in self.blocks if b.kind == "table" and b.table is not None]

    def word_count(self) -> int:
        """Total words across heading/paragraph/quote/list/table text."""
        words = 0
        for block in self.blocks:
            if block.kind in ("heading", "paragraph", "quote", "code"):
                words += len(block.text.split())
            elif block.kind == "list":
                words += sum(len(item.split()) for item in block.items)
            elif block.kind == "table" and block.table is not None:
                # Count the table's content words from its cells, independent of
                # the rendering format (column headers + every cell).
                table = block.table
                words += sum(len(str(col).split()) for col in table.columns)
                words += sum(len(str(cell).split()) for row in table.rows for cell in row)
        return words

    def plain_text(self) -> str:
        """A plain-text rendering — used for length/citation checks and hashing."""
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        if self.subtitle:
            parts.append(self.subtitle)
        for block in self.blocks:
            if block.kind in ("heading", "paragraph", "quote", "code"):
                parts.append(block.text)
            elif block.kind == "list":
                parts.extend(block.items)
            elif block.kind == "table" and block.table is not None:
                parts.append(block.table.to_text())
            elif block.kind == "image" and block.text:
                parts.append(block.text)
        if self.footnotes:
            parts.extend(self.footnotes)
        return "\n\n".join(p for p in parts if p)

    def sections(self) -> list[tuple[DocBlock, list[DocBlock]]]:
        """Group blocks into (heading, body-blocks-until-next-heading) sections.

        A leading run of blocks before the first heading is returned under a
        synthetic empty heading, so a flat document still has one section.
        """
        out: list[tuple[DocBlock, list[DocBlock]]] = []
        current_head: DocBlock | None = None
        body: list[DocBlock] = []
        for block in self.blocks:
            if block.kind == "heading":
                if current_head is not None or body:
                    out.append((current_head or DocBlock(kind="heading", text=""), body))
                current_head = block
                body = []
            else:
                body.append(block)
        if current_head is not None or body:
            out.append((current_head or DocBlock(kind="heading", text=""), body))
        return out
