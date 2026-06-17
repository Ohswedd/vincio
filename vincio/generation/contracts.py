"""Structural document contracts.

:class:`DocumentContract` is to a generated document what
:class:`~vincio.output.schemas.OutputContract` is to model text: it declares the
*shape* a deliverable must have — required sections, table column specs, length
bounds, a citation in every section — and validates a :class:`DocumentModel`
against it. Repair is **formatting-only** (the document-generation mirror of the
JSON-repair path): it normalizes heading levels, derives a missing title, and
trims whitespace, but it never invents content, so a structurally-deficient
document fails loudly instead of being silently padded.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..output.parsers import extract_citations
from .model import DocumentModel

__all__ = [
    "TableSpec",
    "DocumentContract",
    "DocumentValidationStep",
    "DocumentValidationReport",
    "validate_document",
    "repair_formatting",
]


def _norm(text: str) -> str:
    return " ".join(text.split()).strip().lower()


class TableSpec(BaseModel):
    """Required shape for a table the document must contain."""

    # Match a table by its id or (normalized) title; ``"*"`` matches every table.
    match: str = "*"
    required_columns: list[str] = Field(default_factory=list)
    min_rows: int = 0


class DocumentContract(BaseModel):
    """The structural contract a generated document must satisfy."""

    required_sections: list[str] = Field(default_factory=list)
    table_specs: list[TableSpec] = Field(default_factory=list)
    min_words: int | None = None
    max_words: int | None = None
    require_title: bool = True
    # Every non-empty section must carry at least one ``[E1]``-style citation.
    citations_per_section: bool = False
    # Formatting-only repair (heading-level normalization, title derivation,
    # whitespace) — mirrors RepairPolicy.allow_markdown_formatting. Content is
    # never invented; a missing required section is always a violation.
    allow_formatting_repair: bool = True


class DocumentValidationStep(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    repaired: bool = False


class DocumentValidationReport(BaseModel):
    valid: bool = False
    steps: list[DocumentValidationStep] = Field(default_factory=list)
    repairs: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)

    def step(self, name: str, passed: bool, detail: str = "", *, repaired: bool = False) -> None:
        self.steps.append(
            DocumentValidationStep(name=name, passed=passed, detail=detail, repaired=repaired)
        )
        if not passed and detail:
            self.violations.append(f"{name}: {detail}")


def repair_formatting(model: DocumentModel, contract: DocumentContract) -> list[str]:
    """Apply formatting-only repairs in place. Returns a list of actions taken.

    Never adds, removes, or rewrites substantive content — only normalizes
    structure (heading levels can't skip), derives a title from the first
    heading/metadata, and trims trailing whitespace.
    """
    actions: list[str] = []

    # Derive a missing title from metadata or the first heading.
    if contract.require_title and not model.title.strip():
        derived = str(model.metadata.get("title") or "").strip()
        if not derived:
            headings = model.headings()
            if headings:
                derived = headings[0].text.strip()
        if derived:
            model.title = derived
            actions.append("title derived from first heading/metadata")

    # Trim trailing whitespace on text blocks.
    for block in model.blocks:
        if block.text and block.text != block.text.rstrip():
            block.text = block.text.rstrip()
            actions.append("trimmed trailing whitespace")
            break  # report once; loop still trims all below
    for block in model.blocks:
        if block.text:
            block.text = block.text.rstrip()

    # Normalize heading levels so they never skip (h1 → h3 becomes h1 → h2).
    last_level = 0
    changed_levels = False
    for block in model.blocks:
        if block.kind != "heading":
            continue
        if block.level > last_level + 1:
            block.level = last_level + 1
            changed_levels = True
        last_level = block.level
    if changed_levels:
        actions.append("normalized skipped heading levels")

    return actions


def _section_text(body: list) -> str:  # list[DocBlock]
    parts: list[str] = []
    for block in body:
        if block.text:
            parts.append(block.text)
        if block.items:
            parts.extend(block.items)
        if block.table is not None:
            parts.append(block.table.to_text())
    return "\n".join(parts)


def validate_document(
    model: DocumentModel, contract: DocumentContract
) -> DocumentValidationReport:
    """Validate ``model`` against ``contract`` (after any allowed repair)."""
    report = DocumentValidationReport()

    if contract.allow_formatting_repair:
        report.repairs = repair_formatting(model, contract)

    # 1. title
    if contract.require_title:
        ok = bool(model.title.strip())
        report.step("title", ok, "" if ok else "document has no title", repaired=bool(report.repairs))

    # 2. required sections (by normalized heading text/anchor)
    if contract.required_sections:
        present = {_norm(h.text) for h in model.headings()} | {
            _norm(h.anchor) for h in model.headings() if h.anchor
        }
        missing = [s for s in contract.required_sections if _norm(s) not in present]
        report.step(
            "required_sections",
            not missing,
            "" if not missing else f"missing sections: {missing}",
        )

    # 3. table specs
    for index, spec in enumerate(contract.table_specs):
        tables = [b.table for b in model.tables() if b.table is not None]
        if spec.match != "*":
            target = _norm(spec.match)
            tables = [t for t in tables if _norm(t.id) == target or _norm(t.title) == target]
        label = spec.match if spec.match != "*" else f"#{index}"
        if not tables:
            report.step(f"table:{label}", False, f"no table matches {spec.match!r}")
            continue
        problems: list[str] = []
        for table in tables:
            cols = {_norm(c) for c in table.columns}
            absent = [c for c in spec.required_columns if _norm(c) not in cols]
            if absent:
                problems.append(f"{table.id or table.title!r} missing columns {absent}")
            if len(table.rows) < spec.min_rows:
                problems.append(
                    f"{table.id or table.title!r} has {len(table.rows)} rows < {spec.min_rows}"
                )
        report.step(f"table:{label}", not problems, "; ".join(problems))

    # 4. length bounds
    if contract.min_words is not None or contract.max_words is not None:
        words = model.word_count()
        ok = (contract.min_words is None or words >= contract.min_words) and (
            contract.max_words is None or words <= contract.max_words
        )
        bounds = f"[{contract.min_words or 0}, {contract.max_words or '∞'}]"
        report.step("length", ok, "" if ok else f"{words} words outside {bounds}")

    # 5. citation in every section
    if contract.citations_per_section:
        uncited: list[str] = []
        for head, body in model.sections():
            text = _section_text(body)
            if not text.strip():
                continue  # an empty section (e.g. a heading-only divider) is exempt
            if not extract_citations(text + " " + (head.text or "")):
                uncited.append(head.text or "(intro)")
        report.step(
            "citations_per_section",
            not uncited,
            "" if not uncited else f"sections without a citation: {uncited}",
        )

    report.valid = all(s.passed for s in report.steps)
    return report
