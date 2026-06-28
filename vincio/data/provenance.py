"""Cell-level provenance for an analytical answer.

An analytical answer is only trustworthy if you can point at the exact rows and
cells it rests on. This module carries that lineage as first-class, content-bound
evidence — the analytics analogue of a cited report's per-claim entailment.

* :class:`CellCitation` — a reference to one source cell ``table#r<row>!<column>``
  with the value it held, so a cited figure points at the precise cell it came
  from and a tampered source is caught.
* :class:`RowProvenance` — the source cells one result row rests on, and whether
  that lineage is **exact** (a projection/filter result row maps to one source
  row; an aggregate row maps to the group it summarizes) or only table-level.
* :class:`LineageCoverage` — an explicit, never-silent statement of how precise
  the lineage is for a whole result: ``cell`` (every result row traced to its
  source cells) or ``result`` (the result re-derives from the hashed source, but
  per-cell lineage was outside the traced grammar).

The lineage is computed deterministically by the query engine; nothing here calls
a model. ``CellCitation`` renders a stable, parseable reference that slots beside
the ``<source>:p<page>`` / ``<source>:t<start>-<end>`` locators the citation
machinery already uses.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class LineageCoverage(StrEnum):
    """How precisely a result's provenance is traced to its source.

    ``CELL`` — every result row carries the exact source cells it rests on (the
    overwhelming-majority analyst shapes: a single-table projection/filter, or a
    single-table group-by aggregation). ``RESULT`` — per-cell lineage was outside
    the traced grammar (e.g. a multi-table join or a nested subquery), so lineage
    is reported at the source-table level; the result still re-derives from the
    content-hashed source by re-executing the verified query, so it stays
    offline-verifiable. The coverage is always stated, never silently downgraded.
    """

    CELL = "cell"
    RESULT = "result"


class CellCitation(BaseModel):
    """A reference to one source cell an answer rests on.

    ``ref`` renders the stable, parseable locator ``table#r<row>!<column>`` (row
    is 0-based into the source dataset). ``value`` binds the cell's value at query
    time, so a later check can confirm the source still holds it (a tampered cell
    is caught). ``result_column`` is the output column this source cell contributes
    to, so a derived value (``revenue + tax``) cites exactly its operands rather
    than the whole row."""

    table: str
    row: int
    column: str
    value: Any = None
    result_column: str = ""

    @property
    def ref(self) -> str:
        """The stable cell locator, e.g. ``sales#r4!revenue``."""
        return f"{self.table}#r{self.row}!{self.column}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.ref


class RowProvenance(BaseModel):
    """The source cells one result row rests on.

    ``result_row`` is the 0-based index into the result table. ``cells`` are the
    source :class:`CellCitation`s that produced it — one source row for a
    projection/filter result, or every row of a group for an aggregate. ``exact``
    is true when the lineage is cell-precise (false when only the source table
    could be attributed)."""

    result_row: int
    cells: list[CellCitation] = Field(default_factory=list)
    exact: bool = True

    def citations_for(self, column: str) -> list[CellCitation]:
        """The source cells the named result column rests on.

        Matches the cells tagged with that output column (so a derived value cites
        exactly its operands), falling back to a source-column-name match and then
        to every contributing cell for an untagged result."""
        tagged = [c for c in self.cells if c.result_column == column]
        if tagged:
            return tagged
        named = [c for c in self.cells if c.column == column]
        return named or list(self.cells)

    @property
    def refs(self) -> list[str]:
        """The distinct stable locators of every source cell this row rests on."""
        seen: set[str] = set()
        out: list[str] = []
        for c in self.cells:
            if c.ref not in seen:
                seen.add(c.ref)
                out.append(c.ref)
        return out
