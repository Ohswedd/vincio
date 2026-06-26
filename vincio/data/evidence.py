"""Tabular data as first-class context evidence.

:class:`TableEvidence` wraps a :class:`~vincio.data.Dataset` into the evidence
the context compiler already scores, deduplicates, budgets, orders, and cites —
``modality="table"`` evidence whose scorable text and prompt rendering are the
dataset's compact encoding (header-once, schema-declared-once) and whose token
cost is the columnar-accurate count of that encoding, not a per-cell heuristic.

A dataset stays *structured* all the way to the prompt: it is never flattened to
a pipe-joined string or dumped as ``json.dumps``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.tokens import count_tokens
from ..core.types import EvidenceItem, TrustLevel
from .core import Dataset
from .encoders import DataEncoder

__all__ = ["TableEvidence"]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class TableEvidence(BaseModel):
    """A :class:`~vincio.data.Dataset` presented as first-class context evidence.

    Convert it with :meth:`to_evidence_item` (or hand it straight to the context
    compiler's evidence list, or add it to ``app.pending_evidence`` for the next
    run — both coerce it): the resulting :class:`~vincio.core.types.EvidenceItem`
    carries the compact encoding as its scorable text and renders it verbatim into
    the prompt, so the table reaches the model header-once instead of per row.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: Dataset
    source_id: str = "dataset"
    citation: str = ""
    caption: str = ""
    trust_level: TrustLevel = TrustLevel.UNTRUSTED_DOCUMENT
    relevance: float = 0.0
    authority: float = 0.5
    provenance: float = 0.5
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    encoder: DataEncoder | None = None

    def encode(self) -> str:
        """The dataset's compact encoding (using this evidence's encoder, if any)."""
        return self.dataset.encode(self.encoder)

    def token_cost(self, *, model: str | None = None) -> int:
        """The columnar-accurate token cost of the encoded table."""
        return count_tokens(self.encode(), model)

    def to_evidence_item(self) -> EvidenceItem:
        """Project to a provenance-aware :class:`~vincio.core.types.EvidenceItem`
        (``modality="table"``) the context compiler consumes unchanged."""
        encoding = self.encode()
        rows = [[_json_safe(cell) for cell in row] for row in self.dataset.rows()]
        table: dict[str, Any] = {
            "columns": self.dataset.column_names,
            "rows": rows,
            "encoding": encoding,
            "schema": [
                {
                    "name": col.name,
                    "dtype": col.dtype.value,
                    "unit": col.unit,
                    "nullable": col.nullable,
                }
                for col in self.dataset.columns
            ],
        }
        if self.caption:
            table["caption"] = self.caption
        item = EvidenceItem(
            source_id=self.source_id,
            source_type="database",
            modality="table",
            text=encoding,
            table=table,
            token_cost=count_tokens(encoding),
            trust_level=self.trust_level,
            relevance=self.relevance,
            authority=self.authority,
            provenance=self.provenance,
            page=self.page,
            metadata={
                "row_count": self.dataset.row_count,
                "column_count": self.dataset.width,
                **self.metadata,
            },
        )
        if self.citation:
            item.id = self.citation
        return item

    @classmethod
    def from_records(
        cls, records: list[dict[str, Any]], *, name: str = "", source_id: str = "", **kwargs: Any
    ) -> TableEvidence:
        """Build table evidence directly from a list of record mappings."""
        dataset = Dataset.from_records(records, name=name)
        return cls(dataset=dataset, source_id=source_id or name or "dataset", **kwargs)
