"""Data lineage: source → chunk → evidence → output, and erasure-by-source.

The evidence ledger already carries provenance on every item; lineage threads
those links into one queryable chain so two governance questions have a
mechanical answer:

1. *Where did this answer come from?* — :meth:`LineageIndex.trace` returns the
   documents, chunks, evidence, and runs tied to a source.
2. *Forget this source.* — :class:`ErasureResult` records a GDPR
   right-to-erasure sweep that removes a source's chunks from every index, its
   memories, and its cache entries, logged on the hash-chained audit chain.

The :class:`LineageIndex` is populated as the app ingests (source → chunks) and
runs (evidence/source → run). The erasure *orchestration* lives on
``ContextApp.erase_source`` because it spans indexes, memory, and caches; this
module owns the lineage data model and the erasure result it produces.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

__all__ = ["LineageRecord", "LineageIndex", "ErasureResult"]

_logger = logging.getLogger("vincio.governance.lineage")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.types import Chunk, Document, RunResult


class LineageRecord(BaseModel):
    """The full provenance chain for one source."""

    source: str
    documents: list[str] = Field(default_factory=list)
    chunks: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    runs: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.documents or self.chunks or self.evidence or self.runs)


class ErasureResult(BaseModel):
    """Outcome of a right-to-erasure-by-source sweep."""

    source: str
    found: bool = False
    documents_removed: int = 0
    chunks_removed: int = 0
    memories_removed: int = 0
    caches_invalidated: int = 0
    indexes_swept: int = 0
    audit_entry_id: str | None = None

    @property
    def total_removed(self) -> int:
        return self.documents_removed + self.chunks_removed + self.memories_removed


class LineageIndex:
    """Records source → document → chunk → evidence → output edges.

    Lightweight and in-process: every edge is recorded as the app ingests and
    runs, so a later trace or erasure is a dictionary lookup rather than a
    crawl across stores.
    """

    def __init__(self) -> None:
        self._records: dict[str, LineageRecord] = {}
        self._doc_to_source: dict[str, str] = {}
        self._chunk_to_source: dict[str, str] = {}

    def _record(self, source: str) -> LineageRecord:
        return self._records.setdefault(source, LineageRecord(source=source))

    def record_ingest(
        self, source: str, *, documents: list[Document] | None = None, chunks: list[Chunk] | None = None
    ) -> None:
        """Record that ``source`` produced these documents and chunks."""
        record = self._record(source)
        for document in documents or []:
            if document.id not in record.documents:
                record.documents.append(document.id)
            self._doc_to_source[document.id] = source
        for chunk in chunks or []:
            if chunk.id not in record.chunks:
                record.chunks.append(chunk.id)
            self._chunk_to_source[chunk.id] = source
            # A chunk also binds its document to the source.
            self._doc_to_source.setdefault(chunk.document_id, source)
            if chunk.document_id not in record.documents:
                record.documents.append(chunk.document_id)

    def record_run(self, result: RunResult) -> None:
        """Link a run's cited evidence back to the sources it came from.

        Evidence with no registered source (legitimately, tool/memory/web
        origins) is skipped; the count is logged at debug so a dangling
        reference is diagnosable without failing the run.
        """
        run_id = result.run_id
        orphaned = 0
        for item in result.evidence:
            source = self._doc_to_source.get(item.source_id)
            if source is None:
                orphaned += 1
                continue
            record = self._record(source)
            if item.id not in record.evidence:
                record.evidence.append(item.id)
            if run_id and run_id not in record.runs:
                record.runs.append(run_id)
        if orphaned:
            _logger.debug("run %s: %d evidence item(s) had no registered source", run_id, orphaned)

    def trace(self, source: str) -> LineageRecord:
        """Return the lineage for a source name or a document id."""
        if source in self._records:
            return self._records[source]
        mapped = self._doc_to_source.get(source)
        if mapped is not None:
            return self._records[mapped]
        return LineageRecord(source=source)

    def source_of_chunk(self, chunk_id: str) -> str | None:
        return self._chunk_to_source.get(chunk_id)

    def sources(self) -> list[str]:
        return sorted(self._records)

    def chunk_ids_for(self, source: str) -> list[str]:
        return list(self.trace(source).chunks)

    def document_ids_for(self, source: str) -> list[str]:
        return list(self.trace(source).documents)

    def forget(self, source: str) -> LineageRecord:
        """Drop the lineage entry for a source after erasure, returning it."""
        record = self.trace(source)
        key = source if source in self._records else self._doc_to_source.get(source)
        if key is not None:
            self._records.pop(key, None)
        for chunk_id in record.chunks:
            self._chunk_to_source.pop(chunk_id, None)
        for doc_id in record.documents:
            self._doc_to_source.pop(doc_id, None)
        return record

    def to_dict(self) -> dict[str, Any]:
        return {source: record.model_dump(mode="json") for source, record in self._records.items()}
