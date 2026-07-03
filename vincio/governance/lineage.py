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

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.utils import compact_json, sha256_text, utcnow

__all__ = [
    "LineageRecord",
    "LineageIndex",
    "ErasureResult",
    "ErasureProof",
    "build_erasure_proof",
    "verify_erasure_proof",
]

_logger = logging.getLogger("vincio.governance.lineage")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.types import Chunk, Document, RunResult
    from .transparency import ContentSigner


class LineageRecord(BaseModel):
    """The full provenance chain for one source."""

    source: str
    documents: list[str] = Field(default_factory=list)
    chunks: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    runs: list[str] = Field(default_factory=list)
    # generated artifacts (cited documents, images, audio blob keys) whose
    # content derives from this source, so an erasure removes the deliverable too.
    artifacts: list[str] = Field(default_factory=list)
    # registered tabular datasets (by catalog table name) ingested from this
    # source, and the columns each carries — so a metric's provenance and a
    # subject's erasure both reach into the dataset plane.
    datasets: list[str] = Field(default_factory=list)
    dataset_columns: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (
            self.documents
            or self.chunks
            or self.evidence
            or self.runs
            or self.artifacts
            or self.datasets
        )


class ErasureResult(BaseModel):
    """Outcome of a right-to-erasure-by-source sweep."""

    source: str
    found: bool = False
    documents_removed: int = 0
    chunks_removed: int = 0
    memories_removed: int = 0
    caches_invalidated: int = 0
    indexes_swept: int = 0
    # generated artifacts (cited documents, images, audio) tied to the
    # source and removed in the same sweep — so an erased source is erased as
    # evidence, as memory, *and* as generated output in one operation.
    artifacts_removed: int = 0
    # registered datasets removed from the data catalog in the same sweep — so an
    # erased source is erased as evidence, memory, generated output, *and*
    # structured data.
    datasets_removed: int = 0
    audit_entry_id: str | None = None
    # the signed, content-bound proof of exactly what was removed.
    proof: ErasureProof | None = None

    @property
    def total_removed(self) -> int:
        return (
            self.documents_removed
            + self.chunks_removed
            + self.memories_removed
            + self.artifacts_removed
            + self.datasets_removed
        )


class ErasureProof(BaseModel):
    """A signed, content-bound manifest of exactly what an erasure removed.

    The proof records the per-store removal counts *and* a SHA-256 digest over
    the sorted set of removed identifiers (chunk ids, document ids, memory ids,
    artifact keys). The digest binds the proof to the precise removal: a later
    "we deleted everything" claim can be checked against the recorded ids, and
    the optional signature makes the manifest tamper-evident against an attacker
    who can edit the file. It rides the same hash-chained audit log (and its
    Merkle checkpoints) that the citations already use, so erasure is *provable*,
    not merely logged.
    """

    source: str
    created_at: datetime = Field(default_factory=utcnow)
    claim_generator: str = "vincio"
    removed: dict[str, int] = Field(default_factory=dict)  # store -> count
    removed_ids: dict[str, list[str]] = Field(default_factory=dict)  # store -> ids
    content_sha256: str = ""  # digest over the sorted removed-id set
    audit_entry_id: str | None = None
    audit_merkle_root: str | None = None  # the chain root at proof time
    signature: dict[str, Any] | None = None
    key_id: str | None = None

    @property
    def content_hash(self) -> str:
        """Canonical name for :attr:`content_sha256` (the removed-id-set digest).

        The stored field keeps its historical name: it is serialized in
        :meth:`to_dict` and bound by name into :meth:`signing_payload`, both of
        which persisted, signed proofs depend on.
        """
        return self.content_sha256

    def digest_payload(self) -> str:
        """Canonical bytes the content digest covers (the removed-id set)."""
        canonical = {store: sorted(ids) for store, ids in sorted(self.removed_ids.items())}
        # Historical spaced-separator form — content_sha256 inside signed,
        # persisted proofs depends on these exact bytes; do NOT switch to
        # core.utils.compact_json.
        return json.dumps({"source": self.source, "removed_ids": canonical}, sort_keys=True)

    def digest(self) -> str:
        """SHA-256 content hash over :meth:`digest_payload` (the removed-id set)."""
        return sha256_text(self.digest_payload())

    def signing_payload(self) -> str:
        """Deterministic bytes the signature covers (binds the credential)."""
        return compact_json(
            {
                "source": self.source,
                "created_at": self.created_at.isoformat(),
                "claim_generator": self.claim_generator,
                "removed": dict(sorted(self.removed.items())),
                # frozen wire key: covered by persisted signatures — never rename.
                "content_sha256": self.content_sha256,
                "audit_merkle_root": self.audit_merkle_root,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def build_erasure_proof(
    source: str,
    removed_ids: dict[str, list[str]],
    *,
    counts: dict[str, int] | None = None,
    signer: ContentSigner | None = None,
    claim_generator: str | None = None,
    audit_entry_id: str | None = None,
    audit_merkle_root: str | None = None,
) -> ErasureProof:
    """Assemble (and optionally sign) the proof for one erasure sweep."""
    import vincio

    proof = ErasureProof(
        source=source,
        claim_generator=claim_generator or f"vincio/{vincio.__version__}",
        removed=counts or {store: len(ids) for store, ids in removed_ids.items()},
        removed_ids={store: list(ids) for store, ids in removed_ids.items()},
        audit_entry_id=audit_entry_id,
        audit_merkle_root=audit_merkle_root,
    )
    proof.content_sha256 = proof.digest()
    if signer is not None:
        proof.signature = {
            "alg": getattr(signer, "alg", "HMAC-SHA256"),
            "key_id": getattr(signer, "key_id", "default"),
            "value": signer.sign(proof.signing_payload()),
        }
        proof.key_id = getattr(signer, "key_id", "default")
    return proof


def verify_erasure_proof(
    proof: ErasureProof, *, signer: ContentSigner | None = None
) -> bool:
    """Verify a proof's content binding and (if present) its signature.

    Recomputes the digest over the recorded removed-id set and checks it against
    ``content_sha256``; if a signature is present, a ``signer`` with the matching
    key must verify it (a present-but-unverifiable signature is never reported
    valid)."""
    expected = proof.digest()
    if proof.content_hash != expected:
        return False
    if proof.signature is not None:
        if signer is None:
            return False
        return signer.verify(proof.signing_payload(), proof.signature.get("value", ""))
    return True


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
        self._table_to_source: dict[str, str] = {}

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

    def record_artifact(self, source: str, artifact_key: str) -> None:
        """Record that a generated artifact (a cited document, image, or audio
        blob key) derives from ``source`` — so erasure removes it too."""
        record = self._record(source)
        if artifact_key not in record.artifacts:
            record.artifacts.append(artifact_key)

    def record_dataset(self, source: str, table: str, columns: list[str] | None = None) -> None:
        """Record that a registered tabular dataset (catalog table ``table``,
        carrying ``columns``) was ingested from ``source`` — so a metric's
        column-level provenance traces to the source and a right-to-erasure sweep
        removes the dataset alongside the source's documents and memories."""
        record = self._record(source)
        if table not in record.datasets:
            record.datasets.append(table)
        record.dataset_columns[table] = list(columns or [])
        self._table_to_source[table] = source

    def source_of_table(self, table: str) -> str | None:
        """The source a registered dataset table was ingested under, if any."""
        return self._table_to_source.get(table)

    def datasets_for(self, source: str) -> list[str]:
        """The registered dataset tables ingested from a source."""
        return list(self.trace(source).datasets)

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
        for table in record.datasets:
            self._table_to_source.pop(table, None)
        return record

    def to_dict(self) -> dict[str, Any]:
        return {source: record.model_dump(mode="json") for source, record in self._records.items()}
