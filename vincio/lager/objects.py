"""Evidence Objects — the atomic retrieval unit of LAGER.

An :class:`EvidenceObject` is one claim lifted verbatim from a source document:
its ``claim`` text is a **byte-exact slice** of the canonicalized document text
(``claim == canon(document.text)[span[0]:span[1]]``), so every object re-derives
from its source offline. Identity is **content-derived** — the id hashes the
document's canonical text (not ``Document.id``, which is random per load), the
span, and the claim — so re-ingesting the same corpus yields identical ids,
edges, and traces across processes, while any edit to the underlying text
changes them.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import LagerError
from ..core.types import EvidenceItem, TrustLevel

__all__ = [
    "EvidenceObject",
    "EvidencePack",
    "EvidenceRelation",
    "canonical_text",
    "document_key",
]

#: Bumped when extraction output changes shape/semantics, so persisted objects
#: from an older extractor are distinguishable (recorded in metadata).
EXTRACTOR_VERSION = "1"

RelationKind = Literal["supports", "contradicts", "depends_on", "follows"]


def canonical_text(text: str) -> str:
    """The one canonical form spans index into: newline-normalized, once, at
    ingest. Never applied again downstream — spans are byte-exact offsets into
    this string."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def document_key(text: str) -> str:
    """Stable, content-derived document identity (``Document.id`` is random per
    load, so it can never participate in evidence identity).

    Raises :class:`~vincio.core.errors.LagerError` (never a bare
    ``UnicodeEncodeError``) when *text* carries an unpaired surrogate and cannot
    be canonicalized — a lone surrogate reaches here from a lossless
    ``bytes.decode(errors="surrogateescape")`` roundtrip, and the error contract
    requires every public raise to be a ``VincioError``."""
    try:
        encoded = canonical_text(text).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise LagerError(
            "document text contains an unpaired surrogate and cannot be "
            "canonicalized; re-decode the source with a UTF-8-clean codec "
            "(e.g. errors='replace') before ingest",
            details={"position": exc.start},
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _evidence_hash(doc_key: str, span: tuple[int, int], claim: str) -> str:
    hasher = hashlib.sha256()
    for part in (doc_key, str(span[0]), str(span[1]), claim):
        raw = part.encode("utf-8")
        hasher.update(len(raw).to_bytes(8, "big"))
        hasher.update(raw)
    return hasher.hexdigest()


class EvidenceRelation(BaseModel):
    """A typed edge from one Evidence Object to another."""

    kind: RelationKind
    target: str  # EO id
    weight: float = 1.0
    #: Why the edge exists (detector features) — every downstream penalty or
    #: expansion step stays traceable to the signal that created it.
    basis: str = ""


class EvidenceObject(BaseModel):
    """One atomic claim with provenance, entities, relations, and confidence.

    ``kind`` distinguishes prose claims from structural regions (a list, table,
    or code block is one object — never sentence-split, never counted toward
    claim-atomicity statistics).
    """

    id: str
    claim: str
    kind: Literal["claim", "list", "table", "code"] = "claim"
    doc_key: str
    document_id: str  # transport-only; NOT part of identity (random per load)
    source_uri: str | None = None
    title: str | None = None
    span: tuple[int, int]
    section_path: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)  # normalized, sorted
    terms: list[str] = Field(default_factory=list)  # IDF fallback bucket terms
    relations: list[EvidenceRelation] = Field(default_factory=list)
    confidence: float = 0.5
    authority: float = 0.5
    trust_level: TrustLevel = TrustLevel.UNTRUSTED_DOCUMENT
    observed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str = ""

    @classmethod
    def create(
        cls,
        *,
        claim: str,
        doc_key: str,
        span: tuple[int, int],
        **fields: Any,
    ) -> EvidenceObject:
        """Build an object with its content-derived hash and id."""
        content_hash = _evidence_hash(doc_key, span, claim)
        return cls(
            id=f"eo:{content_hash[:16]}",
            claim=claim,
            doc_key=doc_key,
            span=span,
            content_hash=content_hash,
            **fields,
        )

    def verify(self, document_text: str) -> bool:
        """True iff this object re-derives, byte-for-byte, from *document_text*.

        Checks, in order: the canonical text still hashes to ``doc_key``; the
        span is in bounds; the slice equals the claim exactly; and the content
        hash re-derives. All offline, no tokenizer involved."""
        canon = canonical_text(document_text)
        try:
            if document_key(document_text) != self.doc_key:
                return False
        except LagerError:
            return False  # an unencodable candidate can never be this object's source
        start, end = self.span
        if not (0 <= start <= end <= len(canon)):
            return False
        if canon[start:end] != self.claim:
            return False
        return _evidence_hash(self.doc_key, self.span, self.claim) == self.content_hash

    def as_evidence_item(self, *, relevance: float = 0.5, pinned: bool = False) -> EvidenceItem:
        """Bridge into the context compiler: explicit id (the default is a random
        ``new_id``), span and hashes carried for the compile receipt."""
        return EvidenceItem(
            id=self.id,
            source_id=self.source_uri or self.title or self.doc_key[:16],
            source_type="document",
            text=self.claim,
            span=self.span,
            section_path=list(self.section_path),
            relevance=relevance,
            authority=self.authority,
            provenance=0.9,  # byte-exact slice of a known source
            trust_level=self.trust_level,
            pinned=pinned,
            metadata={
                "lager": True,
                "eo_id": self.id,
                "doc_key": self.doc_key,
                "content_hash": self.content_hash,
                "eo_kind": self.kind,
                **({"corroborated_by": self.metadata["corroborated_by"]}
                   if "corroborated_by" in self.metadata else {}),
                # Tenant scope rides through to the runtime's isolation screen —
                # a list of every tenant that owns this (byte-identical) object.
                **({"tenant_ids": self.metadata["tenant_ids"]}
                   if "tenant_ids" in self.metadata else {}),
            },
        )


class EvidencePack(BaseModel):
    """The lazy loop's output: the minimum evidence acquired, with the full
    decision trace — every retrieval step is explainable after the fact."""

    query: str
    objects: list[EvidenceObject] = Field(default_factory=list)
    #: need text → covering EO ids ([] = uncovered)
    coverage: dict[str, list[str]] = Field(default_factory=dict)
    #: (eo_id_a, eo_id_b, basis) pairs the graph flags as contradicting
    contradictions: list[tuple[str, str, str]] = Field(default_factory=list)
    rounds: int = 0
    #: one entry per round: {round, added, gain, coverage, frontier_size}
    gain_trace: list[dict[str, Any]] = Field(default_factory=list)
    #: E0 empty frontier | E1 sufficient | E2 diminishing gain | E3 token budget
    #: | E4 max rounds
    exit_reason: str = ""
    sufficient: bool = False
    #: required needs left uncovered (drives downstream abstention)
    uncovered_needs: list[str] = Field(default_factory=list)
    token_cost: int = 0

    def as_evidence_items(self) -> list[EvidenceItem]:
        """Compiler-ready items; EOs that cover a required need ride pinned so
        the packet is guaranteed to carry the load-bearing evidence."""
        covering: set[str] = set()
        for ids in self.coverage.values():
            covering.update(ids)
        items: list[EvidenceItem] = []
        for rank, obj in enumerate(self.objects):
            relevance = max(0.35, 1.0 - 0.05 * rank)
            items.append(obj.as_evidence_item(relevance=relevance, pinned=obj.id in covering))
        return items

    def verify(self, documents_text: dict[str, str]) -> bool:
        """True iff every object re-derives from its source (keyed by
        ``doc_key``); a missing or tampered source fails."""
        for obj in self.objects:
            text = documents_text.get(obj.doc_key)
            if text is None or not obj.verify(text):
                return False
        return True
