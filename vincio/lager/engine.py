"""The LAGER facade — ingest → retrieve (lazily) → answer, one object.

Every module is injectable: the extractor (any :class:`ClaimExtractor`), the
embedder (None ≙ pure-stdlib deterministic), the planner, and the controller
options — so each stage can be replaced and benchmarked independently, which is
the point of the architecture.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import LagerError
from ..core.types import Document
from .answer import LagerAnswer, generate_answer
from .controller import LazyOptions, LazyRetriever
from .extract import ClaimExtractor, DeterministicClaimExtractor, assign_fallback_terms
from .graph import EvidenceGraph
from .index import EvidenceIndex
from .objects import EvidenceObject, EvidencePack, document_key
from .planner import QueryPlanner

__all__ = ["LagerEngine"]


class LagerEngine:
    """Documents in, reasoning-driven evidence out.

    ``ingest`` extracts Evidence Objects (byte-exact spans, content-derived
    ids), builds the typed graph, and indexes every retrieval signal;
    ``retrieve`` runs the lazy loop; ``answer`` adds one grounded, cited,
    offline-verifiable completion on top.
    """

    def __init__(
        self,
        *,
        extractor: ClaimExtractor | None = None,
        embedder: Any | None = None,
        planner: QueryPlanner | None = None,
        options: LazyOptions | None = None,
    ) -> None:
        self.extractor: ClaimExtractor = extractor or DeterministicClaimExtractor()
        self.planner = planner or QueryPlanner()
        self.options = options or LazyOptions()
        self.graph = EvidenceGraph()
        self.index = EvidenceIndex(embedder=embedder)
        self.objects: list[EvidenceObject] = []
        self._documents_text: dict[str, str] = {}  # doc_key → canonical source

    def __len__(self) -> int:
        return len(self.objects)

    # -- ingestion -------------------------------------------------------------------

    def ingest(self, documents: list[Document]) -> int:
        """Extract, relate, and index *documents*; returns the number of
        Evidence Objects added. Idempotent per document content."""
        added: list[EvidenceObject] = []
        for document in documents:
            key = document_key(document.text or "")
            if key in self._documents_text:
                # Same bytes already ingested. Content-derived identity means one
                # Evidence Object stands for byte-identical documents, so a
                # DIFFERENT tenant that also owns this document is recorded as a
                # co-owner — otherwise a later tenant-scoped run would drop
                # evidence that tenant legitimately holds.
                tenant = getattr(document, "tenant_id", None)
                if tenant:
                    for obj in (*self.objects, *added):
                        if obj.doc_key == key:
                            owners = obj.metadata.setdefault("tenant_ids", [])
                            if tenant not in owners:
                                owners.append(tenant)
                                owners.sort()
                continue
            self._documents_text[key] = document.text or ""
            added.extend(self.extractor.extract(document))
        if not added:
            return 0
        self.objects.extend(added)
        assign_fallback_terms(self.objects)  # corpus-level IDF pass
        self.graph.build(self.objects)
        self.index = EvidenceIndex(embedder=self.index.embedder)
        self.index.add(self.objects)
        return len(added)

    # -- retrieval -------------------------------------------------------------------

    def retrieve(self, query: str) -> EvidencePack:
        """The lazy loop: incremental, graph-guided, self-terminating."""
        if not self.objects:
            raise LagerError(
                "nothing ingested: call engine.ingest(documents) before retrieve()"
            )
        retriever = LazyRetriever(
            self.index, self.graph, options=self.options, planner=self.planner
        )
        return retriever.retrieve(query)

    # -- answering -------------------------------------------------------------------

    async def answer(
        self, query: str, *, provider: Any, model: str
    ) -> LagerAnswer:
        """Retrieve lazily, then one grounded completion with [eo:…] citations."""
        pack = self.retrieve(query)
        return await generate_answer(query, pack, provider=provider, model=model)

    # -- verification ----------------------------------------------------------------

    def verify(self, pack: EvidencePack) -> bool:
        """Every pack object re-derives byte-for-byte from its ingested source."""
        return pack.verify(self._documents_text)

    @property
    def documents_text(self) -> dict[str, str]:
        """doc_key → source text, for offline verification elsewhere."""
        return dict(self._documents_text)
