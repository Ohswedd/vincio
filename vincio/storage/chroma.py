"""Chroma vector index implementing the retrieval Index protocol.

Requires ``pip install "vincio[chroma]"``. Embeddings are computed by Vincio's
embedder (so the same embedder powers every backend); the full chunk is stored
as a JSON metadata field and rehydrated on search.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import StorageError
from ..core.types import Chunk
from ..retrieval.embeddings import Embedder
from ..retrieval.filters import as_predicate
from ..retrieval.indexes import SearchHit, Where

__all__ = ["ChromaVectorIndex"]


class ChromaVectorIndex:
    name = "chroma"

    def __init__(
        self,
        embedder: Embedder,
        *,
        collection: str = "vincio_chunks",
        path: str | None = None,
        client: Any | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise StorageError('Chroma support requires: pip install "vincio[chroma]"') from exc
        self.embedder = embedder
        if client is not None:
            self.client = client
        elif path is not None:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def __len__(self) -> int:
        return self.collection.count()

    def _metadata(self, chunk: Chunk) -> dict[str, Any]:
        # Chroma metadata values must be scalars; keep filterable columns flat
        # and stash the full chunk as JSON for faithful rehydration.
        return {
            "_chunk": chunk.model_dump_json(),
            "document_id": chunk.document_id,
            "tenant_id": chunk.tenant_id or "",
            "kind": chunk.kind,
        }

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = await self.embedder.embed([c.text for c in chunks])
        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=[list(v) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[self._metadata(c) for c in chunks],
        )

    async def delete(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        self.collection.delete(ids=chunk_ids)
        return len(chunk_ids)

    async def search(
        self, query: str, *, top_k: int = 10, where: Where | None = None
    ) -> list[SearchHit]:
        [vector] = await self.embedder.embed([query])
        predicate = as_predicate(where)
        fetch = top_k * 4 if where is not None else top_k
        response = self.collection.query(
            query_embeddings=[list(vector)], n_results=fetch, include=["metadatas", "distances"]
        )
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        hits: list[SearchHit] = []
        for metadata, distance in zip(metadatas, distances, strict=False):
            chunk = Chunk.model_validate_json(metadata["_chunk"])
            if predicate is not None and not predicate(chunk):
                continue
            hits.append(SearchHit(chunk=chunk, score=1.0 - float(distance), source=self.name))
            if len(hits) >= top_k:
                break
        return hits
