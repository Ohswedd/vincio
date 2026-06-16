"""Milvus vector index implementing the retrieval Index protocol.

Requires ``pip install "vincio[milvus]"``. Uses the simple ``MilvusClient``:
a string primary key (the chunk id), a float vector, and the full chunk in a
dynamic ``json`` field rehydrated on search. A pre-built client can be
injected (``client=``) for offline tests — when injected, the SDK is not
imported.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import StorageError
from ..core.types import Chunk
from ..retrieval.embeddings import Embedder, embed_texts
from ..retrieval.indexes import SearchFilter, SearchHit

__all__ = ["MilvusVectorIndex"]


class MilvusVectorIndex:
    name = "milvus"

    def __init__(
        self,
        embedder: Embedder,
        *,
        uri: str = "http://localhost:19530",
        collection: str = "vincio_chunks",
        token: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.embedder = embedder
        self.collection = collection
        self.client = client if client is not None else self._connect(uri, token)
        if not self.client.has_collection(collection):
            self.client.create_collection(
                collection_name=collection,
                dimension=embedder.dim,
                metric_type="COSINE",
                id_type="string",
                max_length=512,
                auto_id=False,
            )

    def _connect(self, uri: str, token: str | None) -> Any:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise StorageError('Milvus support requires: pip install "vincio[milvus]"') from exc
        return MilvusClient(uri=uri, token=token or "")

    def __len__(self) -> int:
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))

    def _row(self, chunk: Chunk, vector: list[float]) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "vector": list(vector),
            "json": chunk.model_dump_json(),
            "document_id": chunk.document_id,
            "tenant_id": chunk.tenant_id or "",
            "kind": chunk.kind,
        }

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = await embed_texts(self.embedder, [c.text for c in chunks], input_type="document")
        rows = [self._row(c, v) for c, v in zip(chunks, vectors, strict=False)]
        self.client.insert(collection_name=self.collection, data=rows)

    async def delete(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        self.client.delete(collection_name=self.collection, ids=list(chunk_ids))
        return len(chunk_ids)

    async def search(
        self, query: str, *, top_k: int = 10, where: SearchFilter | None = None
    ) -> list[SearchHit]:
        [vector] = await embed_texts(self.embedder, [query], input_type="query")
        fetch = top_k * 4 if where is not None else top_k
        results = self.client.search(
            collection_name=self.collection,
            data=[list(vector)],
            limit=fetch,
            output_fields=["json"],
        )
        hits: list[SearchHit] = []
        for hit in results[0] if results else []:
            entity = hit.get("entity") or {}
            chunk = Chunk.model_validate_json(entity["json"])
            if where is not None and not where(chunk):
                continue
            # Milvus returns cosine similarity directly for the COSINE metric.
            hits.append(SearchHit(chunk=chunk, score=float(hit.get("distance", 0.0)), source=self.name))
            if len(hits) >= top_k:
                break
        return hits
