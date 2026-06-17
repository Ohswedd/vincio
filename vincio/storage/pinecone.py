"""Pinecone vector index implementing the retrieval Index protocol.

Requires ``pip install "vincio[pinecone]"``. Serverless indexes are created on
demand; the full chunk rides along as JSON metadata so search rehydrates the
exact :class:`~vincio.core.types.Chunk`.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import StorageError
from ..core.types import Chunk
from ..retrieval.embeddings import Embedder
from ..retrieval.filters import as_predicate
from ..retrieval.indexes import SearchHit, Where

__all__ = ["PineconeVectorIndex"]


class PineconeVectorIndex:
    name = "pinecone"

    def __init__(
        self,
        embedder: Embedder,
        *,
        index_name: str = "vincio-chunks",
        api_key: str | None = None,
        namespace: str = "",
        cloud: str = "aws",
        region: str = "us-east-1",
        client: Any | None = None,
    ) -> None:
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise StorageError('Pinecone support requires: pip install "vincio[pinecone]"') from exc
        self.embedder = embedder
        self.namespace = namespace
        self._pc = client or Pinecone(api_key=api_key)
        existing = {idx["name"] for idx in self._pc.list_indexes()}
        if index_name not in existing:
            self._pc.create_index(
                name=index_name,
                dimension=embedder.dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self.index = self._pc.Index(index_name)

    def __len__(self) -> int:
        stats = self.index.describe_index_stats()
        return int(stats.get("total_vector_count", 0))

    def _metadata(self, chunk: Chunk) -> dict[str, Any]:
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
        self.index.upsert(
            vectors=[
                {"id": chunk.id, "values": list(vector), "metadata": self._metadata(chunk)}
                for chunk, vector in zip(chunks, vectors, strict=False)
            ],
            namespace=self.namespace,
        )

    async def delete(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        self.index.delete(ids=chunk_ids, namespace=self.namespace)
        return len(chunk_ids)

    async def search(
        self, query: str, *, top_k: int = 10, where: Where | None = None
    ) -> list[SearchHit]:
        [vector] = await self.embedder.embed([query])
        predicate = as_predicate(where)
        fetch = top_k * 4 if where is not None else top_k
        response = self.index.query(
            vector=list(vector), top_k=fetch, include_metadata=True, namespace=self.namespace
        )
        hits: list[SearchHit] = []
        for match in response.get("matches") or []:
            metadata = match.get("metadata") or {}
            chunk = Chunk.model_validate_json(metadata["_chunk"])
            if predicate is not None and not predicate(chunk):
                continue
            hits.append(SearchHit(chunk=chunk, score=float(match.get("score", 0.0)), source=self.name))
            if len(hits) >= top_k:
                break
        return hits
