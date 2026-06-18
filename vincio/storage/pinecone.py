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
from ..retrieval.filters import FilterSpec, as_predicate, flat_filter_fields
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
        self.embedder = embedder
        self.namespace = namespace
        # Lazy-import the SDK only when building a real client (like every other
        # adapter), so an injected client works without the package installed.
        if client is None:
            try:
                from pinecone import Pinecone
            except ImportError as exc:
                raise StorageError(
                    'Pinecone support requires: pip install "vincio[pinecone]"'
                ) from exc
            client = Pinecone(api_key=api_key)
        self._pc = client
        existing = {idx["name"] for idx in self._pc.list_indexes()}
        if index_name not in existing:
            from pinecone import ServerlessSpec

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
        # flat filterable fields become Pinecone metadata so a compiled
        # FilterSpec matches server-side.
        return {"_chunk": chunk.model_dump_json(), **flat_filter_fields(chunk)}

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
        # push a FilterSpec down as a Pinecone metadata filter (its real
        # wire format); the as_predicate net guarantees correctness regardless.
        native = where.to_pinecone() if isinstance(where, FilterSpec) else None
        predicate = as_predicate(where)
        fetch = top_k if (native is not None or where is None) else top_k * 4
        query_kwargs: dict[str, Any] = {
            "vector": list(vector),
            "top_k": fetch,
            "include_metadata": True,
            "namespace": self.namespace,
        }
        if native is not None:
            query_kwargs["filter"] = native
        response = self.index.query(**query_kwargs)
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
