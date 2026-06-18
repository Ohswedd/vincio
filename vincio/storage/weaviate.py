"""Weaviate vector index implementing the retrieval Index protocol.

Requires ``pip install "vincio[weaviate]"``. Vincio supplies the vectors (so
one embedder powers every backend); the full chunk rides as a JSON property
and is rehydrated on search. A pre-built client can be injected (``client=``)
for offline tests — when injected, the SDK is not imported.
"""

from __future__ import annotations

import uuid
from typing import Any

from ..core.errors import StorageError
from ..core.types import Chunk
from ..retrieval.embeddings import Embedder, embed_texts
from ..retrieval.filters import FilterSpec, as_predicate, flat_filter_fields
from ..retrieval.indexes import SearchHit, Where

__all__ = ["WeaviateVectorIndex"]


def _object_uuid(chunk_id: str) -> str:
    """Weaviate requires UUID object ids; derive a stable one from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vincio:{chunk_id}"))


class WeaviateVectorIndex:
    name = "weaviate"

    def __init__(
        self,
        embedder: Embedder,
        *,
        url: str = "http://localhost:8080",
        collection: str = "VincioChunks",
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.embedder = embedder
        self.collection_name = collection
        self.client = client if client is not None else self._connect(url, api_key)
        if not self.client.collections.exists(collection):
            self.client.collections.create(collection)
        self.collection = self.client.collections.get(collection)

    def _connect(self, url: str, api_key: str | None) -> Any:
        try:
            import weaviate
            from weaviate.classes.init import Auth
        except ImportError as exc:
            raise StorageError('Weaviate support requires: pip install "vincio[weaviate]"') from exc
        auth = Auth.api_key(api_key) if api_key else None
        return weaviate.connect_to_custom(
            http_host=url, http_secure=url.startswith("https"), auth_credentials=auth
        )

    def __len__(self) -> int:
        return int(self.collection.aggregate.over_all(total_count=True).total_count)

    def _properties(self, chunk: Chunk) -> dict[str, Any]:
        # persist flat filterable fields alongside the blob so a compiled
        # FilterSpec matches server-side (Weaviate properties).
        return {"json": chunk.model_dump_json(), **flat_filter_fields(chunk)}

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = await embed_texts(self.embedder, [c.text for c in chunks], input_type="document")
        for chunk, vector in zip(chunks, vectors, strict=False):
            self.collection.data.insert(
                properties=self._properties(chunk),
                vector=list(vector),
                uuid=_object_uuid(chunk.id),
            )

    async def delete(self, chunk_ids: list[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            try:
                self.collection.data.delete_by_id(_object_uuid(chunk_id))
                removed += 1
            except Exception:  # noqa: BLE001 - id absent / already deleted
                continue
        return removed

    async def search(
        self, query: str, *, top_k: int = 10, where: Where | None = None
    ) -> list[SearchHit]:
        [vector] = await embed_texts(self.embedder, [query], input_type="query")
        # push a FilterSpec into Weaviate's `where` filter server-side; the
        # as_predicate net guarantees correctness regardless. Fetch top_k when a
        # FilterSpec is pushed down (no over-fetch); a callable still post-filters.
        native = where.to_weaviate() if isinstance(where, FilterSpec) else None
        predicate = as_predicate(where)
        fetch = top_k if (native is not None or where is None) else top_k * 4
        kwargs: dict[str, Any] = {"near_vector": list(vector), "limit": fetch}
        if native is not None:
            kwargs["filters"] = native
        try:  # request the distance metadata when the real SDK is present
            from weaviate.classes.query import MetadataQuery

            kwargs["return_metadata"] = MetadataQuery(distance=True)
        except Exception:  # noqa: BLE001 - injected fake / SDK absent
            pass
        response = self.collection.query.near_vector(**kwargs)
        hits: list[SearchHit] = []
        for obj in response.objects:
            chunk = Chunk.model_validate_json(obj.properties["json"])
            if predicate is not None and not predicate(chunk):
                continue
            distance = getattr(getattr(obj, "metadata", None), "distance", None)
            score = (1.0 - float(distance)) if distance is not None else 0.0
            hits.append(SearchHit(chunk=chunk, score=score, source=self.name))
            if len(hits) >= top_k:
                break
        return hits
