"""Elasticsearch and OpenSearch vector indexes implementing the retrieval
Index protocol.

Requires ``pip install "vincio[elasticsearch]"`` (Elasticsearch) or
``pip install "vincio[opensearch]"`` (OpenSearch). Both store a dense vector
plus the full chunk as a JSON field and rehydrate it on search; kNN search
runs against the engine's native vector query. A pre-built client can be
injected (``client=``) for offline tests — when injected, the SDK is not
imported, so the round trip is exercised without the dependency.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import StorageError
from ..core.types import Chunk
from ..retrieval.embeddings import Embedder, embed_texts
from ..retrieval.filters import FilterSpec, as_predicate, flat_filter_fields
from ..retrieval.indexes import SearchHit, Where

__all__ = ["ElasticsearchVectorIndex", "OpenSearchVectorIndex"]


class _ElasticLikeIndex:
    """Shared add/delete/hydrate logic for the Elasticsearch-family engines.

    Subclasses provide the client, the index/mapping creation, and the kNN
    query shape (Elasticsearch 8 and OpenSearch differ on both)."""

    name = "elasticsearch"
    _extra = "elasticsearch"

    def __init__(
        self,
        embedder: Embedder,
        *,
        url: str = "http://localhost:9200",
        index: str = "vincio_chunks",
        api_key: str | None = None,
        client: Any | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.client = client if client is not None else self._connect(url, api_key, client_kwargs)
        self._ensure_index()

    # -- engine-specific seams -------------------------------------------------

    def _connect(self, url: str, api_key: str | None, kwargs: dict[str, Any]) -> Any:
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise StorageError(
                f'Elasticsearch support requires: pip install "vincio[{self._extra}]"'
            ) from exc
        return Elasticsearch(url, api_key=api_key, **kwargs)

    def _ensure_index(self) -> None:
        if not self.client.indices.exists(index=self.index):
            self.client.indices.create(index=self.index, mappings=self._mappings())

    def _mappings(self) -> dict[str, Any]:
        return {
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": self.embedder.dim,
                    "index": True,
                    "similarity": "cosine",
                },
                "json": {"type": "text", "index": False},
                "document_id": {"type": "keyword"},
                "tenant_id": {"type": "keyword"},
                "kind": {"type": "keyword"},
            }
        }

    def _knn_search(self, vector: list[float], fetch: int, native_filter: Any = None) -> Any:
        knn: dict[str, Any] = {
            "field": "vector",
            "query_vector": vector,
            "k": fetch,
            "num_candidates": max(fetch, 50),
        }
        # a kNN `filter` is applied server-side before the top_k cut.
        if native_filter is not None:
            knn["filter"] = native_filter
        return self.client.search(index=self.index, knn=knn, size=fetch)

    # -- Index protocol --------------------------------------------------------

    def __len__(self) -> int:
        return int(self.client.count(index=self.index)["count"])

    def _document(self, chunk: Chunk, vector: list[float]) -> dict[str, Any]:
        # persist flat filterable fields so a compiled FilterSpec `bool`
        # query matches server-side (dynamic-mapped keywords).
        return {"vector": list(vector), "json": chunk.model_dump_json(), **flat_filter_fields(chunk)}

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = await embed_texts(self.embedder, [c.text for c in chunks], input_type="document")
        for chunk, vector in zip(chunks, vectors, strict=False):
            self.client.index(
                index=self.index, id=chunk.id, document=self._document(chunk, vector), refresh=True
            )

    async def delete(self, chunk_ids: list[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            try:
                self.client.delete(index=self.index, id=chunk_id, refresh=True)
                removed += 1
            except Exception:  # noqa: BLE001 - id absent / already deleted
                continue
        return removed

    async def search(
        self, query: str, *, top_k: int = 10, where: Where | None = None
    ) -> list[SearchHit]:
        [vector] = await embed_texts(self.embedder, [query], input_type="query")
        # push a FilterSpec down as an Elasticsearch `bool` filter (its real
        # wire format); the as_predicate net guarantees correctness regardless.
        native = where.to_elasticsearch() if isinstance(where, FilterSpec) else None
        predicate = as_predicate(where)
        fetch = top_k if (native is not None or where is None) else top_k * 4
        response = self._knn_search(list(vector), fetch, native)
        hits: list[SearchHit] = []
        for hit in response["hits"]["hits"]:
            chunk = Chunk.model_validate_json(hit["_source"]["json"])
            if predicate is not None and not predicate(chunk):
                continue
            hits.append(SearchHit(chunk=chunk, score=float(hit.get("_score", 0.0)), source=self.name))
            if len(hits) >= top_k:
                break
        return hits


class ElasticsearchVectorIndex(_ElasticLikeIndex):
    """Elasticsearch 8 dense-vector index (top-level ``knn`` retrieval)."""

    name = "elasticsearch"
    _extra = "elasticsearch"


class OpenSearchVectorIndex(_ElasticLikeIndex):
    """OpenSearch k-NN index (``knn_vector`` field, ``knn`` query)."""

    name = "opensearch"
    _extra = "opensearch"

    def _connect(self, url: str, api_key: str | None, kwargs: dict[str, Any]) -> Any:
        try:
            from opensearchpy import OpenSearch
        except ImportError as exc:
            raise StorageError(
                'OpenSearch support requires: pip install "vincio[opensearch]"'
            ) from exc
        if api_key is not None:
            kwargs.setdefault("api_key", api_key)
        return OpenSearch(url, **kwargs)

    def _ensure_index(self) -> None:
        if not self.client.indices.exists(index=self.index):
            self.client.indices.create(
                index=self.index,
                body={"settings": {"index": {"knn": True}}, "mappings": self._mappings()},
            )

    def _mappings(self) -> dict[str, Any]:
        return {
            "properties": {
                "vector": {"type": "knn_vector", "dimension": self.embedder.dim},
                "json": {"type": "text", "index": False},
                "document_id": {"type": "keyword"},
                "tenant_id": {"type": "keyword"},
                "kind": {"type": "keyword"},
            }
        }

    def _knn_search(self, vector: list[float], fetch: int, native_filter: Any = None) -> Any:
        knn_vector: dict[str, Any] = {"vector": vector, "k": fetch}
        if native_filter is not None:
            knn_vector["filter"] = native_filter  # OpenSearch kNN server-side filter
        return self.client.search(
            index=self.index,
            body={"size": fetch, "query": {"knn": {"vector": knn_vector}}},
        )
