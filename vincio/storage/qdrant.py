"""Qdrant vector index implementing the retrieval Index
protocol. Requires ``pip install "vincio[retrieval]"``."""

from __future__ import annotations

import uuid

from ..core.errors import StorageError
from ..core.types import Chunk
from ..retrieval.embeddings import Embedder
from ..retrieval.filters import FilterSpec, as_predicate
from ..retrieval.indexes import SearchHit, Where

__all__ = ["QdrantVectorIndex"]


def _point_id(chunk_id: str) -> str:
    """Qdrant requires UUID/int ids; derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vincio:{chunk_id}"))


class QdrantVectorIndex:
    name = "qdrant"

    def __init__(
        self,
        embedder: Embedder,
        *,
        url: str = "http://localhost:6333",
        collection: str = "vincio_chunks",
        api_key: str | None = None,
        quantization: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise StorageError(
                'Qdrant support requires: pip install "vincio[retrieval]"'
            ) from exc
        self.embedder = embedder
        self.collection = collection
        self.quantization = quantization
        self.client = QdrantClient(url=url, api_key=api_key)
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
                quantization_config=self._quantization_config(quantization),
            )

    @staticmethod
    def _quantization_config(quantization: str | None):
        """Map ``"scalar"``/``"binary"`` to Qdrant's native quantization config.

        Qdrant keeps full vectors on disk and the quantized copy in RAM, scoring
        the coarse copy first and rescoring with full precision — server-side
        two-stage retrieval, the same coarse→exact pattern as
        :class:`~vincio.retrieval.quantization.TwoStageIndex`. ``None`` leaves
        vectors unquantized.
        """
        if quantization is None:
            return None
        from qdrant_client.models import (
            BinaryQuantization,
            BinaryQuantizationConfig,
            ScalarQuantization,
            ScalarQuantizationConfig,
            ScalarType,
        )

        if quantization == "binary":
            return BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=True))
        if quantization == "scalar":
            return ScalarQuantization(
                scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
            )
        raise StorageError(f"unknown quantization {quantization!r}; use scalar|binary")

    def __len__(self) -> int:
        return self.client.count(self.collection).count

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        from qdrant_client.models import PointStruct

        vectors = await self.embedder.embed([c.text for c in chunks])
        points = [
            PointStruct(
                id=_point_id(chunk.id),
                vector=vector,
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=False)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    async def delete(self, chunk_ids: list[str]) -> int:
        from qdrant_client.models import PointIdsList

        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[_point_id(c) for c in chunk_ids]),
        )
        return len(chunk_ids)

    async def search(
        self, query: str, *, top_k: int = 10, where: Where | None = None
    ) -> list[SearchHit]:
        [vector] = await self.embedder.embed([query])
        # push a FilterSpec into Qdrant's native filter so selectivity is
        # applied server-side — fetch exactly top_k (no over-fetch under-fill)
        # and never round-trip other tenants' rows. A legacy callable still
        # post-filters client-side over a 4x over-fetch.
        native = where.to_qdrant() if isinstance(where, FilterSpec) else None
        predicate = None if native is not None else as_predicate(where)
        fetch = top_k if (native is not None or where is None) else top_k * 4
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=fetch,
            with_payload=True,
            query_filter=native,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            chunk = Chunk.model_validate(point.payload)
            if predicate is not None and not predicate(chunk):
                continue
            hits.append(SearchHit(chunk=chunk, score=float(point.score), source=self.name))
            if len(hits) >= top_k:
                break
        return hits
