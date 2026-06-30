"""Vespa vector index implementing the retrieval Index protocol.

Requires ``pip install "vincio[vespa]"`` and a deployed Vespa application whose
schema has an ``embedding`` tensor field and a rank profile exposing
``closeness`` over a ``nearestNeighbor`` query (the conventional ANN setup).
The full chunk rides in a ``json`` field and is rehydrated on search. A
pre-built ``Vespa`` application can be injected (``app=``) for offline tests —
when injected, the SDK is not imported.
"""

from __future__ import annotations

from typing import Any

from ..core.diagnostics import note_suppressed
from ..core.errors import StorageError
from ..core.types import Chunk
from ..retrieval.embeddings import Embedder, embed_texts
from ..retrieval.filters import as_predicate
from ..retrieval.indexes import SearchHit, Where

__all__ = ["VespaVectorIndex"]


class VespaVectorIndex:
    name = "vespa"

    def __init__(
        self,
        embedder: Embedder,
        *,
        url: str = "http://localhost",
        port: int = 8080,
        schema: str = "vincio_chunk",
        rank_profile: str = "closeness",
        app: Any | None = None,
    ) -> None:
        self.embedder = embedder
        self.schema = schema
        self.rank_profile = rank_profile
        self.app = app if app is not None else self._connect(url, port)
        self._count = 0  # local count for the optimistic path; refined by Vespa when available

    def _connect(self, url: str, port: int) -> Any:
        try:
            from vespa.application import Vespa
        except ImportError as exc:
            raise StorageError('Vespa support requires: pip install "vincio[vespa]"') from exc
        return Vespa(url=url, port=port)

    def __len__(self) -> int:
        try:
            response = self.app.query(
                body={"yql": f"select * from sources {self.schema} where true", "hits": 0}
            )
            total = response.json["root"]["fields"]["totalCount"]
            return int(total)
        except Exception:
            note_suppressed("storage.vespa.count")
            return self._count

    def _fields(self, chunk: Chunk, vector: list[float]) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "embedding": list(vector),
            "json": chunk.model_dump_json(),
            "document_id": chunk.document_id,
            "tenant_id": chunk.tenant_id or "",
            "kind": chunk.kind,
        }

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = await embed_texts(self.embedder, [c.text for c in chunks], input_type="document")
        for chunk, vector in zip(chunks, vectors, strict=False):
            self.app.feed_data_point(
                schema=self.schema, data_id=chunk.id, fields=self._fields(chunk, vector)
            )
        self._count += len(chunks)

    async def delete(self, chunk_ids: list[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            try:
                self.app.delete_data(schema=self.schema, data_id=chunk_id)
                removed += 1
            except Exception:
                note_suppressed("storage.vespa.delete")
                continue
        self._count = max(0, self._count - removed)
        return removed

    async def search(
        self, query: str, *, top_k: int = 10, where: Where | None = None
    ) -> list[SearchHit]:
        [vector] = await embed_texts(self.embedder, [query], input_type="query")
        predicate = as_predicate(where)
        fetch = top_k * 4 if where is not None else top_k
        response = self.app.query(
            body={
                "yql": (
                    f"select * from sources {self.schema} "
                    f"where ({{targetHits:{fetch}}}nearestNeighbor(embedding, q))"
                ),
                "input.query(q)": list(vector),
                "ranking.profile": self.rank_profile,
                "hits": fetch,
            }
        )
        hits: list[SearchHit] = []
        for record in response.hits:
            fields = record.get("fields", {})
            chunk = Chunk.model_validate_json(fields["json"])
            if predicate is not None and not predicate(chunk):
                continue
            hits.append(SearchHit(chunk=chunk, score=float(record.get("relevance", 0.0)), source=self.name))
            if len(hits) >= top_k:
                break
        return hits
