"""LanceDB vector index implementing the retrieval Index protocol.

Requires ``pip install "vincio[lancedb]"``. LanceDB is an embedded, on-disk
vector store (no server); the table is created on first write and cosine
distance powers search.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import StorageError
from ..core.types import Chunk
from ..retrieval.embeddings import Embedder
from ..retrieval.filters import as_predicate
from ..retrieval.indexes import SearchHit, Where

__all__ = ["LanceDBVectorIndex"]


class LanceDBVectorIndex:
    name = "lancedb"

    def __init__(
        self,
        embedder: Embedder,
        *,
        uri: str = ".vincio/lancedb",
        table: str = "vincio_chunks",
        connection: Any | None = None,
    ) -> None:
        try:
            import lancedb
        except ImportError as exc:
            raise StorageError('LanceDB support requires: pip install "vincio[lancedb]"') from exc
        self.embedder = embedder
        self.table_name = table
        self.db = connection or lancedb.connect(uri)
        self._table = self.db.open_table(table) if table in self.db.table_names() else None

    def __len__(self) -> int:
        return self._table.count_rows() if self._table is not None else 0

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
        vectors = await self.embedder.embed([c.text for c in chunks])
        rows = [self._row(c, v) for c, v in zip(chunks, vectors, strict=False)]
        if self._table is None:
            self._table = self.db.create_table(self.table_name, data=rows)
        else:
            self._table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows)

    async def delete(self, chunk_ids: list[str]) -> int:
        if not chunk_ids or self._table is None:
            return 0
        quoted = ", ".join("'" + cid.replace("'", "''") + "'" for cid in chunk_ids)
        self._table.delete(f"id IN ({quoted})")
        return len(chunk_ids)

    async def search(
        self, query: str, *, top_k: int = 10, where: Where | None = None
    ) -> list[SearchHit]:
        if self._table is None:
            return []
        [vector] = await self.embedder.embed([query])
        predicate = as_predicate(where)
        fetch = top_k * 4 if where is not None else top_k
        rows = self._table.search(list(vector)).metric("cosine").limit(fetch).to_list()
        hits: list[SearchHit] = []
        for row in rows:
            chunk = Chunk.model_validate_json(row["json"])
            if predicate is not None and not predicate(chunk):
                continue
            score = 1.0 - float(row.get("_distance", 0.0))
            hits.append(SearchHit(chunk=chunk, score=score, source=self.name))
            if len(hits) >= top_k:
                break
        return hits
