"""Postgres metadata store + pgvector index.
Requires ``pip install "vincio[postgres]"``."""

from __future__ import annotations

import json
from typing import Any

from ..core.errors import StorageError
from ..core.types import Chunk
from ..retrieval.embeddings import Embedder
from ..retrieval.filters import FilterSpec, as_predicate
from ..retrieval.indexes import SearchHit, Where

__all__ = ["PostgresMetadataStore", "PgVectorIndex"]


def _connect(dsn: str):
    try:
        import psycopg
    except ImportError as exc:
        raise StorageError(
            'Postgres support requires: pip install "vincio[postgres]"'
        ) from exc
    return psycopg.connect(dsn, autocommit=True)


_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS vincio_records (
  kind TEXT NOT NULL,
  id TEXT NOT NULL,
  json JSONB NOT NULL,
  PRIMARY KEY (kind, id)
);
CREATE INDEX IF NOT EXISTS idx_vincio_records_kind ON vincio_records(kind);
"""


class PostgresMetadataStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn = _connect(dsn)
        self._apool: Any | None = None  # lazily-opened psycopg3 AsyncConnectionPool
        with self._conn.cursor() as cursor:
            cursor.execute(_TABLES_SQL)

    def close(self) -> None:
        self._conn.close()

    # -- async-first fast path (psycopg3 AsyncConnectionPool) ------------------

    async def _pool(self) -> Any:
        """Lazily open a psycopg3 async connection pool (created inside the
        running loop). Keeps DB I/O off the event loop without per-call threads."""
        if self._apool is None:
            try:
                from psycopg_pool import AsyncConnectionPool
            except ImportError as exc:  # pragma: no cover - optional dep
                raise StorageError(
                    'Async Postgres support requires: pip install "vincio[postgres]" '
                    "(psycopg[pool])"
                ) from exc
            self._apool = AsyncConnectionPool(self._dsn, open=False, kwargs={"autocommit": True})
            await self._apool.open()
        return self._apool

    async def aclose(self) -> None:
        if self._apool is not None:
            await self._apool.close()
            self._apool = None

    async def asave(self, kind: str, record: dict[str, Any]) -> None:
        record_id = record.get("id")
        if not record_id:
            raise StorageError(f"record for {kind!r} has no id")
        pool = await self._pool()
        async with pool.connection() as conn, conn.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO vincio_records (kind, id, json) VALUES (%s, %s, %s) "
                "ON CONFLICT (kind, id) DO UPDATE SET json = EXCLUDED.json",
                (kind, str(record_id), json.dumps(record, default=str)),
            )

    async def aget(self, kind: str, record_id: str) -> dict[str, Any] | None:
        pool = await self._pool()
        async with pool.connection() as conn, conn.cursor() as cursor:
            await cursor.execute(
                "SELECT json FROM vincio_records WHERE kind = %s AND id = %s", (kind, record_id)
            )
            row = await cursor.fetchone()
        return row[0] if row else None

    async def aquery(
        self, kind: str, *, where: dict[str, Any] | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        sql = "SELECT json FROM vincio_records WHERE kind = %s"
        params: list[Any] = [kind]
        for key, value in (where or {}).items():
            sql += " AND json->>%s = %s"
            params.extend([key, str(value)])
        sql += " ORDER BY id DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        pool = await self._pool()
        async with pool.connection() as conn, conn.cursor() as cursor:
            await cursor.execute(sql, params)
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def adelete(self, kind: str, record_id: str) -> bool:
        pool = await self._pool()
        async with pool.connection() as conn, conn.cursor() as cursor:
            await cursor.execute(
                "DELETE FROM vincio_records WHERE kind = %s AND id = %s", (kind, record_id)
            )
            return cursor.rowcount > 0

    async def acount(self, kind: str) -> int:
        pool = await self._pool()
        async with pool.connection() as conn, conn.cursor() as cursor:
            await cursor.execute("SELECT COUNT(*) FROM vincio_records WHERE kind = %s", (kind,))
            row = await cursor.fetchone()
        return row[0] if row else 0

    def save(self, kind: str, record: dict[str, Any]) -> None:
        record_id = record.get("id")
        if not record_id:
            raise StorageError(f"record for {kind!r} has no id")
        with self._conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO vincio_records (kind, id, json) VALUES (%s, %s, %s) "
                "ON CONFLICT (kind, id) DO UPDATE SET json = EXCLUDED.json",
                (kind, str(record_id), json.dumps(record, default=str)),
            )

    def get(self, kind: str, record_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "SELECT json FROM vincio_records WHERE kind = %s AND id = %s", (kind, record_id)
            )
            row = cursor.fetchone()
        return row[0] if row else None

    def query(
        self, kind: str, *, where: dict[str, Any] | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        sql = "SELECT json FROM vincio_records WHERE kind = %s"
        params: list[Any] = [kind]
        for key, value in (where or {}).items():
            sql += " AND json->>%s = %s"
            params.extend([key, str(value)])
        sql += " ORDER BY id DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        with self._conn.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    def delete(self, kind: str, record_id: str) -> bool:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM vincio_records WHERE kind = %s AND id = %s", (kind, record_id)
            )
            return cursor.rowcount > 0

    def count(self, kind: str) -> int:
        with self._conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM vincio_records WHERE kind = %s", (kind,))
            (count,) = cursor.fetchone()
        return count


class PgVectorIndex:
    """pgvector-backed dense index implementing the retrieval Index protocol."""

    name = "pgvector"

    def __init__(self, dsn: str, embedder: Embedder, *, table: str = "vincio_chunks") -> None:
        self.embedder = embedder
        self.table = table
        self._conn = _connect(dsn)
        with self._conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                  id TEXT PRIMARY KEY,
                  document_id TEXT,
                  tenant_id TEXT,
                  kind TEXT,
                  json JSONB NOT NULL,
                  embedding vector({self.embedder.dim})
                )
                """
            )
            # 2.0: GIN index on the chunk jsonb so FilterSpec predicates pushed
            # down as `json ->> ...` / `json -> 'metadata' ->> ...` are indexed.
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_json_gin ON {self.table} USING GIN (json)"
            )

    def __len__(self) -> int:
        with self._conn.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self.table}")
            (count,) = cursor.fetchone()
        return count

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = await self.embedder.embed([c.text for c in chunks])
        with self._conn.cursor() as cursor:
            for chunk, vector in zip(chunks, vectors, strict=False):
                cursor.execute(
                    f"INSERT INTO {self.table} (id, document_id, tenant_id, kind, json, embedding) "
                    "VALUES (%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT (id) DO UPDATE SET json = EXCLUDED.json, embedding = EXCLUDED.embedding",
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.tenant_id,
                        chunk.kind,
                        chunk.model_dump_json(),
                        str(vector),
                    ),
                )

    async def delete(self, chunk_ids: list[str]) -> int:
        removed = 0
        with self._conn.cursor() as cursor:
            for chunk_id in chunk_ids:
                cursor.execute(f"DELETE FROM {self.table} WHERE id = %s", (chunk_id,))
                removed += cursor.rowcount
        return removed

    async def search(
        self, query: str, *, top_k: int = 10, where: Where | None = None
    ) -> list[SearchHit]:
        [vector] = await self.embedder.embed([query])
        # 2.0: push a FilterSpec into the SQL WHERE over the jsonb chunk column
        # so selectivity is applied server-side (GIN-indexed) — fetch exactly
        # top_k, no over-fetch under-fill, and other tenants' rows never leave
        # Postgres. A legacy callable still post-filters over a 4x over-fetch.
        where_sql = "TRUE"
        where_params: list[str] = []
        predicate = None
        if isinstance(where, FilterSpec):
            where_sql, where_params = where.to_sql_where(column="json")
            fetch = top_k
        else:
            predicate = as_predicate(where)
            fetch = top_k * 4 if where is not None else top_k
        with self._conn.cursor() as cursor:
            cursor.execute(
                f"SELECT json, 1 - (embedding <=> %s::vector) AS score FROM {self.table} "
                f"WHERE {where_sql} "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (str(vector), *where_params, str(vector), fetch),
            )
            rows = cursor.fetchall()
        hits: list[SearchHit] = []
        for payload, score in rows:
            chunk = Chunk.model_validate(payload)
            if predicate is not None and not predicate(chunk):
                continue
            hits.append(SearchHit(chunk=chunk, score=float(score), source=self.name))
            if len(hits) >= top_k:
                break
        return hits
