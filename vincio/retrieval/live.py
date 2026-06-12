"""Live indexes: upserts, TTL expiry, and freshness tracking.

:class:`LiveIndex` wraps any ``Index`` so a corpus can change without a
full rebuild: ``upsert`` replaces chunks in place, entries can carry a TTL
after which they silently stop matching (and are purged), and every chunk
is stamped with ``metadata["indexed_at"]`` so freshness flows into evidence
metadata and downstream scoring. Re-embedding migrations live on
:meth:`~vincio.retrieval.indexes.VectorIndex.migrate`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ..core.types import Chunk
from ..core.utils import utcnow
from .indexes import Index, SearchFilter, SearchHit

__all__ = ["LiveIndex"]


class LiveIndex:
    """Mutable wrapper over any index: upsert, TTL, freshness stamps."""

    def __init__(self, inner: Index, *, ttl_seconds: float | None = None) -> None:
        self.inner = inner
        self.ttl_seconds = ttl_seconds
        self.name = f"live[{getattr(inner, 'name', 'index')}]"
        self._indexed_at: dict[str, datetime] = {}
        self._expires_at: dict[str, datetime] = {}

    def __len__(self) -> int:
        return len(self.inner)

    async def add(self, chunks: list[Chunk]) -> None:
        await self.upsert(chunks)

    async def upsert(self, chunks: list[Chunk], *, ttl_seconds: float | None = None) -> None:
        """Insert or replace chunks; stamps freshness and optional expiry."""
        if not chunks:
            return
        now = utcnow()
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        existing = [c.id for c in chunks if c.id in self._indexed_at]
        if existing:
            await self.inner.delete(existing)
        for chunk in chunks:
            chunk.metadata = {**chunk.metadata, "indexed_at": now.isoformat()}
            self._indexed_at[chunk.id] = now
            if ttl is not None:
                self._expires_at[chunk.id] = now + timedelta(seconds=ttl)
            else:
                self._expires_at.pop(chunk.id, None)
        await self.inner.add(chunks)

    async def delete(self, chunk_ids: list[str]) -> int:
        removed = await self.inner.delete(chunk_ids)
        for chunk_id in chunk_ids:
            self._indexed_at.pop(chunk_id, None)
            self._expires_at.pop(chunk_id, None)
        return removed

    async def purge_expired(self, *, now: datetime | None = None) -> int:
        """Drop every entry past its TTL; returns the number removed."""
        now = now or utcnow()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        expired = [cid for cid, deadline in self._expires_at.items() if deadline <= now]
        if not expired:
            return 0
        return await self.delete(expired)

    async def search(
        self, query: str, *, top_k: int = 10, where: SearchFilter | None = None
    ) -> list[SearchHit]:
        await self.purge_expired()
        return await self.inner.search(query, top_k=top_k, where=where)
