"""Live indexes: incremental upserts, TTL expiry, and freshness tracking.

:class:`LiveIndex` wraps any ``Index`` so a corpus can change without a full
rebuild. It adds **content-hash change detection** — an upsert re-embeds only
the chunks whose text actually changed, leaving unchanged chunks untouched — so
re-indexing a mostly-stable corpus is cheap. Entries can carry a TTL after which
they silently stop matching (and are purged), and every chunk is stamped with
``metadata["indexed_at"]`` so freshness flows into evidence metadata and
downstream scoring. ``upsert_stream`` ingests an async stream in batches for
freshness; re-embedding migrations live on
:meth:`~vincio.retrieval.indexes.VectorIndex.migrate`.
"""

from __future__ import annotations

from collections.abc import AsyncIterable
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from ..core.types import Chunk
from ..core.utils import stable_hash, utcnow
from .indexes import Index, SearchFilter, SearchHit

__all__ = ["LiveIndex", "UpsertStats"]


class UpsertStats(BaseModel):
    """What an upsert actually did — the re-embedding it avoided is the win."""

    added: int = 0
    updated: int = 0
    unchanged: int = 0

    @property
    def reembedded(self) -> int:
        return self.added + self.updated

    def merge(self, other: UpsertStats) -> None:
        self.added += other.added
        self.updated += other.updated
        self.unchanged += other.unchanged


def _content_hash(chunk: Chunk) -> str:
    return stable_hash({"text": chunk.text, "kind": chunk.kind})


class LiveIndex:
    """Mutable wrapper over any index: incremental upsert, TTL, freshness."""

    def __init__(self, inner: Index, *, ttl_seconds: float | None = None) -> None:
        self.inner = inner
        self.ttl_seconds = ttl_seconds
        self.name = f"live[{getattr(inner, 'name', 'index')}]"
        self._indexed_at: dict[str, datetime] = {}
        self._expires_at: dict[str, datetime] = {}
        self._content: dict[str, str] = {}  # chunk_id -> content hash

    def __len__(self) -> int:
        return len(self.inner)

    async def add(self, chunks: list[Chunk]) -> None:
        await self.upsert(chunks)

    async def upsert(self, chunks: list[Chunk], *, ttl_seconds: float | None = None) -> UpsertStats:
        """Insert or replace chunks, re-embedding only what changed.

        A chunk whose content hash matches the indexed copy is left in place
        (its freshness/TTL are refreshed); only new or changed chunks are sent
        to the inner index, so re-embedding cost scales with the *delta*, not
        the corpus size.
        """
        stats = UpsertStats()
        if not chunks:
            return stats
        now = utcnow()
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        to_write: list[Chunk] = []
        replaced: list[str] = []
        for chunk in chunks:
            digest = _content_hash(chunk)
            previous = self._content.get(chunk.id)
            # Re-seeing a chunk refreshes its TTL (it is still present), but its
            # freshness stamp tracks the content: unchanged content keeps its
            # original indexed_at, so the LiveIndex's record and the inner
            # index's chunk metadata stay consistent.
            if ttl is not None:
                self._expires_at[chunk.id] = now + timedelta(seconds=ttl)
            else:
                self._expires_at.pop(chunk.id, None)
            if previous == digest:
                stats.unchanged += 1
                continue
            if previous is not None:
                replaced.append(chunk.id)
                stats.updated += 1
            else:
                stats.added += 1
            self._indexed_at[chunk.id] = now
            self._content[chunk.id] = digest
            chunk.metadata = {**chunk.metadata, "indexed_at": now.isoformat()}
            to_write.append(chunk)
        if replaced:
            await self.inner.delete(replaced)
        if to_write:
            await self.inner.add(to_write)
        return stats

    async def upsert_stream(
        self,
        chunks: AsyncIterable[Chunk],
        *,
        batch_size: int = 64,
        ttl_seconds: float | None = None,
    ) -> UpsertStats:
        """Consume an async stream of chunks, upserting in batches for freshness."""
        total = UpsertStats()
        batch: list[Chunk] = []
        async for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= batch_size:
                total.merge(await self.upsert(batch, ttl_seconds=ttl_seconds))
                batch = []
        if batch:
            total.merge(await self.upsert(batch, ttl_seconds=ttl_seconds))
        return total

    async def delete(self, chunk_ids: list[str]) -> int:
        removed = await self.inner.delete(chunk_ids)
        for chunk_id in chunk_ids:
            self._indexed_at.pop(chunk_id, None)
            self._expires_at.pop(chunk_id, None)
            self._content.pop(chunk_id, None)
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
