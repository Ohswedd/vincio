"""Sharded indexing for scale (1.3).

:class:`ShardedIndex` splits a corpus across N backend indexes and queries them
**in parallel**, merging the per-shard hits — so a corpus too large (or too
slow) for one backend can be spread across several and searched concurrently. It
implements the :class:`~vincio.retrieval.indexes.Index` protocol, so it drops
straight into the retrieval engine's fan-out, behind a :class:`LiveIndex`, or
anywhere a single index would go — nothing downstream changes.

Chunks of one document land on the same shard by default (hash of
``document_id``), keeping a document's chunks co-located for coherent retrieval;
pass a custom ``router`` to shard by any rule (tenant, region, recency).
"""

from __future__ import annotations

from collections.abc import Callable

from ..core.concurrency import gather_bounded
from ..core.types import Chunk
from ..core.utils import stable_hash
from .indexes import Index, SearchFilter, SearchHit

__all__ = ["ShardedIndex"]


class ShardedIndex:
    """Routes writes across shards and merges parallel reads (Index protocol)."""

    def __init__(
        self,
        shards: list[Index],
        *,
        router: Callable[[Chunk], int] | None = None,
        max_concurrency: int = 8,
    ) -> None:
        if not shards:
            raise ValueError("ShardedIndex requires at least one shard")
        self.shards = shards
        self.router = router
        self.max_concurrency = max(1, max_concurrency)
        self.name = f"sharded[{len(shards)}]"

    def _shard_for(self, chunk: Chunk) -> int:
        if self.router is not None:
            return self.router(chunk) % len(self.shards)
        # Hash the document id so a document's chunks stay co-located.
        key = chunk.document_id or chunk.id
        return int(stable_hash(key)[:8], 16) % len(self.shards)

    def __len__(self) -> int:
        return sum(len(shard) for shard in self.shards)

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        grouped: dict[int, list[Chunk]] = {}
        for chunk in chunks:
            grouped.setdefault(self._shard_for(chunk), []).append(chunk)
        await gather_bounded(
            (self.shards[i].add(group) for i, group in grouped.items()),
            limit=self.max_concurrency,
        )

    async def search(
        self, query: str, *, top_k: int = 10, where: SearchFilter | None = None
    ) -> list[SearchHit]:
        # Each shard returns its own top_k; merge and keep the global top_k.
        per_shard = await gather_bounded(
            (shard.search(query, top_k=top_k, where=where) for shard in self.shards),
            limit=self.max_concurrency,
        )
        hits: list[SearchHit] = [hit for shard_hits in per_shard for hit in shard_hits]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    async def delete(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        removed = await gather_bounded(
            (shard.delete(chunk_ids) for shard in self.shards), limit=self.max_concurrency
        )
        return sum(removed)
