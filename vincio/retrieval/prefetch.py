"""Speculative retrieval prefetch.

Task classification is a cheap, deterministic step that finishes before any
retrieval runs. The moment the query is normalized and its task type is known,
the embedding the retriever will need can be computed speculatively — warming
the content-addressed embedding cache (and the embedder's connection pool) so
that, by the time retrieval reaches its query-embed step, the vector is already
there. The warm runs concurrently with the rest of preparation (memory recall,
file ingestion) and is cancelled cleanly if it is no longer needed.

The prefetch is purely an accelerator: it shares the app's embedder, so a warm
that lands is a cache hit for retrieval and a warm that does not land costs only
a redundant embed of the query that retrieval would have done anyway. A failed
warm never affects the run.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .embeddings import Embedder, embed_texts

__all__ = ["SpeculativePrefetcher", "PrefetchHandle"]

# Task types whose runs do not retrieve, so there is nothing to warm.
_NON_RETRIEVAL_TASKS = frozenset({"classification", "creative_generation"})


class PrefetchHandle:
    """A handle to an in-flight speculative warm.

    Awaiting :meth:`result` returns the number of texts warmed (``0`` if the
    warm was cancelled or failed); :meth:`cancel` stops it cleanly.
    """

    def __init__(self, task: asyncio.Future[int], queries: list[str]) -> None:
        self._task = task
        self.queries = queries

    @property
    def done(self) -> bool:
        return self._task.done()

    def cancel(self) -> None:
        if not self._task.done():
            self._task.cancel()

    async def result(self) -> int:
        try:
            return await self._task
        except asyncio.CancelledError:
            return 0


class SpeculativePrefetcher:
    """Warms the embedding cache from a predicted query before retrieval runs."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.warmed = 0

    def predict_queries(self, query: str, task_type: Any | None = None) -> list[str]:
        """The queries worth warming for *query* under a predicted task type.

        The normalized query is the highest-confidence prediction — it is the
        exact text retrieval embeds for its primary search — so it is warmed
        whenever the task type retrieves. Non-retrieval task types warm nothing.
        """
        text = (query or "").strip()
        if not text:
            return []
        if task_type is not None and str(getattr(task_type, "value", task_type)) in _NON_RETRIEVAL_TASKS:
            return []
        return [text]

    async def _warm(self, queries: list[str]) -> int:
        if not queries or self.embedder is None:
            return 0
        try:
            # Match retrieval's query embedding exactly (same input_type → same
            # content-addressed cache key) so the warm lands as a cache hit.
            await embed_texts(self.embedder, queries, input_type="query")
        except asyncio.CancelledError:
            raise
        except BaseException:  # noqa: BLE001 - prefetch is best-effort; never break the run
            # Any failure (including a pathological SystemExit from the embedder)
            # stays contained in the warming task — the run is never affected.
            return 0
        self.warmed += 1
        return len(queries)

    def warm(self, query: str, task_type: Any | None = None) -> PrefetchHandle:
        """Start warming the predicted queries concurrently; returns a handle."""
        queries = self.predict_queries(query, task_type)
        task = asyncio.ensure_future(self._warm(queries))
        return PrefetchHandle(task, queries)
