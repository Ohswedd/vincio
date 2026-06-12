"""Embedding interfaces (dense retrieval, extras).

- :class:`LocalHashEmbedder` — deterministic, dependency-free embeddings
  (token-hash bag with char-trigram smoothing). Useful for tests, offline
  mode, and as a fallback; captures lexical similarity, not deep semantics.
- :class:`ProviderEmbedder` — embeddings via any provider that implements
  ``embed`` (OpenAI, Google, Mistral, local servers); large inputs are
  split into bounded batches embedded concurrently.
- :class:`CachedEmbedder` — wraps any embedder with a thread-safe,
  content-addressed cache (in-memory by default, any
  :class:`~vincio.caching.base.CacheBackend` for persistence).
- :class:`BatchingEmbedder` — micro-batches concurrent ``embed`` calls into
  one provider call (request coalescing for embeddings).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import threading
from collections import OrderedDict
from typing import Any, Protocol

from ..core.concurrency import gather_bounded
from ..providers.base import ModelProvider, run_sync

__all__ = [
    "Embedder",
    "LocalHashEmbedder",
    "ProviderEmbedder",
    "CachedEmbedder",
    "BatchingEmbedder",
    "cosine",
]


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        ...


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class LocalHashEmbedder:
    """Deterministic offline embeddings.

    Combines word-level and character-trigram hash features so morphological
    variants ("termination"/"terminate") land near each other.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        tokens = text.lower().split()
        features: list[str] = []
        for token in tokens:
            token = "".join(c for c in token if c.isalnum())
            if not token:
                continue
            features.append(f"w:{token}")
            padded = f"^{token}$"
            features.extend(f"t:{padded[i:i+3]}" for i in range(len(padded) - 2))
        return features

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature in self._features(text):
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 if feature.startswith("w:") else 0.35
            vector[index] += sign * weight
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]


class ProviderEmbedder:
    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str | None = None,
        dim: int = 1536,
        batch_size: int = 64,
        concurrency: int = 4,
    ) -> None:
        self.provider = provider
        self.model = model
        self.dim = dim
        self.batch_size = max(1, batch_size)
        self.concurrency = max(1, concurrency)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) <= self.batch_size:
            vectors = await self.provider.embed(texts, self.model)
        else:
            batches = [
                texts[start : start + self.batch_size]
                for start in range(0, len(texts), self.batch_size)
            ]
            results = await gather_bounded(
                (self.provider.embed(batch, self.model) for batch in batches),
                limit=self.concurrency,
            )
            vectors = [vector for batch_vectors in results for vector in batch_vectors]
        if vectors:
            self.dim = len(vectors[0])
        return vectors


def _content_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class CachedEmbedder:
    """Content-addressed embedding cache, safe under concurrent use.

    Keys are SHA-256 hashes of the text, so identical content embedded from
    any path (chunking, queries, semantic cache) reuses one vector. An
    optional persistent ``backend`` (any :class:`~vincio.caching.base
    .CacheBackend`) survives process restarts; the in-memory LRU fronts it.
    """

    def __init__(self, inner: Embedder, *, max_entries: int = 50_000, backend: Any | None = None) -> None:
        self.inner = inner
        self.max_entries = max_entries
        self.backend = backend  # CacheBackend | None (duck-typed: get/set)
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def dim(self) -> int:
        return self.inner.dim

    def _get_cached(self, key: str) -> list[float] | None:
        with self._lock:
            vector = self._cache.get(key)
            if vector is not None:
                self._cache.move_to_end(key)
                return vector
        if self.backend is not None:
            persisted = self.backend.get(f"emb:{key}")
            if persisted is not None:
                self._store_memory(key, persisted)
                return persisted
        return None

    def _store_memory(self, key: str, vector: list[float]) -> None:
        with self._lock:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [_content_key(text) for text in texts]
        resolved: dict[str, list[float]] = {}
        missing_texts: list[str] = []
        missing_keys: list[str] = []
        for text, key in zip(texts, keys, strict=True):
            if key in resolved:
                continue
            vector = self._get_cached(key)
            if vector is not None:
                resolved[key] = vector
            elif key not in missing_keys:
                missing_keys.append(key)
                missing_texts.append(text)
        if missing_texts:
            vectors = await self.inner.embed(missing_texts)
            for key, vector in zip(missing_keys, vectors, strict=True):
                resolved[key] = vector
                self._store_memory(key, vector)
                if self.backend is not None:
                    self.backend.set(f"emb:{key}", vector, tags=["embeddings"])
            self.misses += len(missing_texts)
        self.hits += len(texts) - len(missing_texts)
        return [resolved[key] for key in keys]

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        return run_sync(self.embed(texts))


class BatchingEmbedder:
    """Coalesces concurrent ``embed`` calls into provider micro-batches.

    Callers that issue small ``embed`` requests at the same time (parallel
    retrieval subqueries, agent steps) are merged into one provider call,
    flushed when the batch fills or after ``window_ms`` of quiet — fewer
    round-trips, same results. Duplicate texts within a flush are sent once.
    """

    def __init__(self, inner: Embedder, *, max_batch: int = 64, window_ms: float = 5.0) -> None:
        self.inner = inner
        self.max_batch = max(1, max_batch)
        self.window_ms = window_ms
        self.flushes = 0
        self._pending: list[tuple[str, asyncio.Future[list[float]]]] = []
        self._timer: asyncio.Task | None = None

    @property
    def dim(self) -> int:
        return self.inner.dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future[list[float]]] = []
        for text in texts:
            future: asyncio.Future[list[float]] = loop.create_future()
            self._pending.append((text, future))
            futures.append(future)
        full_batch: list[tuple[str, asyncio.Future[list[float]]]] | None = None
        if len(self._pending) >= self.max_batch:
            full_batch, self._pending = self._pending, []
        elif self._timer is None or self._timer.done():
            self._timer = asyncio.ensure_future(self._delayed_flush())
        if full_batch is not None:
            await self._flush(full_batch)
        return list(await asyncio.gather(*futures))

    async def aclose(self) -> None:
        """Flush anything pending and stop the timer."""
        batch, self._pending = self._pending, []
        if self._timer is not None and not self._timer.done():
            self._timer.cancel()
            try:
                await self._timer
            except asyncio.CancelledError:
                pass
        if batch:
            await self._flush(batch)

    async def _delayed_flush(self) -> None:
        await asyncio.sleep(self.window_ms / 1000.0)
        batch, self._pending = self._pending, []
        if batch:
            await self._flush(batch)

    async def _flush(self, batch: list[tuple[str, asyncio.Future[list[float]]]]) -> None:
        self.flushes += 1
        unique: dict[str, list[asyncio.Future[list[float]]]] = {}
        for text, future in batch:
            unique.setdefault(text, []).append(future)
        texts = list(unique)
        try:
            vectors = await self.inner.embed(texts)
        except BaseException as exc:  # propagate to every waiter
            for waiters in unique.values():
                for future in waiters:
                    if not future.done():
                        future.set_exception(exc)
                        future.exception()  # consumed below or by the waiter
            if isinstance(exc, asyncio.CancelledError):
                raise
            return
        for text, vector in zip(texts, vectors, strict=True):
            for future in unique[text]:
                if not future.done():
                    future.set_result(vector)
