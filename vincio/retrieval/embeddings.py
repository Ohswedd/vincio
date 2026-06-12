"""Embedding interfaces (dense retrieval, extras).

- :class:`LocalHashEmbedder` — deterministic, dependency-free embeddings
  (token-hash bag with char-trigram smoothing). Useful for tests, offline
  mode, and as a fallback; captures lexical similarity, not deep semantics.
- :class:`ProviderEmbedder` — embeddings via any provider that implements
  ``embed`` (OpenAI, Google, Mistral, local servers).
- :class:`CachedEmbedder` — wraps any embedder with an in-memory cache.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from ..providers.base import ModelProvider, run_sync

__all__ = ["Embedder", "LocalHashEmbedder", "ProviderEmbedder", "CachedEmbedder", "cosine"]


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
    def __init__(self, provider: ModelProvider, *, model: str | None = None, dim: int = 1536) -> None:
        self.provider = provider
        self.model = model
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = await self.provider.embed(texts, self.model)
        if vectors:
            self.dim = len(vectors[0])
        return vectors


class CachedEmbedder:
    def __init__(self, inner: Embedder, *, max_entries: int = 50_000) -> None:
        self.inner = inner
        self.max_entries = max_entries
        self._cache: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0

    @property
    def dim(self) -> int:
        return self.inner.dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            vectors = await self.inner.embed(missing)
            for text, vector in zip(missing, vectors, strict=False):
                if len(self._cache) >= self.max_entries:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[text] = vector
            self.misses += len(missing)
        self.hits += len(texts) - len(missing)
        return [self._cache[t] for t in texts]

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        return run_sync(self.embed(texts))
