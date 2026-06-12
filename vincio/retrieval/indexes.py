"""Retrieval indexes: BM25 sparse, dense vector, with metadata
filtering. Pure-python implementations with no service dependencies; the
storage layer provides Qdrant/pgvector adapters with the same interface.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel

from ..core.types import Chunk
from .embeddings import Embedder, LocalHashEmbedder, cosine

__all__ = ["SearchHit", "SearchFilter", "Index", "BM25Index", "VectorIndex"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    source: str = ""  # which index produced it


SearchFilter = Callable[[Chunk], bool]


def build_filter(
    *,
    tenant_id: str | None = None,
    document_ids: list[str] | None = None,
    kinds: list[str] | None = None,
    metadata_equals: dict[str, Any] | None = None,
) -> SearchFilter:
    def predicate(chunk: Chunk) -> bool:
        if tenant_id is not None and chunk.tenant_id not in (None, tenant_id):
            return False
        if document_ids is not None and chunk.document_id not in document_ids:
            return False
        if kinds is not None and chunk.kind not in kinds:
            return False
        if metadata_equals:
            for key, value in metadata_equals.items():
                if chunk.metadata.get(key) != value:
                    return False
        return True

    return predicate


class Index(Protocol):
    async def add(self, chunks: list[Chunk]) -> None: ...

    async def search(
        self, query: str, *, top_k: int = 10, where: SearchFilter | None = None
    ) -> list[SearchHit]: ...

    async def delete(self, chunk_ids: list[str]) -> int: ...

    def __len__(self) -> int: ...


class BM25Index:
    """Okapi BM25 (k1/b) over in-memory chunks."""

    name = "bm25"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: dict[str, Chunk] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._doc_len: dict[str, int] = {}
        self._total_len = 0

    def __len__(self) -> int:
        return len(self.chunks)

    async def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            if chunk.id in self.chunks:
                await self.delete([chunk.id])
            tokens = _tokenize(chunk.text)
            self.chunks[chunk.id] = chunk
            counts = Counter(tokens)
            self._tf[chunk.id] = counts
            for term in counts:
                self._df[term] += 1
            self._doc_len[chunk.id] = len(tokens)
            self._total_len += len(tokens)

    async def delete(self, chunk_ids: list[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            if chunk_id not in self.chunks:
                continue
            counts = self._tf.pop(chunk_id)
            for term in counts:
                self._df[term] -= 1
                if self._df[term] <= 0:
                    del self._df[term]
            self._total_len -= self._doc_len.pop(chunk_id)
            del self.chunks[chunk_id]
            removed += 1
        return removed

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    async def search(
        self, query: str, *, top_k: int = 10, where: SearchFilter | None = None
    ) -> list[SearchHit]:
        if not self.chunks:
            return []
        query_terms = _tokenize(query)
        average_length = self._total_len / max(1, len(self.chunks))
        scores: dict[str, float] = defaultdict(float)
        for term in query_terms:
            idf = self._idf(term)
            if idf <= 0:
                continue
            for chunk_id, counts in self._tf.items():
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                length_norm = 1 - self.b + self.b * self._doc_len[chunk_id] / average_length
                scores[chunk_id] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)
        hits = []
        for chunk_id, score in scores.items():
            chunk = self.chunks[chunk_id]
            if where is not None and not where(chunk):
                continue
            hits.append(SearchHit(chunk=chunk, score=score, source=self.name))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]


class VectorIndex:
    """Brute-force cosine search over in-memory vectors. For the local/MVP
    path; swap in Qdrant/pgvector adapters (vincio.storage) at scale."""

    name = "vector"

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or LocalHashEmbedder()
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, list[float]] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = await self.embedder.embed([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors, strict=False):
            self.chunks[chunk.id] = chunk
            self.vectors[chunk.id] = vector

    async def delete(self, chunk_ids: list[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
                del self.vectors[chunk_id]
                removed += 1
        return removed

    async def search(
        self, query: str, *, top_k: int = 10, where: SearchFilter | None = None
    ) -> list[SearchHit]:
        if not self.chunks:
            return []
        [query_vector] = await self.embedder.embed([query])
        hits: list[SearchHit] = []
        for chunk_id, vector in self.vectors.items():
            chunk = self.chunks[chunk_id]
            if where is not None and not where(chunk):
                continue
            hits.append(
                SearchHit(chunk=chunk, score=cosine(query_vector, vector), source=self.name)
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
