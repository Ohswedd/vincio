"""Learned sparse retrieval (SPLADE-style impact weighting).

A :class:`SparseEncoder` turns text into a sparse term→weight mapping
("impact vector"). :class:`SparseIndex` stores those vectors in an inverted
index and scores by impact dot product, implementing the same ``Index``
protocol as BM25 and the vector index — so learned sparse fuses with dense,
lexical, and graph retrieval in the existing weighted-RRF merge.

- :class:`LocalImpactEncoder` — deterministic, dependency-free approximation
  of a learned sparse model: sublinear term-frequency impacts plus
  morphological term expansion (SPLADE's neural expansion, approximated by
  stem variants), so "refunds"/"refunded"/"refunding" share mass.
- :class:`CallableSparseEncoder` — adapter for a real served model (SPLADE,
  uniCOIL, ELSER...): pass an async callable
  ``(texts, is_query) -> list[dict[str, float]]``.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from typing import Protocol

from ..core.types import Chunk
from .indexes import SearchFilter, SearchHit

__all__ = [
    "SparseVector",
    "SparseEncoder",
    "LocalImpactEncoder",
    "CallableSparseEncoder",
    "SparseIndex",
]

SparseVector = dict[str, float]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Longest-match-first suffix stripping; coarse but deterministic, and applied
# to both documents and queries so morphological variants meet in stem space.
_SUFFIXES = (
    "ations", "ation", "ities", "ingly", "ments",
    "ment", "ness", "ings", "ions", "ies",
    "ing", "ion", "ers", "ed", "es", "ly", "er", "s", "e",
)


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseEncoder(Protocol):
    async def encode(
        self, texts: list[str], *, is_query: bool = False
    ) -> list[SparseVector]:  # pragma: no cover
        ...


class LocalImpactEncoder:
    """Offline impact encoder: sublinear tf weights + stem expansion."""

    def __init__(self, *, expansion_weight: float = 0.5) -> None:
        self.expansion_weight = expansion_weight

    def encode_one(self, text: str, *, is_query: bool = False) -> SparseVector:
        counts = Counter(_tokenize(text))
        vector: SparseVector = {}
        for term, tf in counts.items():
            impact = 1.0 if is_query else 1.0 + math.log(tf)
            vector[term] = max(vector.get(term, 0.0), impact)
            stem = _stem(term)
            if stem != term:
                expanded = impact * self.expansion_weight
                vector[stem] = max(vector.get(stem, 0.0), expanded)
        return vector

    async def encode(self, texts: list[str], *, is_query: bool = False) -> list[SparseVector]:
        return [self.encode_one(text, is_query=is_query) for text in texts]


class CallableSparseEncoder:
    """Adapter for an external learned sparse model served behind an async
    callable ``(texts, is_query) -> list[dict[str, float]]``."""

    def __init__(self, encode_fn: Callable[[list[str], bool], Awaitable[list[SparseVector]]]) -> None:
        self.encode_fn = encode_fn

    async def encode(self, texts: list[str], *, is_query: bool = False) -> list[SparseVector]:
        return await self.encode_fn(texts, is_query)


class SparseIndex:
    """Inverted impact index over sparse vectors (learned sparse retrieval).

    Scores are impact dot products: ``score(q, d) = Σ_t q[t] · d[t]`` —
    the standard scoring for SPLADE/uniCOIL-style models.
    """

    name = "sparse"

    def __init__(self, encoder: SparseEncoder | None = None) -> None:
        self.encoder = encoder or LocalImpactEncoder()
        self.chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, SparseVector] = {}
        self._postings: dict[str, dict[str, float]] = defaultdict(dict)

    def __len__(self) -> int:
        return len(self.chunks)

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        await self.delete([c.id for c in chunks if c.id in self.chunks])
        vectors = await self.encoder.encode([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.chunks[chunk.id] = chunk
            self._vectors[chunk.id] = vector
            for term, weight in vector.items():
                self._postings[term][chunk.id] = weight

    async def delete(self, chunk_ids: list[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            if chunk_id not in self.chunks:
                continue
            for term in self._vectors.pop(chunk_id):
                postings = self._postings.get(term)
                if postings is not None:
                    postings.pop(chunk_id, None)
                    if not postings:
                        del self._postings[term]
            del self.chunks[chunk_id]
            removed += 1
        return removed

    async def search(
        self, query: str, *, top_k: int = 10, where: SearchFilter | None = None
    ) -> list[SearchHit]:
        if not self.chunks:
            return []
        [query_vector] = await self.encoder.encode([query], is_query=True)
        scores: dict[str, float] = defaultdict(float)
        for term, query_weight in query_vector.items():
            for chunk_id, doc_weight in self._postings.get(term, {}).items():
                scores[chunk_id] += query_weight * doc_weight
        hits: list[SearchHit] = []
        for chunk_id, score in scores.items():
            chunk = self.chunks[chunk_id]
            if where is not None and not where(chunk):
                continue
            hits.append(SearchHit(chunk=chunk, score=score, source=self.name))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
