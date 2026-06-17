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
from .embeddings import Embedder, LocalHashEmbedder, cosine, embed_texts

try:  # optional acceleration — pure-Python cosine stays the zero-dependency default
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only when numpy is absent
    _np = None

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
        # Inverted posting lists term -> {chunk_id: tf}: search scans only the
        # documents that contain each query term instead of every document
        # (sub-linear in corpus size for selective queries).
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
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
            for term, tf in counts.items():
                self._df[term] += 1
                self._postings[term][chunk.id] = tf
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
                postings = self._postings.get(term)
                if postings is not None:
                    postings.pop(chunk_id, None)
                    if not postings:
                        del self._postings[term]
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
            # Only the documents in this term's posting list contribute.
            for chunk_id, tf in self._postings.get(term, {}).items():
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
        # Lazily-built, normalized matrix cache for the optional numpy path.
        self._matrix: Any = None
        self._matrix_ids: list[str] = []
        self._dirty = True

    def __len__(self) -> int:
        return len(self.chunks)

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = await embed_texts(self.embedder, [c.text for c in chunks], input_type="document")
        for chunk, vector in zip(chunks, vectors, strict=False):
            self.chunks[chunk.id] = chunk
            self.vectors[chunk.id] = vector
        self._dirty = True

    async def delete(self, chunk_ids: list[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
                del self.vectors[chunk_id]
                removed += 1
        if removed:
            self._dirty = True
        return removed

    def _ensure_matrix(self) -> bool:
        """Build (or refresh) the row-normalized vector matrix when numpy is
        available. Returns False when numpy is absent or the index is empty, so
        the caller falls back to the pure-Python cosine loop."""
        if _np is None:
            return False
        if self._dirty or self._matrix is None:
            ids = list(self.vectors)
            if not ids:
                self._matrix, self._matrix_ids, self._dirty = None, [], False
                return False
            mat = _np.asarray([self.vectors[i] for i in ids], dtype=float)
            norms = _np.linalg.norm(mat, axis=1)
            norms[norms == 0.0] = 1.0
            self._matrix = mat / norms[:, None]
            self._matrix_ids = ids
            self._dirty = False
        return self._matrix is not None

    async def migrate(self, embedder: Embedder, *, batch_size: int = 64) -> int:
        """Re-embed every stored chunk with a new embedder, in place — a
        model migration without rebuilding the index or re-chunking."""
        self.embedder = embedder
        chunk_ids = list(self.chunks)
        for start in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[start : start + batch_size]
            vectors = await embedder.embed([self.chunks[cid].text for cid in batch])
            for chunk_id, vector in zip(batch, vectors, strict=True):
                self.vectors[chunk_id] = vector
        self._dirty = True
        return len(chunk_ids)

    async def search(
        self, query: str, *, top_k: int = 10, where: SearchFilter | None = None
    ) -> list[SearchHit]:
        if not self.chunks:
            return []
        [query_vector] = await embed_texts(self.embedder, [query], input_type="query")
        # Optional vectorized path: a single matrix-vector product over the
        # row-normalized matrix replaces the per-chunk Python cosine loop.
        if self._ensure_matrix():
            q = _np.asarray(query_vector, dtype=float)
            qn = float(_np.linalg.norm(q)) or 1.0
            sims = self._matrix @ (q / qn)
            hits: list[SearchHit] = []
            # Stable sort so tie ordering matches the pure-Python fallback
            # (insertion order), keeping results reproducible with/without numpy.
            for idx in _np.argsort(-sims, kind="stable"):
                chunk_id = self._matrix_ids[int(idx)]
                chunk = self.chunks[chunk_id]
                if where is not None and not where(chunk):
                    continue
                hits.append(SearchHit(chunk=chunk, score=float(sims[int(idx)]), source=self.name))
                if len(hits) >= top_k:
                    break
            return hits
        hits = []
        for chunk_id, vector in self.vectors.items():
            chunk = self.chunks[chunk_id]
            if where is not None and not where(chunk):
                continue
            hits.append(
                SearchHit(chunk=chunk, score=cosine(query_vector, vector), source=self.name)
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
