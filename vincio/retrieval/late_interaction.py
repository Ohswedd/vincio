"""Late-interaction retrieval (ColBERT-style multi-vector MaxSim).

Documents and queries are embedded *per token*; relevance is the MaxSim sum
``score(Q, D) = Σ_{q∈Q} max_{d∈D} q·d`` — token-level matching that survives
paraphrase and partial overlap where single-vector retrieval averages the
signal away. :class:`LateInteractionIndex` implements the same ``Index``
protocol as BM25/vector/sparse, so late interaction fuses into the existing
weighted-RRF hybrid merge.

For scale, ``compressed=True`` enables PLAID-style two-stage search: token
vectors are clustered into centroids (k-means, deterministic init), every
document keeps compact centroid codes, candidate generation walks the
centroid→document inverted lists with centroid-approximated MaxSim, and only
the surviving candidates get exact multi-vector scoring.

Any :class:`~vincio.retrieval.embeddings.Embedder` supplies token vectors —
the offline :class:`~vincio.retrieval.embeddings.LocalHashEmbedder` by
default, or a provider/ColBERT checkpoint embedder behind the same protocol.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from ..core.types import Chunk
from .embeddings import Embedder, LocalHashEmbedder
from .filters import as_predicate
from .indexes import SearchHit, Where

__all__ = ["LateInteractionIndex"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _kmeans(vectors: list[list[float]], k: int, *, iters: int = 4) -> list[list[float]]:
    """Deterministic k-means over unit vectors (cosine via dot product)."""
    if len(vectors) <= k:
        return [list(v) for v in vectors]
    # Deterministic init: evenly spaced picks over a stable ordering.
    order = sorted(range(len(vectors)), key=lambda i: vectors[i])
    step = len(order) / k
    centroids = [list(vectors[order[int(i * step)]]) for i in range(k)]
    for _ in range(iters):
        assignments: list[list[list[float]]] = [[] for _ in range(k)]
        for vector in vectors:
            best = max(range(k), key=lambda c: _dot(vector, centroids[c]))
            assignments[best].append(vector)
        for index, members in enumerate(assignments):
            if not members:
                continue
            dim = len(members[0])
            mean = [sum(v[d] for v in members) / len(members) for d in range(dim)]
            centroids[index] = _normalize(mean)
    return centroids


class LateInteractionIndex:
    """Multi-vector index scored by MaxSim, with optional PLAID-style
    centroid compression for candidate generation."""

    name = "late_interaction"

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        max_doc_tokens: int = 192,
        max_query_tokens: int = 32,
        compressed: bool = False,
        n_centroids: int = 64,
        n_probe: int = 4,
        rerank_factor: int = 4,
    ) -> None:
        self.embedder = embedder or LocalHashEmbedder(dim=64)
        self.max_doc_tokens = max_doc_tokens
        self.max_query_tokens = max_query_tokens
        self.compressed = compressed
        self.n_centroids = n_centroids
        self.n_probe = max(1, n_probe)
        self.rerank_factor = max(1, rerank_factor)
        self.chunks: dict[str, Chunk] = {}
        self._doc_vectors: dict[str, list[list[float]]] = {}
        # Token-vocabulary vector cache: each distinct token embeds once.
        self._token_vectors: dict[str, list[float]] = {}
        # PLAID-style structures, rebuilt lazily after mutations.
        self._centroids: list[list[float]] = []
        self._doc_codes: dict[str, set[int]] = {}
        self._centroid_docs: dict[int, set[str]] = defaultdict(set)
        self._codes_stale = True

    def __len__(self) -> int:
        return len(self.chunks)

    async def _embed_tokens(self, tokens: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(tokens) if t not in self._token_vectors]
        if missing:
            vectors = await self.embedder.embed(missing)
            for token, vector in zip(missing, vectors, strict=True):
                self._token_vectors[token] = _normalize(vector)
        return [self._token_vectors[t] for t in tokens]

    async def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            # MaxSim takes a max over document tokens, so duplicates are inert:
            # store each distinct token vector once.
            tokens = list(dict.fromkeys(_tokenize(chunk.text)))[: self.max_doc_tokens]
            self.chunks[chunk.id] = chunk
            self._doc_vectors[chunk.id] = await self._embed_tokens(tokens)
        if chunks:
            self._codes_stale = True

    async def delete(self, chunk_ids: list[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
                del self._doc_vectors[chunk_id]
                removed += 1
        if removed:
            self._codes_stale = True
        return removed

    # -- PLAID-style compression ---------------------------------------------------

    def _build_codes(self) -> None:
        all_vectors = [v for vectors in self._doc_vectors.values() for v in vectors]
        self._centroids = _kmeans(all_vectors, self.n_centroids)
        self._doc_codes = {}
        self._centroid_docs = defaultdict(set)
        for chunk_id, vectors in self._doc_vectors.items():
            codes = {
                max(range(len(self._centroids)), key=lambda c: _dot(v, self._centroids[c]))
                for v in vectors
            }
            self._doc_codes[chunk_id] = codes
            for code in codes:
                self._centroid_docs[code].add(chunk_id)
        self._codes_stale = False

    def _candidates(self, query_vectors: list[list[float]]) -> list[str]:
        """Centroid-approximated MaxSim over the inverted lists."""
        centroid_sims = [
            [_dot(q, centroid) for centroid in self._centroids] for q in query_vectors
        ]
        probed: set[int] = set()
        for sims in centroid_sims:
            top = sorted(range(len(sims)), key=sims.__getitem__, reverse=True)[: self.n_probe]
            probed.update(top)
        approx: dict[str, float] = defaultdict(float)
        candidate_ids = {cid for code in probed for cid in self._centroid_docs[code]}
        for chunk_id in candidate_ids:
            codes = self._doc_codes[chunk_id]
            for sims in centroid_sims:
                best = max((sims[code] for code in codes), default=0.0)
                approx[chunk_id] += best
        ranked = sorted(approx, key=approx.__getitem__, reverse=True)
        return ranked

    # -- search ---------------------------------------------------------------------

    def _maxsim(self, query_vectors: list[list[float]], doc_vectors: list[list[float]]) -> float:
        if not doc_vectors:
            return 0.0
        score = 0.0
        for q in query_vectors:
            score += max(_dot(q, d) for d in doc_vectors)
        return score / max(1, len(query_vectors))

    async def search(
        self, query: str, *, top_k: int = 10, where: Where | None = None
    ) -> list[SearchHit]:
        if not self.chunks:
            return []
        predicate = as_predicate(where)
        tokens = _tokenize(query)[: self.max_query_tokens]
        if not tokens:
            return []
        query_vectors = await self._embed_tokens(tokens)
        if self.compressed and len(self.chunks) > self.n_centroids:
            if self._codes_stale:
                self._build_codes()
            candidate_ids = self._candidates(query_vectors)[: top_k * self.rerank_factor]
        else:
            candidate_ids = list(self.chunks)
        hits: list[SearchHit] = []
        for chunk_id in candidate_ids:
            chunk = self.chunks[chunk_id]
            if predicate is not None and not predicate(chunk):
                continue
            score = self._maxsim(query_vectors, self._doc_vectors[chunk_id])
            hits.append(SearchHit(chunk=chunk, score=score, source=self.name))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
