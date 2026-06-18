"""Vector quantization + Matryoshka two-stage retrieval.

At scale, holding every full-precision vector in RAM and scoring all of them is
the cost. The standard answer is **two-stage retrieval**: a cheap coarse pass
over compressed vectors generates candidates, then an exact pass over the
full-precision vectors reranks them — recall of the exact index at a fraction of
the memory and compute. This module ships the building blocks:

* **scalar (int8) and binary quantization** of a vector, plus the matching
  coarse similarity functions; and
* :class:`TwoStageIndex`, an in-memory :class:`~vincio.retrieval.indexes.Index`
  that searches a coarse representation (Matryoshka-truncated then quantized,
  reusing :func:`~vincio.retrieval.embeddings.mrl_truncate`) and reranks the top
  ``rerank_factor × k`` candidates at full precision.

The same coarse/exact pattern is what the Qdrant and pgvector adapters delegate
to their native quantization (``vincio.storage.qdrant`` /
``vincio.storage.postgres``); this implementation is the dependency-free
reference that proves the recall trade-off offline.
"""

from __future__ import annotations

from typing import Any

from ..core.types import Chunk
from ..retrieval.filters import FilterSpec
from .embeddings import Embedder, LocalHashEmbedder, cosine, embed_texts, mrl_truncate
from .indexes import SearchHit, Where

__all__ = [
    "quantize_scalar",
    "quantize_binary",
    "scalar_similarity",
    "binary_similarity",
    "TwoStageIndex",
]


def quantize_scalar(vector: list[float], *, scale: int = 127) -> list[int]:
    """Symmetric int8 quantization of a (roughly unit-norm) vector.

    Each component maps to ``round(x * scale)`` clamped to ``[-scale, scale]``.
    Dot products of the quantized vectors approximate the cosine of the
    originals up to a constant factor — enough to rank candidates coarsely.
    """
    return [max(-scale, min(scale, round(x * scale))) for x in vector]


def quantize_binary(vector: list[float]) -> list[int]:
    """1-bit-per-dimension quantization: the sign of each component (1 / 0).

    32× smaller than float32; Hamming similarity over the sign bits is a
    surprisingly strong coarse ranker for normalized embeddings.
    """
    return [1 if x >= 0.0 else 0 for x in vector]


def scalar_similarity(a: list[int], b: list[int]) -> float:
    return float(sum(x * y for x, y in zip(a, b, strict=False)))


def binary_similarity(a: list[int], b: list[int]) -> float:
    """Fraction of matching sign bits (1.0 = identical, 0.0 = opposite)."""
    if not a:
        return 0.0
    matches = sum(1 for x, y in zip(a, b, strict=False) if x == y)
    return matches / len(a)


def _passes(where: Where | None, chunk: Chunk) -> bool:
    if where is None:
        return True
    if isinstance(where, FilterSpec):
        return where.matches(chunk)
    return bool(where(chunk))


class TwoStageIndex:
    """Matryoshka + quantized coarse search, full-precision exact rerank.

    Stage 1 ranks every chunk by a quantized similarity over a truncated
    ("coarse") view of the embedding and keeps the top ``rerank_factor × k``.
    Stage 2 rescbores those candidates by exact cosine over the full vectors.
    With ``quantization="none"`` and ``coarse_dims=None`` it degrades to a plain
    exact search, so the two-stage path is always a strict refinement of the
    same ranking — that is what the recall gate checks.
    """

    name = "two_stage"

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        coarse_dims: int | None = None,
        quantization: str = "scalar",
        rerank_factor: int = 4,
    ) -> None:
        if quantization not in ("scalar", "binary", "none"):
            raise ValueError(f"unknown quantization {quantization!r}; use scalar|binary|none")
        self.embedder = embedder or LocalHashEmbedder()
        self.coarse_dims = coarse_dims
        self.quantization = quantization
        self.rerank_factor = max(1, rerank_factor)
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, list[float]] = {}
        self.coarse: dict[str, list[int] | list[float]] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    def _coarsen(self, vector: list[float]) -> list[int] | list[float]:
        head = mrl_truncate(vector, self.coarse_dims) if self.coarse_dims else vector
        if self.quantization == "scalar":
            return quantize_scalar(head)
        if self.quantization == "binary":
            return quantize_binary(head)
        return list(head)

    def _coarse_score(self, query: list[int] | list[float], candidate: list[int] | list[float]) -> float:
        if self.quantization == "binary":
            return binary_similarity(query, candidate)  # type: ignore[arg-type]
        if self.quantization == "scalar":
            return scalar_similarity(query, candidate)  # type: ignore[arg-type]
        return cosine(query, candidate)  # type: ignore[arg-type]

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = await embed_texts(self.embedder, [c.text for c in chunks], input_type="document")
        for chunk, vector in zip(chunks, vectors, strict=False):
            self.chunks[chunk.id] = chunk
            self.vectors[chunk.id] = vector
            self.coarse[chunk.id] = self._coarsen(vector)

    async def delete(self, chunk_ids: list[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
                del self.vectors[chunk_id]
                del self.coarse[chunk_id]
                removed += 1
        return removed

    async def search(
        self, query: str, *, top_k: int = 10, where: Where | None = None
    ) -> list[SearchHit]:
        if not self.chunks:
            return []
        [full_query] = await embed_texts(self.embedder, [query], input_type="query")
        coarse_query = self._coarsen(full_query)
        eligible = [cid for cid in self.chunks if _passes(where, self.chunks[cid])]
        # Stage 1: coarse ranking over the compressed view.
        coarse_ranked = sorted(
            eligible, key=lambda cid: self._coarse_score(coarse_query, self.coarse[cid]), reverse=True
        )
        candidate_ids = coarse_ranked[: max(top_k * self.rerank_factor, top_k)]
        # Stage 2: exact rerank over full-precision vectors.
        scored = [
            SearchHit(chunk=self.chunks[cid], score=cosine(full_query, self.vectors[cid]), source=self.name)
            for cid in candidate_ids
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def stats(self) -> dict[str, Any]:
        return {
            "chunks": len(self.chunks),
            "quantization": self.quantization,
            "coarse_dims": self.coarse_dims,
            "rerank_factor": self.rerank_factor,
        }
