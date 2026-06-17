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
from typing import Any, Protocol

from ..core.types import Chunk
from .filters import as_predicate
from .indexes import SearchHit, Where

__all__ = [
    "SparseVector",
    "SparseEncoder",
    "LocalImpactEncoder",
    "CallableSparseEncoder",
    "SpladeEncoder",
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


class SpladeEncoder:
    """Real SPLADE learned-sparse encoder via a local ``transformers`` model (2.1).

    SPLADE expands a passage into term-impact weights over the model vocabulary —
    far richer than the offline :class:`LocalImpactEncoder` approximation. Lazily
    loads a Hugging Face masked-LM (the model the SPLADE checkpoint fine-tuned),
    computes the log-saturated max-pooled term weights, and returns them as a
    :data:`SparseVector`. Injectable via ``encode_fn`` for offline tests, with a
    fallback to :class:`LocalImpactEncoder` when the dependency is missing and
    ``fallback=True``. Install with ``pip install "vincio[splade]"``.
    """

    def __init__(
        self,
        model_name: str = "naver/splade-v3",
        *,
        encode_fn: Callable[[list[str], bool], list[SparseVector]] | None = None,
        fallback: bool = False,
        top_k: int = 256,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self._encode_fn = encode_fn
        self._fallback = fallback
        self._model: Any = None
        self._tokenizer: Any = None
        self._fallback_encoder: LocalImpactEncoder | None = None

    def _ensure(self) -> None:
        if (
            self._encode_fn is not None
            or self._model is not None
            or self._fallback_encoder is not None
        ):
            return
        try:
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForMaskedLM,
                AutoTokenizer,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        except ImportError as exc:
            if self._fallback:
                self._fallback_encoder = LocalImpactEncoder()
                return
            from ..core.errors import ConfigError

            raise ConfigError(
                'the SPLADE encoder requires: pip install "vincio[splade]" '
                "(or construct with fallback=True / inject encode_fn)"
            ) from exc

    def _encode_model(self, texts: list[str]) -> list[SparseVector]:  # pragma: no cover - needs torch
        import torch

        vectors: list[SparseVector] = []
        tokenizer, model = self._tokenizer, self._model
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            # SPLADE term importance: log(1 + relu(logits)) max-pooled over tokens.
            weights = torch.max(
                torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1), dim=1
            ).values.squeeze()
            top = torch.topk(weights, k=min(self.top_k, weights.shape[-1]))
            vector: SparseVector = {}
            for value, idx in zip(top.values.tolist(), top.indices.tolist(), strict=False):
                if value > 0:
                    vector[tokenizer.convert_ids_to_tokens(int(idx))] = float(value)
            vectors.append(vector)
        return vectors

    async def encode(self, texts: list[str], *, is_query: bool = False) -> list[SparseVector]:
        self._ensure()
        if self._encode_fn is not None:
            return list(self._encode_fn(texts, is_query))
        if self._fallback_encoder is not None:
            return await self._fallback_encoder.encode(texts, is_query=is_query)
        return self._encode_model(texts)


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
        self, query: str, *, top_k: int = 10, where: Where | None = None
    ) -> list[SearchHit]:
        if not self.chunks:
            return []
        predicate = as_predicate(where)
        [query_vector] = await self.encoder.encode([query], is_query=True)
        scores: dict[str, float] = defaultdict(float)
        for term, query_weight in query_vector.items():
            for chunk_id, doc_weight in self._postings.get(term, {}).items():
                scores[chunk_id] += query_weight * doc_weight
        hits: list[SearchHit] = []
        for chunk_id, score in scores.items():
            chunk = self.chunks[chunk_id]
            if predicate is not None and not predicate(chunk):
                continue
            hits.append(SearchHit(chunk=chunk, score=score, source=self.name))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
