"""Hybrid indexing over Evidence Objects — no single signal dominates.

Three complementary signals fused with weighted reciprocal-rank fusion:
lexical (BM25 over claim text), entity match (query entities against the
object's normalized entities/terms), and — when an embedder is configured —
dense cosine over claims. Metadata and temporal predicates filter before
fusion. Every ranking tie-breaks (score descending, id ascending), so the
index is deterministic for a fixed corpus; the default (no embedder) path is
pure stdlib.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..providers.base import run_sync
from .extract import content_terms, normalize_entities
from .objects import EvidenceObject

__all__ = ["EvidenceIndex", "fuse_ranked"]

_RRF_K = 60


def fuse_ranked(ranked_lists: list[list[str]], *, weights: list[float] | None = None) -> list[str]:
    """Weighted RRF over id lists; ties break on id so fusion is total-ordered."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = defaultdict(float)
    for ids, weight in zip(ranked_lists, weights, strict=True):
        for rank, eo_id in enumerate(ids):
            scores[eo_id] += weight / (_RRF_K + rank + 1)
    return [eo_id for eo_id, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


class EvidenceIndex:
    """Lexical + entity (+ optional dense) retrieval over Evidence Objects."""

    def __init__(self, *, embedder: Any | None = None) -> None:
        self.embedder = embedder
        self.objects: dict[str, EvidenceObject] = {}
        self._postings: dict[str, list[str]] = defaultdict(list)  # term → EO ids
        self._lengths: dict[str, int] = {}
        self._entity_postings: dict[str, list[str]] = defaultdict(list)
        self._vectors: dict[str, list[float]] = {}
        self._query_vectors: dict[str, list[float]] = {}  # need/query text → cached vector
        self._average_length = 0.0

    def __len__(self) -> int:
        return len(self.objects)

    # -- build -----------------------------------------------------------------------

    def add(self, objects: list[EvidenceObject]) -> None:
        for obj in sorted(objects, key=lambda o: o.id):
            if obj.id in self.objects:
                continue
            self.objects[obj.id] = obj
            terms = content_terms(obj.claim)
            self._lengths[obj.id] = max(len(terms), 1)
            for term in sorted(set(terms)):
                self._postings[term].append(obj.id)
            for key in (*obj.entities, *obj.terms):
                self._entity_postings[key].append(obj.id)
        total = sum(self._lengths.values())
        self._average_length = total / max(len(self._lengths), 1)
        if self.embedder is not None:
            missing = sorted(set(self.objects) - set(self._vectors))
            if missing:
                vectors = run_sync(self.embedder.embed([self.objects[i].claim for i in missing]))
                for eo_id, vector in zip(missing, vectors, strict=True):
                    self._vectors[eo_id] = vector

    # -- signals ---------------------------------------------------------------------

    def lexical(self, query: str, *, limit: int = 32) -> list[str]:
        """BM25 (k1=1.5, b=0.75) over claim terms."""
        terms = content_terms(query)
        if not terms or not self.objects:
            return []
        scores: dict[str, float] = defaultdict(float)
        corpus_size = len(self.objects)
        for term in sorted(set(terms)):
            postings = self._postings.get(term, [])
            if not postings:
                continue
            idf = math.log(1 + (corpus_size - len(postings) + 0.5) / (len(postings) + 0.5))
            frequency = terms.count(term)
            for eo_id in postings:
                length_norm = 1 - 0.75 + 0.75 * (self._lengths[eo_id] / self._average_length)
                scores[eo_id] += idf * (frequency * 2.5) / (frequency + 1.5 * length_norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [eo_id for eo_id, _ in ranked[:limit]]

    def by_entities(self, entities: list[str], *, limit: int = 32) -> list[str]:
        """Objects sharing the most query entities (count desc, id asc)."""
        counts: dict[str, int] = defaultdict(int)
        for entity in sorted(set(e.lower() for e in entities)):
            for eo_id in self._entity_postings.get(entity, []):
                counts[eo_id] += 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [eo_id for eo_id, _ in ranked[:limit]]

    def dense(self, query: str, *, limit: int = 32) -> list[str]:
        """Cosine over claim embeddings; empty when no embedder is configured."""
        if self.embedder is None or not self._vectors:
            return []
        query_vector = run_sync(self.embedder.embed([query]))[0]
        norm_q = math.sqrt(sum(v * v for v in query_vector)) or 1.0
        scored: list[tuple[float, str]] = []
        for eo_id, vector in self._vectors.items():
            norm_v = math.sqrt(sum(v * v for v in vector)) or 1.0
            cosine = sum(a * b for a, b in zip(query_vector, vector, strict=True)) / (norm_q * norm_v)
            scored.append((cosine, eo_id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [eo_id for _, eo_id in scored[:limit]]

    def _query_vector(self, text: str) -> list[float] | None:
        """Embed and memoize a need/query *text* — a whole retrieval's coverage
        checks compare many objects against a handful of distinct need texts, so
        each need is embedded at most once. ``None`` when no embedder is set."""
        if self.embedder is None:
            return None
        cached = self._query_vectors.get(text)
        if cached is not None:
            return cached
        vector = run_sync(self.embedder.embed([text]))[0]
        self._query_vectors[text] = vector
        return vector

    def semantic_similarity(self, text: str, obj: EvidenceObject) -> float | None:
        """Dense cosine in ``[0, 1]`` between *text* and *obj*'s claim embedding,
        or ``None`` when no embedder is configured or *obj* was never vectorized.

        This is the dense signal the **coverage gate** consults (the seeding
        path already fuses ``dense`` via RRF). The object vector is the one
        cached at :meth:`add` time and the need text is embedded once via
        :meth:`_query_vector`, so consulting it during coverage costs at most one
        embed per distinct need text. Returning ``None`` — the pure-stdlib
        default and the only value on the embedder-off path — leaves every caller
        on its lexical fallback, so the default coverage decision is unchanged."""
        if self.embedder is None:
            return None
        vector = self._vectors.get(obj.id)
        if vector is None:
            return None
        query_vector = self._query_vector(text)
        if query_vector is None or len(query_vector) != len(vector):
            return None
        norm_q = math.sqrt(sum(v * v for v in query_vector))
        norm_v = math.sqrt(sum(v * v for v in vector))
        if norm_q == 0.0 or norm_v == 0.0:
            return None
        cosine = sum(a * b for a, b in zip(query_vector, vector, strict=True)) / (norm_q * norm_v)
        return max(0.0, min(1.0, cosine))

    # -- fused seed ------------------------------------------------------------------

    def seed(
        self,
        query: str,
        *,
        entities: list[str] | None = None,
        limit: int = 16,
        where: Callable[[EvidenceObject], bool] | None = None,
        after: datetime | None = None,
    ) -> list[EvidenceObject]:
        """The hybrid entry ranking: lexical + entity (+ dense) fused by RRF,
        then metadata/temporal predicates applied."""
        query_entities = entities if entities is not None else normalize_entities(query)
        signal_lists = [self.lexical(query), self.by_entities(query_entities)]
        weights = [1.0, 0.8]
        if self.embedder is not None:
            signal_lists.append(self.dense(query))
            weights.append(1.0)
        fused = fuse_ranked(signal_lists, weights=weights)
        results: list[EvidenceObject] = []
        for eo_id in fused:
            obj = self.objects[eo_id]
            if where is not None and not where(obj):
                continue
            if after is not None and (obj.observed_at is None or obj.observed_at < after):
                continue
            results.append(obj)
            if len(results) >= limit:
                break
        return results
