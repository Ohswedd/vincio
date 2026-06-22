"""Retrieval engine.

Pipeline: query_understanding → query_rewrite → candidate_generation
(multi-index) → hybrid_merge (RRF) → rerank → evidence_classify →
deduplicate → context_score → return_evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from ..context.scoring import lexical_similarity, near_duplicate_score
from ..core.concurrency import gather_bounded
from ..core.tokens import count_tokens
from ..core.types import Chunk, EvidenceItem, Message, ModelRequest, TrustLevel
from ..core.utils import utcnow
from ..providers.base import ModelProvider
from .indexes import Index, SearchHit, Where
from .query_understanding import QueryExpansion, QueryUnderstanding
from .rerankers import Reranker

__all__ = ["QueryPlan", "RetrievalResult", "RetrievalEngine", "reciprocal_rank_fusion"]


class QueryPlan(BaseModel):
    original: str
    rewritten: str
    subqueries: list[str] = Field(default_factory=list)
    required_facts: list[str] = Field(default_factory=list)
    expansions: list[QueryExpansion] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    evidence: list[EvidenceItem]
    hits: list[SearchHit] = Field(default_factory=list)
    plan: QueryPlan | None = None
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


def reciprocal_rank_fusion(
    result_lists: list[list[SearchHit]], *, k: int = 60, weights: list[float] | None = None
) -> list[SearchHit]:
    """Merge ranked lists with weighted RRF (hybrid_merge)."""
    weights = weights or [1.0] * len(result_lists)
    fused: dict[str, float] = {}
    chunk_map: dict[str, SearchHit] = {}
    for hits, weight in zip(result_lists, weights, strict=False):
        for rank, hit in enumerate(hits):
            fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + weight / (k + rank + 1)
            existing = chunk_map.get(hit.chunk.id)
            if existing is None or hit.score > existing.score:
                chunk_map[hit.chunk.id] = hit
    merged = [
        SearchHit(chunk=chunk_map[cid].chunk, score=score, source="hybrid")
        for cid, score in fused.items()
    ]
    merged.sort(key=lambda h: h.score, reverse=True)
    return merged


# Fusion weight per query-understanding strategy: targeted decompositions
# rank close to subqueries; HyDE/multi-query probes are supporting signals;
# step-back generalizations only nudge the fusion.
_STRATEGY_WEIGHTS = {"decompose": 0.6, "multi_query": 0.5, "hyde": 0.5, "step_back": 0.3}

_SUBQUERY_SPLIT_RE = re.compile(r"(?i)\b(?:and|as well as|plus|also|;)\b")
_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten": {"type": "string"},
        "subqueries": {"type": "array", "items": {"type": "string"}},
        "required_facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["rewritten", "subqueries", "required_facts"],
    "additionalProperties": False,
}


class RetrievalEngine:
    def __init__(
        self,
        indexes: list[Index],
        *,
        reranker: Reranker | None = None,
        index_weights: list[float] | None = None,
        planner_provider: ModelProvider | None = None,
        planner_model: str | None = None,
        candidate_multiplier: int = 4,
        duplicate_threshold: float = 0.9,
        max_concurrency: int = 8,
        query_strategies: list[str] | None = None,
    ) -> None:
        if not indexes:
            raise ValueError("RetrievalEngine requires at least one index")
        self.indexes = indexes
        self.reranker = reranker
        self.index_weights = index_weights or [1.0] * len(indexes)
        self.planner_provider = planner_provider
        self.planner_model = planner_model
        self.candidate_multiplier = candidate_multiplier
        self.duplicate_threshold = duplicate_threshold
        self.max_concurrency = max_concurrency
        self.query_strategies = list(query_strategies or [])
        self.query_understanding = QueryUnderstanding(planner_provider, planner_model)

    # -- query planning ------------------------------------------------------------

    def _heuristic_plan(self, query: str) -> QueryPlan:
        normalized = " ".join(query.split())
        subqueries: list[str] = []
        # Split compound questions into focused subqueries.
        parts = [p.strip(" ?.") for p in _SUBQUERY_SPLIT_RE.split(normalized) if p.strip(" ?.")]
        if len(parts) > 1 and all(len(p.split()) >= 2 for p in parts):
            subqueries = [p for p in parts if len(p.split()) >= 2][:4]
        return QueryPlan(original=query, rewritten=normalized, subqueries=subqueries)

    async def plan_query(self, query: str, *, objective: str = "") -> QueryPlan:
        """Generate the retrieval plan. Uses the LLM planner when
        configured, heuristic decomposition otherwise."""
        if self.planner_provider is None or self.planner_model is None:
            return self._heuristic_plan(query)
        request = ModelRequest(
            model=self.planner_model,
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are a retrieval planner. Given a task, produce: a cleaner "
                        "rewritten search query, 2-5 focused subqueries covering every "
                        "fact needed, and the list of required fact types."
                    ),
                ),
                Message(
                    role="user",
                    content=f"Objective: {objective}\nQuestion: {query}",
                ),
            ],
            output_schema=_PLAN_SCHEMA,
            output_schema_name="query_plan",
            temperature=0.0,
        )
        try:
            response = await self.planner_provider.generate(request)
            payload = response.structured or json.loads(response.text)
            return QueryPlan(
                original=query,
                rewritten=payload.get("rewritten") or query,
                subqueries=[s for s in payload.get("subqueries", []) if s][:6],
                required_facts=[f for f in payload.get("required_facts", []) if f][:10],
            )
        except Exception:  # noqa: BLE001 - planner failure falls back to heuristics
            return self._heuristic_plan(query)

    # -- retrieval ---------------------------------------------------------------------

    async def _search_all(
        self, query: str, *, top_k: int, where: Where | None
    ) -> list[list[SearchHit]]:
        return await gather_bounded(
            (index.search(query, top_k=top_k, where=where) for index in self.indexes),
            limit=self.max_concurrency,
        )

    async def _search_many(
        self, queries: list[str], *, top_k: int, where: Where | None
    ) -> list[list[SearchHit]]:
        """Fan out every (query, index) pair concurrently, preserving the
        query-major ordering of the sequential implementation."""
        results = await gather_bounded(
            (
                index.search(query, top_k=top_k, where=where)
                for query in queries
                for index in self.indexes
            ),
            limit=self.max_concurrency,
        )
        return results

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        where: Where | None = None,
        objective: str = "",
        use_planner: bool = True,
        multi_hop: bool = False,
        max_hops: int = 2,
        strategies: list[str] | None = None,
    ) -> RetrievalResult:
        import time

        started = time.monotonic()
        plan = await self.plan_query(query, objective=objective) if use_planner else QueryPlan(
            original=query, rewritten=query
        )
        strategies = self.query_strategies if strategies is None else strategies
        if strategies:
            plan.expansions = await self.query_understanding.expand(
                query, strategies, objective=objective
            )
        candidate_k = max(top_k * self.candidate_multiplier, top_k)

        # The main query counts more than subqueries and strategy expansions.
        weighted_queries: list[tuple[str, float]] = [(plan.rewritten, 1.0)]
        weighted_queries.extend((subquery, 0.6) for subquery in plan.subqueries)
        for expansion in plan.expansions:
            base = _STRATEGY_WEIGHTS.get(expansion.strategy, 0.5)
            weighted_queries.extend((q, base) for q in expansion.queries)
        seen_query_text = set()
        deduped_queries: list[tuple[str, float]] = []
        for text, base in weighted_queries:
            key = text.lower().strip()
            if not key or key in seen_query_text:
                continue
            seen_query_text.add(key)
            deduped_queries.append((text, base))
        queries = [text for text, _base in deduped_queries]

        # All (query, index) searches run concurrently under one bound.
        result_lists = await self._search_many(queries, top_k=candidate_k, where=where)
        weights: list[float] = []
        for _text, base in deduped_queries:
            weights.extend(base * w for w in self.index_weights)

        merged = reciprocal_rank_fusion(result_lists, weights=weights)

        # Multi-hop: pull entities from the first hits and retrieve again.
        if multi_hop and merged:
            seen_queries = {q.lower() for q in queries}
            for _hop in range(max_hops - 1):
                entities: list[str] = []
                for hit in merged[:top_k]:
                    entities.extend(hit.chunk.entities)
                hop_queries = [
                    f"{plan.rewritten} {entity}"
                    for entity in dict.fromkeys(entities)
                    if entity.lower() not in seen_queries
                ][:3]
                if not hop_queries:
                    break
                for hop_query in hop_queries:
                    seen_queries.add(hop_query.lower())
                hop_lists = await self._search_many(hop_queries, top_k=candidate_k, where=where)
                if not hop_lists:
                    break
                merged = reciprocal_rank_fusion(
                    [merged, *hop_lists], weights=[2.0] + [0.4] * len(hop_lists)
                )

        # Rerank.
        if self.reranker is not None:
            merged = await self.reranker.rerank(plan.rewritten, merged, top_k=max(top_k * 2, top_k))

        # Deduplicate near-identical chunks.
        deduped: list[SearchHit] = []
        for hit in merged:
            if any(
                near_duplicate_score(hit.chunk.text, kept.chunk.text) >= self.duplicate_threshold
                for kept in deduped
            ):
                continue
            deduped.append(hit)
            if len(deduped) >= top_k:
                break

        evidence = [self._to_evidence(hit, plan.rewritten) for hit in deduped]
        return RetrievalResult(
            evidence=evidence,
            hits=deduped,
            plan=plan,
            latency_ms=int((time.monotonic() - started) * 1000),
            metadata={
                "candidates": len(merged),
                "queries": len(queries),
                "strategies": list(strategies or []),
            },
        )

    @staticmethod
    def _to_evidence(hit: SearchHit, query: str) -> EvidenceItem:
        chunk: Chunk = hit.chunk
        metadata = {"chunk_id": chunk.id, "source_uri": chunk.source_uri, "retrieval_score": hit.score}
        text = chunk.text
        token_cost = chunk.token_count
        # Sentence-window retrieval: score on the sentence, cite the window.
        window_text = chunk.metadata.get("window_text")
        if window_text and window_text != text:
            metadata["matched_sentence"] = chunk.metadata.get("matched_sentence", text)
            text = window_text
            token_cost = count_tokens(text)
        # Freshness: surface index/ingest age so downstream scoring and
        # conflict resolution can prefer recent evidence.
        indexed_at = chunk.metadata.get("indexed_at")
        if indexed_at is not None:
            metadata["indexed_at"] = indexed_at
        stamp = chunk.created_at
        if stamp is not None:
            if stamp.tzinfo is None:
                from datetime import UTC

                stamp = stamp.replace(tzinfo=UTC)
            metadata["age_days"] = round((utcnow() - stamp).total_seconds() / 86_400, 3)
        time_range = chunk.metadata.get("time_range")
        return EvidenceItem(
            id=chunk.citation_ref,
            source_id=chunk.document_id,
            source_type="document",
            text=text,
            page=chunk.page,
            time_range=tuple(time_range) if time_range else None,
            section_path=chunk.section_path,
            trust_level=TrustLevel.UNTRUSTED_DOCUMENT,
            relevance=max(0.0, min(1.0, hit.score if hit.score <= 1.0 else lexical_similarity(chunk.text, query) + 0.5)),
            authority=float(chunk.metadata.get("authority", 0.5)),
            provenance=0.9 if chunk.source_uri else 0.5,
            token_cost=token_cost,
            metadata=metadata,
        )
