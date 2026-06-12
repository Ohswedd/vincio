"""Retrieval engine.

Pipeline: query_understanding → query_rewrite → candidate_generation
(multi-index) → hybrid_merge (RRF) → rerank → evidence_classify →
deduplicate → context_score → return_evidence.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from pydantic import BaseModel, Field

from ..context.scoring import lexical_similarity, near_duplicate_score
from ..core.types import Chunk, EvidenceItem, Message, ModelRequest, TrustLevel
from ..providers.base import ModelProvider
from .indexes import Index, SearchFilter, SearchHit
from .rerankers import Reranker

__all__ = ["QueryPlan", "RetrievalResult", "RetrievalEngine", "reciprocal_rank_fusion"]


class QueryPlan(BaseModel):
    original: str
    rewritten: str
    subqueries: list[str] = Field(default_factory=list)
    required_facts: list[str] = Field(default_factory=list)


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
        self, query: str, *, top_k: int, where: SearchFilter | None
    ) -> list[list[SearchHit]]:
        return list(
            await asyncio.gather(
                *(index.search(query, top_k=top_k, where=where) for index in self.indexes)
            )
        )

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        where: SearchFilter | None = None,
        objective: str = "",
        use_planner: bool = True,
        multi_hop: bool = False,
        max_hops: int = 2,
    ) -> RetrievalResult:
        import time

        started = time.monotonic()
        plan = await self.plan_query(query, objective=objective) if use_planner else QueryPlan(
            original=query, rewritten=query
        )
        candidate_k = max(top_k * self.candidate_multiplier, top_k)

        queries = [plan.rewritten, *plan.subqueries]
        result_lists: list[list[SearchHit]] = []
        weights: list[float] = []
        for query_index, q in enumerate(queries):
            lists = await self._search_all(q, top_k=candidate_k, where=where)
            result_lists.extend(lists)
            # The main query counts more than subqueries.
            base = 1.0 if query_index == 0 else 0.6
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
                hop_lists: list[list[SearchHit]] = []
                for hop_query in hop_queries:
                    seen_queries.add(hop_query.lower())
                    hop_lists.extend(
                        await self._search_all(hop_query, top_k=candidate_k, where=where)
                    )
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
            metadata={"candidates": len(merged), "queries": len(queries)},
        )

    @staticmethod
    def _to_evidence(hit: SearchHit, query: str) -> EvidenceItem:
        chunk: Chunk = hit.chunk
        return EvidenceItem(
            id=chunk.citation_ref,
            source_id=chunk.document_id,
            source_type="document",
            text=chunk.text,
            page=chunk.page,
            section_path=chunk.section_path,
            trust_level=TrustLevel.UNTRUSTED_DOCUMENT,
            relevance=max(0.0, min(1.0, hit.score if hit.score <= 1.0 else lexical_similarity(chunk.text, query) + 0.5)),
            authority=float(chunk.metadata.get("authority", 0.5)),
            provenance=0.9 if chunk.source_uri else 0.5,
            token_cost=chunk.token_count,
            metadata={"chunk_id": chunk.id, "source_uri": chunk.source_uri, "retrieval_score": hit.score},
        )
