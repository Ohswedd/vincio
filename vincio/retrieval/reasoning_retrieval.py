"""Reasoning retrieval: retrieve by required facts, not just
query similarity.

A :class:`FactSchema` declares the fact types a task needs (e.g. a refund
decision needs plan, payment status, refund policy...). The engine retrieves
evidence per missing fact, tracks which facts are covered, and reports gaps —
the gaps feed insufficient-evidence behavior and agent planning.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..context.scoring import lexical_similarity
from ..core.types import EvidenceItem
from .engine import RetrievalEngine
from .indexes import Where

__all__ = [
    "FactRequirement",
    "FactSchema",
    "FactCoverage",
    "FactRetrieval",
    "ReasoningRetriever",
]


class FactRequirement(BaseModel):
    name: str  # e.g. "refund_policy"
    description: str = ""  # e.g. "the refund policy applicable to the plan"
    query_hint: str = ""  # extra search terms
    required: bool = True


class FactSchema(BaseModel):
    """Required facts for a task family (fact schema)."""

    task: str
    facts: list[FactRequirement]

    @classmethod
    def from_names(cls, task: str, names: list[str]) -> FactSchema:
        return cls(
            task=task,
            facts=[
                FactRequirement(name=name, description=name.replace("_", " ")) for name in names
            ],
        )


class FactCoverage(BaseModel):
    fact: str
    covered: bool
    evidence_ids: list[str] = Field(default_factory=list)
    best_score: float = 0.0


class FactRetrieval(BaseModel):
    """Result of fact-grounded retrieval: merged evidence plus per-fact coverage.

    ``complete`` is ``False`` while any *required* fact is still uncovered — the
    signal an agent uses to ask for more evidence rather than answer on a gap.
    """

    query: str
    task: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    coverage: list[FactCoverage] = Field(default_factory=list)
    facts_total: int = 0
    facts_covered: int = 0
    missing_facts: list[str] = Field(default_factory=list)
    complete: bool = True

    def covered(self, fact: str) -> bool:
        """Whether *fact* is covered by at least one retrieved evidence item."""
        return any(c.fact == fact and c.covered for c in self.coverage)


class ReasoningRetriever:
    def __init__(
        self,
        engine: RetrievalEngine,
        *,
        coverage_threshold: float = 0.15,
        per_fact_top_k: int = 3,
    ) -> None:
        self.engine = engine
        self.coverage_threshold = coverage_threshold
        self.per_fact_top_k = per_fact_top_k

    def _fact_query(self, fact: FactRequirement, task_query: str) -> str:
        parts = [fact.description or fact.name.replace("_", " ")]
        if fact.query_hint:
            parts.append(fact.query_hint)
        parts.append(task_query)
        return " ".join(parts)

    def _coverage(self, fact: FactRequirement, evidence: list[EvidenceItem]) -> FactCoverage:
        description = fact.description or fact.name.replace("_", " ")
        best_score = 0.0
        ids: list[str] = []
        for item in evidence:
            score = lexical_similarity(item.text or "", description)
            if score >= self.coverage_threshold:
                ids.append(item.id)
            best_score = max(best_score, score)
        return FactCoverage(
            fact=fact.name,
            covered=bool(ids),
            evidence_ids=ids,
            best_score=round(best_score, 4),
        )

    async def retrieve(
        self,
        query: str,
        schema: FactSchema,
        *,
        where: Where | None = None,
        top_k: int = 12,
    ) -> tuple[list[EvidenceItem], list[FactCoverage], dict[str, Any]]:
        """Returns (evidence, per-fact coverage, report)."""
        collected: dict[str, EvidenceItem] = {}
        # Base retrieval for the task itself.
        base = await self.engine.retrieve(query, top_k=top_k, where=where, use_planner=False)
        for item in base.evidence:
            collected[item.id] = item

        coverages: list[FactCoverage] = []
        for fact in schema.facts:
            coverage = self._coverage(fact, list(collected.values()))
            if not coverage.covered:
                # Targeted retrieval for the missing fact.
                fact_result = await self.engine.retrieve(
                    self._fact_query(fact, query),
                    top_k=self.per_fact_top_k,
                    where=where,
                    use_planner=False,
                )
                for item in fact_result.evidence:
                    item.metadata["required_fact"] = fact.name
                    collected.setdefault(item.id, item)
                coverage = self._coverage(fact, list(collected.values()))
            coverages.append(coverage)

        missing = [c.fact for c in coverages if not c.covered]
        report = {
            "facts_total": len(schema.facts),
            "facts_covered": sum(1 for c in coverages if c.covered),
            "missing_facts": missing,
            "complete": not [
                c for c, f in zip(coverages, schema.facts, strict=False) if not c.covered and f.required
            ],
        }
        return list(collected.values()), coverages, report

    async def retrieve_facts(
        self,
        query: str,
        schema: FactSchema,
        *,
        where: Where | None = None,
        top_k: int = 12,
    ) -> FactRetrieval:
        """:meth:`retrieve` packaged as a typed :class:`FactRetrieval` result."""
        evidence, coverage, report = await self.retrieve(
            query, schema, where=where, top_k=top_k
        )
        return FactRetrieval(
            query=query,
            task=schema.task,
            evidence=evidence,
            coverage=coverage,
            facts_total=int(report["facts_total"]),
            facts_covered=int(report["facts_covered"]),
            missing_facts=list(report["missing_facts"]),
            complete=bool(report["complete"]),
        )
