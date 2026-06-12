"""Memory eval harness.

Measures a live :class:`~vincio.memory.engine.MemoryEngine` against labeled
recall cases: recall precision/recall@k, contradiction rate, staleness, and
personalization lift (how much owner-scoped memory beats anonymous recall).
Deterministic and offline; VincioBench runs it as the ``memory`` family and
``benchmarks/budgets.json`` gates the results in CI.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.utils import utcnow
from .policies import detect_contradiction

if TYPE_CHECKING:
    from .engine import MemoryEngine

__all__ = [
    "MemoryEvalCase",
    "MemoryEvalReport",
    "evaluate_memory",
    "contradiction_rate",
    "personalization_dataset",
]


class MemoryEvalCase(BaseModel):
    """One labeled recall: which stored memories *should* surface for a
    query (matched by substring markers), and which must not."""

    id: str = ""
    query: str
    user_id: str | None = None
    agent_id: str | None = None
    tenant_id: str | None = None
    session_id: str | None = None
    task_entities: list[str] = Field(default_factory=list)
    relevant: list[str] = Field(default_factory=list)  # substrings marking relevant memories
    forbidden: list[str] = Field(default_factory=list)  # substrings that must not surface


class MemoryEvalReport(BaseModel):
    metrics: dict[str, float] = Field(default_factory=dict)
    cases: list[dict[str, Any]] = Field(default_factory=list)


def _hits(contents: list[str], markers: list[str]) -> int:
    return sum(1 for marker in markers if any(marker.lower() in c.lower() for c in contents))


def contradiction_rate(engine: MemoryEngine) -> float:
    """Fraction of active memories that contradict at least one other
    active memory of the same owner — unresolved staleness in the store."""
    by_owner: dict[tuple[str, str | None], list[str]] = {}
    for item in engine.store.all_items(statuses=("active", "validated")):
        by_owner.setdefault((item.scope.value, item.owner_id), []).append(item.content)
    total = 0
    contradictory: set[tuple[tuple[str, str | None], int]] = set()
    for owner_key, contents in by_owner.items():
        total += len(contents)
        for (i, a), (j, b) in combinations(enumerate(contents), 2):
            if detect_contradiction(a, b):
                contradictory.add((owner_key, i))
                contradictory.add((owner_key, j))
    return len(contradictory) / total if total else 0.0


async def evaluate_memory(
    engine: MemoryEngine,
    cases: list[MemoryEvalCase],
    *,
    top_k: int = 5,
) -> MemoryEvalReport:
    """Run every case against the engine and aggregate the harness metrics.

    - ``recall_precision`` — fraction of recalled items that are relevant
    - ``recall_at_k`` — fraction of relevant markers covered in the top k
    - ``staleness`` — fraction of recalled items that are expired,
      superseded, or explicitly forbidden by the case
    - ``contradiction_rate`` — unresolved contradictions in the store
    - ``personalization_lift`` — owner-scoped hit rate minus anonymous
      hit rate over the same queries
    """
    report = MemoryEvalReport()
    precisions: list[float] = []
    recalls: list[float] = []
    stale_fractions: list[float] = []
    personalized_hits: list[float] = []
    anonymous_hits: list[float] = []
    now_ids_superseded = {
        item.supersedes
        for item in engine.store.all_items(statuses=())
        if item.supersedes is not None
    }

    for case in cases:
        results = await engine.asearch(
            case.query,
            user_id=case.user_id,
            agent_id=case.agent_id,
            tenant_id=case.tenant_id,
            session_id=case.session_id,
            task_entities=case.task_entities or None,
            top_k=top_k,
        )
        contents = [r.item.content for r in results]
        relevant_items = sum(
            1
            for c in contents
            if any(marker.lower() in c.lower() for marker in case.relevant)
        )
        precision = relevant_items / len(contents) if contents else 0.0
        recall = _hits(contents, case.relevant) / len(case.relevant) if case.relevant else 1.0
        stale = 0
        for result in results:
            item = result.item
            is_forbidden = any(marker.lower() in item.content.lower() for marker in case.forbidden)
            if is_forbidden or item.id in now_ids_superseded or engine._expired(item, utcnow()):
                stale += 1
        stale_fraction = stale / len(results) if results else 0.0
        precisions.append(precision)
        recalls.append(recall)
        stale_fractions.append(stale_fraction)
        if case.relevant and (case.user_id or case.agent_id or case.session_id):
            personalized_hits.append(1.0 if _hits(contents, case.relevant) else 0.0)
            baseline = await engine.asearch(case.query, top_k=top_k)
            baseline_contents = [r.item.content for r in baseline]
            anonymous_hits.append(1.0 if _hits(baseline_contents, case.relevant) else 0.0)
        report.cases.append(
            {
                "id": case.id or case.query[:40],
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "staleness": round(stale_fraction, 4),
                "returned": len(results),
            }
        )

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    lift = _mean(personalized_hits) - _mean(anonymous_hits) if personalized_hits else 0.0
    report.metrics = {
        "recall_precision": round(_mean(precisions), 4),
        "recall_at_k": round(_mean(recalls), 4),
        "staleness": round(_mean(stale_fractions), 4),
        "contradiction_rate": round(contradiction_rate(engine), 4),
        "personalization_lift": round(lift, 4),
        "cases": float(len(cases)),
    }
    return report


def personalization_dataset() -> list[MemoryEvalCase]:
    """A small built-in case set matching the VincioBench memory corpus."""
    return [
        MemoryEvalCase(
            id="style-u1",
            query="what answer style suits this user",
            user_id="u1",
            relevant=["concise"],
            forbidden=["detailed walkthroughs"],
        ),
        MemoryEvalCase(
            id="dept-u1",
            query="which department does the user work in",
            user_id="u1",
            relevant=["compliance"],
        ),
        MemoryEvalCase(
            id="style-u2",
            query="what answer style suits this user",
            user_id="u2",
            relevant=["detailed"],
            forbidden=["concise technical answers"],
        ),
        MemoryEvalCase(
            id="timezone-u3",
            query="what timezone is the user in",
            user_id="u3",
            relevant=["UTC-5"],
            forbidden=["UTC+1"],
        ),
    ]
