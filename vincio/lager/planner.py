"""Query planning — a query becomes explicit information needs.

The needs are the reasoning state the lazy loop retrieves *against*: coverage
is judged per need (structurally, by kind — see the controller), the coverage
denominator is frozen here at plan time, and an unanswered required need is
reported by name so downstream can abstain instead of guessing. Deterministic;
a user-supplied :class:`~vincio.retrieval.reasoning_retrieval.FactSchema` maps
one-to-one onto needs when the task family is known ahead of time.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from .extract import content_terms, normalize_entities

__all__ = ["InformationNeed", "QueryPlanner"]

NeedKind = Literal["lookup", "relation", "temporal", "aggregate", "causal"]

_RELATION_RE = re.compile(
    r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$|,)|"
    r"\bhow\s+(?:does|do|did)\s+(.+?)\s+(?:affect|impact|influence|relate to|depend on|cause)\s+(.+?)(?:\?|$)",
    re.IGNORECASE,
)
_TEMPORAL_RE = re.compile(r"\b(?:when|since|until|before|after|date|year|timeline)\b", re.IGNORECASE)
_AGGREGATE_RE = re.compile(r"\bhow\s+(?:many|much)\b|\b(?:total|average|median|sum|count)\b", re.IGNORECASE)
_CAUSAL_RE = re.compile(r"\bwhy\b|\bbecause\b|\bcaus\w+\b|\breason\b|\broot cause\b", re.IGNORECASE)
_CONJUNCT_SPLIT_RE = re.compile(r"(?:\?|;)\s+|\s+and\s+(?:what|who|when|where|why|how|which)\b", re.IGNORECASE)
_VERB_HINT_RE = re.compile(
    r"\b(?:is|are|was|were|does|do|did|has|have|had|can|will|would|should|happen|"
    r"work|use|make|cause|report|approve|block|launch|require|support)\b",
    re.IGNORECASE,
)


class InformationNeed(BaseModel):
    """One thing the answer requires. ``entities`` and ``kind`` drive the
    structural coverage test; ``required`` gates sufficiency."""

    text: str
    kind: NeedKind = "lookup"
    entities: list[str] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)
    required: bool = True


def _classify(text: str) -> NeedKind:
    if _RELATION_RE.search(text):
        return "relation"
    if _CAUSAL_RE.search(text):
        return "causal"  # a why-question is only answered by a causal claim
    if _AGGREGATE_RE.search(text):
        return "aggregate"
    if _TEMPORAL_RE.search(text):
        return "temporal"
    return "lookup"


def _need_from(text: str) -> InformationNeed:
    stripped = text.strip().rstrip("?.! ")
    return InformationNeed(
        text=stripped,
        kind=_classify(text),
        entities=normalize_entities(text),
        terms=sorted(set(content_terms(text)))[:12],
    )


class QueryPlanner:
    """Deterministic planning: conjunct splitting (only when each side stands
    alone), kind classification, entity extraction — and always at least one
    required need synthesized from the whole query, so sufficiency can never be
    vacuously true."""

    def plan(self, query: str) -> list[InformationNeed]:
        parts = [p for p in _CONJUNCT_SPLIT_RE.split(query) if p and p.strip()]
        needs: list[InformationNeed] = []
        if len(parts) > 1:
            for part in parts:
                # A conjunct is a need of its own only if it independently
                # carries a verb, or an entity plus a content term.
                has_verb = bool(_VERB_HINT_RE.search(part))
                has_anchor = bool(normalize_entities(part)) and len(content_terms(part)) >= 2
                if has_verb or has_anchor:
                    needs.append(_need_from(part))
        if not needs:
            needs = [_need_from(query)]
        # A relation query also needs both endpoints individually resolvable.
        relation = _RELATION_RE.search(query)
        if relation and all(n.kind != "relation" for n in needs):
            needs.insert(0, _need_from(query))
        seen: set[str] = set()
        unique: list[InformationNeed] = []
        for need in needs:
            key = need.text.lower()
            if key not in seen:
                seen.add(key)
                unique.append(need)
        return unique
