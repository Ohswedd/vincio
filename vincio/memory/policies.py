"""Memory write policy and lifecycle.

Write pipeline: extract candidates → classify type → privacy check →
stability check → contradiction check → assign confidence → write with
provenance. Decay follows ``confidence_t = c0 · e^(−λ·age) · usage_boost ·
confirmation_boost``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from ..context.scoring import near_duplicate_score
from ..core.types import MemoryItem, MemoryScope, MemoryType, PrivacyClass
from ..core.utils import utcnow
from ..security.pii import PIIDetector

__all__ = [
    "MemoryCandidate",
    "extract_memory_candidates",
    "classify_memory_type",
    "stability_score",
    "decayed_confidence",
    "importance_score",
    "detect_contradiction",
    "MemoryWritePolicy",
]


class MemoryCandidate(BaseModel):
    content: str
    type: MemoryType = MemoryType.FACT
    scope: MemoryScope = MemoryScope.USER
    confidence: float = 0.7
    entities: list[str] = Field(default_factory=list)
    privacy_class: PrivacyClass = PrivacyClass.INTERNAL
    metadata: dict[str, Any] = Field(default_factory=dict)


_PREFERENCE_RE = re.compile(
    r"(?i)\b(?:prefer|prefers|likes?|dislikes?|always want|please always|never send|favorite|rather)\b"
)
_GOAL_RE = re.compile(r"(?i)\b(?:goal|aiming|objective is|trying to|plan to|by (?:end of|next))\b")
_DECISION_RE = re.compile(r"(?i)\b(?:decided|we will|agreed to|approved|chosen|going with|signed off)\b")
_FACT_CUE_RE = re.compile(
    r"(?i)\b(?:is|are|was|were|has|have|works at|based in|uses|runs on|costs?|renews?)\b"
)
_VOLATILE_RE = re.compile(
    r"(?i)\b(?:today|tomorrow|right now|currently|at the moment|this (?:week|morning|afternoon)|for now|temporarily)\b"
)
_FIRST_PERSON_DURABLE_RE = re.compile(r"(?i)\b(?:my|our|i|we)\b")


def classify_memory_type(content: str) -> MemoryType:
    if _PREFERENCE_RE.search(content):
        return MemoryType.PREFERENCE
    if _DECISION_RE.search(content):
        return MemoryType.DECISION
    if _GOAL_RE.search(content):
        return MemoryType.GOAL
    return MemoryType.FACT


def stability_score(content: str) -> float:
    """How durable is this statement? Volatile time markers reduce stability."""
    score = 0.7
    if _VOLATILE_RE.search(content):
        score -= 0.45
    if _PREFERENCE_RE.search(content) or _DECISION_RE.search(content):
        score += 0.2
    if _FACT_CUE_RE.search(content):
        score += 0.1
    return max(0.0, min(1.0, score))


# Sentences worth remembering: durable user/tenant statements.
def extract_memory_candidates(
    text: str,
    *,
    scope: MemoryScope = MemoryScope.USER,
    min_stability: float = 0.4,
) -> list[MemoryCandidate]:
    """Heuristic candidate extraction (step 1). An LLM extractor
    can replace this via MemoryWritePolicy(extractor=...)."""
    from ..context.compression import split_sentences

    candidates: list[MemoryCandidate] = []
    for sentence in split_sentences(text or ""):
        sentence = sentence.strip()
        if len(sentence.split()) < 4 or len(sentence) > 400:
            continue
        memory_type = classify_memory_type(sentence)
        stability = stability_score(sentence)
        if stability < min_stability:
            continue
        is_personal = bool(_FIRST_PERSON_DURABLE_RE.search(sentence))
        if memory_type == MemoryType.FACT and not is_personal:
            # Generic world facts are retrieval's job, not memory's.
            continue
        candidates.append(
            MemoryCandidate(
                content=sentence,
                type=memory_type,
                scope=scope,
                confidence=0.55 + 0.3 * stability,
                metadata={"stability": round(stability, 3)},
            )
        )
    return candidates


def decayed_confidence(item: MemoryItem, *, decay_lambda: float = 0.01) -> float:
    """Decay formula."""
    updated = item.updated_at
    if updated.tzinfo is None:
        from datetime import UTC

        updated = updated.replace(tzinfo=UTC)
    age_days = max(0.0, (utcnow() - updated).total_seconds() / 86_400)
    usage_boost = 1.0 + min(0.5, 0.05 * item.usage_count)
    confirmation_boost = 1.0 + min(0.5, 0.15 * item.confirmations)
    return min(1.0, item.confidence * math.exp(-decay_lambda * age_days) * usage_boost * confirmation_boost)


_TYPE_IMPORTANCE: dict[MemoryType, float] = {
    MemoryType.PREFERENCE: 0.9,
    MemoryType.DECISION: 0.85,
    MemoryType.GOAL: 0.8,
    MemoryType.SUMMARY: 0.6,
    MemoryType.ENTITY: 0.55,
    MemoryType.RELATIONSHIP: 0.55,
    MemoryType.FACT: 0.5,
}


def importance_score(item: MemoryItem) -> float:
    """How costly it would be to forget this item.

    Importance-weighted retention: heavily used, confirmed, stable
    preferences/decisions survive longer than incidental facts."""
    type_weight = _TYPE_IMPORTANCE.get(item.type, 0.5)
    stability = float(item.metadata.get("stability", 0.7))
    usage = min(1.0, 0.1 * item.usage_count)
    confirmations = min(1.0, 0.25 * item.confirmations)
    score = 0.4 * type_weight + 0.25 * stability + 0.2 * usage + 0.15 * confirmations
    return max(0.0, min(1.0, score))


_NEGATION_RE = re.compile(r"(?i)\b(?:not|no longer|never|stopped|isn't|aren't|doesn't|don't|won't)\b")


def detect_contradiction(new_content: str, old_content: str) -> bool:
    """Same subject, opposite polarity, or same slot with different value."""
    similarity = near_duplicate_score(new_content, old_content)
    if similarity < 0.45:
        return False
    if similarity >= 0.95:
        return False  # restatement, not contradiction
    negation_differs = bool(_NEGATION_RE.search(new_content)) != bool(_NEGATION_RE.search(old_content))
    if negation_differs:
        return True
    # Same subject/verb but diverging values ("prefers X" vs "prefers Y").
    # If one statement's terms are a subset of the other's, it is a
    # refinement, not a contradiction.
    new_terms = set(new_content.lower().split())
    old_terms = set(old_content.lower().split())
    new_only = new_terms - old_terms
    old_only = old_terms - new_terms
    return bool(new_only) and bool(old_only) and similarity >= 0.55


Extractor = Callable[[str], Awaitable[list[MemoryCandidate]]]


class MemoryWritePolicy:
    """Guards every memory write."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.4,
        min_stability: float = 0.4,
        supersede_margin: float = 0.1,
        pii_detector: PIIDetector | None = None,
        allow_pii: bool = False,
        extractor: Extractor | None = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_stability = min_stability
        self.supersede_margin = supersede_margin
        self.pii = pii_detector or PIIDetector()
        self.allow_pii = allow_pii
        self.extractor = extractor

    async def extract(self, text: str, *, scope: MemoryScope = MemoryScope.USER) -> list[MemoryCandidate]:
        if self.extractor is not None:
            return await self.extractor(text)
        return extract_memory_candidates(text, scope=scope, min_stability=self.min_stability)

    def admit(self, candidate: MemoryCandidate) -> tuple[bool, str]:
        """Steps 3-6: privacy, stability, confidence. Returns (ok, reason)."""
        matches = self.pii.detect(candidate.content)
        secrets = [m for m in matches if m.type in ("api_key", "secret")]
        if secrets:
            return False, "contains_credentials"
        sensitive = [m for m in matches if m.confidence >= 0.8 and m.type in ("government_id", "credit_card", "health")]
        if sensitive:
            candidate.privacy_class = PrivacyClass.SENSITIVE
            if not self.allow_pii:
                return False, "sensitive_pii_blocked"
        elif any(m.confidence >= 0.8 for m in matches):
            candidate.privacy_class = PrivacyClass.PII
        stability = float(candidate.metadata.get("stability", stability_score(candidate.content)))
        if stability < self.min_stability:
            return False, "unstable_content"
        if candidate.confidence < self.min_confidence:
            return False, "low_confidence"
        return True, "ok"

    def resolve_conflict(self, new: MemoryCandidate, old: MemoryItem) -> str:
        """Supersede when clearly more confident, else flag."""
        if not detect_contradiction(new.content, old.content):
            return "none"
        if new.confidence > old.confidence + self.supersede_margin:
            return "supersede"
        return "conflict"
