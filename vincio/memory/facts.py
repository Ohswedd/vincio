"""Auto-memory from runs: grounded-fact extraction.

High-confidence, well-grounded facts surfaced during a run become candidate
memories. A fact qualifies only when it is a verifiable claim from the run's
output *and* the cited evidence supports it (lexical support above a
threshold). Extraction is deterministic and offline; admission still goes
through the guarded write policy (privacy, stability, contradiction,
confidence), and the items land as *candidates* — they carry a status
penalty in recall until confirmed.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..context.compression import split_sentences
from ..context.scoring import lexical_similarity
from ..core.types import EvidenceItem

__all__ = ["GroundedFact", "extract_grounded_facts"]

_VERIFIABLE_RE = re.compile(
    r"\d|%|\$|€|\b(is|are|was|were|has|have|will|must|requires?)\b", re.IGNORECASE
)
_CITATION_MARKER_RE = re.compile(r"\s*\[[^\]]{1,60}\]")


class GroundedFact(BaseModel):
    """One output claim with measured evidence support."""

    content: str
    support: float  # max lexical support against the supplied evidence, 0..1
    evidence_ids: list[str] = Field(default_factory=list)


def extract_grounded_facts(
    text: str,
    evidence: list[EvidenceItem],
    *,
    min_support: float = 0.5,
    max_facts: int = 5,
) -> list[GroundedFact]:
    """Extract verifiable, evidence-supported claims from run output.

    Sentences must look like factual claims (numbers, amounts, or copular /
    modal verbs) and reach ``min_support`` lexical similarity against at
    least one evidence item. Returns at most ``max_facts``, strongest
    support first; citation markers are stripped from the stored content.
    """
    if not text or not evidence:
        return []
    # Strip citation markers before splitting: "... days. [D1:C0] Next ..."
    # must split into two sentences, and markers must not be stored.
    stripped = _CITATION_MARKER_RE.sub("", text)
    facts: list[GroundedFact] = []
    for sentence in split_sentences(stripped):
        clean = sentence.strip()
        if len(clean.split()) < 4 or len(clean) > 400:
            continue
        if not _VERIFIABLE_RE.search(clean):
            continue
        support = 0.0
        supporting: list[str] = []
        for item in evidence:
            if not item.text:
                continue
            similarity = lexical_similarity(clean, item.text)
            if similarity > support:
                support = similarity
            if similarity >= min_support:
                supporting.append(item.id)
        if support < min_support:
            continue
        facts.append(
            GroundedFact(content=clean, support=round(support, 4), evidence_ids=supporting[:4])
        )
    facts.sort(key=lambda fact: fact.support, reverse=True)
    return facts[:max_facts]
