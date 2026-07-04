"""Answering over evidence — generation, verification, confidence.

The model receives only the pack's verified minimum: one line per Evidence
Object tagged ``[eo:…]``, contradicting pairs flagged inline so the model must
reconcile them rather than silently pick a side, and an explicit instruction to
abstain when the pack was insufficient. Every generated answer stays traceable:
``verify`` re-derives each cited object from its source bytes offline, and
``estimate_confidence`` is a deterministic function of coverage, evidence
confidence, and contradiction pressure — never a model's self-report.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..core.types import Message, ModelRequest
from .objects import EvidencePack

__all__ = ["LagerAnswer", "build_context", "cited_ids", "estimate_confidence", "verify_answer"]

_CITATION_RE = re.compile(r"\[(eo:[0-9a-f]{16})\]")


class LagerAnswer(BaseModel):
    """A generated answer bound to the pack that produced it."""

    query: str
    text: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    sufficient: bool = False
    uncovered_needs: list[str] = Field(default_factory=list)
    pack: EvidencePack


def build_context(pack: EvidencePack) -> str:
    """The minimum evidence context: one tagged line per object, contradictions
    flagged inline, insufficiency stated plainly."""
    contradicting: dict[str, list[str]] = {}
    for a, b, _basis in pack.contradictions:
        contradicting.setdefault(a, []).append(b)
        contradicting.setdefault(b, []).append(a)
    lines: list[str] = []
    for obj in pack.objects:
        flag = ""
        if obj.id in contradicting:
            others = ", ".join(sorted(contradicting[obj.id]))
            flag = f" (CONFLICTS WITH {others} — reconcile explicitly)"
        lines.append(f"[{obj.id}] {obj.claim}{flag}")
    if not pack.sufficient:
        missing = "; ".join(pack.uncovered_needs) or "the question"
        lines.append(
            f"NOTE: the evidence is INSUFFICIENT to answer: {missing}. "
            "Say so rather than guessing."
        )
    return "\n".join(lines)


_SYSTEM = (
    "Answer strictly from the numbered evidence lines. Cite every factual "
    "statement with its [eo:…] tag. If lines conflict, say which you relied on "
    "and why. If the evidence is marked insufficient, state what is missing "
    "instead of guessing."
)


async def generate_answer(
    query: str,
    pack: EvidencePack,
    *,
    provider,
    model: str,
    temperature: float = 0.0,
) -> LagerAnswer:
    """One grounded completion over the pack (any Vincio provider)."""
    request = ModelRequest(
        model=model,
        messages=[
            Message(role="system", content=_SYSTEM),
            Message(role="user", content=f"Evidence:\n{build_context(pack)}\n\nQuestion: {query}"),
        ],
        temperature=temperature,
    )
    response = await provider.generate(request)
    text = response.text or ""
    return LagerAnswer(
        query=query,
        text=text,
        citations=cited_ids(text),
        confidence=estimate_confidence(pack),
        sufficient=pack.sufficient,
        uncovered_needs=list(pack.uncovered_needs),
        pack=pack,
    )


def cited_ids(text: str) -> list[str]:
    """The [eo:…] tags an answer actually used, in order, deduplicated."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _CITATION_RE.finditer(text):
        if match.group(1) not in seen:
            seen.add(match.group(1))
            ordered.append(match.group(1))
    return ordered


def verify_answer(
    answer: LagerAnswer, documents_text: dict[str, str]
) -> tuple[bool, list[str]]:
    """Offline verification: every citation resolves to a pack object, and every
    pack object re-derives byte-for-byte from its source (keyed by ``doc_key``).
    Returns (ok, problems) — problems name exactly what failed."""
    problems: list[str] = []
    pack_ids = {obj.id for obj in answer.pack.objects}
    for citation in answer.citations:
        if citation not in pack_ids:
            problems.append(f"citation {citation} is not in the evidence pack")
    for obj in answer.pack.objects:
        text = documents_text.get(obj.doc_key)
        if text is None:
            problems.append(f"{obj.id}: source document missing")
        elif not obj.verify(text):
            problems.append(f"{obj.id}: does not re-derive from its source span")
    return (not problems, problems)


def estimate_confidence(pack: EvidencePack) -> float:
    """Deterministic confidence: required-need coverage × mean evidence
    confidence × a contradiction penalty. Zero when the pack is empty."""
    if not pack.objects:
        return 0.0
    covered = sum(1 for ids in pack.coverage.values() if ids)
    coverage_score = covered / max(len(pack.coverage), 1)
    mean_confidence = sum(o.confidence for o in pack.objects) / len(pack.objects)
    contradiction_penalty = 1.0 / (1.0 + 0.5 * len(pack.contradictions))
    return round(coverage_score * mean_confidence * contradiction_penalty, 4)
