"""Context compression and distillation (compress_or_distill).

Three strategies, in increasing order of aggressiveness:

1. **Truncation** — cut at sentence boundaries to fit a token budget.
2. **Extractive compression** — keep the sentences most relevant to the
   query (fast, deterministic, no model call).
3. **Evidence distillation** — convert evidence into ledger claims
  ; uses sentence extraction by default and accepts an async
   LLM distiller for higher-quality claims.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from ..core.tokens import count_tokens
from ..core.types import EvidenceItem
from .scoring import lexical_similarity

__all__ = [
    "CompressionResult",
    "split_sentences",
    "truncate_to_tokens",
    "extractive_compress",
    "distill_evidence_ledger",
    "link_entailments",
]

# Latin punctuation normally separates sentences at whitespace; CJK sentence
# punctuation also separates adjacent sentences without whitespace.  A trailing
# ``[citation]`` belongs to the preceding sentence and must not be detached,
# because citation coverage and entailment are evaluated sentence by sentence.
_SENTENCE_RE = re.compile(r"(?<=[.!?])(?!\s*\[)\s+|(?<=[。！？])(?!\s*\[)\s*|\n{2,}")


class CompressionResult(BaseModel):
    text: str
    original_tokens: int
    compressed_tokens: int
    method: str

    @property
    def ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def truncate_to_tokens(
    text: str, max_tokens: int, *, model: str | None = None
) -> CompressionResult:
    original = count_tokens(text, model)
    if original <= max_tokens:
        return CompressionResult(
            text=text, original_tokens=original, compressed_tokens=original, method="none"
        )
    kept: list[str] = []
    used = 0
    for sentence in split_sentences(text):
        tokens = count_tokens(sentence, model)
        if used + tokens > max_tokens:
            break
        kept.append(sentence)
        used += tokens
    if not kept:  # single huge sentence: hard character cut proportional to budget
        approx_chars = max(1, int(len(text) * max_tokens / original))
        truncated = text[:approx_chars]
        return CompressionResult(
            text=truncated,
            original_tokens=original,
            compressed_tokens=count_tokens(truncated, model),
            method="hard_truncate",
        )
    result = " ".join(kept)
    return CompressionResult(
        text=result,
        original_tokens=original,
        compressed_tokens=count_tokens(result, model),
        method="truncate",
    )


def _verified_truncate(text: str, max_tokens: int) -> tuple[str, int]:
    """Cut *text* to **at most** ``max_tokens``, re-verified against the live
    token counter.

    :func:`truncate_to_tokens`'s single-huge-sentence path estimates a
    proportional character cut and can land *over* the target on token-dense
    text (CJK, code, long identifiers). Callers whose budget is a hard invariant
    — the pinned reservation — need the bound guaranteed: whole-word binary
    search on the verified count, then character halving for a single word
    denser than the budget. Deterministic; returns ``("", 0)`` only when not
    even one character fits."""
    if max_tokens <= 0:
        return "", 0
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    cut = " ".join(words[:lo])
    if not cut and text.strip():  # a single word denser than the budget
        cut = text.strip()
    while cut and count_tokens(cut) > max_tokens:
        cut = cut[: len(cut) // 2].rstrip()
    return cut, count_tokens(cut) if cut else 0


def extractive_compress(
    text: str,
    query: str,
    max_tokens: int,
    *,
    model: str | None = None,
    position_bonus: float = 0.1,
) -> CompressionResult:
    """Keep the sentences most relevant to *query* within *max_tokens*.

    Sentences keep their original order in the output. Early sentences get a
    small bonus (documents often lead with definitions and context).
    """
    original = count_tokens(text, model)
    if original <= max_tokens:
        return CompressionResult(
            text=text, original_tokens=original, compressed_tokens=original, method="none"
        )
    sentences = split_sentences(text)
    scored = []
    for position, sentence in enumerate(sentences):
        score = lexical_similarity(sentence, query)
        score += position_bonus * (1.0 - position / max(1, len(sentences)))
        scored.append((score, position, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)

    chosen: list[tuple[int, str]] = []
    used = 0
    for _score, position, sentence in scored:
        tokens = count_tokens(sentence, model)
        if used + tokens > max_tokens:
            continue
        chosen.append((position, sentence))
        used += tokens
    if not chosen:
        return truncate_to_tokens(text, max_tokens, model=model)
    chosen.sort()
    result = " ".join(sentence for _, sentence in chosen)
    return CompressionResult(
        text=result,
        original_tokens=original,
        compressed_tokens=count_tokens(result, model),
        method="extractive",
    )


# -- evidence ledger -----------------------------------------------------------

LedgerDistiller = Callable[[list[EvidenceItem], str], Awaitable[list[dict[str, Any]]]]


async def distill_evidence_ledger(
    evidence: list[EvidenceItem],
    query: str,
    *,
    max_claims_per_item: int = 3,
    distiller: LedgerDistiller | None = None,
) -> list[dict[str, Any]]:
    """Build an evidence ledger: claims with source provenance.

    With an LLM *distiller*, claims are model-extracted; otherwise the top
    query-relevant sentences of each evidence item become the claims, scored
    by lexical relevance as a confidence proxy.
    """
    if distiller is not None:
        return link_entailments(await distiller(evidence, query))
    ledger: list[dict[str, Any]] = []
    counter = 1
    for item in evidence:
        text = item.scorable_text or item.text or ""
        if not text:
            continue
        sentences = split_sentences(text)
        scored = sorted(
            ((lexical_similarity(s, query), s) for s in sentences),
            key=lambda pair: pair[0],
            reverse=True,
        )[:max_claims_per_item]
        for score, sentence in scored:
            if score <= 0.0 and len(ledger) > 0:
                continue
            ledger.append(
                {
                    "id": f"E{counter}",
                    "source": item.citation_ref,
                    "evidence_id": item.id,
                    "claim": sentence,
                    "supports": [],
                    "confidence": round(min(0.99, 0.5 + score / 2), 2),
                }
            )
            counter += 1
    return link_entailments(ledger)


_LEDGER_NUMERIC_RE = re.compile(r"\d")


def link_entailments(
    ledger: list[dict[str, Any]], *, support_threshold: float = 0.5
) -> list[dict[str, Any]]:
    """Populate ``supports`` / ``contradicts`` links between ledger claims via
    entailment, turning a flat claim list into a claim/contradiction graph.

    Two claims that are topically similar (lexical overlap ≥ ``support_threshold``)
    *corroborate* each other when they come from different sources and agree on
    their salient numeric units, and *contradict* when those numeric units
    disagree — so a downstream check can surface conflicting evidence instead of
    silently averaging it.
    """
    from .llmlingua import salient_units  # lazy: llmlingua imports this module

    for entry in ledger:
        entry.setdefault("supports", [])
        entry["contradicts"] = []
    numeric: dict[str, set[str]] = {
        entry["id"]: {
            u for u in salient_units(entry.get("claim", "")) if _LEDGER_NUMERIC_RE.search(u)
        }
        for entry in ledger
    }
    for a in ledger:
        a_claim = a.get("claim", "")
        for b in ledger:
            if a["id"] == b["id"]:
                continue
            if lexical_similarity(a_claim, b.get("claim", "")) < support_threshold:
                continue
            a_nums, b_nums = numeric[a["id"]], numeric[b["id"]]
            if a_nums and b_nums and a_nums.isdisjoint(b_nums):
                a["contradicts"].append(b["id"])
            elif b.get("source") != a.get("source") and b["id"] not in a["supports"]:
                a["supports"].append(b["id"])
    return ledger
