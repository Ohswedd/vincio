"""Learned prompt compression: an LLMLingua-style compiler pass.

Extractive compression keeps whole sentences. Learned compression goes finer:
it scores the *importance* of every token and drops the low-information ones —
fillers, redundant function words — while protecting the tokens that carry the
answer (numbers, amounts, named entities, citation markers, query terms). That
reaches a higher compression ratio than sentence extraction at the same budget.

The scorer is deterministic and offline by default (an information-content
heuristic over rarity, query relevance, and token class). An optional
``learned`` hook lets a real token-importance model drive it without changing
the call site. Either way the pass is **faithfulness-gated**: it is only
*adopted* (installed on the compiler) when it preserves the cited-fact set under
eval — :func:`compression_faithfulness` and :func:`faithfulness_preserved`
measure that, and :mod:`vincio.optimize.compression_tuning` gates the adoption.

The compressor matches the call signature of
:func:`~vincio.context.compression.extractive_compress`, so it is a drop-in for
``ContextCompiler.compressor`` and slots into the budgeting stage unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ..core.tokens import count_tokens
from .compression import (
    CompressionResult,
    extractive_compress,
    truncate_to_tokens,
)

__all__ = [
    "TokenImportanceScorer",
    "LLMLinguaCompressor",
    "salient_units",
    "compression_faithfulness",
    "faithfulness_preserved",
]

# Token classes. Protected tokens are never dropped: they carry the answer.
_PROTECTED_RE = re.compile(r"\d|%|\$|€|£")
_CITATION_RE = re.compile(r"\[[^\]]{1,60}\]")
_ENTITY_RE = re.compile(r"^[A-Z][a-zA-Z]")
# Tokenizer that keeps a bracketed citation span atomic (even when it contains
# spaces, e.g. "[Smith and Jones 2020]") so fine pruning can protect it whole.
_TOKEN_RE = re.compile(r"\[[^\]]{1,60}\]|\S+")

# High-frequency function words carry little information and compress away first.
_STOPWORDS = frozenset(
    """a an the of to in on at by for with from into over under and or but nor so yet
    is are was were be been being am do does did have has had having will would shall
    should can could may might must this that these those it its as if then than such
    there here we you they he she them his her their our your my me i us about above
    below very also just only really quite rather somewhat which who whom whose what
    when where why how not no nor very""".split()
)


# learned(tokens, query) -> per-token importance in [0, 1].
LearnedScorer = Callable[[list[str], str], list[float]]


class TokenImportanceScorer:
    """Deterministic token-importance scoring with an optional learned hook.

    Each token gets a score in ``[0, 1]``: protected tokens (numbers, amounts,
    citation markers) and named entities score at the top, query-overlapping
    and rare/long content words score high, and stopwords score near zero.
    ``learned`` overrides the heuristic entirely when supplied.
    """

    def __init__(self, *, learned: LearnedScorer | None = None) -> None:
        self.learned = learned

    @staticmethod
    def is_protected(token: str) -> bool:
        return bool(_PROTECTED_RE.search(token) or _CITATION_RE.fullmatch(token))

    def score(self, tokens: list[str], query: str) -> list[float]:
        if self.learned is not None:
            learned_scores = list(self.learned(tokens, query))
            if len(learned_scores) == len(tokens):
                return [max(0.0, min(1.0, float(s))) for s in learned_scores]
        query_terms = {t.lower() for t in re.findall(r"\w+", query)}
        scores: list[float] = []
        for token in tokens:
            lower = token.lower().strip(".,;:!?\"'()")
            if self.is_protected(token):
                scores.append(1.0)
                continue
            if not lower or not any(c.isalnum() for c in token):
                scores.append(0.05)  # bare punctuation
                continue
            if lower in _STOPWORDS:
                scores.append(0.1)
                continue
            score = 0.45
            if lower in query_terms:
                score += 0.4
            if _ENTITY_RE.match(token):  # capitalized → likely entity
                score += 0.3
            score += min(0.2, max(0, len(lower) - 4) * 0.03)  # longer words carry more
            scores.append(min(1.0, score))
        return scores


class LLMLinguaCompressor:
    """Token-importance compressor (callable, drop-in for ``extractive_compress``).

    Two stages, coarse-to-fine: first keep the most query-relevant sentences to
    a generous intermediate budget, then prune the lowest-importance tokens
    within them down to ``max_tokens`` — never dropping a protected token. If
    token pruning cannot reach the budget without sacrificing protected content,
    it falls back to extractive compression, so the budget is always honoured.
    """

    def __init__(
        self,
        *,
        scorer: TokenImportanceScorer | None = None,
        min_keep_ratio: float = 0.1,
        coarse_overshoot: float = 1.5,
    ) -> None:
        self.scorer = scorer or TokenImportanceScorer()
        self.min_keep_ratio = min_keep_ratio
        self.coarse_overshoot = coarse_overshoot

    def __call__(
        self, text: str, query: str, max_tokens: int, *, model: str | None = None
    ) -> CompressionResult:
        original = count_tokens(text, model)
        if original <= max_tokens:
            return CompressionResult(
                text=text, original_tokens=original, compressed_tokens=original, method="none"
            )
        # Coarse: keep the most relevant sentences to an intermediate budget so
        # fine pruning operates on already-relevant material.
        coarse_budget = int(max_tokens * self.coarse_overshoot)
        coarse = extractive_compress(text, query, coarse_budget, model=model)
        working = coarse.text if coarse.compressed_tokens <= coarse_budget else text

        # Fine: token-level pruning by importance, protecting the answer tokens.
        # Citation spans are tokenized atomically so a multi-word marker like
        # "[Smith and Jones 2020]" is protected whole, not fragmented.
        tokens = _TOKEN_RE.findall(working)
        if not tokens:
            return truncate_to_tokens(text, max_tokens, model=model)
        scores = self.scorer.score(tokens, query)
        order = sorted(range(len(tokens)), key=lambda i: scores[i])  # least important first
        drop: set[int] = set()
        floor = max(1, int(len(tokens) * self.min_keep_ratio))
        for index in order:
            if len(tokens) - len(drop) <= floor:
                break
            if self.scorer.is_protected(tokens[index]):
                continue
            candidate = " ".join(tok for i, tok in enumerate(tokens) if i not in drop and i != index)
            drop.add(index)
            if count_tokens(candidate, model) <= max_tokens:
                break
        kept = " ".join(tok for i, tok in enumerate(tokens) if i not in drop)
        compressed_tokens = count_tokens(kept, model)
        if not kept or compressed_tokens > max_tokens:
            # Could not hit the budget while protecting answer tokens.
            return extractive_compress(text, query, max_tokens, model=model)
        return CompressionResult(
            text=kept,
            original_tokens=original,
            compressed_tokens=compressed_tokens,
            method="llmlingua",
        )


# -- faithfulness --------------------------------------------------------------


def salient_units(text: str) -> set[str]:
    """The answer-bearing units of a passage: numbers, amounts, percentages,
    citation markers, and named entities. Compression is faithful when these
    survive; everything else is connective tissue."""
    units: set[str] = set()
    for marker in _CITATION_RE.findall(text):
        units.add(marker.lower())
    for token in text.split():
        cleaned = token.strip(".,;:!?\"'()")
        if not cleaned or _CITATION_RE.fullmatch(cleaned):
            continue  # citation markers already captured above
        if _PROTECTED_RE.search(cleaned):
            units.add(cleaned.lower())
        elif _ENTITY_RE.match(cleaned) and cleaned.lower() not in _STOPWORDS:
            units.add(cleaned.lower())
    return units


def compression_faithfulness(original: str, compressed: str, *, query: str = "") -> float:
    """Fraction of the original's salient units preserved in the compressed text.

    1.0 means every number, amount, citation, and entity survived; lower values
    mean answer-bearing content was lost. ``query`` is accepted for symmetry
    with the scorer and reserved for query-aware weighting.
    """
    original_units = salient_units(original)
    if not original_units:
        return 1.0
    kept = salient_units(compressed)
    preserved = sum(1 for unit in original_units if unit in kept)
    return round(preserved / len(original_units), 4)


def faithfulness_preserved(
    facts: list[str], compressed_text: str, *, threshold: float = 0.8
) -> bool:
    """True when the cited-fact set survives compression at ``threshold``.

    Each fact's salient units must be present in the compressed text; the gate
    passes when the mean per-fact preservation clears ``threshold``. Used to
    block adopting a compressor that would drop the evidence an answer cites.
    """
    facts = [f for f in facts if f and f.strip()]
    if not facts:
        return True
    kept = salient_units(compressed_text)
    scores: list[float] = []
    for fact in facts:
        units = salient_units(fact)
        if not units:
            continue
        scores.append(sum(1 for unit in units if unit in kept) / len(units))
    if not scores:
        return True
    return (sum(scores) / len(scores)) >= threshold
