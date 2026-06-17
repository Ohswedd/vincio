"""Context candidate scoring.

Implements the context item utility:

    Score(c_i | τ) = w_r·relevance + w_n·novelty + w_a·authority + w_f·freshness
                   + w_p·provenance + w_q·answerability + w_m·memory_value
                   - w_t·token_cost - w_d·duplication - w_k·leakage_risk

All component scores are normalized to [0, 1]. Relevance defaults to a fast
lexical estimator and accepts an embedding-similarity callback for semantic
scoring when an embedder is available.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.tokens import count_tokens
from ..core.utils import utcnow
from ..retrieval.embeddings import cosine

__all__ = [
    "CandidateType",
    "ContextScores",
    "ScoringWeights",
    "ContextCandidate",
    "ContextScorer",
    "lexical_similarity",
    "shingle_similarity",
    "containment_similarity",
    "near_duplicate_score",
]

CandidateType = Literal[
    "instruction", "memory", "evidence", "tool_result", "example", "schema", "policy"
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the and or but if then else of to in on at for with by from as is are was were be been "
    "this that these those it its do does did not no can could should would will i you he she we "
    "they what which who when where how why".split()
)


def _stem(token: str) -> str:
    """Light suffix stripping so morphological variants match
    (refunds→refund, annually/annual→annu, payment/pays→pay)."""
    if len(token) <= 3:
        return token
    for suffix in ("ingly", "ment", "ness", "ally", "ing", "ed", "ly", "ies", "es", "al", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


@lru_cache(maxsize=4096)
def _terms(text: str) -> frozenset[str]:
    """Stemmed content terms. Memoized: the O(n²) dedupe/conflict loops and
    incremental recompiles hit the same texts repeatedly."""
    return frozenset(
        _stem(t)
        for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())
    )


def lexical_similarity(a: str, b: str) -> float:
    """Symmetric lexical overlap in [0, 1] with IDF-free dampening."""
    terms_a, terms_b = _terms(a), _terms(b)
    if not terms_a or not terms_b:
        return 0.0
    overlap = len(terms_a & terms_b)
    return overlap / math.sqrt(len(terms_a) * len(terms_b))


@lru_cache(maxsize=4096)
def _shingles(text: str, size: int = 3) -> frozenset[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < size:
        return frozenset({" ".join(tokens)}) if tokens else frozenset()
    return frozenset(" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1))


def shingle_similarity(a: str, b: str, *, size: int = 3) -> float:
    """Jaccard similarity over word shingles — near-duplicate detection."""
    sa, sb = _shingles(a, size), _shingles(b, size)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def containment_similarity(a: str, b: str) -> float:
    """Max containment of content terms — catches near-duplicates that differ
    by a few filler words (where shingle Jaccard under-reports)."""
    terms_a, terms_b = _terms(a), _terms(b)
    if not terms_a or not terms_b:
        return 0.0
    overlap = len(terms_a & terms_b)
    return max(overlap / len(terms_a), overlap / len(terms_b))


def near_duplicate_score(a: str, b: str) -> float:
    return max(shingle_similarity(a, b), containment_similarity(a, b))


class ContextScores(BaseModel):
    relevance: float = 0.0
    novelty: float = 1.0
    authority: float = 0.5
    freshness: float = 0.5
    provenance: float = 0.5
    question_answerability: float = 0.0
    memory_value: float = 0.0
    token_cost: float = 0.0  # normalized [0,1]
    duplication: float = 0.0
    leakage_risk: float = 0.0
    total: float = 0.0


class ScoringWeights(BaseModel):
    """w_* scoring weights. Defaults favor relevant, novel, grounded items."""

    relevance: float = 1.0
    novelty: float = 0.3
    authority: float = 0.25
    freshness: float = 0.15
    provenance: float = 0.2
    question_answerability: float = 0.5
    memory_value: float = 0.3
    token_cost: float = 0.2
    duplication: float = 0.8
    leakage_risk: float = 2.0


class ContextCandidate(BaseModel):
    """A scored candidate for inclusion in the context packet."""

    id: str
    type: CandidateType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_cost: int = 0
    scores: ContextScores = Field(default_factory=ContextScores)
    source: Any = None  # original EvidenceItem / MemoryItem / ToolResult
    required: bool = False  # instructions/schema/policy are always kept
    created_at: datetime | None = None
    authority: float = 0.5
    provenance: float = 0.5
    leakage_risk: float = 0.0


class ContextScorer:
    # Weight on the reranker's cross-encoder verdict when blending it into the
    # relevance term (the rest is the query similarity). The cross-encoder is
    # the stronger signal, so it dominates while similarity still contributes.
    _UPSTREAM_BLEND = 0.65

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        *,
        similarity_fn: Callable[[str, str], float] | None = None,
        max_token_cost: int = 2000,
        freshness_half_life_days: float = 90.0,
    ) -> None:
        self.weights = weights or ScoringWeights()
        self.similarity_fn = similarity_fn or lexical_similarity
        self.max_token_cost = max_token_cost
        self.freshness_half_life_days = freshness_half_life_days
        # Optional embedding vectors keyed by candidate/query text. When set
        # (opt-in semantic scoring), relevance, novelty, duplication, and
        # near-duplicate detection use cosine over these instead of lexical
        # overlap. Defaults stay lexical when unset — fully additive.
        self._vectors: dict[str, list[float]] | None = None

    def set_embeddings(self, vectors: dict[str, list[float]] | None) -> None:
        """Install (or clear) the embedding vector cache used for semantic scoring."""
        self._vectors = vectors or None

    @property
    def semantic(self) -> bool:
        return self._vectors is not None

    def _semantic_sim(self, a: str, b: str) -> float | None:
        """Cosine over cached embeddings, or ``None`` when either vector is absent."""
        if self._vectors is None:
            return None
        va = self._vectors.get(a)
        vb = self._vectors.get(b)
        if va is None or vb is None:
            return None
        return max(0.0, min(1.0, cosine(va, vb)))

    def query_similarity(self, content: str, query: str) -> float:
        """Semantic cosine vs the query when embeddings are present; else lexical."""
        sim = self._semantic_sim(content, query)
        if sim is not None:
            return sim
        return self.similarity_fn(content, query)

    def diversity_similarity(self, a: str, b: str) -> float:
        """Similarity used by novelty/duplication/near-dup: semantic when
        embeddings are present, otherwise word-shingle overlap."""
        sim = self._semantic_sim(a, b)
        if sim is not None:
            return sim
        return shingle_similarity(a, b)

    def near_duplicate(self, a: str, b: str) -> float:
        """Near-duplicate score: semantic cosine when embeddings are present
        (catches paraphrases that differ in wording), otherwise the lexical
        shingle/containment maximum."""
        sim = self._semantic_sim(a, b)
        if sim is not None:
            return sim
        return near_duplicate_score(a, b)

    # -- component scores -------------------------------------------------------

    def relevance(self, candidate: ContextCandidate, query: str) -> float:
        if not query or not candidate.content:
            return 0.0
        base = min(1.0, self.query_similarity(candidate.content, query))
        # In semantic mode, blend the reranker's verdict (set on evidence as
        # ``upstream_relevance``) into relevance instead of using it only as a
        # min-relevance gate, so the cross-encoder drives selection and
        # ordering. Default (lexical) behavior is unchanged.
        if self.semantic:
            upstream = candidate.metadata.get("upstream_relevance")
            if upstream is not None:
                up = max(0.0, min(1.0, float(upstream)))
                return self._UPSTREAM_BLEND * up + (1.0 - self._UPSTREAM_BLEND) * base
        return base

    def freshness(self, candidate: ContextCandidate) -> float:
        if candidate.created_at is None:
            return 0.5
        created = candidate.created_at
        if created.tzinfo is None:
            from datetime import UTC

            created = created.replace(tzinfo=UTC)
        age_days = max(0.0, (utcnow() - created).total_seconds() / 86_400)
        return 0.5 ** (age_days / self.freshness_half_life_days)

    def novelty(self, candidate: ContextCandidate, selected: list[ContextCandidate]) -> float:
        if not selected:
            return 1.0
        max_sim = max(
            self.diversity_similarity(candidate.content, other.content) for other in selected
        )
        return 1.0 - max_sim

    def duplication(self, candidate: ContextCandidate, selected: list[ContextCandidate]) -> float:
        if not selected:
            return 0.0
        return max(self.diversity_similarity(candidate.content, other.content) for other in selected)

    def answerability(self, candidate: ContextCandidate, query: str) -> float:
        """Does this item plausibly contain an answer? Question terms covered + facts present."""
        if not query:
            return 0.0
        question_terms = _terms(query)
        if not question_terms:
            return 0.0
        content_terms = _terms(candidate.content)
        coverage = len(question_terms & content_terms) / len(question_terms)
        has_specifics = 1.0 if re.search(r"\d|%|\$|€", candidate.content) else 0.6
        return coverage * has_specifics

    def normalized_token_cost(self, candidate: ContextCandidate) -> float:
        tokens = candidate.token_cost or count_tokens(candidate.content)
        return min(1.0, tokens / self.max_token_cost)

    # -- total ---------------------------------------------------------------------

    def score(
        self,
        candidate: ContextCandidate,
        *,
        query: str,
        selected: list[ContextCandidate] | None = None,
        memory_value: float | None = None,
    ) -> ContextScores:
        selected = selected or []
        w = self.weights
        scores = ContextScores(
            relevance=self.relevance(candidate, query),
            novelty=self.novelty(candidate, selected),
            authority=candidate.authority,
            freshness=self.freshness(candidate),
            provenance=candidate.provenance,
            question_answerability=self.answerability(candidate, query),
            memory_value=memory_value
            if memory_value is not None
            else (candidate.scores.memory_value if candidate.type == "memory" else 0.0),
            token_cost=self.normalized_token_cost(candidate),
            duplication=self.duplication(candidate, selected),
            leakage_risk=candidate.leakage_risk,
        )
        scores.total = (
            w.relevance * scores.relevance
            + w.novelty * scores.novelty
            + w.authority * scores.authority
            + w.freshness * scores.freshness
            + w.provenance * scores.provenance
            + w.question_answerability * scores.question_answerability
            + w.memory_value * scores.memory_value
            - w.token_cost * scores.token_cost
            - w.duplication * scores.duplication
            - w.leakage_risk * scores.leakage_risk
        )
        candidate.scores = scores
        return scores
