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
import unicodedata
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..core.tokens import count_tokens
from ..core.utils import utcnow
from ..retrieval.embeddings import cosine_with_norms, vector_norm
from .vectorized import HAS_NUMPY, matrix_vector_cosine, row_normalize, weighted_totals

if TYPE_CHECKING:
    from .features import FeatureArena

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

# Below this candidate count, NumPy array construction costs more than it saves,
# so batched scoring stays on the pure-Python reduction even when NumPy is present.
_VECTOR_BATCH_MIN = 32

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
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
    terms: set[str] = set()
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOPWORDS or (len(token) <= 1 and not token.isdigit()):
            continue
        terms.add(_stem(token))
        # Languages without whitespace word boundaries (CJK, Thai, Lao,
        # Khmer, Myanmar) need character shingles for lexical evidence
        # support. This is script-based, not a finite language list.
        scripts = {
            unicodedata.name(char, "").split()[0]
            for char in token
            if char.isalpha() and unicodedata.name(char, "")
        }
        if scripts.intersection({"CJK", "HIRAGANA", "KATAKANA", "THAI", "LAO", "KHMER", "MYANMAR"}):
            terms.update(token[index : index + 2] for index in range(len(token) - 1))
    return frozenset(terms)


def _lexical_from_terms(terms_a: frozenset[str], terms_b: frozenset[str]) -> float:
    """Lexical overlap from precomputed term sets (the kernel behind
    :func:`lexical_similarity`), so a feature arena can supply the terms once."""
    if not terms_a or not terms_b:
        return 0.0
    overlap = len(terms_a & terms_b)
    return overlap / math.sqrt(len(terms_a) * len(terms_b))


def lexical_similarity(a: str, b: str) -> float:
    """Symmetric lexical overlap in [0, 1] with IDF-free dampening."""
    return _lexical_from_terms(_terms(a), _terms(b))


@lru_cache(maxsize=4096)
def _shingles(text: str, size: int = 3) -> frozenset[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < size:
        return frozenset({" ".join(tokens)}) if tokens else frozenset()
    return frozenset(" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1))


def _shingle_jaccard(sa: frozenset[str], sb: frozenset[str]) -> float:
    """Jaccard of precomputed shingle sets (the kernel behind
    :func:`shingle_similarity`)."""
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def shingle_similarity(a: str, b: str, *, size: int = 3) -> float:
    """Jaccard similarity over word shingles — near-duplicate detection."""
    return _shingle_jaccard(_shingles(a, size), _shingles(b, size))


def _containment_from_terms(terms_a: frozenset[str], terms_b: frozenset[str]) -> float:
    """Max term containment from precomputed term sets (the kernel behind
    :func:`containment_similarity`)."""
    if not terms_a or not terms_b:
        return 0.0
    overlap = len(terms_a & terms_b)
    return max(overlap / len(terms_a), overlap / len(terms_b))


def containment_similarity(a: str, b: str) -> float:
    """Max containment of content terms — catches near-duplicates that differ
    by a few filler words (where shingle Jaccard under-reports)."""
    return _containment_from_terms(_terms(a), _terms(b))


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
    """A scored candidate for inclusion in the context packet.

    A candidate may be text, an image, a table, or a video clip. ``content``
    always holds the scorable text surrogate (the text itself, an image
    caption/OCR, a table's Markdown, or a clip's transcript), so relevance,
    novelty, dedup, and ordering work uniformly across modalities, while
    ``image`` / ``table`` / ``video`` carry the non-text payload and ``modality``
    drives a modality-aware token cost.
    """

    id: str
    type: CandidateType
    content: str
    modality: Literal["text", "image", "table", "video"] = "text"
    image: Any = None  # ImageRef for image candidates
    table: dict[str, Any] | None = None  # structured table for table candidates
    video: Any = None  # VideoRef for video candidates
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
        # Per-compile feature arena (optional): when installed, the lexical
        # similarity passes read each candidate's terms/shingles from it instead
        # of re-deriving them through the bounded global cache. The arena is a
        # fresh per-compile object, so a scorer carrying one is never shared.
        self._features: FeatureArena | None = None
        # Per-vector-set L2-norm cache for semantic cosine, so a vector's norm is
        # computed once and reused across every pairwise comparison rather than
        # recomputed on each call. Reset whenever the vector set changes.
        self._norm_cache: dict[str, float] = {}
        # The default lexical estimator can route through the arena; a custom
        # similarity callback is opaque and is always called directly.
        self._lexical_default = self.similarity_fn is lexical_similarity

    def set_embeddings(self, vectors: dict[str, list[float]] | None) -> None:
        """Install (or clear) the embedding vector cache used for semantic scoring."""
        self._vectors = vectors or None
        self._norm_cache = {}

    def set_features(self, arena: FeatureArena | None) -> None:
        """Install (or clear) the per-compile feature arena."""
        self._features = arena

    def _terms_of(self, text: str) -> frozenset[str]:
        arena = self._features
        return arena.terms(text) if arena is not None else _terms(text)

    def _shingles_of(self, text: str, size: int = 3) -> frozenset[str]:
        arena = self._features
        return arena.shingles(text, size) if arena is not None else _shingles(text, size)

    def _norm_of(self, text: str, vector: list[float]) -> float:
        cache = self._norm_cache
        norm = cache.get(text)
        if norm is None:
            norm = vector_norm(vector)
            cache[text] = norm
        return norm

    @property
    def semantic(self) -> bool:
        return self._vectors is not None

    def _semantic_sim(self, a: str, b: str) -> float | None:
        """Cosine over cached embeddings, or ``None`` when either vector is absent.

        Each vector's L2 norm is computed once (memoized by text) and reused, so
        comparing one candidate against many never recomputes its norm; the value
        is bit-for-bit identical to recomputing both norms on every call."""
        if self._vectors is None:
            return None
        va = self._vectors.get(a)
        vb = self._vectors.get(b)
        if va is None or vb is None:
            return None
        sim = cosine_with_norms(va, vb, self._norm_of(a, va), self._norm_of(b, vb))
        return max(0.0, min(1.0, sim))

    def query_similarity(self, content: str, query: str) -> float:
        """Semantic cosine vs the query when embeddings are present; else lexical."""
        sim = self._semantic_sim(content, query)
        if sim is not None:
            return sim
        if self._lexical_default:
            return _lexical_from_terms(self._terms_of(content), self._terms_of(query))
        return self.similarity_fn(content, query)

    def diversity_similarity(self, a: str, b: str) -> float:
        """Similarity used by novelty/duplication/near-dup: semantic when
        embeddings are present, otherwise word-shingle overlap."""
        sim = self._semantic_sim(a, b)
        if sim is not None:
            return sim
        return _shingle_jaccard(self._shingles_of(a), self._shingles_of(b))

    def near_duplicate(self, a: str, b: str) -> float:
        """Near-duplicate score: semantic cosine when embeddings are present
        (catches paraphrases that differ in wording), otherwise the lexical
        shingle/containment maximum."""
        sim = self._semantic_sim(a, b)
        if sim is not None:
            return sim
        return max(
            _shingle_jaccard(self._shingles_of(a), self._shingles_of(b)),
            _containment_from_terms(self._terms_of(a), self._terms_of(b)),
        )

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
        return max(
            self.diversity_similarity(candidate.content, other.content) for other in selected
        )

    def answerability(self, candidate: ContextCandidate, query: str) -> float:
        """Does this item plausibly contain an answer? Question terms covered + facts present."""
        if not query:
            return 0.0
        question_terms = self._terms_of(query)
        if not question_terms:
            return 0.0
        content_terms = self._terms_of(candidate.content)
        coverage = len(question_terms & content_terms) / len(question_terms)
        has_specifics = 1.0 if re.search(r"\d|%|\$|€", candidate.content) else 0.6
        return coverage * has_specifics

    # Representative token cost of a media candidate when none was supplied, so
    # an image/table competes for the budget on its true (non-zero) footprint
    # rather than the length of its short caption.
    _IMAGE_TOKEN_COST = {"low": 85, "high": 765, "auto": 512}
    _VIDEO_TOKEN_COST = {"low": 256, "high": 2048, "auto": 1024}

    def modality_token_cost(self, candidate: ContextCandidate) -> int:
        """Token footprint of a candidate, modality-aware."""
        if candidate.token_cost:
            return candidate.token_cost
        if candidate.modality == "image":
            detail = getattr(candidate.image, "detail", "auto") if candidate.image else "auto"
            return self._IMAGE_TOKEN_COST.get(detail, self._IMAGE_TOKEN_COST["auto"])
        if candidate.modality == "video":
            detail = getattr(candidate.video, "detail", "auto") if candidate.video else "auto"
            return self._VIDEO_TOKEN_COST.get(detail, self._VIDEO_TOKEN_COST["auto"])
        if candidate.modality == "table" and candidate.table:
            # First-class table evidence carries a compact encoding: cost it by
            # the tokens the model receives. A raw table dict falls back to the
            # per-cell heuristic.
            encoding = candidate.table.get("encoding")
            if encoding:
                return count_tokens(str(encoding))
            rows = candidate.table.get("rows") or []
            cols = candidate.table.get("columns") or []
            cells = sum(len(r) for r in rows) if rows else 0
            return max(count_tokens(candidate.content), 3 * (cells + len(cols)))
        return count_tokens(candidate.content)

    def normalized_token_cost(self, candidate: ContextCandidate) -> float:
        return min(1.0, self.modality_token_cost(candidate) / self.max_token_cost)

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

    # -- batched (single-pass) scoring --------------------------------------------

    def _batch_relevance(self, candidates: list[ContextCandidate], query: str) -> list[float]:
        """Relevance of every candidate against *query*.

        In semantic mode with NumPy present the query similarities are one
        matrix–vector product over the cached candidate embeddings; the
        reranker-verdict blend is then applied per item exactly as
        :meth:`relevance` does. Otherwise each candidate falls back to the
        per-item path, so the result is identical either way.
        """
        if not (query and self.semantic and HAS_NUMPY and self._vectors is not None):
            return [self.relevance(c, query) for c in candidates]
        vectors = [self._vectors.get(c.content) for c in candidates]
        query_vec = self._vectors.get(query)
        if query_vec is None or any(v is None for v in vectors):
            return [self.relevance(c, query) for c in candidates]
        normalized = row_normalize(vectors)  # type: ignore[arg-type]
        if normalized is None:  # pragma: no cover - guarded by HAS_NUMPY
            return [self.relevance(c, query) for c in candidates]
        bases = matrix_vector_cosine(normalized, query_vec)
        out: list[float] = []
        for candidate, base in zip(candidates, bases, strict=True):
            base = min(1.0, base)
            upstream = candidate.metadata.get("upstream_relevance")
            if upstream is not None:
                up = max(0.0, min(1.0, float(upstream)))
                out.append(self._UPSTREAM_BLEND * up + (1.0 - self._UPSTREAM_BLEND) * base)
            else:
                out.append(base)
        return out

    def score_batch(self, candidates: list[ContextCandidate], query: str) -> None:
        """Score every candidate against an empty selection in one pass.

        Equivalent to calling :meth:`score` on each candidate with
        ``selected=[]`` (novelty 1, duplication 0): the per-component scores are
        computed once and reduced against the signed weight vector together
        (a single matrix–vector product under NumPy, an equivalent weighted sum
        otherwise), and each :class:`ContextScores` is built without per-item
        validation. The totals — and therefore the selection that follows — are
        identical to the per-candidate loop.
        """
        if not candidates:
            return
        w = self.weights
        construct = ContextScores.model_construct  # skip per-item validation
        if HAS_NUMPY and len(candidates) >= _VECTOR_BATCH_MIN:
            # NumPy fast lane: semantic relevance and the weighted reduction each
            # collapse to a single matrix product over the whole candidate set.
            n = len(candidates)
            relevance = self._batch_relevance(candidates, query)
            authority = [c.authority for c in candidates]
            freshness = [self.freshness(c) for c in candidates]
            provenance = [c.provenance for c in candidates]
            answerability = [self.answerability(c, query) for c in candidates]
            memory_value = [
                c.scores.memory_value if c.type == "memory" else 0.0 for c in candidates
            ]
            token_cost = [self.normalized_token_cost(c) for c in candidates]
            leakage = [c.leakage_risk for c in candidates]
            totals = weighted_totals(
                [
                    relevance,
                    [1.0] * n,
                    authority,
                    freshness,
                    provenance,
                    answerability,
                    memory_value,
                    token_cost,
                    [0.0] * n,
                    leakage,
                ],
                [
                    w.relevance,
                    w.novelty,
                    w.authority,
                    w.freshness,
                    w.provenance,
                    w.question_answerability,
                    w.memory_value,
                    -w.token_cost,
                    -w.duplication,
                    -w.leakage_risk,
                ],
            )
            for i, candidate in enumerate(candidates):
                candidate.scores = construct(
                    relevance=relevance[i],
                    novelty=1.0,
                    authority=authority[i],
                    freshness=freshness[i],
                    provenance=provenance[i],
                    question_answerability=answerability[i],
                    memory_value=memory_value[i],
                    token_cost=token_cost[i],
                    duplication=0.0,
                    leakage_risk=leakage[i],
                    total=totals[i],
                )
            return
        # Pure-Python: a single pass with the total accumulated inline and the
        # scores built without per-item validation — identical results to
        # :meth:`score` with an empty selection.
        wr, wn, wa, wf, wp = w.relevance, w.novelty, w.authority, w.freshness, w.provenance
        wq, wm, wt, wk = w.question_answerability, w.memory_value, w.token_cost, w.leakage_risk
        for candidate in candidates:
            rel = self.relevance(candidate, query)
            auth = candidate.authority
            fresh = self.freshness(candidate)
            prov = candidate.provenance
            ans = self.answerability(candidate, query)
            mem = candidate.scores.memory_value if candidate.type == "memory" else 0.0
            tok = self.normalized_token_cost(candidate)
            leak = candidate.leakage_risk
            candidate.scores = construct(
                relevance=rel,
                novelty=1.0,
                authority=auth,
                freshness=fresh,
                provenance=prov,
                question_answerability=ans,
                memory_value=mem,
                token_cost=tok,
                duplication=0.0,
                leakage_risk=leak,
                total=(
                    wr * rel
                    + wn
                    + wa * auth
                    + wf * fresh
                    + wp * prov
                    + wq * ans
                    + wm * mem
                    - wt * tok
                    - wk * leak
                ),
            )
