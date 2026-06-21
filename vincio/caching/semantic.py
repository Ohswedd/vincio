"""Learned semantic cache (caching/semantic).

Exact-match prompt caching serves a byte-identical request for free; the rung
above it is **near-miss reuse** — answering a request that is *semantically
equivalent* (not byte-identical) to a recent one straight from cache. The risk
is obvious: serve a near-miss that is not actually equivalent and the answer is
wrong. :class:`LearnedSemanticCache` makes that decision safe by never serving
below a **calibrated acceptance threshold** — a threshold *learned from the
platform's own traces* so that an accepted near-miss meets a precision target,
rather than a hand-picked similarity constant.

The cache is the response-side analogue of the reasoning module's
:class:`~vincio.caching.ReasoningTraceCache`: a bounded LRU that lives under the
same resident-memory budget the rest of the platform holds, evicting
lowest-recency-first by both an entry count and an optional byte ceiling, and
tracking its hit rate so the saving is observable.

Three properties make near-miss serving trustworthy:

- **Calibration** — :class:`ThresholdCalibrator` fits the acceptance threshold
  from labelled trace pairs so accepted near-misses clear a precision target;
  if the target is unreachable it falls back to ``1.0`` (near-miss serving
  effectively off) rather than guess.
- **Auditability** — every accepted near-miss is recorded as a
  :class:`SemanticCacheHit` (the query, the matched entry, the similarity, the
  threshold in force), so a served answer can always be traced to its source.
- **Reversibility & gating** — a drifting cache is caught by
  :class:`SemanticCacheGate`, the eval-replay no-regression check that mirrors
  the model-swap gate, and any entry can be :meth:`~LearnedSemanticCache.revoke`\\
  d so a bad decision is undone.

Insertion order is tracked by a monotonic counter (not wall-clock), so a seeded
run is reproducible; freshness (TTL) reads an injectable ``clock`` so a test can
pin time. The cache holds only the cached payload and the query embedding —
nothing it cannot keep resident under the budget.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from ..retrieval.embeddings import Embedder, cosine

__all__ = [
    "SemanticCachePolicy",
    "CalibrationExample",
    "CalibrationReport",
    "ThresholdCalibrator",
    "SemanticCacheEntry",
    "SemanticCacheHit",
    "SemanticCacheStats",
    "LearnedSemanticCache",
    "SemanticGateCase",
    "SemanticGateReport",
    "SemanticCacheGate",
]

# Per-entry structural overhead charged on top of the query text, embedding, and
# serialized payload — the key, scope, schema ref, counters, and bookkeeping.
# Mirrors the reasoning cache's per-entry charge so the budgets compose.
_ENTRY_OVERHEAD_BYTES = 256


class SemanticCachePolicy(BaseModel):
    """Opt-in policy for the learned semantic cache.

    ``threshold`` is the acceptance bar in cosine similarity; calibration may
    raise it (never below ``min_floor``). ``target_precision`` is the precision
    an accepted near-miss must clear on the calibration set. ``min_floor`` is a
    hard lower bound the calibrated threshold can never go beneath — the "never
    serve a near-miss below the bar" guarantee. The remaining fields bound the
    cache's resident footprint and freshness.
    """

    enabled: bool = False
    threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    target_precision: float = Field(default=0.95, ge=0.0, le=1.0)
    min_floor: float = Field(default=0.80, ge=0.0, le=1.0)
    ttl_s: float | None = Field(default=3600.0, ge=0.0)
    max_entries: int = Field(default=2048, ge=1)
    max_resident_bytes: int | None = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# Threshold calibration (the "learned" in learned semantic cache)
# ---------------------------------------------------------------------------


class CalibrationExample(BaseModel):
    """One labelled near-miss observation used to calibrate the threshold.

    ``similarity`` is the cosine similarity between a query and a candidate
    cache entry; ``equivalent`` is the ground-truth label (would serving the
    candidate's answer have been correct?). Mined from the platform's own
    traces — pairs of requests with a known same/different answer.
    """

    similarity: float = Field(ge=-1.0, le=1.0)
    equivalent: bool
    query: str = ""
    candidate_query: str = ""


class CalibrationReport(BaseModel):
    """The outcome of fitting an acceptance threshold from labelled examples.

    ``threshold`` is the fitted bar; ``achieved_precision`` and ``recall`` are
    measured at that bar on the calibration set. ``calibrated`` is ``False``
    when no threshold at or above ``min_floor`` reached ``target_precision`` —
    in which case ``threshold`` is ``1.0`` (near-miss serving is effectively
    disabled) rather than an unsafe guess. ``floored`` marks that ``min_floor``
    was the binding constraint.
    """

    threshold: float
    target_precision: float
    achieved_precision: float
    recall: float
    n_examples: int
    n_positive: int
    calibrated: bool
    floored: bool


class ThresholdCalibrator:
    """Fit a calibrated acceptance threshold from labelled near-miss examples.

    Chooses the **lowest** similarity threshold at or above ``min_floor`` whose
    accepted set clears ``target_precision`` — lowest because a lower bar admits
    more true near-misses (higher recall) while the precision target keeps the
    false ones out. If no such threshold exists the calibrator refuses to guess:
    it returns ``threshold=1.0`` with ``calibrated=False``, so the cache serves
    only near-identical requests until better calibration data arrives. This is
    the same discipline a judge ensemble uses to earn its gating weight.
    """

    def __init__(self, *, target_precision: float = 0.95, min_floor: float = 0.80) -> None:
        if not 0.0 <= target_precision <= 1.0:
            raise ValueError("target_precision must be in [0, 1]")
        if not 0.0 <= min_floor <= 1.0:
            raise ValueError("min_floor must be in [0, 1]")
        self.target_precision = target_precision
        self.min_floor = min_floor

    def calibrate(self, examples: Sequence[CalibrationExample]) -> CalibrationReport:
        """Fit the threshold; see the class docstring for the rule."""
        n = len(examples)
        n_positive = sum(1 for e in examples if e.equivalent)
        candidates = sorted({round(e.similarity, 6) for e in examples} | {self.min_floor, 1.0})

        chosen: tuple[float, float, float] | None = None  # (threshold, precision, recall)
        # Ascending scan: the first qualifying threshold is the lowest, i.e. the
        # one with the highest recall that still clears the precision target.
        for tau in candidates:
            if tau < self.min_floor:
                continue
            accepted = [e for e in examples if e.similarity >= tau]
            if not accepted:
                continue
            true_pos = sum(1 for e in accepted if e.equivalent)
            precision = true_pos / len(accepted)
            recall = (true_pos / n_positive) if n_positive else 0.0
            if precision + 1e-12 >= self.target_precision:
                chosen = (tau, precision, recall)
                break

        if chosen is None:
            return CalibrationReport(
                threshold=1.0,
                target_precision=self.target_precision,
                achieved_precision=self._precision_at(examples, 1.0),
                recall=self._recall_at(examples, 1.0, n_positive),
                n_examples=n,
                n_positive=n_positive,
                calibrated=False,
                floored=False,
            )

        threshold, precision, recall = chosen
        return CalibrationReport(
            threshold=threshold,
            target_precision=self.target_precision,
            achieved_precision=precision,
            recall=recall,
            n_examples=n,
            n_positive=n_positive,
            calibrated=True,
            floored=abs(threshold - self.min_floor) < 1e-9,
        )

    @staticmethod
    def _precision_at(examples: Sequence[CalibrationExample], tau: float) -> float:
        accepted = [e for e in examples if e.similarity >= tau]
        if not accepted:
            return 0.0
        return sum(1 for e in accepted if e.equivalent) / len(accepted)

    @staticmethod
    def _recall_at(examples: Sequence[CalibrationExample], tau: float, n_positive: int) -> float:
        if not n_positive:
            return 0.0
        return sum(1 for e in examples if e.equivalent and e.similarity >= tau) / n_positive


# ---------------------------------------------------------------------------
# Cache entries, hits, and stats
# ---------------------------------------------------------------------------


class SemanticCacheEntry(BaseModel):
    """One cached response keyed by its query embedding.

    ``value`` is the cached payload (e.g. a serialized ``ModelResponse`` or a
    plain answer). ``policy_scope`` and ``schema_ref`` partition the cache: a
    near-miss is only eligible within the same scope (model + stable prompt
    head) and output schema, so two answers are never swapped across a
    boundary that would change what a correct answer looks like.
    """

    key: str
    query: str
    vector: list[float]
    value: Any = None
    policy_scope: str
    schema_ref: str | None = None
    response_tokens: int = 0
    nbytes: int = 0
    seq: int = 0
    stored_at: float = 0.0
    hits: int = 0


class SemanticCacheHit(BaseModel):
    """An auditable record of one near-miss decision.

    Recorded for every *accepted* hit so a served answer can be traced to the
    entry that produced it, the similarity that cleared the bar, and the
    threshold in force at the time — and reversed via
    :meth:`LearnedSemanticCache.revoke`.
    """

    query: str
    matched_key: str
    matched_query: str
    similarity: float
    threshold: float
    accepted: bool
    policy_scope: str
    schema_ref: str | None = None
    seq: int = 0
    value: Any = None


class SemanticCacheStats(BaseModel):
    """Hit-rate, savings, residency, and the calibration in force.

    The residency analogue lives in ``resident_bytes`` (held under the budget);
    the saving surfaces as ``served`` near-miss hits and ``tokens_saved`` — the
    output tokens those hits did not have to generate, which show up in the cost
    report as $0-billed calls.
    """

    entries: int
    hits: int
    misses: int
    near_misses_rejected: int
    served: int
    tokens_saved: int
    hit_rate: float
    threshold: float
    calibrated: bool
    resident_bytes: int
    max_entries: int
    max_resident_bytes: int | None


class LearnedSemanticCache:
    """Bounded, calibrated, auditable near-miss response cache.

    A lookup embeds the query, scans the entries that share its ``policy_scope``
    and ``schema_ref`` (and have not expired), and returns the most-similar
    entry **only if** its similarity clears the calibrated threshold. Below the
    bar is a miss — the near-miss is recorded as rejected but never served.
    Accepted hits are logged for audit and can be revoked. The cache evicts
    lowest-recency-first to fit both the entry-count and resident-byte ceilings,
    the same discipline :class:`~vincio.caching.ReasoningTraceCache` uses.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        policy: SemanticCachePolicy | None = None,
        calibration: CalibrationReport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.embedder = embedder
        self.policy = policy or SemanticCachePolicy()
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, SemanticCacheEntry] = OrderedDict()
        self._audit: list[SemanticCacheHit] = []
        self._seq = 0
        self.resident_bytes = 0
        self.hits = 0
        self.misses = 0
        self.near_misses_rejected = 0
        self.served = 0
        self.tokens_saved = 0
        self._calibration = calibration
        self.threshold = calibration.threshold if calibration is not None else self.policy.threshold

    # -- calibration ------------------------------------------------------

    def calibrate(self, examples: Sequence[CalibrationExample]) -> CalibrationReport:
        """Fit and install the acceptance threshold from labelled examples.

        Updates :attr:`threshold` in place and returns the report; subsequent
        lookups serve only at or above the fitted bar.
        """
        report = ThresholdCalibrator(
            target_precision=self.policy.target_precision, min_floor=self.policy.min_floor
        ).calibrate(examples)
        with self._lock:
            self._calibration = report
            self.threshold = report.threshold
        return report

    async def calibrate_from_pairs(
        self, pairs: Sequence[tuple[str, str, bool]]
    ) -> CalibrationReport:
        """Calibrate from raw labelled query pairs, embedding them first.

        Each pair is ``(query, candidate_query, equivalent)`` — the natural
        shape mined from traces (two requests and whether one's answer would
        have served the other). Embeds the queries, computes their cosine
        similarities, and calibrates on the result.
        """
        texts: list[str] = []
        for left, right, _ in pairs:
            texts.append(left)
            texts.append(right)
        vectors = await self.embedder.embed(texts) if texts else []
        examples: list[CalibrationExample] = []
        for idx, (left, right, equivalent) in enumerate(pairs):
            sim = cosine(vectors[2 * idx], vectors[2 * idx + 1])
            examples.append(
                CalibrationExample(
                    similarity=sim, equivalent=equivalent, query=left, candidate_query=right
                )
            )
        return self.calibrate(examples)

    @property
    def calibration(self) -> CalibrationReport | None:
        """The calibration report in force, or ``None`` if uncalibrated."""
        return self._calibration

    # -- lookup / store ---------------------------------------------------

    async def lookup(
        self, query: str, *, policy_scope: str, schema_ref: str | None = None
    ) -> SemanticCacheHit | None:
        """Return an accepted near-miss for ``query``, or ``None``.

        Embeds the query once, finds the best in-scope, in-schema, unexpired
        entry, and accepts it only if its similarity clears the threshold. A
        below-bar best match is counted as a rejected near-miss and not served.
        """
        query = query.strip()
        if not query:
            with self._lock:
                self.misses += 1
            return None
        [vector] = await self.embedder.embed([query])
        now = self._clock()
        with self._lock:
            best_score = -1.0
            best_entry: SemanticCacheEntry | None = None
            for entry in self._entries.values():
                if entry.policy_scope != policy_scope or entry.schema_ref != schema_ref:
                    continue
                if self._expired(entry, now):
                    continue
                score = cosine(vector, entry.vector)
                if score > best_score:
                    best_score, best_entry = score, entry
            if best_entry is None:
                self.misses += 1
                return None
            accepted = best_score >= self.threshold
            hit = SemanticCacheHit(
                query=query,
                matched_key=best_entry.key,
                matched_query=best_entry.query,
                similarity=best_score,
                threshold=self.threshold,
                accepted=accepted,
                policy_scope=policy_scope,
                schema_ref=schema_ref,
                seq=self._next_seq(),
                value=best_entry.value if accepted else None,
            )
            if not accepted:
                self.near_misses_rejected += 1
                self.misses += 1
                return None
            self.hits += 1
            self.served += 1
            self.tokens_saved += max(0, best_entry.response_tokens)
            best_entry.hits += 1
            self._entries.move_to_end(best_entry.key)
            self._audit.append(hit)
            return hit

    async def store(
        self,
        query: str,
        value: Any,
        *,
        policy_scope: str,
        schema_ref: str | None = None,
        response_tokens: int = 0,
    ) -> SemanticCacheEntry:
        """Insert (or refresh) a cached answer keyed by ``query``'s embedding.

        Re-storing the same query in the same scope replaces the prior entry
        (its bytes are swapped, not double-counted). Evicts lowest-recency-first
        to fit both ceilings after insertion.
        """
        query = query.strip()
        [vector] = await self.embedder.embed([query])
        key = self._entry_key(query, policy_scope, schema_ref)
        entry = SemanticCacheEntry(
            key=key,
            query=query,
            vector=vector,
            value=value,
            policy_scope=policy_scope,
            schema_ref=schema_ref,
            response_tokens=max(0, response_tokens),
            stored_at=self._clock(),
        )
        entry.nbytes = self._estimate_bytes(entry)
        with self._lock:
            entry.seq = self._next_seq()
            existing = self._entries.pop(key, None)
            if existing is not None:
                self.resident_bytes -= existing.nbytes
            self._entries[key] = entry
            self.resident_bytes += entry.nbytes
            self._evict()
        return entry.model_copy()

    # -- audit / reversal -------------------------------------------------

    def audit(self) -> list[SemanticCacheHit]:
        """The append-only log of accepted near-miss decisions (a copy)."""
        with self._lock:
            return [h.model_copy() for h in self._audit]

    def revoke(self, key: str) -> bool:
        """Drop the entry behind a near-miss so it can never serve again.

        The reversal half of the safety contract: an entry implicated in a bad
        hit (its key is on the :class:`SemanticCacheHit`) is removed and its
        bytes reclaimed. Returns ``True`` if an entry was removed.
        """
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                return False
            self.resident_bytes -= entry.nbytes
            return True

    def clear(self) -> int:
        """Drop every entry, reclaim the footprint, and return the count.

        Matches the legacy :class:`~vincio.caching.SemanticCache` contract so
        the :class:`~vincio.caching.InvalidationManager` can clear it on a
        policy / schema / scope change. The audit log and counters persist.
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self.resident_bytes = 0
            return count

    # -- introspection ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> SemanticCacheStats:
        """Hit-rate, savings, residency, and the calibration in force."""
        with self._lock:
            total = self.hits + self.misses
            return SemanticCacheStats(
                entries=len(self._entries),
                hits=self.hits,
                misses=self.misses,
                near_misses_rejected=self.near_misses_rejected,
                served=self.served,
                tokens_saved=self.tokens_saved,
                hit_rate=round(self.hits / total, 4) if total else 0.0,
                threshold=self.threshold,
                calibrated=bool(self._calibration and self._calibration.calibrated),
                resident_bytes=self.resident_bytes,
                max_entries=self.policy.max_entries,
                max_resident_bytes=self.policy.max_resident_bytes,
            )

    # -- internals --------------------------------------------------------

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _expired(self, entry: SemanticCacheEntry, now: float) -> bool:
        ttl = self.policy.ttl_s
        return ttl is not None and (now - entry.stored_at) > ttl

    @staticmethod
    def _entry_key(query: str, policy_scope: str, schema_ref: str | None) -> str:
        from ..core.utils import stable_hash

        return "scache:" + stable_hash({"q": query, "scope": policy_scope, "schema": schema_ref})

    @staticmethod
    def _estimate_bytes(entry: SemanticCacheEntry) -> int:
        import json

        value_bytes = (
            len(json.dumps(entry.value, default=str).encode("utf-8")) if entry.value else 0
        )
        return (
            _ENTRY_OVERHEAD_BYTES
            + len(entry.query.encode("utf-8"))
            + 8 * len(entry.vector)
            + value_bytes
        )

    def _evict(self) -> None:
        while len(self._entries) > self.policy.max_entries:
            _, victim = self._entries.popitem(last=False)
            self.resident_bytes -= victim.nbytes
        ceiling = self.policy.max_resident_bytes
        if ceiling is not None:
            # Keep at least one entry: a single oversized payload is still a hit.
            while self.resident_bytes > ceiling and len(self._entries) > 1:
                _, victim = self._entries.popitem(last=False)
                self.resident_bytes -= victim.nbytes


# ---------------------------------------------------------------------------
# Safety gate (eval-replay no-regression check)
# ---------------------------------------------------------------------------


def _answer_text(value: Any) -> str:
    """Best-effort extraction of an answer string from a cached payload."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        return text if isinstance(text, str) else ""
    text_attr = getattr(value, "text", None)
    return text_attr if isinstance(text_attr, str) else str(value)


def lexical_quality(candidate: str, reference: str) -> float:
    """Token-F1 overlap of ``candidate`` against ``reference`` in ``[0, 1]``.

    The dependency-free default quality signal for the semantic cache gate: a
    served near-miss is "at-least-as-good" when its overlap with the live answer
    clears the floor. Callers needing a stronger signal pass their own scorer
    (e.g. an :mod:`vincio.evals` judge) to :class:`SemanticCacheGate`.
    """
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    if not cand_tokens and not ref_tokens:
        return 1.0
    if not cand_tokens or not ref_tokens:
        return 0.0
    cand_set, ref_set = set(cand_tokens), set(ref_tokens)
    overlap = len(cand_set & ref_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand_set)
    recall = overlap / len(ref_set)
    return 2 * precision * recall / (precision + recall)


class SemanticGateCase(BaseModel):
    """One probe for the cache gate: a query and its live (reference) answer.

    The gate asks the cache for a near-miss on ``query`` and, if one is served,
    checks it is at-least-as-good as ``reference_answer`` — the answer a live
    call produced for this query.
    """

    query: str
    reference_answer: str
    policy_scope: str
    schema_ref: str | None = None


class SemanticGateReport(BaseModel):
    """The verdict of an eval-replay regression check on the cache.

    ``passed`` is ``False`` if any served near-miss scored below the quality
    floor against its reference answer — the same "no regression at a fixed
    budget" rule that gates a model swap, applied to the cache. ``regressions``
    names the offending queries so they can be revoked or the threshold raised.
    """

    passed: bool
    cases: int
    served: int
    regressions: list[str] = Field(default_factory=list)
    mean_quality: float
    min_quality: float
    quality_floor: float
    reason: str


class SemanticCacheGate:
    """Gate a learned semantic cache on replayed cases before it ships.

    The cache analogue of :class:`~vincio.evals.SwapGate`: instead of replaying
    golden traces through a candidate *model* and checking it does not regress,
    it replays probe cases through the *cache* and checks every served near-miss
    is at-least-as-good as the live answer at a fixed (near-zero) budget. A cache
    whose calibration has drifted — serving near-misses that are no longer
    equivalent — fails the gate and is caught before it reaches production.
    """

    def __init__(
        self,
        *,
        quality_floor: float = 0.9,
        scorer: Callable[[str, str], float] | Callable[[str, str], Awaitable[float]] | None = None,
    ) -> None:
        if not 0.0 <= quality_floor <= 1.0:
            raise ValueError("quality_floor must be in [0, 1]")
        self.quality_floor = quality_floor
        self._scorer = scorer or lexical_quality

    async def evaluate(
        self, cache: LearnedSemanticCache, cases: Sequence[SemanticGateCase]
    ) -> SemanticGateReport:
        """Replay ``cases`` through ``cache`` and return the no-regression verdict."""
        served = 0
        regressions: list[str] = []
        qualities: list[float] = []
        for case in cases:
            hit = await cache.lookup(
                case.query, policy_scope=case.policy_scope, schema_ref=case.schema_ref
            )
            if hit is None:
                continue  # a miss costs a live call — never a quality regression
            served += 1
            quality = await self._score(_answer_text(hit.value), case.reference_answer)
            qualities.append(quality)
            if quality + 1e-9 < self.quality_floor:
                regressions.append(case.query)
        mean_quality = sum(qualities) / len(qualities) if qualities else 1.0
        min_quality = min(qualities) if qualities else 1.0
        passed = not regressions
        if passed:
            reason = (
                f"{served}/{len(cases)} served near-misses all cleared the "
                f"{self.quality_floor:.2f} quality floor"
            )
        else:
            reason = (
                f"{len(regressions)} served near-miss(es) regressed below the "
                f"{self.quality_floor:.2f} quality floor"
            )
        return SemanticGateReport(
            passed=passed,
            cases=len(cases),
            served=served,
            regressions=regressions,
            mean_quality=round(mean_quality, 6),
            min_quality=round(min_quality, 6),
            quality_floor=self.quality_floor,
            reason=reason,
        )

    async def _score(self, candidate: str, reference: str) -> float:
        result = self._scorer(candidate, reference)
        if isinstance(result, Awaitable):
            return await result
        return result
