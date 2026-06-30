"""Long-horizon context engineering: compaction, intra-run decay, a governor.

Million-token, multi-day, multi-session agent runs degrade when context is
accumulated naïvely: stale spans crowd out fresh signal ("context rot") and the
resident footprint grows without bound, blowing both quality and budget. This
module composes primitives the platform already owns — the context compiler's
footprint estimator, the memory subsystem's exponential decay model, and the
content-addressed evidence store's cross-process ``materialize()`` — into an
explicit per-run **context governor** that keeps a long run inside a declared
*context budget* without losing recall.

Three pieces, all deterministic and offline (no model required):

* :class:`RelevanceDecay` — the memory subsystem's exponential decay applied
  *within a single run*, so a candidate admitted many steps ago loses weight
  before it crowds out fresh signal. Demotions surface in the excluded-context
  report.
* :class:`ContextCompactor` — hierarchical, provenance-preserving compaction
  that folds cold spans of a run into the memory OS as summaries (keyed by the
  content hashes of the evidence they cover, full text written to a
  content-addressed :class:`~vincio.context.evidence_store.EvidenceStore`) and
  *pages the full text back on demand*, so the live packet stays small without
  losing the ability to recover detail. A summary is itself a span, so
  summaries compact again into higher levels as the horizon grows.
* :class:`ContextGovernor` — a per-run controller that holds a
  :class:`ContextBudget` (live tokens, resident bytes, KV-cache footprint) the
  way the cost report holds a dollar budget. As a run grows it decays stale
  spans, compacts the coldest ones once the budget is exceeded, and reports the
  live footprint — bounded as the horizon scales 10×.

This is the *cross-run, long-horizon* governor. The *intra-loop* compactor bound
to one :class:`~vincio.agents.executor.AgentExecutor` step loop lives separately
as :class:`vincio.agents.compaction.LoopCompactor`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.diagnostics import note_suppressed
from ..core.tokens import count_tokens
from ..core.utils import new_id
from .compression import extractive_compress, split_sentences
from .evidence_store import EvidenceStore, InMemoryEvidenceStore, content_hash
from .footprint import ENTRY_OVERHEAD_BYTES
from .scoring import lexical_similarity

__all__ = [
    "RelevanceDecay",
    "ContextBudget",
    "ContextBudgetReport",
    "RunSpan",
    "CompactionRecord",
    "ContextCompactor",
    "ContextGovernor",
]


def _summarize(text: str, *, max_tokens: int) -> str:
    """Extractive summary of *text* within a token budget (deterministic).

    Mirrors :func:`vincio.memory.summarizers.extractive_summary` but stays inside
    :mod:`vincio.context` so the long-horizon governor carries no import-time
    dependency on the memory subsystem (the memory engine is a runtime arg)."""
    if not text.strip():
        return ""
    focus = " ".join(split_sentences(text)[:3])
    return extractive_compress(text, focus, max_tokens).text


class RelevanceDecay(BaseModel):
    """Exponential intra-run relevance decay (the memory recency model, per run).

    A span admitted ``age`` steps ago keeps a fraction ``0.5 ** (age /
    half_life_steps)`` of its base relevance, clamped to ``[floor, 1]``. Fresh
    signal therefore outweighs stale signal of equal base relevance, and a span
    whose decayed weight falls below the governor's threshold is demoted (and
    surfaced in the excluded-context report) before it can crowd the packet.
    """

    half_life_steps: float = Field(default=8.0, gt=0.0)
    floor: float = Field(default=0.0, ge=0.0, le=1.0)

    def weight(self, age_steps: int) -> float:
        """Decay multiplier in ``[floor, 1]`` for a span ``age_steps`` old."""
        if age_steps <= 0:
            return 1.0
        return float(max(self.floor, 0.5 ** (age_steps / self.half_life_steps)))

    def decayed(self, base_relevance: float, age_steps: int) -> float:
        """Base relevance scaled by the age-decay multiplier."""
        return max(0.0, min(1.0, base_relevance)) * self.weight(age_steps)


class ContextBudget(BaseModel):
    """A per-run context budget: the residency analogue of a dollar budget.

    Any cap left ``None`` is unbounded. ``kv_bytes_per_token`` estimates the
    decode KV-cache footprint from the live prompt-token count (KV cache grows
    linearly in context length), so a long run is held against the memory the
    serving engine actually pays, not just the token count.
    """

    max_tokens: int | None = None
    max_resident_bytes: int | None = None
    max_kv_cache_bytes: int | None = None
    kv_bytes_per_token: int = Field(default=2048, ge=0)

    def kv_cache_bytes(self, live_tokens: int) -> int:
        """Estimated decode KV-cache bytes for *live_tokens* of live context."""
        return live_tokens * self.kv_bytes_per_token


class RunSpan(BaseModel):
    """One unit of context admitted to a long run at a given step.

    ``kind`` is ``evidence`` / ``memory`` / ``tool_result`` for raw spans and
    ``summary`` for a compaction product. ``level`` is the compaction level
    (0 = raw, 1 = summary of raw, 2 = summary of summaries, …). ``relevance`` is
    the base utility at admission; ``effective_relevance`` is recomputed by the
    governor each step under intra-run decay. ``covered_hashes`` and
    ``source_ids`` carry provenance so a summary can page its originals back and
    a citation trail survives compaction. ``pinned`` spans (e.g. the objective)
    never decay and are never compacted or evicted.
    """

    id: str = Field(default_factory=lambda: new_id("span"))
    step: int = 0
    kind: str = "evidence"
    text: str = ""
    token_cost: int = 0
    content_hash: str = ""
    source_ids: list[str] = Field(default_factory=list)
    relevance: float = 1.0
    effective_relevance: float = 1.0
    level: int = 0
    covered_hashes: list[str] = Field(default_factory=list)
    pinned: bool = False
    decayed: bool = False

    def model_post_init(self, _context: Any) -> None:  # noqa: D401 - pydantic hook
        if not self.token_cost:
            self.token_cost = count_tokens(self.text)
        if not self.content_hash and self.text:
            self.content_hash = content_hash(self.text)
        self.effective_relevance = self.relevance


class CompactionRecord(BaseModel):
    """Provenance of one compaction: which spans folded into which summary."""

    summary_id: str
    level: int
    covered_span_ids: list[str]
    covered_hashes: list[str]
    source_ids: list[str]
    tokens_before: int
    tokens_after: int
    memory_id: str | None = None

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


class ContextBudgetReport(BaseModel):
    """The live context footprint, the way ``cost_report`` holds dollar spend."""

    live_tokens: int
    resident_bytes: int
    kv_cache_bytes: int
    span_count: int
    summary_count: int
    compaction_count: int
    compacted_tokens_saved: int
    paged_in: int
    decayed_count: int
    step: int
    within_budget: bool
    budget: ContextBudget

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ContextCompactor:
    """Hierarchical, provenance-preserving compaction of cold run spans.

    Folds a batch of cold spans into a single extractive **summary span** whose
    full source text is written to a content-addressed
    :class:`~vincio.context.evidence_store.EvidenceStore` (so it pages back
    losslessly via the same cross-process path slim packets use) and whose gist
    is optionally written into the memory OS as an audited ``SUMMARY`` memory
    carrying the covered content hashes and source ids. Because a summary is
    itself a :class:`RunSpan`, compacting summaries again yields higher levels —
    a hierarchy that keeps the live packet bounded no matter the horizon.
    """

    def __init__(
        self,
        *,
        store: EvidenceStore | None = None,
        memory: Any | None = None,
        owner_id: str = "run",
        scope: Any = None,
        summary_tokens: int = 120,
        summarizer: Any | None = None,
    ) -> None:
        self.store = store if store is not None else InMemoryEvidenceStore()
        self.memory = memory
        self.owner_id = owner_id
        self.scope = scope
        self.summary_tokens = summary_tokens
        # ``summarizer(text, max_tokens=...) -> str``; defaults to the offline,
        # deterministic extractive summarizer.
        self.summarizer = summarizer or _summarize

    def store_span(self, span: RunSpan) -> str:
        """Persist a span's full text under its content hash; return the hash."""
        if span.text:
            self.store.put(span.text)
        return span.content_hash

    def page_in(self, hashes: list[str]) -> dict[str, str]:
        """Materialize cold span text back from the content-addressed store.

        The cross-process recovery path: a summary references its originals by
        hash, and this resolves them on demand without holding the text live.
        """
        recovered: dict[str, str] = {}
        for digest in hashes:
            text = self.store.get(digest)
            if text is not None:
                recovered[digest] = text
        return recovered

    def compact(self, spans: list[RunSpan], *, level: int | None = None) -> tuple[RunSpan, CompactionRecord]:
        """Fold *spans* into one summary span, preserving provenance.

        Writes each span's full text to the store (page-out target), summarizes
        the batch, optionally writes the gist into the memory OS, and returns the
        replacement summary span and a :class:`CompactionRecord`.
        """
        if not spans:
            raise ValueError("compact() requires at least one span")
        # Page the originals out to the content-addressed store first, so they
        # are recoverable before the live copies are dropped.
        covered_hashes: list[str] = []
        source_ids: list[str] = []
        for span in spans:
            covered_hashes.extend(self._covered_hashes(span))
            source_ids.extend(span.source_ids or ([span.content_hash] if span.content_hash else []))
            self.store_span(span)
        # De-duplicate while preserving order (a needle may appear once).
        covered_hashes = list(dict.fromkeys(covered_hashes))
        source_ids = list(dict.fromkeys(source_ids))

        combined = "\n".join(s.text for s in spans if s.text)
        summary_text = self.summarizer(combined, max_tokens=self.summary_tokens) or combined
        out_level = (max(s.level for s in spans) + 1) if level is None else level
        summary = RunSpan(
            kind="summary",
            text=summary_text,
            source_ids=source_ids,
            relevance=max((s.relevance for s in spans), default=0.5),
            level=out_level,
            covered_hashes=covered_hashes,
            step=max((s.step for s in spans), default=0),
        )
        # Provenance-preserving write into the audited memory OS, when present.
        memory_id = self._write_summary_memory(summary, source_ids, covered_hashes)
        record = CompactionRecord(
            summary_id=summary.id,
            level=out_level,
            covered_span_ids=[s.id for s in spans],
            covered_hashes=covered_hashes,
            source_ids=source_ids,
            tokens_before=sum(s.token_cost for s in spans),
            tokens_after=summary.token_cost,
            memory_id=memory_id,
        )
        return summary, record

    @staticmethod
    def _covered_hashes(span: RunSpan) -> list[str]:
        """Hashes a span makes recoverable: its own raw text, or — for a summary
        — the hashes it already covers (so re-compaction keeps the full trail)."""
        if span.kind == "summary" and span.covered_hashes:
            return list(span.covered_hashes)
        return [span.content_hash] if span.content_hash else []

    def _write_summary_memory(
        self, summary: RunSpan, source_ids: list[str], covered_hashes: list[str]
    ) -> str | None:
        if self.memory is None or not summary.text:
            return None
        from ..core.types import MemoryScope, MemoryType

        scope = self.scope if self.scope is not None else MemoryScope.SESSION
        kwargs: dict[str, Any] = {}
        owner_key = {
            MemoryScope.USER: "user_id",
            MemoryScope.AGENT: "agent_id",
            MemoryScope.SESSION: "session_id",
            MemoryScope.TENANT: "tenant_id",
            MemoryScope.ORGANIZATION: "tenant_id",
        }.get(MemoryScope(scope))
        if owner_key:
            kwargs[owner_key] = self.owner_id
        try:
            item = self.memory.remember(
                summary.text,
                scope=scope,
                type=MemoryType.SUMMARY,
                confidence=0.6,
                metadata={
                    "origin": "context_compaction",
                    "level": summary.level,
                    "covered_hashes": covered_hashes,
                    "source_ids": source_ids,
                },
                **kwargs,
            )
        except Exception:
            note_suppressed("context.longhorizon.memory_write")
            return None
        return str(item.id)


class ContextGovernor:
    """Per-run controller that holds a context budget across a long horizon.

    Admit each turn's context with :meth:`admit` (or :meth:`admit_packet`). On
    every admission the governor (1) re-applies intra-run decay so stale spans
    lose weight, then (2) while the live footprint is over budget, compacts the
    coldest non-recent spans into a summary (paging their text to the
    content-addressed store) — or evicts the lowest-utility span when no
    compactor is configured — until the footprint fits. :meth:`recall` answers a
    query over the live spans and *pages cold detail back* from the summaries
    that cover it, so recall survives compaction. :meth:`report` surfaces the
    live footprint the way the cost report surfaces spend.
    """

    def __init__(
        self,
        budget: ContextBudget | None = None,
        *,
        decay: RelevanceDecay | None = None,
        compactor: ContextCompactor | None = None,
        keep_recent_spans: int = 4,
        decay_threshold: float = 0.15,
        compact_batch: int = 4,
    ) -> None:
        self.budget = budget or ContextBudget()
        self.decay = decay or RelevanceDecay()
        self.compactor = compactor
        self.keep_recent_spans = max(0, keep_recent_spans)
        self.decay_threshold = decay_threshold
        self.compact_batch = max(1, compact_batch)
        self.spans: list[RunSpan] = []
        self.step = 0
        self.compactions: list[CompactionRecord] = []
        self.excluded: list[dict[str, Any]] = []
        self.paged_in = 0

    # -- admission ------------------------------------------------------------

    def admit(
        self,
        text: str,
        *,
        kind: str = "evidence",
        relevance: float = 1.0,
        token_cost: int | None = None,
        source_ids: list[str] | None = None,
        pinned: bool = False,
        id: str | None = None,
    ) -> RunSpan:
        """Admit one span at the next step, then govern the live footprint."""
        self.step += 1
        span = RunSpan(
            id=id or new_id("span"),
            step=self.step,
            kind=kind,
            text=text,
            token_cost=token_cost or 0,
            source_ids=list(source_ids or []),
            relevance=relevance,
            pinned=pinned,
        )
        self.spans.append(span)
        self._govern()
        return span

    def admit_packet(self, packet: Any, *, store: EvidenceStore | None = None) -> list[RunSpan]:
        """Admit every evidence entry of a compiled :class:`~vincio.context.ContextPacket`.

        The natural multi-session hook: feed each ``app.run`` result's packet to a
        persistent governor and the long-horizon footprint stays bounded across
        the whole conversation. Slim packets are materialized first so the span
        text is present.
        """
        packet.materialize(store)
        admitted: list[RunSpan] = []
        for entry in packet.evidence_items:
            text = entry.get("text") or ""
            if not text:
                continue
            admitted.append(
                self.admit(
                    text,
                    kind="evidence",
                    relevance=float(entry.get("relevance", 1.0) or 0.0),
                    token_cost=int(entry.get("token_cost", 0) or 0),
                    source_ids=[str(entry.get("source_id") or entry.get("id") or "")],
                )
            )
        return admitted

    def admit_evidence(self, evidence_items: list[Any]) -> list[RunSpan]:
        """Admit a sequence of :class:`~vincio.core.types.EvidenceItem`-like records.

        The :meth:`~vincio.core.app.ContextApp.run` hook: feed each result's
        ``.evidence`` to a persistent governor so the long-horizon footprint stays
        bounded across a multi-session conversation."""
        admitted: list[RunSpan] = []
        for item in evidence_items:
            text = getattr(item, "text", None) or ""
            if not text:
                continue
            admitted.append(
                self.admit(
                    text,
                    kind="evidence",
                    relevance=float(getattr(item, "relevance", 1.0) or 0.0),
                    token_cost=int(getattr(item, "token_cost", 0) or 0),
                    source_ids=[str(getattr(item, "source_id", None) or getattr(item, "id", "") or "")],
                )
            )
        return admitted

    # -- governance loop ------------------------------------------------------

    def _recent_ids(self) -> set[str]:
        if not self.keep_recent_spans:
            return set()
        ordered = sorted(self.spans, key=lambda s: s.step, reverse=True)
        return {s.id for s in ordered[: self.keep_recent_spans]}

    def _apply_decay(self) -> None:
        """Recompute every span's effective relevance under intra-run decay and
        record newly-demoted spans in the excluded-context report."""
        for span in self.spans:
            if span.pinned:
                span.effective_relevance = span.relevance
                continue
            span.effective_relevance = self.decay.decayed(span.relevance, self.step - span.step)
            if span.effective_relevance < self.decay_threshold and not span.decayed:
                span.decayed = True
                self.excluded.append(
                    {
                        "id": span.id,
                        "reason": "intra_run_decay",
                        "step": span.step,
                        "age_steps": self.step - span.step,
                        "effective_relevance": round(span.effective_relevance, 4),
                    }
                )

    def _cold_spans(self) -> list[RunSpan]:
        """Compaction/eviction candidates: not pinned, not in the recent window,
        oldest and least relevant first."""
        recent = self._recent_ids()
        cold = [s for s in self.spans if not s.pinned and s.id not in recent]
        cold.sort(key=lambda s: (s.effective_relevance, s.step))
        return cold

    def _govern(self) -> None:
        self._apply_decay()
        guard = 0
        while self._over_budget() and guard < 10_000:
            guard += 1
            before = (self.live_tokens, self.resident_bytes, len(self.spans))
            cold = self._cold_spans()
            if not cold:
                break
            batch = self._oldest_batch(cold)
            if self.compactor is not None and self._compaction_reduces(batch):
                self._compact_batch(batch)
            else:
                self._evict(cold[0])
            # Stop rather than spin if a pass reduced nothing on any axis.
            after = (self.live_tokens, self.resident_bytes, len(self.spans))
            if after[0] >= before[0] and after[1] >= before[1] and after[2] >= before[2]:
                break

    def _oldest_batch(self, cold: list[RunSpan]) -> list[RunSpan]:
        """The oldest contiguous batch of cold spans to fold together."""
        by_age = sorted(cold, key=lambda s: s.step)
        return by_age[: self.compact_batch]

    @staticmethod
    def _compaction_reduces(batch: list[RunSpan]) -> bool:
        """Compaction is worth attempting only when the batch carries tokens to
        fold; an empty batch (or all-empty spans) falls through to eviction."""
        return bool(batch) and sum(s.token_cost for s in batch) > 0

    def _compact_batch(self, batch: list[RunSpan]) -> None:
        assert self.compactor is not None  # noqa: S101 - the caller compacts only when a compactor is configured
        summary, record = self.compactor.compact(batch)
        # If the summary did not actually shrink the batch, fall back to eviction
        # of the coldest member to guarantee forward progress.
        if record.tokens_after >= record.tokens_before:
            self._evict(min(batch, key=lambda s: s.effective_relevance))
            return
        batch_ids = {s.id for s in batch}
        first_index = min(i for i, s in enumerate(self.spans) if s.id in batch_ids)
        summary.effective_relevance = self.decay.decayed(summary.relevance, self.step - summary.step)
        self.spans = [s for s in self.spans if s.id not in batch_ids]
        self.spans.insert(min(first_index, len(self.spans)), summary)
        self.compactions.append(record)
        for span in batch:
            self.excluded.append(
                {
                    "id": span.id,
                    "reason": "compacted_into_summary",
                    "summary_id": summary.id,
                    "level": summary.level,
                }
            )

    def _evict(self, span: RunSpan) -> None:
        # Page the evicted span out so it can still be recovered, when a store
        # is available, before dropping the live copy.
        if self.compactor is not None:
            self.compactor.store_span(span)
        self.spans = [s for s in self.spans if s is not span]
        self.excluded.append({"id": span.id, "reason": "context_budget_exceeded", "step": span.step})

    # -- footprint ------------------------------------------------------------

    @property
    def live_tokens(self) -> int:
        return sum(s.token_cost for s in self.spans)

    @property
    def resident_bytes(self) -> int:
        return ENTRY_OVERHEAD_BYTES * len(self.spans) + sum(
            len(s.text.encode("utf-8")) for s in self.spans
        )

    @property
    def kv_cache_bytes(self) -> int:
        return self.budget.kv_cache_bytes(self.live_tokens)

    def _over_budget(self) -> bool:
        b = self.budget
        if b.max_tokens is not None and self.live_tokens > b.max_tokens:
            return True
        if b.max_resident_bytes is not None and self.resident_bytes > b.max_resident_bytes:
            return True
        if b.max_kv_cache_bytes is not None and self.kv_cache_bytes > b.max_kv_cache_bytes:
            return True
        return False

    def within_budget(self) -> bool:
        return not self._over_budget()

    # -- recall (page cold detail back) --------------------------------------

    def recall(self, query: str, *, top_k: int = 5, page_in: bool = True) -> list[str]:
        """Recall the spans most relevant to *query*, paging cold detail back.

        Scores live spans directly. For a summary span, also scores the original
        cold spans it covers — materialized on demand from the content-addressed
        store — so a fact compacted out of the live packet is still recoverable
        by query. This is what keeps recall high as the horizon grows.
        """
        scored: list[tuple[float, str, bool]] = []
        for span in self.spans:
            scored.append((lexical_similarity(query, span.text), span.text, False))
            if page_in and span.kind == "summary" and span.covered_hashes and self.compactor is not None:
                recovered = self.compactor.page_in(span.covered_hashes)
                for text in recovered.values():
                    scored.append((lexical_similarity(query, text), text, True))
        scored.sort(key=lambda t: t[0], reverse=True)
        out: list[str] = []
        seen: set[str] = set()
        for score, text, paged in scored[: top_k * 2]:
            if score <= 0.0 or text in seen:
                continue
            seen.add(text)
            if paged:
                self.paged_in += 1
            out.append(text)
            if len(out) >= top_k:
                break
        return out

    def materialize_summary(self, span: RunSpan) -> dict[str, str]:
        """Page back the full originals a summary span covers."""
        if self.compactor is None or not span.covered_hashes:
            return {}
        recovered = self.compactor.page_in(span.covered_hashes)
        self.paged_in += len(recovered)
        return recovered

    # -- reporting ------------------------------------------------------------

    def report(self) -> ContextBudgetReport:
        """The live context-budget report (footprint, compactions, decay)."""
        return ContextBudgetReport(
            live_tokens=self.live_tokens,
            resident_bytes=self.resident_bytes,
            kv_cache_bytes=self.kv_cache_bytes,
            span_count=len(self.spans),
            summary_count=sum(1 for s in self.spans if s.kind == "summary"),
            compaction_count=len(self.compactions),
            compacted_tokens_saved=sum(r.tokens_saved for r in self.compactions),
            paged_in=self.paged_in,
            decayed_count=sum(1 for s in self.spans if s.decayed),
            step=self.step,
            within_budget=self.within_budget(),
            budget=self.budget,
        )

    def excluded_report(self) -> list[dict[str, Any]]:
        """Decayed / compacted / evicted spans, for the excluded-context report."""
        return list(self.excluded)
