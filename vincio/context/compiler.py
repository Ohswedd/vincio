"""Context Compiler — the core innovation.

Pipeline::

    collect_candidates → normalize_candidates → classify_candidates
    → score_candidates → remove_duplicates → resolve_conflicts
    → compress_or_distill → allocate_budget → order_context
    → render_context_packet → validate_packet

Inputs: objective, user message, memory items, evidence, tool results,
business rules, output schema, policies, budget. Outputs: Context IR,
Context Packet, evidence ledger, excluded-context report, budget report.
"""

from __future__ import annotations

import heapq
import re
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import ContextCompileError
from ..core.tokens import count_tokens, count_tokens_many
from ..core.types import (
    Budget,
    Constraint,
    EvidenceItem,
    Example,
    Instruction,
    MemoryItem,
    Objective,
    PolicySet,
    ToolResult,
    ToolSpec,
    UserInput,
)
from ..core.utils import new_id, stable_hash
from .arena import CandidateArena, PreparedCandidates
from .budgeting import BudgetAllocator
from .compression import distill_evidence_ledger, extractive_compress
from .features import FeatureArena
from .footprint import estimate_resident_bytes
from .ir import ContextIR, OutputContractRef
from .llmlingua import salient_units
from .packet import ContextPacket
from .scoring import (
    ContextCandidate,
    ContextScorer,
    ScoringWeights,
    _terms,
    lexical_similarity,
)

__all__ = [
    "ContextCompilerOptions",
    "CompiledContext",
    "CompileStreamEvent",
    "ContextCompiler",
]


class ContextCompilerOptions(BaseModel):
    min_score: float = 0.05  # inclusion threshold on total utility
    min_relevance: float = 0.05  # evidence below this is excluded as irrelevant
    duplicate_threshold: float = 0.85
    conflict_authority_gap: float = 0.25
    conflict_freshness_gap: float = 0.25
    use_evidence_ledger: bool = False
    compress_evidence: bool = True
    max_evidence_items: int = 24
    max_memory_items: int = 8
    ordering: Literal["relevance", "authority", "recency", "boundary_sandwich"] = "relevance"
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    slim_packets: bool = False  # packets reference evidence text by hash
    # Opt-in embedding-driven selection. When enabled *and* a semantic
    # embedder is threaded in, relevance/novelty/dedup/conflict use cosine over
    # cached embeddings and ``_select`` runs MMR with ``mmr_lambda``. Off by
    # default — the default hash embedder is not semantic, so selection stays
    # lexical unless a real embedder is configured and this is turned on.
    semantic_scoring: bool = False
    mmr_lambda: float = 0.7  # MMR relevance/diversity trade-off (1.0 = pure relevance)
    # Reserve response (and one tool-loop round) tokens out of the input budget
    # so the allocator accounts for the full window, not input only. Capped
    # so it never starves context. On by default; small corpora are unaffected
    # because the flexible blocks still fit everything.
    reserve_response_tokens: bool = True
    # Warm candidate arena: when the candidate set (inputs + privacy scope) is
    # unchanged since a recent compile, reuse the collected, normalized, and
    # privacy-screened candidates instead of rebuilding them, so a steady-state
    # recompile is dominated by the query-dependent scoring and selection.
    # Correctness-preserving (the reused state is query-independent), on by
    # default. The reused candidates are fresh per-compile copies, so the
    # shared compiler stays concurrency-safe.
    reuse_candidate_set: bool = True
    # Single-pass feature arena: derive each candidate's stemmed terms, shingles,
    # and similarity-blocking tokens exactly once per compile and thread them
    # through the dedup, conflict, and selection passes, instead of re-deriving
    # them pass after pass through the bounded global cache (which thrashes on the
    # 10k+ pools the streaming pre-filter exercises). Selection-preserving — the
    # features are byte-identical to the per-pass derivation, so the same context
    # is selected — and concurrency-safe, since the arena is a fresh per-compile
    # object. On by default; turn off to fall back to the per-pass derivation.
    single_pass_selection: bool = True
    # Per-app resident-memory ceiling for the compiled packet, in bytes. When
    # set and the selected context would exceed it, the compiler slims the
    # packet and evicts the lowest-utility evidence until the estimate is under
    # the ceiling, recording each eviction in the excluded report. ``None``
    # leaves the footprint unbounded (the default).
    max_resident_bytes: int | None = None
    # Streaming candidate pre-filter. When set and the evidence candidate pool
    # exceeds this cap, a single streaming pass keeps the top ``max_candidates``
    # by a cheap lexical relevance proxy (a bounded heap) and drops exact
    # duplicates by a bounded content fingerprint, *before* the full multi-signal
    # scoring, the O(n²) dedup/conflict passes, and any embedding materialization
    # run — so the expensive stages and the resident vector footprint are bounded
    # by the cap, not by the raw pool size, as a 10k+ candidate corpus grows.
    # Each drop is recorded in the excluded report. ``None`` collects and scores
    # every candidate (the default, behavior unchanged). Memory and tool-result
    # candidates are never pre-filtered (they carry their own small caps).
    max_candidates: int | None = None


class CompiledContext(BaseModel):
    ir: ContextIR
    packet: ContextPacket
    excluded_report: list[dict[str, Any]] = Field(default_factory=list)
    budget_report: dict[str, Any] = Field(default_factory=dict)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    token_count: int = 0
    # Estimated resident-memory footprint of the packet's context, in bytes
    # (deterministic; see :mod:`vincio.context.footprint`). Held under any
    # declared ``max_resident_bytes`` ceiling and surfaced in the cost summary.
    resident_bytes: int = 0
    from_cache: bool = False
    # Original (pre-selection) inputs, retained in memory for partial
    # recompiles; excluded from dumps so packets and caches stay slim.
    source_evidence: list[EvidenceItem] = Field(default_factory=list, exclude=True)
    source_memory: list[MemoryItem] = Field(default_factory=list, exclude=True)
    source_tool_results: list[ToolResult] = Field(default_factory=list, exclude=True)


class CompileStreamEvent(BaseModel):
    """An event from :meth:`ContextCompiler.compile_streaming`.

    - ``prefix`` — the always-included, evidence-independent context (objective,
      instructions, constraints, task), emitted before any candidate is scored.
    - ``evidence`` — the selected evidence entries, once selection finishes.
    - ``done`` — the terminal event carrying the full :class:`CompiledContext`.
    """

    type: Literal["prefix", "evidence", "done"]
    text: str = ""
    blocks: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    result: CompiledContext | None = None


_NEGATION_RE = re.compile(r"\b(not|no|never|cannot|can't|won't|isn't|doesn't|without)\b")
_NUMERIC_RE = re.compile(r"\d")
# Numbers that index a document structure (Section 5, Clause 7, Figure 2) are
# references, not values: two passages that agree but cite different section
# numbers are not in conflict, so these are excluded from value comparison.
_STRUCTURAL_REF_RE = re.compile(
    r"(?i)\b(?:section|clause|chapter|article|appendix|annex|schedule|exhibit|part|rule|"
    r"paragraph|para|page|line|note|figure|fig|table|step|item|version|v|no)\.?\s*#?\s*"
    r"(\d+(?:\.\d+)*)"
)


def _media_identity(e: EvidenceItem) -> str | None:
    """Stable hash of an evidence item's non-text payload (image/table/video),
    or ``None`` for plain text — distinguishes media candidates in the cache
    signature so two clips that share a caption don't collide."""
    if e.modality == "text":
        return None
    payload: Any = None
    if e.image is not None:
        payload = e.image.model_dump(mode="json")
    elif e.video is not None:
        payload = e.video.model_dump(mode="json")
    elif e.table is not None:
        payload = e.table
    return stable_hash(payload) if payload is not None else None


def _coerce_evidence(evidence: list[Any] | None) -> list[EvidenceItem]:
    """Normalize an evidence list so a :class:`~vincio.data.Dataset` or
    :class:`~vincio.data.TableEvidence` (anything exposing ``to_evidence_item``)
    is projected to a ``modality="table"`` :class:`EvidenceItem`; plain evidence
    items pass through unchanged. Lets a dataset be dropped straight into the
    compiler's evidence list (or ``app.pending_evidence``) as table evidence."""
    if not evidence:
        return []
    return [
        item.to_evidence_item() if hasattr(item, "to_evidence_item") else item
        for item in evidence
    ]


def _looks_negated(text: str) -> bool:
    return bool(_NEGATION_RE.search(text.lower()))


def _value_units(text: str) -> set[str]:
    """The numeric/date salient units of *text* (numbers, percentages,
    currency, dates) — the values two same-topic passages can disagree on.
    Bare structural references (section/clause/figure numbers) are excluded."""
    structural = {m.group(1) for m in _STRUCTURAL_REF_RE.finditer(text)}
    return {u for u in salient_units(text) if _NUMERIC_RE.search(u) and u not in structural}


def _value_disagreement(a: str, b: str) -> dict[str, Any] | None:
    """Detect a value-level contradiction between two same-topic passages.

    Replaces the old negation-XOR trigger (which missed every value
    disagreement): two near-paraphrases that cite *different* numbers/dates
    (e.g. "refund within 30 days" vs "within 14 days"), or that differ in
    polarity (one negates, the other does not), are in conflict. Returns a
    structured delta, or ``None`` when there is no disagreement.
    """
    units_a, units_b = _value_units(a), _value_units(b)
    if units_a and units_b and units_a != units_b:
        return {
            "kind": "value_disagreement",
            "a_values": sorted(units_a),
            "b_values": sorted(units_b),
            "differing": sorted(units_a.symmetric_difference(units_b)),
        }
    if _looks_negated(a) != _looks_negated(b):
        return {"kind": "polarity_disagreement", "a_values": [], "b_values": [], "differing": []}
    return None


class ContextCompiler:
    def __init__(
        self,
        options: ContextCompilerOptions | None = None,
        *,
        cache: Any | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.options = options or ContextCompilerOptions()
        self.scorer = ContextScorer(self.options.weights)
        # Semantic embedder threaded in by the app when semantic scoring is on
        # (opt-in). Used only to build per-compile embedding vectors; the shared
        # ``self.scorer`` stays vector-less so concurrent compiles never race.
        self.embedder = embedder
        self.allocator = BudgetAllocator()
        self.cache = cache  # ContextCompileCache | None
        self.cache_hits = 0
        # Warm candidate arena: reuse the prepared candidate set across compiles
        # whose inputs are unchanged (see ``reuse_candidate_set``).
        self.arena = CandidateArena() if self.options.reuse_candidate_set else None
        self.arena_hits = 0
        # Count of candidates dropped by the streaming pre-filter (see
        # ``max_candidates``) — observability for the bounded big-data path.
        self.prefilter_drops = 0
        # The inline evidence compressor. Defaults to extractive
        # compression; a learned compressor (e.g. ``LLMLinguaCompressor``) is a
        # drop-in with the same signature, installed via
        # ``app.use_learned_compression(...)`` once faithfulness-gated.
        self.compressor: Any = extractive_compress

    def _fresh_scorer(self, features: FeatureArena | None) -> ContextScorer:
        """A per-compile scorer mirroring the shared one's configuration, carrying
        this compile's feature arena. Used whenever per-compile state (a feature
        arena or embedding vectors) must not touch the shared, concurrently-used
        ``self.scorer``."""
        scorer = ContextScorer(
            self.options.weights,
            similarity_fn=self.scorer.similarity_fn,
            max_token_cost=self.scorer.max_token_cost,
            freshness_half_life_days=self.scorer.freshness_half_life_days,
        )
        scorer.set_features(features)
        return scorer

    async def _scorer_for(
        self,
        candidates: list[ContextCandidate],
        query: str,
        features: FeatureArena | None = None,
    ) -> ContextScorer:
        """The scorer for this compile. Lexical mode returns the shared,
        vector-less scorer (or a fresh one carrying the feature arena, so the
        shared instance stays state-free under concurrent compiles). Semantic mode
        batch-embeds candidate contents and the query (content-addressed, so
        repeats are cheap) and installs the vectors on a fresh scorer — keeping
        the shared instance race-free."""
        if not (self.options.semantic_scoring and self.embedder is not None):
            return self.scorer if features is None else self._fresh_scorer(features)
        # Semantic mode always works on a fresh scorer (it carries per-compile
        # vectors); the feature arena rides on the same fresh instance.
        fallback = self.scorer if features is None else self._fresh_scorer(features)
        seen: set[str] = set()
        texts: list[str] = []
        for candidate in candidates:
            if candidate.content and candidate.content not in seen:
                seen.add(candidate.content)
                texts.append(candidate.content)
        if query and query not in seen:
            seen.add(query)
            texts.append(query)
        if not texts:
            return fallback
        try:
            vectors_list = await self.embedder.embed(texts)
        except Exception:  # noqa: BLE001 - fall back to lexical if embedding fails
            return fallback
        vectors = {text: vec for text, vec in zip(texts, vectors_list, strict=False)}
        scorer = self._fresh_scorer(features)
        scorer.set_embeddings(vectors)
        return scorer

    def _signature(
        self,
        *,
        objective: Objective,
        user_input: UserInput,
        instructions: list[Instruction],
        constraints: list[Constraint],
        examples: list[Example],
        evidence: list[EvidenceItem],
        memory: list[MemoryItem],
        tool_results: list[ToolResult],
        tool_specs: list[ToolSpec],
        output_contract: OutputContractRef | None,
        budget: Budget,
        policies: PolicySet,
    ) -> dict[str, Any]:
        """Content signature covering every input that affects compilation."""
        return {
            "objective": (objective.text, objective.task_type.value),
            "input": (user_input.text, user_input.tenant_id),
            "instructions": [i.text for i in instructions],
            "constraints": [c.text for c in constraints],
            "examples": [(e.input, e.output) for e in examples],
            "evidence": [
                (e.id, e.text, e.authority, e.provenance, e.relevance, e.token_cost, e.source_id)
                for e in evidence
            ],
            "memory": [
                (
                    m.id,
                    m.content,
                    m.confidence,
                    m.scope.value,
                    m.privacy_class.value,
                    m.owner_id,
                    m.updated_at.isoformat() if m.updated_at else None,
                )
                for m in memory
            ],
            "tool_results": [
                (t.id, t.tool_name, t.status, str(t.output), t.error) for t in tool_results
            ],
            "tool_specs": [t.model_dump(mode="json") for t in tool_specs],
            "contract": output_contract.model_dump(mode="json") if output_contract else None,
            "budget": budget.model_dump(mode="json"),
            "policies": policies.model_dump(mode="json"),
            "options": self.options.model_dump(mode="json"),
        }

    def _candidate_signature(
        self,
        *,
        evidence: list[EvidenceItem],
        memory: list[MemoryItem],
        tool_results: list[ToolResult],
        privacy: str,
        tenant_id: str | None,
    ) -> dict[str, Any]:
        """Query-independent signature of the candidate set.

        Candidate collection, normalization, and the privacy screen depend only
        on the inputs and the run's privacy scope — never on the query or
        budget — so two compiles with this same signature share their prepared
        candidate set. Mirrors the evidence/memory/tool fields the full compile
        signature uses, with the modality and media identity that distinguish
        image/table evidence and the privacy scope the screen reads.
        """
        return {
            "evidence": [
                (
                    e.id,
                    e.scorable_text,
                    e.modality,
                    e.authority,
                    e.provenance,
                    e.relevance,
                    e.token_cost,
                    e.source_id,
                    e.page,  # affects citation_ref, which _collect caches in metadata
                    e.time_range,  # temporal locator → citation_ref for clip evidence
                    _media_identity(e),
                )
                for e in evidence
            ],
            "memory": [
                (
                    m.id,
                    m.content,
                    m.confidence,
                    m.scope.value,
                    m.privacy_class.value,
                    m.owner_id,
                    bool(m.source_trace_id),
                    m.updated_at.isoformat() if m.updated_at else None,
                )
                for m in memory
            ],
            "tool_results": [
                (t.id, t.tool_name, t.status, str(t.output), t.error) for t in tool_results
            ],
            "privacy": privacy,
            "tenant_id": tenant_id,
        }

    # -- candidate collection -----------------------------------------------------

    def _collect(
        self,
        *,
        evidence: list[EvidenceItem],
        memory: list[MemoryItem],
        tool_results: list[ToolResult],
    ) -> list[ContextCandidate]:
        candidates: list[ContextCandidate] = []
        for raw in evidence:
            # A Dataset / TableEvidence is first-class: it projects to a
            # ``modality="table"`` EvidenceItem carrying its compact encoding.
            item = raw.to_evidence_item() if hasattr(raw, "to_evidence_item") else raw
            # text, image, table, and video evidence are all candidates. The
            # scorable surrogate (text / caption / table encoding / transcript)
            # drives relevance/dedup/ordering; image/table/video carry the
            # non-text payload.
            text = item.scorable_text.strip()
            if not text and item.modality == "text":
                continue
            if not text and item.image is None and item.table is None and item.video is None:
                continue
            token_cost = item.token_cost or item.estimated_token_cost() or count_tokens(text)
            candidates.append(
                ContextCandidate(
                    id=item.id,
                    type="evidence",
                    content=text,
                    modality=item.modality,
                    image=item.image,
                    table=item.table,
                    video=item.video,
                    token_cost=token_cost,
                    source=item,
                    authority=item.authority,
                    provenance=item.provenance,
                    created_at=None,
                    metadata={
                        "citation_ref": item.citation_ref,
                        "source_id": item.source_id,
                        "modality": item.modality,
                        # Relevance assigned upstream (retrieval/reranker); the
                        # lexical gate must not discard semantically-matched
                        # evidence that shares no surface terms with the query.
                        "upstream_relevance": item.relevance,
                    },
                )
            )
        for mem in memory:
            content = mem.content.strip()
            if not content:
                continue
            leakage = 0.0
            if mem.privacy_class.value in ("pii", "sensitive"):
                leakage = 0.6
            candidate = ContextCandidate(
                id=mem.id,
                type="memory",
                content=content,
                token_cost=count_tokens(content),
                source=mem,
                authority=mem.confidence,
                provenance=0.8 if mem.source_trace_id else 0.5,
                created_at=mem.updated_at,
                leakage_risk=leakage,
                metadata={"scope": mem.scope.value, "type": mem.type.value},
            )
            candidate.scores.memory_value = mem.confidence
            candidates.append(candidate)
        for result in tool_results:
            text = str(result.output if result.output is not None else result.error or "").strip()
            if not text:
                continue
            candidates.append(
                ContextCandidate(
                    id=result.id,
                    type="tool_result",
                    content=text,
                    token_cost=count_tokens(text),
                    source=result,
                    authority=0.8 if result.status == "ok" else 0.2,
                    provenance=0.9,
                    metadata={"tool": result.tool_name, "status": result.status},
                )
            )
        return candidates

    # -- streaming candidate pre-filter -------------------------------------------

    def _prefilter_candidates(
        self,
        candidates: list[ContextCandidate],
        query: str,
        excluded: list[dict[str, Any]],
    ) -> list[ContextCandidate]:
        """Bound the evidence candidate pool before full scoring.

        When ``max_candidates`` is set and the evidence pool exceeds it, a single
        streaming pass keeps only the most promising candidates so the expensive
        stages downstream — the full multi-signal scoring, the O(n²) dedup and
        conflict passes, and any per-candidate embedding materialization — never
        see more than the cap, however large the raw corpus grows. The pass is
        bounded in memory two ways: a min-heap of at most ``max_candidates``
        survivors ranked by a cheap lexical relevance proxy against the query
        (top-K, never the whole sorted pool), and a capped content-fingerprint
        set that drops exact duplicates (a reservoir-style overflow stops the set
        growing on a huge stream). Required candidates and non-evidence
        candidates (memory, tool results) always pass through, and every drop is
        recorded in the excluded report so the pruning is auditable. Returns the
        survivors in their original order (full scoring re-ranks them).
        """
        cap = self.options.max_candidates
        if cap is None or cap <= 0:
            return candidates
        evidence = [c for c in candidates if c.type == "evidence" and not c.required]
        if len(evidence) <= cap:
            return candidates

        # Bounded exact-duplicate fingerprints: small ints, capped so the set
        # never grows without bound on a huge stream (overflow stops adding —
        # dedup becomes best-effort within the bound rather than unbounded).
        seen: set[int] = set()
        dedup_cap = max(cap * 8, 4096)
        dedup_overflow = False
        # Min-heap of (proxy_relevance, order_index, candidate). The order index
        # makes the comparison total and breaks ties deterministically toward the
        # earlier candidate, never reaching the candidate object itself.
        heap: list[tuple[float, int, ContextCandidate]] = []
        drops = 0
        for index, candidate in enumerate(evidence):
            fingerprint = hash(candidate.content)
            if fingerprint in seen:
                excluded.append({"id": candidate.id, "reason": "prefiltered_duplicate"})
                drops += 1
                continue
            if not dedup_overflow:
                seen.add(fingerprint)
                if len(seen) > dedup_cap:
                    dedup_overflow = True
            proxy = lexical_similarity(candidate.content, query) if query else 0.0
            if len(heap) < cap:
                heapq.heappush(heap, (proxy, index, candidate))
            elif (proxy, -index) > (heap[0][0], -heap[0][1]):
                evicted = heapq.heapreplace(heap, (proxy, index, candidate))[2]
                excluded.append({"id": evicted.id, "reason": "prefiltered_low_relevance"})
                drops += 1
            else:
                excluded.append({"id": candidate.id, "reason": "prefiltered_low_relevance"})
                drops += 1
        self.prefilter_drops += drops
        kept_ids = {candidate.id for _, _, candidate in heap}
        return [
            c
            for c in candidates
            if c.type != "evidence" or c.required or c.id in kept_ids
        ]

    # -- normalization / dedup / conflicts ----------------------------------------

    @staticmethod
    def _normalize(candidates: list[ContextCandidate]) -> list[ContextCandidate]:
        for candidate in candidates:
            candidate.content = " ".join(candidate.content.split())
        # Count the tokens of every candidate still missing a cost in one batch,
        # rather than one memoized call per item.
        pending = [c for c in candidates if not c.token_cost and c.content]
        if pending:
            counts = count_tokens_many([c.content for c in pending])
            for candidate, count in zip(pending, counts, strict=True):
                candidate.token_cost = count
        return [c for c in candidates if c.content]

    @staticmethod
    def _block_tokens(text: str, arena: FeatureArena | None = None) -> frozenset[str]:
        """Tokens a candidate is indexed under for similarity blocking. Two
        passages that share none of these have shingle *and* containment
        similarity 0, so they can be skipped without computing either — making
        the blocking pass exact, not an approximation. Reads the feature arena
        when one is threaded in, so the tokens are derived once per compile."""
        if arena is not None:
            return arena.block_tokens(text)
        return frozenset(re.findall(r"[a-z0-9]+", text.lower())) | _terms(text)

    def _remove_duplicates(
        self,
        candidates: list[ContextCandidate],
        excluded: list[dict[str, Any]],
        scorer: ContextScorer,
    ) -> list[ContextCandidate]:
        kept: list[ContextCandidate] = []
        semantic = scorer.semantic
        arena = scorer._features
        # Lexical mode: inverted token index so each candidate is compared only
        # against kept items that could possibly be near-duplicates (near-linear
        # for diverse pools, exact at the 0.85 threshold). Semantic mode compares
        # all kept items (cosine catches paraphrases with no shared tokens).
        index: dict[str, list[int]] = defaultdict(list)
        for candidate in sorted(candidates, key=lambda c: c.scores.total, reverse=True):
            if semantic:
                neighbors: list[ContextCandidate] = kept
                tokens: frozenset[str] = frozenset()
            else:
                positions: set[int] = set()
                tokens = self._block_tokens(candidate.content, arena)
                for token in tokens:
                    positions.update(index.get(token, ()))
                neighbors = [kept[p] for p in positions]
            duplicate_of = None
            for existing in neighbors:
                if scorer.near_duplicate(candidate.content, existing.content) >= self.options.duplicate_threshold:
                    duplicate_of = existing.id
                    break
            if duplicate_of is not None:
                excluded.append(
                    {"id": candidate.id, "reason": "duplicate", "duplicate_of": duplicate_of}
                )
            else:
                if not semantic:
                    pos = len(kept)
                    for token in tokens:
                        index[token].append(pos)
                kept.append(candidate)
        return kept

    def _conflict_pairs(
        self, candidates: list[ContextCandidate], scorer: ContextScorer
    ) -> list[tuple[int, int]]:
        """Index pairs of same-type evidence/memory candidates worth comparing
        for conflict. Lexical mode blocks on shared tokens (exact); semantic
        mode returns all same-type pairs."""
        same_type = [
            i for i, c in enumerate(candidates) if c.type in ("evidence", "memory")
        ]
        if scorer.semantic:
            return [
                (a, b)
                for ia, a in enumerate(same_type)
                for b in same_type[ia + 1 :]
                if candidates[a].type == candidates[b].type
            ]
        arena = scorer._features
        index: dict[str, list[int]] = defaultdict(list)
        for i in same_type:
            for token in self._block_tokens(candidates[i].content, arena):
                index[token].append(i)
        pairs: set[tuple[int, int]] = set()
        for members in index.values():
            for ia in range(len(members)):
                for ib in range(ia + 1, len(members)):
                    i, j = members[ia], members[ib]
                    if candidates[i].type == candidates[j].type:
                        pairs.add((i, j) if i < j else (j, i))
        return sorted(pairs)

    def _resolve_conflicts(
        self,
        candidates: list[ContextCandidate],
        excluded: list[dict[str, Any]],
        scorer: ContextScorer,
    ) -> tuple[list[ContextCandidate], list[dict[str, Any]]]:
        """Higher authority wins; newer wins on similar authority; otherwise
        keep both and report a structured conflict delta to the model.

        The trigger is a salient-unit value disagreement (or polarity flip), not
        the old negation-XOR heuristic that missed every numeric/date conflict.
        """
        conflicts: list[dict[str, Any]] = []
        dropped: set[str] = set()
        for i, j in self._conflict_pairs(candidates, scorer):
            a, b = candidates[i], candidates[j]
            if a.id in dropped or b.id in dropped:
                continue
            similarity = scorer.diversity_similarity(a.content, b.content)
            if not (0.30 <= similarity < self.options.duplicate_threshold):
                continue
            delta = _value_disagreement(a.content, b.content)
            if delta is None:
                continue
            authority_gap = a.authority - b.authority
            if abs(authority_gap) > self.options.conflict_authority_gap:
                loser = b if authority_gap > 0 else a
                winner = a if authority_gap > 0 else b
                dropped.add(loser.id)
                excluded.append(
                    {"id": loser.id, "reason": "conflict_lower_authority", "superseded_by": winner.id}
                )
                continue
            freshness_a = scorer.freshness(a)
            freshness_b = scorer.freshness(b)
            if abs(freshness_a - freshness_b) > self.options.conflict_freshness_gap:
                loser = b if freshness_a > freshness_b else a
                winner = a if freshness_a > freshness_b else b
                dropped.add(loser.id)
                excluded.append(
                    {"id": loser.id, "reason": "conflict_stale", "superseded_by": winner.id}
                )
                continue
            conflicts.append(
                {
                    "a": a.id,
                    "b": b.id,
                    "note": "unresolved conflict; both included — report the discrepancy",
                    **delta,
                }
            )
        return [c for c in candidates if c.id not in dropped], conflicts

    # -- selection under budget ------------------------------------------------------

    @staticmethod
    def _update_max_sim(
        remaining: list[ContextCandidate],
        chosen: ContextCandidate,
        max_sim: dict[str, float],
        scorer: ContextScorer,
    ) -> None:
        """Fold the just-selected item's similarity into each remaining
        candidate's running max — the only diversity update needed per pick."""
        for candidate in remaining:
            sim = scorer.diversity_similarity(candidate.content, chosen.content)
            if sim > max_sim[candidate.id]:
                max_sim[candidate.id] = sim

    def _select(
        self,
        candidates: list[ContextCandidate],
        *,
        budget_tokens: int,
        max_items: int,
        query: str,
        excluded: list[dict[str, Any]],
        scorer: ContextScorer,
    ) -> list[ContextCandidate]:
        """Maximal-marginal-relevance selection: utility minus a diversity
        penalty against the already-selected set.

        Scored once against an empty selection, then each pick only folds the
        newly-selected item into a running per-candidate max-similarity — O(n·k)
        instead of re-scoring the whole pool every pick (O(n²)). In semantic mode
        relevance and diversity are embedding cosine with an ``mmr_lambda``
        trade-off; otherwise the lexical weighted utility (identical to before).
        """
        pool: list[ContextCandidate] = []
        for candidate in candidates:
            # Relevance gate: evidence that shares nothing with the task is
            # context pollution regardless of authority/novelty baselines.
            if candidate.type == "evidence":
                relevance = scorer.relevance(candidate, query)
                answerability = scorer.answerability(candidate, query)
                upstream = float(candidate.metadata.get("upstream_relevance") or 0.0)
                if max(relevance, answerability, upstream) < self.options.min_relevance:
                    excluded.append(
                        {"id": candidate.id, "reason": "low_relevance", "score": round(relevance, 4)}
                    )
                    continue
            pool.append(candidate)

        # Score once with an empty selection: novelty=1, duplication=0.
        scorer.score_batch(pool, query)
        w = scorer.weights
        semantic = scorer.semantic
        lam = self.options.mmr_lambda
        diversity_weight = w.novelty + w.duplication
        base = {c.id: c.scores.total for c in pool}
        static = {c.id: base[c.id] - w.novelty for c in pool}  # utility minus diversity terms
        max_sim = {c.id: 0.0 for c in pool}

        def effective(candidate: ContextCandidate) -> float:
            sim = max_sim[candidate.id]
            if semantic:
                return lam * static[candidate.id] - (1.0 - lam) * sim
            # Equivalent to the original per-iteration total:
            # static + w.novelty*(1-sim) - w.duplication*sim.
            return base[candidate.id] - diversity_weight * sim

        selected: list[ContextCandidate] = []
        used = 0
        remaining = list(pool)
        while remaining and len(selected) < max_items:
            best = max(remaining, key=effective)
            best_total = effective(best)
            best.scores.duplication = max_sim[best.id]
            best.scores.novelty = 1.0 - max_sim[best.id]
            best.scores.total = best_total
            if best_total < self.options.min_score:
                excluded.append(
                    {"id": best.id, "reason": "low_relevance", "score": round(best_total, 4)}
                )
                for rem in remaining:
                    if rem is best:
                        continue
                    rem.scores.total = effective(rem)
                    excluded.append(
                        {"id": rem.id, "reason": "low_relevance", "score": round(rem.scores.total, 4)}
                    )
                break
            remaining.remove(best)
            if used + best.token_cost > budget_tokens:
                # Only compress text: a media item's scorable surrogate is a short
                # caption/transcript whose token cost reflects the media payload,
                # not the surrogate — compressing it would undercount the budget.
                if (
                    self.options.compress_evidence
                    and best.type == "evidence"
                    and best.modality == "text"
                ):
                    remaining_budget = budget_tokens - used
                    if remaining_budget >= 32:
                        compressed = self.compressor(best.content, query, remaining_budget)
                        if compressed.compressed_tokens <= remaining_budget:
                            best.content = compressed.text
                            best.token_cost = compressed.compressed_tokens
                            best.metadata["compressed"] = compressed.method
                            selected.append(best)
                            used += best.token_cost
                            self._update_max_sim(remaining, best, max_sim, scorer)
                            continue
                excluded.append(
                    {"id": best.id, "reason": "budget_exceeded", "token_cost": best.token_cost}
                )
                continue
            selected.append(best)
            used += best.token_cost
            self._update_max_sim(remaining, best, max_sim, scorer)
        return selected

    def _order(self, selected: list[ContextCandidate], query: str) -> list[ContextCandidate]:
        """Ranked evidence; boundary sandwich puts the strongest
        items at the start and end (mitigates lost-in-the-middle)."""
        mode = self.options.ordering
        if mode == "authority":
            return sorted(selected, key=lambda c: c.authority, reverse=True)
        if mode == "recency":
            return sorted(
                selected,
                key=lambda c: c.created_at.timestamp() if c.created_at else 0.0,
                reverse=True,
            )
        ranked = sorted(selected, key=lambda c: c.scores.total, reverse=True)
        if mode == "boundary_sandwich" and len(ranked) > 2:
            front: list[ContextCandidate] = []
            back: list[ContextCandidate] = []
            for index, candidate in enumerate(ranked):
                (front if index % 2 == 0 else back).append(candidate)
            return front + back[::-1]
        return ranked

    def _enforce_footprint(
        self,
        evidence: list[ContextCandidate],
        memory: list[ContextCandidate],
        slim: bool,
        excluded: list[dict[str, Any]],
    ) -> tuple[bool, list[ContextCandidate]]:
        """Hold the packet's estimated footprint under ``max_resident_bytes``.

        First slims the packet (lossless — text moves to a hash reference);
        then, if still over, evicts the lowest-utility evidence one at a time,
        recording each eviction, until the estimate fits or no evidence remains.
        Returns the effective slim flag and the surviving (order-preserved)
        evidence.
        """
        ceiling = self.options.max_resident_bytes
        assert ceiling is not None
        memory_texts = [c.content for c in memory]

        def estimate(items: list[ContextCandidate], slim_flag: bool) -> int:
            return estimate_resident_bytes(
                [c.content for c in items], memory_texts, slim=slim_flag
            )

        if estimate(evidence, slim) <= ceiling:
            return slim, evidence
        slim = True  # slim first — it never drops evidence
        if estimate(evidence, slim) <= ceiling:
            return slim, evidence

        # Evict the lowest-utility evidence until the estimate fits.
        kept = list(evidence)
        order = sorted(range(len(kept)), key=lambda i: kept[i].scores.total)  # ascending utility
        evict: set[int] = set()
        for index in order:
            if estimate([kept[i] for i in range(len(kept)) if i not in evict], slim) <= ceiling:
                break
            evict.add(index)
            victim = kept[index]
            excluded.append(
                {
                    "id": victim.id,
                    "reason": "memory_budget_exceeded",
                    "token_cost": victim.token_cost,
                }
            )
        survivors = [kept[i] for i in range(len(kept)) if i not in evict]
        return slim, survivors

    # -- main entry --------------------------------------------------------------

    async def compile(
        self,
        *,
        objective: Objective,
        user_input: UserInput,
        instructions: list[Instruction] | None = None,
        constraints: list[Constraint] | None = None,
        examples: list[Example] | None = None,
        evidence: list[EvidenceItem] | None = None,
        memory: list[MemoryItem] | None = None,
        tool_results: list[ToolResult] | None = None,
        tool_specs: list[ToolSpec] | None = None,
        output_contract: OutputContractRef | None = None,
        budget: Budget | None = None,
        policies: PolicySet | None = None,
        trace_parent_id: str | None = None,
    ) -> CompiledContext:
        budget = budget or Budget()
        policies = policies or PolicySet()
        evidence = _coerce_evidence(evidence)
        query = (user_input.text or "") or objective.text
        excluded: list[dict[str, Any]] = []

        cache_key: str | None = None
        if self.cache is not None:
            cache_key = self.cache.key(
                self._signature(
                    objective=objective,
                    user_input=user_input,
                    instructions=list(instructions or []),
                    constraints=list(constraints or []),
                    examples=list(examples or []),
                    evidence=list(evidence or []),
                    memory=list(memory or []),
                    tool_results=list(tool_results or []),
                    tool_specs=list(tool_specs or []),
                    output_contract=output_contract,
                    budget=budget,
                    policies=policies,
                )
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                compiled = CompiledContext.model_validate(cached)
                compiled.from_cache = True
                # The compiled content is identical, but each run gets its
                # own packet identity and trace linkage.
                compiled.packet.id = new_id("ctx")
                compiled.packet.trace_parent_id = trace_parent_id
                compiled.source_evidence = list(evidence or [])
                compiled.source_memory = list(memory or [])
                compiled.source_tool_results = list(tool_results or [])
                return compiled

        # 1-3. collect, normalize, classify (type is assigned at collection).
        # The warm candidate arena reuses this query-independent prep when the
        # candidate set is unchanged since a recent compile.
        arena_key: str | None = None
        prepared: PreparedCandidates | None = None
        if self.arena is not None:
            arena_key = self.arena.fingerprint(
                self._candidate_signature(
                    evidence=list(evidence or []),
                    memory=list(memory or []),
                    tool_results=list(tool_results or []),
                    privacy=policies.privacy,
                    tenant_id=user_input.tenant_id,
                )
            )
            prepared = self.arena.get(arena_key)

        if prepared is not None:
            self.arena_hits += 1
            candidates = prepared.candidates
            excluded.extend(prepared.excluded)
        else:
            candidates = self._collect(
                evidence=evidence or [], memory=memory or [], tool_results=tool_results or []
            )
            candidates = self._normalize(candidates)

            # Policy screen: drop items whose privacy scope conflicts with the run.
            static_excluded: list[dict[str, Any]] = []
            if policies.privacy != "open":
                allowed: list[ContextCandidate] = []
                for candidate in candidates:
                    source = candidate.source
                    owner = getattr(source, "owner_id", None) or getattr(source, "tenant_id", None)
                    if (
                        candidate.type == "memory"
                        and owner
                        and user_input.tenant_id
                        and getattr(source, "scope", None) is not None
                        and str(source.scope.value) in ("tenant", "organization")
                        and owner != user_input.tenant_id
                    ):
                        static_excluded.append(
                            {"id": candidate.id, "reason": "privacy_scope_mismatch"}
                        )
                        continue
                    allowed.append(candidate)
                candidates = allowed
            excluded.extend(static_excluded)
            if self.arena is not None and arena_key is not None:
                self.arena.put(
                    arena_key,
                    PreparedCandidates(candidates=candidates, excluded=static_excluded),
                )

        # 3.5. streaming candidate pre-filter: when the evidence pool exceeds
        # ``max_candidates``, bound it by a cheap lexical relevance proxy and a
        # fingerprint dedup *before* scoring, dedup/conflict, and embedding — so
        # those stages and the resident vector footprint never see more than the
        # cap. Query-dependent, so it runs per compile (never cached in the arena)
        # and is a no-op when the cap is unset or the pool already fits.
        if self.options.max_candidates is not None:
            candidates = self._prefilter_candidates(candidates, query, excluded)

        # 4. score (with a per-compile scorer — the feature arena and any semantic
        # embeddings are installed on a fresh scorer so the shared instance stays
        # state-free and concurrent compiles never race on mutable state). The
        # arena derives each candidate's lexical features once and threads them
        # through scoring, dedup, conflict, and selection.
        features = FeatureArena() if self.options.single_pass_selection else None
        scorer = await self._scorer_for(candidates, query, features)
        scorer.score_batch(candidates, query)

        # 5. dedupe
        candidates = self._remove_duplicates(candidates, excluded, scorer)

        # 6. conflicts
        candidates, conflicts = self._resolve_conflicts(candidates, excluded, scorer)

        # 8. budget allocation (uses fixed costs for known blocks)
        instruction_tokens = sum(count_tokens(i.text) for i in instructions or [])
        constraint_tokens = sum(count_tokens(c.text) for c in constraints or [])
        example_tokens = sum(count_tokens(e.input + e.output) for e in examples or [])
        task_tokens = count_tokens(query)
        schema_tokens = (
            count_tokens(str(output_contract.schema_def)) if output_contract and output_contract.schema_def else 0
        )
        reserve_tokens = 0
        if self.options.reserve_response_tokens:
            reserve = budget.max_output_tokens
            if tool_specs:
                reserve += budget.max_output_tokens  # headroom for one tool-loop round
            # Never let the reservation starve the context blocks.
            reserve_tokens = min(reserve, budget.max_input_tokens // 4)
        allocation = self.allocator.allocate(
            budget.max_input_tokens,
            task_type=objective.task_type,
            fixed_costs={
                "instructions": instruction_tokens + constraint_tokens,
                "examples": example_tokens,
                "user_task": task_tokens,
                "schema": schema_tokens,
            },
            reserve_tokens=reserve_tokens,
        )

        evidence_pool = [c for c in candidates if c.type == "evidence"]
        memory_pool = [c for c in candidates if c.type == "memory"]
        tool_pool = [c for c in candidates if c.type == "tool_result"]

        # 7+8. select under per-block budgets (compression happens inline).
        selected_evidence = self._select(
            evidence_pool,
            budget_tokens=allocation.block("evidence").tokens,
            max_items=self.options.max_evidence_items,
            query=query,
            excluded=excluded,
            scorer=scorer,
        )
        selected_memory = self._select(
            memory_pool,
            budget_tokens=allocation.block("memory").tokens,
            max_items=self.options.max_memory_items,
            query=query,
            excluded=excluded,
            scorer=scorer,
        )
        selected_tools = self._select(
            tool_pool,
            budget_tokens=allocation.block("tool_results").tokens,
            max_items=self.options.max_evidence_items,
            query=query,
            excluded=excluded,
            scorer=scorer,
        )
        allocation.block("evidence").used_tokens = sum(c.token_cost for c in selected_evidence)
        allocation.block("memory").used_tokens = sum(c.token_cost for c in selected_memory)
        allocation.block("tool_results").used_tokens = sum(c.token_cost for c in selected_tools)

        # 9. order
        selected_evidence = self._order(selected_evidence, query)

        # Per-app resident-memory ceiling: slim the packet and evict the
        # lowest-utility evidence until the estimated footprint fits.
        effective_slim = self.options.slim_packets
        if self.options.max_resident_bytes is not None:
            effective_slim, selected_evidence = self._enforce_footprint(
                selected_evidence, selected_memory, effective_slim, excluded
            )
            allocation.block("evidence").used_tokens = sum(
                c.token_cost for c in selected_evidence
            )

        # Rebuild typed items from selected candidates (possibly compressed).
        final_evidence: list[EvidenceItem] = []
        for candidate in selected_evidence:
            item: EvidenceItem = candidate.source
            final_evidence.append(
                item.model_copy(
                    update={
                        "text": candidate.content,
                        "relevance": candidate.scores.relevance,
                        "token_cost": candidate.token_cost,
                    }
                )
            )
        final_memory: list[MemoryItem] = [c.source for c in selected_memory]
        memory_excluded = [e for e in excluded if any(
            e.get("id") == c.id for c in memory_pool if c not in selected_memory
        )]

        ledger: list[dict[str, Any]] = []
        if self.options.use_evidence_ledger and final_evidence:
            ledger = await distill_evidence_ledger(final_evidence, query)

        ir = ContextIR(
            objective=objective,
            instructions=list(instructions or []),
            constraints=list(constraints or []),
            examples=list(examples or []),
            input=user_input,
            memory=final_memory,
            evidence=final_evidence,
            tool_specs=list(tool_specs or []),
            output_contract=output_contract or OutputContractRef(),
            budgets=budget,
            policies=policies,
            evidence_ledger=ledger,
            metadata={"conflicts": conflicts} if conflicts else {},
        )

        token_count = (
            instruction_tokens
            + constraint_tokens
            + example_tokens
            + task_tokens
            + schema_tokens
            + allocation.block("evidence").used_tokens
            + allocation.block("memory").used_tokens
            + allocation.block("tool_results").used_tokens
        )

        # 10-11. render + validate packet.
        if token_count > budget.max_input_tokens:
            raise ContextCompileError(
                f"compiled context ({token_count} tokens) exceeds budget "
                f"({budget.max_input_tokens})",
                details={"token_count": token_count},
            )
        packet = ContextPacket.from_ir(
            ir,
            excluded_report=excluded,
            budget_report=allocation.report(),
            conflicts=conflicts,
            memory_excluded=memory_excluded,
            trace_parent_id=trace_parent_id,
            token_count=token_count,
            slim=effective_slim,
        )
        resident_bytes = estimate_resident_bytes(
            [c.content for c in selected_evidence],
            [c.content for c in selected_memory],
            slim=effective_slim,
        )
        compiled = CompiledContext(
            ir=ir,
            packet=packet,
            excluded_report=excluded,
            budget_report=allocation.report(),
            conflicts=conflicts,
            token_count=token_count,
            resident_bytes=resident_bytes,
            source_evidence=list(evidence or []),
            source_memory=list(memory or []),
            source_tool_results=list(tool_results or []),
        )
        if self.cache is not None and cache_key is not None:
            self.cache.set(
                cache_key,
                compiled.model_dump(mode="json"),
                source_ids=sorted({e.source_id for e in compiled.ir.evidence if e.source_id}),
            )
        return compiled

    @staticmethod
    def _prefix_blocks(
        objective: Objective,
        instructions: list[Instruction],
        constraints: list[Constraint],
        user_input: UserInput,
    ) -> dict[str, Any]:
        """The evidence-independent prefix: objective, instructions,
        constraints, and the user task. Known before any candidate is scored."""
        return {
            "objective": objective.text,
            "instructions": [i.text for i in instructions],
            "constraints": [c.text for c in constraints],
            "task": user_input.text or "",
        }

    async def compile_streaming(
        self,
        *,
        objective: Objective,
        user_input: UserInput,
        instructions: list[Instruction] | None = None,
        constraints: list[Constraint] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompileStreamEvent]:
        """Compile, streaming the result as it becomes available.

        The stable prefix — objective, instructions, constraints, and the task —
        is always included and does not depend on which evidence is selected, so
        it is emitted first, *before* any candidate is scored: a downstream
        consumer (the prompt compiler or the provider transport) can begin on
        the prefix while selection runs. The selected evidence follows once
        scoring and selection finish, then a terminal ``done`` event carries the
        full :class:`CompiledContext`. Back-pressure is the async generator
        itself — scoring and rendering advance only as the consumer pulls.

        ``"".join`` of the prefix/evidence is not the contract; the terminal
        ``done`` event's :class:`CompiledContext` is authoritative and identical
        to :meth:`compile` for the same inputs.
        """
        instructions = list(instructions or [])
        constraints = list(constraints or [])
        blocks = self._prefix_blocks(objective, instructions, constraints, user_input)
        prefix_text = "\n".join(
            part
            for part in (
                blocks["objective"],
                "\n".join(blocks["instructions"]),
                "\n".join(blocks["constraints"]),
                blocks["task"],
            )
            if part
        )
        yield CompileStreamEvent(type="prefix", text=prefix_text, blocks=blocks)

        compiled = await self.compile(
            objective=objective,
            user_input=user_input,
            instructions=instructions,
            constraints=constraints,
            **kwargs,
        )
        yield CompileStreamEvent(type="evidence", evidence=compiled.packet.evidence_items)
        yield CompileStreamEvent(type="done", result=compiled)

    async def recompile(
        self,
        previous: CompiledContext,
        *,
        objective: Objective | None = None,
        user_input: UserInput | None = None,
        add_evidence: list[EvidenceItem] | None = None,
        remove_evidence_ids: list[str] | None = None,
        add_memory: list[MemoryItem] | None = None,
        remove_memory_ids: list[str] | None = None,
        budget: Budget | None = None,
    ) -> CompiledContext:
        """Partial recompile after a packet edit.

        Re-runs the pipeline over the previous compile's *original* inputs
        with the given edits applied, instead of re-collecting from scratch.
        Unchanged texts hit the memoized tokenizers (and, when a compile
        cache is attached, an unchanged edit set is a full cache hit), so
        the cost is proportional to the edit, not the packet.
        """
        removed_evidence = set(remove_evidence_ids or [])
        removed_memory = set(remove_memory_ids or [])
        evidence = [e for e in previous.source_evidence if e.id not in removed_evidence]
        evidence.extend(add_evidence or [])
        memory = [m for m in previous.source_memory if m.id not in removed_memory]
        memory.extend(add_memory or [])
        ir = previous.ir
        return await self.compile(
            objective=objective or ir.objective,
            user_input=user_input or ir.input,
            instructions=ir.instructions,
            constraints=ir.constraints,
            examples=ir.examples,
            evidence=evidence,
            memory=memory,
            tool_results=previous.source_tool_results,
            tool_specs=ir.tool_specs,
            output_contract=ir.output_contract,
            budget=budget or ir.budgets,
            policies=ir.policies,
            trace_parent_id=previous.packet.trace_parent_id,
        )
