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

import re
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import ContextCompileError
from ..core.tokens import count_tokens
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
from ..core.utils import new_id
from .budgeting import BudgetAllocator
from .compression import distill_evidence_ledger, extractive_compress
from .ir import ContextIR, OutputContractRef
from .llmlingua import salient_units
from .packet import ContextPacket
from .scoring import (
    ContextCandidate,
    ContextScorer,
    ScoringWeights,
    _terms,
)

__all__ = ["ContextCompilerOptions", "CompiledContext", "ContextCompiler"]


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
    # Opt-in embedding-driven selection (1.7). When enabled *and* a semantic
    # embedder is threaded in, relevance/novelty/dedup/conflict use cosine over
    # cached embeddings and ``_select`` runs MMR with ``mmr_lambda``. Off by
    # default — the default hash embedder is not semantic, so selection stays
    # lexical unless a real embedder is configured and this is turned on.
    semantic_scoring: bool = False
    mmr_lambda: float = 0.7  # MMR relevance/diversity trade-off (1.0 = pure relevance)
    # Reserve response (and one tool-loop round) tokens out of the input budget
    # so the allocator accounts for the full window, not input only (1.7). Capped
    # so it never starves context. On by default; small corpora are unaffected
    # because the flexible blocks still fit everything.
    reserve_response_tokens: bool = True


class CompiledContext(BaseModel):
    ir: ContextIR
    packet: ContextPacket
    excluded_report: list[dict[str, Any]] = Field(default_factory=list)
    budget_report: dict[str, Any] = Field(default_factory=dict)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    token_count: int = 0
    from_cache: bool = False
    # Original (pre-selection) inputs, retained in memory for partial
    # recompiles; excluded from dumps so packets and caches stay slim.
    source_evidence: list[EvidenceItem] = Field(default_factory=list, exclude=True)
    source_memory: list[MemoryItem] = Field(default_factory=list, exclude=True)
    source_tool_results: list[ToolResult] = Field(default_factory=list, exclude=True)


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
        # The inline evidence compressor (1.4). Defaults to extractive
        # compression; a learned compressor (e.g. ``LLMLinguaCompressor``) is a
        # drop-in with the same signature, installed via
        # ``app.use_learned_compression(...)`` once faithfulness-gated.
        self.compressor: Any = extractive_compress

    async def _scorer_for(
        self, candidates: list[ContextCandidate], query: str
    ) -> ContextScorer:
        """The scorer for this compile. Lexical mode returns the shared,
        vector-less scorer. Semantic mode batch-embeds candidate contents and
        the query (content-addressed, so repeats are cheap) and installs the
        vectors on a fresh scorer — keeping the shared instance race-free."""
        if not (self.options.semantic_scoring and self.embedder is not None):
            return self.scorer
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
            return self.scorer
        try:
            vectors_list = await self.embedder.embed(texts)
        except Exception:  # noqa: BLE001 - fall back to lexical if embedding fails
            return self.scorer
        vectors = {text: vec for text, vec in zip(texts, vectors_list, strict=False)}
        scorer = ContextScorer(
            self.options.weights,
            similarity_fn=self.scorer.similarity_fn,
            max_token_cost=self.scorer.max_token_cost,
            freshness_half_life_days=self.scorer.freshness_half_life_days,
        )
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

    # -- candidate collection -----------------------------------------------------

    def _collect(
        self,
        *,
        evidence: list[EvidenceItem],
        memory: list[MemoryItem],
        tool_results: list[ToolResult],
    ) -> list[ContextCandidate]:
        candidates: list[ContextCandidate] = []
        for item in evidence:
            # 2.0: text, image, and table evidence are all candidates. The
            # scorable surrogate (text / caption / table Markdown) drives
            # relevance/dedup/ordering; image/table carry the non-text payload.
            text = item.scorable_text.strip()
            if not text and item.modality == "text":
                continue
            if not text and item.image is None and item.table is None:
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

    # -- normalization / dedup / conflicts ----------------------------------------

    @staticmethod
    def _normalize(candidates: list[ContextCandidate]) -> list[ContextCandidate]:
        for candidate in candidates:
            candidate.content = " ".join(candidate.content.split())
            if not candidate.token_cost:
                candidate.token_cost = count_tokens(candidate.content)
        return [c for c in candidates if c.content]

    @staticmethod
    def _block_tokens(text: str) -> set[str]:
        """Tokens a candidate is indexed under for similarity blocking. Two
        passages that share none of these have shingle *and* containment
        similarity 0, so they can be skipped without computing either — making
        the blocking pass exact, not an approximation."""
        raw = set(re.findall(r"[a-z0-9]+", text.lower()))
        return raw | set(_terms(text))

    def _remove_duplicates(
        self,
        candidates: list[ContextCandidate],
        excluded: list[dict[str, Any]],
        scorer: ContextScorer,
    ) -> list[ContextCandidate]:
        kept: list[ContextCandidate] = []
        semantic = scorer.semantic
        # Lexical mode: inverted token index so each candidate is compared only
        # against kept items that could possibly be near-duplicates (near-linear
        # for diverse pools, exact at the 0.85 threshold). Semantic mode compares
        # all kept items (cosine catches paraphrases with no shared tokens).
        index: dict[str, list[int]] = defaultdict(list)
        for candidate in sorted(candidates, key=lambda c: c.scores.total, reverse=True):
            if semantic:
                neighbors: list[ContextCandidate] = kept
            else:
                positions: set[int] = set()
                tokens = self._block_tokens(candidate.content)
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
                    for token in self._block_tokens(candidate.content):
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
        index: dict[str, list[int]] = defaultdict(list)
        for i in same_type:
            for token in self._block_tokens(candidates[i].content):
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
        for candidate in pool:
            scorer.score(candidate, query=query, selected=[])
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
                if self.options.compress_evidence and best.type == "evidence":
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
        candidates = self._collect(
            evidence=evidence or [], memory=memory or [], tool_results=tool_results or []
        )
        candidates = self._normalize(candidates)

        # Policy screen: drop items whose privacy scope conflicts with the run.
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
                    excluded.append({"id": candidate.id, "reason": "privacy_scope_mismatch"})
                    continue
                allowed.append(candidate)
            candidates = allowed

        # 4. score (with a per-compile scorer — semantic embeddings are installed
        # on a fresh scorer so the shared instance stays vector-less and concurrent
        # compiles never race on mutable state).
        scorer = await self._scorer_for(candidates, query)
        for candidate in candidates:
            scorer.score(candidate, query=query)

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
            slim=self.options.slim_packets,
        )
        compiled = CompiledContext(
            ir=ir,
            packet=packet,
            excluded_report=excluded,
            budget_report=allocation.report(),
            conflicts=conflicts,
            token_count=token_count,
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
