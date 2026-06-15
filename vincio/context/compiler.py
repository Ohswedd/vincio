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
from .packet import ContextPacket
from .scoring import (
    ContextCandidate,
    ContextScorer,
    ScoringWeights,
    near_duplicate_score,
    shingle_similarity,
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


def _looks_negated(text: str) -> bool:
    import re

    return bool(re.search(r"\b(not|no|never|cannot|can't|won't|isn't|doesn't|without)\b", text.lower()))


class ContextCompiler:
    def __init__(self, options: ContextCompilerOptions | None = None, *, cache: Any | None = None) -> None:
        self.options = options or ContextCompilerOptions()
        self.scorer = ContextScorer(self.options.weights)
        self.allocator = BudgetAllocator()
        self.cache = cache  # ContextCompileCache | None
        self.cache_hits = 0
        # The inline evidence compressor (1.4). Defaults to extractive
        # compression; a learned compressor (e.g. ``LLMLinguaCompressor``) is a
        # drop-in with the same signature, installed via
        # ``app.use_learned_compression(...)`` once faithfulness-gated.
        self.compressor: Any = extractive_compress

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
            text = (item.text or "").strip()
            if not text:
                continue
            candidates.append(
                ContextCandidate(
                    id=item.id,
                    type="evidence",
                    content=text,
                    token_cost=item.token_cost or count_tokens(text),
                    source=item,
                    authority=item.authority,
                    provenance=item.provenance,
                    created_at=None,
                    metadata={
                        "citation_ref": item.citation_ref,
                        "source_id": item.source_id,
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

    def _remove_duplicates(
        self, candidates: list[ContextCandidate], excluded: list[dict[str, Any]]
    ) -> list[ContextCandidate]:
        kept: list[ContextCandidate] = []
        for candidate in sorted(candidates, key=lambda c: c.scores.total, reverse=True):
            duplicate_of = None
            for existing in kept:
                if near_duplicate_score(candidate.content, existing.content) >= self.options.duplicate_threshold:
                    duplicate_of = existing.id
                    break
            if duplicate_of is not None:
                excluded.append(
                    {"id": candidate.id, "reason": "duplicate", "duplicate_of": duplicate_of}
                )
            else:
                kept.append(candidate)
        return kept

    def _resolve_conflicts(
        self,
        candidates: list[ContextCandidate],
        excluded: list[dict[str, Any]],
    ) -> tuple[list[ContextCandidate], list[dict[str, Any]]]:
        """Higher authority wins; newer wins on similar authority;
        otherwise keep both and report the conflict to the model."""
        conflicts: list[dict[str, Any]] = []
        dropped: set[str] = set()
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                if a.id in dropped or b.id in dropped:
                    continue
                if a.type != b.type or a.type not in ("evidence", "memory"):
                    continue
                similarity = shingle_similarity(a.content, b.content)
                if not (0.30 <= similarity < self.options.duplicate_threshold):
                    continue
                if _looks_negated(a.content) == _looks_negated(b.content):
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
                freshness_a = self.scorer.freshness(a)
                freshness_b = self.scorer.freshness(b)
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
                    }
                )
        return [c for c in candidates if c.id not in dropped], conflicts

    # -- selection under budget ------------------------------------------------------

    def _select(
        self,
        candidates: list[ContextCandidate],
        *,
        budget_tokens: int,
        max_items: int,
        query: str,
        excluded: list[dict[str, Any]],
    ) -> list[ContextCandidate]:
        """Greedy utility-per-token selection with novelty rescoring."""
        selected: list[ContextCandidate] = []
        used = 0
        pool = []
        for candidate in candidates:
            # Relevance gate: evidence that shares nothing with the task is
            # context pollution regardless of authority/novelty baselines.
            if candidate.type == "evidence":
                relevance = self.scorer.relevance(candidate, query)
                answerability = self.scorer.answerability(candidate, query)
                upstream = float(candidate.metadata.get("upstream_relevance") or 0.0)
                if max(relevance, answerability, upstream) < self.options.min_relevance:
                    excluded.append(
                        {
                            "id": candidate.id,
                            "reason": "low_relevance",
                            "score": round(relevance, 4),
                        }
                    )
                    continue
            pool.append(candidate)
        while pool and len(selected) < max_items:
            for candidate in pool:
                self.scorer.score(candidate, query=query, selected=selected)
            pool.sort(key=lambda c: c.scores.total, reverse=True)
            best = pool.pop(0)
            if best.scores.total < self.options.min_score:
                excluded.append(
                    {"id": best.id, "reason": "low_relevance", "score": round(best.scores.total, 4)}
                )
                for remaining in pool:
                    excluded.append(
                        {
                            "id": remaining.id,
                            "reason": "low_relevance",
                            "score": round(remaining.scores.total, 4),
                        }
                    )
                break
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
                            continue
                excluded.append(
                    {"id": best.id, "reason": "budget_exceeded", "token_cost": best.token_cost}
                )
                continue
            selected.append(best)
            used += best.token_cost
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

        # 4. score
        for candidate in candidates:
            self.scorer.score(candidate, query=query)

        # 5. dedupe
        candidates = self._remove_duplicates(candidates, excluded)

        # 6. conflicts
        candidates, conflicts = self._resolve_conflicts(candidates, excluded)

        # 8. budget allocation (uses fixed costs for known blocks)
        instruction_tokens = sum(count_tokens(i.text) for i in instructions or [])
        constraint_tokens = sum(count_tokens(c.text) for c in constraints or [])
        example_tokens = sum(count_tokens(e.input + e.output) for e in examples or [])
        task_tokens = count_tokens(query)
        schema_tokens = (
            count_tokens(str(output_contract.schema_def)) if output_contract and output_contract.schema_def else 0
        )
        allocation = self.allocator.allocate(
            budget.max_input_tokens,
            task_type=objective.task_type,
            fixed_costs={
                "instructions": instruction_tokens + constraint_tokens,
                "examples": example_tokens,
                "user_task": task_tokens,
                "schema": schema_tokens,
            },
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
        )
        selected_memory = self._select(
            memory_pool,
            budget_tokens=allocation.block("memory").tokens,
            max_items=self.options.max_memory_items,
            query=query,
            excluded=excluded,
        )
        selected_tools = self._select(
            tool_pool,
            budget_tokens=allocation.block("tool_results").tokens,
            max_items=self.options.max_evidence_items,
            query=query,
            excluded=excluded,
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
