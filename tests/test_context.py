"""Context compiler unit tests (context scoring, token budgeting)."""

import pytest

from vincio.context import (
    BudgetAllocator,
    ContextCompiler,
    ContextCompilerOptions,
    ContextScorer,
    extractive_compress,
    lexical_similarity,
    shingle_similarity,
    split_sentences,
    truncate_to_tokens,
)
from vincio.context.scoring import ContextCandidate, near_duplicate_score
from vincio.core.types import (
    Budget,
    EvidenceItem,
    Instruction,
    MemoryItem,
    Objective,
    TaskType,
    UserInput,
)


class TestScoring:
    def test_lexical_similarity(self):
        assert lexical_similarity("refund policy for plans", "what is the refund policy") > 0.3
        assert lexical_similarity("refund policy", "quantum chromodynamics") == 0.0

    def test_lexical_similarity_supports_scripts_without_spaces(self):
        assert (
            lexical_similarity(
                "Pythonの最新安定版は3.14です",
                "公式情報ではPythonの最新安定版は3.14です",
            )
            > 0.5
        )
        assert lexical_similarity("最新版本是3.14", "官方文档说明最新版本是3.14") > 0.5

    def test_near_duplicate_catches_filler_variants(self):
        a = "The contract renews automatically unless terminated 60 days before renewal."
        b = "The contract renews automatically unless terminated 60 days before the renewal."
        assert near_duplicate_score(a, b) >= 0.85
        assert shingle_similarity(a, "Completely different sentence about bananas.") < 0.1

    def test_scorer_components(self):
        scorer = ContextScorer()
        candidate = ContextCandidate(
            id="c1",
            type="evidence",
            content="The refund window is 30 days for Pro plans costing $20.",
            authority=0.9,
        )
        scores = scorer.score(candidate, query="What is the refund window?")
        assert scores.relevance > 0.2
        assert scores.question_answerability > 0.1
        assert scores.total > 0

    def test_leakage_risk_penalized(self):
        scorer = ContextScorer()
        risky = ContextCandidate(
            id="r", type="memory", content="refund window is 30 days", leakage_risk=0.9
        )
        safe = ContextCandidate(
            id="s", type="memory", content="refund window is 30 days", leakage_risk=0.0
        )
        assert (
            scorer.score(risky, query="refund window").total
            < scorer.score(safe, query="refund window").total
        )


class TestBudgeting:
    def test_default_allocation_sums_to_total(self):
        allocation = BudgetAllocator().allocate(10_000)
        assert sum(b.tokens for b in allocation.blocks.values()) <= 10_000
        assert allocation.block("evidence").tokens > allocation.block("examples").tokens

    def test_task_adaptive_allocation(self):
        allocator = BudgetAllocator()
        qa = allocator.allocation_for(TaskType.DOCUMENT_QA)
        classification = allocator.allocation_for(TaskType.CLASSIFICATION)
        assert qa["evidence"] > 0.6
        assert classification["evidence"] == 0.0

    def test_fixed_costs_charged(self):
        allocation = BudgetAllocator().allocate(
            1000, fixed_costs={"instructions": 100, "user_task": 50, "examples": 0, "schema": 0}
        )
        assert allocation.block("instructions").tokens == 100
        assert allocation.block("evidence").tokens > 0


class TestCompression:
    def test_multilingual_sentence_split_preserves_trailing_citations(self):
        assert split_sentences("First fact. [E1]") == ["First fact. [E1]"]
        assert split_sentences("最初の事実です。次の事実です。 [E2]") == [
            "最初の事実です。",
            "次の事実です。 [E2]",
        ]

    def test_truncate_to_tokens(self):
        text = "First sentence here. Second sentence follows. Third one too. " * 20
        result = truncate_to_tokens(text, 30)
        assert result.compressed_tokens <= 30
        assert result.method in ("truncate", "hard_truncate")

    def test_extractive_keeps_relevant(self):
        text = (
            "The weather is nice today. The refund window is 30 days. "
            "Cats are wonderful pets. Refunds require an invoice ID."
        )
        result = extractive_compress(text, "refund window invoice", 25)
        assert "refund" in result.text.lower()
        assert "cats" not in result.text.lower()


class TestContextCompiler:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_evidence):
        compiler = ContextCompiler(ContextCompilerOptions(use_evidence_ledger=True))
        compiled = await compiler.compile(
            objective=Objective("Review contract renewal risk", task_type=TaskType.DOCUMENT_QA),
            user_input=UserInput(text="Which renewal clauses are risky?"),
            instructions=[Instruction("Use only provided documents")],
            evidence=sample_evidence
            + [
                EvidenceItem(
                    id="e1b",
                    source_id="D1",
                    text="The contract renews automatically unless terminated 60 days before the renewal.",
                    authority=0.4,
                    relevance=0.8,
                )
            ],
            memory=[MemoryItem(content="User prefers concise summaries.", confidence=0.9)],
            budget=Budget(max_input_tokens=2000),
        )
        kept = {e.id for e in compiled.ir.evidence}
        assert "e1" in kept and "e2" in kept
        assert "e3" not in kept  # irrelevant
        assert "e1b" not in kept  # near-duplicate
        reasons = {x["id"]: x["reason"] for x in compiled.excluded_report}
        assert reasons.get("e3") == "low_relevance"
        assert reasons.get("e1b") == "duplicate"
        assert compiled.packet.spec_hash
        assert compiled.ir.evidence_ledger
        assert compiled.token_count <= 2000

    @pytest.mark.asyncio
    async def test_budget_compression(self):
        long_evidence = EvidenceItem(
            id="big",
            source_id="D1",
            text="The refund policy allows returns. " * 200,
            relevance=0.9,
        )
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=Objective("answer", task_type=TaskType.DOCUMENT_QA),
            user_input=UserInput(text="What is the refund policy?"),
            evidence=[long_evidence],
            budget=Budget(max_input_tokens=300),
        )
        assert compiled.token_count <= 300

    @pytest.mark.asyncio
    async def test_privacy_scope_exclusion(self):
        from vincio.core.types import MemoryScope

        foreign = MemoryItem(
            content="Tenant Acme pays 50k annually",
            scope=MemoryScope.TENANT,
            owner_id="acme",
            confidence=0.9,
        )
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=Objective("answer"),
            user_input=UserInput(text="what does the tenant pay annually", tenant_id="other_co"),
            memory=[foreign],
            budget=Budget(max_input_tokens=2000),
        )
        assert not compiled.ir.memory
        assert any(x["reason"] == "privacy_scope_mismatch" for x in compiled.excluded_report)
