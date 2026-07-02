"""Vincio context compiler: IR, packet, scoring, budgeting, compression."""

from .budgeting import DEFAULT_ALLOCATION, BlockBudget, BudgetAllocation, BudgetAllocator
from .compiler import (
    CompiledContext,
    CompileStreamEvent,
    ContextCompiler,
    ContextCompilerOptions,
)
from .compression import (
    CompressionResult,
    distill_evidence_ledger,
    extractive_compress,
    split_sentences,
    truncate_to_tokens,
)
from .ir import ContextIR, OutputContractRef
from .llmlingua import (
    LLMLinguaCompressor,
    TokenImportanceScorer,
    compression_faithfulness,
    faithfulness_preserved,
    salient_units,
)
from .longhorizon import (
    CompactionRecord,
    ContextBudget,
    ContextBudgetReport,
    ContextCompactor,
    ContextGovernor,
    RelevanceDecay,
    RunSpan,
)
from .packet import ContextPacket
from .receipt import (
    BudgetSummary,
    CompileReceipt,
    ConflictSummary,
    PrivacySummary,
    ReceiptItem,
    RenderInfo,
)
from .scoring import (
    ContextCandidate,
    ContextScorer,
    ContextScores,
    ScoringWeights,
    lexical_similarity,
    shingle_similarity,
)

__all__ = [
    "BudgetAllocator",
    "BudgetAllocation",
    "BlockBudget",
    "DEFAULT_ALLOCATION",
    "CompiledContext",
    "CompileStreamEvent",
    "ContextCompiler",
    "ContextCompilerOptions",
    "CompressionResult",
    "distill_evidence_ledger",
    "extractive_compress",
    "split_sentences",
    "truncate_to_tokens",
    "LLMLinguaCompressor",
    "TokenImportanceScorer",
    "compression_faithfulness",
    "faithfulness_preserved",
    "salient_units",
    "ContextIR",
    "OutputContractRef",
    # long-horizon context engineering
    "ContextCompactor",
    "ContextGovernor",
    "ContextBudget",
    "ContextBudgetReport",
    "RelevanceDecay",
    "RunSpan",
    "CompactionRecord",
    "ContextPacket",
    # compile receipt
    "CompileReceipt",
    "ReceiptItem",
    "ConflictSummary",
    "BudgetSummary",
    "PrivacySummary",
    "RenderInfo",
    "ContextCandidate",
    "ContextScorer",
    "ContextScores",
    "ScoringWeights",
    "lexical_similarity",
    "shingle_similarity",
]
