"""Vincio context compiler: IR, packet, scoring, budgeting, compression."""

from .budgeting import DEFAULT_ALLOCATION, BlockBudget, BudgetAllocation, BudgetAllocator
from .compiler import CompiledContext, ContextCompiler, ContextCompilerOptions
from .compression import (
    CompressionResult,
    distill_evidence_ledger,
    extractive_compress,
    split_sentences,
    truncate_to_tokens,
)
from .ir import ContextIR, OutputContractRef
from .packet import ContextPacket
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
    "ContextCompiler",
    "ContextCompilerOptions",
    "CompressionResult",
    "distill_evidence_ledger",
    "extractive_compress",
    "split_sentences",
    "truncate_to_tokens",
    "ContextIR",
    "OutputContractRef",
    "ContextPacket",
    "ContextCandidate",
    "ContextScorer",
    "ContextScores",
    "ScoringWeights",
    "lexical_similarity",
    "shingle_similarity",
]
