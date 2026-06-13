"""Vincio memory engine."""

from .consolidation import ConsolidationReport, MemoryConsolidator
from .engine import MemoryEngine, MemorySearchResult, ScopedMemory
from .evals import (
    MemoryEvalCase,
    MemoryEvalReport,
    contradiction_rate,
    evaluate_memory,
    personalization_dataset,
)
from .facts import GroundedFact, extract_grounded_facts
from .graph import MemoryEdge, MemoryGraph, MemoryNode
from .policies import (
    MemoryCandidate,
    MemoryWritePolicy,
    classify_memory_type,
    decayed_confidence,
    detect_contradiction,
    extract_memory_candidates,
    importance_score,
    stability_score,
)
from .stores import InMemoryMemoryStore, MemoryStore, SQLiteMemoryStore
from .summarizers import SessionSummarizer, extractive_summary

__all__ = [
    "MemoryEngine",
    "MemorySearchResult",
    "ScopedMemory",
    "ConsolidationReport",
    "MemoryConsolidator",
    "MemoryEvalCase",
    "MemoryEvalReport",
    "contradiction_rate",
    "evaluate_memory",
    "personalization_dataset",
    "GroundedFact",
    "extract_grounded_facts",
    "MemoryEdge",
    "MemoryGraph",
    "MemoryNode",
    "MemoryCandidate",
    "MemoryWritePolicy",
    "classify_memory_type",
    "decayed_confidence",
    "detect_contradiction",
    "extract_memory_candidates",
    "importance_score",
    "stability_score",
    "InMemoryMemoryStore",
    "MemoryStore",
    "SQLiteMemoryStore",
    "SessionSummarizer",
    "extractive_summary",
]
