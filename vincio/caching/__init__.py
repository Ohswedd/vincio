"""Vincio caching system."""

from __future__ import annotations

from .base import CacheBackend, InMemoryCache, SQLiteCache
from .compilation import ChunkCache, ContextCompileCache, PromptCompileCache
from .invalidation import InvalidationManager
from .kvreuse import (
    KVPrefixEntry,
    KVPrefixObservation,
    KVPrefixPool,
    KVReuseReport,
    kv_prefix_key,
)
from .layers import (
    ContextPacketCache,
    EvalResultCache,
    ResponseCache,
    RetrievalCache,
    SemanticCache,
)
from .reasoning import ReasoningTrace, ReasoningTraceCache, reasoning_prefix_key
from .semantic import (
    CalibrationExample,
    CalibrationReport,
    LearnedSemanticCache,
    SemanticCacheEntry,
    SemanticCacheGate,
    SemanticCacheHit,
    SemanticCachePolicy,
    SemanticCacheStats,
    SemanticGateCase,
    SemanticGateReport,
    ThresholdCalibrator,
    lexical_quality,
)

__all__ = [
    "CacheBackend",
    "InMemoryCache",
    "SQLiteCache",
    "InvalidationManager",
    "ChunkCache",
    "ContextCompileCache",
    "ContextPacketCache",
    "EvalResultCache",
    "PromptCompileCache",
    "ResponseCache",
    "RetrievalCache",
    "SemanticCache",
    "ReasoningTrace",
    "ReasoningTraceCache",
    "reasoning_prefix_key",
    # learned semantic cache & near-miss KV reuse
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
    "lexical_quality",
    "kv_prefix_key",
    "KVPrefixEntry",
    "KVPrefixObservation",
    "KVReuseReport",
    "KVPrefixPool",
]
