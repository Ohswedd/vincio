"""Vincio retrieval engine."""

from .chunking import CHUNKERS, chunk_document
from .embeddings import CachedEmbedder, Embedder, LocalHashEmbedder, ProviderEmbedder, cosine
from .engine import QueryPlan, RetrievalEngine, RetrievalResult, reciprocal_rank_fusion
from .graph_retrieval import EntityGraph, GraphPath
from .indexes import BM25Index, Index, SearchFilter, SearchHit, VectorIndex, build_filter
from .reasoning_retrieval import FactCoverage, FactRequirement, FactSchema, ReasoningRetriever
from .rerankers import (
    AuthorityReranker,
    CrossEncoderReranker,
    HeuristicReranker,
    LLMReranker,
    RecencyReranker,
    Reranker,
    build_reranker,
)

__all__ = [
    "CHUNKERS",
    "chunk_document",
    "CachedEmbedder",
    "Embedder",
    "LocalHashEmbedder",
    "ProviderEmbedder",
    "cosine",
    "QueryPlan",
    "RetrievalEngine",
    "RetrievalResult",
    "reciprocal_rank_fusion",
    "EntityGraph",
    "GraphPath",
    "BM25Index",
    "Index",
    "SearchFilter",
    "SearchHit",
    "VectorIndex",
    "build_filter",
    "FactCoverage",
    "FactRequirement",
    "FactSchema",
    "ReasoningRetriever",
    "AuthorityReranker",
    "CrossEncoderReranker",
    "HeuristicReranker",
    "LLMReranker",
    "RecencyReranker",
    "Reranker",
    "build_reranker",
]
