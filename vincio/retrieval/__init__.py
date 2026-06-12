"""Vincio retrieval engine."""

from .chunking import CHUNKERS, chunk_document
from .embeddings import CachedEmbedder, Embedder, LocalHashEmbedder, ProviderEmbedder, cosine
from .engine import QueryPlan, RetrievalEngine, RetrievalResult, reciprocal_rank_fusion
from .graph_retrieval import EntityGraph, GraphPath
from .graphrag import Community, GraphRAG, detect_communities
from .hierarchy import AutoMergingIndex, contextualize_chunks
from .indexes import BM25Index, Index, SearchFilter, SearchHit, VectorIndex, build_filter
from .late_interaction import LateInteractionIndex
from .live import LiveIndex
from .query_understanding import QUERY_STRATEGIES, QueryExpansion, QueryUnderstanding
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
from .sparse import CallableSparseEncoder, LocalImpactEncoder, SparseEncoder, SparseIndex

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
    "Community",
    "GraphRAG",
    "detect_communities",
    "AutoMergingIndex",
    "contextualize_chunks",
    "BM25Index",
    "Index",
    "SearchFilter",
    "SearchHit",
    "VectorIndex",
    "build_filter",
    "LateInteractionIndex",
    "LiveIndex",
    "QUERY_STRATEGIES",
    "QueryExpansion",
    "QueryUnderstanding",
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
    "CallableSparseEncoder",
    "LocalImpactEncoder",
    "SparseEncoder",
    "SparseIndex",
]
