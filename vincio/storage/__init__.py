"""Vincio storage layer.

Optional adapters (postgres/qdrant/neo4j/redis/duckdb) import lazily —
access them via their modules or the factory helpers so the core package
works without the optional dependencies installed.
"""

from __future__ import annotations

from .base import (
    BlobStore,
    FileBlobStore,
    InMemoryMetadataStore,
    MetadataStore,
    build_metadata_store,
    create_metadata_store,
    parse_storage_url,
)
from .index_regression import IndexRegressionArtifact, IndexRegressionStore, config_key
from .shared_state import (
    IdempotencyStore,
    InMemoryIdempotencyStore,
    InMemoryRateLimiter,
    RateLimitDecision,
    RateLimiter,
    TenantQuotaManager,
)
from .sqlite import SQLiteMetadataStore
from .vectorstores import VECTOR_BACKENDS, build_vector_index

__all__ = [
    "BlobStore",
    "FileBlobStore",
    "InMemoryMetadataStore",
    "MetadataStore",
    "build_metadata_store",
    "create_metadata_store",
    "parse_storage_url",
    "SQLiteMetadataStore",
    "VECTOR_BACKENDS",
    "build_vector_index",
    # shared server state
    "RateLimiter",
    "RateLimitDecision",
    "InMemoryRateLimiter",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "TenantQuotaManager",
    # index/retrieval regression artifacts
    "IndexRegressionArtifact",
    "IndexRegressionStore",
    "config_key",
]
