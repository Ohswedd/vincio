"""Vincio storage layer.

Optional adapters (postgres/qdrant/neo4j/redis/duckdb) import lazily —
access them via their modules or the factory helpers so the core package
works without the optional dependencies installed.
"""

from .base import (
    BlobStore,
    FileBlobStore,
    InMemoryMetadataStore,
    MetadataStore,
    create_metadata_store,
    parse_storage_url,
)
from .sqlite import SQLiteMetadataStore
from .vectorstores import VECTOR_BACKENDS, build_vector_index

__all__ = [
    "BlobStore",
    "FileBlobStore",
    "InMemoryMetadataStore",
    "MetadataStore",
    "create_metadata_store",
    "parse_storage_url",
    "SQLiteMetadataStore",
    "VECTOR_BACKENDS",
    "build_vector_index",
]
