"""Vector-store factory.

One entry point for every backend that implements the retrieval
:class:`~vincio.retrieval.indexes.Index` protocol. The in-memory backend has
no dependencies; the rest import their client lazily and raise a clear
:class:`~vincio.core.errors.StorageError` when the extra is not installed::

    from vincio.retrieval import LocalHashEmbedder
    from vincio.storage import build_vector_index

    index = build_vector_index("qdrant", LocalHashEmbedder(), url="http://localhost:6333")
"""

from __future__ import annotations

from typing import Any

from ..core.errors import ConfigError
from ..retrieval.embeddings import Embedder

__all__ = ["VECTOR_BACKENDS", "build_vector_index"]

VECTOR_BACKENDS = ("memory", "qdrant", "pgvector", "chroma", "pinecone", "lancedb")


def build_vector_index(kind: str, embedder: Embedder, **options: Any) -> Any:
    """Construct a vector index by backend name.

    ``memory`` (alias ``local``) is the dependency-free brute-force index.
    ``pgvector`` requires a ``dsn=`` keyword; the others accept their usual
    connection options (``url``/``api_key``/``path``/``uri``/…).
    """
    if kind in ("memory", "local"):
        from ..retrieval.indexes import VectorIndex

        return VectorIndex(embedder)
    if kind == "qdrant":
        from .qdrant import QdrantVectorIndex

        return QdrantVectorIndex(embedder, **options)
    if kind in ("pgvector", "postgres"):
        from .postgres import PgVectorIndex

        dsn = options.pop("dsn", None)
        if not dsn:
            raise ConfigError("pgvector vector index requires a dsn= keyword")
        return PgVectorIndex(dsn, embedder, **options)
    if kind == "chroma":
        from .chroma import ChromaVectorIndex

        return ChromaVectorIndex(embedder, **options)
    if kind == "pinecone":
        from .pinecone import PineconeVectorIndex

        return PineconeVectorIndex(embedder, **options)
    if kind == "lancedb":
        from .lancedb import LanceDBVectorIndex

        return LanceDBVectorIndex(embedder, **options)
    raise ConfigError(f"unknown vector backend {kind!r}; known: {list(VECTOR_BACKENDS)}")
