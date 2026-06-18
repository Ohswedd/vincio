"""Content-addressed compilation caches.

Incremental & cached compilation: the prompt-compile, chunking, and
context-compile stages are pure functions of their inputs, so each gets a
content-addressed cache — unchanged inputs are never recomputed. Keys cover
*every* input that affects the output (content, options, compiler version),
which makes the caches safe to leave enabled.
"""

from __future__ import annotations

from typing import Any

from ..core.utils import stable_hash
from .base import CacheBackend, InMemoryCache

__all__ = ["PromptCompileCache", "ChunkCache", "ContextCompileCache"]


class PromptCompileCache:
    """Caches compiled prompts keyed by spec hash + task + context blocks."""

    def __init__(self, backend: CacheBackend | None = None, *, ttl_s: float | None = 3600.0) -> None:
        self.backend = backend or InMemoryCache()
        self.ttl_s = ttl_s

    def key(self, payload: dict[str, Any]) -> str:
        return "pcomp:" + stable_hash(payload)

    def get(self, key: str) -> dict[str, Any] | None:
        return self.backend.get(key)

    def set(self, key: str, compiled_dump: dict[str, Any], *, spec_hash: str = "") -> None:
        tags = ["prompt_compile"]
        if spec_hash:
            tags.append(f"prompt:{spec_hash}")
        self.backend.set(key, compiled_dump, ttl_s=self.ttl_s, tags=tags)


class ChunkCache:
    """Caches chunking output keyed by document content + strategy + sizes.

    The key is the document *content* hash (not its id), so re-loading an
    unchanged file — or the same content under a new Document id — is a hit.
    Chunk ids and document ids are restored from the requesting document so
    provenance stays correct.
    """

    def __init__(self, backend: CacheBackend | None = None, *, ttl_s: float | None = None) -> None:
        self.backend = backend or InMemoryCache(default_ttl_s=None)
        self.ttl_s = ttl_s

    def key(self, *, content: str, strategy: str, size: int, overlap: int) -> str:
        return "chunk:" + stable_hash(
            {"content": content, "strategy": strategy, "size": size, "overlap": overlap}
        )

    def get(self, key: str) -> list[dict[str, Any]] | None:
        return self.backend.get(key)

    def set(self, key: str, chunk_dumps: list[dict[str, Any]]) -> None:
        self.backend.set(key, chunk_dumps, ttl_s=self.ttl_s, tags=["chunks"])


class ContextCompileCache:
    """Caches full context-compile results keyed by every compile input.

    A hit returns the compiled IR + packet + reports without re-running
    scoring, dedup, conflict resolution, or selection. Tagged per evidence
    source so document updates invalidate precisely.
    """

    def __init__(self, backend: CacheBackend | None = None, *, ttl_s: float | None = 600.0) -> None:
        self.backend = backend or InMemoryCache()
        self.ttl_s = ttl_s

    def key(self, payload: dict[str, Any]) -> str:
        return "ccomp:" + stable_hash(payload)

    def get(self, key: str) -> dict[str, Any] | None:
        return self.backend.get(key)

    def set(self, key: str, compiled_dump: dict[str, Any], *, source_ids: list[str] | None = None) -> None:
        tags = ["context_compile", "context_packets"]
        tags.extend(f"doc:{source_id}" for source_id in (source_ids or [])[:64])
        self.backend.set(key, compiled_dump, ttl_s=self.ttl_s, tags=tags)
