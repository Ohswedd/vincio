"""Cache layers: response, retrieval, context packet, eval
result, and semantic caches over a shared backend. Tool-result caching lives
in the tool runtime; embedding caching in CachedEmbedder."""

from __future__ import annotations

from typing import Any

from ..core.types import ModelRequest, ModelResponse
from ..core.utils import stable_hash
from ..retrieval.embeddings import Embedder, cosine
from .base import CacheBackend, InMemoryCache

__all__ = ["ResponseCache", "RetrievalCache", "ContextPacketCache", "EvalResultCache", "SemanticCache"]


class ResponseCache:
    """Exact-match model response cache, keyed by the full request hash."""

    def __init__(self, backend: CacheBackend | None = None, *, ttl_s: float | None = 3600.0) -> None:
        self.backend = backend or InMemoryCache()
        self.ttl_s = ttl_s

    def get(self, request: ModelRequest) -> ModelResponse | None:
        payload = self.backend.get(f"resp:{request.hash}")
        if payload is None:
            return None
        return ModelResponse.model_validate(payload)

    def set(self, request: ModelRequest, response: ModelResponse, *, prompt_version: str = "") -> None:
        tags = ["responses", f"model:{request.model}"]
        if prompt_version:
            tags.append(f"prompt:{prompt_version}")
        self.backend.set(
            f"resp:{request.hash}",
            response.model_dump(mode="json", exclude={"raw"}),
            ttl_s=self.ttl_s,
            tags=tags,
        )


class RetrievalCache:
    """Caches retrieval results keyed by query + filters + index version."""

    def __init__(self, backend: CacheBackend | None = None, *, ttl_s: float | None = 600.0) -> None:
        self.backend = backend or InMemoryCache()
        self.ttl_s = ttl_s

    def key(self, query: str, *, top_k: int, tenant_id: str | None, index_version: str) -> str:
        return "retr:" + stable_hash(
            {"q": query, "k": top_k, "tenant": tenant_id, "v": index_version}
        )

    def get(self, key: str) -> list[dict[str, Any]] | None:
        return self.backend.get(key)

    def set(self, key: str, evidence_dumps: list[dict[str, Any]], *, document_ids: list[str]) -> None:
        tags = ["retrieval", *(f"doc:{d}" for d in document_ids[:64])]
        self.backend.set(key, evidence_dumps, ttl_s=self.ttl_s, tags=tags)


class ContextPacketCache:
    """Caches compiled context packets keyed by their input signature."""

    def __init__(self, backend: CacheBackend | None = None, *, ttl_s: float | None = 600.0) -> None:
        self.backend = backend or InMemoryCache()
        self.ttl_s = ttl_s

    def key(self, *, objective: str, query: str, evidence_ids: list[str], memory_ids: list[str], schema_ref: str | None) -> str:
        return "ctx:" + stable_hash(
            {"obj": objective, "q": query, "ev": sorted(evidence_ids), "mem": sorted(memory_ids), "schema": schema_ref}
        )

    def get(self, key: str) -> dict[str, Any] | None:
        return self.backend.get(key)

    def set(self, key: str, packet_dump: dict[str, Any], *, prompt_version: str = "") -> None:
        tags = ["context_packets"]
        if prompt_version:
            tags.append(f"prompt:{prompt_version}")
        self.backend.set(key, packet_dump, ttl_s=self.ttl_s, tags=tags)


class EvalResultCache:
    """Caches per-case eval outputs keyed by (case, system config hash)."""

    def __init__(self, backend: CacheBackend | None = None, *, ttl_s: float | None = None) -> None:
        self.backend = backend or InMemoryCache(default_ttl_s=None)
        self.ttl_s = ttl_s

    def key(self, case_id: str, config_hash: str) -> str:
        return f"eval:{config_hash}:{case_id}"

    def get(self, case_id: str, config_hash: str) -> dict[str, Any] | None:
        return self.backend.get(self.key(case_id, config_hash))

    def set(self, case_id: str, config_hash: str, result: dict[str, Any]) -> None:
        self.backend.set(self.key(case_id, config_hash), result, ttl_s=self.ttl_s, tags=["evals"])


class SemanticCache:
    """Similarity-matched response cache.

    A hit requires: similarity ≥ threshold AND matching policy scope AND
    matching output schema AND entry freshness — only then may a cached
    answer substitute a model call. Use for low-risk tasks only.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        threshold: float = 0.97,
        max_entries: int = 5000,
        ttl_s: float | None = 3600.0,
    ) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._entries: list[dict[str, Any]] = []
        self.hits = 0
        self.misses = 0

    async def get(
        self,
        query: str,
        *,
        policy_scope: str,
        schema_ref: str | None,
    ) -> Any | None:
        import time

        if not self._entries:
            self.misses += 1
            return None
        [vector] = await self.embedder.embed([query])
        now = time.monotonic()
        best_score, best_entry = 0.0, None
        for entry in self._entries:
            if entry["policy_scope"] != policy_scope or entry["schema_ref"] != schema_ref:
                continue
            if self.ttl_s is not None and now - entry["stored_at"] > self.ttl_s:
                continue
            score = cosine(vector, entry["vector"])
            if score > best_score:
                best_score, best_entry = score, entry
        if best_entry is not None and best_score >= self.threshold:
            self.hits += 1
            return best_entry["value"]
        self.misses += 1
        return None

    async def set(
        self,
        query: str,
        value: Any,
        *,
        policy_scope: str,
        schema_ref: str | None,
    ) -> None:
        import time

        [vector] = await self.embedder.embed([query])
        self._entries.append(
            {
                "query": query,
                "vector": vector,
                "value": value,
                "policy_scope": policy_scope,
                "schema_ref": schema_ref,
                "stored_at": time.monotonic(),
            }
        )
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

    def clear(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count
