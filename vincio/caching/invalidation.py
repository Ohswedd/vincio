"""Cache invalidation.

Maps domain events to tag-based invalidation across all registered cache
backends:

- document updated   → ``doc:<id>``, ``retrieval``, ``context_packets``
- policy changed     → everything policy-scoped (responses, packets, semantic)
- prompt version     → ``prompt:<version>``, ``responses``, ``context_packets``
- schema changed     → ``responses``, ``context_packets``
- tool data stale    → handled by the tool runtime cache (per-tool clear)
- scope changed      → full clear (tenant/user boundaries moved)
"""

from __future__ import annotations

from typing import Any

from ..core.events import EventBus
from .base import CacheBackend

__all__ = ["InvalidationManager"]


class InvalidationManager:
    def __init__(self, backends: list[CacheBackend] | None = None) -> None:
        self.backends: list[CacheBackend] = list(backends or [])
        self._semantic_caches: list[Any] = []

    def register(self, backend: CacheBackend) -> None:
        self.backends.append(backend)

    def register_semantic(self, semantic_cache: Any) -> None:
        self._semantic_caches.append(semantic_cache)

    def _invalidate_tags(self, tags: list[str]) -> int:
        removed = 0
        for backend in self.backends:
            for tag in tags:
                removed += backend.invalidate_tag(tag)
        return removed

    # -- triggers ------------------------------------------------------

    def document_updated(self, document_id: str) -> int:
        return self._invalidate_tags([f"doc:{document_id}", "retrieval", "context_packets"])

    def policy_changed(self) -> int:
        removed = self._invalidate_tags(["responses", "context_packets", "retrieval"])
        for cache in self._semantic_caches:
            removed += cache.clear()
        return removed

    def prompt_version_changed(self, version: str | None = None) -> int:
        tags = ["responses", "context_packets", "prompt_compile"]
        if version:
            tags.insert(0, f"prompt:{version}")
        return self._invalidate_tags(tags)

    def output_schema_changed(self) -> int:
        removed = self._invalidate_tags(["responses", "context_packets"])
        for cache in self._semantic_caches:
            removed += cache.clear()
        return removed

    def scope_changed(self) -> int:
        removed = 0
        for backend in self.backends:
            removed += backend.clear()
        for cache in self._semantic_caches:
            removed += cache.clear()
        return removed

    # -- event-bus wiring ----------------------------------------------------------------

    def attach(self, bus: EventBus) -> None:
        bus.subscribe("document.updated", lambda e: self.document_updated(e.payload.get("document_id", "")))
        bus.subscribe("policy.changed", lambda e: self.policy_changed())
        bus.subscribe("prompt.version_changed", lambda e: self.prompt_version_changed(e.payload.get("version")))
        bus.subscribe("schema.changed", lambda e: self.output_schema_changed())
        bus.subscribe("scope.changed", lambda e: self.scope_changed())
