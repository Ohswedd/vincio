"""Memory Engine: layered, scored, decaying memory.

Layers map to scopes: L0 working memory (per-run dict), L1 session, L2/L3
episodic+semantic (user scope, separated by type/age), L4 tenant/org, L5
knowledge graph (MemoryGraph over all items).

Retrieval scoring::

    MemoryValue = relevance · recency · confidence · scope_match · stability
                  ─────────────────────────────────────────────────────────
                  token_cost + privacy_risk + staleness_penalty
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..context.scoring import lexical_similarity, near_duplicate_score
from ..core.errors import MemoryPolicyError
from ..core.tokens import count_tokens
from ..core.types import MemoryItem, MemoryScope, MemoryType, PrivacyClass
from ..core.utils import utcnow
from .graph import MemoryGraph
from .policies import MemoryCandidate, MemoryWritePolicy, decayed_confidence
from .stores import InMemoryMemoryStore, MemoryStore

__all__ = ["MemorySearchResult", "MemoryEngine"]


class MemorySearchResult(BaseModel):
    item: MemoryItem
    score: float
    components: dict[str, float] = Field(default_factory=dict)


class MemoryEngine:
    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        write_policy: MemoryWritePolicy | None = None,
        decay_lambda: float = 0.01,
        min_confidence: float = 0.25,
        graph_enabled: bool = True,
    ) -> None:
        self.store = store or InMemoryMemoryStore()
        self.write_policy = write_policy or MemoryWritePolicy()
        self.decay_lambda = decay_lambda
        self.min_confidence = min_confidence
        self.graph = MemoryGraph() if graph_enabled else None
        self.working: dict[str, Any] = {}  # L0 working memory (one run)
        self.conflicts: list[dict[str, str]] = []

    # -- writes (write_fact / policy pipeline) ----------------------

    def write_fact(
        self,
        content: str,
        *,
        scope: MemoryScope | str = MemoryScope.USER,
        owner_id: str | None = None,
        type: MemoryType | str = MemoryType.FACT,
        confidence: float = 0.8,
        source_trace_id: str | None = None,
        privacy_class: PrivacyClass | str = PrivacyClass.INTERNAL,
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        """Direct, policy-checked write of a single memory."""
        candidate = MemoryCandidate(
            content=content,
            type=MemoryType(type),
            scope=MemoryScope(scope),
            confidence=confidence,
            entities=entities or [],
            privacy_class=PrivacyClass(privacy_class),
            metadata=metadata or {},
        )
        item = self._admit_and_store(candidate, owner_id=owner_id, source_trace_id=source_trace_id)
        if item is None:
            raise MemoryPolicyError(
                f"memory write rejected: {candidate.metadata.get('rejected_reason', 'policy')}",
                details={"content": content[:120]},
            )
        return item

    def _admit_and_store(
        self,
        candidate: MemoryCandidate,
        *,
        owner_id: str | None,
        source_trace_id: str | None,
    ) -> MemoryItem | None:
        ok, reason = self.write_policy.admit(candidate)
        if not ok:
            candidate.metadata["rejected_reason"] = reason
            return None
        # Contradiction / duplicate handling against existing memories.
        existing_items = self.store.all_items(scope=candidate.scope, owner_id=owner_id)
        superseded: str | None = None
        for existing in existing_items:
            if near_duplicate_score(candidate.content, existing.content) >= 0.95:
                # Restatement: confirm the existing memory instead of duplicating.
                existing.confirmations += 1
                existing.usage_count += 1
                existing.updated_at = utcnow()
                self.store.put(existing)
                return existing
            resolution = self.write_policy.resolve_conflict(candidate, existing)
            if resolution == "supersede":
                existing.status = "archived"
                self.store.put(existing)
                superseded = existing.id
            elif resolution == "conflict":
                self.conflicts.append(
                    {"new": candidate.content[:120], "existing_id": existing.id, "status": "needs_confirmation"}
                )
                candidate.metadata["conflict_with"] = existing.id
        item = MemoryItem(
            scope=candidate.scope,
            type=candidate.type,
            content=candidate.content,
            owner_id=owner_id,
            confidence=candidate.confidence,
            source_trace_id=source_trace_id,
            privacy_class=candidate.privacy_class,
            status="active",
            entities=candidate.entities,
            supersedes=superseded,
            metadata=candidate.metadata,
        )
        self.store.put(item)
        if self.graph is not None:
            self.graph.add_memory(item)
        return item

    async def ingest(
        self,
        text: str,
        *,
        scope: MemoryScope = MemoryScope.USER,
        owner_id: str | None = None,
        source_trace_id: str | None = None,
    ) -> list[MemoryItem]:
        """Extract-and-write pipeline over free text."""
        written: list[MemoryItem] = []
        for candidate in await self.write_policy.extract(text, scope=scope):
            item = self._admit_and_store(candidate, owner_id=owner_id, source_trace_id=source_trace_id)
            if item is not None:
                written.append(item)
        return written

    # -- retrieval ----------------------------------------------------

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        session_id: str | None = None,
        task_entities: list[str] | None = None,
        top_k: int = 8,
        max_privacy: PrivacyClass = PrivacyClass.PII,
    ) -> list[MemorySearchResult]:
        privacy_order = [
            PrivacyClass.PUBLIC,
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL,
            PrivacyClass.PII,
            PrivacyClass.SENSITIVE,
        ]
        max_privacy_rank = privacy_order.index(max_privacy)

        results: list[MemorySearchResult] = []
        for item in self.store.all_items():
            # Scope/owner match (scope match + privacy policy).
            scope_match = self._scope_match(item, user_id=user_id, tenant_id=tenant_id, session_id=session_id)
            if scope_match == 0.0:
                continue
            if privacy_order.index(item.privacy_class) > max_privacy_rank:
                continue
            confidence = decayed_confidence(item, decay_lambda=self.decay_lambda)
            if confidence < self.min_confidence:
                continue
            relevance = lexical_similarity(item.content, query)
            if task_entities:
                entity_hits = sum(
                    1 for entity in task_entities if entity.lower() in item.content.lower()
                )
                relevance = min(1.0, relevance + 0.25 * entity_hits)
            if relevance <= 0.0 and item.type not in (MemoryType.PREFERENCE, MemoryType.GOAL):
                continue
            recency = self._recency(item)
            stability = float(item.metadata.get("stability", 0.7))
            token_cost = count_tokens(item.content) / 200.0
            privacy_risk = {
                PrivacyClass.PUBLIC: 0.0,
                PrivacyClass.INTERNAL: 0.02,
                PrivacyClass.CONFIDENTIAL: 0.1,
                PrivacyClass.PII: 0.25,
                PrivacyClass.SENSITIVE: 0.5,
            }[item.privacy_class]
            staleness_penalty = max(0.0, 0.5 - recency) * 0.4
            # Preferences/goals are useful even with low query overlap.
            base_relevance = relevance if relevance > 0 else 0.15
            numerator = base_relevance * recency * confidence * scope_match * stability
            denominator = token_cost + privacy_risk + staleness_penalty + 0.05
            score = numerator / denominator
            results.append(
                MemorySearchResult(
                    item=item,
                    score=round(score, 6),
                    components={
                        "relevance": round(relevance, 4),
                        "recency": round(recency, 4),
                        "confidence": round(confidence, 4),
                        "scope_match": scope_match,
                        "stability": stability,
                        "token_cost": round(token_cost, 4),
                        "privacy_risk": privacy_risk,
                    },
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        selected = results[:top_k]
        for result in selected:
            result.item.usage_count += 1
            self.store.put(result.item)
        return selected

    @staticmethod
    def _scope_match(
        item: MemoryItem, *, user_id: str | None, tenant_id: str | None, session_id: str | None
    ) -> float:
        if item.scope == MemoryScope.GLOBAL:
            return 0.6
        if item.scope == MemoryScope.SESSION:
            if session_id and item.owner_id == session_id:
                return 1.0
            return 0.0
        if item.scope == MemoryScope.USER:
            if item.owner_id is None or (user_id and item.owner_id == user_id):
                return 1.0
            return 0.0
        if item.scope in (MemoryScope.TENANT, MemoryScope.ORGANIZATION):
            if item.owner_id is None or (tenant_id and item.owner_id == tenant_id):
                return 0.9
            return 0.0
        return 0.0

    @staticmethod
    def _recency(item: MemoryItem) -> float:
        updated = item.updated_at
        if updated.tzinfo is None:
            from datetime import UTC

            updated = updated.replace(tzinfo=UTC)
        age_days = max(0.0, (utcnow() - updated).total_seconds() / 86_400)
        return 0.5 ** (age_days / 60.0)

    # -- lifecycle -------------------------------------------------------------

    def confirm(self, memory_id: str) -> MemoryItem:
        item = self.store.get(memory_id)
        if item is None:
            raise MemoryPolicyError(f"memory not found: {memory_id}")
        item.confirmations += 1
        item.confidence = min(1.0, item.confidence + 0.1)
        item.status = "active"
        item.updated_at = utcnow()
        self.store.put(item)
        return item

    def delete(self, memory_id: str) -> bool:
        item = self.store.get(memory_id)
        if item is None:
            return False
        item.status = "deleted"
        self.store.put(item)
        return self.store.delete(memory_id)

    def decay_pass(self) -> dict[str, int]:
        """Transition decayed/expired items (candidate→…→archived)."""
        now = utcnow()
        decayed = archived = expired = 0
        for item in self.store.all_items(statuses=("active", "validated", "decayed")):
            if item.expires_at is not None:
                expires = item.expires_at
                if expires.tzinfo is None:
                    from datetime import UTC

                    expires = expires.replace(tzinfo=UTC)
                if expires <= now:
                    item.status = "archived"
                    self.store.put(item)
                    expired += 1
                    continue
            confidence = decayed_confidence(item, decay_lambda=self.decay_lambda)
            if confidence < self.min_confidence / 2:
                item.status = "archived"
                archived += 1
            elif confidence < self.min_confidence and item.status != "decayed":
                item.status = "decayed"
                decayed += 1
            else:
                continue
            self.store.put(item)
        return {"decayed": decayed, "archived": archived, "expired": expired}

    # -- stats / eval support ----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        items = self.store.all_items(statuses=())
        by_status: dict[str, int] = {}
        by_scope: dict[str, int] = {}
        for item in items:
            by_status[item.status] = by_status.get(item.status, 0) + 1
            by_scope[item.scope.value] = by_scope.get(item.scope.value, 0) + 1
        return {
            "total": len(items),
            "by_status": by_status,
            "by_scope": by_scope,
            "conflicts_pending": len(self.conflicts),
        }
