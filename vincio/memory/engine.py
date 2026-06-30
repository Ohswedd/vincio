"""Memory Engine: layered, scored, decaying memory.

Layers map to scopes: L0 working memory (per-run dict), L1 session, L2/L3
episodic+semantic (user scope, separated by type/age), L4 tenant/org, L5
knowledge graph (MemoryGraph over all items). Agent scope gives every agent
durable memory of its own alongside the user's.

Retrieval is hybrid: lexical and vector relevance fuse with graph adjacency
in one scored query. Scoring::

    MemoryValue = relevance · recency · confidence · scope_match · stability · status
                  ─────────────────────────────────────────────────────────────────
                  token_cost + privacy_risk + staleness_penalty
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..context.scoring import lexical_similarity, near_duplicate_score
from ..core.errors import MemoryPolicyError
from ..core.tokens import count_tokens
from ..core.types import EvidenceItem, MemoryItem, MemoryScope, MemoryType, PrivacyClass, ToolResult
from ..core.utils import utcnow
from ..providers.base import run_sync
from ..retrieval.embeddings import Embedder, cosine
from .facts import GroundedFact
from .graph import MemoryGraph
from .policies import (
    MemoryCandidate,
    MemoryWritePolicy,
    classify_memory_type,
    decayed_confidence,
    importance_score,
)
from .stores import InMemoryMemoryStore, MemoryStore

if TYPE_CHECKING:
    from ..governance.consent import ConsentLedger
    from ..security.audit import AuditLog
    from .consolidation import ConsolidationReport

__all__ = ["MemorySearchResult", "ScopedMemory", "MemoryEngine"]

_STATUS_WEIGHT = {"active": 1.0, "validated": 0.95, "candidate": 0.7}
_SEARCH_STATUSES = ("active", "validated", "candidate")


class MemorySearchResult(BaseModel):
    item: MemoryItem
    score: float
    components: dict[str, float] = Field(default_factory=dict)


class ScopedMemory:
    """Mem0-style handle bound to one owner: ``engine.for_user("u1")``.

    Every call delegates to the engine, so writes still pass the write
    policy and reads still pay the full scoring pipeline."""

    def __init__(self, engine: MemoryEngine, *, scope: MemoryScope, owner_id: str) -> None:
        self.engine = engine
        self.scope = scope
        self.owner_id = owner_id

    def _owner_kwargs(self) -> dict[str, Any]:
        key = {
            MemoryScope.USER: "user_id",
            MemoryScope.AGENT: "agent_id",
            MemoryScope.SESSION: "session_id",
            MemoryScope.TEAM: "team_id",
            MemoryScope.TENANT: "tenant_id",
            MemoryScope.ORGANIZATION: "tenant_id",
        }[self.scope]
        return {key: self.owner_id}

    def remember(self, content: str, **kwargs: Any) -> MemoryItem:
        kwargs.setdefault("scope", self.scope)
        return self.engine.remember(content, **self._owner_kwargs(), **kwargs)

    def recall(self, query: str, *, top_k: int = 5) -> list[MemoryItem]:
        return self.engine.recall(query, top_k=top_k, **self._owner_kwargs())

    async def arecall(self, query: str, *, top_k: int = 5) -> list[MemoryItem]:
        return await self.engine.arecall(query, top_k=top_k, **self._owner_kwargs())

    def forget(self, memory_id: str, *, reason: str = "user_request") -> bool:
        return self.engine.forget(memory_id, reason=reason)

    def items(self) -> list[MemoryItem]:
        return [
            item
            for item in self.engine.store.all_items(scope=self.scope, statuses=_SEARCH_STATUSES)
            if item.owner_id == self.owner_id
        ]

    def export(self) -> list[dict[str, Any]]:
        return self.engine.export_owner_data(self.owner_id, scope=self.scope)


class MemoryEngine:
    """Layered, guarded, decaying long-term memory with hybrid recall.

    Writes pass a configurable policy (``guarded`` / ``open`` / ``off``); recall
    fuses lexical, vector, and graph signals and honors scope, ACLs, consent,
    and bi-temporal as-of queries. Supports decay/TTL, conflict-resolving
    ``correct()``, consolidation with provenance, and audited GDPR hygiene.
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        write_policy: MemoryWritePolicy | None = None,
        decay_lambda: float = 0.01,
        min_confidence: float = 0.25,
        graph_enabled: bool = True,
        embedder: Embedder | None = None,
        vector_weight: float = 0.5,
        retention_weight: float = 0.5,
        ttl_days: Mapping[str, float] | None = None,
        audit: AuditLog | None = None,
        consent_ledger: ConsentLedger | None = None,
        privacy_accountant: Any | None = None,
        privacy_mechanism: Any | None = None,
    ) -> None:
        self.store = store or InMemoryMemoryStore()
        self.write_policy = write_policy or MemoryWritePolicy()
        self.decay_lambda = decay_lambda
        self.min_confidence = min_confidence
        self.graph = MemoryGraph() if graph_enabled else None
        self.embedder = embedder
        self.vector_weight = max(0.0, min(1.0, vector_weight))
        self.retention_weight = max(0.0, min(1.0, retention_weight))
        self.ttl_days = dict(ttl_days or {})
        self.audit = audit
        # Optional consent ledger: when set, recall drops any memory whose
        # ``purpose`` no longer has active consent for its subject (``owner_id``).
        self.consent_ledger = consent_ledger
        # Optional differential-privacy accountant: when set, consolidating a
        # subject's episodes charges the subject's privacy budget and refuses
        # (or down-weights) a consolidation that would exceed it.
        self.privacy_accountant = privacy_accountant
        self.privacy_mechanism = privacy_mechanism
        self.working: dict[str, Any] = {}  # L0 working memory (one run)
        self.conflicts: list[dict[str, str]] = []
        self._embedding_cache: dict[str, list[float]] = {}

    def _record(self, action: str, **fields: Any) -> None:
        if self.audit is not None:
            self.audit.record(action, **fields)

    # -- personalization API (scoped handles + remember/recall) -----

    def for_user(self, user_id: str) -> ScopedMemory:
        return ScopedMemory(self, scope=MemoryScope.USER, owner_id=user_id)

    def for_agent(self, agent_id: str) -> ScopedMemory:
        return ScopedMemory(self, scope=MemoryScope.AGENT, owner_id=agent_id)

    def for_session(self, session_id: str) -> ScopedMemory:
        return ScopedMemory(self, scope=MemoryScope.SESSION, owner_id=session_id)

    def for_team(self, team_id: str) -> ScopedMemory:
        """Team-shared memory: one owner is the team; per-memory ACLs gate
        which members may recall an item."""
        return ScopedMemory(self, scope=MemoryScope.TEAM, owner_id=team_id)

    def remember(
        self,
        content: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        team_id: str | None = None,
        tenant_id: str | None = None,
        scope: MemoryScope | str | None = None,
        type: MemoryType | str | None = None,
        confidence: float = 0.8,
        entities: list[str] | None = None,
        privacy_class: PrivacyClass | str = PrivacyClass.INTERNAL,
        source_trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_days: float | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        acl: list[str] | None = None,
        purpose: str | None = None,
        consent_id: str | None = None,
    ) -> MemoryItem:
        """Ergonomic write: infers scope from the most specific owner id
        given (session > agent > team > user > tenant) and classifies the memory
        type when not stated. Still policy-checked end to end. Bi-temporal
        validity (``valid_from`` / ``valid_to``), a per-memory ``acl``, and a
        GDPR ``purpose`` / ``consent_id`` ride through unchanged."""
        inferred: list[tuple[MemoryScope, str | None]] = [
            (MemoryScope.SESSION, session_id),
            (MemoryScope.AGENT, agent_id),
            (MemoryScope.TEAM, team_id),
            (MemoryScope.USER, user_id),
            (MemoryScope.TENANT, tenant_id),
        ]
        if scope is not None:
            scope = MemoryScope(scope)
            owner_id = (
                dict(inferred).get(scope)
                or session_id
                or agent_id
                or team_id
                or user_id
                or tenant_id
            )
        else:
            scope, owner_id = next(
                ((s, o) for s, o in inferred if o is not None), (MemoryScope.GLOBAL, None)
            )
        memory_type = MemoryType(type) if type is not None else classify_memory_type(content)
        meta = dict(metadata or {})
        meta.setdefault("tier", "episodic" if scope == MemoryScope.SESSION else "semantic")
        return self.write_fact(
            content,
            scope=scope,
            owner_id=owner_id,
            type=memory_type,
            confidence=confidence,
            source_trace_id=source_trace_id,
            privacy_class=privacy_class,
            entities=entities,
            metadata=meta,
            ttl_days=ttl_days,
            valid_from=valid_from,
            valid_to=valid_to,
            acl=acl,
            purpose=purpose,
            consent_id=consent_id,
        )

    async def arecall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        team_id: str | None = None,
        tenant_id: str | None = None,
        session_id: str | None = None,
        top_k: int = 5,
        task_entities: list[str] | None = None,
        as_of: datetime | None = None,
        reader: str | None = None,
    ) -> list[MemoryItem]:
        results = await self.asearch(
            query,
            user_id=user_id,
            agent_id=agent_id,
            team_id=team_id,
            tenant_id=tenant_id,
            session_id=session_id,
            top_k=top_k,
            task_entities=task_entities,
            as_of=as_of,
            reader=reader,
        )
        return [r.item for r in results]

    def recall(self, query: str, **kwargs: Any) -> list[MemoryItem]:
        """Sync wrapper over :meth:`arecall`."""
        return run_sync(self.arecall(query, **kwargs))

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
        ttl_days: float | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        acl: list[str] | None = None,
        purpose: str | None = None,
        consent_id: str | None = None,
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
        item = self._admit_and_store(
            candidate,
            owner_id=owner_id,
            source_trace_id=source_trace_id,
            ttl_days=ttl_days,
            valid_from=valid_from,
            valid_to=valid_to,
            acl=acl,
            purpose=purpose,
            consent_id=consent_id,
        )
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
        status: str = "active",
        ttl_days: float | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        acl: list[str] | None = None,
        purpose: str | None = None,
        consent_id: str | None = None,
    ) -> MemoryItem | None:
        ok, reason = self.write_policy.admit(candidate)
        if not ok:
            candidate.metadata["rejected_reason"] = reason
            return None
        now = utcnow()
        # Contradiction / duplicate handling against existing memories.
        existing_items = self.store.all_items(scope=candidate.scope, owner_id=owner_id)
        superseded: str | None = None
        for existing in existing_items:
            if near_duplicate_score(candidate.content, existing.content) >= 0.95:
                # Restatement: confirm the existing memory instead of duplicating.
                existing.confirmations += 1
                existing.usage_count += 1
                existing.status = "active"
                existing.updated_at = now
                self.store.put(existing)
                return existing
            resolution = self.write_policy.resolve_conflict(candidate, existing)
            if resolution == "supersede":
                existing.status = "archived"
                # Bi-temporal correction: close the old fact's valid
                # interval at the moment the new one takes effect, so as-of
                # recall before that moment still returns the prior value.
                if existing.valid_to is None:
                    existing.valid_to = valid_from or now
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
            status=status,  # type: ignore[arg-type]
            entities=candidate.entities,
            supersedes=superseded,
            valid_from=valid_from,
            valid_to=valid_to,
            acl=list(acl or []),
            purpose=purpose,
            consent_id=consent_id,
            metadata=candidate.metadata,
        )
        ttl = ttl_days if ttl_days is not None else self.ttl_days.get(item.scope.value)
        if item.expires_at is None and ttl:
            item.expires_at = now + timedelta(days=float(ttl))
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

    def write_back(
        self,
        *,
        evidence: list[EvidenceItem] | None = None,
        tool_results: list[ToolResult] | None = None,
        facts: list[GroundedFact] | None = None,
        owner_id: str | None = None,
        session_id: str | None = None,
        source_trace_id: str | None = None,
    ) -> list[MemoryItem]:
        """Write confirmed evidence, tool results, and grounded run facts
        back as *candidate* memories with provenance. Candidates carry a
        status penalty in retrieval until confirmed, and every recall
        utility-scores them against the task before they enter a packet.
        ``facts`` are evidence-supported output claims (auto-memory):
        their confidence scales with measured support and they go through
        the same guarded admission as every other write."""
        scope = MemoryScope.SESSION if session_id else MemoryScope.USER
        write_owner = session_id or owner_id
        written: list[MemoryItem] = []

        def _write(content: str, meta: dict[str, Any], confidence: float) -> None:
            candidate = MemoryCandidate(
                content=content.strip()[:400],
                type=MemoryType.FACT,
                scope=scope,
                confidence=confidence,
                metadata={"tier": "episodic", **meta},
            )
            item = self._admit_and_store(
                candidate, owner_id=write_owner, source_trace_id=source_trace_id, status="candidate"
            )
            if item is not None:
                written.append(item)

        for ev in evidence or []:
            if not (ev.text or "").strip():
                continue
            _write(
                ev.text or "",
                {"origin": "evidence", "source_id": ev.source_id, "evidence_id": ev.id},
                confidence=min(0.7, 0.4 + 0.3 * ev.provenance),
            )
        for tool in tool_results or []:
            if tool.status != "ok" or tool.output is None:
                continue
            _write(
                f"Tool {tool.tool_name} returned: {str(tool.output)}",
                {"origin": "tool", "tool_name": tool.tool_name, "call_id": tool.call_id},
                confidence=0.55,
            )
        for fact in facts or []:
            if not fact.content.strip():
                continue
            _write(
                fact.content,
                {
                    "origin": "run_fact",
                    "support": round(fact.support, 4),
                    "evidence_ids": list(fact.evidence_ids[:4]),
                },
                confidence=min(0.8, 0.35 + 0.45 * fact.support),
            )
        return written

    # -- retrieval ----------------------------------------------------

    def _vector_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    async def _vectors_for(self, texts: list[str]) -> list[list[float]]:
        assert self.embedder is not None
        keys = [self._vector_key(text) for text in texts]
        missing = [(key, text) for key, text in zip(keys, texts, strict=True) if key not in self._embedding_cache]
        if missing:
            vectors = await self.embedder.embed([text for _key, text in missing])
            for (key, _text), vector in zip(missing, vectors, strict=True):
                self._embedding_cache[key] = vector
        return [self._embedding_cache[key] for key in keys]

    async def asearch(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        team_id: str | None = None,
        tenant_id: str | None = None,
        session_id: str | None = None,
        task_entities: list[str] | None = None,
        top_k: int = 8,
        max_privacy: PrivacyClass = PrivacyClass.PII,
        as_of: datetime | None = None,
        reader: str | None = None,
    ) -> list[MemorySearchResult]:
        """Hybrid recall: lexical + vector relevance fused with graph
        adjacency, in one scored, scope- and privacy-filtered query.

        ``as_of`` makes recall **bi-temporal** — it returns the memories
        that were *valid* at that moment, including ones later superseded
        (their ``valid_to`` was closed by the correction); ``reader`` enforces
        per-memory ACLs so team-shared memory only surfaces to permitted
        members; and a configured :attr:`consent_ledger` drops any item whose
        ``purpose`` lost consent."""
        privacy_order = [
            PrivacyClass.PUBLIC,
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL,
            PrivacyClass.PII,
            PrivacyClass.SENSITIVE,
        ]
        max_privacy_rank = privacy_order.index(max_privacy)
        now = utcnow()
        moment = as_of or now
        # As-of recall must see superseded/archived items that were valid then,
        # so it widens the status filter; current recall keeps the live set.
        statuses = () if as_of is not None else _SEARCH_STATUSES

        # Pass 1: scope/privacy/lifetime/validity/ACL/consent filters.
        eligible: list[tuple[MemoryItem, float, float]] = []
        for item in self.store.all_items(statuses=statuses):
            scope_match = self._scope_match(
                item,
                user_id=user_id,
                agent_id=agent_id,
                team_id=team_id,
                tenant_id=tenant_id,
                session_id=session_id,
            )
            if scope_match == 0.0:
                continue
            if privacy_order.index(item.privacy_class) > max_privacy_rank:
                continue
            if self._expired(item, now):
                continue
            # Bi-temporal validity: items carrying a valid interval must contain
            # the recall moment; non-bi-temporal items (no interval set) always
            # pass, so this is fully backward compatible.
            if (item.valid_from is not None or item.valid_to is not None) and not item.valid_at(
                moment
            ):
                continue
            # Per-memory ACL: a populated ACL admits only listed readers.
            if not item.readable_by(reader):
                continue
            # Consent / purpose: drop a memory whose purpose lost consent.
            if (
                self.consent_ledger is not None
                and item.purpose
                and item.owner_id
                and not self.consent_ledger.allows(item.owner_id, item.purpose)
            ):
                continue
            confidence = decayed_confidence(item, decay_lambda=self.decay_lambda)
            if confidence < self.min_confidence:
                continue
            eligible.append((item, scope_match, confidence))

        # Graph adjacency: memories linked to the task's entities (and the
        # supersedes chain) get a relevance boost in the same query.
        graph_hits: set[str] = set()
        if self.graph is not None and task_entities:
            for entity in task_entities:
                graph_hits.update(self.graph.memories_about(entity))

        # Vector relevance for all eligible items in one batched call.
        query_vector: list[float] | None = None
        item_vectors: dict[str, list[float]] = {}
        if self.embedder is not None and eligible:
            vectors = await self._vectors_for([query] + [item.content for item, _s, _c in eligible])
            query_vector = vectors[0]
            item_vectors = {item.id: vec for (item, _s, _c), vec in zip(eligible, vectors[1:], strict=True)}

        results: list[MemorySearchResult] = []
        for item, scope_match, confidence in eligible:
            lexical = lexical_similarity(item.content, query)
            vector = 0.0
            if query_vector is not None:
                vector = max(0.0, cosine(query_vector, item_vectors[item.id]))
                relevance = (1.0 - self.vector_weight) * lexical + self.vector_weight * vector
            else:
                relevance = lexical
            if task_entities:
                entity_hits = sum(
                    1 for entity in task_entities if entity.lower() in item.content.lower()
                )
                relevance = min(1.0, relevance + 0.25 * entity_hits)
            graph_boost = 0.2 if item.id in graph_hits else 0.0
            relevance = min(1.0, relevance + graph_boost)
            if relevance <= 0.03 and not graph_boost and item.type not in (
                MemoryType.PREFERENCE,
                MemoryType.GOAL,
            ):
                continue
            recency = self._recency(item)
            stability = float(item.metadata.get("stability", 0.7))
            status_weight = _STATUS_WEIGHT.get(item.status, 0.7)
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
            numerator = base_relevance * recency * confidence * scope_match * stability * status_weight
            denominator = token_cost + privacy_risk + staleness_penalty + 0.05
            score = numerator / denominator
            results.append(
                MemorySearchResult(
                    item=item,
                    score=round(score, 6),
                    components={
                        "relevance": round(relevance, 4),
                        "lexical": round(lexical, 4),
                        "vector": round(vector, 4),
                        "graph": graph_boost,
                        "recency": round(recency, 4),
                        "confidence": round(confidence, 4),
                        "scope_match": scope_match,
                        "stability": stability,
                        "status": status_weight,
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

    def search(self, query: str, **kwargs: Any) -> list[MemorySearchResult]:
        """Sync wrapper over :meth:`asearch`."""
        return run_sync(self.asearch(query, **kwargs))

    @staticmethod
    def _scope_match(
        item: MemoryItem,
        *,
        user_id: str | None,
        agent_id: str | None = None,
        team_id: str | None = None,
        tenant_id: str | None,
        session_id: str | None,
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
        if item.scope == MemoryScope.AGENT:
            if agent_id and item.owner_id == agent_id:
                return 1.0
            return 0.0
        if item.scope == MemoryScope.TEAM:
            if team_id and item.owner_id == team_id:
                return 1.0
            return 0.0
        if item.scope in (MemoryScope.TENANT, MemoryScope.ORGANIZATION):
            if item.owner_id is None or (tenant_id and item.owner_id == tenant_id):
                return 0.9
            return 0.0
        return 0.0

    @staticmethod
    def _expired(item: MemoryItem, now: Any) -> bool:
        if item.expires_at is None:
            return False
        expires = item.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        return expires <= now

    @staticmethod
    def _recency(item: MemoryItem) -> float:
        updated = item.updated_at
        if updated.tzinfo is None:
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

    def edit(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        type: MemoryType | str | None = None,
        confidence: float | None = None,
        entities: list[str] | None = None,
    ) -> MemoryItem:
        """User-driven correction. New content re-passes the write policy;
        the change is recorded in the audit log."""
        item = self.store.get(memory_id)
        if item is None:
            raise MemoryPolicyError(f"memory not found: {memory_id}")
        previous = item.content
        if content is not None:
            candidate = MemoryCandidate(
                content=content,
                type=item.type,
                scope=item.scope,
                confidence=confidence if confidence is not None else item.confidence,
                privacy_class=item.privacy_class,
                metadata=dict(item.metadata),
            )
            ok, reason = self.write_policy.admit(candidate)
            if not ok:
                raise MemoryPolicyError(f"memory edit rejected: {reason}", details={"id": memory_id})
            item.content = content
            item.privacy_class = candidate.privacy_class
        if type is not None:
            item.type = MemoryType(type)
        if confidence is not None:
            item.confidence = confidence
        if entities is not None:
            item.entities = entities
        item.metadata["edited"] = True
        item.updated_at = utcnow()
        self.store.put(item)
        if self.graph is not None and content is not None:
            self.graph.add_memory(item)
        self._record(
            "memory_edit",
            user_id=item.owner_id if item.scope == MemoryScope.USER else None,
            resource=item.id,
            details={"previous": previous[:120], "current": item.content[:120]},
        )
        return item

    def correct(
        self,
        memory_id: str,
        new_content: str,
        *,
        valid_from: datetime | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        """Bi-temporal correction: close the existing memory's valid
        interval and open a new one carrying the corrected content.

        Unlike :meth:`edit` (which mutates the record in place, losing history),
        ``correct`` preserves the old value so an as-of recall before the
        correction still returns what was believed true then. The old item is
        archived with its ``valid_to`` set; the new item inherits the scope,
        owner, ACL, and purpose, with ``valid_from`` at the correction moment."""
        item = self.store.get(memory_id)
        if item is None:
            raise MemoryPolicyError(f"memory not found: {memory_id}")
        moment = valid_from or utcnow()
        if item.valid_to is None:
            item.valid_to = moment
        item.status = "archived"
        item.updated_at = utcnow()
        self.store.put(item)
        new_item = self.write_fact(
            new_content,
            scope=item.scope,
            owner_id=item.owner_id,
            type=item.type,
            confidence=confidence if confidence is not None else item.confidence,
            privacy_class=item.privacy_class,
            entities=item.entities,
            metadata={**item.metadata, "corrected_from": item.id},
            valid_from=moment,
            acl=list(item.acl),
            purpose=item.purpose,
            consent_id=item.consent_id,
        )
        new_item.supersedes = item.id
        self.store.put(new_item)
        self._record(
            "memory_correct",
            user_id=item.owner_id if item.scope == MemoryScope.USER else None,
            resource=item.id,
            details={"new_id": new_item.id, "valid_to": moment.isoformat()},
        )
        return new_item

    def delete(self, memory_id: str) -> bool:
        item = self.store.get(memory_id)
        if item is None:
            return False
        item.status = "deleted"
        self.store.put(item)
        removed = self.store.delete(memory_id)
        self._record(
            "memory_delete",
            user_id=item.owner_id if item.scope == MemoryScope.USER else None,
            resource=memory_id,
            details={"scope": item.scope.value},
        )
        return removed

    def forget(self, memory_id: str, *, reason: str = "user_request") -> bool:
        """Explicit user-driven deletion; the reason lands in the audit log."""
        item = self.store.get(memory_id)
        if item is None:
            return False
        item.status = "deleted"
        self.store.put(item)
        removed = self.store.delete(memory_id)
        self._record(
            "memory_delete",
            user_id=item.owner_id if item.scope == MemoryScope.USER else None,
            resource=memory_id,
            details={"scope": item.scope.value, "reason": reason},
        )
        return removed

    def export_owner_data(
        self, owner_id: str, *, scope: MemoryScope | None = None
    ) -> list[dict[str, Any]]:
        """GDPR-style data access/portability: every stored memory for an
        owner, all statuses, as plain JSON-able dicts. Audited."""
        items = [
            item
            for item in self.store.all_items(scope=scope, statuses=())
            if item.owner_id == owner_id
        ]
        self._record(
            "memory_export",
            user_id=owner_id,
            details={"count": len(items), "scope": scope.value if scope else "all"},
        )
        return [item.model_dump(mode="json") for item in items]

    def erase_owner_data(self, owner_id: str) -> int:
        """GDPR-style right to erasure: hard-delete every memory owned by
        *owner_id* (all scopes and statuses) and rebuild the graph. Audited."""
        items = [
            item for item in self.store.all_items(statuses=()) if item.owner_id == owner_id
        ]
        for item in items:
            self.store.delete(item.id)
        if self.graph is not None:
            self.graph = MemoryGraph()
            for item in self.store.all_items(statuses=_SEARCH_STATUSES):
                self.graph.add_memory(item)
        self._record("memory_erase", user_id=owner_id, details={"count": len(items)})
        return len(items)

    def decay_pass(self) -> dict[str, int]:
        """Transition decayed/expired items (candidate→…→archived).
        Retention is importance-weighted: heavily used, confirmed, stable
        memories tolerate lower decayed confidence before archival."""
        now = utcnow()
        decayed = archived = expired = 0
        for item in self.store.all_items(statuses=("active", "validated", "candidate", "decayed")):
            if self._expired(item, now):
                item.status = "archived"
                self.store.put(item)
                expired += 1
                continue
            confidence = decayed_confidence(item, decay_lambda=self.decay_lambda)
            retention = 1.0 - self.retention_weight * importance_score(item)
            if confidence < (self.min_confidence / 2) * retention:
                item.status = "archived"
                archived += 1
            elif confidence < self.min_confidence * retention and item.status != "decayed":
                item.status = "decayed"
                decayed += 1
            else:
                continue
            self.store.put(item)
        return {"decayed": decayed, "archived": archived, "expired": expired}

    # -- consolidation -----------------------------------------------------------

    async def consolidate(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        summarizer: Any | None = None,
    ) -> ConsolidationReport:
        """Episodic→semantic consolidation for one session (see
        :class:`~vincio.memory.consolidation.MemoryConsolidator`)."""
        from .consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(self, summarizer=summarizer)
        return await consolidator.consolidate_session(
            session_id, user_id=user_id, agent_id=agent_id
        )

    async def promote_aged_episodes(
        self,
        *,
        min_age_days: float = 7.0,
        user_id: str | None = None,
        summarizer: Any | None = None,
    ) -> list[ConsolidationReport]:
        """Consolidate every session whose episodic memories have all aged past
        *min_age_days* — the periodic background tier transition (see
        :meth:`~vincio.memory.consolidation.MemoryConsolidator.promote_aged_episodes`)."""
        from .consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(self, summarizer=summarizer)
        return await consolidator.promote_aged_episodes(
            min_age_days=min_age_days, user_id=user_id
        )

    # -- stats / eval support ----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        items = self.store.all_items(statuses=())
        by_status: dict[str, int] = {}
        by_scope: dict[str, int] = {}
        by_tier: dict[str, int] = {}
        for item in items:
            by_status[item.status] = by_status.get(item.status, 0) + 1
            by_scope[item.scope.value] = by_scope.get(item.scope.value, 0) + 1
            tier = str(item.metadata.get("tier", "semantic"))
            by_tier[tier] = by_tier.get(tier, 0) + 1
        return {
            "total": len(items),
            "by_status": by_status,
            "by_scope": by_scope,
            "by_tier": by_tier,
            "conflicts_pending": len(self.conflicts),
        }
