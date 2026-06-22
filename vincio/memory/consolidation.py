"""Consolidation tiers: episodic→semantic promotion with provenance.

Episodic memories (session-scoped observations, evidence and tool
write-backs) are summarized into a few durable semantic memories, promoted
to longer-lived scopes, and deduplicated — with full provenance retained:
promoted items record ``consolidated_from`` ids, and source items are
archived (never silently dropped) with a ``consolidated_into`` backref.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..context.scoring import near_duplicate_score
from ..core.errors import MemoryPolicyError
from ..core.types import MemoryItem, MemoryScope
from ..core.utils import utcnow
from .summarizers import SessionSummarizer

if TYPE_CHECKING:
    from .engine import MemoryEngine

__all__ = ["ConsolidationReport", "MemoryConsolidator"]


class ConsolidationReport(BaseModel):
    session_id: str | None = None
    examined: int = 0
    promoted: int = 0
    deduplicated: int = 0
    archived: int = 0
    items: list[MemoryItem] = Field(default_factory=list)
    # Differential-privacy accounting (set only when an accountant is attached).
    privacy_refused: bool = False
    privacy_epsilon: float | None = None  # cumulative ε the subject has spent


class MemoryConsolidator:
    """Runs the episodic→semantic tier transitions for a memory engine."""

    def __init__(
        self,
        engine: MemoryEngine,
        *,
        summarizer: SessionSummarizer | None = None,
        min_items: int = 2,
        dedup_threshold: float = 0.92,
    ) -> None:
        self.engine = engine
        self.summarizer = summarizer or SessionSummarizer()
        self.min_items = min_items
        self.dedup_threshold = dedup_threshold

    async def consolidate_session(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> ConsolidationReport:
        """Summarize one session's episodic memories into semantic memories
        promoted to the user (or agent) scope, then archive the episodes."""
        episodes = [
            item
            for item in self.engine.store.all_items(
                scope=MemoryScope.SESSION, statuses=("active", "validated", "candidate")
            )
            if item.owner_id == session_id
        ]
        report = ConsolidationReport(session_id=session_id, examined=len(episodes))
        if len(episodes) < self.min_items:
            return report
        episodes.sort(key=lambda item: item.created_at)
        session_text = "\n".join(item.content for item in episodes)
        if user_id is not None:
            target_scope, target_owner = MemoryScope.USER, user_id
        elif agent_id is not None:
            target_scope, target_owner = MemoryScope.AGENT, agent_id
        else:
            target_scope, target_owner = MemoryScope.SESSION, session_id

        # Differential-privacy gate: consolidating a subject's episodes into a
        # durable summary is a release of aggregated information about that
        # subject. When an accountant is attached, charge the subject's privacy
        # budget; a consolidation that would exceed it is refused (the subject's
        # data simply stays in its short-lived episodic form) and recorded.
        spend = self._charge_privacy(subject=target_owner, session_id=session_id)
        if spend is False:
            report.privacy_refused = True
            return report
        if spend is not None:
            report.privacy_epsilon = spend.cumulative_epsilon

        summaries = await self.summarizer.summarize(
            session_text, scope=target_scope, owner_id=target_owner, session_id=session_id
        )
        source_ids = [item.id for item in episodes]
        promoted: list[MemoryItem] = []
        for summary in summaries:
            metadata = {
                **summary.metadata,
                "tier": "semantic",
                "consolidated_from": source_ids,
            }
            try:
                item = self.engine.write_fact(
                    summary.content,
                    scope=target_scope,
                    owner_id=target_owner,
                    type=summary.type,
                    confidence=summary.confidence,
                    source_trace_id=summary.source_trace_id,
                    metadata=metadata,
                )
            except MemoryPolicyError:
                continue
            promoted.append(item)
        promoted_ids = [item.id for item in promoted]
        for episode in episodes:
            episode.status = "archived"
            episode.metadata["consolidated_into"] = promoted_ids
            episode.updated_at = utcnow()
            self.engine.store.put(episode)
        report.promoted = len(promoted)
        report.archived = len(episodes)
        report.items = promoted
        report.deduplicated = self.dedup(scope=target_scope, owner_id=target_owner)
        self.engine._record(
            "memory_consolidate",
            user_id=user_id,
            details={
                "session_id": session_id,
                "examined": report.examined,
                "promoted": report.promoted,
                "deduplicated": report.deduplicated,
            },
        )
        return report

    def _charge_privacy(self, *, subject: str | None, session_id: str) -> Any:
        """Charge the subject's DP budget for this consolidation.

        Returns ``None`` when no accountant is attached (proceed unaccounted),
        ``False`` when the budget refuses the consolidation (skip it), or the
        recorded :class:`~vincio.governance.privacy.PrivacySpend` on success.
        """
        accountant = getattr(self.engine, "privacy_accountant", None)
        if accountant is None or not subject:
            return None
        mechanism = getattr(self.engine, "privacy_mechanism", None)
        # A consolidation releases a deterministic, extractive summary it cannot make
        # noisier, so it must pay the *full* mechanism cost — a down-weight (which
        # assumes a more-private release) is not honestly realizable here. Gate on the
        # full cost: a clean fit proceeds; anything else (over budget, or a budget
        # that would only admit a down-weighted release) is refused.
        decision = accountant.check(subject, mechanism)
        if decision.action != "allow":
            accountant.note_refusal(decision, operation="memory_consolidation")
            return False
        return accountant.record(
            subject,
            mechanism,
            operation="memory_consolidation",
            round_id=session_id,
            details={"session_id": session_id},
        )

    def dedup(self, *, scope: MemoryScope | None = None, owner_id: str | None = None) -> int:
        """Merge near-duplicate active memories. The survivor keeps the
        higher confidence, absorbs the other's confirmations, and records
        ``merged_from`` provenance; the duplicate is archived."""
        items = [
            item
            for item in self.engine.store.all_items(scope=scope, statuses=("active", "validated"))
            if owner_id is None or item.owner_id == owner_id
        ]
        merged = 0
        for index, item in enumerate(items):
            if item.status == "archived":
                continue
            for other in items[index + 1 :]:
                if other.status == "archived" or other.owner_id != item.owner_id:
                    continue
                if near_duplicate_score(item.content, other.content) < self.dedup_threshold:
                    continue
                survivor, duplicate = (
                    (item, other) if item.confidence >= other.confidence else (other, item)
                )
                survivor.confirmations += duplicate.confirmations + 1
                survivor.usage_count += duplicate.usage_count
                survivor.entities = sorted(set(survivor.entities) | set(duplicate.entities))
                merged_from = survivor.metadata.setdefault("merged_from", [])
                merged_from.append(duplicate.id)
                survivor.updated_at = utcnow()
                duplicate.status = "archived"
                duplicate.metadata["merged_into"] = survivor.id
                self.engine.store.put(survivor)
                self.engine.store.put(duplicate)
                merged += 1
        return merged

    async def promote_aged_episodes(
        self, *, min_age_days: float = 7.0, user_id: str | None = None
    ) -> list[ConsolidationReport]:
        """Consolidate every session whose episodic memories have all aged
        past *min_age_days* — the periodic background tier transition."""
        sessions: dict[str, list[MemoryItem]] = {}
        for item in self.engine.store.all_items(
            scope=MemoryScope.SESSION, statuses=("active", "validated", "candidate")
        ):
            if item.owner_id:
                sessions.setdefault(item.owner_id, []).append(item)
        reports: list[ConsolidationReport] = []
        now = utcnow()
        for session_id, episodes in sessions.items():
            ages: list[float] = []
            for episode in episodes:
                updated = episode.updated_at
                if updated.tzinfo is None:
                    from datetime import UTC

                    updated = updated.replace(tzinfo=UTC)
                ages.append((now - updated).total_seconds() / 86_400)
            if min(ages) < min_age_days:
                continue
            owner = user_id or self._session_user(episodes)
            reports.append(await self.consolidate_session(session_id, user_id=owner))
        return reports

    @staticmethod
    def _session_user(episodes: list[MemoryItem]) -> str | None:
        for episode in episodes:
            value = episode.metadata.get("user_id")
            if value:
                return str(value)
        return None
