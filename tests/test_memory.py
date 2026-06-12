"""Memory engine unit tests (memory policies)."""

import pytest

from vincio.core.errors import MemoryPolicyError
from vincio.core.types import MemoryItem, MemoryScope
from vincio.memory import (
    MemoryEngine,
    MemoryGraph,
    SQLiteMemoryStore,
    decayed_confidence,
    detect_contradiction,
    extract_memory_candidates,
    extractive_summary,
    stability_score,
)


class TestWritePolicy:
    def test_extracts_durable_statements(self):
        candidates = extract_memory_candidates(
            "I prefer email summaries every Friday. The sky is blue today. "
            "We decided to migrate to Kubernetes next quarter."
        )
        contents = " ".join(c.content for c in candidates)
        assert "email summaries" in contents
        assert "Kubernetes" in contents
        assert "sky is blue" not in contents  # volatile + impersonal

    def test_stability_score(self):
        assert stability_score("I prefer dark mode") > stability_score(
            "I'm at the office right now"
        )

    def test_contradiction_detection(self):
        assert detect_contradiction(
            "User prefers detailed verbose answers", "User prefers concise technical answers"
        )
        assert not detect_contradiction(
            "User prefers concise answers in email", "User prefers concise answers"
        )
        assert not detect_contradiction("User works at Acme", "Cats are mammals")

    def test_secret_write_blocked(self):
        engine = MemoryEngine()
        with pytest.raises(MemoryPolicyError):
            engine.write_fact("api_key = sk-supersecret1234567890", scope="user", owner_id="u1")


class TestEngine:
    def test_restatement_confirms_instead_of_duplicating(self):
        engine = MemoryEngine()
        first = engine.write_fact("User prefers concise answers", scope="user", owner_id="u1", type="preference")
        second = engine.write_fact("User prefers concise answers", scope="user", owner_id="u1", type="preference")
        assert first.id == second.id
        assert second.confirmations == 1

    def test_contradiction_supersedes(self):
        engine = MemoryEngine()
        old = engine.write_fact(
            "User prefers concise technical answers", scope="user", owner_id="u1",
            type="preference", confidence=0.6,
        )
        new = engine.write_fact(
            "User prefers detailed verbose answers", scope="user", owner_id="u1",
            type="preference", confidence=0.95,
        )
        assert new.supersedes == old.id
        archived = engine.store.get(old.id)
        assert archived is None or archived.status == "archived"

    def test_scope_isolation(self):
        engine = MemoryEngine()
        engine.write_fact("Tenant Acme pays annually", scope="tenant", owner_id="acme")
        results_other = engine.search("annual payment", user_id="u1", tenant_id="other")
        assert not any("Acme" in r.item.content for r in results_other)
        results_acme = engine.search("annual payment", user_id="u1", tenant_id="acme")
        assert any("Acme" in r.item.content for r in results_acme)

    def test_search_scoring_components(self):
        engine = MemoryEngine()
        engine.write_fact("User prefers bullet points", scope="user", owner_id="u1", type="preference")
        results = engine.search("how should answers be formatted", user_id="u1")
        assert results
        assert set(results[0].components) >= {"relevance", "recency", "confidence", "scope_match"}

    def test_decay(self):
        from datetime import timedelta

        from vincio.core.utils import utcnow

        item = MemoryItem(content="old fact", confidence=0.5)
        item.updated_at = utcnow() - timedelta(days=365)
        assert decayed_confidence(item, decay_lambda=0.01) < 0.5 * 0.5

    def test_decay_pass_archives(self):
        from datetime import timedelta

        from vincio.core.utils import utcnow

        engine = MemoryEngine(min_confidence=0.4)
        item = engine.write_fact("User prefers dark mode", scope="user", owner_id="u1", confidence=0.41, type="preference")
        item.updated_at = utcnow() - timedelta(days=2000)
        engine.store.put(item)
        stats = engine.decay_pass()
        assert stats["archived"] + stats["decayed"] >= 1

    def test_confirm_and_delete(self):
        engine = MemoryEngine()
        item = engine.write_fact("User works at Globex", scope="user", owner_id="u1")
        confirmed = engine.confirm(item.id)
        assert confirmed.confirmations == 1
        assert engine.delete(item.id) is True
        assert engine.store.get(item.id) is None


class TestStores:
    def test_sqlite_roundtrip(self, tmp_path):
        store = SQLiteMemoryStore(tmp_path / "memory.db")
        engine = MemoryEngine(store=store)
        item = engine.write_fact("User works at Globex Corporation", scope="user", owner_id="u9")
        loaded = store.get(item.id)
        assert loaded is not None
        assert loaded.content == item.content
        assert store.all_items(scope=MemoryScope.USER, owner_id="u9")
        store.close()


class TestGraphAndSummaries:
    def test_memory_graph_projection(self):
        graph = MemoryGraph()
        item = MemoryItem(
            content="User prefers concise answers",
            owner_id="u1",
            entities=["Acme"],
        )
        graph.add_memory(item)
        assert graph.memories_about("Acme") == [item.id]
        assert graph.memories_for_owner("u1") == [item.id]

    def test_extractive_summary(self):
        text = (
            "We discussed the migration plan. The team decided to use Postgres. "
            "Lunch was great. The deadline is March 15."
        )
        summary = extractive_summary(text, max_tokens=20, focus="decisions deadline")
        assert "Postgres" in summary or "deadline" in summary.lower() or "March" in summary

    @pytest.mark.asyncio
    async def test_session_summarizer_extractive(self):
        from vincio.memory import SessionSummarizer

        items = await SessionSummarizer().summarize(
            "The team decided to adopt Vincio. Rollout starts next month.", owner_id="u1"
        )
        assert items and items[0].type.value == "summary"


# ---------------------------------------------------------------------------
# 0.4 — Memory & personalization
# ---------------------------------------------------------------------------


class TestPersonalizationAPI:
    def test_remember_infers_scope_and_type(self):
        engine = MemoryEngine()
        item = engine.remember("I prefer dark mode in every editor", user_id="u1")
        assert item.scope == MemoryScope.USER
        assert item.owner_id == "u1"
        assert item.type.value == "preference"
        assert item.metadata["tier"] == "semantic"

    def test_remember_session_scope_is_episodic(self):
        engine = MemoryEngine()
        item = engine.remember("We reviewed the Q3 contract terms", session_id="s1")
        assert item.scope == MemoryScope.SESSION
        assert item.owner_id == "s1"
        assert item.metadata["tier"] == "episodic"

    def test_scoped_handles_isolate_owners(self):
        engine = MemoryEngine()
        engine.for_user("u1").remember("User prefers concise answers")
        engine.for_agent("bot1").remember("Agent escalates billing disputes to finance")
        user_items = engine.for_user("u1").recall("answer style")
        agent_items = engine.for_agent("bot1").recall("billing disputes")
        assert any("concise" in i.content for i in user_items)
        assert not any("finance" in i.content for i in user_items)
        assert any("finance" in i.content for i in agent_items)

    def test_agent_scope_requires_matching_agent(self):
        engine = MemoryEngine()
        engine.remember("Agent always verifies invoices twice", agent_id="bot1")
        assert not engine.search("verify invoices", user_id="u9")
        results = engine.search("verify invoices", agent_id="bot1")
        assert results and results[0].item.scope == MemoryScope.AGENT

    def test_recall_returns_items(self):
        engine = MemoryEngine()
        engine.remember("User prefers weekly summaries by email", user_id="u1")
        items = engine.recall("how often to send summaries", user_id="u1")
        assert items and "weekly" in items[0].content


class TestHybridRecall:
    def _hybrid_engine(self):
        from vincio.retrieval.embeddings import LocalHashEmbedder

        return MemoryEngine(embedder=LocalHashEmbedder())

    def test_vector_recall_catches_morphological_variants(self):
        engine = self._hybrid_engine()
        engine.write_fact(
            "User prefers contracts terminated before renewal", scope="user",
            owner_id="u1", type="preference",
        )
        results = engine.search("terminate the contract", user_id="u1")
        assert results
        assert results[0].components["vector"] > 0.05

    def test_graph_boost_for_task_entities(self):
        engine = self._hybrid_engine()
        engine.write_fact(
            "User decided to consolidate Acme invoices quarterly", scope="user",
            owner_id="u1", entities=["Acme"],
        )
        boosted = engine.search("invoicing cadence", user_id="u1", task_entities=["Acme"])
        plain = engine.search("invoicing cadence", user_id="u1")
        assert boosted and boosted[0].components["graph"] > 0.0
        assert boosted[0].components["relevance"] > plain[0].components["relevance"]

    def test_embedding_cache_is_content_addressed(self):
        engine = self._hybrid_engine()
        engine.write_fact("User prefers concise answers", scope="user", owner_id="u1", type="preference")
        engine.search("answer style", user_id="u1")
        cached = len(engine._embedding_cache)
        engine.search("answer style", user_id="u1")
        assert len(engine._embedding_cache) == cached


class TestConsolidation:
    @pytest.mark.asyncio
    async def test_session_consolidation_promotes_with_provenance(self):
        engine = MemoryEngine()
        engine.remember("We agreed to migrate the billing stack to Postgres", session_id="s1")
        engine.remember("The rollout deadline is March 15 next quarter", session_id="s1")
        report = await engine.consolidate("s1", user_id="u1")
        assert report.examined == 2
        assert report.promoted >= 1
        episode_ids = set()
        for item in report.items:
            assert item.scope == MemoryScope.USER
            assert item.owner_id == "u1"
            assert item.metadata["tier"] == "semantic"
            episode_ids.update(item.metadata["consolidated_from"])
        assert len(episode_ids) == 2
        archived = engine.store.all_items(statuses=("archived",))
        assert len(archived) == 2
        for episode in archived:
            assert episode.metadata["consolidated_into"]

    @pytest.mark.asyncio
    async def test_consolidation_below_min_items_is_noop(self):
        engine = MemoryEngine()
        engine.remember("We agreed to migrate billing to Postgres", session_id="s1")
        report = await engine.consolidate("s1", user_id="u1")
        assert report.promoted == 0
        assert engine.store.all_items(statuses=("archived",)) == []

    def test_dedup_merges_near_duplicates_with_provenance(self):
        from vincio.core.types import MemoryItem as Item
        from vincio.memory import MemoryConsolidator

        engine = MemoryEngine()
        first = Item(content="User prefers concise technical answers", owner_id="u1", confidence=0.9)
        second = Item(content="User prefers concise technical answers always", owner_id="u1", confidence=0.6)
        engine.store.put(first)
        engine.store.put(second)
        merged = MemoryConsolidator(engine).dedup(owner_id="u1")
        assert merged == 1
        survivor = engine.store.get(first.id)
        assert survivor.metadata["merged_from"] == [second.id]
        duplicate = engine.store.get(second.id)
        assert duplicate.status == "archived"
        assert duplicate.metadata["merged_into"] == first.id


class TestHygiene:
    def test_ttl_default_applied_per_scope(self):
        engine = MemoryEngine(ttl_days={"session": 1.0})
        session_item = engine.remember("We agreed to review contract clauses quarterly", session_id="s1")
        user_item = engine.remember("User prefers concise answers", user_id="u1")
        assert session_item.expires_at is not None
        assert user_item.expires_at is None

    def test_expired_items_excluded_from_recall(self):
        from datetime import timedelta

        from vincio.core.utils import utcnow

        engine = MemoryEngine()
        item = engine.write_fact("User prefers concise answers", scope="user", owner_id="u1", type="preference")
        item.expires_at = utcnow() - timedelta(days=1)
        engine.store.put(item)
        assert engine.search("answer style", user_id="u1") == []
        stats = engine.decay_pass()
        assert stats["expired"] == 1

    def test_importance_weighted_retention(self):
        from datetime import timedelta

        from vincio.core.utils import utcnow
        from vincio.memory import importance_score

        engine = MemoryEngine(min_confidence=0.4, retention_weight=1.0)
        fact = engine.write_fact("Office is based in Berlin Mitte", scope="user", owner_id="u1", confidence=0.41)
        preference = engine.write_fact(
            "User prefers concise answers", scope="user", owner_id="u1",
            type="preference", confidence=0.41,
        )
        preference.usage_count = 10
        preference.confirmations = 4
        aged = utcnow() - timedelta(days=250)
        fact.updated_at = aged
        preference.updated_at = aged
        engine.store.put(fact)
        engine.store.put(preference)
        assert importance_score(preference) > importance_score(fact)
        engine.decay_pass()
        assert engine.store.get(fact.id).status == "archived"
        assert engine.store.get(preference.id).status in ("active", "decayed")

    def test_forget_and_edit_flow_through_audit(self):
        from vincio.security.audit import AuditLog

        audit = AuditLog(directory=None)
        engine = MemoryEngine(audit=audit)
        item = engine.write_fact("User works at Globex", scope="user", owner_id="u1")
        engine.edit(item.id, content="User works at Initech in Austin")
        assert engine.store.get(item.id).content == "User works at Initech in Austin"
        assert engine.forget(item.id, reason="user_request") is True
        actions = [e.action for e in audit.entries]
        assert "memory_edit" in actions
        assert "memory_delete" in actions
        delete_entry = next(e for e in audit.entries if e.action == "memory_delete")
        assert delete_entry.details["reason"] == "user_request"
        assert audit.verify_chain()

    def test_edit_rejects_credentials(self):
        engine = MemoryEngine()
        item = engine.write_fact("User works at Globex", scope="user", owner_id="u1")
        with pytest.raises(MemoryPolicyError):
            engine.edit(item.id, content="api_key = sk-supersecret1234567890")

    def test_export_and_erase_owner_data(self):
        from vincio.security.audit import AuditLog

        audit = AuditLog(directory=None)
        engine = MemoryEngine(audit=audit)
        engine.write_fact("User prefers concise answers", scope="user", owner_id="u1", type="preference")
        engine.write_fact("User timezone is UTC-5", scope="user", owner_id="u1")
        engine.write_fact("Other user prefers verbose answers", scope="user", owner_id="u2", type="preference")
        exported = engine.export_owner_data("u1")
        assert len(exported) == 2
        assert all(record["owner_id"] == "u1" for record in exported)
        erased = engine.erase_owner_data("u1")
        assert erased == 2
        assert engine.search("answer style", user_id="u1") == []
        assert engine.search("answer style", user_id="u2")
        actions = [e.action for e in audit.entries]
        assert "memory_export" in actions and "memory_erase" in actions


class TestWriteBack:
    def test_evidence_and_tools_become_candidates(self):
        from vincio.core.types import EvidenceItem, ToolResult

        engine = MemoryEngine()
        evidence = EvidenceItem(
            source_id="contract.md",
            text="The Pro plan refund window is 30 days from purchase",
            provenance=0.9,
        )
        tool = ToolResult(
            call_id="c1", tool_name="billing_lookup", status="ok",
            output={"invoice": "INV-9", "amount": 42.0},
        )
        failed = ToolResult(call_id="c2", tool_name="billing_lookup", status="error")
        written = engine.write_back(
            evidence=[evidence], tool_results=[tool, failed], session_id="s1",
            source_trace_id="tr1",
        )
        assert len(written) == 2
        assert all(item.status == "candidate" for item in written)
        assert written[0].metadata["origin"] == "evidence"
        assert written[1].metadata["origin"] == "tool"
        assert all(item.source_trace_id == "tr1" for item in written)
        results = engine.search("refund window", session_id="s1")
        assert results and results[0].components["status"] == 0.7

    def test_confirmed_candidate_becomes_active(self):
        engine = MemoryEngine()
        from vincio.core.types import EvidenceItem

        evidence = EvidenceItem(source_id="doc", text="User prefers invoices in EUR currency")
        written = engine.write_back(evidence=[evidence], owner_id="u1")
        assert written[0].status == "candidate"
        confirmed = engine.confirm(written[0].id)
        assert confirmed.status == "active"


class TestMemoryEvalHarness:
    def _seeded_engine(self):
        engine = MemoryEngine()
        engine.write_fact("User prefers concise technical answers", scope="user", owner_id="u1", type="preference")
        engine.write_fact("User works in the compliance department", scope="user", owner_id="u1")
        engine.write_fact("User prefers detailed walkthroughs", scope="user", owner_id="u2", type="preference")
        engine.write_fact("User timezone is UTC+1", scope="user", owner_id="u3", confidence=0.6)
        engine.write_fact("User timezone is UTC-5", scope="user", owner_id="u3", confidence=0.9)
        return engine

    @pytest.mark.asyncio
    async def test_evaluate_memory_metrics(self):
        from vincio.memory import evaluate_memory, personalization_dataset

        engine = self._seeded_engine()
        report = await evaluate_memory(engine, personalization_dataset(), top_k=3)
        assert report.metrics["recall_at_k"] >= 0.75
        assert report.metrics["recall_precision"] > 0.0
        assert report.metrics["staleness"] == 0.0
        assert report.metrics["contradiction_rate"] <= 0.1
        assert report.metrics["personalization_lift"] > 0.0
        assert len(report.cases) == 4

    def test_contradiction_rate_flags_unresolved_conflicts(self):
        from vincio.core.types import MemoryItem as Item
        from vincio.memory import contradiction_rate

        engine = MemoryEngine()
        engine.store.put(Item(content="User prefers detailed verbose answers", owner_id="u1"))
        engine.store.put(Item(content="User prefers concise technical answers", owner_id="u1"))
        assert contradiction_rate(engine) > 0.0
