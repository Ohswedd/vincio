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
