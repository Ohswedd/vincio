"""3.0 provable erasure + consent/purpose + bi-temporal memory:
signed erasure proofs across stores, the ConsentLedger, and as-of / ACL recall."""

from datetime import UTC, datetime

from vincio import ContextApp
from vincio.core.types import MemoryItem
from vincio.governance import (
    ConsentLedger,
    ErasureProof,
    HmacSigner,
    LawfulBasis,
    Purpose,
    build_erasure_proof,
    verify_erasure_proof,
)
from vincio.governance.consent import ConsentDecision
from vincio.memory import MemoryEngine
from vincio.security.access import AccessController, Principal

# ---------------------------------------------------------------------------
# Provable erasure
# ---------------------------------------------------------------------------


class TestErasureProof:
    def test_proof_binds_to_removed_ids(self):
        proof = build_erasure_proof("docs/policy.md", {"chunks": ["c1", "c2"], "memories": ["m1"]})
        assert isinstance(proof, ErasureProof)
        assert proof.content_sha256
        assert verify_erasure_proof(proof)

    def test_tampered_proof_fails_verification(self):
        proof = build_erasure_proof("s", {"chunks": ["c1", "c2"]})
        # Editing the recorded id set without recomputing the digest breaks the
        # content binding — the proof no longer verifies.
        proof.removed_ids["chunks"].append("c3")
        assert not verify_erasure_proof(proof)

    def test_signed_proof_requires_verifier(self):
        signer = HmacSigner("erasure-key")
        proof = build_erasure_proof("s", {"chunks": ["c1"]}, signer=signer)
        assert proof.signature is not None
        # A present signature with no verifier is never reported valid.
        assert not verify_erasure_proof(proof)
        assert verify_erasure_proof(proof, signer=signer)
        # A wrong key fails.
        assert not verify_erasure_proof(proof, signer=HmacSigner("wrong-key"))

    def test_app_erase_source_emits_signed_proof(self, sample_docs_dir, offline_config, tmp_cwd):
        app = ContextApp(name="erase", config=offline_config)
        app.content_signer = HmacSigner("app-erasure-key")
        app.add_source("docs", path=str(sample_docs_dir), retrieval="bm25")
        result = app.erase_source("docs")
        assert result.found
        assert result.chunks_removed > 0
        assert result.proof is not None
        assert result.proof.signature is not None
        assert verify_erasure_proof(result.proof, signer=app.content_signer)
        # The proof is anchored to the audit chain and a dedicated entry exists.
        assert "erasure_proof" in [e.action for e in app.audit.entries]

    def test_erase_source_covers_artifacts(self, sample_docs_dir, offline_config, tmp_cwd):
        app = ContextApp(name="erase2", config=offline_config)
        app.add_source("docs", path=str(sample_docs_dir), retrieval="bm25")
        # A generated artifact derived from the source is tracked and erased too.
        app.lineage.record_artifact("docs", "reports/summary.pdf")
        result = app.erase_source("docs")
        assert result.artifacts_removed == 1
        assert "artifacts" in (result.proof.removed_ids if result.proof else {})

    def test_erase_source_idempotent(self, sample_docs_dir, offline_config, tmp_cwd):
        app = ContextApp(name="erase3", config=offline_config)
        app.add_source("docs", path=str(sample_docs_dir), retrieval="bm25")
        first = app.erase_source("docs")
        assert first.found
        second = app.erase_source("docs")
        assert not second.found
        assert second.total_removed == 0


# ---------------------------------------------------------------------------
# Consent ledger
# ---------------------------------------------------------------------------


class TestConsentLedger:
    def test_grant_and_check(self):
        ledger = ConsentLedger()
        ledger.grant("u1", [Purpose.PERSONALIZATION], lawful_basis=LawfulBasis.CONSENT)
        decision = ledger.check("u1", Purpose.PERSONALIZATION)
        assert isinstance(decision, ConsentDecision)
        assert decision.allowed
        assert decision.lawful_basis == "consent"

    def test_no_record_denies_by_default(self):
        ledger = ConsentLedger()
        assert not ledger.allows("u1", Purpose.ANALYTICS)

    def test_revoke_denies(self):
        ledger = ConsentLedger()
        ledger.grant("u1", [Purpose.PERSONALIZATION, Purpose.ANALYTICS])
        ledger.revoke("u1", purpose=Purpose.ANALYTICS)
        assert ledger.allows("u1", Purpose.PERSONALIZATION)
        assert not ledger.allows("u1", Purpose.ANALYTICS)

    def test_default_allow_mode(self):
        ledger = ConsentLedger(default_allow=True)
        assert ledger.allows("anyone", Purpose.SERVICE)

    def test_access_controller_check_purpose(self):
        ledger = ConsentLedger()
        ledger.grant("u1", [Purpose.PERSONALIZATION])
        ctl = AccessController(consent_ledger=ledger)
        granted = ctl.check_purpose(Principal(user_id="u1"), purpose="personalization")
        assert granted.allowed and granted.lawful_basis == "consent"
        denied = ctl.check_purpose(Principal(user_id="u2"), purpose="personalization")
        assert not denied.allowed

    def test_check_purpose_no_ledger_is_noop_allow(self):
        ctl = AccessController()
        decision = ctl.check_purpose(Principal(user_id="u1"), purpose="service")
        assert decision.allowed


# ---------------------------------------------------------------------------
# Bi-temporal memory + ACL + team scope
# ---------------------------------------------------------------------------


class TestBiTemporalMemory:
    def test_correct_preserves_as_of_history(self):
        engine = MemoryEngine()
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        item = engine.write_fact("User lives in Berlin", scope="user", owner_id="u1", valid_from=t0)
        engine.correct(item.id, "User lives in Munich", valid_from=datetime(2026, 3, 1, tzinfo=UTC))
        # Current recall returns the corrected value …
        current = engine.recall("where does the user live", user_id="u1")
        assert any("Munich" in m.content for m in current)
        assert not any("Berlin" in m.content for m in current)
        # … but an as-of recall before the correction returns the prior value.
        past = engine.recall(
            "where does the user live", user_id="u1", as_of=datetime(2026, 2, 1, tzinfo=UTC)
        )
        assert any("Berlin" in m.content for m in past)

    def test_valid_at_interval(self):
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        t1 = datetime(2026, 6, 1, tzinfo=UTC)
        item = MemoryItem(content="x", valid_from=t0, valid_to=t1)
        assert item.valid_at(datetime(2026, 3, 1, tzinfo=UTC))
        assert not item.valid_at(datetime(2025, 12, 1, tzinfo=UTC))
        assert not item.valid_at(t1)  # half-open interval

    def test_acl_gates_team_recall(self):
        engine = MemoryEngine()
        team = engine.for_team("eng")
        team.remember("Rotated the prod deploy key", acl=["alice"])
        assert engine.recall("deploy key", team_id="eng", reader="alice")
        assert not engine.recall("deploy key", team_id="eng", reader="bob")
        # An open (no-ACL) team memory is visible to any reader.
        team.remember("Standup is at 10am")
        assert engine.recall("standup", team_id="eng", reader="bob")

    def test_team_scope_isolation(self):
        engine = MemoryEngine()
        engine.for_team("eng").remember("Eng-only note", acl=["alice"])
        assert not engine.recall("note", team_id="sales", reader="alice")

    def test_recall_drops_withdrawn_consent(self):
        ledger = ConsentLedger()
        ledger.grant("u1", [Purpose.PERSONALIZATION])
        engine = MemoryEngine(consent_ledger=ledger)
        engine.write_fact(
            "User prefers concise answers",
            scope="user",
            owner_id="u1",
            type="preference",
            purpose="personalization",
        )
        assert engine.recall("answer style", user_id="u1")
        ledger.revoke("u1")
        assert not engine.recall("answer style", user_id="u1")


class TestMemoryItemSchema:
    def test_new_fields_default_backward_compatible(self):
        item = MemoryItem(content="x")
        assert item.valid_from is None
        assert item.valid_to is None
        assert item.acl == []
        assert item.purpose is None
        assert item.consent_id is None
        # An item with no interval is always valid; an empty ACL is open.
        assert item.valid_at(datetime.now(UTC))
        assert item.readable_by(None)

    def test_sqlite_round_trips_bitemporal_fields(self, tmp_path):
        from vincio.memory.stores import SQLiteMemoryStore

        store = SQLiteMemoryStore(tmp_path / "m.db")
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        store.put(
            MemoryItem(content="x", owner_id="u1", valid_from=t0, acl=["alice"], purpose="service")
        )
        items = store.all_items(statuses=())
        assert len(items) == 1
        loaded = items[0]
        assert loaded.valid_from == t0
        assert loaded.acl == ["alice"]
        assert loaded.purpose == "service"
        store.close()
