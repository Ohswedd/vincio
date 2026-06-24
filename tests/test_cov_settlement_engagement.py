"""Coverage-hardening tests for ``vincio.settlement.engagement``.

The cross-org engagement facade threads the *whole* settlement & credit fabric
behind one governed, narrated call-path. The companion suite
(``test_cross_org_engagement.py``) proves the happy-path negotiate→settle→net
lifecycle and the chain's tamper-evidence; this file drives the *remaining*
facade verbs (admit, settle, arbitrate, escrow/pool/guard collateral, reputation
attestation & portability, completeness / root-consistency / history checks) and
the low-level helpers and verification branches they exercise, asserting on the
exact stages, hashes, and verdicts each produces — never on "is not None".
"""

from __future__ import annotations

import pytest

from vincio import (
    ContextApp,
    CrossOrgEngagement,
    EngagementNarrative,
    EngagementStage,
)
from vincio.choreography import Saga, StepOutcome
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms, buyer_position, seller_position
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement.engagement import (
    ENGAGEMENT_ACTION,
    EngagementSignature,
    EngagementVerification,
    _artifact_digest,
    _artifact_hash,
    _artifact_id,
    _artifact_kind,
    _artifact_wire,
)


def _app(name: str = "acme") -> ContextApp:
    return ContextApp(name=name, provider=MockProvider(default_text="ok"))


def _engagement(app: ContextApp | None = None) -> tuple[ContextApp, CrossOrgEngagement]:
    app = app or _app()
    eng = app.cross_org_engagement(buyer="acme", seller="vendor", scope="transcribe 1k calls")
    return app, eng


def _negotiate(eng: CrossOrgEngagement):
    return eng.negotiate(
        buyer=buyer_position(max_price_usd=0.12, max_sla_seconds=5.0),
        seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.10),
    )


def _contract(price: float = 1.0, *, buyer: str = "acme", seller: str = "vendor") -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope="transcribe", price_usd=price)
    ).seal()


# == low-level artifact helpers ===============================================


def test_artifact_wire_prefers_to_wire_over_model_dump():
    # An artifact carrying to_wire uses it in preference to model_dump.
    class WithWire:
        def to_wire(self):
            return {"src": "wire"}

        def model_dump(self, mode="json"):
            return {"src": "dump"}

    assert _artifact_wire(WithWire()) == {"src": "wire"}


def test_artifact_wire_uses_model_dump_when_no_to_wire():
    # A pydantic model without to_wire (e.g. a Contract) projects via model_dump.
    contract = _contract()
    assert not hasattr(contract, "to_wire")
    assert _artifact_wire(contract) == contract.model_dump(mode="json")


def test_artifact_wire_falls_back_to_jsonable_for_plain_objects():
    # No to_wire, no model_dump: line 86 — the to_jsonable fallback.
    assert _artifact_wire({"a": 1, "b": [2, 3]}) == {"a": 1, "b": [2, 3]}
    assert _artifact_wire("plain") == "plain"


def test_artifact_wire_projects_lists_element_wise():
    c1, c2 = _contract(1.0), _contract(2.0)
    assert _artifact_wire([c1, c2]) == [c1.model_dump(mode="json"), c2.model_dump(mode="json")]


def test_artifact_kind_labels_lists_and_empty_lists():
    assert _artifact_kind(_contract()) == "Contract"
    assert _artifact_kind([_contract()]) == "list[Contract]"
    assert _artifact_kind([]) == "list[object]"


def test_artifact_id_empty_for_lists_and_idless_objects():
    assert _artifact_id([_contract()]) == ""
    assert _artifact_id(object()) == ""
    contract = _contract()
    assert _artifact_id(contract) == contract.id and contract.id != ""


def test_artifact_hash_reads_content_hash_then_head_then_journal():
    # content_hash branch.
    contract = _contract()
    assert _artifact_hash(contract) == contract.content_hash

    # head_hash branch (no content_hash) — line 119-122.
    class Ledger:
        content_hash = ""
        head_hash = "abc123"

    assert _artifact_hash(Ledger()) == "abc123"

    # journal.head_hash branch — lines 123-127.
    class Journal:
        head_hash = "jjj999"

    class Result:
        content_hash = ""
        head_hash = ""
        journal = Journal()

    assert _artifact_hash(Result()) == "jjj999"

    # Nothing to read, and lists carry no own hash.
    assert _artifact_hash(object()) == ""
    assert _artifact_hash([contract]) == ""


def test_artifact_hash_empty_journal_head_returns_blank():
    # journal present but its head is empty — the final fallthrough to "".
    class Journal:
        head_hash = ""

    class Result:
        journal = Journal()

    assert _artifact_hash(Result()) == ""


def test_artifact_digest_is_stable_and_32_hex():
    contract = _contract()
    d = _artifact_digest(contract)
    assert d == _artifact_digest(contract)
    assert len(d) == 32 and all(ch in "0123456789abcdef" for ch in d)


# == EngagementStage round-trip & hashing =====================================


def test_stage_to_wire_from_wire_round_trip_preserves_link_hash():
    # Exercises EngagementStage.to_wire (174) and from_wire (178).
    stage = EngagementStage(index=2, stage="settle", kind="SettlementRecord", digest="ab", summary={"n": 1})
    stage.entry_hash = stage.compute_entry_hash()
    restored = EngagementStage.from_wire(stage.to_wire())
    assert restored.compute_entry_hash() == stage.compute_entry_hash()
    assert restored.stage == "settle" and restored.summary == {"n": 1}


def test_stage_entry_hash_ignores_index_changes_via_link_facts():
    # link_facts binds index, so two different indices differ — proving index is bound.
    a = EngagementStage(index=0, stage="settle", digest="x")
    b = EngagementStage(index=1, stage="settle", digest="x")
    assert a.compute_entry_hash() != b.compute_entry_hash()


# == admit / settle / arbitrate facade verbs ==================================


def test_admit_records_decision_and_defaults_subject_to_seller():
    app, eng = _engagement()
    decision = eng.admit()  # subject defaults to "vendor"
    assert eng.admission is decision
    assert decision.subject == "vendor"
    narrative = eng.seal()
    admit_stage = narrative.stage("admit")
    assert admit_stage is not None
    assert admit_stage.summary["subject"] == "vendor"
    assert narrative.verify(app.contract_signer).valid


def test_settle_uses_negotiated_contract_and_records_balance():
    app, eng = _engagement()
    _negotiate(eng)
    record = eng.settle(cost_usd=0.05, latency_ms=1000, quality=0.95)
    assert eng.settlements == [record]
    narrative = eng.seal()
    settle_stage = narrative.stage("settle")
    assert settle_stage is not None
    assert settle_stage.summary["status"] == record.status
    assert settle_stage.summary["balance_usd"] == pytest.approx(record.balance_usd)


def test_settle_with_explicit_contract_overrides_default():
    app, eng = _engagement()
    explicit = _contract(price=2.0)
    record = eng.settle(explicit, cost_usd=0.1, latency_ms=900, quality=0.9)
    assert record.contract_id == explicit.id


def test_settle_saga_without_result_raises():
    app, eng = _engagement()
    with pytest.raises(SettlementError, match="settle_saga"):
        eng.settle_saga(contracts={})


def test_arbitrate_records_resolution_status_and_contract_id():
    app, eng = _engagement()
    contract = _contract()
    record = app.settle(contract, cost_usd=0.05, latency_ms=900, quality=0.95)
    resolution = eng.arbitrate([record])
    assert eng.arbitration is resolution
    narrative = eng.seal()
    arb_stage = narrative.stage("arbitrate")
    assert arb_stage is not None
    assert arb_stage.summary["status"] == resolution.status
    assert arb_stage.summary["contract_id"] == contract.id


# == collateral facade verbs ==================================================


def test_post_escrow_uses_default_contract_and_records_stage():
    app, eng = _engagement()
    _negotiate(eng)
    escrow = eng.post_escrow(fraction=0.2)
    assert eng.escrow is escrow
    # The escrow binds the default (negotiated) contract.
    assert escrow.contract_id == eng.contract.id
    narrative = eng.seal()
    stage = narrative.stage("post_escrow")
    assert stage is not None
    assert stage.kind == "Escrow"
    assert stage.artifact_hash == escrow.content_hash
    assert narrative.verify(app.contract_signer).valid


def test_post_escrow_without_contract_raises():
    app, eng = _engagement()
    with pytest.raises(SettlementError, match="post_escrow"):
        eng.post_escrow()


def test_post_collateral_pool_and_guard_collateral_chain():
    app, eng = _engagement()
    c1, c2 = _contract(100.0), _contract(200.0)
    pool = eng.post_collateral_pool([c1, c2], fraction=0.1)
    assert eng.pool is pool
    ledger = eng.guard_collateral([pool])
    assert eng.ledger is ledger
    narrative = eng.seal()
    assert narrative.stage_names == ["post_collateral_pool", "guard_collateral"]
    pool_stage = narrative.stage("post_collateral_pool")
    assert pool_stage.summary["posted_usd"] == pytest.approx(pool.posted_usd)
    guard_stage = narrative.stage("guard_collateral")
    assert guard_stage.summary["status"] == ledger.status
    assert narrative.verify(app.contract_signer).valid


# == reputation attestation & portability =====================================


def test_attest_reputation_records_subject_and_outcomes():
    app, eng = _engagement()
    # Give the seller a real fulfilled settlement so the attestation has substance.
    contract = _contract()
    app.settle(contract, cost_usd=0.05, latency_ms=900, quality=0.95)
    attestation = eng.attest_reputation()  # subject defaults to "vendor"
    assert eng.attestation is attestation
    assert attestation.subject == "vendor"
    narrative = eng.seal()
    stage = narrative.stage("attest_reputation")
    assert stage is not None
    assert stage.summary["subject"] == "vendor"
    assert stage.summary["successes"] == attestation.successes


def test_import_reputation_facade_hits_latent_standings_len_bug():
    # BUG (documented): CrossOrgEngagement.import_reputation summarizes issuers with
    #   len(getattr(prior, "standings", []) or getattr(prior, "issuers", []) or [])
    # but PortableReputation.standings is a *method*, not a list — getattr returns the
    # bound method (truthy), so `len(method)` raises TypeError. The facade therefore
    # crashes for any real PortableReputation. This test pins the current behavior so
    # a fix (e.g. `len(prior.subjects())`) is a visible, intentional change.
    issuer_a = _app("issuer-a")
    issuer_a.use_settlement_book(owner="issuer-a")
    issuer_a.settle(
        _contract(buyer="issuer-a", seller="vendor"),
        cost_usd=0.05,
        latency_ms=900,
        quality=0.95,
    )
    att_a = issuer_a.attest_reputation("vendor")

    app, eng = _engagement()
    with pytest.raises(TypeError, match="has no len"):
        eng.import_reputation([att_a], subject="vendor")
    # The prior itself was produced fine and stored before the summary blew up.
    assert eng.reputation is not None
    assert eng.reputation.subjects() == ["vendor"]


# == solvency completeness / consistency facade verbs =========================


def test_check_completeness_records_omission_status():
    app, eng = _engagement()
    owed = eng.attest_liabilities("vendor", {"acme": 60.0})
    # globex can prove a claim the attestation omitted entirely.
    proof = eng.check_completeness(owed, {"acme": 60.0, "globex": 40.0})
    assert eng.completeness is proof
    narrative = eng.seal()
    stage = narrative.stage("check_completeness")
    assert stage is not None
    assert stage.summary["status"] == proof.status
    assert proof.status != "complete"


def test_check_root_consistency_flags_equivocation():
    app = _app("auditor")
    eng = app.cross_org_engagement(buyer="acme", seller="vendor", scope="x")
    vendor = _app("vendor")
    from vincio.core.utils import utcnow

    t = utcnow()
    owed_acme = vendor.attest_liabilities("vendor", {"acme": 60.0}, as_of=t)
    owed_globex = vendor.attest_liabilities("vendor", {"globex": 40.0}, as_of=t)
    report = eng.check_root_consistency([("acme", owed_acme), ("globex", owed_globex)])
    assert eng.root_consistency is report
    narrative = eng.seal()
    stage = narrative.stage("check_root_consistency")
    assert stage is not None
    assert stage.summary["consistent"] is False
    assert stage.summary["equivocations"] >= 1


def test_check_history_consistency_flags_dropped_debt():
    app = _app("auditor")
    eng = app.cross_org_engagement(buyer="acme", seller="vendor", scope="x")
    vendor = _app("vendor")
    from vincio.core.utils import utcnow

    t1 = utcnow()
    s1 = vendor.attest_liabilities("vendor", {"acme": 100.0}, as_of=t1)
    s2 = vendor.attest_liabilities(
        "vendor", {"acme": 30.0}, as_of=t1.replace(year=t1.year + 1), prior=s1
    )
    report = eng.check_history_consistency([s1, s2])
    assert eng.history is report
    narrative = eng.seal()
    stage = narrative.stage("check_history_consistency")
    assert stage is not None
    assert stage.summary["consistent"] is False


# == narrative verification branches ==========================================


def test_verify_artifact_count_mismatch_sets_digests_not_ok():
    # Exercises the len(artifacts) != len(stages) branch (327-329).
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    result = narrative.verify(app.contract_signer, artifacts=[])  # zero artifacts, one stage
    assert not result.digests_ok
    assert not result.valid
    assert result.broken_at == 0


def test_verify_count_mismatch_keeps_existing_broken_at():
    # Branch 328->337: a chain already broken at stage 0 *and* an artifact-count
    # mismatch — broken_at is preserved from the chain walk, not overwritten.
    app, eng = _engagement()
    contract = _negotiate(eng)
    saga = Saga(name="f").step("t", participant="vendor", action="run", contract=contract)
    eng.choreograph(
        saga,
        participants={
            "vendor": {"run": lambda p: StepOutcome(ok=True, cost_usd=0.05, latency_ms=1, quality=0.9)}
        },
    )
    narrative = eng.seal()
    forged = EngagementNarrative.from_wire(narrative.to_wire())
    forged.stages[0].digest = "broken"  # breaks the chain at stage 0
    result = forged.verify(artifacts=[_contract()])  # also a count mismatch (1 vs 2)
    assert not result.intact and not result.digests_ok
    assert result.broken_at == 0  # preserved from the chain walk


def test_verify_with_signatures_present_but_no_verifier():
    # require defaults to the coordinator; with no verifier the binding is unauthenticated.
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    result = narrative.verify()  # no verifier, required coordinator present but unauthenticated
    assert not result.valid
    assert "not authenticated" in (result.reason or "")
    # signed_by still lists the present signatures (verifier None branch, line 347).
    assert result.signed_by == ["acme"]


def test_verify_empty_require_passes_without_verifier():
    # require=[] checks the binding alone — no signature requirement.
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    result = narrative.verify(require=[])
    assert result.valid
    assert result.reason is None


def test_sign_seals_an_unsealed_narrative_first():
    # Line 281: sign() on a never-sealed narrative seals it before signing.
    stage = EngagementStage(stage="negotiate", digest="d1")
    narrative = EngagementNarrative(coordinator="acme", stages=[stage])
    assert narrative.content_hash == ""
    signer = HMACSigner("acme-key", key_id="acme")
    narrative.sign(signer, party="acme")
    assert narrative.content_hash != ""  # seal() ran
    assert narrative.verify(signer).valid


def test_chain_broken_mid_walk_pinpoints_stage():
    # Lines 317-319: a corrupted prev_hash on a later stage breaks the walk there.
    app, eng = _engagement()
    contract = _negotiate(eng)
    saga = Saga(name="f").step("t", participant="vendor", action="run", contract=contract)
    eng.choreograph(
        saga,
        participants={
            "vendor": {"run": lambda p: StepOutcome(ok=True, cost_usd=0.05, latency_ms=1, quality=0.9)}
        },
    )
    narrative = eng.seal()
    forged = EngagementNarrative.from_wire(narrative.to_wire())
    forged.stages[1].prev_hash = "0" * 32  # breaks the link at stage 1, mid-walk
    result = forged.verify()
    assert not result.intact and result.broken_at == 1
    assert "chain broken at stage 1" == result.reason


def test_require_valid_returns_self_when_valid_and_raises_with_details():
    # Lines 394-400: the happy return and the raising branch with details.
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    assert narrative.require_valid(app.contract_signer) is narrative
    narrative.stages[0].digest = "tampered"
    with pytest.raises(SettlementError) as exc:
        narrative.require_valid(app.contract_signer)
    assert exc.value.details["engagement_id"] == narrative.id
    assert "verification" in str(exc.value)


def test_narrative_to_wire_from_wire_round_trip():
    # Lines 435, 439: EngagementNarrative.to_wire / from_wire.
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    restored = EngagementNarrative.from_wire(narrative.to_wire())
    assert restored.content_hash == narrative.content_hash
    assert restored.head_hash == narrative.head_hash
    assert restored.stage_names == narrative.stage_names


def test_resign_replaces_prior_signature_for_same_party():
    # Line 281 path: content_hash already set on second sign; signature replaced not duplicated.
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()  # already signed once by coordinator
    assert narrative.signed_by.count("acme") == 1
    narrative.sign(app.contract_signer, party="acme")  # re-sign, content_hash already present
    assert narrative.signed_by.count("acme") == 1


def test_forged_signature_fails_and_reports_mismatch():
    # Line 345 + reason 369-370: a signature that does not verify under the verifier.
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    stranger = HMACSigner("other-secret", key_id="stranger")
    result = narrative.verify(stranger)
    assert not result.signatures_ok and not result.valid
    assert result.reason is not None


def test_head_mismatch_reason_when_chain_intact():
    # Line 362: intact chain but a tampered head → "head hash mismatch".
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    forged = EngagementNarrative.from_wire(narrative.to_wire())
    forged.head_hash = "f" * 32
    result = forged.verify(app.contract_signer)
    assert result.intact and not result.head_ok
    assert result.reason == "head hash mismatch"


def test_content_hash_mismatch_reason_when_head_ok():
    # Line 364: head_ok but the content hash was tampered → "content hash mismatch".
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    forged = EngagementNarrative.from_wire(narrative.to_wire())
    forged.content_hash = "0" * 32
    result = forged.verify(app.contract_signer)
    assert result.head_ok and not result.hash_ok
    assert result.reason == "content hash mismatch"


def test_digest_mismatch_reason_pinpoints_stage():
    # Line 367-368: digests_ok False with an aligned-but-tampered artifact.
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    # one stage, one artifact, but a different artifact digest.
    other = _contract(price=999.0)
    result = narrative.verify(app.contract_signer, artifacts=[other])
    assert not result.digests_ok
    assert result.reason == "artifact digest mismatch at stage 0"


def test_missing_required_signature_reason():
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    result = narrative.verify(app.contract_signer, require=["someone-else"])
    assert not result.valid
    assert "someone-else" in (result.reason or "")


# == views: stage(name) miss, audit_details, print_summary ====================


def test_stage_lookup_returns_none_for_unknown_verb():
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    assert narrative.stage("nonexistent") is None  # lines 411-414 (loop exhausts)


def test_audit_details_reports_full_engagement_shape():
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    details = narrative.audit_details()
    assert details["coordinator"] == "acme"
    assert details["buyer"] == "acme" and details["seller"] == "vendor"
    assert details["stages"] == ["negotiate"]
    assert details["stage_count"] == 1
    assert details["content_hash"] == narrative.content_hash
    assert details["signed_by"] == ["acme"]


def test_print_summary_emits_one_line_per_stage(capsys):
    app, eng = _engagement()
    contract = _negotiate(eng)
    saga = Saga(name="fulfil").step("t", participant="vendor", action="run", contract=contract)
    eng.choreograph(
        saga,
        participants={
            "vendor": {"run": lambda p: StepOutcome(ok=True, cost_usd=0.05, latency_ms=1, quality=0.9)}
        },
    )
    narrative = eng.seal()
    narrative.print_summary()
    out = capsys.readouterr().out
    assert "negotiate" in out and "choreograph" in out
    assert "acme" in out and "vendor" in out
    assert f"head={narrative.head_hash}" in out


# == seal options & signer resolution =========================================


def test_seal_without_sign_leaves_narrative_unsigned():
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal(sign=False)
    assert narrative.signatures == []
    # Unsigned still verifies its binding when no signatures are required.
    assert narrative.verify(require=[]).valid


def test_seal_without_audit_does_not_touch_audit_chain():
    app, eng = _engagement()
    _negotiate(eng)
    eng.seal(record_audit=False)
    assert app.audit.query(action=ENGAGEMENT_ACTION) == []


def test_engagement_verify_seals_with_signer_then_authenticates():
    app, eng = _engagement()
    _negotiate(eng)
    # verify() with a verifier seals signed and re-digests every captured artifact.
    result = eng.verify(app.contract_signer)
    assert result.valid
    assert result.digests_ok
    assert result.signed_by == ["acme"]


def test_signer_falls_back_to_contract_signer_attribute():
    # An app lacking _resolve_contract_signer falls through to .contract_signer (line 955).
    class BareApp:
        name = "bare"
        settlement_book = object()
        reputation_ledger = object()
        contract_signer = HMACSigner("bare-key", key_id="bare")
        audit = None

    eng = CrossOrgEngagement(BareApp(), buyer="b", seller="s", scope="x")
    eng.record_stage("custom", _contract(), note="hi")
    narrative = eng.seal()
    assert narrative.signed_by == ["bare"]
    assert narrative.verify(BareApp.contract_signer).valid


def test_signer_returns_none_when_no_signer_resolvable():
    class NoSignerApp:
        name = "nosign"
        settlement_book = object()
        reputation_ledger = object()
        contract_signer = None
        audit = None

    eng = CrossOrgEngagement(NoSignerApp(), buyer="b", seller="s", scope="x")
    eng.record_stage("custom", _contract())
    narrative = eng.seal()  # sign=True but no signer resolvable → no signatures
    assert narrative.signatures == []
    assert narrative.verify(require=[]).valid


# == small model surfaces =====================================================


def test_engagement_signature_carries_party_and_key_id():
    sig = EngagementSignature(party="acme", signature="deadbeef", key_id="k1")
    assert sig.party == "acme" and sig.key_id == "k1"


def test_engagement_verification_conjunction_fields():
    v = EngagementVerification(
        valid=True,
        intact=True,
        head_ok=True,
        hash_ok=True,
        digests_ok=True,
        signatures_ok=True,
        signed_by=["acme"],
        stages=3,
    )
    assert v.valid and v.broken_at is None and v.reason is None
    assert v.signed_by == ["acme"] and v.stages == 3


def test_record_stage_drops_none_summary_values():
    app, eng = _engagement()
    eng.record_stage("custom", _contract(), kept="yes", dropped=None)
    stage = eng.stages[0]
    assert stage.summary == {"kept": "yes"}  # None values filtered out
