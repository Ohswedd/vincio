"""The cross-org engagement lifecycle facade — composition, narrative, conformance.

The capstone that unifies the cross-org settlement & credit fabric into one coherent,
conformance-proven system. These tests prove the facade is *purely compositional* (it
delegates to the same primitives, which stay usable directly), that it narrates the
whole pipeline into one hash-linked, signed, offline-verifiable narrative, and that a
tamper introduced anywhere — a re-ordered stage, an edited digest, a forged signature,
or an edited underlying artifact — is caught from the bytes alone.
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
from vincio.negotiation import buyer_position, seller_position
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner


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


def _deliver(eng: CrossOrgEngagement, contract):
    saga = Saga(name="fulfil").step(
        "transcribe", participant="vendor", action="run", contract=contract
    )
    parts = {
        "vendor": {
            "run": lambda p: StepOutcome(
                ok=True, cost_usd=0.05, latency_ms=1200, quality=0.95, output={"t": 1}
            )
        }
    }
    return eng.choreograph(saga, participants=parts)


# -- construction & setup -----------------------------------------------------


def test_engagement_ensures_book_and_ledger():
    app = _app()
    assert app.settlement_book is None
    eng = app.cross_org_engagement(buyer="acme", seller="vendor", scope="x")
    assert app.settlement_book is not None
    assert app.reputation_ledger is not None
    assert eng.coordinator == "acme"


def test_engagement_respects_preconfigured_book():
    app = _app()
    book = app.use_settlement_book(owner="custom")
    app.cross_org_engagement(buyer="acme", seller="vendor")
    assert app.settlement_book is book  # not overwritten


# -- the happy-path lifecycle threads into a narrative ------------------------


def test_full_lifecycle_threads_a_verifiable_narrative():
    app, eng = _engagement()
    contract = _negotiate(eng)
    assert eng.contract is contract
    delivery = _deliver(eng, contract)
    assert delivery.status == "completed"
    records = eng.settle_saga(contracts={contract.id: contract})
    assert len(records) == 1
    netting = eng.net()
    assert netting.verify().valid

    narrative = eng.seal()
    assert narrative.stage_names == ["negotiate", "choreograph", "settle_saga", "net"]
    result = narrative.verify(app.contract_signer)
    assert result.valid
    assert result.intact and result.head_ok and result.hash_ok
    assert result.signed_by == ["acme"]
    assert result.stages == 4


def test_captured_artifacts_are_exposed_as_attributes():
    app, eng = _engagement()
    contract = _negotiate(eng)
    _deliver(eng, contract)
    eng.settle_saga(contracts={contract.id: contract})
    eng.net()
    assert eng.contract is contract
    assert eng.delivery is not None
    assert eng.settlements and eng.netting is not None


def test_solvency_and_insolvency_stages():
    app, eng = _engagement()
    reserves = eng.attest_custody("vendor", {"omnibus": 50.0})
    owed = eng.attest_liabilities("vendor", {"bank": 60.0, "acme": 40.0})
    proof = eng.prove_solvency(reserves, owed)
    assert proof.status == "insolvent"
    schedule = app.build_seniority_schedule("vendor", [["bank"], ["acme"]])
    resolution = eng.resolve_insolvency(reserves, owed, schedule)
    assert eng.insolvency is resolution
    narrative = eng.seal()
    assert narrative.stage_names == [
        "attest_custody",
        "attest_liabilities",
        "prove_solvency",
        "resolve_insolvency",
    ]
    assert narrative.verify(app.contract_signer).valid


# -- the facade is compositional ----------------------------------------------


def test_primitives_remain_usable_directly():
    # The facade adds no new economic logic: the underlying app methods still work
    # on their own, byte-for-byte the same, whether or not an engagement wraps them.
    app, eng = _engagement()
    contract = _negotiate(eng)
    direct = app.settle(contract, cost_usd=0.05, latency_ms=1000, quality=0.95)
    assert direct.verify(app.contract_signer, require=["acme"]).valid
    assert direct.status in ("settled", "breached")


def test_record_stage_escape_hatch():
    app, eng = _engagement()
    _negotiate(eng)
    decision = app.admit("vendor")
    eng.record_stage("admit", decision, subject="vendor")
    narrative = eng.seal()
    assert "admit" in narrative.stage_names
    assert narrative.verify(app.contract_signer).valid


# -- tamper-evidence: a tamper anywhere is caught -----------------------------


def test_tampered_stage_digest_is_caught():
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    forged = EngagementNarrative.from_wire(narrative.to_wire())
    forged.stages[0].digest = "deadbeef"
    result = forged.verify()
    assert not result.valid and not result.intact
    assert result.broken_at == 0


def test_reordered_stages_are_caught():
    app, eng = _engagement()
    contract = _negotiate(eng)
    _deliver(eng, contract)
    eng.settle_saga(contracts={contract.id: contract})
    narrative = eng.seal()
    forged = EngagementNarrative.from_wire(narrative.to_wire())
    forged.stages[1], forged.stages[2] = forged.stages[2], forged.stages[1]
    assert not forged.verify().valid


def test_tampered_head_is_caught():
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    forged = EngagementNarrative.from_wire(narrative.to_wire())
    forged.head_hash = "0" * 32
    assert not forged.verify().head_ok


def test_tampered_content_hash_is_caught():
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    forged = EngagementNarrative.from_wire(narrative.to_wire())
    forged.content_hash = "0" * 32
    assert not forged.verify().hash_ok


def test_tampered_underlying_artifact_is_caught():
    app, eng = _engagement()
    reserves = eng.attest_custody("vendor", {"omnibus": 120.0})
    owed = eng.attest_liabilities("vendor", {"acme": 40.0})
    eng.prove_solvency(reserves, owed)
    assert eng.verify(app.contract_signer).valid
    owed.liabilities_usd = 999.0  # tamper a captured artifact after sealing
    result = eng.verify(app.contract_signer)
    assert not result.digests_ok and not result.valid


def test_forged_signature_is_rejected():
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    stranger = HMACSigner("other-secret", key_id="stranger")
    assert narrative.verify(app.contract_signer).valid
    assert not narrative.verify(stranger).valid


def test_require_valid_raises_on_tamper():
    app, eng = _engagement()
    _negotiate(eng)
    narrative = eng.seal()
    narrative.require_valid(app.contract_signer)  # does not raise
    narrative.stages[0].summary["price_usd"] = -1.0
    with pytest.raises(SettlementError):
        narrative.require_valid(app.contract_signer)


# -- audit, persistence, edge cases -------------------------------------------


def test_engagement_lands_on_audit_chain():
    app, eng = _engagement()
    _negotiate(eng)
    eng.seal()
    entries = app.audit.query(action="cross_org_engagement")
    assert len(entries) == 1
    assert app.audit.verify_chain()


def test_narrative_wire_roundtrip():
    app, eng = _engagement()
    contract = _negotiate(eng)
    _deliver(eng, contract)
    narrative = eng.seal()
    restored = EngagementNarrative.from_wire(narrative.to_wire())
    assert restored.content_hash == narrative.content_hash
    assert restored.stage_names == narrative.stage_names
    assert restored.verify(app.contract_signer).valid


def test_no_agreement_raises_by_default():
    app, eng = _engagement()
    with pytest.raises(SettlementError):
        eng.negotiate(
            buyer=buyer_position(max_price_usd=0.01, max_sla_seconds=5.0),
            seller=seller_position(min_price_usd=5.0, ideal_price_usd=9.0),
        )


def test_no_agreement_recorded_when_not_required():
    app, eng = _engagement()
    result = eng.negotiate(
        buyer=buyer_position(max_price_usd=0.01, max_sla_seconds=5.0),
        seller=seller_position(min_price_usd=5.0, ideal_price_usd=9.0),
        require_agreement=False,
    )
    assert result.status != "agreement"
    narrative = eng.seal()
    assert narrative.stage_names == ["negotiate"]


def test_settle_without_contract_raises():
    app, eng = _engagement()
    with pytest.raises(SettlementError):
        eng.settle()


def test_stages_property_is_a_copy():
    app, eng = _engagement()
    _negotiate(eng)
    snapshot = eng.stages
    snapshot[0].digest = "mutated"
    # The live engagement is unaffected by mutating the returned copy.
    assert eng.seal().verify(app.contract_signer).valid


def test_resealing_after_more_stages_extends_the_narrative():
    app, eng = _engagement()
    contract = _negotiate(eng)
    first = eng.seal()
    assert first.stage_names == ["negotiate"]
    _deliver(eng, contract)
    second = eng.seal()
    assert second.stage_names == ["negotiate", "choreograph"]
    assert second.id != first.id
    assert second.verify(app.contract_signer).valid


def test_empty_engagement_seals_and_verifies():
    app, eng = _engagement()
    narrative = eng.seal()
    assert narrative.stages == []
    assert narrative.verify(app.contract_signer).valid


def test_stage_link_facts_exclude_timestamp():
    # Two stages with identical economic content but different timestamps must
    # produce the same link hash, so the chain is reproducible.
    a = EngagementStage(stage="settle", digest="abc", summary={"x": 1})
    b = EngagementStage(stage="settle", digest="abc", summary={"x": 1})
    assert a.compute_entry_hash() == b.compute_entry_hash()
