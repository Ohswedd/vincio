"""Cross-org collateralized settlement & escrow.

Backing an admission-required collateral fraction with a posted, content-bound escrow:
an ``Escrow`` binds the collateral to a specific ``Contract`` and counterparty into a
signed, offline-verifiable artifact, settling the contract releases the whole stake on a
fulfilled delivery and forfeits a bounded, pinpointed slice proportional to the shortfall
on a breach, and every post / release / forfeiture recomputes from the bytes alone and
lands on the hash-chained audit log — driven by the same settlement verdict the books
already close on.
"""

from __future__ import annotations

import pytest

from vincio import (
    ContextApp,
    Escrow,
    EscrowConfig,
    admit,
    post_escrow,
    settle_contract,
    settle_escrow,
)
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement.escrow import ESCROW_ACTION

ACME = HMACSigner("acme-key", key_id="acme")
VENDOR = HMACSigner("vendor-key", key_id="vendor")


def _contract(price: float = 1.0, *, buyer: str = "acme", seller: str = "vendor") -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope="transcribe", price_usd=price)
    ).seal()


def _app(name: str = "acme") -> ContextApp:
    app = ContextApp(name=name, provider=MockProvider(default_text="ok"))
    app.use_reputation_ledger()
    app.use_settlement_book()
    return app


# -- config validation --------------------------------------------------------


def test_default_config_is_coherent():
    EscrowConfig().validate_coherent()


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1])
def test_incoherent_config_raises(value):
    with pytest.raises(SettlementError):
        EscrowConfig(max_forfeit_fraction=value).validate_coherent()


# -- posting collateral -------------------------------------------------------


def test_post_from_admission_decision_holds_the_required_fraction():
    contract = _contract(price=2.0)
    decision = admit("vendor")  # newcomer → a positive escrow fraction
    escrow = post_escrow(contract, decision=decision)
    assert escrow.escrow_fraction == pytest.approx(decision.escrow_fraction)
    assert escrow.amount_usd == pytest.approx(round(decision.escrow_fraction * 2.0, 6))
    assert escrow.state == "posted"
    assert escrow.poster == "vendor"  # the admitted counterparty backs its delivery
    assert escrow.beneficiary == "acme"
    assert escrow.decision_id == decision.id
    assert escrow.verify().valid


def test_post_from_explicit_fraction():
    escrow = post_escrow(_contract(price=10.0), fraction=0.25)
    assert escrow.amount_usd == pytest.approx(2.5)
    assert escrow.escrow_fraction == pytest.approx(0.25)
    assert escrow.verify().valid


def test_post_from_flat_amount_is_authoritative():
    escrow = post_escrow(_contract(price=10.0), amount=3.0)
    assert escrow.amount_usd == pytest.approx(3.0)
    assert escrow.escrow_fraction == 0.0  # nothing to re-derive against
    assert escrow.verify().valid


def test_post_reads_admission_stamp_off_the_contract_terms():
    decision = admit("vendor")
    terms = decision.apply_to_terms(ContractTerms(scope="work", price_usd=5.0))
    contract = Contract(buyer="acme", seller="vendor", terms=terms).seal()
    escrow = post_escrow(contract)  # no explicit source — reads the stamp
    assert escrow.escrow_fraction == pytest.approx(decision.escrow_fraction)
    assert escrow.amount_usd == pytest.approx(round(decision.escrow_fraction * 5.0, 6))
    assert escrow.decision_id == decision.id


def test_post_without_a_source_raises():
    with pytest.raises(SettlementError):
        post_escrow(_contract())


def test_post_binds_the_specific_contract():
    contract = _contract()
    escrow = post_escrow(contract, fraction=0.3)
    assert escrow.contract_id == contract.id
    assert escrow.contract_hash == contract.content_hash


# -- release on a clean delivery ----------------------------------------------


def test_fulfilled_delivery_releases_the_whole_stake():
    contract = _contract(price=1.0)
    escrow = post_escrow(contract, fraction=0.4)
    record = settle_contract(contract, cost_usd=0.5)  # within the agreed price
    settle_escrow(escrow, record)
    assert escrow.is_released
    assert escrow.released_usd == pytest.approx(escrow.amount_usd)
    assert escrow.forfeited_usd == 0.0
    assert escrow.breaches == []
    assert escrow.settlement_hash == record.content_hash
    assert escrow.verify().valid


# -- forfeiture on a breach ---------------------------------------------------


def test_breach_forfeits_proportional_slice_and_releases_the_rest():
    contract = _contract(price=1.0)
    escrow = post_escrow(contract, fraction=0.5)  # $0.50 posted
    record = settle_contract(contract, cost_usd=1.5)  # 50% cost overrun
    settle_escrow(escrow, record)
    assert escrow.is_forfeited
    assert escrow.shortfall_fraction == pytest.approx(0.5)
    assert escrow.forfeited_usd == pytest.approx(0.25)  # half of the stake
    assert escrow.released_usd == pytest.approx(0.25)  # the remainder, never the whole
    assert 0.0 < escrow.forfeited_usd < escrow.amount_usd
    assert escrow.breaches == ["price"]
    assert escrow.verify().valid


def test_forfeiture_scales_with_severity():
    contract = _contract(price=1.0)
    small = post_escrow(contract, fraction=0.5)
    big = post_escrow(contract, fraction=0.5)
    settle_escrow(small, settle_contract(contract, cost_usd=1.1))  # 10% over
    settle_escrow(big, settle_contract(contract, cost_usd=1.8))  # 80% over
    assert big.forfeited_usd > small.forfeited_usd


def test_total_miss_capped_below_whole_stake_when_configured():
    contract = _contract(price=1.0)
    escrow = post_escrow(contract, fraction=0.5, config=EscrowConfig(max_forfeit_fraction=0.8))
    record = settle_contract(contract, cost_usd=3.0)  # 200% over → shortfall clamps to 1
    settle_escrow(escrow, record)
    # The cap guarantees a residual is always released to the poster.
    assert escrow.forfeited_usd == pytest.approx(0.8 * escrow.amount_usd)
    assert escrow.released_usd == pytest.approx(0.2 * escrow.amount_usd)
    assert escrow.released_usd > 0.0
    assert escrow.verify().valid


def test_sla_breach_pinpointed_and_forfeited():
    contract = Contract(
        buyer="acme",
        seller="vendor",
        terms=ContractTerms(scope="work", price_usd=1.0, sla_seconds=1.0),
    ).seal()
    escrow = post_escrow(contract, fraction=0.5)
    record = settle_contract(contract, cost_usd=0.5, latency_ms=1500.0)  # 50% over the 1s SLA
    settle_escrow(escrow, record)
    assert escrow.is_forfeited
    assert "sla" in escrow.breaches
    assert escrow.verify().valid


def test_cannot_resolve_twice():
    contract = _contract()
    escrow = post_escrow(contract, fraction=0.3)
    record = settle_contract(contract, cost_usd=0.5)
    settle_escrow(escrow, record)
    with pytest.raises(SettlementError):
        settle_escrow(escrow, record)


def test_cannot_resolve_against_a_different_contract():
    escrow = post_escrow(_contract(), fraction=0.3)
    other = settle_contract(_contract(), cost_usd=0.5)  # a different contract id
    with pytest.raises(SettlementError):
        settle_escrow(escrow, other)


# -- offline verification -----------------------------------------------------


def test_unsealed_escrow_is_invalid():
    escrow = Escrow(
        contract_id="c1",
        buyer="acme",
        seller="vendor",
        poster="vendor",
        beneficiary="acme",
        amount_usd=1.0,
    )
    result = escrow.verify()
    assert not result.valid
    assert "not sealed" in (result.reason or "")


def test_tampered_amount_caught_even_after_reseal():
    escrow = post_escrow(_contract(price=1.0), fraction=0.4)
    escrow.amount_usd = 999.0
    escrow.seal()  # recompute the hash to match the inflated amount
    result = escrow.verify()
    assert result.hash_ok  # hash matches the tampered field
    assert not result.terms_sound  # but the amount no longer re-derives from the fraction
    assert not result.valid


def test_tampered_forfeiture_caught_even_after_reseal():
    contract = _contract(price=1.0)
    escrow = post_escrow(contract, fraction=0.5)
    settle_escrow(escrow, settle_contract(contract, cost_usd=1.5))
    escrow.forfeited_usd = 0.0  # lie: forfeit nothing despite the breach
    escrow.released_usd = escrow.amount_usd
    escrow.seal()
    result = escrow.verify()
    assert result.hash_ok
    assert not result.terms_sound
    assert not result.valid


def test_tampered_state_without_reseal_breaks_the_hash():
    escrow = post_escrow(_contract(), fraction=0.3)
    escrow.state = "released"  # do not reseal
    assert not escrow.verify().hash_ok


def test_require_valid_raises_on_tamper():
    escrow = post_escrow(_contract(price=1.0), fraction=0.4)
    escrow.amount_usd = 5.0
    escrow.seal()
    with pytest.raises(SettlementError):
        escrow.require_valid()


def test_require_valid_returns_self_when_sound():
    escrow = post_escrow(_contract(), fraction=0.3)
    assert escrow.require_valid() is escrow


# -- signing ------------------------------------------------------------------


def test_sign_and_verify_with_a_verifier():
    escrow = post_escrow(_contract(), fraction=0.3).sign(VENDOR, party="vendor")
    result = escrow.verify(VENDOR, require=["vendor"])
    assert result.valid
    assert result.signatures_ok
    assert result.signed_by == ["vendor"]


def test_forged_signature_is_caught():
    escrow = post_escrow(_contract(), fraction=0.3).sign(VENDOR, party="vendor")
    result = escrow.verify(ACME, require=["vendor"])  # wrong key
    assert not result.signatures_ok
    assert not result.valid


def test_missing_required_signature_is_caught():
    escrow = post_escrow(_contract(), fraction=0.3)  # unsigned
    result = escrow.verify(VENDOR, require=["vendor"])
    assert not result.valid
    assert "vendor" in (result.reason or "")


def test_only_a_contract_party_can_sign():
    escrow = post_escrow(_contract(), fraction=0.3)
    with pytest.raises(SettlementError):
        escrow.sign(ACME, party="stranger")


def test_resolution_clears_stale_signatures():
    contract = _contract()
    escrow = post_escrow(contract, fraction=0.3).sign(VENDOR, party="vendor")
    assert escrow.signed_by == ["vendor"]
    settle_escrow(escrow, settle_contract(contract, cost_usd=0.5))
    # The content hash changed on resolution, so the posted-state signature is dropped.
    assert escrow.signatures == []
    assert escrow.verify().valid  # binding + disposition still verify offline


# -- determinism & wire -------------------------------------------------------


def test_same_collateral_and_outcome_hash_identically():
    contract = _contract(price=1.0)
    a = post_escrow(contract, fraction=0.4)
    b = post_escrow(contract, fraction=0.4)
    assert a.compute_hash() == b.compute_hash()


def test_wire_roundtrip_preserves_verification():
    contract = _contract(price=1.0)
    escrow = post_escrow(contract, fraction=0.5)
    settle_escrow(escrow, settle_contract(contract, cost_usd=1.5))
    restored = Escrow.from_wire(escrow.to_wire())
    assert restored.content_hash == escrow.content_hash
    assert restored.verify().valid
    assert restored.forfeited_usd == escrow.forfeited_usd


# -- app & book wiring --------------------------------------------------------


def test_app_post_escrow_signs_and_audits():
    app = _app()
    contract = _contract()
    decision = app.admit("vendor")
    escrow = app.post_escrow(contract, decision=decision)
    assert escrow.audit_id is not None
    assert escrow.signed_by == ["acme"]
    assert len(app.audit.query(action=ESCROW_ACTION)) == 1
    assert app.audit.verify_chain()
    assert escrow.verify(app.contract_signer, require=["acme"]).valid


def test_app_settle_resolves_an_attached_escrow_in_place():
    app = _app()
    contract = _contract(price=1.0)
    escrow = app.post_escrow(contract, fraction=0.5)
    record = app.settle(contract, cost_usd=1.5, escrow=escrow)  # a breach
    assert record.status == "breached"
    assert escrow.is_forfeited
    assert escrow.settlement_hash == record.content_hash
    # Both the posting and the forfeiture landed on the audit chain.
    assert len(app.audit.query(action=ESCROW_ACTION)) == 2
    assert app.audit.verify_chain()
    assert escrow.verify(app.contract_signer).valid


def test_app_settle_escrow_releases_on_clean_delivery():
    app = _app()
    contract = _contract(price=1.0)
    escrow = app.post_escrow(contract, fraction=0.5)
    record = app.settle(contract, cost_usd=0.5)
    resolved = app.settle_escrow(escrow, record)
    assert resolved.is_released
    assert resolved.released_usd == pytest.approx(resolved.amount_usd)
    assert app.settlement_book.verify(app.contract_signer).intact


def test_book_settle_with_escrow_keeps_the_book_intact():
    app = _app()
    contract = _contract(price=1.0)
    escrow = app.post_escrow(contract, fraction=0.4)
    app.settle(contract, cost_usd=2.0, escrow=escrow)
    assert app.settlement_book.verify().intact
    assert escrow.is_forfeited
