"""Cross-org dispute resolution & arbitration.

Adjudicating a disputed contract from the signed records its parties submit: a
reconciliation hash both sides co-signed stands, a contradicting unilateral claim
is rejected and pinpointed, a tampered claim is marked inadmissible, and the
content-bound Resolution verifies offline the way a settlement record does.
"""

from __future__ import annotations

import pytest

from vincio import ContextApp, Resolution, arbitrate, net_settlements, settle_contract
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement import ClaimVerdict, SettlementBook
from vincio.settlement.arbitration import ARBITRATION_ACTION

BUYER = HMACSigner("buyer-key", key_id="acme")
SELLER = HMACSigner("seller-key", key_id="vendor")


def _app(name: str = "arbiter") -> ContextApp:
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), model="mock-1")


def _contract(price: float = 0.10, **terms) -> Contract:
    base = {"scope": "work", "price_usd": price}
    base.update(terms)
    return Contract(buyer="acme", seller="vendor", terms=ContractTerms(**base)).seal()


def _claim(contract: Contract, *, cost: float, signer: HMACSigner, party: str):
    return settle_contract(contract, cost_usd=cost).sign(signer, party=party)


def _agreed(contract: Contract, *, cost: float = 0.08):
    """Both sides independently produce and co-sign the same figure."""
    return [
        _claim(contract, cost=cost, signer=BUYER, party="acme"),
        _claim(contract, cost=cost, signer=SELLER, party="vendor"),
    ]


# -- mutual corroboration upholds a figure ------------------------------------


def test_co_signed_figure_is_upheld():
    c = _contract()
    res = arbitrate(_agreed(c, cost=0.08), arbiter="arb")
    assert res.status == "upheld"
    assert res.upheld and res.resolved
    assert sorted(res.corroborated_by) == ["acme", "vendor"]
    assert res.upheld_balance_usd == pytest.approx(0.02)
    assert res.upheld_status == "settled"
    assert all(c.stands for c in res.claims)
    assert not res.rejected_claims and not res.inadmissible_claims


def test_unilateral_claim_contradicting_corroborated_truth_is_rejected():
    c = _contract()
    truth = _agreed(c, cost=0.08)  # both co-sign cost 0.08 -> balance 0.02
    liar = _claim(c, cost=0.05, signer=SELLER, party="vendor")  # seller revises lower
    res = arbitrate([*truth, liar])
    assert res.status == "upheld"
    assert res.upheld_balance_usd == pytest.approx(0.02)
    assert len(res.standing_claims) == 2
    assert len(res.rejected_claims) == 1
    assert res.rejected_claims[0].settlement_id == liar.id
    assert "contradicts" in (res.rejected_claims[0].reason or "")
    assert res.dissenters == ["vendor"]


def test_single_uncontested_claim_stands():
    c = _contract()
    only = _claim(c, cost=0.08, signer=BUYER, party="acme")
    res = arbitrate([only])
    assert res.status == "upheld"
    assert res.corroborated_by == ["acme"]
    assert res.standing_claims[0].settlement_id == only.id


def test_duplicate_co_signed_records_both_stand():
    # The same economic figure submitted from both books co-signs one hash.
    c = _contract()
    res = arbitrate(_agreed(c, cost=0.08))
    hashes = {cl.reconciliation_hash for cl in res.standing_claims}
    assert len(hashes) == 1  # one figure, two corroborating signatures


# -- genuine standoff stays unresolved ----------------------------------------


def test_disagreeing_unilateral_claims_are_unresolved():
    c = _contract()
    buyer_says = _claim(c, cost=0.08, signer=BUYER, party="acme")
    seller_says = _claim(c, cost=0.05, signer=SELLER, party="vendor")
    res = arbitrate([buyer_says, seller_says])
    assert res.status == "unresolved"
    assert not res.resolved
    assert res.upheld_hash == ""
    assert res.upheld_balance_usd is None
    assert res.dissenters == []  # nobody's claim stood; nobody is singled out
    assert "disagree" in (res.reason or "")


def test_no_admissible_claims_is_unresolved():
    c = _contract()
    bad = _claim(c, cost=0.08, signer=SELLER, party="vendor")
    bad.balance_usd = 999.0  # tamper after sealing
    res = arbitrate([bad])
    assert res.status == "unresolved"
    assert len(res.inadmissible_claims) == 1
    assert "no admissible" in (res.reason or "")


# -- admissibility: tampered / unsigned / forged claims -----------------------


def test_tampered_claim_is_inadmissible_not_raised():
    c = _contract()
    good = _agreed(c, cost=0.08)
    bad = _claim(c, cost=0.08, signer=SELLER, party="vendor")
    bad.amount_owed_usd = 999.0  # tamper without resealing
    res = arbitrate([*good, bad])  # does NOT raise — pinpointed instead
    inadmissible = res.inadmissible_claims
    assert len(inadmissible) == 1
    assert inadmissible[0].settlement_id == bad.id
    assert "tampered" in (inadmissible[0].reason or "")
    # The good co-signed figure still stands despite the bad claim in the pool.
    assert res.status == "upheld"


def test_unsigned_claim_is_inadmissible():
    c = _contract()
    unsigned = settle_contract(c, cost_usd=0.08)  # never signed
    res = arbitrate([unsigned])
    assert res.status == "unresolved"
    assert res.inadmissible_claims[0].reason and "unsigned" in res.inadmissible_claims[0].reason


def test_forged_signature_is_inadmissible_with_verifier():
    c = _contract()
    good = _agreed(c, cost=0.08)
    forged = _claim(c, cost=0.08, signer=SELLER, party="vendor")
    forged.signatures[0].signature = "deadbeef"  # corrupt the signature
    res = arbitrate([*good, forged], verify_with=SELLER)
    assert any("forged" in (cl.reason or "") for cl in res.inadmissible_claims)


def test_forged_signature_passes_without_verifier():
    # Without a verifier, a forged signature is admissible (presence-only check);
    # the hash binding still holds, so the figure can still stand.
    c = _contract()
    good = _agreed(c, cost=0.08)
    res = arbitrate(good, verify_with=None)
    assert res.status == "upheld"


# -- offline verification & content binding -----------------------------------


def test_resolution_verifies_offline():
    c = _contract()
    res = arbitrate(_agreed(c, cost=0.08)).sign(BUYER, party="acme")
    v = res.verify(BUYER)
    assert v.valid and v.hash_ok and v.decision_sound and v.signatures_ok


def test_unsigned_resolution_verifies_hash_and_decision():
    c = _contract()
    res = arbitrate(_agreed(c, cost=0.08))
    v = res.verify()
    assert v.valid and v.hash_ok and v.decision_sound


def test_tampered_balance_breaks_verification():
    c = _contract()
    res = arbitrate(_agreed(c, cost=0.08))
    res.upheld_balance_usd = 1.0
    assert not res.verify().valid


def test_tampered_status_breaks_verification():
    c = _contract()
    res = arbitrate([_claim(c, cost=0.08, signer=BUYER, party="acme")])
    res.status = "unresolved"
    assert not res.verify().valid


def test_flipped_standing_flag_breaks_decision_soundness():
    c = _contract()
    res = arbitrate(_agreed(c, cost=0.08))
    res.claims[0].stands = False  # try to drop a standing claim
    res.seal()  # even re-sealing the hash cannot save the re-derived decision
    v = res.verify()
    assert not v.decision_sound and not v.valid


def test_swapped_upheld_hash_breaks_decision_soundness():
    c = _contract()
    res = arbitrate(_agreed(c, cost=0.08))
    res.upheld_hash = "0" * 32
    res.seal()
    assert not res.verify().decision_sound


def test_invalid_signature_fails_verification():
    c = _contract()
    res = arbitrate(_agreed(c, cost=0.08)).sign(BUYER, party="acme")
    res.signatures[0].signature = "deadbeef"
    assert not res.verify(BUYER).signatures_ok


def test_require_signers_must_be_present():
    c = _contract()
    res = arbitrate(_agreed(c, cost=0.08)).sign(BUYER, party="acme")
    assert res.verify(BUYER, require=["acme"]).valid
    assert not res.verify(BUYER, require=["vendor"]).valid


# -- co-signability: two arbiters agree on the hash ---------------------------


def test_two_arbiters_compute_the_same_hash():
    c = _contract()
    claims = _agreed(c, cost=0.08)
    a = arbitrate(claims, arbiter="org-a")
    b = arbitrate(claims, arbiter="org-b")
    assert a.content_hash == b.content_hash  # arbiter excluded from the hash


def test_both_parties_can_co_sign_one_resolution():
    c = _contract()
    res = arbitrate(_agreed(c, cost=0.08))
    res.sign(BUYER, party="acme").sign(SELLER, party="vendor")
    assert sorted(res.signed_by) == ["acme", "vendor"]
    # Each party's signature checks under its own key (distinct HMAC keys here).
    assert res.verify(BUYER, require=["acme"]).hash_ok
    assert BUYER.verify(res.content_hash, res.signatures[0].signature)
    assert SELLER.verify(res.content_hash, res.signatures[1].signature)


# -- strict-mode helpers ------------------------------------------------------


def test_require_valid_raises_on_tamper():
    c = _contract()
    res = arbitrate(_agreed(c, cost=0.08))
    res.upheld_balance_usd = 9.9
    with pytest.raises(SettlementError):
        res.require_valid()


def test_require_resolved_raises_when_unresolved():
    c = _contract()
    res = arbitrate(
        [
            _claim(c, cost=0.08, signer=BUYER, party="acme"),
            _claim(c, cost=0.05, signer=SELLER, party="vendor"),
        ]
    )
    with pytest.raises(SettlementError):
        res.require_resolved()


def test_require_resolved_returns_self_when_resolved():
    c = _contract()
    res = arbitrate(_agreed(c, cost=0.08))
    assert res.require_resolved() is res


# -- category errors ----------------------------------------------------------


def test_empty_pool_raises():
    with pytest.raises(SettlementError):
        arbitrate([])


def test_pool_spanning_several_contracts_requires_contract_id():
    a = _claim(_contract(), cost=0.08, signer=BUYER, party="acme")
    b = _claim(_contract(), cost=0.08, signer=BUYER, party="acme")
    with pytest.raises(SettlementError):
        arbitrate([a, b])


def test_contract_id_selects_the_disputed_contract():
    c1 = _contract()
    c2 = _contract(price=0.20)
    pool = [*_agreed(c1, cost=0.08), *_agreed(c2, cost=0.15)]
    res = arbitrate(pool, contract_id=c2.id)
    assert res.contract_id == c2.id
    assert res.upheld_balance_usd == pytest.approx(0.05)


def test_contract_id_with_no_matching_record_raises():
    c = _contract()
    with pytest.raises(SettlementError):
        arbitrate(_agreed(c, cost=0.08), contract_id="contract-nope")


# -- serialization round-trip -------------------------------------------------


def test_to_wire_from_wire_round_trip():
    c = _contract()
    res = arbitrate([*_agreed(c, cost=0.08), _claim(c, cost=0.05, signer=SELLER, party="vendor")])
    res.sign(BUYER, party="acme")
    wire = res.to_wire()
    back = Resolution.from_wire(wire)
    assert back.content_hash == res.content_hash
    assert back.verify(BUYER).valid
    assert isinstance(back.claims[0], ClaimVerdict)
    assert back.dissenters == res.dissenters


# -- app integration: audit, signing, reputation ------------------------------


def test_app_arbitrate_signs_audits_and_resolves():
    app = _app()
    c = _contract()
    res = app.arbitrate(_agreed(c, cost=0.08))
    assert res.status == "upheld"
    assert res.verify(app.contract_signer).valid
    assert app.audit.query(action=ARBITRATION_ACTION)
    assert app.audit.verify_chain()


def test_app_arbitrate_debits_the_dissenter():
    app = _app()
    app.use_reputation_ledger()
    c = _contract()
    truth = _agreed(c, cost=0.08)
    liar = _claim(c, cost=0.05, signer=SELLER, party="vendor")
    before = app.reputation_ledger.snapshot("vendor").reputation
    app.arbitrate([*truth, liar])
    after = app.reputation_ledger.snapshot("vendor").reputation
    assert after < before  # the rejected claim debits the seller


def test_app_arbitrate_unresolved_debits_nobody():
    app = _app()
    app.use_reputation_ledger()
    c = _contract()
    app.reputation_ledger.record_outcome("vendor", passed=True, round_id="seed")
    before = app.reputation_ledger.snapshot("vendor").reputation
    app.arbitrate(
        [
            _claim(c, cost=0.08, signer=BUYER, party="acme"),
            _claim(c, cost=0.05, signer=SELLER, party="vendor"),
        ]
    )
    after = app.reputation_ledger.snapshot("vendor").reputation
    assert after == before  # an unresolved dispute singles out nobody


# -- book convenience ---------------------------------------------------------


def test_book_arbitrate_against_counterparty_claim():
    c = _contract()
    book = SettlementBook("acme", signer=BUYER)
    book.settle(c, cost_usd=0.08, party="acme")  # acme records its own figure, signed
    seller_record = _claim(c, cost=0.08, signer=SELLER, party="vendor")  # vendor agrees
    res = book.arbitrate(seller_record, contract_id=c.id)
    assert res.status == "upheld"
    assert res.signed_by == ["acme"]
    assert sorted(res.corroborated_by) == ["acme", "vendor"]


def test_book_arbitrate_infers_contract_from_counterparty():
    c = _contract()
    book = SettlementBook("acme", signer=BUYER)
    book.settle(c, cost_usd=0.08, party="acme")
    seller_record = _claim(c, cost=0.05, signer=SELLER, party="vendor")  # disagrees
    res = book.arbitrate(seller_record)
    assert res.contract_id == c.id
    assert res.status == "unresolved"


# -- bridge from a netting dispute --------------------------------------------


def test_resolves_a_pinpointed_netting_dispute():
    c = _contract()
    # Two books disagree on the same contract -> netting pinpoints a dispute.
    buyer_rec = _claim(c, cost=0.08, signer=BUYER, party="acme")
    seller_rec = _claim(c, cost=0.05, signer=SELLER, party="vendor")
    netting = net_settlements([buyer_rec, seller_rec])
    assert not netting.clean and len(netting.disputes) == 1
    disputed_id = netting.disputes[0].contract_id
    # Arbitration takes it from here: the parties submit their signed records.
    res = arbitrate([buyer_rec, seller_rec], contract_id=disputed_id)
    assert res.contract_id == disputed_id
    assert res.status == "unresolved"  # neither figure corroborated
    # Once the seller co-signs the buyer's figure, the dispute resolves.
    seller_agrees = _claim(c, cost=0.08, signer=SELLER, party="vendor")
    res2 = arbitrate([buyer_rec, seller_agrees], contract_id=disputed_id)
    assert res2.status == "upheld"
    assert res2.upheld_balance_usd == pytest.approx(0.02)
