"""Cross-org custody liability attestation & proof-of-solvency.

A ``CustodyAttestation`` proves the capital a counterparty *holds*, but reserves are only one
side of the ledger — a counterparty solvent against one buyer's pledges may be under-water once
*every* obligation it owes is counted. A ``LiabilityAttestation`` makes the liability side
evidence-backed too, and ``prove_solvency`` folds the two proofs into a bounded, offline-verifiable
``SolvencyProof`` (reserves − liabilities) that ``guard_collateral(solvency=)`` reads as a
solvency-adjusted held figure — bounding a pledge against capital not already owed elsewhere and
pinpointing an ``InsolvencyBreach`` when the liabilities exceed the reserves.
"""

from __future__ import annotations

import pytest

from vincio import (
    CollateralLedger,
    ContextApp,
    InsolvencyBreach,
    LiabilityAttestation,
    LiabilityLine,
    SolvencyProof,
    attest_custody,
    attest_liabilities,
    guard_collateral,
    post_collateral_pool,
    prove_solvency,
)
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement.solvency import LIABILITY_ACTION, SOLVENCY_ACTION

ATTESTOR = HMACSigner("attestor-key", key_id="attestor")
CUSTODIAN = HMACSigner("custodian-key", key_id="custodian")
VENDOR = HMACSigner("vendor-key", key_id="vendor")
FORGER = HMACSigner("forger-key", key_id="forger")


def _app(name: str = "auditor") -> ContextApp:
    app = ContextApp(name=name, provider=MockProvider(default_text="ok"))
    app.use_settlement_book()
    return app


def _contract(
    scope: str, price: float = 100.0, *, buyer: str = "acme", seller: str = "vendor"
) -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope=scope, price_usd=price)
    ).seal()


# == liability attestation ====================================================

# -- construction & coercion --------------------------------------------------


def test_attest_liabilities_defaults_to_self_attested():
    att = attest_liabilities("vendor", 60.0)
    assert att.attestor == "vendor"  # self-attested
    assert att.poster == "vendor"
    assert att.self_attested
    assert att.liabilities_usd == pytest.approx(60.0)
    assert len(att.liabilities) == 1
    assert att.liabilities[0].creditor == "liabilities"


def test_third_party_attestor_is_not_self_attested():
    att = attest_liabilities("vendor", 60.0, attestor="auditor")
    assert att.attestor == "auditor"
    assert not att.self_attested


def test_liabilities_accept_mapping_list_and_tuples():
    from_map = attest_liabilities("vendor", {"acme": 40.0, "globex": 20.0})
    from_lines = attest_liabilities(
        "vendor", [LiabilityLine(creditor="acme", amount_usd=40.0), ("globex", 20.0)]
    )
    assert from_map.liabilities_usd == pytest.approx(60.0)
    assert from_lines.liabilities_usd == pytest.approx(60.0)
    assert from_map.creditors == ["acme", "globex"]


def test_total_re_derives_from_line_items():
    att = attest_liabilities("vendor", {"a": 10.0, "b": 20.0, "c": 5.0})
    assert att.liabilities_usd == pytest.approx(35.0)
    assert att.verify().liabilities_sound


def test_negative_obligation_is_refused():
    with pytest.raises(SettlementError, match="negative"):
        attest_liabilities("vendor", {"a": 10.0, "b": -5.0})


def test_empty_liabilities_prove_zero():
    att = attest_liabilities("vendor", [])
    assert att.liabilities_usd == pytest.approx(0.0)
    assert att.verify().valid


# -- signing ------------------------------------------------------------------


def test_only_the_attestor_signs():
    att = attest_liabilities("vendor", 60.0, attestor="auditor")
    att.sign(ATTESTOR)
    assert att.signed_by == ["auditor"]
    assert att.verify(ATTESTOR, require=["auditor"]).valid


def test_signing_as_a_non_attestor_is_refused():
    att = attest_liabilities("vendor", 60.0, attestor="auditor")
    with pytest.raises(SettlementError, match="signed by its attestor"):
        att.sign(VENDOR, party="vendor")


def test_re_signing_replaces_the_prior_signature():
    att = attest_liabilities("vendor", 60.0, attestor="auditor")
    att.sign(ATTESTOR)
    att.sign(ATTESTOR)
    assert len(att.signatures) == 1


# -- verification & tamper-evidence -------------------------------------------


def test_unsealed_attestation_is_invalid():
    att = attest_liabilities("vendor", 60.0)
    att.content_hash = ""
    result = att.verify()
    assert not result.valid and not result.hash_ok


def test_tampered_total_caught_even_after_reseal():
    att = attest_liabilities("vendor", {"acme": 60.0})
    att.liabilities_usd = 1.0  # under-state the debt
    assert not att.verify().hash_ok
    att.seal()  # re-seal the lie; the total no longer re-derives
    result = att.verify()
    assert result.hash_ok
    assert not result.liabilities_sound
    assert not result.valid


def test_tampered_line_item_caught():
    att = attest_liabilities("vendor", {"acme": 60.0})
    att.liabilities[0].amount_usd = 1.0  # lie about the obligation without touching the total
    att.seal()
    assert not att.verify().liabilities_sound


def test_forged_attestor_signature_refused_with_verifier():
    att = attest_liabilities("vendor", 60.0, attestor="auditor")
    att.sign(ATTESTOR)
    att.signatures[0].signature = FORGER.sign(att.content_hash)
    result = att.verify(ATTESTOR)
    assert not result.signatures_ok
    assert not result.valid


def test_require_valid_raises_on_tamper():
    att = attest_liabilities("vendor", 60.0)
    att.liabilities_usd = 1.0
    att.seal()
    with pytest.raises(SettlementError, match="failed verification"):
        att.require_valid()


# -- determinism & serialization ----------------------------------------------


def test_two_attestors_compute_the_same_hash():
    from datetime import UTC, datetime

    t = datetime(2026, 1, 1, tzinfo=UTC)
    a = attest_liabilities("vendor", {"acme": 40.0, "globex": 20.0}, as_of=t)
    b = attest_liabilities("vendor", [("globex", 20.0), ("acme", 40.0)], as_of=t)
    assert a.compute_hash() == b.compute_hash()  # order-independent


def test_wire_roundtrip_preserves_verification():
    att = attest_liabilities("vendor", {"acme": 60.0}, attestor="auditor")
    att.sign(ATTESTOR)
    restored = LiabilityAttestation.from_wire(att.to_wire())
    assert restored.content_hash == att.content_hash
    assert restored.verify(ATTESTOR).valid


def test_audit_details_are_json_safe():
    att = attest_liabilities("vendor", {"acme": 60.0}, attestor="auditor")
    details = att.audit_details()
    assert details["poster"] == "vendor"
    assert details["attestor"] == "auditor"
    assert details["self_attested"] is False
    assert details["liabilities_usd"] == pytest.approx(60.0)


# -- app / book integration ---------------------------------------------------


def test_app_attest_liabilities_signs_and_audits():
    app = _app("auditor")
    att = app.attest_liabilities("vendor", {"acme": 60.0})
    assert att.attestor == "auditor"
    assert att.signed_by == ["auditor"]
    assert att.audit_id is not None
    entries = app.audit.query(action=LIABILITY_ACTION)
    assert len(entries) == 1
    assert entries[0].decision == "attested"
    assert att.verify(app.contract_signer).valid


def test_app_self_attested_is_recorded_as_self():
    app = _app("vendor")
    att = app.attest_liabilities("vendor", 60.0)  # the app attests its own liabilities
    assert att.self_attested
    assert app.audit.query(action=LIABILITY_ACTION)[0].decision == "self_attested"


def test_book_attest_liabilities_signs_as_owner():
    app = _app("auditor")
    book = app.settlement_book
    att = book.attest_liabilities("vendor", {"acme": 60.0})
    assert att.attestor == "auditor"
    assert att.signed_by == ["auditor"]
    assert att.verify(book.signer, require=["auditor"]).valid


def test_app_attest_liabilities_third_party_not_signed_by_app():
    app = _app("acme")  # neither attestor nor poster
    att = app.attest_liabilities("vendor", 60.0, attestor="auditor")
    assert att.attestor == "auditor"
    assert att.signed_by == []


# == proof-of-solvency ========================================================


def _proofs(reserves, liabilities, *, poster: str = "vendor"):
    res = attest_custody(poster, reserves, custodian="custodian")
    liab = attest_liabilities(poster, liabilities, attestor="auditor")
    return res, liab


# -- folding & margin ---------------------------------------------------------


def test_solvent_margin_is_reserves_minus_liabilities():
    res, liab = _proofs(80.0, {"acme": 60.0})
    proof = prove_solvency(res, liab)
    assert proof.poster == "vendor"
    assert proof.custodian == "custodian"
    assert proof.attestor == "auditor"
    assert proof.reserves_usd == pytest.approx(80.0)
    assert proof.liabilities_usd == pytest.approx(60.0)
    assert proof.margin_usd == pytest.approx(20.0)
    assert proof.solvency_adjusted_held == pytest.approx(20.0)
    assert proof.solvent and not proof.insolvent
    assert proof.status == "solvent"
    assert proof.breach is None
    assert proof.custody_hash == res.content_hash
    assert proof.liability_hash == liab.content_hash
    assert proof.verify().valid


def test_insolvency_breach_is_pinpointed():
    res, liab = _proofs(50.0, {"acme": 120.0})
    proof = prove_solvency(res, liab)
    assert proof.insolvent and not proof.solvent
    assert proof.margin_usd == pytest.approx(-70.0)
    assert proof.solvency_adjusted_held == pytest.approx(0.0)  # floored: no free capital
    breach = proof.breach
    assert isinstance(breach, InsolvencyBreach)
    assert breach.poster == "vendor"
    assert breach.custodian == "custodian"
    assert breach.attestor == "auditor"
    assert breach.custody_hash == res.content_hash
    assert breach.liability_hash == liab.content_hash
    assert breach.reserves_usd == pytest.approx(50.0)
    assert breach.liabilities_usd == pytest.approx(120.0)
    assert breach.shortfall_usd == pytest.approx(70.0)
    assert proof.verify().valid


def test_exactly_solvent_has_no_breach():
    res, liab = _proofs(60.0, 60.0)
    proof = prove_solvency(res, liab)
    assert proof.solvent
    assert proof.margin_usd == pytest.approx(0.0)
    assert proof.solvency_adjusted_held == pytest.approx(0.0)
    assert proof.breach is None


# -- poster resolution --------------------------------------------------------


def test_mismatched_posters_require_explicit_poster():
    res = attest_custody("vendor", 80.0)
    liab = attest_liabilities("globex", 60.0)
    with pytest.raises(SettlementError, match="explicit poster"):
        prove_solvency(res, liab)


def test_explicit_poster_must_match_both_attestations():
    res = attest_custody("vendor", 80.0)
    liab = attest_liabilities("globex", 60.0)
    with pytest.raises(SettlementError, match="not the poster"):
        prove_solvency(res, liab, poster="vendor")


# -- refusing tampered / forged / mismatched ----------------------------------


def test_tampered_reserves_refused():
    res, liab = _proofs(80.0, 60.0)
    res.reserves_usd = 9_999.0
    res.seal()  # re-seal the lie; the total no longer re-derives
    with pytest.raises(SettlementError, match="tampered"):
        prove_solvency(res, liab)


def test_tampered_liabilities_refused():
    res, liab = _proofs(80.0, {"acme": 60.0})
    liab.liabilities_usd = 1.0
    liab.seal()
    with pytest.raises(SettlementError, match="tampered"):
        prove_solvency(res, liab)


def test_forged_attestor_refused_with_verifier():
    # The reserve proof is left unsigned, so the single verifier checks only the (forged)
    # liability attestor signature it is given.
    res = attest_custody("vendor", 80.0)
    liab = attest_liabilities("vendor", 60.0, attestor="auditor")
    liab.sign(ATTESTOR)
    liab.signatures[0].signature = FORGER.sign(liab.content_hash)
    with pytest.raises(SettlementError, match="invalid attestor signature"):
        prove_solvency(res, liab, verifier=ATTESTOR)


# -- verification & tamper-evidence -------------------------------------------


def test_unsealed_proof_is_invalid():
    res, liab = _proofs(80.0, 60.0)
    proof = prove_solvency(res, liab)
    proof.content_hash = ""
    result = proof.verify()
    assert not result.valid
    assert result.reason is not None and "not sealed" in result.reason


def test_tampered_margin_caught_even_after_reseal():
    res, liab = _proofs(80.0, 60.0)
    proof = prove_solvency(res, liab)
    proof.margin_usd = 9_999.0  # overstate the free capital
    assert not proof.verify().hash_ok
    proof.seal()  # recompute the hash to match the lie
    result = proof.verify()
    assert result.hash_ok
    assert not result.margin_sound  # the margin re-derives from the proven figures
    assert not result.valid


def test_flipped_verdict_caught():
    res, liab = _proofs(50.0, 120.0)
    proof = prove_solvency(res, liab)
    proof.breach = None  # hide the insolvency
    proof.seal()
    result = proof.verify()
    assert result.hash_ok
    assert not result.margin_sound
    assert not result.valid


def test_fabricated_breach_is_caught():
    res, liab = _proofs(80.0, 60.0)  # solvent
    proof = prove_solvency(res, liab)
    proof.breach = InsolvencyBreach(
        poster="vendor", reserves_usd=80.0, liabilities_usd=60.0, shortfall_usd=20.0
    )
    proof.seal()
    assert not proof.verify().margin_sound  # a breach with a positive margin is rejected


def test_solvency_require_valid_raises_on_tamper():
    res, liab = _proofs(80.0, 60.0)
    proof = prove_solvency(res, liab)
    proof.margin_usd = 1.0
    proof.seal()
    with pytest.raises(SettlementError, match="failed verification"):
        proof.require_valid()


def test_require_solvent_raises_when_insolvent():
    res, liab = _proofs(50.0, 120.0)
    proof = prove_solvency(res, liab)
    with pytest.raises(SettlementError, match="insolvent"):
        proof.require_solvent()


def test_require_solvent_returns_self_when_solvent():
    res, liab = _proofs(80.0, 60.0)
    proof = prove_solvency(res, liab)
    assert proof.require_solvent() is proof


# -- determinism & serialization ----------------------------------------------


def test_two_folders_compute_the_same_hash():
    res, liab = _proofs(80.0, 60.0)
    from datetime import UTC, datetime

    t = datetime(2026, 1, 1, tzinfo=UTC)
    a = prove_solvency(res, liab, as_of=t)
    b = prove_solvency(res, liab, as_of=t)
    assert a.compute_hash() == b.compute_hash()


def test_wire_roundtrip_preserves_breach():
    res, liab = _proofs(50.0, 120.0)
    proof = prove_solvency(res, liab)
    restored = SolvencyProof.from_wire(proof.to_wire())
    assert restored.content_hash == proof.content_hash
    assert restored.verify().valid
    assert restored.breach is not None
    assert restored.insolvent


def test_solvency_audit_details_are_json_safe():
    res, liab = _proofs(50.0, 120.0)
    details = prove_solvency(res, liab).audit_details()
    assert details["poster"] == "vendor"
    assert details["status"] == "insolvent"
    assert details["margin_usd"] == pytest.approx(-70.0)
    assert details["shortfall_usd"] == pytest.approx(70.0)


# -- app / book integration ---------------------------------------------------


def test_app_prove_solvency_signs_and_audits():
    app = _app("auditor")
    res = app.attest_custody("vendor", {"omnibus": 80.0})
    liab = app.attest_liabilities("vendor", {"acme": 60.0})
    proof = app.prove_solvency(res, liab)
    assert proof.signed_by == ["auditor"]
    assert proof.audit_id is not None
    assert proof.status == "solvent"
    entries = app.audit.query(action=SOLVENCY_ACTION)
    assert len(entries) == 1
    assert entries[0].decision == "solvent"
    assert proof.verify(app.contract_signer).valid


def test_book_prove_solvency_records_insolvent():
    app = _app("auditor")
    res = app.attest_custody("vendor", 50.0)
    liab = app.attest_liabilities("vendor", 120.0)
    proof = app.settlement_book.prove_solvency(res, liab)
    assert proof.insolvent
    assert any(
        e.resource == "vendor" and e.decision == "insolvent"
        for e in app.audit.query(action=SOLVENCY_ACTION)
    )


# == guard integration ========================================================


def test_solvency_proof_reads_as_solvency_adjusted_held():
    pool = post_collateral_pool([_contract("a", 100.0), _contract("b", 50.0)], fraction=0.1)
    # pledges 15; reserves 80 − liabilities 60 = 20 free, which covers 15.
    res, liab = _proofs(80.0, 60.0)
    ledger = guard_collateral([pool], solvency=prove_solvency(res, liab))
    assert ledger.reserves_proven
    assert ledger.solvency_adjusted
    assert ledger.held_usd == pytest.approx(20.0)  # the unencumbered capital
    assert ledger.reserves_usd == pytest.approx(20.0)
    assert ledger.gross_reserves_usd == pytest.approx(80.0)
    assert ledger.liabilities_usd == pytest.approx(60.0)
    assert ledger.solvency_margin_usd == pytest.approx(20.0)
    assert ledger.attestor == "auditor"
    assert not ledger.under_reserved  # 20 free covers 15 pledged
    assert not ledger.insolvent
    assert ledger.verify().valid


def test_under_reserved_when_liabilities_eat_the_free_capital():
    pool = post_collateral_pool([_contract("a", 100.0), _contract("b", 200.0)], fraction=0.1)
    # pledges 30; reserves 80 − liabilities 65 = 15 free, short of 30 pledged.
    res, liab = _proofs(80.0, 65.0)
    ledger = guard_collateral([pool], solvency=prove_solvency(res, liab))
    assert ledger.solvency_adjusted
    assert not ledger.insolvent  # solvent overall (margin 15 > 0) ...
    assert ledger.under_reserved  # ... but the free capital does not cover these pledges
    breach = ledger.reserve_breach
    assert breach is not None
    assert breach.reserves_usd == pytest.approx(15.0)  # the solvency-adjusted held figure
    assert breach.pledged_usd == pytest.approx(30.0)
    assert breach.shortfall_usd == pytest.approx(15.0)
    assert ledger.verify().valid


def test_insolvent_counterparty_has_no_free_capital():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.1)  # pledges 10
    res, liab = _proofs(50.0, 120.0)
    ledger = guard_collateral([pool], solvency=prove_solvency(res, liab))
    assert ledger.insolvent
    assert ledger.held_usd == pytest.approx(0.0)
    assert ledger.gross_reserves_usd == pytest.approx(50.0)
    assert ledger.under_reserved  # nothing backs the pledges
    assert ledger.over_committed
    assert ledger.verify().valid


def test_solvency_adjusted_held_bounds_beneficiary_claims():
    ba = _contract("ba", 100.0, buyer="acme")  # requires 10
    bb = _contract("bb", 300.0, buyer="globex")  # requires 30
    pool = post_collateral_pool([ba, bb], fraction=0.1)  # pledges 40 (10 acme : 30 globex)
    res, liab = _proofs(80.0, 60.0)  # 20 free
    ledger = guard_collateral([pool], solvency=prove_solvency(res, liab))
    assert ledger.under_reserved
    assert ledger.claim("acme").secured_usd == pytest.approx(5.0)  # 10/40 of 20
    assert ledger.claim("globex").secured_usd == pytest.approx(15.0)  # 30/40 of 20
    assert sum(c.secured_usd for c in ledger.claims) == pytest.approx(20.0)


def test_ledger_require_solvent_raises_when_insolvent():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.1)
    res, liab = _proofs(50.0, 120.0)
    ledger = guard_collateral([pool], solvency=prove_solvency(res, liab))
    with pytest.raises(SettlementError, match="insolvent"):
        ledger.require_solvent()


def test_ledger_require_solvent_returns_self_when_solvent():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.1)
    res, liab = _proofs(80.0, 60.0)
    ledger = guard_collateral([pool], solvency=prove_solvency(res, liab))
    assert ledger.require_solvent() is ledger


def test_held_and_solvency_are_mutually_exclusive():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.1)
    res, liab = _proofs(80.0, 60.0)
    with pytest.raises(SettlementError, match="one source"):
        guard_collateral([pool], held=10.0, solvency=prove_solvency(res, liab))


def test_custody_and_solvency_are_mutually_exclusive():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.1)
    res, liab = _proofs(80.0, 60.0)
    with pytest.raises(SettlementError, match="one source"):
        guard_collateral([pool], custody=res, solvency=prove_solvency(res, liab))


def test_tampered_solvency_proof_is_refused_at_guard():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.1)
    res, liab = _proofs(80.0, 60.0)
    proof = prove_solvency(res, liab)
    proof.margin_usd = 9_999.0
    proof.seal()  # re-seal the lie; the margin no longer re-derives
    with pytest.raises(SettlementError, match="tampered"):
        guard_collateral([pool], solvency=proof)


def test_solvency_proof_for_a_different_poster_is_refused():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.1)  # poster = vendor
    res, liab = _proofs(80.0, 60.0, poster="globex")
    proof = prove_solvency(res, liab)
    with pytest.raises(SettlementError, match="not the poster"):
        guard_collateral([pool], solvency=proof)


def test_forged_solvency_signature_is_refused_with_verifier():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.1)
    res, liab = _proofs(80.0, 60.0)
    proof = prove_solvency(res, liab)
    proof.sign(CUSTODIAN, party="custodian")
    proof.signatures[0].signature = FORGER.sign(proof.content_hash)
    with pytest.raises(SettlementError, match="invalid signature"):
        guard_collateral([pool], solvency=proof, verify_with=CUSTODIAN)


def test_ledger_hash_binds_the_solvency_proof():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.1)
    res, liab = _proofs(80.0, 60.0)
    proof = prove_solvency(res, liab)
    a = guard_collateral([pool], solvency=proof)
    b = guard_collateral([pool], solvency=proof)
    assert a.compute_hash() == b.compute_hash()
    other_res, other_liab = _proofs(80.0, 70.0)  # different liabilities → different margin
    other = guard_collateral([pool], solvency=prove_solvency(other_res, other_liab))
    assert other.compute_hash() != a.compute_hash()


def test_solvency_wire_roundtrip_preserves_fields():
    pool = post_collateral_pool([_contract("a", 100.0), _contract("b", 200.0)], fraction=0.1)
    res, liab = _proofs(80.0, 65.0)
    ledger = guard_collateral([pool], solvency=prove_solvency(res, liab))
    restored = CollateralLedger.from_wire(ledger.to_wire())
    assert restored.content_hash == ledger.content_hash
    assert restored.verify().valid
    assert restored.solvency_adjusted
    assert restored.reserve_breach is not None
    assert restored.liabilities_usd == pytest.approx(65.0)


def test_under_reserved_breach_re_derives_even_after_reseal():
    pool = post_collateral_pool([_contract("a", 100.0), _contract("b", 200.0)], fraction=0.1)
    res, liab = _proofs(80.0, 65.0)  # 15 free < 30 pledged
    ledger = guard_collateral([pool], solvency=prove_solvency(res, liab))
    ledger.reserve_breach.shortfall_usd = 0.0  # hide the breach
    ledger.seal()
    result = ledger.verify()
    assert result.hash_ok
    assert not result.terms_sound
    assert not result.valid


def test_tampered_solvency_margin_on_ledger_is_caught():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.1)
    res, liab = _proofs(80.0, 60.0)
    ledger = guard_collateral([pool], solvency=prove_solvency(res, liab))
    ledger.solvency_margin_usd = 999.0  # overstate the margin without touching the held figure
    ledger.seal()
    assert ledger.verify().hash_ok
    assert not ledger.verify().terms_sound  # held no longer reconciles to max(0, margin)


def test_app_guard_collateral_with_solvency_signs_and_audits():
    app = ContextApp(name="acme", provider=MockProvider(default_text="ok"))
    app.use_settlement_book()
    pool = app.post_collateral_pool(
        [_contract("a", 100.0), _contract("b", 200.0)], fraction=0.1
    )  # pledges 30
    res = app.attest_custody("vendor", {"omnibus": 80.0})
    liab = app.attest_liabilities("vendor", {"acme": 65.0})
    proof = app.prove_solvency(res, liab)
    ledger = app.guard_collateral([pool], solvency=proof)
    assert ledger.under_reserved
    assert ledger.solvency_adjusted
    assert ledger.audit_details()["under_reserved_usd"] == pytest.approx(15.0)
    assert ledger.audit_details()["solvency_margin_usd"] == pytest.approx(15.0)
    assert ledger.audit_id is not None
    assert ledger.verify(app.contract_signer).valid
    assert len(app.audit.query(action="rehypothecation")) == 1
    assert app.audit.verify_chain()


def test_summary_mentions_insolvent(capsys):
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.1)
    res, liab = _proofs(50.0, 120.0)
    ledger = guard_collateral([pool], solvency=prove_solvency(res, liab))
    ledger.print_summary()
    out = capsys.readouterr().out
    assert "insolvent" in out
