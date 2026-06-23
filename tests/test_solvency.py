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

from datetime import UTC, datetime

import pytest

from vincio import (
    CollateralLedger,
    CompletenessProof,
    ContextApp,
    Discharge,
    EquivocationProof,
    HistoryConsistencyProof,
    HistoryConsistencyReport,
    InclusionProof,
    InsolvencyBreach,
    LiabilityAttestation,
    LiabilityLine,
    OmissionBreach,
    RootCommitment,
    RootConsistencyReport,
    SolvencyProof,
    attest_custody,
    attest_liabilities,
    check_completeness,
    check_history_consistency,
    check_root_consistency,
    discharge_liability,
    guard_collateral,
    post_collateral_pool,
    prove_equivocation,
    prove_solvency,
)
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement.solvency import (
    COMPLETENESS_ACTION,
    DISCHARGE_ACTION,
    EQUIVOCATION_ACTION,
    HISTORY_ACTION,
    LIABILITY_ACTION,
    SOLVENCY_ACTION,
)

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


# == liability inclusion proofs ===============================================

# -- the attestation's Merkle commitment --------------------------------------


def test_attestation_commits_a_merkle_root():
    att = attest_liabilities("vendor", {"acme": 60.0, "globex": 40.0})
    assert att.liabilities_root
    assert att.liabilities_root == att.compute_root()
    assert att.verify().valid and att.verify().liabilities_sound


def test_root_is_order_independent_but_content_dependent():
    a = attest_liabilities("vendor", [("acme", 60.0), ("globex", 40.0)])
    b = attest_liabilities("vendor", [("globex", 40.0), ("acme", 60.0)])
    assert a.liabilities_root == b.liabilities_root  # canonical creditor-sorted order
    c = attest_liabilities("vendor", {"acme": 60.0, "globex": 41.0})
    assert c.liabilities_root != a.liabilities_root  # a changed amount changes the root


def test_tampered_root_is_caught_even_after_reseal():
    att = attest_liabilities("vendor", {"acme": 60.0, "globex": 40.0})
    att.liabilities_root = "0" * 32  # forge the commitment ...
    att.content_hash = att.compute_hash()  # ... and re-seal the hash to match the lie
    result = att.verify()
    assert result.hash_ok  # the hash matches the forged facts
    assert not result.liabilities_sound  # but the root no longer re-derives
    assert not result.valid


def test_empty_attestation_has_a_well_defined_root():
    att = attest_liabilities("vendor", {})
    assert att.liabilities_root and att.liabilities_root == att.compute_root()
    assert att.verify().valid


# -- inclusion proofs ---------------------------------------------------------


def test_inclusion_proof_verifies_against_the_attestation():
    att = attest_liabilities("vendor", {"acme": 60.0, "globex": 40.0, "initech": 25.0})
    for creditor, amount in (("acme", 60.0), ("globex", 40.0), ("initech", 25.0)):
        proof = att.inclusion_proof(creditor)
        assert isinstance(proof, InclusionProof)
        assert proof.creditor == creditor
        assert proof.amount_usd == pytest.approx(amount)
        assert proof.liability_hash == att.content_hash
        assert proof.liabilities_root == att.liabilities_root
        result = proof.verify(att)
        assert result.valid and result.path_ok and result.bound_ok


def test_inclusion_proof_path_verifies_standalone():
    att = attest_liabilities("vendor", {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0})
    proof = att.inclusion_proof("c")
    # The path alone reconstructs the committed root, without the attestation in hand.
    assert proof.verify().path_ok


def test_tampered_inclusion_leaf_is_caught():
    att = attest_liabilities("vendor", {"acme": 60.0, "globex": 40.0})
    proof = att.inclusion_proof("acme")
    forged = InclusionProof.from_wire(proof.to_wire())
    forged.amount_usd = 9_999.0  # claim a bigger debt than the attested leaf
    assert not forged.verify().path_ok  # the path no longer reconstructs the root
    assert not forged.verify(att).valid


def test_inclusion_proof_root_from_another_attestation_is_refused():
    att = attest_liabilities("vendor", {"acme": 60.0, "globex": 40.0})
    other = attest_liabilities("vendor", {"acme": 60.0, "globex": 40.0, "extra": 1.0})
    proof = att.inclusion_proof("acme")
    lifted = InclusionProof.from_wire(proof.to_wire())
    lifted.liabilities_root = other.liabilities_root  # forge the committed root
    lifted.liability_hash = other.content_hash
    # The path no longer reconstructs the (forged) root, and it isn't bound to `att`.
    assert not lifted.verify(att).valid


def test_inclusion_proof_for_unknown_creditor_is_refused():
    att = attest_liabilities("vendor", {"acme": 60.0})
    with pytest.raises(SettlementError):
        att.inclusion_proof("nobody")


def test_inclusion_proof_for_ambiguous_creditor_is_refused():
    att = attest_liabilities("vendor", [("acme", 60.0), ("acme", 10.0)])
    with pytest.raises(SettlementError):
        att.inclusion_proof("acme")


def test_inclusion_proofs_lists_every_line():
    att = attest_liabilities("vendor", [("acme", 60.0), ("acme", 10.0), ("globex", 40.0)])
    proofs = att.inclusion_proofs()
    assert len(proofs) == 3
    assert all(p.verify(att).valid for p in proofs)


def test_forged_attestor_signature_invalidates_bound_inclusion_proof():
    att = attest_liabilities("vendor", {"acme": 60.0}, attestor="auditor").sign(ATTESTOR)
    proof = att.inclusion_proof("acme")
    assert proof.verify(att, ATTESTOR).valid
    assert not proof.verify(att, FORGER).valid  # the bound attestation fails signature check


def test_inclusion_require_valid_raises_on_tamper():
    att = attest_liabilities("vendor", {"acme": 60.0})
    proof = att.inclusion_proof("acme")
    proof.amount_usd = 1.0
    with pytest.raises(SettlementError):
        proof.require_valid()


def test_app_and_book_inclusion_proof():
    app = _app()
    owed = app.attest_liabilities("vendor", {"acme": 60.0, "globex": 40.0})
    proof = app.inclusion_proof(owed, "acme")
    assert proof.verify(owed, app.contract_signer).valid
    book_proof = app.settlement_book.inclusion_proof(owed, "globex")
    assert book_proof.verify(owed).valid


# == liability completeness ===================================================

# -- the completeness check ---------------------------------------------------


def test_complete_when_every_claim_is_attested():
    att = attest_liabilities("vendor", {"acme": 60.0, "globex": 40.0})
    check = check_completeness(att, {"acme": 60.0, "globex": 40.0})
    assert check.complete and check.status == "complete"
    assert not check.breaches
    assert check.attested_usd == pytest.approx(100.0)
    assert check.completed_usd == pytest.approx(100.0)
    assert check.understated_usd == pytest.approx(0.0)
    assert check.verify().valid


def test_omitted_creditor_surfaces_an_omission_breach():
    att = attest_liabilities("vendor", {"acme": 60.0})
    check = check_completeness(att, {"acme": 60.0, "globex": 40.0})
    assert not check.complete and check.status == "incomplete"
    assert check.omitted_creditors == ["globex"]
    (breach,) = check.breaches
    assert isinstance(breach, OmissionBreach)
    assert breach.creditor == "globex"
    assert breach.omitted is True
    assert breach.attested_usd == pytest.approx(0.0)
    assert breach.claimed_usd == pytest.approx(40.0)
    assert breach.understatement_usd == pytest.approx(40.0)
    assert check.completed_usd == pytest.approx(100.0)
    assert check.understated_usd == pytest.approx(40.0)


def test_under_stated_creditor_surfaces_an_omission_breach():
    att = attest_liabilities("vendor", {"acme": 60.0})
    check = check_completeness(att, {"acme": 90.0})  # acme can prove it is owed 90, not 60
    (breach,) = check.breaches
    assert breach.omitted is False  # listed, but under-stated
    assert breach.understatement_usd == pytest.approx(30.0)
    assert check.completed_usd == pytest.approx(90.0)


def test_completeness_accepts_liability_line_and_pair_claims():
    att = attest_liabilities("vendor", {"acme": 60.0})
    check = check_completeness(
        att, [LiabilityLine(creditor="globex", amount_usd=40.0), ("zeta", 5.0)]
    )
    assert check.omitted_creditors == ["globex", "zeta"]
    assert check.completed_usd == pytest.approx(105.0)


def test_completeness_derives_claims_from_settlement_records():
    # A creditor's own settled records back its claim: it delivered work it is owed for.
    seller = _app("globex")
    buyer_signer = HMACSigner("vendor-key", key_id="vendor")
    contract = _contract("delivery", price=40.0, buyer="vendor", seller="globex")
    record = seller.settle(contract, cost_usd=40.0)
    record.sign(buyer_signer, party="vendor")
    att = attest_liabilities("vendor", {"acme": 60.0})  # omits globex entirely
    check = seller.settlement_book.check_completeness(att)  # claims derived from records
    assert check.omitted_creditors == ["globex"]
    assert check.completed_usd == pytest.approx(100.0)


def test_negative_claim_is_refused():
    att = attest_liabilities("vendor", {"acme": 60.0})
    with pytest.raises(SettlementError):
        check_completeness(att, {"acme": -1.0})


def test_completeness_refuses_a_tampered_attestation():
    att = attest_liabilities("vendor", {"acme": 60.0})
    att.liabilities_usd = 1.0
    att.seal()
    with pytest.raises(SettlementError):
        check_completeness(att, {"acme": 60.0})


def test_completeness_forged_attestor_refused_with_verifier():
    att = attest_liabilities("vendor", {"acme": 60.0}, attestor="auditor").sign(ATTESTOR)
    att.signatures[0].signature = "deadbeef"  # forge the attestor signature
    with pytest.raises(SettlementError):
        check_completeness(att, {"acme": 60.0}, verifier=ATTESTOR)


# -- offline verification of the completeness check ---------------------------


def test_completeness_hash_recomputes_offline():
    att = attest_liabilities("vendor", {"acme": 60.0})
    check = check_completeness(att, {"acme": 60.0, "globex": 40.0})
    restored = CompletenessProof.from_wire(check.to_wire())
    assert restored.content_hash == check.content_hash
    assert restored.verify().valid
    assert restored.omitted_creditors == ["globex"]


def test_hidden_omission_is_caught_even_after_reseal():
    att = attest_liabilities("vendor", {"acme": 60.0})
    check = check_completeness(att, {"acme": 60.0, "globex": 40.0})
    check.breaches = []  # hide the omission ...
    check.completed_usd = check.attested_usd  # ... and lower the completed total
    check.seal()  # recompute the hash to match the lie
    result = check.verify()
    assert result.hash_ok  # the hash matches the forged facts
    assert not result.completeness_sound  # but the completed total no longer re-derives
    assert not result.valid


def test_tampered_completed_total_is_caught():
    att = attest_liabilities("vendor", {"acme": 60.0})
    check = check_completeness(att, {"acme": 60.0, "globex": 40.0})
    check.completed_usd = 9_999.0
    check.seal()
    assert not check.verify().completeness_sound


def test_fabricated_breach_with_no_understatement_is_caught():
    att = attest_liabilities("vendor", {"acme": 60.0})
    check = check_completeness(att, {"acme": 60.0})
    assert check.complete
    check.breaches = [
        OmissionBreach(
            poster="vendor",
            attestor="vendor",
            creditor="acme",
            attested_usd=60.0,
            claimed_usd=60.0,
            understatement_usd=0.0,
            omitted=False,
        )
    ]
    check.seal()
    assert not check.verify().completeness_sound  # a "breach" with no shortfall isn't real


def test_require_complete_raises_on_omission():
    att = attest_liabilities("vendor", {"acme": 60.0})
    check = check_completeness(att, {"acme": 60.0, "globex": 40.0})
    with pytest.raises(SettlementError):
        check.require_complete()


def test_require_complete_returns_self_when_complete():
    att = attest_liabilities("vendor", {"acme": 60.0})
    check = check_completeness(att, {"acme": 60.0})
    assert check.require_complete() is check


def test_completeness_audit_details_are_json_safe():
    att = attest_liabilities("vendor", {"acme": 60.0})
    check = check_completeness(att, {"acme": 60.0, "globex": 40.0})
    details = check.audit_details()
    import json

    json.dumps(details)
    assert details["status"] == "incomplete"
    assert details["omitted_creditors"] == ["globex"]


def test_completeness_summary_mentions_omission(capsys):
    att = attest_liabilities("vendor", {"acme": 60.0})
    check = check_completeness(att, {"acme": 60.0, "globex": 40.0})
    check.print_summary()
    out = capsys.readouterr().out
    assert "globex" in out and "omitted" in out


def test_app_check_completeness_signs_and_audits():
    app = _app()
    owed = app.attest_liabilities("vendor", {"acme": 60.0})
    check = app.check_completeness(owed, {"acme": 60.0, "globex": 40.0})
    assert check.audit_id is not None
    assert len(app.audit.query(action=COMPLETENESS_ACTION)) == 1
    assert check.verify(app.contract_signer, require=["auditor"]).valid


# == completeness folded into proof-of-solvency ===============================


def test_completeness_bounds_the_solvency_margin():
    res, liab = _proofs(200.0, {"acme": 60.0})  # attestor lists only 60 owed
    check = check_completeness(liab, {"acme": 60.0, "globex": 40.0})  # globex proves +40
    proof = prove_solvency(res, liab, completeness=check)
    assert proof.completeness_adjusted
    assert proof.attested_liabilities_usd == pytest.approx(60.0)
    assert proof.liabilities_usd == pytest.approx(100.0)  # completed total
    assert proof.understated_usd == pytest.approx(40.0)
    assert proof.margin_usd == pytest.approx(100.0)  # 200 − 100, not 200 − 60
    assert proof.verify().valid


def test_completeness_can_tip_a_proof_into_insolvency():
    res, liab = _proofs(80.0, {"acme": 60.0})  # solvent on the attestor's figure (margin 20)
    check = check_completeness(liab, {"acme": 60.0, "globex": 50.0})  # +50 hidden
    proof = prove_solvency(res, liab, completeness=check)
    assert proof.insolvent
    assert proof.breach is not None
    assert proof.breach.shortfall_usd == pytest.approx(30.0)  # 110 owed vs 80 held
    assert proof.solvency_adjusted_held == pytest.approx(0.0)


def test_complete_check_leaves_the_margin_unchanged():
    res, liab = _proofs(200.0, {"acme": 60.0, "globex": 40.0})
    check = check_completeness(liab, {"acme": 60.0, "globex": 40.0})  # nothing omitted
    proof = prove_solvency(res, liab, completeness=check)
    assert proof.completeness_adjusted  # the check was folded ...
    assert proof.understated_usd == pytest.approx(0.0)  # ... but raised nothing
    assert proof.liabilities_usd == pytest.approx(100.0)
    assert proof.margin_usd == pytest.approx(100.0)


def test_completeness_for_a_different_attestation_is_refused():
    res, liab = _proofs(200.0, {"acme": 60.0})
    other = attest_liabilities("vendor", {"acme": 60.0, "globex": 40.0})
    check = check_completeness(other, {"acme": 60.0})  # bound to `other`, not `liab`
    with pytest.raises(SettlementError):
        prove_solvency(res, liab, completeness=check)


def test_completeness_for_a_different_poster_is_refused():
    res = attest_custody("vendor", 200.0, custodian="custodian")
    liab = attest_liabilities("vendor", {"acme": 60.0}, attestor="auditor")
    other_poster = attest_liabilities("globex", {"acme": 60.0}, attestor="auditor")
    check = check_completeness(other_poster, {"acme": 90.0})
    with pytest.raises(SettlementError):
        prove_solvency(res, liab, completeness=check)


def test_tampered_completeness_is_refused_at_prove_solvency():
    res, liab = _proofs(200.0, {"acme": 60.0})
    check = check_completeness(liab, {"acme": 90.0})
    check.completed_usd = 60.0  # hide the understatement ...
    check.seal()  # ... and re-seal
    with pytest.raises(SettlementError):
        prove_solvency(res, liab, completeness=check)


def test_solvency_completed_figure_cannot_drop_below_attested():
    # The completed liabilities can only *raise* the attestor's figure. A forged proof that
    # lowers the completed total below the attested figure no longer re-derives.
    res, liab = _proofs(200.0, {"acme": 60.0})
    check = check_completeness(liab, {"acme": 60.0, "globex": 40.0})
    proof = prove_solvency(res, liab, completeness=check)
    proof.liabilities_usd = 50.0  # below the attested 60 ...
    proof.margin_usd = 150.0
    proof.seal()  # ... and re-seal
    assert not proof.verify().margin_sound


def test_completeness_folds_through_app_and_guard():
    app = _app()
    res = app.attest_custody("vendor", 200.0)
    owed = app.attest_liabilities("vendor", {"acme": 60.0})
    check = app.check_completeness(owed, {"acme": 60.0, "globex": 40.0})
    proof = app.prove_solvency(res, owed, completeness=check)
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.5)  # pledges 50
    ledger = app.guard_collateral([pool], solvency=proof)
    assert ledger.solvency_adjusted
    assert ledger.held_usd == pytest.approx(100.0)  # 200 − 100 completed, not 200 − 60
    assert ledger.liabilities_usd == pytest.approx(100.0)
    assert ledger.gross_reserves_usd == pytest.approx(200.0)
    assert ledger.verify(app.contract_signer).valid


def test_solvency_wire_roundtrip_preserves_completeness_fields():
    res, liab = _proofs(200.0, {"acme": 60.0})
    check = check_completeness(liab, {"acme": 60.0, "globex": 40.0})
    proof = prove_solvency(res, liab, completeness=check)
    restored = SolvencyProof.from_wire(proof.to_wire())
    assert restored.completeness_hash == check.content_hash
    assert restored.completeness_adjusted
    assert restored.attested_liabilities_usd == pytest.approx(60.0)
    assert restored.verify().valid


# == liability non-equivocation & root consistency ============================

# A poster issues its liability attestation per relationship, so it can sign a *smaller* root for
# one creditor and a different one for another — each creditor's inclusion proof verifying against
# the root it was shown while the totals disagree. Completeness catches an omission only when the
# omitted creditor folds its own claim; equivocation is caught by comparing the signed roots.

_AS_OF = datetime(2026, 1, 1, tzinfo=UTC)
_AS_OF2 = datetime(2026, 1, 2, tzinfo=UTC)


def _signed(poster, liabilities, *, attestor="auditor", as_of=_AS_OF, signer=ATTESTOR):
    att = attest_liabilities(poster, liabilities, attestor=attestor, as_of=as_of)
    if signer is not None:
        att.sign(signer)
    return att


# -- root commitments ---------------------------------------------------------


def test_root_commitment_carries_signed_root_without_line_items():
    att = _signed("vendor", {"acme": 60.0, "globex": 40.0})
    commitment = att.root_commitment()
    assert commitment.poster == "vendor"
    assert commitment.attestor == "auditor"
    assert commitment.liabilities_root == att.liabilities_root
    assert commitment.liability_hash == att.content_hash
    assert commitment.liabilities_usd == pytest.approx(100.0)
    assert commitment.as_of == _AS_OF
    # The commitment shares the root and total, never the per-creditor line items.
    assert "acme" not in commitment.model_dump_json()
    assert "globex" not in commitment.model_dump_json()


def test_root_commitment_verifies_and_refuses_a_forged_signature():
    att = _signed("vendor", {"acme": 60.0})
    commitment = att.root_commitment()
    assert commitment.verify(ATTESTOR).valid
    assert commitment.verify(ATTESTOR).signed_by == ["auditor"]
    # A signature minted with a different key over the same hash does not verify.
    assert not commitment.verify(FORGER).valid
    assert commitment.verify(FORGER).reason


def test_unsigned_root_commitment_is_committed_but_unattributed():
    att = attest_liabilities("vendor", {"acme": 60.0}, attestor="auditor").seal()
    commitment = att.root_commitment()
    assert commitment.signature is None
    result = commitment.verify(ATTESTOR)
    assert result.committed and result.valid
    assert result.signed_by == []  # nothing attributed to the attestor


def test_root_commitment_conflict_detection():
    a = _signed("vendor", {"acme": 60.0}).root_commitment()
    b = _signed("vendor", {"globex": 40.0}).root_commitment()
    same = _signed("vendor", {"acme": 60.0}).root_commitment()
    later = _signed("vendor", {"globex": 40.0}, as_of=_AS_OF2).root_commitment()
    assert a.conflicts_with(b)  # same (poster, attestor, as_of), different root
    assert b.conflicts_with(a)  # symmetric
    assert not a.conflicts_with(same)  # identical root is not a conflict
    assert not a.conflicts_with(later)  # a different instant is a distinct snapshot
    assert a.consistency_key == ("vendor", "auditor", _AS_OF.isoformat())


def test_root_commitment_wire_roundtrip():
    commitment = _signed("vendor", {"acme": 60.0}).root_commitment()
    restored = RootCommitment.from_wire(commitment.to_wire())
    assert restored.verify(ATTESTOR).valid
    assert restored.liabilities_root == commitment.liabilities_root


# -- the equivocation proof ---------------------------------------------------


def test_prove_equivocation_builds_a_valid_proof():
    a = _signed("vendor", {"acme": 60.0})
    b = _signed("vendor", {"globex": 40.0})
    proof = prove_equivocation(
        a, b, verifier=ATTESTOR, first_creditor="acme", second_creditor="globex"
    )
    result = proof.verify(ATTESTOR)
    assert result.valid and result.attestor_signed and result.conflict_ok
    assert proof.poster == "vendor"
    assert proof.attestor == "auditor"
    assert proof.as_of == _AS_OF
    assert proof.liabilities_gap_usd == pytest.approx(20.0)
    assert set(proof.roots) == {a.liabilities_root, b.liabilities_root}
    assert set(proof.creditors) == {"acme", "globex"}


def test_equivocation_proof_is_canonical_regardless_of_input_order():
    a = _signed("vendor", {"acme": 60.0})
    b = _signed("vendor", {"globex": 40.0})
    one = prove_equivocation(
        a, b, verifier=ATTESTOR, first_creditor="acme", second_creditor="globex"
    )
    two = prove_equivocation(
        b, a, verifier=ATTESTOR, first_creditor="globex", second_creditor="acme"
    )
    # Canonical content-hash ordering means the same conflict yields the same proof either way.
    assert one.content_hash == two.content_hash
    assert one.first_hash == two.first_hash
    assert one.first_creditor == two.first_creditor


def test_equivocation_proof_without_verifier_is_valid_but_unattested():
    a = _signed("vendor", {"acme": 60.0})
    b = _signed("vendor", {"globex": 40.0})
    proof = prove_equivocation(a, b)
    result = proof.verify()  # no verifier — signatures not checked
    assert result.valid and result.conflict_ok
    assert not result.attestor_signed  # attribution requires the verifier


def test_prove_equivocation_refuses_a_forged_conflicting_root():
    honest = _signed("vendor", {"acme": 60.0})
    forged = _signed("vendor", {"globex": 40.0}, signer=FORGER)  # signed with the wrong key
    with pytest.raises(SettlementError, match="invalid attestor signature"):
        prove_equivocation(honest, forged, verifier=ATTESTOR)


def test_prove_equivocation_refuses_an_unsigned_root_with_a_verifier():
    honest = _signed("vendor", {"acme": 60.0})
    unsigned = attest_liabilities("vendor", {"globex": 40.0}, attestor="auditor").seal()
    with pytest.raises(SettlementError, match="not signed by its attestor"):
        prove_equivocation(honest, unsigned, verifier=ATTESTOR)


def test_prove_equivocation_refuses_a_tampered_attestation():
    honest = _signed("vendor", {"acme": 60.0})
    tampered = _signed("vendor", {"globex": 40.0})
    tampered.liabilities_usd = 1.0  # total no longer re-derives
    with pytest.raises(SettlementError, match="tampered"):
        prove_equivocation(honest, tampered, verifier=ATTESTOR)


def test_prove_equivocation_refuses_different_posters():
    a = _signed("vendor", {"acme": 60.0})
    b = _signed("supplier", {"globex": 40.0})
    with pytest.raises(SettlementError, match="different posters"):
        prove_equivocation(a, b, verifier=ATTESTOR)


def test_prove_equivocation_refuses_different_instants():
    a = _signed("vendor", {"acme": 60.0}, as_of=_AS_OF)
    b = _signed("vendor", {"globex": 40.0}, as_of=_AS_OF2)
    with pytest.raises(SettlementError, match="different instants"):
        prove_equivocation(a, b, verifier=ATTESTOR)


def test_prove_equivocation_refuses_identical_roots():
    a = _signed("vendor", {"acme": 60.0})
    b = _signed("vendor", {"acme": 60.0})  # same lines → same root
    with pytest.raises(SettlementError, match="same root"):
        prove_equivocation(a, b, verifier=ATTESTOR)


def test_equivocation_proof_hash_recomputes_and_catches_tamper():
    proof = prove_equivocation(
        _signed("vendor", {"acme": 60.0}), _signed("vendor", {"globex": 40.0})
    )
    assert proof.verify().hash_ok
    proof.first_root = "deadbeef"  # tamper with a recorded root
    result = proof.verify()
    assert not result.valid
    assert not result.hash_ok or not result.conflict_ok


def test_equivocation_proof_reporter_signature():
    # The reporter co-signs with the fabric verification secret (the shared-key convention the
    # settlement records use), so one verifier checks both the attestor and reporter signatures.
    reporter = HMACSigner("attestor-key", key_id="acme")
    proof = prove_equivocation(
        _signed("vendor", {"acme": 60.0}), _signed("vendor", {"globex": 40.0})
    )
    proof.sign(reporter, party="acme")  # the creditor lodging the accusation
    assert proof.verify(ATTESTOR, require=["acme"]).valid
    # Validity does not require the reporter's signature when none is demanded.
    bare = prove_equivocation(
        _signed("vendor", {"acme": 60.0}), _signed("vendor", {"globex": 40.0})
    )
    assert bare.verify(ATTESTOR).valid


def test_equivocation_proof_require_valid_raises_on_tampered_pairing():
    proof = prove_equivocation(
        _signed("vendor", {"acme": 60.0}), _signed("vendor", {"globex": 40.0})
    )
    proof.second = proof.first  # collapse the conflict
    with pytest.raises(SettlementError, match="failed verification"):
        proof.require_valid()


def test_equivocation_proof_wire_roundtrip():
    proof = prove_equivocation(
        _signed("vendor", {"acme": 60.0}), _signed("vendor", {"globex": 40.0}), verifier=ATTESTOR
    )
    restored = EquivocationProof.from_wire(proof.to_wire())
    assert restored.verify(ATTESTOR).valid
    assert restored.liabilities_gap_usd == pytest.approx(20.0)


def test_equivocation_audit_details_are_json_safe():
    import json

    proof = prove_equivocation(
        _signed("vendor", {"acme": 60.0}), _signed("vendor", {"globex": 40.0})
    )
    payload = proof.audit_details()
    assert json.loads(json.dumps(payload))["poster"] == "vendor"
    assert payload["liabilities_gap_usd"] == pytest.approx(20.0)


def test_equivocation_summary_mentions_the_two_roots(capsys):
    proof = prove_equivocation(
        _signed("vendor", {"acme": 60.0}),
        _signed("vendor", {"globex": 40.0}),
        first_creditor="acme",
        second_creditor="globex",
    )
    proof.print_summary()
    out = capsys.readouterr().out
    assert "vendor" in out and "root A" in out and "root B" in out


# -- the root-consistency scan ------------------------------------------------


def test_check_root_consistency_passes_an_honest_set():
    a = _signed("vendor", {"acme": 60.0})
    same = _signed("vendor", {"acme": 60.0})  # the same root shown to two creditors
    report = check_root_consistency([("acme", a), ("globex", same)], verifier=ATTESTOR)
    assert report.consistent
    assert report.checked == 2
    assert report.keys == 1
    assert report.equivocations == []


def test_check_root_consistency_surfaces_an_equivocation():
    a = _signed("vendor", {"acme": 60.0})
    b = _signed("vendor", {"globex": 40.0})
    report = check_root_consistency([("acme", a), ("globex", b)], verifier=ATTESTOR)
    assert not report.consistent
    assert len(report.equivocations) == 1
    assert report.equivocating_posters == ["vendor"]
    assert report.equivocations[0].verify(ATTESTOR).valid


def test_check_root_consistency_minimal_proofs_for_three_roots():
    atts = [
        ("acme", _signed("vendor", {"acme": 60.0})),
        ("globex", _signed("vendor", {"globex": 40.0})),
        ("initech", _signed("vendor", {"initech": 25.0})),
    ]
    report = check_root_consistency(atts, verifier=ATTESTOR)
    # Three distinct roots for one (poster, attestor, as_of) → k−1 = 2 chaining proofs.
    assert len(report.equivocations) == 2
    assert all(p.verify(ATTESTOR).valid for p in report.equivocations)


def test_check_root_consistency_excludes_forged_roots_from_evidence():
    honest = _signed("vendor", {"acme": 60.0})
    forged = _signed("vendor", {"globex": 40.0}, signer=FORGER)
    # With the verifier the forged root is inadmissible, so no false accusation is raised.
    report = check_root_consistency([("acme", honest), ("globex", forged)], verifier=ATTESTOR)
    assert report.consistent
    assert report.checked == 1  # only the honest one counted


def test_check_root_consistency_excludes_tampered_roots():
    honest = _signed("vendor", {"acme": 60.0})
    tampered = _signed("vendor", {"globex": 40.0})
    tampered.liabilities_usd = 999.0  # total no longer re-derives
    report = check_root_consistency([honest, tampered], verifier=ATTESTOR)
    assert report.consistent
    assert report.checked == 1


def test_check_root_consistency_separates_posters_and_instants():
    atts = [
        _signed("vendor", {"acme": 60.0}, as_of=_AS_OF),
        _signed("vendor", {"globex": 40.0}, as_of=_AS_OF),  # equivocation for vendor@AS_OF
        _signed("supplier", {"acme": 10.0}, as_of=_AS_OF),  # lone snapshot, no conflict
        _signed("vendor", {"acme": 60.0}, as_of=_AS_OF2),  # later snapshot, no conflict
    ]
    report = check_root_consistency(atts, verifier=ATTESTOR)
    assert report.equivocating_posters == ["vendor"]
    assert len(report.equivocations) == 1
    assert report.keys == 3  # vendor@1, supplier@1, vendor@2


def test_require_consistent_raises_and_returns_self():
    consistent = check_root_consistency([_signed("vendor", {"acme": 60.0})], verifier=ATTESTOR)
    assert consistent.require_consistent() is consistent
    bad = check_root_consistency(
        [_signed("vendor", {"acme": 60.0}), _signed("vendor", {"globex": 40.0})], verifier=ATTESTOR
    )
    with pytest.raises(SettlementError, match="inconsistent"):
        bad.require_consistent()


def test_root_consistency_report_wire_roundtrip():
    report = check_root_consistency(
        [_signed("vendor", {"acme": 60.0}), _signed("vendor", {"globex": 40.0})], verifier=ATTESTOR
    )
    restored = RootConsistencyReport.from_wire(report.to_wire())
    assert not restored.consistent
    assert restored.equivocations[0].verify(ATTESTOR).valid


def test_check_root_consistency_rejects_malformed_items():
    with pytest.raises(SettlementError, match="must be LiabilityAttestation"):
        check_root_consistency(["not-an-attestation"])


# -- app / book wiring --------------------------------------------------------


def test_app_check_root_consistency_audits_and_dings_reputation():
    app = _app("auditor")
    app.use_reputation_ledger()
    a = app.attest_liabilities("vendor", {"acme": 60.0}, as_of=_AS_OF)
    b = app.attest_liabilities("vendor", {"globex": 40.0}, as_of=_AS_OF)
    report = app.check_root_consistency(
        [("acme", a), ("globex", b)], verify_with=app.contract_signer
    )
    assert not report.consistent
    assert len(app.audit.query(action=EQUIVOCATION_ACTION)) == 1
    assert app.audit.verify_chain()
    assert app.reputation_ledger.weight("vendor") < 1.0  # equivocation counts as a failure
    assert report.equivocations[0].audit_id is not None


def test_book_check_root_consistency_audits_and_dings_reputation():
    from vincio.optimize.reputation import ReputationLedger
    from vincio.security.audit import AuditLog
    from vincio.settlement.book import SettlementBook

    ledger = ReputationLedger()
    book = SettlementBook(owner="auditor", signer=ATTESTOR, audit=AuditLog(), reputation=ledger)
    a = book.attest_liabilities("vendor", {"acme": 60.0}, attestor="auditor", as_of=_AS_OF)
    b = book.attest_liabilities("vendor", {"globex": 40.0}, attestor="auditor", as_of=_AS_OF)
    report = book.check_root_consistency([("acme", a), ("globex", b)], verify_with=ATTESTOR)
    assert not report.consistent
    assert len(book.audit.query(action=EQUIVOCATION_ACTION)) == 1
    assert ledger.weight("vendor") < 1.0


def test_app_check_root_consistency_dings_each_poster_once():
    # Three conflicting roots for one poster produce two pairwise proofs but a single
    # reputation debit — three roots are one equivocating counterparty, audited per conflict.
    app = _app("auditor")
    app.use_reputation_ledger()
    atts = [
        ("acme", app.attest_liabilities("vendor", {"acme": 60.0}, as_of=_AS_OF)),
        ("globex", app.attest_liabilities("vendor", {"globex": 40.0}, as_of=_AS_OF)),
        ("initech", app.attest_liabilities("vendor", {"initech": 25.0}, as_of=_AS_OF)),
    ]
    report = app.check_root_consistency(atts, verify_with=app.contract_signer)
    assert len(report.equivocations) == 2  # k − 1 pairwise proofs
    assert len(app.audit.query(action=EQUIVOCATION_ACTION)) == 2  # every conflict audited
    # A single failure recorded for the poster, not one per pairwise proof.
    snapshot = app.reputation_ledger.snapshot("vendor")
    assert snapshot.failures == pytest.approx(1.0)
    assert snapshot.rounds == 1


def test_app_check_root_consistency_can_skip_side_effects():
    app = _app("auditor")
    app.use_reputation_ledger()
    a = app.attest_liabilities("vendor", {"acme": 60.0}, as_of=_AS_OF)
    b = app.attest_liabilities("vendor", {"globex": 40.0}, as_of=_AS_OF)
    report = app.check_root_consistency(
        [a, b], verify_with=app.contract_signer, record_reputation=False, record_audit=False
    )
    assert not report.consistent
    assert len(app.audit.query(action=EQUIVOCATION_ACTION)) == 0
    assert app.reputation_ledger.weight("vendor") == pytest.approx(
        app.reputation_ledger.weight("x")
    )


# == liability history consistency & snapshot monotonicity ====================

# The cross-party signing convention the settlement records use: every party signs with the shared
# fabric verification secret, distinguished only by ``key_id`` (its identity), so one verifier
# checks the attestor's and the creditors' signatures alike.
_AS_OF3 = datetime(2026, 1, 3, tzinfo=UTC)
_ACME = HMACSigner("attestor-key", key_id="acme")
_GLOBEX = HMACSigner("attestor-key", key_id="globex")


def _history(poster="vendor", attestor="auditor"):
    """A three-snapshot linked history where acme's $100 obligation is constant."""
    s1 = _signed(poster, {"acme": 100.0, "globex": 40.0}, attestor=attestor, as_of=_AS_OF)
    s2 = attest_liabilities(
        poster, {"acme": 100.0, "globex": 40.0}, attestor=attestor, as_of=_AS_OF2, prior=s1
    ).sign(ATTESTOR)
    s3 = attest_liabilities(
        poster, {"acme": 100.0, "globex": 40.0}, attestor=attestor, as_of=_AS_OF3, prior=s2
    ).sign(ATTESTOR)
    return s1, s2, s3


# -- linked liability history -------------------------------------------------


def test_link_binds_prior_into_the_signed_hash():
    s1 = _signed("vendor", {"acme": 100.0})
    s2 = attest_liabilities("vendor", {"acme": 100.0}, attestor="auditor", as_of=_AS_OF2, prior=s1)
    assert s2.has_prior
    assert s2.prior_hash == s1.content_hash
    assert s2.prior_root == s1.liabilities_root
    assert s2.prior_as_of == s1.as_of
    # The link is bound into the content hash: re-pointing the predecessor breaks verification.
    s2.prior_hash = "deadbeef"
    assert not s2.verify().hash_ok


def test_no_prior_hashes_identically_to_an_unlinked_attestation():
    # The link is additive: an attestation with no predecessor hashes exactly as one issued before
    # linked history existed, so no stored hash or test vector shifts.
    plain = attest_liabilities("vendor", {"acme": 100.0}, attestor="auditor", as_of=_AS_OF)
    assert not plain.has_prior
    assert "prior" not in plain.attestation_facts()
    # Same facts, same hash as the equivalent attestation the non-equivocation tests build.
    twin = attest_liabilities("vendor", {"acme": 100.0}, attestor="auditor", as_of=_AS_OF)
    assert plain.content_hash == twin.content_hash


def test_link_to_refuses_a_back_dated_predecessor():
    later = _signed("vendor", {"acme": 100.0}, as_of=_AS_OF2)
    with pytest.raises(SettlementError, match="strictly later"):
        attest_liabilities("vendor", {"acme": 100.0}, attestor="auditor", as_of=_AS_OF, prior=later)


def test_link_to_refuses_a_different_counterparty():
    other = _signed("supplier", {"acme": 100.0})
    with pytest.raises(SettlementError, match="one\n? *counterparty|one counterparty"):
        attest_liabilities(
            "vendor", {"acme": 100.0}, attestor="auditor", as_of=_AS_OF2, prior=other
        )


def test_back_dated_link_is_caught_from_the_bytes():
    s1 = _signed("vendor", {"acme": 100.0}, as_of=_AS_OF)
    s2 = attest_liabilities("vendor", {"acme": 100.0}, attestor="auditor", as_of=_AS_OF2, prior=s1)
    # Tamper the successor's own as_of to before the predecessor's — re-seal to match the hash.
    s2.as_of = datetime(2025, 12, 31, tzinfo=UTC)
    s2.seal()
    assert not s2.verify().liabilities_sound  # the back-dated link fails the soundness check


def test_partial_prior_link_is_refused():
    s = attest_liabilities("vendor", {"acme": 100.0}, attestor="auditor", as_of=_AS_OF2)
    s.prior_hash = "abc"  # a hash without a root or instant is malformed
    s.seal()
    assert not s.verify().liabilities_sound


# -- discharges ---------------------------------------------------------------


def test_discharge_is_signed_by_the_creditor():
    d = discharge_liability("vendor", "acme", 70.0, as_of=_AS_OF2).sign(_ACME)
    assert d.creditor == "acme"
    assert d.amount_usd == pytest.approx(70.0)
    assert d.verify(ATTESTOR).valid
    assert d.verify(ATTESTOR, require=["acme"]).valid


def test_discharge_signing_as_non_creditor_is_refused():
    d = discharge_liability("vendor", "acme", 70.0)
    with pytest.raises(SettlementError, match="signed by its creditor"):
        d.sign(ATTESTOR, party="vendor")


def test_discharge_negative_amount_refused():
    with pytest.raises(SettlementError, match="negative"):
        discharge_liability("vendor", "acme", -1.0)


def test_discharge_tamper_caught_even_after_reseal():
    d = discharge_liability("vendor", "acme", 70.0).sign(_ACME)
    d.amount_usd = 999.0
    assert not d.verify().hash_ok
    d.seal()  # recompute the hash to match the tampered amount
    # The forged figure no longer carries the creditor's signature over it.
    assert not d.verify(ATTESTOR).signatures_ok


def test_discharge_wire_roundtrip():
    d = discharge_liability("vendor", "acme", 70.0, as_of=_AS_OF2, note="invoice-7").sign(_ACME)
    restored = Discharge.from_wire(d.to_wire())
    assert restored.verify(ATTESTOR).valid
    assert restored.amount_usd == pytest.approx(70.0)


# -- monotonicity walk --------------------------------------------------------


def test_constant_history_is_consistent():
    report = check_history_consistency(list(_history()), verifier=ATTESTOR)
    assert report.consistent
    assert report.chains == 1
    assert report.checked == 3
    proof = report.proofs[0]
    assert proof.monotone and proof.chain_linked
    assert proof.verify(ATTESTOR).valid


def test_increasing_obligation_is_monotone():
    s1 = _signed("vendor", {"acme": 50.0}, as_of=_AS_OF)
    s2 = attest_liabilities("vendor", {"acme": 90.0}, attestor="auditor", as_of=_AS_OF2, prior=s1)
    s2.sign(ATTESTOR)
    report = check_history_consistency([s1, s2], verifier=ATTESTOR)
    assert report.consistent  # a debt growing is fine; only a silent drop is a breach


def test_unexplained_drop_is_a_breach():
    s1, s2, _ = _history()
    s3 = attest_liabilities(
        "vendor", {"acme": 30.0, "globex": 40.0}, attestor="auditor", as_of=_AS_OF3, prior=s2
    ).sign(ATTESTOR)
    report = check_history_consistency([s1, s2, s3], verifier=ATTESTOR)
    assert not report.consistent
    assert report.breaching_posters == ["vendor"]
    proof = report.proofs[0]
    assert proof.verify(ATTESTOR).valid
    assert proof.breaching_creditors == ["acme"]
    breach = proof.breaches[0]
    assert breach.dropped_usd == pytest.approx(70.0)
    assert breach.unexplained_usd == pytest.approx(70.0)
    assert breach.prior_hash == s2.content_hash and breach.next_hash == s3.content_hash


def test_a_signed_discharge_explains_a_drop():
    s1, s2, _ = _history()
    s3 = attest_liabilities(
        "vendor", {"acme": 30.0, "globex": 40.0}, attestor="auditor", as_of=_AS_OF3, prior=s2
    ).sign(ATTESTOR)
    settled = discharge_liability("vendor", "acme", 70.0, as_of=_AS_OF3).sign(_ACME)
    report = check_history_consistency([s1, s2, s3], discharges=[settled], verifier=ATTESTOR)
    assert report.consistent
    proof = report.proofs[0]
    assert proof.monotone
    assert len(proof.discharges) == 1  # only the discharge that explained a drop is embedded
    assert proof.verify(ATTESTOR).valid


def test_a_partial_discharge_leaves_a_residual_breach():
    s1, s2, _ = _history()
    s3 = attest_liabilities(
        "vendor", {"acme": 30.0, "globex": 40.0}, attestor="auditor", as_of=_AS_OF3, prior=s2
    ).sign(ATTESTOR)
    settled = discharge_liability("vendor", "acme", 50.0, as_of=_AS_OF3).sign(_ACME)
    report = check_history_consistency([s1, s2, s3], discharges=[settled], verifier=ATTESTOR)
    assert not report.consistent
    breach = report.proofs[0].breaches[0]
    assert breach.discharged_usd == pytest.approx(50.0)
    assert breach.unexplained_usd == pytest.approx(20.0)


def test_a_forged_discharge_does_not_explain_a_drop():
    s1, s2, _ = _history()
    s3 = attest_liabilities(
        "vendor", {"acme": 30.0, "globex": 40.0}, attestor="auditor", as_of=_AS_OF3, prior=s2
    ).sign(ATTESTOR)
    # The poster signs a "discharge" with the wrong secret while naming the creditor — refused.
    forged = discharge_liability("vendor", "acme", 70.0, as_of=_AS_OF3).sign(
        HMACSigner("forger-key", key_id="acme"), party="acme"
    )
    report = check_history_consistency([s1, s2, s3], discharges=[forged], verifier=ATTESTOR)
    assert not report.consistent  # a forged release cannot paper over the drop
    assert len(report.proofs[0].discharges) == 0


def test_a_discharge_outside_the_window_does_not_apply():
    s1, s2, _ = _history()
    s3 = attest_liabilities(
        "vendor", {"acme": 30.0, "globex": 40.0}, attestor="auditor", as_of=_AS_OF3, prior=s2
    ).sign(ATTESTOR)
    # The release is dated after the drop's snapshot window (_AS_OF2, _AS_OF3].
    late = discharge_liability("vendor", "acme", 70.0, as_of=datetime(2026, 2, 1, tzinfo=UTC))
    late.sign(_ACME)
    report = check_history_consistency([s1, s2, s3], discharges=[late], verifier=ATTESTOR)
    assert not report.consistent


def test_one_discharge_cannot_explain_two_drops():
    # acme drops 100→60 (t1→t2) and 60→20 (t2→t3); a single $40 release covers only one transition.
    s1 = _signed("vendor", {"acme": 100.0}, as_of=_AS_OF)
    s2 = attest_liabilities("vendor", {"acme": 60.0}, attestor="auditor", as_of=_AS_OF2, prior=s1)
    s2.sign(ATTESTOR)
    s3 = attest_liabilities("vendor", {"acme": 20.0}, attestor="auditor", as_of=_AS_OF3, prior=s2)
    s3.sign(ATTESTOR)
    d12 = discharge_liability("vendor", "acme", 40.0, as_of=_AS_OF2).sign(_ACME)
    report = check_history_consistency([s1, s2, s3], discharges=[d12], verifier=ATTESTOR)
    assert not report.consistent  # the second drop remains unexplained
    assert len(report.proofs[0].breaches) == 1
    assert report.proofs[0].breaches[0].next_hash == s3.content_hash


def test_separate_posters_walk_independently():
    s1, s2, s3 = _history("vendor")
    o1 = _signed("supplier", {"acme": 50.0}, as_of=_AS_OF)
    o2 = attest_liabilities("supplier", {"acme": 10.0}, attestor="auditor", as_of=_AS_OF2, prior=o1)
    o2.sign(ATTESTOR)
    report = check_history_consistency([s1, s2, s3, o1, o2], verifier=ATTESTOR)
    assert report.chains == 2
    assert report.breaching_posters == ["supplier"]  # only supplier dropped acme's debt


def test_single_snapshot_yields_no_proof():
    report = check_history_consistency([_signed("vendor", {"acme": 100.0})], verifier=ATTESTOR)
    assert report.consistent
    assert report.chains == 0


def test_same_instant_snapshots_collapse_to_one_step():
    # Two roots for one instant are the domain of non-equivocation, not history: the walk keeps one
    # per instant so the proof stays a strictly increasing, verifiable sequence.
    a = _signed("vendor", {"acme": 100.0}, as_of=_AS_OF)
    b = _signed("vendor", {"globex": 40.0}, as_of=_AS_OF)  # same instant, different root
    c = attest_liabilities(
        "vendor", {"acme": 100.0}, attestor="auditor", as_of=_AS_OF2, prior=a
    ).sign(ATTESTOR)
    report = check_history_consistency([a, b, c], verifier=ATTESTOR)
    assert report.chains == 1
    proof = report.proofs[0]
    assert proof.snapshot_count == 2  # one per instant
    assert proof.verify(ATTESTOR).valid


# -- inadmissible evidence ----------------------------------------------------


def test_tampered_snapshot_is_excluded():
    s1, s2, s3 = _history()
    s2.liabilities_usd = 1.0  # total no longer re-derives
    # Excluding the tampered middle snapshot still leaves a constant acme across s1, s3.
    report = check_history_consistency([s1, s2, s3], verifier=ATTESTOR)
    assert report.checked == 2
    assert report.consistent


def test_unsigned_snapshot_is_excluded_with_a_verifier():
    s1, s2, _ = _history()
    s3 = attest_liabilities(
        "vendor", {"acme": 30.0, "globex": 40.0}, attestor="auditor", as_of=_AS_OF3, prior=s2
    )  # never signed
    report = check_history_consistency([s1, s2, s3], verifier=ATTESTOR)
    assert report.checked == 2  # the unsigned drop is not admitted as evidence
    assert report.consistent


# -- proof verification & wire ------------------------------------------------


def test_history_proof_dropped_breach_is_caught():
    s1, s2, _ = _history()
    s3 = attest_liabilities(
        "vendor", {"acme": 30.0, "globex": 40.0}, attestor="auditor", as_of=_AS_OF3, prior=s2
    ).sign(ATTESTOR)
    proof = check_history_consistency([s1, s2, s3], verifier=ATTESTOR).proofs[0]
    proof.breaches = []  # forge away the breach
    result = proof.verify(ATTESTOR)
    assert not result.valid
    assert not result.hash_ok or not result.monotone_sound


def test_history_proof_require_monotone_raises():
    s1, s2, _ = _history()
    s3 = attest_liabilities(
        "vendor", {"acme": 30.0, "globex": 40.0}, attestor="auditor", as_of=_AS_OF3, prior=s2
    ).sign(ATTESTOR)
    proof = check_history_consistency([s1, s2, s3], verifier=ATTESTOR).proofs[0]
    with pytest.raises(SettlementError, match="not monotone"):
        proof.require_monotone()


def test_history_require_linked_raises_on_unlinked_chain():
    # Snapshots with no prior commitment are still walked, but are not a contiguous chain.
    s1 = _signed("vendor", {"acme": 100.0}, as_of=_AS_OF)
    s2 = _signed("vendor", {"acme": 100.0}, as_of=_AS_OF2)
    proof = check_history_consistency([s1, s2], verifier=ATTESTOR).proofs[0]
    assert proof.monotone and not proof.chain_linked
    with pytest.raises(SettlementError, match="contiguous"):
        proof.require_linked()


def test_history_proof_wire_roundtrip_preserves_breach():
    s1, s2, _ = _history()
    s3 = attest_liabilities(
        "vendor", {"acme": 30.0, "globex": 40.0}, attestor="auditor", as_of=_AS_OF3, prior=s2
    ).sign(ATTESTOR)
    settled = discharge_liability("vendor", "acme", 20.0, as_of=_AS_OF3).sign(_ACME)
    proof = check_history_consistency([s1, s2, s3], discharges=[settled], verifier=ATTESTOR).proofs[
        0
    ]
    restored = HistoryConsistencyProof.from_wire(proof.to_wire())
    assert restored.verify(ATTESTOR).valid
    assert restored.breaches[0].unexplained_usd == pytest.approx(50.0)


def test_history_proof_reporter_signature():
    reporter = HMACSigner("attestor-key", key_id="auditor")
    proof = check_history_consistency(list(_history()), verifier=ATTESTOR).proofs[0]
    proof.sign(reporter, party="auditor")
    assert proof.verify(ATTESTOR, require=["auditor"]).valid


def test_history_report_wire_roundtrip():
    report = check_history_consistency(list(_history()), verifier=ATTESTOR)
    restored = HistoryConsistencyReport.from_wire(report.to_wire())
    assert restored.consistent
    assert restored.proofs[0].verify(ATTESTOR).valid


def test_history_rejects_malformed_items():
    with pytest.raises(SettlementError):
        check_history_consistency(["not-an-attestation"])


def test_history_rejects_malformed_discharges():
    with pytest.raises(SettlementError, match="discharges must be"):
        check_history_consistency(list(_history()), discharges=["nope"], verifier=ATTESTOR)


def test_history_audit_details_are_json_safe():
    import json

    proof = check_history_consistency(list(_history()), verifier=ATTESTOR).proofs[0]
    json.dumps(proof.audit_details())
    d = discharge_liability("vendor", "acme", 70.0).sign(_ACME)
    json.dumps(d.audit_details())


def test_summary_mentions_unexplained_drop(capsys):
    s1, s2, _ = _history()
    s3 = attest_liabilities(
        "vendor", {"acme": 30.0, "globex": 40.0}, attestor="auditor", as_of=_AS_OF3, prior=s2
    ).sign(ATTESTOR)
    proof = check_history_consistency([s1, s2, s3], verifier=ATTESTOR).proofs[0]
    proof.print_summary()
    out = capsys.readouterr().out
    assert "inconsistent" in out and "acme" in out


# -- app / book wiring --------------------------------------------------------


def test_app_check_history_consistency_audits_and_dings_reputation():
    app = _app("auditor")
    app.use_reputation_ledger()
    s1 = app.attest_liabilities("vendor", {"acme": 100.0}, as_of=_AS_OF)
    s2 = app.attest_liabilities("vendor", {"acme": 30.0}, as_of=_AS_OF2, prior=s1)
    report = app.check_history_consistency([s1, s2], verify_with=app.contract_signer)
    assert not report.consistent
    assert report.breaching_posters == ["vendor"]
    assert app.reputation_ledger.weight("vendor") < 1.0
    assert report.proofs[0].audit_id is not None
    assert len(app.audit.query(action=HISTORY_ACTION)) == 1
    assert report.proofs[0].verify(app.contract_signer).valid


def test_app_check_history_consistency_can_skip_side_effects():
    app = _app("auditor")
    app.use_reputation_ledger()
    s1 = app.attest_liabilities("vendor", {"acme": 100.0}, as_of=_AS_OF)
    s2 = app.attest_liabilities("vendor", {"acme": 30.0}, as_of=_AS_OF2, prior=s1)
    report = app.check_history_consistency(
        [s1, s2], verify_with=app.contract_signer, record_reputation=False, record_audit=False
    )
    assert not report.consistent
    assert len(app.audit.query(action=HISTORY_ACTION)) == 0
    assert app.reputation_ledger.weight("vendor") == pytest.approx(
        app.reputation_ledger.weight("x")
    )


def test_app_discharge_liability_signs_and_audits():
    app = _app("acme")
    discharge = app.discharge_liability("vendor", 70.0, as_of=_AS_OF2)
    assert discharge.creditor == "acme"
    assert discharge.verify(app.contract_signer).valid
    assert discharge.audit_id is not None
    assert len(app.audit.query(action=DISCHARGE_ACTION)) == 1


def test_book_check_history_consistency_records_inconsistent():
    from vincio.optimize.reputation import ReputationLedger
    from vincio.security.audit import AuditLog
    from vincio.settlement.book import SettlementBook

    ledger = ReputationLedger()
    book = SettlementBook(owner="auditor", signer=ATTESTOR, audit=AuditLog(), reputation=ledger)
    s1 = book.attest_liabilities("vendor", {"acme": 100.0}, attestor="auditor", as_of=_AS_OF)
    s2 = book.attest_liabilities(
        "vendor", {"acme": 30.0}, attestor="auditor", as_of=_AS_OF2, prior=s1
    )
    report = book.check_history_consistency([s1, s2], verify_with=ATTESTOR)
    assert not report.consistent
    assert len(book.audit.query(action=HISTORY_ACTION)) == 1
    assert ledger.weight("vendor") < 1.0
    assert report.proofs[0].verify(ATTESTOR).valid


def test_book_discharge_liability_signs_as_owner():
    from vincio.security.audit import AuditLog
    from vincio.settlement.book import SettlementBook

    book = SettlementBook(owner="acme", signer=_ACME, audit=AuditLog())
    discharge = book.discharge_liability("vendor", 70.0)
    assert discharge.creditor == "acme"
    assert discharge.verify(ATTESTOR).valid  # shared fabric secret verifies the owner's signature
    assert len(book.audit.query(action=DISCHARGE_ACTION)) == 1
