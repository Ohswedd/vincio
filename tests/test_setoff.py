"""Cross-org insolvency set-off & close-out netting (3.43).

A creditor of an insolvent estate is often *also* a debtor of it. A signed, mutually-agreed
``SetOffStatement`` collapses the obligations running both ways between a poster and one creditor to
the poster's bounded net liability, and ``resolve_insolvency(set_off=…)`` folds those statements to
reduce each creditor to its net claim *before* the waterfall — a creditor in debit recovers nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from vincio import (
    ContextApp,
    SetOffStatement,
    attest_custody,
    attest_liabilities,
    build_set_off_statement,
    check_completeness,
    resolve_insolvency,
    set_off_from_records,
)
from vincio.core.errors import SettlementError
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement.setoff import SETOFF_ACTION

# Cross-party signing convention: every party signs with the shared fabric secret, distinguished
# only by key_id (its identity), so one verifier checks every party's signature alike.
FABRIC = "fabric-secret"
VENDOR = HMACSigner(FABRIC, key_id="vendor")
ACME = HMACSigner(FABRIC, key_id="acme")
AUDITOR = HMACSigner(FABRIC, key_id="auditor")
VERIFIER = HMACSigner(FABRIC, key_id="any")
FORGER = HMACSigner("forger-secret", key_id="acme")

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _reserves(amount: float = 60.0, *, poster: str = "vendor") -> object:
    return attest_custody(poster, {"omnibus": amount}, custodian="custodian")


def _owed(mapping: dict[str, float] | None = None, *, poster: str = "vendor") -> object:
    return attest_liabilities(poster, mapping or {"bank": 50.0, "acme": 30.0}, attestor="auditor")


def _signed(statement: SetOffStatement) -> SetOffStatement:
    """Co-sign a statement as both the poster and the creditor (a mutual close-out)."""
    statement.sign(HMACSigner(FABRIC, key_id=statement.poster), party=statement.poster)
    statement.sign(HMACSigner(FABRIC, key_id=statement.creditor), party=statement.creditor)
    return statement


# == set-off statement ========================================================

# -- construction & net derivation --------------------------------------------


def test_poster_still_owes_when_owed_exceeds_owing():
    st = build_set_off_statement("vendor", "acme", 30.0, 12.0)
    assert st.poster_net_claim_usd == 18.0
    assert st.net_debtor == "vendor" and st.net_creditor == "acme"
    assert st.net_usd == 18.0
    assert st.direction == "poster_owes"
    assert not st.creditor_in_debit


def test_creditor_in_debit_when_owing_exceeds_owed():
    st = build_set_off_statement("vendor", "acme", 30.0, 50.0)
    assert st.poster_net_claim_usd == 0.0  # floored at zero
    assert st.net_debtor == "acme" and st.net_creditor == "vendor"
    assert st.net_usd == 20.0
    assert st.creditor_in_debit
    assert st.direction == "creditor_in_debit"


def test_equal_obligations_eliminate():
    st = build_set_off_statement("vendor", "acme", 30.0, 30.0)
    assert st.eliminated
    assert st.net_debtor == "" and st.net_creditor == ""
    assert st.net_usd == 0.0
    assert st.direction == "eliminated"


def test_set_off_usd_is_the_min_of_the_two_sides():
    assert build_set_off_statement("vendor", "acme", 30.0, 12.0).set_off_usd == 12.0
    assert build_set_off_statement("vendor", "acme", 30.0, 50.0).set_off_usd == 30.0


def test_same_party_on_both_sides_is_refused():
    with pytest.raises(SettlementError):
        build_set_off_statement("vendor", "vendor", 10.0, 5.0)


def test_negative_figure_is_refused():
    with pytest.raises(SettlementError):
        build_set_off_statement("vendor", "acme", -1.0, 5.0)


# -- hashing, signing, verification -------------------------------------------


def test_statement_is_sealed_and_verifies():
    st = build_set_off_statement("vendor", "acme", 30.0, 12.0)
    assert st.content_hash
    assert st.verify().valid


def test_references_order_does_not_change_the_hash():
    a = build_set_off_statement("vendor", "acme", 30.0, 12.0, references=["c2", "c1"], as_of=T0)
    b = build_set_off_statement("vendor", "acme", 30.0, 12.0, references=["c1", "c2"], as_of=T0)
    assert a.content_hash == b.content_hash


def test_tampered_owing_caught_even_after_reseal():
    st = build_set_off_statement("vendor", "acme", 30.0, 12.0, as_of=T0)
    st.owing_usd = 28.0  # inflate what the creditor is said to owe back (wipe out its recovery)
    assert not st.verify().hash_ok
    st.seal()  # re-seal the lie -> the net no longer re-derives
    assert not st.verify().net_sound


def test_tampered_net_caught_even_after_reseal():
    st = build_set_off_statement("vendor", "acme", 30.0, 12.0)
    st.net_usd = 999.0
    st.seal()
    assert st.verify().hash_ok
    assert not st.verify().net_sound


def test_mutual_signing_and_require_mutual():
    st = build_set_off_statement("vendor", "acme", 30.0, 12.0)
    st.sign(VENDOR, party="vendor")
    assert not st.verify(VERIFIER, require_mutual=True).valid  # only one side signed
    st.sign(ACME, party="acme")
    assert st.mutual
    assert st.verify(VERIFIER, require_mutual=True).valid


def test_forged_signature_refused():
    st = _signed(build_set_off_statement("vendor", "acme", 30.0, 12.0))
    st.sign(FORGER, party="acme")  # wrong key for acme
    assert not st.verify(VERIFIER, require_mutual=True).signatures_ok


def test_re_signing_replaces_prior_signature():
    st = build_set_off_statement("vendor", "acme", 30.0, 12.0)
    st.sign(VENDOR, party="vendor")
    st.sign(VENDOR, party="vendor")
    assert st.signed_by == ["vendor"]


def test_require_valid_raises_on_tamper():
    st = build_set_off_statement("vendor", "acme", 30.0, 12.0)
    st.content_hash = "deadbeef"
    with pytest.raises(SettlementError):
        st.require_valid()


def test_statement_wire_roundtrip():
    st = _signed(build_set_off_statement("vendor", "acme", 30.0, 12.0))
    back = SetOffStatement.from_wire(st.to_wire())
    assert back.verify(VERIFIER, require_mutual=True).valid
    assert back.poster_net_claim_usd == 18.0


def test_audit_details_json_safe():
    st = build_set_off_statement("vendor", "acme", 30.0, 12.0)
    details = st.audit_details()
    assert details["poster"] == "vendor" and details["creditor"] == "acme"
    assert details["direction"] == "poster_owes"


# -- derived from the existing artifacts --------------------------------------


def test_set_off_from_records_reads_attestation_and_records():
    from vincio.settlement.book import settle_contract
    from vincio.settlement.record import SettlementRecord

    owed = attest_liabilities("vendor", {"acme": 30.0, "bank": 50.0}, attestor="auditor")
    # acme (buyer) owes vendor (seller) $12 across a settled contract.
    rec = SettlementRecord(contract_id="c1", buyer="acme", seller="vendor", amount_owed_usd=12.0)
    rec.seal()
    st = set_off_from_records("vendor", "acme", owed, [rec])
    assert st.owed_usd == 30.0  # from the attestation's line for acme
    assert st.owing_usd == 12.0  # from the records where acme owes vendor
    assert st.references == ["c1"]
    assert st.poster_net_claim_usd == 18.0
    assert settle_contract is not None  # import smoke


def test_set_off_from_records_dedupes_the_same_settlement():
    from vincio.settlement.record import SettlementRecord

    owed = attest_liabilities("vendor", {"acme": 30.0}, attestor="auditor")
    rec = SettlementRecord(contract_id="c1", buyer="acme", seller="vendor", amount_owed_usd=12.0)
    rec.seal()
    # The same settlement seen from both books must not be double-counted.
    st = set_off_from_records("vendor", "acme", owed, [rec, rec.model_copy(deep=True)])
    assert st.owing_usd == 12.0


def test_set_off_from_records_refuses_tampered_record():
    from vincio.settlement.record import SettlementRecord

    owed = attest_liabilities("vendor", {"acme": 30.0}, attestor="auditor")
    rec = SettlementRecord(contract_id="c1", buyer="acme", seller="vendor", amount_owed_usd=12.0)
    rec.seal()
    rec.amount_owed_usd = 999.0  # tamper without re-sealing
    with pytest.raises(SettlementError):
        set_off_from_records("vendor", "acme", owed, [rec])


def test_set_off_from_records_refuses_wrong_poster_attestation():
    owed = attest_liabilities("globex", {"acme": 30.0}, attestor="auditor")
    with pytest.raises(SettlementError):
        set_off_from_records("vendor", "acme", owed, [])


# == close-out netting into the waterfall =====================================


def test_set_off_reduces_the_creditor_to_its_net_claim():
    owed = _owed({"bank": 50.0, "acme": 30.0})
    st = _signed(build_set_off_statement("vendor", "acme", 30.0, 12.0))
    res = resolve_insolvency(_reserves(68.0), owed, set_off=[st])
    assert res.gross_liabilities_usd == 80.0
    assert res.liabilities_usd == 68.0  # acme netted from 30 to 18
    assert res.set_off_usd == 12.0
    acme = res.recovery_of("acme")
    assert acme.gross_claim_usd == 30.0 and acme.set_off_usd == 12.0 and acme.claim_usd == 18.0
    assert res.solvent  # 68 reserves cover the 68 net


def test_creditor_in_debit_recovers_nothing():
    owed = _owed({"bank": 50.0, "acme": 30.0})
    st = _signed(build_set_off_statement("vendor", "acme", 30.0, 40.0))  # acme owes more
    res = resolve_insolvency(_reserves(60.0), owed, set_off=[st])
    acme = res.recovery_of("acme")
    assert acme.claim_usd == 0.0 and acme.recovery_usd == 0.0
    assert acme.made_whole  # no net claim -> no shortfall borne
    assert res.liabilities_usd == 50.0  # only the bank's claim is distributable


def test_set_off_shrinks_the_distributable_estate():
    # Without set-off the 60 reserves cover only 60% of 80; with acme's 12 set off, 60 of 68.
    owed = _owed({"bank": 50.0, "acme": 30.0})
    plain = resolve_insolvency(_reserves(60.0), owed)
    st = _signed(build_set_off_statement("vendor", "acme", 30.0, 12.0))
    netted = resolve_insolvency(_reserves(60.0), owed, set_off=[st])
    assert netted.recovery_rate > plain.recovery_rate


def test_empty_set_off_list_is_a_no_op():
    res = resolve_insolvency(_reserves(60.0), _owed(), set_off=[])
    assert not res.set_off
    assert res.set_off_hashes == []


# -- refusals -----------------------------------------------------------------


def test_one_sided_set_off_refused():
    owed = _owed()
    st = build_set_off_statement("vendor", "acme", 30.0, 12.0)
    st.sign(VENDOR, party="vendor")  # only the poster signed
    with pytest.raises(SettlementError):
        resolve_insolvency(_reserves(60.0), owed, set_off=[st], verifier=VERIFIER)


def test_over_stated_set_off_refused():
    # The statement claims vendor owes acme $40, but the attestation shows $30.
    owed = _owed({"bank": 50.0, "acme": 30.0})
    st = _signed(build_set_off_statement("vendor", "acme", 40.0, 12.0))
    with pytest.raises(SettlementError):
        resolve_insolvency(_reserves(60.0), owed, set_off=[st])


def test_set_off_for_wrong_poster_refused():
    owed = _owed()
    st = _signed(build_set_off_statement("globex", "acme", 30.0, 12.0))
    with pytest.raises(SettlementError):
        resolve_insolvency(_reserves(60.0), owed, set_off=[st])


def test_creditor_set_off_twice_refused():
    owed = _owed({"bank": 50.0, "acme": 30.0})
    a = _signed(build_set_off_statement("vendor", "acme", 30.0, 6.0))
    b = _signed(build_set_off_statement("vendor", "acme", 30.0, 6.0))
    with pytest.raises(SettlementError):
        resolve_insolvency(_reserves(60.0), owed, set_off=[a, b])


def test_tampered_set_off_refused():
    owed = _owed()
    st = _signed(build_set_off_statement("vendor", "acme", 30.0, 12.0))
    st.owing_usd = 5.0  # tamper without re-sealing
    with pytest.raises(SettlementError):
        resolve_insolvency(_reserves(60.0), owed, set_off=[st])


# -- verification of a netted resolution --------------------------------------


def test_netted_resolution_verifies_offline():
    owed = _owed({"bank": 50.0, "acme": 30.0})
    st = _signed(build_set_off_statement("vendor", "acme", 30.0, 12.0))
    res = resolve_insolvency(_reserves(60.0), owed, set_off=[st])
    assert res.verify().valid and res.verify().distribution_sound
    assert res.set_off


def test_inflated_set_off_caught_even_after_reseal():
    owed = _owed({"bank": 50.0, "acme": 30.0})
    st = _signed(build_set_off_statement("vendor", "acme", 30.0, 12.0))
    res = resolve_insolvency(_reserves(60.0), owed, set_off=[st])
    acme = res.recovery_of("acme")
    acme.set_off_usd = 25.0  # claim more was netted than really was (without lowering the claim)
    res.seal()
    assert res.verify().hash_ok
    assert not res.verify().distribution_sound


def test_set_off_binding_detects_a_substituted_statement():
    owed = _owed({"bank": 50.0, "acme": 30.0})
    st = _signed(build_set_off_statement("vendor", "acme", 30.0, 12.0))
    res = resolve_insolvency(_reserves(60.0), owed, set_off=[st])
    assert res.verify(VERIFIER, set_off=[st]).set_off_bound
    # A different statement (more netted) does not bind.
    other = _signed(build_set_off_statement("vendor", "acme", 30.0, 20.0))
    assert not res.verify(VERIFIER, set_off=[other]).set_off_bound


def test_set_off_binding_requires_mutual_statement():
    owed = _owed({"bank": 50.0, "acme": 30.0})
    st = _signed(build_set_off_statement("vendor", "acme", 30.0, 12.0))
    res = resolve_insolvency(_reserves(60.0), owed, set_off=[st])
    # Strip acme's signature: the bound statement no longer verifies mutually.
    one_sided = SetOffStatement.from_wire(st.to_wire())
    one_sided.signatures = [s for s in one_sided.signatures if s.party != "acme"]
    assert not res.verify(VERIFIER, set_off=[one_sided]).set_off_bound


def test_two_folders_compute_the_same_hash_with_set_off():
    owed = attest_liabilities("vendor", {"bank": 50.0, "acme": 30.0}, attestor="auditor", as_of=T0)
    cust = attest_custody("vendor", {"omnibus": 60.0}, custodian="custodian", as_of=T0)
    st = _signed(build_set_off_statement("vendor", "acme", 30.0, 12.0, as_of=T0))
    a = resolve_insolvency(cust, owed, set_off=[st], as_of=T0)
    b = resolve_insolvency(cust, owed, set_off=[st], as_of=T0)
    assert a.content_hash == b.content_hash


def test_set_off_composes_with_completeness():
    # The attestor omits globex; completeness adds it; set-off then nets acme.
    owed = attest_liabilities("vendor", {"bank": 50.0, "acme": 30.0}, attestor="auditor")
    check = check_completeness(owed, {"bank": 50.0, "acme": 30.0, "globex": 40.0})
    st = _signed(build_set_off_statement("vendor", "acme", 30.0, 12.0))
    res = resolve_insolvency(_reserves(80.0), owed, completeness=check, set_off=[st])
    assert res.recovery_of("globex") is not None  # the omitted creditor is paid
    assert res.gross_liabilities_usd == 120.0  # 50 + 30 + 40 completed
    assert res.liabilities_usd == 108.0  # acme netted 30 -> 18
    assert res.attested_liabilities_usd == 80.0  # the attestor's figure unchanged


# == app & book wiring ========================================================


def _app(name: str = "auditor") -> ContextApp:
    app = ContextApp(name=name, provider=MockProvider(default_text="ok"))
    app.use_settlement_book(owner=name)
    return app


def test_app_build_set_off_statement_signs_and_audits():
    app = _app(name="vendor")
    st = app.build_set_off_statement("vendor", "acme", 30.0, 12.0)
    assert st.verify(app.contract_signer).valid
    assert "vendor" in st.signed_by
    assert len(app.audit.query(action=SETOFF_ACTION)) == 1
    assert app.audit.query(action=SETOFF_ACTION)[0].decision == "poster_owes"


def test_app_resolve_insolvency_with_set_off():
    app = _app()
    reserves = app.attest_custody("vendor", {"omnibus": 60.0})
    owed = app.attest_liabilities("vendor", {"bank": 50.0, "acme": 30.0})
    # The two counterparties co-sign the close-out (a mutually-agreed set-off).
    st = _signed(build_set_off_statement("vendor", "acme", 30.0, 12.0))
    res = app.resolve_insolvency(reserves, owed, set_off=[st])
    assert res.verify(app.contract_signer).valid  # the resolution is signed by the app
    assert res.recovery_of("acme").claim_usd == 18.0
    assert res.set_off_usd == 12.0


def test_book_build_set_off_from_its_own_records():
    from vincio.settlement.record import SettlementRecord

    app = _app(name="vendor")
    book = app.settlement_book
    # acme owes vendor $12 on a settled contract carried in vendor's own book.
    rec = SettlementRecord(contract_id="c1", buyer="acme", seller="vendor", amount_owed_usd=12.0)
    rec.seal()
    book.append(rec)
    owed = attest_liabilities("vendor", {"acme": 30.0, "bank": 50.0}, attestor="auditor")
    st = book.build_set_off_statement("vendor", "acme", liabilities=owed)
    assert st.owed_usd == 30.0 and st.owing_usd == 12.0
    assert "vendor" in st.signed_by
    assert len(app.audit.query(action=SETOFF_ACTION)) == 1
