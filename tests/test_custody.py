"""Cross-org collateral custody attestation & proof-of-reserves.

The rehypothecation guard bounds a counterparty's pledges against the capital it ``held`` —
but that holdings figure was *asserted*, not proven. A ``CustodyAttestation`` makes it
evidence-backed: a custodian (or the poster's own signed reserve record) issues a signed,
content-bound proof-of-reserves over the capital actually held, itemized into reserve lines
whose total re-derives on every verify. It verifies offline from the bytes alone — a tampered
reserve figure or a forged custodian is caught — and the book / app path signs and audits it.
"""

from __future__ import annotations

import pytest

from vincio import (
    ContextApp,
    CustodyAttestation,
    ReserveLine,
    attest_custody,
)
from vincio.core.errors import SettlementError
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement.custody import CUSTODY_ACTION

CUSTODIAN = HMACSigner("custodian-key", key_id="custodian")
VENDOR = HMACSigner("vendor-key", key_id="vendor")
FORGER = HMACSigner("forger-key", key_id="forger")


def _app(name: str = "custodian") -> ContextApp:
    app = ContextApp(name=name, provider=MockProvider(default_text="ok"))
    app.use_settlement_book()
    return app


# -- construction & reserve coercion ------------------------------------------


def test_attest_custody_defaults_to_self_custody():
    att = attest_custody("vendor", 80.0)
    assert att.custodian == "vendor"  # self-custody
    assert att.poster == "vendor"
    assert att.self_custody
    assert att.reserves_usd == pytest.approx(80.0)
    assert len(att.reserves) == 1
    assert att.reserves[0].account == "reserves"


def test_third_party_custodian_is_not_self_custody():
    att = attest_custody("vendor", 80.0, custodian="custodian")
    assert att.custodian == "custodian"
    assert not att.self_custody


def test_reserves_accept_mapping_list_and_tuples():
    from_map = attest_custody("vendor", {"omnibus": 50.0, "escrow": 30.0})
    from_lines = attest_custody(
        "vendor", [ReserveLine(account="omnibus", amount_usd=50.0), ("escrow", 30.0)]
    )
    assert from_map.reserves_usd == pytest.approx(80.0)
    assert from_lines.reserves_usd == pytest.approx(80.0)
    assert from_map.accounts == ["escrow", "omnibus"]


def test_total_re_derives_from_line_items():
    att = attest_custody("vendor", {"a": 10.0, "b": 20.0, "c": 5.0})
    assert att.reserves_usd == pytest.approx(35.0)
    assert att.verify().reserves_sound


def test_negative_holding_is_refused():
    with pytest.raises(SettlementError, match="negative"):
        attest_custody("vendor", {"a": 10.0, "b": -5.0})


def test_empty_reserves_prove_zero():
    att = attest_custody("vendor", [])
    assert att.reserves_usd == pytest.approx(0.0)
    assert att.verify().valid


# -- signing ------------------------------------------------------------------


def test_only_the_custodian_signs():
    att = attest_custody("vendor", 80.0, custodian="custodian")
    att.sign(CUSTODIAN)
    assert att.signed_by == ["custodian"]
    assert att.verify(CUSTODIAN, require=["custodian"]).valid


def test_signing_as_a_non_custodian_is_refused():
    att = attest_custody("vendor", 80.0, custodian="custodian")
    with pytest.raises(SettlementError, match="signed by its custodian"):
        att.sign(VENDOR, party="vendor")


def test_re_signing_replaces_the_prior_signature():
    att = attest_custody("vendor", 80.0, custodian="custodian")
    att.sign(CUSTODIAN)
    att.sign(CUSTODIAN)
    assert len(att.signatures) == 1


# -- verification & tamper-evidence -------------------------------------------


def test_unsealed_attestation_is_invalid():
    att = attest_custody("vendor", 80.0)
    att.content_hash = ""
    result = att.verify()
    assert not result.valid and not result.hash_ok


def test_tampered_total_caught_even_after_reseal():
    att = attest_custody("vendor", {"omnibus": 80.0})
    att.reserves_usd = 9_999.0
    # Without re-sealing the hash no longer recomputes.
    assert not att.verify().hash_ok
    # Re-sealing the lie still fails: the total no longer re-derives from the line items.
    att.seal()
    result = att.verify()
    assert result.hash_ok
    assert not result.reserves_sound
    assert not result.valid


def test_tampered_line_item_caught():
    att = attest_custody("vendor", {"omnibus": 80.0})
    att.reserves[0].amount_usd = 1.0  # lie about the holding without touching the total
    att.seal()
    assert not att.verify().reserves_sound


def test_forged_custodian_signature_refused_with_verifier():
    att = attest_custody("vendor", 80.0, custodian="custodian")
    att.sign(CUSTODIAN)
    att.signatures[0].signature = FORGER.sign(att.content_hash)  # forge over the real hash
    result = att.verify(CUSTODIAN)
    assert not result.signatures_ok
    assert not result.valid


def test_require_valid_raises_on_tamper():
    att = attest_custody("vendor", 80.0)
    att.reserves_usd = 1.0
    att.seal()
    with pytest.raises(SettlementError, match="failed verification"):
        att.require_valid()


# -- determinism & serialization ----------------------------------------------


def test_two_custodians_compute_the_same_hash():
    from datetime import UTC, datetime

    t = datetime(2026, 1, 1, tzinfo=UTC)
    a = attest_custody("vendor", {"omnibus": 50.0, "escrow": 30.0}, as_of=t)
    b = attest_custody("vendor", [("escrow", 30.0), ("omnibus", 50.0)], as_of=t)
    assert a.compute_hash() == b.compute_hash()  # order-independent


def test_wire_roundtrip_preserves_verification():
    att = attest_custody("vendor", {"omnibus": 80.0}, custodian="custodian")
    att.sign(CUSTODIAN)
    restored = CustodyAttestation.from_wire(att.to_wire())
    assert restored.content_hash == att.content_hash
    assert restored.verify(CUSTODIAN).valid


def test_audit_details_are_json_safe():
    att = attest_custody("vendor", {"omnibus": 80.0}, custodian="custodian")
    details = att.audit_details()
    assert details["poster"] == "vendor"
    assert details["custodian"] == "custodian"
    assert details["self_custody"] is False
    assert details["reserves_usd"] == pytest.approx(80.0)


# -- app / book integration ---------------------------------------------------


def test_app_attest_custody_signs_and_audits():
    app = _app("custodian")
    att = app.attest_custody("vendor", {"omnibus": 80.0})
    assert att.custodian == "custodian"
    assert att.signed_by == ["custodian"]
    assert att.audit_id is not None
    entries = app.audit.query(action=CUSTODY_ACTION)
    assert len(entries) == 1
    assert entries[0].decision == "custodied"
    assert att.verify(app.contract_signer).valid


def test_app_self_custody_is_recorded_as_self():
    app = _app("vendor")
    att = app.attest_custody("vendor", 80.0)  # the app attests its own reserves
    assert att.self_custody
    assert app.audit.query(action=CUSTODY_ACTION)[0].decision == "self_custody"


def test_book_attest_custody_signs_as_owner():
    app = _app("custodian")
    book = app.settlement_book
    att = book.attest_custody("vendor", {"omnibus": 80.0})
    assert att.custodian == "custodian"
    assert att.signed_by == ["custodian"]
    assert att.verify(book.signer, require=["custodian"]).valid


def test_app_attest_custody_third_party_not_signed_by_app():
    # When the app is neither custodian nor the attestation is its own, it should not sign.
    app = _app("acme")
    att = app.attest_custody("vendor", 80.0, custodian="custodian")
    assert att.custodian == "custodian"
    assert att.signed_by == []  # acme is not the custodian, so it does not sign
