"""Cross-org collateral rehypothecation guards & re-use bounds.

A counterparty may back many contracts with one pooled stake — but a pool only ever
re-allocates capital *within itself*. When the counterparty pledges the *same* stake across
more than one pool, nothing bounds the **re-use**: the same capital is double-counted, over-
stating what actually backs each deal. The ``CollateralLedger`` is the rehypothecation guard:
it folds a counterparty's ``CollateralPool``\\ s into one view, reconciles what they
collectively pledge against the capital it actually holds, surfaces the same capital pledged
twice as a bounded, pinpointed ``ReuseBreach``, and bounds each beneficiary's claim to its
deterministic pari-passu share — reading only the signed, content-bound pools (a tampered one
is refused) and landing the guard on the hash-chained audit log.
"""

from __future__ import annotations

import pytest

from vincio import (
    BeneficiaryClaim,
    CollateralLedger,
    ContextApp,
    ReuseBreach,
    guard_collateral,
    post_collateral_pool,
)
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement.rehypothecation import REHYPOTHECATION_ACTION

VENDOR = HMACSigner("vendor-key", key_id="vendor")
ACME = HMACSigner("acme-key", key_id="acme")


def _contract(
    scope: str, price: float = 100.0, *, buyer: str = "acme", seller: str = "vendor"
) -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope=scope, price_usd=price)
    ).seal()


def _app(name: str = "acme") -> ContextApp:
    app = ContextApp(name=name, provider=MockProvider(default_text="ok"))
    app.use_reputation_ledger()
    app.use_settlement_book()
    return app


# -- folding & defaults -------------------------------------------------------


def test_single_pool_no_reuse_is_within_bounds():
    c1, c2 = _contract("a", 100.0), _contract("b", 200.0)
    pool = post_collateral_pool([c1, c2], fraction=0.1)  # vendor posts 30
    ledger = guard_collateral([pool])
    assert ledger.poster == "vendor"
    assert ledger.pledged_usd == pytest.approx(30.0)
    assert ledger.held_usd == pytest.approx(30.0)  # default = pledged (no duplicates)
    assert ledger.reuse_usd == pytest.approx(0.0)
    assert not ledger.over_committed
    assert ledger.within_bounds
    assert ledger.status == "within_bounds"
    assert ledger.breaches == []
    assert ledger.verify().valid


def test_default_poster_resolves_from_shared_poster():
    pool = post_collateral_pool([_contract("a")], fraction=0.1)
    ledger = guard_collateral([pool])
    assert ledger.poster == "vendor"


def test_mismatched_posters_require_explicit_poster():
    # Two pools posted by different sellers (the buyer backs one as the poster).
    p1 = post_collateral_pool([_contract("a", seller="vendor")], fraction=0.1)
    p2 = post_collateral_pool([_contract("b", seller="globex")], fraction=0.1)
    with pytest.raises(SettlementError, match="do not share one poster"):
        guard_collateral([p1, p2])


def test_explicit_poster_must_own_every_pool():
    p1 = post_collateral_pool([_contract("a", seller="vendor")], fraction=0.1)
    p2 = post_collateral_pool([_contract("b", seller="globex")], fraction=0.1)
    with pytest.raises(SettlementError, match="not posted by"):
        guard_collateral([p1, p2], poster="vendor")


def test_empty_pool_set_is_refused():
    with pytest.raises(SettlementError, match="at least one"):
        guard_collateral([])


def test_negative_held_is_refused():
    pool = post_collateral_pool([_contract("a")], fraction=0.1)
    with pytest.raises(SettlementError, match="non-negative"):
        guard_collateral([pool], held=-1.0)


# -- the cross-pool re-use bound ----------------------------------------------


def test_same_contract_across_pools_is_a_pinpointed_reuse_breach():
    c1, c2 = _contract("a", 100.0), _contract("b", 200.0)
    p1 = post_collateral_pool([c1, c2], fraction=0.1)  # c1 -> 10, c2 -> 20
    p2 = post_collateral_pool([c1], fraction=0.1)  # c1 re-pledged -> 10
    ledger = guard_collateral([p1, p2])
    # pledged = 30 + 10 = 40; the duplicate is c1 (10 pledged in two pools).
    assert ledger.pledged_usd == pytest.approx(40.0)
    assert ledger.duplicate_pledge_usd == pytest.approx(10.0)
    # default held = pledged − duplicate = 30, so the re-use surfaces.
    assert ledger.held_usd == pytest.approx(30.0)
    assert ledger.reuse_usd == pytest.approx(10.0)
    assert ledger.over_committed
    assert ledger.status == "over_committed"
    assert len(ledger.breaches) == 1
    breach = ledger.breaches[0]
    assert isinstance(breach, ReuseBreach)
    assert breach.contract_id == c1.id
    assert sorted(breach.pools) == sorted([p1.id, p2.id])
    assert breach.pledged_usd == pytest.approx(20.0)
    assert breach.secured_usd == pytest.approx(10.0)
    assert breach.excess_usd == pytest.approx(10.0)
    assert ledger.breach(c1.id) is breach
    assert ledger.verify().valid


def test_explicit_held_below_distinct_commitment_is_over_committed():
    # No duplicate contract, but the poster does not hold what the pools pledge.
    c1, c2 = _contract("a", 100.0), _contract("b", 200.0)
    pool = post_collateral_pool([c1, c2], fraction=0.5)  # pledges 150
    ledger = guard_collateral([pool], held=100.0)
    assert ledger.pledged_usd == pytest.approx(150.0)
    assert ledger.reuse_usd == pytest.approx(50.0)
    assert ledger.over_committed
    assert ledger.breaches == []  # no double-pledged contract — an aggregate shortfall
    assert ledger.verify().valid


def test_held_above_pledged_is_within_bounds():
    pool = post_collateral_pool([_contract("a")], fraction=0.1)
    ledger = guard_collateral([pool], held=1000.0)
    assert not ledger.over_committed
    assert ledger.available_usd == pytest.approx(990.0)
    assert ledger.reuse_usd == pytest.approx(0.0)


def test_settled_contracts_no_longer_pledge_capital():
    c1, c2 = _contract("a", 100.0), _contract("b", 200.0)
    pool = post_collateral_pool([c1, c2], fraction=0.1)  # posts 30
    from vincio.settlement import settle_contract

    pool.draw(settle_contract(c1, cost_usd=60.0))  # clean — c1 released, no longer open
    ledger = guard_collateral([pool])
    # Only c2 (20) still pledges; the pool balance is unchanged (clean release).
    assert [c.contract_id for p in ledger.pools for c in p.open_contracts] == [c2.id]
    assert ledger.pools[0].committed_usd == pytest.approx(20.0)
    assert ledger.verify().valid


# -- beneficiary-claim priority -----------------------------------------------


def test_beneficiaries_apportioned_pari_passu_under_scarcity():
    # vendor sells to two distinct buyers -> two beneficiaries on one poster.
    ca = _contract("a", 100.0, buyer="acme")  # beneficiary acme, requires 10
    cb = _contract("b", 300.0, buyer="globex")  # beneficiary globex, requires 30
    pool = post_collateral_pool([ca, cb], fraction=0.1)  # pledges 40
    ledger = guard_collateral([pool], held=20.0)  # only half the pledge is held
    assert ledger.over_committed
    acme = ledger.claim("acme")
    globex = ledger.claim("globex")
    assert isinstance(acme, BeneficiaryClaim)
    # 10:30 claims share the held 20 proportionally -> 5:15.
    assert acme.claim_usd == pytest.approx(10.0)
    assert acme.secured_usd == pytest.approx(5.0)
    assert acme.unsecured_usd == pytest.approx(5.0)
    assert acme.share == pytest.approx(0.5)
    assert not acme.is_secured
    assert globex.secured_usd == pytest.approx(15.0)
    assert globex.unsecured_usd == pytest.approx(15.0)
    # The held capital is fully and exactly apportioned, never over-promised.
    assert sum(c.secured_usd for c in ledger.claims) == pytest.approx(20.0)
    assert ledger.verify().valid


def test_beneficiaries_fully_secured_when_held_covers_every_claim():
    ca = _contract("a", 100.0, buyer="acme")
    cb = _contract("b", 300.0, buyer="globex")
    pool = post_collateral_pool([ca, cb], fraction=0.1)  # pledges 40
    ledger = guard_collateral([pool], held=40.0)
    assert all(c.is_secured for c in ledger.claims)
    assert all(c.unsecured_usd == pytest.approx(0.0) for c in ledger.claims)
    assert all(c.share == pytest.approx(1.0) for c in ledger.claims)


def test_duplicate_contract_claim_counted_once_per_beneficiary():
    # The same contract pledged in two pools is a re-use, but the beneficiary can only
    # forfeit it once: its claim counts the largest single pledge, not the sum.
    c1 = _contract("a", 100.0)  # beneficiary acme, requires 10
    p1 = post_collateral_pool([c1], fraction=0.1)
    p2 = post_collateral_pool([c1], fraction=0.1)
    ledger = guard_collateral([p1, p2], held=10.0)
    acme = ledger.claim("acme")
    assert acme.claim_usd == pytest.approx(10.0)  # counted once
    assert acme.is_secured  # 10 held covers the single claim
    assert sorted(acme.pools) == sorted([p1.id, p2.id])


# -- offline verification & content binding -----------------------------------


def test_tampered_total_caught_even_after_reseal():
    c1 = _contract("a", 100.0)
    p1 = post_collateral_pool([c1], fraction=0.1)
    p2 = post_collateral_pool([c1], fraction=0.1)
    ledger = guard_collateral([p1, p2])
    assert ledger.over_committed
    # Hide the over-commitment by zeroing the re-use and re-sealing.
    ledger.reuse_usd = 0.0
    ledger.available_usd = 0.0
    assert not ledger.verify().hash_ok
    ledger.seal()  # recompute the hash to match the tampered figure
    result = ledger.verify()
    assert result.hash_ok  # the hash matches the lie ...
    assert not result.terms_sound  # ... but the re-use re-derives from the bytes
    assert not result.valid


def test_tampered_claim_caught():
    ca = _contract("a", 100.0, buyer="acme")
    cb = _contract("b", 300.0, buyer="globex")
    pool = post_collateral_pool([ca, cb], fraction=0.1)
    ledger = guard_collateral([pool], held=20.0)
    ledger.claim("acme").secured_usd = 20.0  # over-promise one beneficiary
    ledger.seal()
    assert ledger.verify().hash_ok
    assert not ledger.verify().terms_sound


def test_tampered_breach_caught():
    c1 = _contract("a", 100.0)
    p1 = post_collateral_pool([c1], fraction=0.1)
    p2 = post_collateral_pool([c1], fraction=0.1)
    ledger = guard_collateral([p1, p2])
    ledger.breaches[0].excess_usd = 0.0  # downplay the double-pledge
    ledger.seal()
    assert not ledger.verify().terms_sound


def test_unsealed_ledger_is_invalid():
    pool = post_collateral_pool([_contract("a")], fraction=0.1)
    ledger = guard_collateral([pool])
    ledger.content_hash = ""
    result = ledger.verify()
    assert not result.valid
    assert result.reason is not None and "not sealed" in result.reason


def test_fold_order_independent_and_wire_roundtrip():
    c1, c2 = _contract("a", 100.0), _contract("b", 200.0)
    p1 = post_collateral_pool([c1], fraction=0.1)
    p2 = post_collateral_pool([c2], fraction=0.1)
    a = guard_collateral([p1, p2])
    b = guard_collateral([p2, p1])
    assert a.compute_hash() == b.compute_hash()
    restored = CollateralLedger.from_wire(a.to_wire())
    assert restored.content_hash == a.content_hash
    assert restored.verify().valid
    assert restored.reuse_usd == pytest.approx(a.reuse_usd)


def test_two_folders_compute_the_same_hash():
    c1 = _contract("a", 100.0)
    p1 = post_collateral_pool([c1], fraction=0.1)
    p2 = post_collateral_pool([c1], fraction=0.1)
    a = guard_collateral([p1, p2], held=15.0)
    b = guard_collateral([p1, p2], held=15.0)
    assert a.compute_hash() == b.compute_hash()


# -- refusing tampered / forged pools -----------------------------------------


def test_tampered_pool_is_refused_at_fold():
    pool = post_collateral_pool([_contract("a")], fraction=0.1)
    pool.contracts[0].allocated_usd = 9_999.0  # lie without re-sealing
    with pytest.raises(SettlementError, match="tampered"):
        guard_collateral([pool])


def test_forged_pool_signature_is_refused_with_verifier():
    pool = post_collateral_pool([_contract("a")], fraction=0.1)
    pool.sign(VENDOR, party="vendor")
    # Swap in a signature from a different key the verifier will reject.
    forged = HMACSigner("evil-key", key_id="vendor")
    pool.signatures[0].signature = forged.sign(pool.content_hash)
    with pytest.raises(SettlementError, match="invalid signature"):
        guard_collateral([pool], verify_with=VENDOR)


# -- signing & strict guards --------------------------------------------------


def test_sign_and_require_poster_signature():
    pool = post_collateral_pool([_contract("a")], fraction=0.1)
    ledger = guard_collateral([pool])
    ledger.sign(VENDOR, party="vendor")
    assert ledger.signed_by == ["vendor"]
    assert ledger.verify(VENDOR, require=["vendor"]).valid
    assert not ledger.verify(VENDOR, require=["acme"]).valid


def test_resign_replaces_prior_signature():
    pool = post_collateral_pool([_contract("a")], fraction=0.1)
    ledger = guard_collateral([pool])
    ledger.sign(VENDOR, party="vendor")
    ledger.sign(VENDOR, party="vendor")
    assert ledger.signed_by == ["vendor"]


def test_require_within_bounds_raises_when_over_committed():
    c1 = _contract("a", 100.0)
    p1 = post_collateral_pool([c1], fraction=0.1)
    p2 = post_collateral_pool([c1], fraction=0.1)
    ledger = guard_collateral([p1, p2])
    with pytest.raises(SettlementError, match="over-committed"):
        ledger.require_within_bounds()


def test_require_within_bounds_returns_self_when_clean():
    pool = post_collateral_pool([_contract("a")], fraction=0.1)
    ledger = guard_collateral([pool])
    assert ledger.require_within_bounds() is ledger


def test_require_valid_raises_on_tamper():
    pool = post_collateral_pool([_contract("a")], fraction=0.1)
    ledger = guard_collateral([pool])
    ledger.pledged_usd = 0.0
    ledger.seal()
    with pytest.raises(SettlementError, match="failed verification"):
        ledger.require_valid()


# -- app & book wiring --------------------------------------------------------


def test_app_guard_collateral_signs_and_audits():
    app = _app("acme")
    c1 = _contract("a", 100.0)
    p1 = app.post_collateral_pool([c1], fraction=0.1)
    p2 = app.post_collateral_pool([c1], fraction=0.1)  # re-pledge
    ledger = app.guard_collateral([p1, p2])
    assert ledger.over_committed
    assert ledger.signed_by == ["acme"]
    assert ledger.audit_id is not None
    assert ledger.verify(app.contract_signer).valid
    entries = app.audit.query(action=REHYPOTHECATION_ACTION)
    assert len(entries) == 1
    assert entries[0].decision == "over_committed"
    assert app.audit.verify_chain()


def test_app_guard_collateral_can_skip_sign_and_audit():
    app = _app("acme")
    pool = app.post_collateral_pool([_contract("a")], fraction=0.1)
    ledger = app.guard_collateral([pool], sign=False, record_audit=False)
    assert ledger.signed_by == []
    assert ledger.audit_id is None
    assert app.audit.query(action=REHYPOTHECATION_ACTION) == []


def test_book_guard_collateral_signs_as_owner_and_audits():
    app = _app("acme")
    c1 = _contract("a", 100.0)
    p1 = app.post_collateral_pool([c1], fraction=0.1)
    p2 = app.post_collateral_pool([c1], fraction=0.1)
    ledger = app.settlement_book.guard_collateral([p1, p2])
    assert ledger.signed_by == ["acme"]
    assert ledger.over_committed
    assert any(e.resource == ledger.id for e in app.audit.query(action=REHYPOTHECATION_ACTION))


def test_explicit_held_through_app():
    app = _app("acme")
    c1, c2 = _contract("a", 100.0), _contract("b", 200.0)
    pool = app.post_collateral_pool([c1, c2], fraction=0.5)  # pledges 150
    ledger = app.guard_collateral([pool], held=100.0)
    assert ledger.reuse_usd == pytest.approx(50.0)
    assert ledger.audit_details()["reuse_usd"] == pytest.approx(50.0)


# -- reporting ----------------------------------------------------------------


def test_audit_details_and_summary_are_jsonable(capsys):
    c1 = _contract("a", 100.0)
    p1 = post_collateral_pool([c1], fraction=0.1)
    p2 = post_collateral_pool([c1], fraction=0.1)
    ledger = guard_collateral([p1, p2])
    details = ledger.audit_details()
    assert details["status"] == "over_committed"
    assert details["poster"] == "vendor"
    assert c1.id in details["breaches"]
    ledger.print_summary()
    out = capsys.readouterr().out
    assert "over_committed" in out
