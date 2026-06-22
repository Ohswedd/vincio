"""Cross-org collateral pooling & cross-contract margin.

A counterparty running many concurrent contracts posts a single stake — a ``CollateralPool``
margin account — that backs them all at a deterministic, offline-verifiable allocation
(proportional to each contract's admission-required collateral): settling a contract draws a
bounded, pinpointed forfeiture from the shared stake on a breach and releases the rest back to
the available balance on a clean delivery, a pool committed below the collateral its open
contracts require surfaces a bounded top-up obligation, and every post / draw / release / top-up
recomputes from the bytes alone and lands on the hash-chained audit log — driven by the same
settlement verdict the books already close on.
"""

from __future__ import annotations

import pytest

from vincio import (
    CollateralPool,
    ContextApp,
    EscrowConfig,
    admit,
    post_collateral_pool,
    settle_contract,
)
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement.collateral import COLLATERAL_ACTION, draw_pool

ACME = HMACSigner("acme-key", key_id="acme")
VENDOR = HMACSigner("vendor-key", key_id="vendor")


def _contract(
    scope: str, price: float = 1.0, *, buyer: str = "acme", seller: str = "vendor"
) -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope=scope, price_usd=price)
    ).seal()


def _app(name: str = "acme") -> ContextApp:
    app = ContextApp(name=name, provider=MockProvider(default_text="ok"))
    app.use_reputation_ledger()
    app.use_settlement_book()
    return app


# -- posting a pool -----------------------------------------------------------


def test_post_from_uniform_fraction_allocates_proportionally():
    c1, c2, c3 = _contract("a", 100.0), _contract("b", 200.0), _contract("c", 300.0)
    pool = post_collateral_pool([c1, c2, c3], fraction=0.1)
    # required = 10 / 20 / 30; posted defaults to the total required (60).
    assert pool.posted_usd == pytest.approx(60.0)
    assert pool.required_open_usd == pytest.approx(60.0)
    assert [c.allocated_usd for c in pool.contracts] == pytest.approx([10.0, 20.0, 30.0])
    assert pool.available_usd == pytest.approx(0.0)
    assert pool.topup_usd == 0.0
    assert pool.poster == "vendor"  # the common seller backs its delivery
    assert pool.status == "posted"
    assert pool.verify().valid


def test_post_from_admission_decisions_per_contract():
    c1, c2 = _contract("a", 100.0), _contract("b", 100.0)
    d = admit("vendor")  # a newcomer → a positive escrow fraction
    pool = post_collateral_pool([c1, c2], decisions={c1.id: d, c2.id: d})
    expected = round(d.escrow_fraction * 100.0, 6)
    assert pool.contract(c1.id).required_usd == pytest.approx(expected)
    assert pool.contract(c1.id).decision_id == d.id
    assert pool.verify().valid


def test_post_reads_admission_stamp_off_each_contract():
    d = admit("vendor")
    terms = d.apply_to_terms(ContractTerms(scope="work", price_usd=50.0))
    contract = Contract(buyer="acme", seller="vendor", terms=terms).seal()
    pool = post_collateral_pool([contract])  # no explicit source — reads the stamp
    assert pool.contract(contract.id).escrow_fraction == pytest.approx(d.escrow_fraction)
    assert pool.contract(contract.id).required_usd == pytest.approx(
        round(d.escrow_fraction * 50.0, 6)
    )


def test_post_without_a_source_raises():
    with pytest.raises(SettlementError):
        post_collateral_pool([_contract("a")])


def test_empty_pool_raises():
    with pytest.raises(SettlementError):
        post_collateral_pool([])


def test_default_poster_requires_a_common_seller():
    c1 = _contract("a", 100.0, seller="vendor")
    c2 = _contract("b", 100.0, seller="other")
    with pytest.raises(SettlementError):
        post_collateral_pool([c1, c2], fraction=0.1)  # no shared seller, no explicit poster
    # an explicit poster that is a party to both is fine
    pool = post_collateral_pool([c1, c2], fraction=0.1, poster="acme")
    assert pool.poster == "acme"


def test_poster_must_be_a_party_to_every_contract():
    with pytest.raises(SettlementError):
        post_collateral_pool([_contract("a", 100.0)], fraction=0.1, poster="stranger")


def test_pool_binds_each_specific_contract():
    c1 = _contract("a", 100.0)
    pool = post_collateral_pool([c1], fraction=0.3)
    pc = pool.contract(c1.id)
    assert pc.contract_hash == c1.content_hash
    assert pc.beneficiary == "acme"  # the counterparty to the vendor poster


# -- frees capital on a clean delivery ----------------------------------------


def test_clean_delivery_frees_capital_for_the_next_contract():
    c1, c2 = _contract("a", 100.0), _contract("b", 100.0)
    pool = post_collateral_pool([c1, c2], fraction=0.5)  # 50 + 50, posted 100
    pc = pool.draw(settle_contract(c1, cost_usd=60.0))  # within price — clean
    assert pc.state == "released"
    assert pc.forfeited_usd == 0.0
    assert pc.released_usd == pytest.approx(50.0)  # its whole requirement freed
    # The balance is untouched (nothing drawn) but the committed requirement dropped,
    # so capital is freed back to available for the next deal.
    assert pool.balance_usd == pytest.approx(100.0)
    assert pool.required_open_usd == pytest.approx(50.0)
    assert pool.available_usd == pytest.approx(50.0)
    assert pool.status == "active"
    assert pool.verify().valid


# -- draws a forfeiture on a breach -------------------------------------------


def test_breach_draws_proportional_slice_from_the_shared_stake():
    c1, c2 = _contract("a", 100.0), _contract("b", 100.0)
    pool = post_collateral_pool([c1, c2], fraction=0.5)  # 50 each, posted 100
    pc = pool.draw(settle_contract(c1, cost_usd=150.0))  # 50% over
    assert pc.state == "forfeited"
    assert pc.shortfall_fraction == pytest.approx(0.5)
    assert pc.forfeited_usd == pytest.approx(25.0)  # half its $50 requirement
    assert pc.released_usd == pytest.approx(25.0)  # the rest freed
    assert pc.breaches == ["price"]
    assert pool.drawn_usd == pytest.approx(25.0)
    assert pool.balance_usd == pytest.approx(75.0)  # the forfeiture left the pool
    assert pool.verify().valid


def test_forfeiture_scales_with_severity():
    c1, c2 = _contract("a", 100.0), _contract("b", 100.0)
    small = post_collateral_pool([c1], fraction=0.5)
    big = post_collateral_pool([c2], fraction=0.5)
    small.draw(settle_contract(c1, cost_usd=110.0))  # 10% over
    big.draw(settle_contract(c2, cost_usd=180.0))  # 80% over
    assert big.drawn_usd > small.drawn_usd


def test_max_forfeit_cap_guarantees_a_residual():
    c1 = _contract("a", 100.0)
    pool = post_collateral_pool([c1], fraction=0.5, config=EscrowConfig(max_forfeit_fraction=0.8))
    pc = pool.draw(settle_contract(c1, cost_usd=300.0))  # 200% over → shortfall clamps to 1
    assert pc.forfeited_usd == pytest.approx(0.8 * 50.0)
    assert pc.released_usd == pytest.approx(0.2 * 50.0)
    assert pc.released_usd > 0.0
    assert pool.verify().valid


def test_sla_breach_pinpointed_and_forfeited():
    contract = Contract(
        buyer="acme",
        seller="vendor",
        terms=ContractTerms(scope="work", price_usd=100.0, sla_seconds=1.0),
    ).seal()
    pool = post_collateral_pool([contract], fraction=0.5)
    pc = pool.draw(settle_contract(contract, cost_usd=50.0, latency_ms=1500.0))  # 50% over SLA
    assert pc.state == "forfeited"
    assert "sla" in pc.breaches
    assert pool.verify().valid


def test_cannot_draw_an_unbacked_contract():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.3)
    other = settle_contract(_contract("b", 100.0), cost_usd=50.0)
    with pytest.raises(SettlementError):
        pool.draw(other)


def test_cannot_draw_a_contract_twice():
    c1 = _contract("a", 100.0)
    pool = post_collateral_pool([c1], fraction=0.3)
    record = settle_contract(c1, cost_usd=50.0)
    pool.draw(record)
    with pytest.raises(SettlementError):
        pool.draw(record)


def test_fully_settled_pool_reports_residual_balance():
    c1, c2 = _contract("a", 100.0), _contract("b", 100.0)
    pool = post_collateral_pool([c1, c2], fraction=0.5)  # posted 100
    pool.draw(settle_contract(c1, cost_usd=60.0))  # clean
    pool.draw(settle_contract(c2, cost_usd=150.0))  # 50% over → forfeits 25
    assert pool.status == "settled"
    assert pool.required_open_usd == 0.0
    assert pool.residual_usd == pytest.approx(75.0)  # 100 posted − 25 drawn
    assert pool.available_usd == pytest.approx(75.0)
    assert pool.verify().valid


# -- under-collateralization & top-up -----------------------------------------


def test_under_posting_surfaces_a_bounded_topup():
    c1, c2 = _contract("a", 100.0), _contract("b", 100.0)
    pool = post_collateral_pool([c1, c2], fraction=0.2, posted=30.0)  # need 40, posted 30
    assert pool.needs_topup
    assert pool.topup_usd == pytest.approx(10.0)
    assert pool.available_usd == pytest.approx(-10.0)
    assert pool.coverage == pytest.approx(0.75)
    # allocations pro-rate down by the coverage the balance affords.
    assert [c.allocated_usd for c in pool.contracts] == pytest.approx([15.0, 15.0])
    assert pool.verify().valid  # an under-collateralized pool is still verifiable


def test_top_up_clears_the_obligation():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.4, posted=30.0)
    assert pool.topup_usd == pytest.approx(10.0)
    pool.top_up(10.0)
    assert pool.posted_usd == pytest.approx(40.0)
    assert pool.topup_usd == 0.0
    assert pool.available_usd == pytest.approx(0.0)
    assert pool.coverage == pytest.approx(1.0)
    assert pool.verify().valid


def test_top_up_must_be_positive():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.4)
    with pytest.raises(SettlementError):
        pool.top_up(0.0)


def test_backing_a_new_contract_can_over_commit():
    c1 = _contract("a", 100.0)
    pool = post_collateral_pool([c1], fraction=0.4)  # need 40, posted 40
    assert not pool.needs_topup
    c2 = _contract("b", 100.0)
    pool.back(c2, fraction=0.4)  # now needs 80 against 40 posted
    assert pool.required_open_usd == pytest.approx(80.0)
    assert pool.topup_usd == pytest.approx(40.0)
    assert pool.contract(c2.id) is not None
    assert pool.verify().valid


def test_cannot_back_the_same_contract_twice():
    c1 = _contract("a", 100.0)
    pool = post_collateral_pool([c1], fraction=0.4)
    with pytest.raises(SettlementError):
        pool.back(c1, fraction=0.4)


# -- offline verification -----------------------------------------------------


def test_unsealed_pool_is_invalid():
    pool = CollateralPool(poster="vendor")
    pool.content_hash = ""
    result = pool.verify()
    assert not result.valid
    assert "not sealed" in (result.reason or "")


def test_tampered_allocation_caught_even_after_reseal():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.4)
    pool.contracts[0].allocated_usd = 999.0
    pool.seal()  # recompute the hash to match the tampered allocation
    result = pool.verify()
    assert result.hash_ok
    assert not result.terms_sound
    assert not result.valid


def test_tampered_balance_caught_even_after_reseal():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.4)
    pool.balance_usd = 999.0  # lie: more in the pool than was posted minus drawn
    pool.seal()
    assert not pool.verify().terms_sound


def test_tampered_forfeiture_caught_even_after_reseal():
    c1 = _contract("a", 100.0)
    pool = post_collateral_pool([c1], fraction=0.5)
    pool.draw(settle_contract(c1, cost_usd=150.0))  # a breach
    pool.contracts[0].forfeited_usd = 0.0  # lie: forfeit nothing
    pool.contracts[0].released_usd = pool.contracts[0].required_usd
    pool.seal()
    result = pool.verify()
    assert result.hash_ok
    assert not result.terms_sound


def test_tampered_required_caught():
    c1 = _contract("a", 100.0)
    pool = post_collateral_pool([c1], fraction=0.4)
    pool.contracts[0].required_usd = 999.0  # no longer fraction × price
    pool.seal()
    assert not pool.verify().terms_sound


def test_tampered_state_without_reseal_breaks_the_hash():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.3)
    pool.contracts[0].state = "released"  # do not reseal
    assert not pool.verify().hash_ok


def test_require_valid_raises_on_tamper():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.4)
    pool.posted_usd = 5.0
    pool.seal()
    with pytest.raises(SettlementError):
        pool.require_valid()


# -- signing ------------------------------------------------------------------


def test_sign_and_verify_with_a_verifier():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.3).sign(VENDOR, party="vendor")
    result = pool.verify(VENDOR, require=["vendor"])
    assert result.valid
    assert result.signatures_ok
    assert result.signed_by == ["vendor"]


def test_forged_signature_is_caught():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.3).sign(VENDOR, party="vendor")
    result = pool.verify(ACME, require=["vendor"])  # wrong key
    assert not result.signatures_ok
    assert not result.valid


def test_only_a_pool_party_can_sign():
    pool = post_collateral_pool([_contract("a", 100.0)], fraction=0.3)
    with pytest.raises(SettlementError):
        pool.sign(ACME, party="stranger")


def test_draw_clears_stale_signatures():
    c1 = _contract("a", 100.0)
    pool = post_collateral_pool([c1], fraction=0.3).sign(VENDOR, party="vendor")
    assert pool.signed_by == ["vendor"]
    pool.draw(settle_contract(c1, cost_usd=50.0))
    assert pool.signatures == []  # the content hash changed on the draw
    assert pool.verify().valid


# -- determinism & wire -------------------------------------------------------


def test_same_pool_hashes_identically_regardless_of_contract_order():
    c1, c2 = _contract("a", 100.0), _contract("b", 200.0)
    a = post_collateral_pool([c1, c2], fraction=0.4)
    b = post_collateral_pool([c2, c1], fraction=0.4)  # reversed order
    assert a.compute_hash() == b.compute_hash()


def test_wire_roundtrip_preserves_verification():
    c1, c2 = _contract("a", 100.0), _contract("b", 100.0)
    pool = post_collateral_pool([c1, c2], fraction=0.5)
    pool.draw(settle_contract(c1, cost_usd=150.0))  # a breach
    restored = CollateralPool.from_wire(pool.to_wire())
    assert restored.content_hash == pool.content_hash
    assert restored.verify().valid
    assert restored.drawn_usd == pool.drawn_usd


def test_module_level_draw_pool():
    c1 = _contract("a", 100.0)
    pool = post_collateral_pool([c1], fraction=0.5)
    pc = draw_pool(pool, settle_contract(c1, cost_usd=150.0))
    assert pc.state == "forfeited"
    assert pool.verify().valid


# -- app & book wiring --------------------------------------------------------


def test_app_post_collateral_pool_signs_and_audits():
    app = _app()
    c1, c2 = _contract("a", 100.0), _contract("b", 100.0)
    d = app.admit("vendor")
    pool = app.post_collateral_pool([c1, c2], decisions=d)
    assert pool.audit_id is not None
    assert pool.signed_by == ["acme"]  # the book owner co-signs as its side
    assert len(app.audit.query(action=COLLATERAL_ACTION)) == 1
    assert app.audit.verify_chain()
    assert pool.verify(app.contract_signer, require=["acme"]).valid


def test_app_settle_draws_against_an_attached_pool_in_place():
    app = _app()
    c1 = _contract("a", 100.0)
    pool = app.post_collateral_pool([c1], fraction=0.5)
    record = app.settle(c1, cost_usd=150.0, pool=pool)  # a breach
    assert record.status == "breached"
    assert pool.contract(c1.id).state == "forfeited"
    assert pool.contract(c1.id).settlement_hash == record.content_hash
    # Both the posting and the draw landed on the audit chain.
    assert len(app.audit.query(action=COLLATERAL_ACTION)) == 2
    assert app.audit.verify_chain()
    assert pool.verify(app.contract_signer).valid


def test_app_draw_pool_releases_on_clean_delivery():
    app = _app()
    c1 = _contract("a", 100.0)
    pool = app.post_collateral_pool([c1], fraction=0.5)
    record = app.settle(c1, cost_usd=50.0)
    resolved = app.draw_pool(pool, record)
    assert resolved.contract(c1.id).state == "released"
    assert app.settlement_book.verify(app.contract_signer).intact


def test_book_settle_with_pool_keeps_the_book_intact():
    app = _app()
    c1 = _contract("a", 100.0)
    pool = app.post_collateral_pool([c1], fraction=0.4)
    app.settle(c1, cost_usd=200.0, pool=pool)
    assert app.settlement_book.verify().intact
    assert pool.contract(c1.id).state == "forfeited"
