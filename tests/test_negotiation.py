"""Agent negotiation & contracting: bounded bargain, signed contract, reputation."""

from __future__ import annotations

import pytest

from vincio import ContextApp
from vincio.a2a import connect_a2a_in_process
from vincio.core.errors import ConfigError, ContractError, NegotiationError
from vincio.negotiation import (
    A2ANegotiator,
    Contract,
    ContractTerms,
    LocalParty,
    Negotiation,
    NegotiationBudget,
    NegotiationPosition,
    Offer,
    buyer_position,
    select_offer,
    seller_position,
)
from vincio.negotiation.engine import IssuePreference
from vincio.optimize.reputation import ReputationLedger
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner


def _app(name: str = "market") -> ContextApp:
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), model="mock-1")


def _buyer() -> NegotiationPosition:
    return buyer_position(
        max_price_usd=0.10,
        ideal_price_usd=0.0,
        max_sla_seconds=5.0,
        ideal_sla_seconds=0.5,
        min_quality=0.7,
        ideal_quality=1.0,
    )


def _seller() -> NegotiationPosition:
    return seller_position(
        min_price_usd=0.04,
        ideal_price_usd=0.14,
        min_sla_seconds=1.0,
        ideal_sla_seconds=6.0,
        max_quality=0.95,
        ideal_quality=0.7,
    )


# -- utility & concession ----------------------------------------------------


def test_issue_utility_direction_and_bounds():
    # Buyer prefers low price: ideal below reserve.
    issue = IssuePreference(name="price_usd", ideal=0.0, reserve=0.10)
    assert issue.utility(0.0) == 1.0  # ideal
    assert issue.utility(0.10) == 0.0  # reservation
    assert 0.0 < issue.utility(0.05) < 1.0
    # Beyond the reservation is negative (genuinely penalized); above the ideal caps.
    assert issue.utility(0.20) < 0.0
    assert issue.utility(-0.05) == 1.0


def test_concession_is_monotone_toward_reservation():
    pos = _seller()
    levels = [pos.concession_level(t, 8) for t in range(9)]
    assert levels[0] == 1.0
    assert all(a >= b - 1e-12 for a, b in zip(levels, levels[1:], strict=False))


def test_position_coherence_validation():
    with pytest.raises(NegotiationError):
        NegotiationPosition(role="buyer", issues=[], concession=1.0).validate_coherent()
    with pytest.raises(NegotiationError):
        NegotiationPosition(
            role="buyer",
            issues=[IssuePreference(name="price_usd", ideal=0.0, reserve=0.1)],
            concession=0.0,
        ).validate_coherent()


# -- the bargain -------------------------------------------------------------


@pytest.mark.asyncio
async def test_negotiation_reaches_agreement_within_budget():
    app = _app()
    result = await app.anegotiate(
        "transcribe calls",
        buyer=_buyer(),
        seller=_seller(),
        budget=NegotiationBudget(max_rounds=8),
        buyer_id="acme",
        seller_id="vendor",
    )
    assert result.status == "agreement"
    assert result.agreed
    assert 0 < result.rounds <= 8
    terms = result.contract.terms
    # Agreed terms sit inside both parties' acceptable regions.
    assert 0.04 <= terms.price_usd <= 0.10
    assert 1.0 <= terms.sla_seconds <= 5.0
    assert 0.7 <= terms.quality_floor <= 0.95


@pytest.mark.asyncio
async def test_termination_guaranteed_when_no_overlap():
    # Buyer will pay at most 0.02; seller wants at least 0.10 — empty ZOPA.
    buyer = buyer_position(
        max_price_usd=0.02, ideal_price_usd=0.0, max_sla_seconds=2.0, min_quality=0.9
    )
    seller = seller_position(
        min_price_usd=0.10, ideal_price_usd=0.20, min_sla_seconds=5.0, max_quality=0.5
    )
    app = _app()
    result = await app.anegotiate(
        "job", buyer=buyer, seller=seller, budget=NegotiationBudget(max_rounds=6)
    )
    assert result.status in ("no_agreement", "walk_away")
    assert result.contract is None
    assert result.rounds <= 6
    # Partial result still carries the trace.
    assert result.offers
    assert result.last_buyer_offer is not None
    assert result.last_seller_offer is not None


@pytest.mark.asyncio
async def test_deadline_returns_partial_result():
    # A clock that jumps past the deadline immediately on the first check.
    ticks = iter([0.0, 100.0, 200.0, 300.0])

    def clock() -> float:
        try:
            return next(ticks)
        except StopIteration:
            return 999.0

    neg = Negotiation(
        LocalParty("b", _buyer()),
        LocalParty("s", _seller()),
        budget=NegotiationBudget(max_rounds=8, deadline_s=1.0),
        clock=clock,
    )
    result = await neg.run("job")
    assert result.status == "no_agreement"
    assert result.deadline_hit is True


def test_budget_validation():
    with pytest.raises(NegotiationError):
        NegotiationBudget(max_rounds=0).validate_coherent()
    with pytest.raises(NegotiationError):
        NegotiationBudget(deadline_s=-1.0).validate_coherent()


# -- the contract ------------------------------------------------------------


@pytest.mark.asyncio
async def test_contract_is_signed_and_offline_verifiable():
    app = _app()
    result = await app.anegotiate("job", buyer=_buyer(), seller=_seller(), buyer_id="acme", seller_id="vendor")
    contract = result.contract
    assert contract is not None
    assert contract.fully_signed
    verification = contract.verify(app.contract_signer)
    assert verification.valid
    assert verification.hash_ok and verification.signatures_ok
    assert set(verification.signed_by) == {"acme", "vendor"}


@pytest.mark.asyncio
async def test_contract_tamper_is_detected():
    app = _app()
    result = await app.anegotiate("job", buyer=_buyer(), seller=_seller())
    contract = result.contract
    contract.terms.price_usd += 0.5  # tamper a term after signing
    assert not contract.verify(app.contract_signer).valid


def test_contract_signature_verifies_cross_process_with_shared_key():
    signer = HMACSigner("shared-secret", key_id="org")
    contract = Contract(
        buyer="acme", seller="vendor", terms=ContractTerms(scope="x", price_usd=0.05)
    ).seal()
    contract.sign(signer, party="acme")
    contract.sign(signer, party="vendor")
    # A fresh verifier holding only the key validates from the bytes alone.
    fresh = HMACSigner("shared-secret", key_id="org")
    assert contract.verify(fresh).valid
    # The wrong key fails.
    assert not contract.verify(HMACSigner("other", key_id="org")).valid


def test_resigning_replaces_not_accumulates():
    signer = HMACSigner("k")
    contract = Contract(buyer="a", seller="b", terms=ContractTerms(price_usd=0.1)).seal()
    contract.sign(signer, party="a")
    contract.sign(signer, party="a")
    assert contract.signed_by.count("a") == 1


def test_contract_to_budget_and_check():
    contract = Contract(
        buyer="a",
        seller="b",
        terms=ContractTerms(scope="x", price_usd=0.10, sla_seconds=3.0, quality_floor=0.8),
    ).seal()
    budget = contract.to_budget()
    assert budget.max_cost_usd == 0.10
    assert budget.max_latency_ms == 3000
    # Fulfilled within terms.
    ok = contract.check(cost_usd=0.08, latency_ms=2500, quality=0.9)
    assert ok.fulfilled and not ok.breaches
    # Breach on all three dimensions.
    bad = contract.check(cost_usd=0.20, latency_ms=4000, quality=0.5)
    assert not bad.fulfilled
    assert len(bad.breaches) == 3
    with pytest.raises(ContractError):
        contract.check(cost_usd=0.20, raise_on_breach=True)


def test_require_valid_raises_on_bad_signature():
    contract = Contract(buyer="a", seller="b", terms=ContractTerms(price_usd=0.1)).seal()
    with pytest.raises(ContractError):
        contract.require_valid(HMACSigner("k"))  # no signatures present


# -- reputation weighting ----------------------------------------------------


@pytest.mark.asyncio
async def test_reputation_discounts_a_regressing_seller():
    app = _app()
    ledger = app.use_reputation_ledger()
    for _ in range(30):
        ledger.record_outcome("vendor", passed=False, round_id="r")
        ledger.record_outcome("trusty", passed=True, round_id="r")
    bad = await app.anegotiate("job", buyer=_buyer(), seller=_seller(), buyer_id="acme", seller_id="vendor")
    good = await app.anegotiate("job", buyer=_buyer(), seller=_seller(), buyer_id="acme", seller_id="trusty")
    # Both still close a deal — the regressor is discounted, not singled out.
    assert bad.agreed and good.agreed
    # The discounted seller had to concede more, so the buyer pays no more.
    assert bad.contract.terms.price_usd <= good.contract.terms.price_usd + 1e-9


@pytest.mark.asyncio
async def test_select_offer_prefers_the_reliable_seller():
    app = _app()
    ledger = app.use_reputation_ledger()
    for _ in range(20):
        ledger.record_outcome("vendor", passed=False, round_id="r")
        ledger.record_outcome("trusty", passed=True, round_id="r")
    buyer = _buyer()
    bad = await app.anegotiate("job", buyer=buyer, seller=_seller(), buyer_id="acme", seller_id="vendor")
    good = await app.anegotiate("job", buyer=buyer, seller=_seller(), buyer_id="acme", seller_id="trusty")
    best = select_offer([bad, good], buyer, reputation=ledger)
    assert best is not None and best.seller == "trusty"


def test_reputation_weight_is_bounded():
    ledger = ReputationLedger()
    for _ in range(50):
        ledger.record_outcome("bad", passed=False, round_id="r")
    w = ledger.weight("bad")
    assert ledger.config.weight_floor <= w <= ledger.config.weight_ceiling


@pytest.mark.asyncio
async def test_enforce_contract_records_reputation_on_breach():
    app = _app()
    ledger = app.use_reputation_ledger()
    result = await app.anegotiate("job", buyer=_buyer(), seller=_seller(), seller_id="vendor")
    before = ledger.weight("vendor")
    app.enforce_contract(result.contract, cost_usd=999.0, latency_ms=999999, quality=0.0)
    after = ledger.weight("vendor")
    assert after < before  # a breach debits the seller


# -- audit -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_outcome_and_contract_recorded_on_audit_chain():
    app = _app()
    result = await app.anegotiate("job", buyer=_buyer(), seller=_seller(), buyer_id="acme", seller_id="vendor")
    neg_entries = app.audit.query(action="negotiation")
    contract_entries = app.audit.query(action="contract_signed")
    assert neg_entries and neg_entries[-1].decision == "agreement"
    assert contract_entries and contract_entries[-1].resource == result.contract.id
    assert result.contract.audit_id == contract_entries[-1].id
    assert app.audit.verify_chain()


# -- A2A fabric --------------------------------------------------------------


@pytest.mark.asyncio
async def test_negotiation_over_a2a_matches_local():
    app = _app()
    seller = _seller()
    # Local reference.
    local = await app.anegotiate("job", buyer=_buyer(), seller=seller, buyer_id="acme", seller_id="vendor")
    # Same bargain, seller reached over A2A in-process.
    server = app.serve_negotiation(LocalParty("vendor", seller), name="vendor")
    client = connect_a2a_in_process(server)
    remote = A2ANegotiator(client, member_id="vendor", role="seller")
    over_a2a = await app.anegotiate("job", buyer=_buyer(), seller=remote, buyer_id="acme")
    assert over_a2a.status == local.status
    assert over_a2a.agreed
    assert over_a2a.contract.terms.canonical() == local.contract.terms.canonical()


@pytest.mark.asyncio
async def test_a2a_negotiator_pins_member_identity():
    app = _app()
    # A seller that lies about its party id on the wire is still recorded as the
    # directory-resolved member id, so reputation cannot be spoofed.
    server = app.serve_negotiation(LocalParty("spoofer", _seller()), name="vendor")
    client = connect_a2a_in_process(server)
    remote = A2ANegotiator(client, member_id="vendor", role="seller")
    offer = await remote.respond("job", _open_offer(), 1, NegotiationBudget(max_rounds=8))
    assert offer.party == "vendor"


def _open_offer() -> Offer:
    return Offer(
        party="acme",
        role="buyer",
        terms=ContractTerms(scope="job", price_usd=0.0, sla_seconds=0.5, quality_floor=1.0),
        round_index=0,
    )


def test_offer_wire_round_trip():
    offer = _open_offer()
    restored = Offer.from_wire(offer.to_wire())
    assert restored.party == offer.party
    assert restored.terms.canonical() == offer.terms.canonical()
    assert restored.round_index == offer.round_index


def test_negotiate_rejects_wrong_role_position():
    app = _app()
    with pytest.raises(ConfigError):
        app.negotiate("job", buyer=_seller(), seller=_seller())
