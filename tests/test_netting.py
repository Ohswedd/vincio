"""Cross-org settlement netting & multilateral clearing.

Folding a fleet's bilateral settlement books into the minimal set of net
obligations: dedup across both sides, dispute pinpointing, multilateral clearing,
and the content-bound, offline-verifiable NettingSet.
"""

from __future__ import annotations

import pytest

from vincio import ContextApp, NettingSet, net_books, net_settlements, settle_contract
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement import (
    BilateralNet,
    NetObligation,
    NetPosition,
    NettingDispute,
    SettlementBook,
)


def _app(name: str = "clearer") -> ContextApp:
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), model="mock-1")


def _contract(buyer: str, seller: str, price: float = 0.10, **terms) -> Contract:
    base = {"scope": "work", "price_usd": price}
    base.update(terms)
    return Contract(buyer=buyer, seller=seller, terms=ContractTerms(**base)).seal()


def _settled(buyer: str, seller: str, price: float = 0.10, cost: float = 0.05):
    return settle_contract(_contract(buyer, seller, price), cost_usd=cost)


# -- gross obligations & positions --------------------------------------------


def test_directed_obligation_from_each_settlement():
    ns = net_settlements([_settled("acme", "vendor", 0.10)])
    assert ns.gross_edges == 1
    g = ns.gross[0]
    assert g.debtor == "acme" and g.creditor == "vendor" and g.amount_usd == 0.10
    assert g.settlements == 1


def test_payables_aggregate_per_directed_pair():
    # Two contracts from acme to vendor sum into one gross obligation.
    ns = net_settlements([_settled("acme", "vendor", 0.10), _settled("acme", "vendor", 0.05)])
    assert ns.gross_edges == 1
    assert ns.gross[0].amount_usd == pytest.approx(0.15)
    assert ns.gross[0].settlements == 2


def test_net_positions_sum_to_zero():
    ns = net_settlements(
        [_settled("a", "b", 0.10), _settled("b", "c", 0.06), _settled("c", "a", 0.04)]
    )
    assert sum(p.net_usd for p in ns.positions) == pytest.approx(0.0)
    pos = {p.party: p.net_usd for p in ns.positions}
    assert pos["a"] == pytest.approx(-0.06)
    assert pos["b"] == pytest.approx(0.04)
    assert pos["c"] == pytest.approx(0.02)


def test_position_owed_and_due_break_down():
    ns = net_settlements([_settled("a", "b", 0.10), _settled("c", "a", 0.03)])
    a = ns.position("a")
    assert isinstance(a, NetPosition)
    assert a.owed_usd == pytest.approx(0.10)  # a owes b
    assert a.due_usd == pytest.approx(0.03)  # c owes a
    assert a.net_usd == pytest.approx(-0.07)
    assert a.is_debtor and not a.is_creditor


def test_self_dealing_and_zero_price_are_dropped():
    ns = net_settlements(
        [_settled("a", "a", 0.10), _settled("a", "b", 0.0), _settled("a", "b", 0.05)]
    )
    assert ns.gross_edges == 1
    assert ns.gross[0].amount_usd == 0.05


# -- bilateral netting --------------------------------------------------------


def test_bilateral_net_collapses_two_directions():
    ns = net_settlements([_settled("acme", "vendor", 0.10), _settled("vendor", "acme", 0.04)])
    assert len(ns.bilateral) == 1
    bn = ns.bilateral[0]
    assert isinstance(bn, BilateralNet)
    assert {bn.party_low, bn.party_high} == {"acme", "vendor"}
    assert bn.net_debtor == "acme" and bn.net_creditor == "vendor"
    assert bn.net_amount_usd == pytest.approx(0.06)


def test_equal_opposing_flows_net_to_zero():
    ns = net_settlements([_settled("acme", "vendor", 0.10), _settled("vendor", "acme", 0.10)])
    bn = ns.bilateral[0]
    assert bn.net_amount_usd == 0.0 and bn.net_debtor == "" and bn.net_creditor == ""
    assert ns.cleared_transfers == 0  # nothing to move


# -- multilateral clearing ----------------------------------------------------


def test_cycle_clears_to_fewer_transfers():
    # A 3-org cycle (3 gross edges) clears to 2 transfers (<= N-1).
    ns = net_settlements(
        [_settled("a", "b", 0.10), _settled("b", "c", 0.06), _settled("c", "a", 0.04)]
    )
    assert ns.gross_edges == 3
    assert ns.cleared_transfers == 2
    assert ns.reduction == 1
    assert all(isinstance(o, NetObligation) for o in ns.obligations)
    # The debtor 'a' funds both creditors; total moved is far below the gross.
    assert ns.total_gross_usd == pytest.approx(0.20)
    assert ns.total_cleared_usd == pytest.approx(0.06)


def test_cleared_set_is_at_most_n_minus_one():
    recs = [_settled("a", "b", 0.10), _settled("b", "c", 0.10), _settled("c", "d", 0.10)]
    ns = net_settlements(recs)
    n = len([p for p in ns.positions if abs(p.net_usd) > 1e-9])
    assert ns.cleared_transfers <= max(0, n - 1)


def test_clearing_conserves_every_position():
    ns = net_settlements(
        [_settled("a", "b", 0.20), _settled("b", "c", 0.05), _settled("a", "c", 0.07)]
    )
    flow = {p.party: 0.0 for p in ns.positions}
    for o in ns.obligations:
        flow[o.creditor] += o.amount_usd
        flow[o.debtor] -= o.amount_usd
    for p in ns.positions:
        assert flow[p.party] == pytest.approx(p.net_usd)


def test_clearing_is_deterministic():
    recs = [_settled("a", "b", 0.10), _settled("b", "c", 0.06), _settled("c", "a", 0.04)]
    a = net_settlements(recs)
    b = net_settlements(list(reversed(recs)))
    assert a.content_hash == b.content_hash
    assert [(o.debtor, o.creditor, o.amount_usd) for o in a.obligations] == [
        (o.debtor, o.creditor, o.amount_usd) for o in b.obligations
    ]


# -- dedup & dispute pinpointing ----------------------------------------------


def test_same_settlement_from_both_sides_is_deduped():
    c = _contract("acme", "vendor", 0.10)
    buyer_rec = settle_contract(c, cost_usd=0.08)
    seller_rec = settle_contract(c, cost_usd=0.08)  # co-signs the same hash
    assert buyer_rec.content_hash == seller_rec.content_hash
    ns = net_settlements([buyer_rec, seller_rec])
    assert ns.settlements == 1  # not double-counted
    assert ns.gross_edges == 1
    assert ns.gross[0].amount_usd == 0.10


def test_disagreement_is_a_dispute_not_silently_netted():
    c = _contract("acme", "vendor", 0.10)
    ours = settle_contract(c, cost_usd=0.08)
    theirs = settle_contract(c, cost_usd=0.09)  # different delivered cost
    ns = net_settlements([ours, theirs])
    assert not ns.clean
    assert len(ns.disputes) == 1
    d = ns.disputes[0]
    assert isinstance(d, NettingDispute)
    assert d.contract_id == c.id
    assert len(d.hashes) == 2 and sorted(d.parties) == ["acme", "vendor"]
    assert ns.settlements == 0  # the disputed contract is excluded


def test_require_clean_raises_on_dispute():
    c = _contract("acme", "vendor", 0.10)
    ns = net_settlements([settle_contract(c, cost_usd=0.08), settle_contract(c, cost_usd=0.09)])
    with pytest.raises(SettlementError):
        ns.require_clean()


def test_dispute_excludes_only_the_disputed_contract():
    c1 = _contract("acme", "vendor", 0.10)
    c2 = _contract("acme", "data", 0.05)
    ns = net_settlements(
        [
            settle_contract(c1, cost_usd=0.08),
            settle_contract(c1, cost_usd=0.09),  # disputes c1
            settle_contract(c2, cost_usd=0.04),  # c2 still nets
        ]
    )
    assert len(ns.disputes) == 1 and ns.disputes[0].contract_id == c1.id
    assert ns.settlements == 1
    assert ns.position("data").net_usd == pytest.approx(0.05)


def test_tampered_source_record_is_refused():
    rec = _settled("acme", "vendor", 0.10)
    rec.amount_owed_usd = 999.0  # tamper without resealing
    with pytest.raises(SettlementError):
        net_settlements([rec])


def test_forged_signature_is_refused_with_verifier():
    rec = _settled("acme", "vendor", 0.10).sign(HMACSigner("real", key_id="acme"), party="acme")
    # A different key cannot verify the signature → refuse to net.
    with pytest.raises(SettlementError):
        net_settlements([rec], verifier=HMACSigner("other", key_id="acme"))


# -- the netting set: content-bound & offline-verifiable ----------------------


def test_set_signs_and_verifies_offline():
    ns = net_settlements([_settled("a", "b", 0.10), _settled("b", "c", 0.04)], owner="clearer")
    signer = HMACSigner("k", key_id="clearer")
    ns.sign(signer, party="clearer")
    verdict = ns.verify(signer)
    assert verdict.valid and verdict.hash_ok and verdict.conserves
    assert verdict.positions_balanced and verdict.signatures_ok
    assert ns.signed_by == ["clearer"]


def test_tampered_obligation_is_caught():
    ns = net_settlements([_settled("a", "b", 0.10), _settled("b", "c", 0.04)])
    signer = HMACSigner("k")
    ns.sign(signer, party="a")
    ns.obligations[0].amount_usd = 999.0
    verdict = ns.verify(signer)
    assert not verdict.valid
    assert not verdict.hash_ok  # the netting hash no longer recomputes


def test_tampered_position_breaks_conservation():
    ns = net_settlements([_settled("a", "b", 0.10), _settled("b", "c", 0.04)])
    # Edit a position without re-sealing: the hash breaks; were it resealed,
    # conservation would still fail because the obligations no longer reproduce it.
    ns.positions[0].net_usd = ns.positions[0].net_usd + 5.0
    reverify = ns.verify()
    assert not reverify.hash_ok
    ns.seal()
    after = ns.verify()
    assert not after.positions_balanced or not after.conserves
    assert not after.valid


def test_resealed_wrong_amount_breaks_conservation():
    # A fabricated transfer amount between two real parties: re-sealing restores the
    # hash, but the cleared transfers no longer reproduce the positions.
    ns = net_settlements([_settled("a", "b", 0.10), _settled("b", "c", 0.04)])
    ns.obligations[0].amount_usd = round(ns.obligations[0].amount_usd + 0.05, 9)
    ns.seal()
    verdict = ns.verify()
    assert verdict.hash_ok and verdict.positions_balanced
    assert not verdict.conserves and not verdict.valid


def test_transfer_to_party_outside_fleet_is_unconserved():
    ns = net_settlements([_settled("a", "b", 0.10)])
    ns.obligations = [NetObligation(debtor="a", creditor="ghost", amount_usd=0.10)]
    ns.seal()
    assert not ns.verify().conserves


def test_two_clearers_compute_the_same_hash():
    recs = [_settled("a", "b", 0.10), _settled("b", "c", 0.06), _settled("c", "a", 0.04)]
    one = net_settlements(recs, owner="x")
    two = net_settlements(recs, owner="y")
    # Owner is metadata, not an economic term → both co-sign the same hash.
    assert one.content_hash == two.content_hash


def test_require_valid_raises_on_broken_set():
    ns = net_settlements([_settled("a", "b", 0.10)])
    ns.obligations.append(NetObligation(debtor="a", creditor="ghost", amount_usd=1.0))
    ns.seal()
    with pytest.raises(SettlementError):
        ns.require_valid()


def test_set_wire_roundtrip():
    ns = net_settlements([_settled("a", "b", 0.10), _settled("b", "c", 0.04)], owner="clearer")
    ns.sign(HMACSigner("k", key_id="clearer"), party="clearer")
    back = NettingSet.from_wire(ns.to_wire())
    assert back.content_hash == ns.content_hash
    assert back.verify().valid
    assert back.signed_by == ["clearer"]


def test_obligations_for_party():
    ns = net_settlements([_settled("a", "b", 0.10), _settled("b", "c", 0.06), _settled("c", "a", 0.04)])
    a_obls = ns.obligations_for("a")
    assert a_obls and all("a" in (o.debtor, o.creditor) for o in a_obls)


# -- net_books & the single-org view ------------------------------------------


def test_net_books_clears_a_fleet():
    b1 = SettlementBook("acme", signer=HMACSigner("k1", key_id="acme"))
    b1.settle(_contract("acme", "vendor", 0.10), cost_usd=0.05)
    b2 = SettlementBook("vendor", signer=HMACSigner("k2", key_id="vendor"))
    b2.settle(_contract("vendor", "acme", 0.04), cost_usd=0.02)
    ns = net_books([b1, b2], owner="clearer")
    assert ns.cleared_transfers == 1
    o = ns.obligations[0]
    assert o.debtor == "acme" and o.creditor == "vendor"
    assert o.amount_usd == pytest.approx(0.06)
    assert ns.verify().valid


def test_net_books_require_intact_catches_tamper():
    b1 = SettlementBook("acme")
    b1.settle(_contract("acme", "vendor", 0.10), cost_usd=0.05)
    b1.records[0].balance_usd = 1.0  # tamper the book
    with pytest.raises(SettlementError):
        net_books([b1], require_intact=True)


def test_book_nets_its_own_records():
    book = SettlementBook("acme", signer=HMACSigner("k", key_id="acme"))
    book.settle(_contract("acme", "vendor", 0.10), cost_usd=0.05)
    book.settle(_contract("data", "acme", 0.07), cost_usd=0.04)  # acme is the seller here
    ns = book.net()
    assert ns.owner == "acme"
    assert ns.signed_by == ["acme"]  # signed as the owner
    acme = ns.position("acme")
    assert acme.net_usd == pytest.approx(-0.03)  # owes vendor 0.10, owed 0.07
    assert ns.verify(HMACSigner("k", key_id="acme")).valid


# -- the app surface ----------------------------------------------------------


def test_app_clear_settlements_signs_and_audits():
    app = _app("clearer")
    book = SettlementBook("acme")
    book.settle(_contract("acme", "vendor", 0.10), cost_usd=0.05)
    book.settle(_contract("vendor", "acme", 0.04), cost_usd=0.02)
    netting = app.clear_settlements(books=[book])
    assert netting.owner == "clearer"
    assert "clearer" in netting.signed_by
    assert netting.verify(app.contract_signer).valid
    assert app.audit.query(action="netting")
    assert app.audit.verify_chain()


def test_app_clear_settlements_uses_attached_book():
    app = _app("acme")
    app.use_settlement_book()
    app.settle(_contract("acme", "vendor", 0.10), cost_usd=0.05)
    app.settle(_contract("data", "acme", 0.06), cost_usd=0.03)
    netting = app.clear_settlements()
    assert netting.settlements == 2
    assert netting.verify().valid


def test_app_clear_settlements_records_and_disputes_audit_decision():
    app = _app("clearer")
    c = _contract("acme", "vendor", 0.10)
    netting = app.clear_settlements(
        records=[settle_contract(c, cost_usd=0.08), settle_contract(c, cost_usd=0.09)]
    )
    assert not netting.clean
    entry = app.audit.query(action="netting")[0]
    assert entry.decision == "disputed"


def test_app_clear_empty_when_no_settlements():
    app = _app("clearer")
    netting = app.clear_settlements()
    assert netting.settlements == 0
    assert netting.obligations == []
    assert netting.verify().valid  # vacuously conserved


def test_clear_matches_min_scan_oracle_on_random_positions():
    # The heap-with-lazy-invalidation clearing must reproduce the two-min-scan
    # greedy byte-for-byte (the obligation sequence is content-bound).
    import random

    from vincio.settlement.netting import _TOLERANCE, _clear, _r

    def oracle(positions):
        debt = {p.party: -p.net_usd for p in positions if p.net_usd < -_TOLERANCE}
        credit = {p.party: p.net_usd for p in positions if p.net_usd > _TOLERANCE}
        obligations = []
        while debt and credit:
            d_party = min(debt, key=lambda p: (-debt[p], p))
            c_party = min(credit, key=lambda p: (-credit[p], p))
            transfer = _r(min(debt[d_party], credit[c_party]))
            if transfer <= _TOLERANCE:
                break
            obligations.append((d_party, c_party, transfer))
            debt[d_party] = _r(debt[d_party] - transfer)
            credit[c_party] = _r(credit[c_party] - transfer)
            if debt[d_party] <= _TOLERANCE:
                del debt[d_party]
            if credit[c_party] <= _TOLERANCE:
                del credit[c_party]
        return sorted(obligations)

    rng = random.Random(20260703)
    for _trial in range(300):
        n = rng.randint(0, 14)
        positions = []
        for i in range(n):
            a = _r(
                rng.choice([-1, 1])
                * rng.choice([0.0, 0.01, 0.05, 0.05, round(rng.uniform(0, 2), 2),
                              rng.uniform(0, 2)])
            )
            positions.append(
                NetPosition(party=f"org{i}", owed_usd=max(0.0, -a),
                            due_usd=max(0.0, a), net_usd=a)
            )
        got = [(o.debtor, o.creditor, o.amount_usd) for o in _clear(positions)]
        assert got == oracle(positions)
