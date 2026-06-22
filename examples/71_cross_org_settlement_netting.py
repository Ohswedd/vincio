"""Cross-org settlement netting & multilateral clearing — closing the books once.

Vincio lets agents negotiate contracts, run durable cross-org sagas, and settle the
delivered work into signed, offline-verifiable records. This example adds the next
rung: **netting** those bilateral settlements across a whole fleet into a single
minimal set of net obligations, so an org that is both a buyer and a seller across a
web of contracts closes its books once. It is a library-side clearing calculation,
never a hosted clearing house or a payment rail.

Five steps, all offline and deterministic:

  1. A fleet of bilateral settlements — three orgs, each owing the next.
  2. Net them: the directed payables fold into one net position per org, summing to
     zero, and clear to the minimal set of transfers (fewer than the gross edges).
  3. Offline-verifiable: the cleared set is content-bound and signed; it recomputes
     from the bytes alone — positions balance, transfers reproduce them.
  4. A dispute is pinpointed: two books that disagree on a contract are flagged, not
     silently netted away.
  5. One org closes its own book: its net position against each counterparty.

Everything here is opt-in and additive; this is a library capability inside your
process, never a hosted clearing house or a payment processor.
"""

from __future__ import annotations

from vincio import ContextApp, net_settlements, settle_contract
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.settlement import SettlementBook


def a_contract(buyer: str, seller: str, price: float) -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope="work", price_usd=price)
    ).seal()


def main() -> None:
    app = ContextApp(name="clearer", provider=MockProvider(default_text="ok"))

    # 1. A fleet of bilateral settlements: acme→vendor $0.10, vendor→data $0.06,
    #    data→acme $0.04 — a cycle of obligations across three orgs.
    fleet = [
        settle_contract(a_contract("acme", "vendor", 0.10), cost_usd=0.08),
        settle_contract(a_contract("vendor", "data", 0.06), cost_usd=0.05),
        settle_contract(a_contract("data", "acme", 0.04), cost_usd=0.03),
    ]
    print(f"1. Fleet: {len(fleet)} bilateral settlements across 3 orgs.")

    # 2. Net the fleet: directed payables fold into one net position per org.
    netting = app.clear_settlements(records=fleet)
    print(
        f"2. Netted {netting.settlements} settlements: {netting.gross_edges} gross "
        f"obligations → {netting.cleared_transfers} cleared transfers "
        f"(${netting.total_gross_usd:.2f} owed gross, only ${netting.total_cleared_usd:.2f} moves)."
    )
    for p in netting.positions:
        side = "creditor" if p.is_creditor else "debtor" if p.is_debtor else "flat"
        print(f"     position {p.party}: net ${p.net_usd:+.2f} ({side})")
    for o in netting.obligations:
        print(f"     transfer {o.debtor} → {o.creditor}: ${o.amount_usd:.2f}")

    # 3. The cleared set is content-bound and offline-verifiable.
    verdict = netting.verify(app.contract_signer)
    print(
        f"3. Verifies offline={verdict.valid} (hash={verdict.hash_ok}, "
        f"positions balance={verdict.positions_balanced}, conserves={verdict.conserves}); "
        f"signed by {netting.signed_by}; on the audit chain={bool(app.audit.query(action='netting'))}."
    )

    # 4. A disagreement is pinpointed as a dispute, never silently netted.
    c = a_contract("acme", "vendor", 0.10)
    disputed = net_settlements(
        [settle_contract(c, cost_usd=0.08), settle_contract(c, cost_usd=0.09)]
    )
    print(
        f"4. Disagreeing books: clean={disputed.clean}; disputes pinpointed="
        f"{[d.contract_id[:12] for d in disputed.disputes]} (excluded from the clearing)."
    )

    # 5. One org closes its own book: its net position against each counterparty.
    book = SettlementBook("acme")
    book.settle(a_contract("acme", "vendor", 0.10), cost_usd=0.08)
    book.settle(a_contract("data", "acme", 0.07), cost_usd=0.05)  # acme is the seller here
    own = book.net()
    acme = own.position("acme")
    print(
        f"5. acme closes its book: owes ${acme.owed_usd:.2f}, is owed ${acme.due_usd:.2f} "
        f"→ net ${acme.net_usd:+.2f}; cleared in {own.cleared_transfers} transfer(s)."
    )


if __name__ == "__main__":
    main()
