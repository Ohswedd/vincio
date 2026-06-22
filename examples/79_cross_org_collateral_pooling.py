"""Cross-org collateral pooling & cross-contract margin.

An ``Escrow`` backs *one* contract with collateral held against its delivery — but a
counterparty running many concurrent contracts must lock **separate** collateral per deal,
even though its breaches and clean deliveries across those contracts net out. Capital is
stranded contract-by-contract the way bilateral settlements were stranded book-by-book
before netting folded them. This example adds the next rung: a ``CollateralPool`` — a
single posted stake (a margin account) that backs many contracts at a deterministic,
offline-verifiable allocation, the collateral analogue of the ``NettingSet``.

Five steps, all offline and deterministic:

  1. A vendor admitted on conservative terms **posts one stake** backing three concurrent
     contracts at once, each allocated a share proportional to its admission-required
     collateral — capital posted once, not locked deal-by-deal.
  2. A clean delivery **frees** its committed capital back to the available balance, ready
     to back the next contract — the shared stake is reused, not stranded.
  3. A breach **draws** a bounded, pinpointed slice from the shared stake (proportional to
     the shortfall the settlement measured) and releases the rest — covered from the pool,
     never the whole stake, never punitive.
  4. Backing a fourth contract over-commits the pool, surfacing a bounded **top-up**
     obligation rather than silently over-committing; topping up clears it.
  5. The whole lifecycle is offline-verifiable: the allocations re-derive and the balance
     reconciles from the bytes (a tampered allocation is caught even after re-sealing), and
     every post, draw, and top-up lands on the hash-chained audit log.

Everything here is opt-in and additive; this is a library capability inside your process,
never a hosted clearing house, a margin custodian, or a payment rail.
"""

from __future__ import annotations

from vincio import AdmissionConfig, ContextApp
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider


def a_contract(buyer: str, seller: str, scope: str, price: float) -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope=scope, price_usd=price)
    ).seal()


def main() -> None:
    buyer = ContextApp(name="acme", provider=MockProvider(default_text="ok"))
    buyer.use_reputation_ledger()
    buyer.use_settlement_book()

    # A thin, brand-new counterparty is admitted on conservative terms — including a
    # required collateral fraction that backs every deal it runs.
    decision = buyer.admit("vendor", config=AdmissionConfig(parity_exposure_usd=1000.0))
    c1 = a_contract("acme", "vendor", "transcribe batch A", price=100.0)
    c2 = a_contract("acme", "vendor", "transcribe batch B", price=200.0)
    c3 = a_contract("acme", "vendor", "transcribe batch C", price=300.0)

    # 1. Post one stake backing all three contracts. Each is allocated a share proportional
    #    to its admission-required collateral — capital posted once, not deal-by-deal.
    pool = buyer.post_collateral_pool([c1, c2, c3], decisions=decision)
    print(
        f"1. Posted one ${pool.posted_usd:,.2f} stake backing {len(pool.contracts)} contracts "
        f"({decision.escrow_fraction:.0%} of each price): "
        + ", ".join(f"{c.contract_id[-4:]}→${c.allocated_usd:,.2f}" for c in pool.contracts)
        + f" — verifiable offline: {pool.verify().valid}."
    )

    # 2. A clean delivery frees its committed capital back to the available balance.
    buyer.settle(c1, cost_usd=60.0, pool=pool)  # within the agreed $100
    print(
        f"2. Clean delivery of A: ${pool.contract(c1.id).released_usd:,.2f} freed — "
        f"committed now ${pool.required_open_usd:,.2f}, available ${pool.available_usd:,.2f} "
        f"to back the next deal. Nothing drawn from the stake."
    )

    # 3. A breach draws a bounded slice from the shared stake, proportional to the shortfall.
    buyer.settle(c2, cost_usd=300.0, pool=pool)  # 50% over the $200 price
    drawn = pool.contract(c2.id)
    print(
        f"3. Breach on B ({drawn.shortfall_fraction:.0%} over): ${drawn.forfeited_usd:,.2f} "
        f"drawn from the shared stake ({', '.join(drawn.breaches)}), "
        f"${drawn.released_usd:,.2f} freed — pool balance now ${pool.balance_usd:,.2f}, "
        f"never the whole stake."
    )

    # 4. Backing a fourth contract over-commits the pool, surfacing a bounded top-up.
    c4 = a_contract("acme", "vendor", "transcribe batch D", price=400.0)
    pool.back(c4, decision=decision)
    print(
        f"4. Backing a 4th contract over-commits: open contracts need "
        f"${pool.required_open_usd:,.2f} against ${pool.balance_usd:,.2f} held — top-up "
        f"obligation ${pool.topup_usd:,.2f}. "
    )
    pool.top_up(pool.topup_usd)
    print(
        f"   Topped up to ${pool.posted_usd:,.2f}: obligation cleared "
        f"(top-up ${pool.topup_usd:,.2f}), every open contract covered."
    )

    # 5. Offline-verifiable: the allocations re-derive and the balance reconciles; a tampered
    #    allocation is caught even after re-sealing, and every transition is on the chain.
    tampered = buyer.post_collateral_pool([c3], fraction=0.5)
    tampered.contracts[0].allocated_usd = 9_999.0  # lie about what backs the deal
    tampered.seal()  # recompute the hash to match the tampered allocation
    print(
        f"5. Offline-verifiable: the live pool verifies={pool.verify(buyer.contract_signer).valid}; "
        f"a re-sealed inflated allocation is still caught (terms_sound="
        f"{tampered.verify().terms_sound}); "
        f"{len(buyer.audit.query(action='collateral_pool'))} pool transitions on the audit "
        f"chain, book intact={buyer.settlement_book.verify(buyer.contract_signer).intact}."
    )


if __name__ == "__main__":
    main()
