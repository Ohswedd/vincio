"""Cross-org collateralized settlement & escrow.

Admission now sets a required collateral / escrow fraction on a thin or low-trust
counterparty's contract — but the fraction is still only a *number stamped on the
terms*; nothing **holds** it, releases it on a clean delivery, or forfeits a slice on a
breach. A counterparty admitted on conservative terms posts no actual collateral, so the
escrow the admission policy asked for has no teeth, and a breach is still only debited to
reputation after the fact. This example adds the next rung: an ``Escrow`` that makes the
posted collateral a **verifiable, offline escrow bound to the contract** — held against
delivery and settled deterministically.

Five steps, all offline and deterministic:

  1. Admission sets a required escrow fraction on a thin counterparty's contract, and an
     ``Escrow`` **posts** that collateral — bound to the specific contract, the held
     amount re-deriving from the admission posture.
  2. A clean delivery **releases** the whole stake back to the poster — the collateral
     was held against delivery, and delivery held.
  3. A breach **forfeits** a bounded, pinpointed slice proportional to the shortfall the
     settlement measured — never the whole stake, never punitive — and releases the rest.
  4. Every post, release, and forfeiture is offline-verifiable: the disposition
     re-derives from the bytes, so a tampered forfeiture is caught from the bytes alone,
     and every transition lands on the hash-chained audit log.
  5. The escrow folds into the existing settlement path: ``app.settle(escrow=...)``
     resolves the collateral in place against the same record verdict the books close on
     — the collateral closes the same loop the settlement does.

Everything here is opt-in and additive; this is a library capability inside your
process, never a hosted escrow service or a payment rail.
"""

from __future__ import annotations

from vincio import AdmissionConfig, ContextApp
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider


def a_contract(buyer: str, seller: str, price: float) -> Contract:
    return Contract(
        buyer=buyer,
        seller=seller,
        terms=ContractTerms(scope="transcribe 1k calls", price_usd=price),
    ).seal()


def main() -> None:
    buyer = ContextApp(name="acme", provider=MockProvider(default_text="ok"))
    buyer.use_reputation_ledger()
    buyer.use_settlement_book()

    # A thin, brand-new counterparty is admitted on conservative terms — including a
    # required escrow fraction that backs the deal.
    decision = buyer.admit("vendor", config=AdmissionConfig(parity_exposure_usd=1000.0))
    contract = a_contract("acme", "vendor", price=100.0)

    # 1. Post the admission-required collateral as a content-bound escrow, bound to this
    #    specific contract. The held amount re-derives from the admission posture.
    escrow = buyer.post_escrow(contract, decision=decision)
    print(
        f"1. Posted escrow: ${escrow.amount_usd:,.2f} held "
        f"({escrow.escrow_fraction:.0%} of the ${contract.terms.price_usd:,.0f} contract), "
        f"bound to {escrow.contract_id} — verifiable offline: {escrow.verify().valid}."
    )

    # 2. A clean delivery (cost within the agreed price) releases the whole stake.
    clean = buyer.post_escrow(contract, decision=decision)
    record_ok = buyer.settle(contract, cost_usd=60.0, escrow=clean)
    print(
        f"2. Clean delivery (cost ${record_ok.delivered_cost_usd:,.0f} ≤ "
        f"${contract.terms.price_usd:,.0f}): escrow {clean.state} — "
        f"${clean.released_usd:,.2f} released to {clean.poster}, nothing forfeited."
    )

    # 3. A breach (a cost overrun) forfeits a bounded slice proportional to the shortfall.
    breached_escrow = buyer.post_escrow(contract, decision=decision)
    record_bad = buyer.settle(contract, cost_usd=140.0, escrow=breached_escrow)  # 40% over
    print(
        f"3. Breach (cost ${record_bad.delivered_cost_usd:,.0f} > "
        f"${contract.terms.price_usd:,.0f}, {breached_escrow.shortfall_fraction:.0%} over): "
        f"escrow {breached_escrow.state} — ${breached_escrow.forfeited_usd:,.2f} forfeited to "
        f"{breached_escrow.beneficiary} ({', '.join(breached_escrow.breaches)}), "
        f"${breached_escrow.released_usd:,.2f} released — proportional, never the whole stake."
    )

    # 4. Offline-verifiable: the disposition re-derives from the bytes; a tampered
    #    forfeiture is caught even after re-sealing, and every transition is on the chain.
    tampered = buyer.post_escrow(contract, fraction=0.5)
    buyer.settle_escrow(tampered, record_bad)
    tampered.forfeited_usd = 0.0  # lie: forfeit nothing despite the breach
    tampered.released_usd = tampered.amount_usd
    tampered.seal()  # recompute the hash to match the tampered split
    print(
        f"4. Offline-verifiable: a sound escrow verifies={breached_escrow.verify().valid}; a "
        f"re-sealed zeroed forfeiture is still caught (terms_sound="
        f"{tampered.verify().terms_sound}); "
        f"{len(buyer.audit.query(action='escrow'))} escrow transitions on the audit chain."
    )

    # 5. The whole settlement ledger — records and escrows alike — verifies offline.
    print(
        f"5. Folded into the settlement path: the book stays tamper-evident "
        f"(intact={buyer.settlement_book.verify(buyer.contract_signer).intact}); the "
        f"collateral closed the same loop the settlement did, no hosted escrow service."
    )


if __name__ == "__main__":
    main()
