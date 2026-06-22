"""Cross-org collateral rehypothecation guards & re-use bounds.

A ``CollateralPool`` lets a counterparty back many contracts with one posted stake, freeing
capital as clean deliveries release it — but a pool only ever *re-allocates* capital **within
itself**. When the counterparty pledges the *same* stake across more than one pool (or re-
pledges collateral a beneficiary already has a claim on), nothing bounds the **re-use**: the
same capital is double-counted, over-stating what actually backs each deal — the collateral
analogue of a settlement record double-counted before netting deduplicated it. This example
adds the next rung: a ``CollateralLedger`` — the **rehypothecation guard** that folds a
counterparty's pools into one view and bounds what its posted stake is committed to.

Four steps, all offline and deterministic:

  1. A vendor backs two separate pools with what it claims is independent capital — but a
     ledger that folds them and reconciles the pledge against the capital it actually holds
     surfaces the **same stake pledged twice** as a bounded, pinpointed re-use breach.
  2. When the held capital is scarce, each beneficiary's claim is bounded to its
     deterministic **pari-passu share**, so a forfeiture cannot pay one beneficiary out of
     capital another has first claim on.
  3. The guard reads only the signed, content-bound pools: a tampered pool is **refused** at
     fold time, and the re-use bound re-derives from the bytes even after re-sealing.
  4. Every guard lands on the hash-chained audit log and verifies offline from the bytes
     alone — `app.guard_collateral` signs it as the org and records it on the chain.

Everything here is opt-in and additive; this is a library capability inside your process,
never a hosted custodian, a clearing house, a rehypothecation registry, or a payment rail.
"""

from __future__ import annotations

from vincio import ContextApp
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider


def a_contract(buyer: str, seller: str, scope: str, price: float) -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope=scope, price_usd=price)
    ).seal()


def main() -> None:
    acme = ContextApp(name="acme", provider=MockProvider(default_text="ok"))
    acme.use_settlement_book()

    # A vendor backs two pools. Contract A is *re-pledged* — it appears in both pools, so the
    # same collateral is committed twice even though each pool verifies on its own.
    c_a = a_contract("acme", "vendor", "transcribe batch A", price=100.0)
    c_b = a_contract("acme", "vendor", "transcribe batch B", price=200.0)
    pool_one = acme.post_collateral_pool([c_a, c_b], fraction=0.1)  # pledges 30
    pool_two = acme.post_collateral_pool([c_a], fraction=0.1)  # re-pledges A (+10)

    # 1. Fold the pools into one ledger. It reconciles the gross pledge against the capital
    #    the poster actually holds and pinpoints the double-pledged contract.
    ledger = acme.guard_collateral([pool_one, pool_two])
    breach = ledger.breaches[0]
    print(
        f"1. Folded 2 pools: ${ledger.pledged_usd:,.2f} pledged against "
        f"${ledger.held_usd:,.2f} held — re-use ${ledger.reuse_usd:,.2f}. "
        f"Contract {breach.contract_id[-4:]} pledged across {len(breach.pools)} pools "
        f"(${breach.excess_usd:,.2f} double-pledged); over-committed: {ledger.over_committed}."
    )

    # 2. Beneficiary-claim priority: a stake backing two beneficiaries, with only half the
    #    capital actually held, is apportioned pari passu — no one paid out of another's share.
    to_acme = a_contract("acme", "vendor", "deal for acme", price=100.0)  # claim 10
    to_globex = a_contract("globex", "vendor", "deal for globex", price=300.0)  # claim 30
    multi = acme.post_collateral_pool([to_acme, to_globex], fraction=0.1)  # pledges 40
    scarce = acme.guard_collateral([multi], held=20.0)  # vendor really holds only 20
    shares = ", ".join(
        f"{c.beneficiary}→${c.secured_usd:,.2f}/${c.claim_usd:,.2f}" for c in scarce.claims
    )
    print(
        f"2. ${scarce.held_usd:,.2f} held against ${scarce.pledged_usd:,.2f} pledged for two "
        f"beneficiaries — bounded to deterministic shares: {shares} "
        f"(exposed: ${sum(c.unsecured_usd for c in scarce.claims):,.2f})."
    )

    # 3. The guard reads only signed, content-bound pools: a tampered pool is refused, and a
    #    re-sealed lie about the re-use bound is still caught from the bytes.
    tampered = acme.post_collateral_pool([a_contract("acme", "vendor", "x", 100.0)], fraction=0.5)
    tampered.contracts[0].allocated_usd = 9_999.0  # lie without re-sealing
    try:
        acme.guard_collateral([tampered], record_audit=False)
        refused = False
    except SettlementError:
        refused = True
    resealed = acme.guard_collateral([pool_one, pool_two], sign=False, record_audit=False)
    resealed.reuse_usd = 0.0  # hide the over-commitment ...
    resealed.seal()  # ... and recompute the hash to match
    print(
        f"3. Tampered pool refused at fold: {refused}; a re-sealed inflated ledger still "
        f"fails (hash_ok={resealed.verify().hash_ok}, terms_sound={resealed.verify().terms_sound})."
    )

    # 4. Auditable & offline: the live guard verifies and is on the hash-chained audit log.
    print(
        f"4. Offline-verifiable: the live ledger verifies="
        f"{ledger.verify(acme.contract_signer).valid}; "
        f"{len(acme.audit.query(action='rehypothecation'))} guards on the audit chain, "
        f"book intact={acme.settlement_book.verify(acme.contract_signer).intact}."
    )


if __name__ == "__main__":
    main()
