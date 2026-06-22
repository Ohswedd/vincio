"""Agent negotiation & contracting — a bounded, signed, reputation-weighted deal.

Vincio governs a fabric of agents over A2A, scores per-member reliability with a
reputation ledger, and discounts an unreliable member's pull on a federated round.
This example adds the next rung: two agents in a multi-org crew **negotiate a
contract** — a buyer agent and a seller agent converge on a price/SLA/scope
agreement under a hard budget, the contract is a typed, signed, audited artifact
both sides verify offline, and the counterparty's reputation weights the deal.

Six steps, all offline and deterministic:

  1. Bargain: a bounded offer/counter negotiation converges on terms — the
     negotiation analogue of a bounded crew round, with guaranteed termination.
  2. Contract: the agreement is signed by both parties and verifies offline from
     the bytes alone; a tampered term is caught.
  3. Enforce: the agreed terms lower to a Budget the runtime enforces, and a
     breach of the delivered work is detected like any other budget overrun.
  4. Terminate: a bargain with no overlapping acceptable region ends in a clean
     no-deal, and a wall-clock deadline returns a partial result — never a loop.
  5. Reputation: a regressing seller's offers are discounted (it must concede
     more) without being singled out, and the reputation-weighted best deal wins.
  6. Fabric: the same bargain runs against a counterparty reached over A2A,
     byte-for-byte the same as locally, every outcome on the audit chain.

Everything here is opt-in and additive; this is a library capability inside your
process, never a hosted marketplace.
"""

from __future__ import annotations

import asyncio

from vincio import ContextApp
from vincio.a2a import connect_a2a_in_process
from vincio.negotiation import (
    A2ANegotiator,
    LocalParty,
    NegotiationBudget,
    buyer_position,
    select_offer,
    seller_position,
)
from vincio.providers import MockProvider


def make_buyer():
    # The buyer wants a low price, a fast turnaround, and high quality; the
    # `max_*` / `min_*` values are its reservation (walk-away) points.
    return buyer_position(
        max_price_usd=0.10,
        ideal_price_usd=0.0,
        max_sla_seconds=5.0,
        ideal_sla_seconds=0.5,
        min_quality=0.7,
        ideal_quality=1.0,
    )


def make_seller():
    # The seller wants a high price, a loose SLA, and a low committed quality floor.
    return seller_position(
        min_price_usd=0.04,
        ideal_price_usd=0.14,
        min_sla_seconds=1.0,
        ideal_sla_seconds=6.0,
        max_quality=0.95,
        ideal_quality=0.7,
    )


async def main() -> None:
    app = ContextApp(name="acme", provider=MockProvider(default_text="ok"), model="mock-1")

    # 1. Bargain ------------------------------------------------------------
    result = await app.anegotiate(
        "transcribe 1,000 support calls",
        buyer=make_buyer(),
        seller=make_seller(),
        budget=NegotiationBudget(max_rounds=8),
        buyer_id="acme",
        seller_id="vendor",
    )
    print(f"1. status={result.status} rounds={result.rounds}")
    terms = result.contract.terms
    print(
        f"   agreed: ${terms.price_usd:.4f}  SLA {terms.sla_seconds:.2f}s  "
        f"quality≥{terms.quality_floor:.3f}"
    )

    # 2. Contract: signed, offline-verifiable, tamper-evident ---------------
    contract = result.contract
    print(f"2. signed_by={contract.signed_by}  verifies={contract.verify(app.contract_signer).valid}")
    tampered = contract.model_copy(deep=True)
    tampered.terms.price_usd = 0.01
    print(f"   tampered contract verifies={tampered.verify(app.contract_signer).valid}")

    # 3. Enforce like a budget ----------------------------------------------
    budget = contract.to_budget()
    print(
        f"3. contract → budget: max_cost=${budget.max_cost_usd:.4f} "
        f"max_latency={budget.max_latency_ms}ms"
    )
    fulfillment = app.enforce_contract(
        contract, cost_usd=terms.price_usd * 2, latency_ms=999_999, quality=0.1
    )
    print(f"   delivered work fulfilled={fulfillment.fulfilled} breaches={fulfillment.breaches}")

    # 4. Guaranteed termination ---------------------------------------------
    no_overlap = await app.anegotiate(
        "job",
        buyer=buyer_position(max_price_usd=0.02, ideal_price_usd=0.0, max_sla_seconds=2.0, min_quality=0.9),
        seller=seller_position(min_price_usd=0.10, ideal_price_usd=0.2, min_sla_seconds=5.0, max_quality=0.5),
        budget=NegotiationBudget(max_rounds=6),
    )
    print(f"4. no overlapping region → status={no_overlap.status} (bounded, no contract)")

    # 5. Reputation weights the deal ----------------------------------------
    ledger = app.use_reputation_ledger()
    for _ in range(30):
        ledger.record_outcome("vendor", passed=False, round_id="history")  # a regressor
        ledger.record_outcome("trusty", passed=True, round_id="history")  # a reliable peer
    bad = await app.anegotiate("job", buyer=make_buyer(), seller=make_seller(), buyer_id="acme", seller_id="vendor")
    good = await app.anegotiate("job", buyer=make_buyer(), seller=make_seller(), buyer_id="acme", seller_id="trusty")
    print(
        f"5. discounted seller conceded to ${bad.contract.terms.price_usd:.4f} "
        f"vs reliable ${good.contract.terms.price_usd:.4f} (both still close a deal)"
    )
    winner = select_offer([bad, good], make_buyer(), reputation=ledger)
    print(f"   reputation-weighted selection → {winner.seller}")

    # 6. Over the A2A fabric ------------------------------------------------
    clean = ContextApp(name="clean", provider=MockProvider(default_text="ok"), model="mock-1")
    local = await clean.anegotiate("job", buyer=make_buyer(), seller=make_seller(), buyer_id="acme", seller_id="vendor")
    server = clean.serve_negotiation(LocalParty("vendor", make_seller()), name="vendor")
    client = connect_a2a_in_process(server)
    remote = A2ANegotiator(client, member_id="vendor", role="seller")
    over_a2a = await clean.anegotiate("job", buyer=make_buyer(), seller=remote, buyer_id="acme")
    parity = over_a2a.contract.terms.canonical() == local.contract.terms.canonical()
    print(f"6. A2A bargain reaches the same terms as local: {parity}")

    # Every outcome is on the hash-chained audit log.
    print(
        f"\naudit: {len(app.audit.query(action='negotiation'))} negotiations, "
        f"{len(app.audit.query(action='contract_signed'))} contracts, chain intact="
        f"{app.audit.verify_chain()}"
    )


if __name__ == "__main__":
    asyncio.run(main())
