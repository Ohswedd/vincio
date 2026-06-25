# Guide: agent negotiation & contracting

Vincio already governs a fabric of agents over A2A and the MCP registry, scores
per-member reliability with a Beta-Bernoulli reputation ledger, and discounts an
unreliable member's pull on a federated round. This guide covers the next rung:
**bounded negotiation and contracting** between agents in a multi-org crew. A
buyer agent and a seller agent converge on a price / SLA / scope contract under a
hard budget; the contract is a typed, signed, audited artifact both sides can
verify offline; and the counterparty's reputation weights the deal, all in the
same governed, audited, budgeted runtime, never a hosted marketplace.

This is additive (`vincio.negotiation`); it changes nothing about how a single
agent runs, and runs fully offline against deterministic local parties.

## A bounded bargain

A negotiation is the analogue of a bounded crew round: **termination is
guaranteed**. Each party holds a private `NegotiationPosition`, for each issue an
*ideal* value and a *reservation* (walk-away) value, and concedes from its ideal
toward its reservation as the deadline approaches (a time-dependent tactic). A
party accepts the opponent's offer the moment it is at least as good as the offer
it would make next, so the bargain ends in a deal when the parties' acceptable
regions overlap and in a clean "no agreement" when they do not.

```python
from vincio import ContextApp
from vincio.negotiation import buyer_position, seller_position, NegotiationBudget

app = ContextApp(name="marketplace")

buyer = buyer_position(
    max_price_usd=0.10,      # reservation: the most the buyer will pay
    ideal_price_usd=0.0,     # ideal: free
    max_sla_seconds=5.0,     # reservation: the slowest turnaround it accepts
    ideal_sla_seconds=0.5,
    min_quality=0.7,         # reservation: the lowest quality it accepts
    ideal_quality=1.0,
)
seller = seller_position(
    min_price_usd=0.04,      # reservation: the least the seller will take
    ideal_price_usd=0.14,
    min_sla_seconds=1.0,
    ideal_sla_seconds=6.0,
    max_quality=0.95,        # reservation: the highest quality it will commit to
    ideal_quality=0.7,
)

result = app.negotiate(
    "transcribe 1,000 support calls",
    buyer=buyer,
    seller=seller,
    budget=NegotiationBudget(max_rounds=8),
    buyer_id="acme",
    seller_id="vendor",
)

print(result.status)          # "agreement" | "no_agreement" | "walk_away"
print(result.rounds)          # bounded by max_rounds
if result.agreed:
    print(result.contract.terms.price_usd, result.contract.terms.sla_seconds)
```

The `NegotiationBudget` is the guarantee: `max_rounds` bounds the offer exchanges,
and an optional `deadline_s` returns a **partial result** the moment it is hit. A
no-deal carries the full `offers` trace and each side's last offer, so a deadline
outcome is inspectable, never a bare failure.

Tune how hard a party bargains with `concession` (the exponent of the concession
curve: `< 1` tough/"boulware", `> 1` generous/"conceder") and `min_utility` (the
hard floor below which the party will neither offer nor accept).

## A typed, signed, verifiable contract

On agreement a `Contract` is minted and **signed by both parties**. It verifies
**offline** from the bytes alone, the content hash recomputes from the stored
terms and every signature checks, so a tampered term or a forged signature is
caught without the live parties.

```python
contract = result.contract
verification = contract.verify(app.contract_signer)
assert verification.valid          # hash recomputes + both signatures verify
assert contract.signed_by == ["acme", "vendor"]

contract.terms.price_usd = 0.01    # tamper
assert not contract.verify(app.contract_signer).valid
```

By default contracts are signed with the audit chain's signer when one is
configured, otherwise a per-app key (`app.contract_signer`). For third-party
verifiability without sharing a secret, pass an `Ed25519Signer`:

```python
from vincio.security.audit import Ed25519Signer

signer = Ed25519Signer(private_key=...)   # the seller holds the public half
result = app.negotiate("...", buyer=buyer, seller=seller, signer=signer)
```

## Enforced like a budget

A contract is a hard cap, not a hope. `to_budget()` lowers the agreed price and SLA
into a `Budget` the runtime already enforces, and `check()` compares delivered work
against the terms:

```python
config = RunConfig(budget=contract.to_budget())   # price → max_cost_usd, SLA → max_latency_ms
delivered = app.run("...", config=config)

fulfillment = app.enforce_contract(
    contract, cost_usd=delivered.cost_usd, latency_ms=delivered.latency_ms, quality=0.92
)
print(fulfillment.fulfilled, fulfillment.breaches)
```

`enforce_contract` records the verdict on the hash-chained audit log and, when a
reputation ledger is attached, credits the seller on fulfilment or **debits it on
a breach**, so a breached SLA discounts the seller's future offers. That closes the
loop: negotiation outcomes feed reputation, and reputation weights negotiation.

## Reputation weights the deal

When a reputation ledger is attached (`app.use_reputation_ledger()`), it weights
each party's view of the *counterparty's* offers. A repeatedly-regressing seller's
offers are **discounted, never zeroed, never singled out**: the weight stays in
`[floor, 1]`, so the discounted seller can still close a deal by conceding more (a
risk premium), and a reformed seller recovers. With several competing sellers,
`select_offer` picks the deal that maximizes the buyer's reputation-weighted
utility, so reliability, not just price, decides the winner.

```python
from vincio.negotiation import select_offer

ledger = app.use_reputation_ledger()
# ... reputation accrues from past gate/contract outcomes ...

deals = [
    app.negotiate("job", buyer=buyer, seller=seller, seller_id=sid)
    for sid in ("vendor-a", "vendor-b")
]
best = select_offer(deals, buyer, reputation=ledger)   # reputation-weighted winner
```

## Over the A2A fabric

A counterparty can live in **another organization, reached over A2A**. Expose a
local party as an A2A agent and drive it remotely with an `A2ANegotiator`, the
local engine bargains against it exactly as it would a local party, and every offer
exchange is a bounded, audited A2A task.

```python
from vincio.negotiation import LocalParty, A2ANegotiator
from vincio.a2a import connect_a2a_in_process

# Seller org: expose its negotiating party over A2A.
seller_party = LocalParty("vendor", seller)
server = app.serve_negotiation(seller_party, name="vendor")

# Buyer org: reach it as a remote party.
client = connect_a2a_in_process(server)          # or connect_a2a(url) over HTTP
remote_seller = A2ANegotiator(client, member_id="vendor", role="seller")

result = app.negotiate("job", buyer=buyer, seller=remote_seller, buyer_id="acme")
```

The remote party's identity is pinned to the directory-resolved member id, not a
self-asserted one on the wire, so a reputation lookup cannot be spoofed.

## What it is not

This is a library capability inside your process, not a hosted marketplace. There
is no clearinghouse, no escrow service, no central order book, a negotiation is two
agents exchanging typed offers under a budget, and a contract is a signed file you
hold and verify yourself. Everything that looks operational is something you run.
