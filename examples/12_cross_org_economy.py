"""The cross-organization agent economy — one narrative arc, end to end.

Two organizations — buyer ``acme`` and seller ``vendor`` — transact through agents
over Vincio's A2A fabric across nine stages: negotiate a signed contract, choreograph
a compensating cross-org saga, meter & settle the work, net a fleet's books, arbitrate
a dispute, make standing portable, prove solvency, resolve an insolvency by seniority,
and thread it all through the ``CrossOrgEngagement`` facade.

The invariant that ties the arc together: **every artifact is typed, signed,
content-bound, and verifies from the bytes alone** — a tamper anywhere is caught.
Everything is opt-in and additive: a library capability inside your process, never a
hosted marketplace, clearing house, or payment rail. Runs fully offline on the mock.
"""

from __future__ import annotations

import asyncio

from vincio import (
    ContextApp,
    EngagementNarrative,
    arbitrate,
    attest_custody,
    attest_liabilities,
    build_seniority_schedule,
    resolve_insolvency,
    settle_contract,
)
from vincio.choreography import Saga, StepOutcome
from vincio.negotiation import Contract, ContractTerms, buyer_position, seller_position
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def make_app(name: str = "acme") -> ContextApp:
    """A fresh governed org. The mock provider keeps this fully offline."""
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), model="mock-1")


def a_contract(buyer: str, seller: str, *, scope: str = "transcribe 1k calls", price: float = 0.10) -> Contract:
    """A sealed, signed contract — the typed artifact both orgs verify offline."""
    return Contract(buyer=buyer, seller=seller, terms=ContractTerms(scope=scope, price_usd=price)).seal()


async def section_negotiate(app: ContextApp) -> Contract:
    banner("1. Negotiate — bounded, signed, tamper-evident")
    # anegotiate runs a bounded bilateral bargain (guaranteed to terminate) between a
    # buyer and seller position and returns a signed Contract. max_/min_ are walk-away
    # reservation points; ideal_ are aspirations — the agents converge in between.
    buyer = buyer_position(max_price_usd=0.12, ideal_price_usd=0.04, max_sla_seconds=5.0, min_quality=0.7)
    seller = seller_position(min_price_usd=0.04, ideal_price_usd=0.10, min_sla_seconds=1.0, max_quality=0.95)
    result = await app.anegotiate("transcribe 1,000 support calls", buyer=buyer, seller=seller,
                                  buyer_id="acme", seller_id="vendor")
    contract = result.contract
    print(f"agreed in {result.rounds} rounds: ${contract.terms.price_usd:.4f} "
          f"SLA {contract.terms.sla_seconds:.2f}s quality>={contract.terms.quality_floor:.2f}")

    # The signature verifies from the bytes; mutating a term without resealing breaks it.
    tampered = contract.model_copy(deep=True)
    tampered.terms.price_usd = 0.01
    print(f"signed_by={contract.signed_by} verifies={contract.verify(app.contract_signer).valid}; "
          f"tampered verifies={tampered.verify(app.contract_signer).valid}")
    return contract


async def section_choreograph(app: ContextApp) -> None:
    banner("2. Choreograph — durable saga with compensation")
    # A saga is a multi-org workflow where each step names the action that *undoes* it.
    # On failure the completed steps roll back in reverse order; the journal is
    # hash-chained, so the whole run verifies offline. Use it for cross-org work that
    # must be all-or-nothing without a shared database or a distributed transaction.
    def build_saga() -> Saga:
        return (Saga(name="fulfil-order")
                .step("reserve", participant="warehouse", action="reserve", compensation="release")
                .step("charge", participant="payments", action="charge", compensation="refund")
                .step("ship", participant="warehouse", action="ship"))

    log: list[str] = []
    healthy = {
        "warehouse": {"reserve": lambda p: log.append("reserve") or {"ticket": "WH-1"},
                      "release": lambda p: log.append("release") or {},
                      "ship": lambda p: log.append("ship") or {"tracking": "TRK-9"}},
        "payments": {"charge": lambda p: log.append("charge") or {"receipt": "PAY-7"},
                     "refund": lambda p: log.append("refund") or {}},
    }
    ok = await app.achoreograph(build_saga(), participants=healthy, input={"sku": "A1"})
    print(f"forward: status={ok.status} ran={log}")

    # The last step fails → the completed steps compensate in reverse (ship never ran).
    comp: list[str] = []
    failing = {
        "warehouse": {"reserve": lambda p: {"ticket": "WH-2"},
                      "release": lambda p: comp.append("release") or {},
                      "ship": lambda p: StepOutcome(ok=False, error="carrier unavailable")},
        "payments": {"charge": lambda p: {"receipt": "PAY-8"},
                     "refund": lambda p: comp.append("refund") or {}},
    }
    rolled = await app.achoreograph(build_saga(), participants=failing)
    print(f"failure: status={rolled.status} compensated_in_reverse={comp}; "
          f"journal verifies={ok.journal.verify().intact}")


def section_settle(app: ContextApp, contract: Contract) -> None:
    banner("3. Settle — metered, reconciled, signed")
    # A meter accrues real usage against the agreed price; settle() reconciles delivery
    # vs terms into a signed SettlementRecord (a *record*, never a payment rail). A
    # breach settles to a breached record with the offending dimensions pinpointed —
    # not an exception — so books always close to a verifiable artifact.
    priced = a_contract("acme", "vendor", price=0.10)
    meter = app.meter(priced)
    meter.accrue(units=500, cost_usd=0.04, latency_ms=1200, quality=0.95, step="batch-1")
    meter.accrue(units=500, cost_usd=0.03, latency_ms=900, quality=0.92, step="batch-2")
    reading = meter.reading()
    record = app.settle(priced, reading=reading)
    print(f"metered {reading.units:g} units → status={record.status} balance=${record.balance_usd:+.2f} "
          f"verifies={record.verify(app.contract_signer, require=['acme']).valid}")

    breached = app.settle(a_contract("acme", "vendor", price=0.05, scope="tight job"), cost_usd=0.08, quality=0.6)
    print(f"overrun → status={breached.status} overrun=${breached.overrun_usd:.2f} breaches={breached.breaches}")


def section_net(app: ContextApp) -> None:
    banner("4. Net — multilateral clearing")
    # Netting folds a web of bilateral settlements into the minimal set of net
    # obligations (≤ N−1 transfers). Positions sum to zero and the cleared transfers
    # reproduce them — so an org that is both buyer and seller closes its books once,
    # moving far less cash than the gross would suggest.
    fleet = [  # a cycle: acme owes vendor owes data owes acme
        settle_contract(a_contract("acme", "vendor", price=0.10), cost_usd=0.08),
        settle_contract(a_contract("vendor", "data", price=0.06), cost_usd=0.05),
        settle_contract(a_contract("data", "acme", price=0.04), cost_usd=0.03),
    ]
    netting = app.clear_settlements(records=fleet)
    verdict = netting.verify(app.contract_signer)
    print(f"{netting.gross_edges} gross obligations → {netting.cleared_transfers} transfers "
          f"(${netting.total_gross_usd:.2f} gross, only ${netting.total_cleared_usd:.2f} moves)")
    print(f"verifies={verdict.valid} (positions_balanced={verdict.positions_balanced}, conserves={verdict.conserves})")


def section_arbitrate() -> None:
    banner("5. Arbitrate — which figure stands")
    # Arbitration decides deterministically over signed records: a co-signed
    # reconciliation is upheld, a unilateral contradiction is rejected and pinpointed,
    # and a genuine standoff is left honestly unresolved. Parties share the fabric
    # secret and are distinguished only by key_id.
    buyer, seller = HMACSigner("fabric-secret", key_id="acme"), HMACSigner("fabric-secret", key_id="vendor")
    app = make_app("arbiter")
    app.use_reputation_ledger()
    c = a_contract("acme", "vendor", price=0.10)

    def claim(*, cost: float, signer: HMACSigner, party: str):
        return settle_contract(c, cost_usd=cost).sign(signer, party=party)

    acme_says = claim(cost=0.08, signer=buyer, party="acme")
    standoff = arbitrate([acme_says, claim(cost=0.05, signer=seller, party="vendor")], contract_id=c.id)
    print(f"uncorroborated → status={standoff.status} ({standoff.reason})")

    vendor_agrees = claim(cost=0.08, signer=seller, party="vendor")  # seller co-signs the buyer's figure
    res = app.arbitrate([acme_says, vendor_agrees])
    print(f"co-signed → status={res.status} upheld=${res.upheld_balance_usd:+.2f} by {res.corroborated_by}")

    # A bad-faith contradicting claim is rejected AND dings the signer's reputation.
    before = app.reputation_ledger.snapshot("vendor").reputation
    disputed = app.arbitrate([acme_says, vendor_agrees, claim(cost=0.05, signer=seller, party="vendor")])
    after = app.reputation_ledger.snapshot("vendor").reputation
    print(f"bad-faith → status={disputed.status} dissenters={disputed.dissenters} reputation {before:.3f}→{after:.3f}")


def section_attest_reputation() -> None:
    banner("6. Reputation attestation — standing that travels")
    # Each org derives a signed ReputationAttestation from its own signed records. A
    # buyer with no local history imports a bundle and the evidence pools into one
    # bounded prior — so reputation is portable without a central rating agency, and a
    # self-attestation carries no weight.
    acme = make_app("acme")
    acme.use_settlement_book()
    for _ in range(4):
        acme.settle(a_contract("acme", "vendor", price=0.10), cost_usd=0.06)  # fulfilled
    acme.settle(a_contract("acme", "vendor", price=0.04), cost_usd=0.09)  # one breach
    acme_att = acme.attest_reputation("vendor")

    globex = make_app("globex")
    globex.use_settlement_book()
    for _ in range(3):
        globex.settle(a_contract("globex", "vendor", price=0.10), cost_usd=0.05)
    globex_att = globex.attest_reputation("vendor")
    print(f"acme attests {acme_att.settled}ok/{acme_att.breached}fail (rep {acme_att.reputation:.3f}), "
          f"verifies={acme_att.verify(acme.contract_signer).valid}")

    buyer = make_app("buyer")
    buyer.use_reputation_ledger()
    prior = buyer.import_reputation([acme_att, globex_att])
    standing = prior.standing("vendor")
    print(f"new buyer pools {len(prior.counted)} attestations → rep {standing.reputation:.3f} from {standing.issuers}; "
          f"a stranger gets prior {prior.weight('stranger'):.3f}")


def section_solvency() -> None:
    banner("7. Proof-of-solvency — reserves vs. liabilities")
    # prove_solvency folds proof-of-reserves against proof-of-liabilities into a
    # margin, reading only signed content-bound artifacts. A counterparty solvent
    # against one creditor may be under water once *every* obligation is counted; an
    # insolvency surfaces as a bounded, pinpointed breach rather than a bare 'false'.
    auditor = make_app("auditor")
    auditor.use_settlement_book()
    reserves = auditor.attest_custody("vendor", {"omnibus": 80.0})

    proof = auditor.prove_solvency(reserves, auditor.attest_liabilities("vendor", {"globex": 35.0, "initech": 15.0}))
    print(f"solvent: ${proof.reserves_usd:,.0f} held − ${proof.liabilities_usd:,.0f} owed = "
          f"${proof.margin_usd:,.0f} free; verifies={proof.verify(auditor.contract_signer).valid}")

    insolvent = auditor.prove_solvency(reserves, auditor.attest_liabilities("vendor", {"globex": 70.0, "initech": 50.0}))
    print(f"insolvent: same ${insolvent.reserves_usd:,.0f} vs ${insolvent.liabilities_usd:,.0f} owed → "
          f"short ${insolvent.breach.shortfall_usd:,.0f} (attestor {insolvent.breach.attestor!r})")


def section_waterfall() -> None:
    banner("8. Insolvency waterfall — who gets what")
    # When reserves can't cover every claim, a signed SenioritySchedule ranks the
    # creditors and resolve_insolvency pays out by rank, then pari-passu within a
    # tranche. Each creditor's bounded recovery and the shortfall it bears are
    # pinpointed, and the resolution is hash-bound to the schedule it applied.
    custodian, auditor_signer, bank = (HMACSigner("fabric-secret", key_id=k) for k in ("custodian", "auditor", "bank"))
    reserves = attest_custody("vendor", {"omnibus": 60.0}, custodian="custodian").sign(custodian)
    owed = attest_liabilities("vendor", {"bank": 50.0, "acme": 30.0, "globex": 20.0}, attestor="auditor").sign(auditor_signer)
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]]).sign(bank, party="bank")  # bank senior

    app = make_app("auditor")
    app.use_settlement_book(owner="auditor")
    app.use_reputation_ledger()
    resolution = app.resolve_insolvency(reserves, owed, schedule, verifier=auditor_signer)
    print(f"$60 reserves vs $100 owed → distributed ${resolution.distributed_usd:,.0f}; "
          f"{resolution.shortfall_bearers} bear ${resolution.shortfall_usd:,.0f}")
    for r in sorted(resolution.recoveries, key=lambda r: (r.rank, r.creditor)):
        mark = "made whole" if r.made_whole else f"short ${r.shortfall_usd:,.0f}"
        print(f"  rank {r.rank} {r.creditor}: ${r.recovery_usd:,.0f}/${r.claim_usd:,.0f} ({r.recovery_rate:.0%}) — {mark}")

    clean = resolve_insolvency(reserves, owed, schedule, verifier=auditor_signer).sign(auditor_signer, party="auditor")
    print(f"verifies={clean.verify(auditor_signer, schedule=schedule).valid}, bound to its schedule by hash")


def section_engagement() -> None:
    banner("9. CrossOrgEngagement — the whole fabric as one system")
    # The facade is purely compositional — each method delegates to the same app.*
    # primitive above — and seals the run into one hash-linked, signed, offline-
    # verifiable narrative. This is how you run the whole arc as one governed,
    # audited, narrated call-path; a tamper anywhere (re-ordered stage, edited bytes)
    # is caught by the chain.
    app = make_app("acme")
    eng = app.cross_org_engagement(buyer="acme", seller="vendor", scope="transcribe 1k calls")
    contract = eng.negotiate(buyer=buyer_position(max_price_usd=0.12, max_sla_seconds=5.0),
                             seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.10))
    saga = (Saga(name="fulfil")
            .step("transcribe", participant="vendor", action="run", contract=contract)
            .step("deliver", participant="vendor", action="deliver", compensation="recall"))
    parts = {"vendor": {
        "run": lambda p: StepOutcome(ok=True, cost_usd=0.05, latency_ms=1200, quality=0.95, output={"text": "..."}),
        "deliver": lambda p: StepOutcome(ok=True, output={"delivered": True}),
        "recall": lambda p: {"recalled": True}}}
    eng.choreograph(saga, participants=parts)
    eng.settle_saga(contracts={contract.id: contract})
    eng.net()
    reserves = eng.attest_custody("vendor", {"omnibus": 120.0})
    eng.prove_solvency(reserves, eng.attest_liabilities("vendor", {"acme": 40.0, "globex": 30.0}))

    narrative = eng.seal()
    whole = eng.verify(app.contract_signer)
    forged = EngagementNarrative.from_wire(narrative.to_wire())
    forged.stages[2], forged.stages[3] = forged.stages[3], forged.stages[2]  # re-order two stages
    print(f"sealed {len(narrative.stages)} stages [{', '.join(narrative.stage_names)}]")
    print(f"verifies: valid={whole.valid} (chain_intact={whole.intact}, digests_ok={whole.digests_ok}); "
          f"re-ordered narrative caught={not forged.verify().valid}")


async def main() -> None:
    app = make_app("acme")
    app.use_reputation_ledger()
    app.use_settlement_book()

    contract = await section_negotiate(app)
    await section_choreograph(app)
    section_settle(app, contract)
    section_net(app)
    section_arbitrate()
    section_attest_reputation()
    section_solvency()
    section_waterfall()
    section_engagement()

    print(f"\naudit chain intact={app.audit.verify_chain()}")


if __name__ == "__main__":
    asyncio.run(main())
