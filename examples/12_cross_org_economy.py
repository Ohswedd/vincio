"""The cross-organization agent economy.

One coherent story of two organizations — buyer ``acme`` and seller ``vendor`` —
transacting through agents over Vincio's A2A fabric, end to end. They negotiate a
bounded, signed Contract; choreograph a durable compensating cross-org Saga; meter
and settle the delivered work; net a fleet's multilateral books; arbitrate a dispute
over signed records; make earned standing portable as a reputation attestation;
prove the counterparty solvent (reserves vs. liabilities); resolve an insolvency by
a seniority waterfall; and finally thread the whole pipeline through the
``CrossOrgEngagement`` lifecycle facade — one signed, offline-verifiable narrative.

Every artifact is typed, signed, content-bound, and verifies from the bytes alone.
Everything is opt-in and additive: a library capability inside your process, never a
hosted marketplace, clearing house, or payment rail. Runs fully offline on the
deterministic mock provider.
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
    prove_solvency,
    resolve_insolvency,
    settle_contract,
)
from vincio.choreography import Saga, StepOutcome
from vincio.negotiation import (
    Contract,
    ContractTerms,
    buyer_position,
    seller_position,
)
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def make_app(name: str = "acme") -> ContextApp:
    """A fresh governed org. The mock provider keeps this fully offline."""
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), model="mock-1")


def a_contract(buyer: str, seller: str, *, scope: str = "transcribe 1k calls", price: float = 0.10) -> Contract:
    """A sealed, signed contract — the typed artifact both orgs verify offline."""
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope=scope, price_usd=price)
    ).seal()


# --------------------------------------------------------------------------- #
# 1. Negotiate a bounded, signed contract                                     #
# --------------------------------------------------------------------------- #
async def section_negotiate(app: ContextApp) -> Contract:
    """A buyer and seller agent converge on price/SLA/quality under a hard budget.

    The bargain is bounded (guaranteed to terminate), the agreement is signed by
    both parties, and it verifies offline from the bytes — a tampered term is caught.
    """
    banner("1. Negotiate — bounded, signed, tamper-evident")

    # max_*/min_* are reservation (walk-away) points; ideal_* are aspirations.
    buyer = buyer_position(
        max_price_usd=0.12, ideal_price_usd=0.04, max_sla_seconds=5.0, min_quality=0.7
    )
    seller = seller_position(
        min_price_usd=0.04, ideal_price_usd=0.10, min_sla_seconds=1.0, max_quality=0.95
    )
    result = await app.anegotiate(
        "transcribe 1,000 support calls",
        buyer=buyer,
        seller=seller,
        buyer_id="acme",
        seller_id="vendor",
    )
    contract = result.contract
    terms = contract.terms
    print(f"status={result.status} rounds={result.rounds}")
    print(f"agreed: ${terms.price_usd:.4f}  SLA {terms.sla_seconds:.2f}s  quality>={terms.quality_floor:.2f}")

    # Offline-verifiable: the signature checks from the bytes; mutating a term breaks it.
    print(f"signed_by={contract.signed_by}  verifies={contract.verify(app.contract_signer).valid}")
    tampered = contract.model_copy(deep=True)
    tampered.terms.price_usd = 0.01  # mutate without resealing
    print(f"tampered contract verifies={tampered.verify(app.contract_signer).valid}")
    return contract


# --------------------------------------------------------------------------- #
# 2. Choreograph a durable, compensating cross-org saga                       #
# --------------------------------------------------------------------------- #
async def section_choreograph(app: ContextApp) -> None:
    """A long-running workflow spanning two orgs, with deterministic compensation.

    Each step names the org that performs it and the action that undoes it on
    rollback. A failure unwinds the already-completed steps in reverse order, and
    the hash-chained journal verifies offline.
    """
    banner("2. Choreograph — durable saga with compensation")

    def build_saga() -> Saga:
        return (
            Saga(name="fulfil-order")
            .step("reserve", participant="warehouse", action="reserve", compensation="release")
            .step("charge", participant="payments", action="charge", compensation="refund")
            .step("ship", participant="warehouse", action="ship")
        )

    # Forward path: every step succeeds and completes in order.
    log: list[str] = []
    healthy = {
        "warehouse": {
            "reserve": lambda p: log.append("reserve") or {"ticket": "WH-1"},
            "release": lambda p: log.append("release") or {},
            "ship": lambda p: log.append("ship") or {"tracking": "TRK-9"},
        },
        "payments": {
            "charge": lambda p: log.append("charge") or {"receipt": "PAY-7"},
            "refund": lambda p: log.append("refund") or {},
        },
    }
    ok = await app.achoreograph(build_saga(), participants=healthy, input={"sku": "A1"})
    print(f"forward: status={ok.status} completed={ok.completed_steps} (ran {log})")

    # A failure on the last step rolls back the completed ones in reverse order.
    comp: list[str] = []
    failing = {
        "warehouse": {
            "reserve": lambda p: {"ticket": "WH-2"},
            "release": lambda p: comp.append("release") or {},
            "ship": lambda p: StepOutcome(ok=False, error="carrier unavailable"),
        },
        "payments": {
            "charge": lambda p: {"receipt": "PAY-8"},
            "refund": lambda p: comp.append("refund") or {},
        },
    }
    rolled = await app.achoreograph(build_saga(), participants=failing)
    print(f"failure: status={rolled.status} compensated={rolled.compensated_steps} (reverse: {comp})")
    print(f"journal verifies offline={ok.journal.verify().intact}")


# --------------------------------------------------------------------------- #
# 3. Meter & settle the contracted work                                       #
# --------------------------------------------------------------------------- #
def section_settle(app: ContextApp, contract: Contract) -> None:
    """Close the books: usage accrues against the price, then reconciles to a record.

    A settlement record reconciles delivered work against the agreed terms — signed,
    offline-verifiable, and never a payment rail. A breach reconciles to a *settled*
    breached record (not an error), with the breaching dimensions pinpointed.
    """
    banner("3. Settle — metered, reconciled, signed")

    # A priced contract for the metered delivery.
    priced = a_contract("acme", "vendor", price=0.10)
    meter = app.meter(priced)
    meter.accrue(units=500, cost_usd=0.04, latency_ms=1200, quality=0.95, step="batch-1")
    meter.accrue(units=500, cost_usd=0.03, latency_ms=900, quality=0.92, step="batch-2")
    reading = meter.reading()
    print(f"metered {reading.units:g} units / {reading.events} events: cost=${reading.cost_usd:.2f} quality={reading.quality:.2f}")

    record = app.settle(priced, reading=reading)
    print(f"settled status={record.status} owed=${record.amount_owed_usd:.2f} balance=${record.balance_usd:+.2f}")
    print(f"verifies offline={record.verify(app.contract_signer, require=['acme']).valid}")

    # An overrun against a tight floor settles to a breach, dimensions pinpointed.
    breach_c = a_contract("acme", "vendor", price=0.05, scope="tight job")
    breached = app.settle(breach_c, cost_usd=0.08, quality=0.6)
    print(f"breach status={breached.status} overrun=${breached.overrun_usd:.2f} breaches={breached.breaches}")


# --------------------------------------------------------------------------- #
# 4. Net a fleet's multilateral books                                         #
# --------------------------------------------------------------------------- #
def section_net(app: ContextApp) -> None:
    """Fold a web of bilateral settlements into one minimal set of net obligations.

    An org that is both buyer and seller across many contracts closes its books
    once. The cleared set is content-bound and signed; positions sum to zero and
    the minimal transfers reproduce them.
    """
    banner("4. Net — multilateral clearing")

    # A cycle: acme owes vendor, vendor owes data, data owes acme.
    fleet = [
        settle_contract(a_contract("acme", "vendor", price=0.10), cost_usd=0.08),
        settle_contract(a_contract("vendor", "data", price=0.06), cost_usd=0.05),
        settle_contract(a_contract("data", "acme", price=0.04), cost_usd=0.03),
    ]
    netting = app.clear_settlements(records=fleet)
    print(
        f"{netting.gross_edges} gross obligations -> {netting.cleared_transfers} cleared transfers "
        f"(${netting.total_gross_usd:.2f} gross, only ${netting.total_cleared_usd:.2f} moves)"
    )
    for o in netting.obligations:
        print(f"  transfer {o.debtor} -> {o.creditor}: ${o.amount_usd:.2f}")
    verdict = netting.verify(app.contract_signer)
    print(f"verifies offline={verdict.valid} (positions balance={verdict.positions_balanced}, conserves={verdict.conserves})")


# --------------------------------------------------------------------------- #
# 5. Arbitrate a dispute over signed records                                  #
# --------------------------------------------------------------------------- #
def section_arbitrate() -> None:
    """Two parties submit signed records; a deterministic arbitration decides.

    A reconciliation both parties co-signed is upheld; a contradicting unilateral
    claim is rejected and pinpointed; a genuine standoff is left honestly unresolved.
    Parties sign with the shared fabric secret, distinguished only by key_id.
    """
    banner("5. Arbitrate — which figure stands")

    buyer = HMACSigner("fabric-secret", key_id="acme")
    seller = HMACSigner("fabric-secret", key_id="vendor")
    app = make_app("arbiter")
    app.use_reputation_ledger()

    c = a_contract("acme", "vendor", price=0.10)

    def claim(*, cost: float, signer: HMACSigner, party: str):
        return settle_contract(c, cost_usd=cost).sign(signer, party=party)

    # The two sides disagree on the delivered cost: neither figure is corroborated.
    acme_says = claim(cost=0.08, signer=buyer, party="acme")
    vendor_says = claim(cost=0.05, signer=seller, party="vendor")
    standoff = arbitrate([acme_says, vendor_says], contract_id=c.id)
    print(f"standoff: status={standoff.status} ({standoff.reason})")

    # The seller co-signs the buyer's figure: one reconciliation hash, upheld.
    vendor_agrees = claim(cost=0.08, signer=seller, party="vendor")
    res = app.arbitrate([acme_says, vendor_agrees])
    print(f"co-signed: status={res.status} upheld balance=${res.upheld_balance_usd:+.2f} corroborated_by={res.corroborated_by}")

    # A bad-faith contradicting claim is rejected, pinpointed, and dinged on reputation.
    before = app.reputation_ledger.snapshot("vendor").reputation
    liar = claim(cost=0.05, signer=seller, party="vendor")
    disputed = app.arbitrate([acme_says, vendor_agrees, liar])
    after = app.reputation_ledger.snapshot("vendor").reputation
    print(f"bad-faith: status={disputed.status} dissenters={disputed.dissenters} reputation {before:.3f} -> {after:.3f}")


# --------------------------------------------------------------------------- #
# 6. Make earned standing portable                                            #
# --------------------------------------------------------------------------- #
def section_attest_reputation() -> None:
    """Two orgs attest a vendor's standing; a new buyer pools them into a prior.

    Each issuer derives a signed ``ReputationAttestation`` from its own signed
    records. A buyer with no local history imports the bundle, the evidence pools
    into one bounded prior, and a self-attestation is refused.
    """
    banner("6. Reputation attestation — standing that travels")

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

    print(f"acme attests {acme_att.settled}ok/{acme_att.breached}fail (reputation {acme_att.reputation:.3f})")
    print(f"globex attests {globex_att.settled}ok/{globex_att.breached}fail (reputation {globex_att.reputation:.3f})")
    print(f"acme attestation verifies offline={acme_att.verify(acme.contract_signer).valid}")

    # A brand-new buyer with no history pools the bundle into one bounded prior.
    buyer = make_app("buyer")
    buyer.use_reputation_ledger()
    prior = buyer.import_reputation([acme_att, globex_att])
    standing = prior.standing("vendor")
    print(f"imported {len(prior.counted)} attestation(s): pooled reputation {standing.reputation:.3f} from {standing.issuers}")
    print(f"an unknown party gets the benefit-of-the-doubt prior {prior.weight('stranger'):.3f}")


# --------------------------------------------------------------------------- #
# 7. Prove solvency (reserves vs liabilities)                                 #
# --------------------------------------------------------------------------- #
def section_solvency() -> ContextApp:
    """Fold proof-of-reserves against proof-of-liabilities into a solvency margin.

    A counterparty solvent against one buyer's pledges may be under water once
    *every* obligation is counted. ``prove_solvency`` reads only signed,
    content-bound artifacts; an insolvency surfaces as a bounded, pinpointed breach.
    """
    banner("7. Proof-of-solvency — reserves vs. liabilities")

    auditor = make_app("auditor")
    auditor.use_settlement_book()

    # The vendor proves $80 held but owes $50 across all creditors: $30 free.
    reserves = auditor.attest_custody("vendor", {"omnibus": 80.0})
    owed = auditor.attest_liabilities("vendor", {"globex": 35.0, "initech": 15.0})
    proof = auditor.prove_solvency(reserves, owed)
    print(
        f"solvent: reserves ${proof.reserves_usd:,.0f} - liabilities ${proof.liabilities_usd:,.0f} "
        f"= ${proof.margin_usd:,.0f} free (solvent={proof.solvent})"
    )

    # The same $80, but $120 owed: insolvent by $40, pinpointed to the attestor.
    deep_owed = auditor.attest_liabilities("vendor", {"globex": 70.0, "initech": 50.0})
    insolvent = auditor.prove_solvency(reserves, deep_owed)
    breach = insolvent.breach
    print(f"insolvent: ${insolvent.reserves_usd:,.0f} held vs ${insolvent.liabilities_usd:,.0f} owed -> short ${breach.shortfall_usd:,.0f} (attestor {breach.attestor!r})")
    print(f"proof verifies offline={proof.verify(auditor.contract_signer).valid}")
    return auditor


# --------------------------------------------------------------------------- #
# 8. Resolve an insolvency by seniority waterfall                             #
# --------------------------------------------------------------------------- #
def section_waterfall() -> None:
    """Distribute scarce reserves by seniority, then pari-passu within a tranche.

    When reserves cannot cover every obligation, a signed ``SenioritySchedule``
    ranks the creditors and ``resolve_insolvency`` says who gets what — each
    creditor's bounded recovery and the shortfall it bears are pinpointed.
    """
    banner("8. Insolvency waterfall — who gets what")

    # One shared fabric secret; key_id distinguishes each party's identity.
    custodian = HMACSigner("fabric-secret", key_id="custodian")
    auditor_signer = HMACSigner("fabric-secret", key_id="auditor")
    bank = HMACSigner("fabric-secret", key_id="bank")

    # $60 held against $100 owed: a $40 shortfall to resolve.
    reserves = attest_custody("vendor", {"omnibus": 60.0}, custodian="custodian").sign(custodian)
    owed = attest_liabilities(
        "vendor", {"bank": 50.0, "acme": 30.0, "globex": 20.0}, attestor="auditor"
    ).sign(auditor_signer)
    proof = prove_solvency(reserves, owed, verifier=auditor_signer)
    print(f"shortfall: reserves ${proof.reserves_usd:,.0f} vs liabilities ${proof.liabilities_usd:,.0f} ({proof.status})")

    # The bank is senior (rank 0); acme and globex are junior (rank 1). The bank
    # co-signs the inter-creditor order, making it non-repudiable.
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]]).sign(bank, party="bank")

    app = make_app("auditor")
    app.use_settlement_book(owner="auditor")
    app.use_reputation_ledger()
    resolution = app.resolve_insolvency(reserves, owed, schedule, verify_with=auditor_signer)
    print(f"waterfall ({resolution.status}): distributed ${resolution.distributed_usd:,.0f} of ${resolution.liabilities_usd:,.0f}; {resolution.shortfall_bearers} bear ${resolution.shortfall_usd:,.0f}")
    for r in sorted(resolution.recoveries, key=lambda r: (r.rank, r.creditor)):
        mark = "made whole" if r.made_whole else f"short ${r.shortfall_usd:,.0f}"
        print(f"  rank {r.rank} {r.creditor}: ${r.recovery_usd:,.0f} of ${r.claim_usd:,.0f} ({r.recovery_rate:.0%}) — {mark}")

    # Verifies from the bytes; an over-stated recovery is refused.
    clean = resolve_insolvency(reserves, owed, schedule, verifier=auditor_signer).sign(auditor_signer, party="auditor")
    print(f"verifies offline={clean.verify(auditor_signer, schedule=schedule).valid}; bound to its schedule by hash")


# --------------------------------------------------------------------------- #
# 9. The CrossOrgEngagement lifecycle facade                                  #
# --------------------------------------------------------------------------- #
def section_engagement() -> None:
    """Thread the whole pipeline behind one governed, audited, narrated call-path.

    ``CrossOrgEngagement`` is purely compositional — each method delegates to the
    same ``app.*`` primitive — and seals the run into one hash-linked, signed,
    offline-verifiable narrative. A tamper anywhere (re-ordered stage, edited
    artifact bytes) is caught.
    """
    banner("9. CrossOrgEngagement — the whole fabric as one system")

    app = make_app("acme")
    eng = app.cross_org_engagement(buyer="acme", seller="vendor", scope="transcribe 1k calls")

    # Negotiate -> choreograph -> settle -> net -> prove solvency, all narrated.
    contract = eng.negotiate(
        buyer=buyer_position(max_price_usd=0.12, max_sla_seconds=5.0),
        seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.10),
    )
    print(f"negotiated ({eng.negotiation.status}): {contract.buyer} <=> {contract.seller} @ ${contract.terms.price_usd:.2f}")

    saga = (
        Saga(name="fulfil")
        .step("transcribe", participant="vendor", action="run", contract=contract)
        .step("deliver", participant="vendor", action="deliver", compensation="recall")
    )
    parts = {
        "vendor": {
            "run": lambda p: StepOutcome(ok=True, cost_usd=0.05, latency_ms=1200, quality=0.95, output={"text": "..."}),
            "deliver": lambda p: StepOutcome(ok=True, output={"delivered": True}),
            "recall": lambda p: {"recalled": True},
        }
    }
    delivery = eng.choreograph(saga, participants=parts)
    print(f"delivered ({delivery.status}): steps {delivery.completed_steps}; journal intact={delivery.journal.verify().intact}")

    records = eng.settle_saga(contracts={contract.id: contract})
    print(f"settled {len(records)} contract(s): " + ", ".join(f"{r.seller}:{r.status} (${r.balance_usd:+.2f})" for r in records))

    netting = eng.net()
    print(f"netted: clean={netting.clean}, {len(netting.positions)} position(s), {len(netting.obligations)} transfer(s)")

    reserves = eng.attest_custody("vendor", {"omnibus": 120.0})
    owed = eng.attest_liabilities("vendor", {"acme": 40.0, "globex": 30.0})
    solvency = eng.prove_solvency(reserves, owed)
    print(f"solvency ({solvency.status}): reserves ${reserves.reserves_usd:.0f} - liabilities ${owed.liabilities_usd:.0f} = margin ${solvency.margin_usd:+.0f}")

    # Seal into one signed narrative on the chain; verify the whole chain offline.
    narrative = eng.seal()
    whole = eng.verify(app.contract_signer)
    forged = EngagementNarrative.from_wire(narrative.to_wire())
    forged.stages[2], forged.stages[3] = forged.stages[3], forged.stages[2]  # re-order two stages
    print(f"sealed: {len(narrative.stages)} stages [{', '.join(narrative.stage_names)}]")
    print(
        f"verifies offline: valid={whole.valid} (chain intact={whole.intact}, artifacts match={whole.digests_ok}); "
        f"a re-ordered narrative is caught={not forged.verify().valid}"
    )


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
