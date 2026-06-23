"""Cross-org engagement lifecycle — the settlement & credit fabric as one system.

Twenty rungs delivered the cross-org *primitives* — negotiation and contracting,
choreographed delivery, metering and settlement, multilateral netting, dispute
arbitration, portable reputation, reputation-gated admission, collateral escrow /
pooling / rehypothecation guards, proof-of-reserves and proof-of-solvency, liability
completeness / non-equivocation / history, and insolvency resolution by seniority
waterfall with close-out set-off — each signed, content-bound, and offline-verifiable
on its own. This capstone example composes them: a single `CrossOrgEngagement`
(`app.cross_org_engagement`) threads the whole pipeline behind one governed, audited
call-path and seals it into one hash-linked, offline-verifiable narrative.

The facade is *purely compositional* — every lifecycle method delegates to the same
`app.*` entry point you could call directly, so the primitives stay unchanged and
usable on their own; the engagement only captures and **narrates** them.

Seven steps, all offline and deterministic:

  1. Negotiate the contract for the engagement's scope between two orgs.
  2. Choreograph the contracted delivery as a durable saga — the participant for one
     step is *discovered* from the governed directory by capability.
  3. Close the books: settle every contract the saga ran under.
  4. Net the settled books into one minimal cleared set.
  5. Prove the counterparty's solvency (proof-of-reserves vs proof-of-liabilities).
  6. Seal the engagement into a signed, content-bound narrative on the audit chain.
  7. Verify the whole chain — and every captured artifact — offline; a tamper anywhere
     is caught.

Everything here is opt-in and additive; this is a library capability inside your
process, never a hosted marketplace, clearing house, or payment processor.
"""

from __future__ import annotations

from vincio import ContextApp, EngagementNarrative
from vincio.a2a.protocol import AgentCard, AgentSkill
from vincio.choreography import Saga, StepOutcome
from vincio.negotiation import buyer_position, seller_position
from vincio.providers import MockProvider


def _vendor_card(name: str, capability: str = "transcription") -> AgentCard:
    return AgentCard(
        name=name,
        description=f"{name} — performs {capability}",
        skills=[AgentSkill(id="run", name="run", description=capability, tags=[capability])],
    )


def main() -> None:
    app = ContextApp(name="acme", provider=MockProvider(default_text="ok"))

    # One facade threads the whole fabric. It ensures a durable settlement book and a
    # reputation ledger, then narrates every primitive it delegates to.
    eng = app.cross_org_engagement(buyer="acme", seller="vendor", scope="transcribe 1k calls")

    # 1. Negotiate the contract for the engagement's scope.
    contract = eng.negotiate(
        buyer=buyer_position(max_price_usd=0.12, max_sla_seconds=5.0),
        seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.10),
    )
    print(
        f"1. Negotiated ({eng.negotiation.status}): {contract.buyer} ⇄ {contract.seller} "
        f"@ ${contract.terms.price_usd:.2f} for {contract.terms.scope!r}"
    )

    # 2. Choreograph the contracted delivery. The first step is *discovered* — its
    #    participant is resolved by capability from the governed, allow-listed directory.
    directory = app.agent_directory(allow=["vendor*"])
    directory.register(_vendor_card("vendor"))
    saga = (
        Saga(name="fulfil")
        .step("transcribe", action="run", capability="transcription", contract=contract)
        .step("deliver", participant="vendor", action="deliver", compensation="recall")
    )
    parts = {
        "vendor": {
            "run": lambda p: StepOutcome(ok=True, cost_usd=0.05, latency_ms=1200, quality=0.95, output={"text": "…"}),
            "deliver": lambda p: StepOutcome(ok=True, output={"delivered": True}),
            "recall": lambda p: {"recalled": True},
        }
    }
    delivery = eng.choreograph(saga, participants=parts, directory=directory)
    bound = delivery.bindings["transcribe"].org
    print(
        f"2. Delivered ({delivery.status}): steps {delivery.completed_steps}; "
        f"'transcribe' discovered → {bound}; journal intact={delivery.journal.verify().intact}"
    )

    # 3. Close the books on every contract the saga ran under.
    records = eng.settle_saga(contracts={contract.id: contract})
    print(
        f"3. Settled {len(records)} contract(s): "
        + ", ".join(f"{r.seller}:{r.status} (balance ${r.balance_usd:+.2f})" for r in records)
    )

    # 4. Net the settled books into one minimal cleared set.
    netting = eng.net()
    print(
        f"4. Netted: clean={netting.clean}, {len(netting.positions)} position(s), "
        f"{len(netting.obligations)} cleared transfer(s); verifies offline={netting.verify().valid}"
    )

    # 5. Prove the counterparty solvent: proof-of-reserves against proof-of-liabilities.
    reserves = eng.attest_custody("vendor", {"omnibus": 120.0})
    owed = eng.attest_liabilities("vendor", {"acme": 40.0, "globex": 30.0})
    solvency = eng.prove_solvency(reserves, owed)
    print(
        f"5. Solvency ({solvency.status}): reserves ${reserves.reserves_usd:.0f} − "
        f"liabilities ${owed.liabilities_usd:.0f} = margin ${solvency.margin_usd:+.0f}"
    )

    # 6. Seal the whole engagement into one signed, content-bound narrative on the chain.
    narrative = eng.seal()
    print(
        f"6. Engagement sealed: {len(narrative.stages)} stages "
        f"[{', '.join(narrative.stage_names)}]; "
        f"{len(app.audit.query(action='cross_org_engagement'))} engagement(s) on the chain."
    )

    # 7. Verify the whole chain — and every captured artifact — offline. A tamper anywhere
    #    is caught: re-order a stage, edit an artifact's bytes, and verification fails.
    whole = eng.verify(app.contract_signer)
    forged = EngagementNarrative.from_wire(narrative.to_wire())
    forged.stages[2], forged.stages[3] = forged.stages[3], forged.stages[2]  # re-order two stages
    print(
        f"7. Verifies offline: valid={narrative.verify(app.contract_signer).valid}, "
        f"chain intact={whole.intact}, artifacts match={whole.digests_ok}, "
        f"signed_by={whole.signed_by}; a re-ordered narrative is caught="
        f"{not forged.verify().valid}; audit chain intact={app.audit.verify_chain()}."
    )

    assert whole.valid and not forged.verify().valid


if __name__ == "__main__":
    main()
