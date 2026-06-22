"""Cross-org transitive trust & Sybil-resistant attestation weighting.

Reputation is now portable, current, and discoverable — but every counted issuer's
evidence pools into the prior with equal pull, weighted only by *how much* it
attests, not by *how much the importer trusts the issuer*. A clutch of unknown peers
can therefore out-evidence a few an importer has lived through, and an adversary can
spin up *Sybil* issuers that all vouch the same way. This example adds the next
rung — weighing each issuer's evidence by the importer's **own trust in that
issuer**, a bounded, transitive web-of-trust rooted in its local ledger — so pull
follows earned trust, not issuer count, with no central trust authority.

Five steps, all offline and deterministic:

  1. A buyer pools two equal attestations with no trust weighting: equal pull, the
     pre-trust behavior — strictly opt-in.
  2. Turn on issuer-trust weighting (``trust_config``): the issuer the buyer knows
     first-hand out-pulls the unknown one with equal evidence, the unknown one
     floored rather than zeroed, every multiplier pinpointed.
  3. Sybil resistance: a clutch of unknown issuers all vouching the same way cannot
     out-evidence one trusted issuer's adverse outcomes.
  4. Bounded transitivity: a trusted issuer lends weight one hop to an issuer *it*
     attests; a chain beyond the depth bound manufactures no standing.
  5. The weighted prior drops into the negotiation path unchanged — the same
     ``weight(member_id)`` a local reputation exposes.

Everything here is opt-in and additive; this is a library capability inside your
process, never a hosted trust authority or a reputation bureau.
"""

from __future__ import annotations

from vincio import (
    ContextApp,
    TrustConfig,
    attest_reputation,
    build_trust_model,
    combine_attestations,
    settle_contract,
)
from vincio.negotiation import (
    Contract,
    ContractTerms,
    buyer_position,
    select_offer,
    seller_position,
)
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner


def a_contract(issuer: str, subject: str, price: float = 0.10) -> Contract:
    return Contract(
        buyer=issuer, seller=subject, terms=ContractTerms(scope="transcribe", price_usd=price)
    ).seal()


def attestation(issuer: str, subject: str, *, settled: int = 0, breached: int = 0):
    """A signed attestation by ``issuer`` over ``subject``'s earned standing."""
    records = [settle_contract(a_contract(issuer, subject), cost_usd=0.06) for _ in range(settled)]
    records += [
        settle_contract(a_contract(issuer, subject, price=0.04), cost_usd=0.09)
        for _ in range(breached)
    ]
    return attest_reputation(records, subject, issuer=issuer).sign(HMACSigner(f"{issuer}-key", key_id=issuer))


def main() -> None:
    # The buyer knows "acme" first-hand from past dealings; "stranger" is unknown.
    buyer = ContextApp(name="buyer", provider=MockProvider(default_text="ok"))
    buyer.use_reputation_ledger()
    for _ in range(10):
        buyer.reputation_ledger.record_outcome("acme", passed=True, round_id="prior-deal")

    trusted = attestation("acme", "vendor", settled=4)
    unknown = attestation("stranger", "vendor", settled=4)

    # 1. No trust weighting → equal pull, exactly as before (opt-in).
    plain = combine_attestations([trusted, unknown])
    print(
        f"1. Equal pull (no trust): vendor rests on {plain.standing('vendor').issuers} with "
        f"{plain.standing('vendor').successes:g} pooled successes — the pre-trust behavior."
    )

    # 2. Turn on issuer-trust weighting: the known issuer out-pulls the unknown one.
    weighted = buyer.import_reputation([trusted, unknown], trust_config=TrustConfig(), weight=False)
    standing = weighted.standing("vendor")
    print(
        f"2. Trust-weighted: acme (known first-hand) pulls at "
        f"{standing.issuer_trust['acme']:.2f}, stranger (unknown) floored at "
        f"{standing.issuer_trust['stranger']:.2f} — discounted, never zeroed, pinpointed."
    )

    # 3. Sybil resistance: five unknown issuers vouch glowingly; one trusted issuer
    #    reports the vendor regressed. Pull follows trust, so the trusted word wins.
    sybils = [attestation(f"sybil{i}", "vendor", settled=4) for i in range(5)]
    trusted_bad = attestation("acme", "vendor", breached=4)
    sybil_weighted = buyer.import_reputation(
        [*sybils, trusted_bad], trust_config=TrustConfig(), weight=False
    )
    sybil_plain = combine_attestations([*sybils, trusted_bad])
    print(
        f"3. Sybil resistance: 5 unknown issuers vouch positive, 1 trusted reports a "
        f"regression — weighted reputation {sybil_weighted.standing('vendor').reputation:.2f} "
        f"vs naive {sybil_plain.standing('vendor').reputation:.2f}; the Sybils cannot outvote "
        f"earned trust (each floored at "
        f"{sybil_weighted.standing('vendor').issuer_trust['sybil0']:.2f})."
    )

    # 4. Bounded transitivity: acme vouches for "broker" as a counterparty; broker then
    #    attests the vendor — so broker inherits trust one hop out, decayed.
    acme_on_broker = attestation("acme", "broker", settled=8)
    broker_on_vendor = attestation("broker", "vendor", settled=4)
    one_hop = build_trust_model(
        [acme_on_broker, broker_on_vendor], base=buyer.reputation_ledger, config=TrustConfig(max_depth=1)
    )
    zero_hop = build_trust_model(
        [acme_on_broker, broker_on_vendor], base=buyer.reputation_ledger, config=TrustConfig(max_depth=0)
    )
    broker = one_hop.assessment("broker")
    print(
        f"4. Transitive (≤1 hop): broker vouched by {broker.vouched_by} inherits trust "
        f"{broker.trust:.2f} (< acme's {one_hop.trust_in('acme'):.2f}, attenuated by decay); "
        f"with the depth bound at 0 it stays floored at {zero_hop.trust_in('broker'):.2f}."
    )

    # 5. The weighted prior drops into the negotiation path unchanged.
    pos = buyer_position(max_price_usd=0.10, max_sla_seconds=5.0)
    reliable_offer = buyer.negotiate(
        "transcribe", buyer=pos, seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.12),
        buyer_id="buyer", seller_id="acme",
    )
    unknown_offer = buyer.negotiate(
        "transcribe", buyer=pos, seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.12),
        buyer_id="buyer", seller_id="stranger",
    )
    # acme attests itself reliable as a seller; stranger has a thin, unknown standing.
    seller_prior = buyer.import_reputation(
        [attestation("globex", "acme", settled=8), attestation("globex", "stranger", breached=6)],
        trust_config=TrustConfig(),
        weight=False,
    )
    chosen = select_offer([reliable_offer, unknown_offer], pos, reputation=seller_prior)
    print(
        f"5. The trust-weighted prior weights the negotiation unchanged: the buyer "
        f"selects {chosen.seller!r} — the same weight(member_id) a local reputation exposes."
    )


if __name__ == "__main__":
    main()
