"""Cross-org reputation-gated admission & progressive exposure.

Reputation is now portable, current, discoverable, and trust-weighted — but it is
still only ever *consulted* as a soft weight on a negotiation; nothing **acts** on a
too-thin or too-low standing to bound how much a new counterparty is trusted with up
front. A brand-new or low-trust counterparty is admitted to a contract on the same
terms as a long-trusted one, with the regression caught only after the fact. This
example adds the next rung: an ``AdmissionPolicy`` that maps the standing the fabric
already earns to a **graduated exposure posture** — a maximum contract value, a
required escrow fraction, and an SLA-strictness factor — so onboarding an unknown org
is safe by construction, not by hope.

Five steps, all offline and deterministic:

  1. A brand-new counterparty is admitted on *conservative* terms — a discounted
     ceiling and posted collateral — rather than refused. Discounted exposure, never
     a hard gate.
  2. As corroborated, settled history accrues the ceiling **ramps** toward parity, the
     escrow fraction falling away — trust earned over real deliveries unlocks exposure
     the way a credit line builds.
  3. A regression walks the ceiling back — bounded and reversible, pinpointed at every
     step.
  4. Every decision is offline-verifiable: the exposure terms re-derive from the bound
     standing, so a tampered ceiling is caught from the bytes alone, and the decision
     lands on the hash-chained audit log.
  5. The decision folds into the existing path: it clamps a buyer's negotiating
     position to the ceiling, so the bargain can only converge within the admitted
     exposure — no new code path through the engine.

Everything here is opt-in and additive; this is a library capability inside your
process, never a hosted underwriting service.
"""

from __future__ import annotations

from vincio import AdmissionConfig, ContextApp, attest_reputation, settle_contract
from vincio.negotiation import Contract, ContractTerms, buyer_position, seller_position
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner

ACME = HMACSigner("acme-key", key_id="acme")
GLOBEX = HMACSigner("globex-key", key_id="globex")


def a_contract(issuer: str, subject: str, price: float = 0.10) -> Contract:
    return Contract(
        buyer=issuer, seller=subject, terms=ContractTerms(scope="transcribe", price_usd=price)
    ).seal()


def attestation(issuer: str, subject: str, signer: HMACSigner, *, settled: int = 0):
    """A signed attestation by ``issuer`` over ``subject``'s earned standing."""
    records = [settle_contract(a_contract(issuer, subject), cost_usd=0.06) for _ in range(settled)]
    return attest_reputation(records, subject, issuer=issuer).sign(signer)


def main() -> None:
    # A policy that admits a fully-trusted counterparty to $1,000 contracts, demanding
    # up to half in collateral and a 2× tighter SLA at the conservative end.
    policy = AdmissionConfig(parity_exposure_usd=1000.0, max_escrow_fraction=0.5, min_sla_factor=0.5)

    buyer = ContextApp(name="buyer", provider=MockProvider(default_text="ok"))
    buyer.use_reputation_ledger()

    # 1. A brand-new counterparty: admitted conservatively, never refused.
    newcomer = buyer.admit("vendor", config=policy)
    print(
        f"1. New counterparty admitted on conservative terms: "
        f"${newcomer.max_contract_value_usd:,.0f} ceiling "
        f"({newcomer.exposure_fraction:.0%} of parity), {newcomer.escrow_fraction:.0%} escrow, "
        f"SLA ×{newcomer.sla_factor:.2f} — discounted exposure, never a hard gate."
    )

    # 2. Settled, corroborated history ramps the ceiling toward parity. Two other orgs
    #    attest the vendor's deliveries; with no local history yet the imported prior
    #    decides — corroboration from several issuers, not one self-asserted number.
    buyer.import_reputation(
        [
            attestation("acme", "vendor", ACME, settled=6),
            attestation("globex", "vendor", GLOBEX, settled=6),
        ]
    )
    ramped = buyer.admit("vendor", config=policy)
    print(
        f"2. After settled, corroborated history (issuers {ramped.issuers}): "
        f"ceiling ramps to ${ramped.max_contract_value_usd:,.0f} "
        f"({ramped.exposure_fraction:.0%} of parity), escrow down to {ramped.escrow_fraction:.0%} — "
        f"exposure unlocked the way a credit line builds."
    )

    # 3. A regression walks the ceiling back. The buyer now lives through breaches of
    #    its own; local first-hand evidence wins over what others still attest.
    for _ in range(15):
        buyer.reputation_ledger.record_outcome("vendor", passed=False, round_id="breach")
    walked_back = buyer.admit("vendor", config=policy)
    print(
        f"3. After a regression the buyer lived through: ceiling walked back to "
        f"${walked_back.max_contract_value_usd:,.0f} ({walked_back.exposure_fraction:.0%} of "
        f"parity), escrow back up to {walked_back.escrow_fraction:.0%} — local evidence wins, "
        f"bounded and reversible."
    )

    # 4. Offline-verifiable: the terms re-derive from the bound standing; a tampered
    #    ceiling is caught even after re-sealing, and the decision is on the audit chain.
    verified = ramped.verify().valid
    tampered = buyer.admit("vendor", config=policy, record_audit=False)
    tampered.max_contract_value_usd = 1_000_000.0
    tampered.seal()  # recompute the hash to match the inflated ceiling
    print(
        f"4. Offline-verifiable: a sound decision verifies={verified}; a re-sealed inflated "
        f"ceiling is still caught (terms_sound={tampered.verify().terms_sound}); "
        f"{len(buyer.audit.query(action='reputation_admission'))} decisions on the audit chain."
    )

    # 5. Folds into the existing negotiation path: the ceiling clamps the buyer's
    #    position, so the bargain can only converge within the admitted exposure.
    admit = buyer.admit("vendor", config=AdmissionConfig(parity_exposure_usd=0.08))
    bounded = admit.bound_position(
        buyer_position(max_price_usd=10.0, ideal_price_usd=0.01, max_sla_seconds=5.0)
    )
    result = buyer.negotiate(
        "transcribe 1k calls",
        buyer=bounded,
        seller=seller_position(min_price_usd=0.01, ideal_price_usd=0.05),
        buyer_id="buyer",
        seller_id="vendor",
    )
    closed = result.contract.terms.price_usd if result.agreed else None
    print(
        f"5. Folded into the negotiation path: the bargain converges at "
        f"${closed:.4f} ≤ ${admit.max_contract_value_usd:.4f} ceiling — the admitted "
        f"exposure bounds the deal, no new code path through the engine."
    )


if __name__ == "__main__":
    main()
