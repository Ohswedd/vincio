"""Cross-org reputation attestation & portability — standing that travels.

Vincio lets agents negotiate contracts, run durable cross-org sagas, settle the
delivered work into signed records, net a fleet's books, and arbitrate a dispute —
each closing the reputation loop. But the standing they earn lives inside one org's
own ledger: a *new* counterparty has no way to trust it without a hosted reputation
bureau. This example adds the last rung: making earned standing **portable**. An
org issues a signed ``ReputationAttestation`` over a counterparty's standing, a
prospective counterparty verifies it from the bytes alone, and several issuers'
attestations combine into a bounded, evidence-weighted prior that weights the next
negotiation. It is reputation that travels the fabric, never a central service.

Five steps, all offline and deterministic:

  1. Two orgs each settle work delivered by a vendor and issue a signed attestation
     over its earned standing — derived only from their own signed records.
  2. Offline-verifiable: an attestation recomputes from the bytes alone; a tampered
     score is caught even after re-sealing, and a forged issuer is refused.
  3. A buyer with no local history imports the bundle: the evidence pools into one
     bounded prior, an issuer that vouches for itself is refused, and an unknown
     party falls back to the benefit-of-the-doubt prior.
  4. The imported prior weights a negotiation under the same ``[floor, 1]`` rule a
     local reputation does — a regressor is discounted without being singled out.
  5. Local standing still wins: a counterparty the buyer has lived through keeps its
     own earned ledger score over what others attest.

Everything here is opt-in and additive; this is a library capability inside your
process, never a hosted reputation service.
"""

from __future__ import annotations

from vincio import ContextApp, attest_reputation, combine_attestations
from vincio.negotiation import Contract, ContractTerms, buyer_position, seller_position
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner

ACME = HMACSigner("acme-key", key_id="acme")
GLOBEX = HMACSigner("globex-key", key_id="globex")


def a_contract(seller: str = "vendor", price: float = 0.10) -> Contract:
    return Contract(
        buyer="acme", seller=seller, terms=ContractTerms(scope="transcribe", price_usd=price)
    ).seal()


def main() -> None:
    # 1. Two orgs settle work the vendor delivered and attest its earned standing.
    acme = ContextApp(name="acme", provider=MockProvider(default_text="ok"))
    acme.use_settlement_book()
    for _ in range(4):
        acme.settle(a_contract("vendor"), cost_usd=0.06)  # delivered under price → fulfilled
    acme.settle(a_contract("vendor", 0.04), cost_usd=0.09)  # one overrun → breached
    acme_att = acme.attest_reputation("vendor")

    globex = ContextApp(name="globex", provider=MockProvider(default_text="ok"))
    globex.use_settlement_book()
    for _ in range(3):
        globex.settle(a_contract("vendor"), cost_usd=0.05)
    globex_att = globex.attest_reputation("vendor")

    print(
        f"1. Attestations issued — acme: {acme_att.settled}✓/{acme_att.breached}✗ "
        f"(reputation {acme_att.reputation:.3f}); globex: {globex_att.settled}✓/"
        f"{globex_att.breached}✗ (reputation {globex_att.reputation:.3f})."
    )

    # 2. Offline-verifiable: recomputes from the bytes; a tamper or forgery is caught.
    verified = acme_att.verify(acme.contract_signer).valid
    tampered = acme.attest_reputation("vendor")
    tampered.reputation = 0.99
    tampered.seal()  # recompute the hash to match the tampered score
    print(
        f"2. acme attestation verifies offline={verified}; a re-sealed tampered "
        f"score is still caught (evidence_sound={tampered.verify().evidence_sound})."
    )

    # 3. A buyer with no local history imports the bundle. A self-attestation (the
    #    vendor vouching for itself) is refused; the rest pools into one prior.
    vendor_self = attest_reputation(
        [acme.settlement_book.records[0]], "vendor", issuer="vendor"
    ).sign(GLOBEX)
    buyer = ContextApp(name="buyer", provider=MockProvider(default_text="ok"))
    buyer.use_reputation_ledger()
    prior = buyer.import_reputation([acme_att, globex_att, vendor_self])
    standing = prior.standing("vendor")
    print(
        f"3. Imported {len(prior.counted)} attestation(s), refused/excluded "
        f"{len(prior.refused) + len(prior.excluded)} (self-attestation): pooled "
        f"reputation {standing.reputation:.3f} from issuers {standing.issuers}; an "
        f"unknown party gets the prior weight {prior.weight('stranger'):.3f}."
    )

    # 4. The imported prior weights a negotiation against the never-before-seen vendor.
    result = buyer.negotiate(
        "transcribe 1k calls",
        buyer=buyer_position(max_price_usd=0.10, max_sla_seconds=5.0, ideal_price_usd=0.04),
        seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.12),
        buyer_id="buyer",
        seller_id="vendor",
    )
    print(
        f"4. Negotiation weighted by the imported prior: status={result.status}; "
        f"vendor's offers discounted by weight {prior.weight('vendor'):.3f} "
        f"(in [floor, 1] — discounted, never zeroed)."
    )

    # 5. Local standing still wins for a counterparty the buyer has lived through.
    for _ in range(5):
        buyer.reputation_ledger.record_outcome("vendor", passed=True, round_id="local")
    print(
        f"5. After local delivery, the buyer trusts its own ledger "
        f"({buyer.reputation_ledger.weight('vendor'):.3f}) over the imported prior — "
        f"effective weight now {prior.weight('vendor'):.3f}."
    )


if __name__ == "__main__":
    main()
