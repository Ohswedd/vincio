"""Cross-org dispute resolution & arbitration — settling which figure stands.

Vincio lets agents negotiate contracts, run durable cross-org sagas, settle the
delivered work into signed records, and net a fleet's books into a minimal cleared
set — pinpointing any disagreement as a dispute. This example adds the next rung:
**resolving** that dispute. Each party submits its signed settlement records and a
deterministic ``Arbitration`` decides which figure stands — a reconciliation hash
both parties co-signed is upheld, a contradicting unilateral claim is rejected and
pinpointed, and a genuine standoff is left honestly unresolved. It is a library-side
protocol, never a hosted arbitration service or a court of record.

Five steps, all offline and deterministic:

  1. A disagreement: netting pinpoints two books that disagree on a contract.
  2. Resolve it — but neither figure is corroborated, so it is left unresolved.
  3. The seller co-signs the buyer's figure: now a record stands, upheld.
  4. A bad-faith revision: a unilateral claim contradicting the co-signed truth is
     rejected and its claimant pinpointed (and debited on reputation).
  5. Offline-verifiable: the resolution is content-bound and signed; it recomputes
     from the bytes alone, and a tampered claim is marked inadmissible, not dropped.

Everything here is opt-in and additive; this is a library capability inside your
process, never a hosted arbitration service or a payment processor.
"""

from __future__ import annotations

from vincio import ContextApp, arbitrate, net_settlements, settle_contract
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner

BUYER = HMACSigner("acme-key", key_id="acme")
SELLER = HMACSigner("vendor-key", key_id="vendor")


def a_contract(price: float = 0.10) -> Contract:
    return Contract(
        buyer="acme", seller="vendor", terms=ContractTerms(scope="work", price_usd=price)
    ).seal()


def claim(contract: Contract, *, cost: float, signer: HMACSigner, party: str):
    return settle_contract(contract, cost_usd=cost).sign(signer, party=party)


def main() -> None:
    app = ContextApp(name="arbiter", provider=MockProvider(default_text="ok"))
    app.use_reputation_ledger()

    # 1. A disagreement: acme's book says cost 0.08 (balance 0.02), vendor's says
    #    cost 0.05 (balance 0.05). Netting pinpoints the contract as a dispute.
    c = a_contract(0.10)
    acme_says = claim(c, cost=0.08, signer=BUYER, party="acme")
    vendor_says = claim(c, cost=0.05, signer=SELLER, party="vendor")
    netting = net_settlements([acme_says, vendor_says])
    print(
        f"1. Netting clean={netting.clean}; dispute pinpointed on "
        f"{[d.contract_id[:12] for d in netting.disputes]} (excluded from clearing)."
    )

    # 2. Arbitrate: each party submits its signed record. Neither figure is
    #    corroborated by both sides, so the dispute is honestly left unresolved.
    standoff = arbitrate([acme_says, vendor_says], contract_id=c.id)
    print(f"2. Arbitrated: status={standoff.status} — {standoff.reason}")

    # 3. The seller co-signs the buyer's figure (cost 0.08). Now both sides agree on
    #    one reconciliation hash, so that record stands — upheld.
    vendor_agrees = claim(c, cost=0.08, signer=SELLER, party="vendor")
    res = app.arbitrate([acme_says, vendor_agrees])
    print(
        f"3. Both co-sign: status={res.status}, balance ${res.upheld_balance_usd:+.2f} "
        f"corroborated by {res.corroborated_by}."
    )

    # 4. A bad-faith revision: the seller later submits a contradicting unilateral
    #    claim (cost 0.05). It is rejected, pinpointed, and debited on reputation.
    before = app.reputation_ledger.snapshot("vendor").reputation
    liar = claim(c, cost=0.05, signer=SELLER, party="vendor")
    disputed = app.arbitrate([acme_says, vendor_agrees, liar])
    after = app.reputation_ledger.snapshot("vendor").reputation
    print(
        f"4. Bad-faith revision: status={disputed.status}; rejected "
        f"{[cl.settlement_id[:12] for cl in disputed.rejected_claims]}; "
        f"dissenters={disputed.dissenters}; reputation {before:.3f} → {after:.3f}."
    )

    # 5. The resolution is content-bound and offline-verifiable; a tampered claim is
    #    marked inadmissible, never silently dropped.
    verdict = disputed.verify(app.contract_signer)
    bad = claim(c, cost=0.08, signer=SELLER, party="vendor")
    bad.amount_owed_usd = 999.0  # tamper without resealing
    with_bad = arbitrate([acme_says, vendor_agrees, bad])
    print(
        f"5. Verifies offline={verdict.valid} (hash={verdict.hash_ok}, "
        f"decision={verdict.decision_sound}); signed by {disputed.signed_by}; "
        f"on the audit chain={bool(app.audit.query(action='arbitration'))}. "
        f"Tampered claim inadmissible="
        f"{[cl.settlement_id[:12] for cl in with_bad.inadmissible_claims]}."
    )


if __name__ == "__main__":
    main()
