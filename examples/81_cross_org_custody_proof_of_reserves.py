"""Cross-org collateral custody attestation & proof-of-reserves.

A ``CollateralLedger`` bounds a counterparty's pledges across its ``CollateralPool``\\ s
against the capital it actually ``held`` — but that holdings figure was the one input the
guard *trusted*: it was **asserted**, not proven. A counterparty over-stating its real
reserves still passed the guard, the way a self-asserted reputation score passed before an
attestation made standing verifiable. This example adds the next rung: a ``CustodyAttestation``
— a signed, content-bound **proof-of-reserves** the guard reads as the held figure, so a
re-use bound rests on a proven number rather than a promise.

Four steps, all offline and deterministic:

  1. A custodian attests the capital a vendor actually holds — itemized into reserve accounts
     whose total re-derives on every verify — and the guard reads it as the held figure
     (``custody=``), bounding the pledges against **proven** reserves.
  2. When the proven reserves fall below what the pools pledge, the shortfall surfaces as a
     bounded, pinpointed **under-reserved breach** rather than passing on an inflated claim.
  3. The proof reads only signed, content-bound artifacts: a tampered reserve figure, a
     forged custodian, and an attestation for a different poster are each **refused**.
  4. Every attestation and guard lands on the hash-chained audit log and verifies offline
     from the bytes alone — `app.attest_custody` / `app.guard_collateral`.

Everything here is opt-in and additive; this is a library capability inside your process,
never a hosted custodian, a proof-of-reserves auditor, or a trusted third party.
"""

from __future__ import annotations

from vincio import ContextApp, attest_custody, guard_collateral, post_collateral_pool
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider


def a_contract(buyer: str, seller: str, scope: str, price: float) -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope=scope, price_usd=price)
    ).seal()


def main() -> None:
    # A custodian org attests a vendor's reserves; an acme buyer guards the vendor's pools.
    custodian = ContextApp(name="custodian", provider=MockProvider(default_text="ok"))
    custodian.use_settlement_book()
    acme = ContextApp(name="acme", provider=MockProvider(default_text="ok"))
    acme.use_settlement_book()

    # The vendor backs two contracts with one pool.
    c_a = a_contract("acme", "vendor", "transcribe batch A", price=100.0)
    c_b = a_contract("acme", "vendor", "transcribe batch B", price=200.0)
    pool = acme.post_collateral_pool([c_a, c_b], fraction=0.1)  # pledges 30

    # 1. The custodian issues a signed proof-of-reserves over the capital the vendor holds.
    #    The guard reads it as the held figure instead of an asserted number — the proof's
    #    content hash and re-derived total make it tamper-evident from the bytes alone.
    proof = custodian.attest_custody("vendor", {"omnibus": 40.0, "escrow": 10.0})  # 50 proven
    covered = acme.guard_collateral([pool], custody=proof)
    print(
        f"1. Custodian attests ${proof.reserves_usd:,.2f} across {len(proof.reserves)} accounts "
        f"(signature verifies={proof.verify(custodian.contract_signer).valid}); the guard bounds "
        f"${covered.pledged_usd:,.2f} pledged against proven reserves "
        f"(reserves_proven={covered.reserves_proven}, under_reserved={covered.under_reserved})."
    )

    # 2. A thinner attestation proves less than the pledges — the shortfall is pinpointed.
    thin = custodian.attest_custody("vendor", {"omnibus": 20.0})  # only 20 proven
    under = acme.guard_collateral([pool], custody=thin)
    breach = under.reserve_breach
    print(
        f"2. A ${thin.reserves_usd:,.2f} proof against ${under.pledged_usd:,.2f} pledged is "
        f"under-reserved by ${breach.shortfall_usd:,.2f} — pinpointed against custodian "
        f"{breach.custodian!r}; require_reserved() would raise."
    )

    # 3. The proof reads only signed, content-bound artifacts. A tampered reserve figure and an
    #    attestation for a different poster are caught from the bytes alone; a forged custodian
    #    is caught when the guard is given the custodian's verifier.
    tampered = attest_custody("vendor", {"omnibus": 50.0})
    tampered.reserves_usd = 9_999.0  # lie about the total ...
    tampered.seal()  # ... and re-seal; the total no longer re-derives
    try:
        guard_collateral([pool], custody=tampered)
        tamper_refused = False
    except SettlementError:
        tamper_refused = True
    wrong = attest_custody("globex", 50.0)  # attests globex, not the pool's vendor
    try:
        guard_collateral([pool], custody=wrong)
        poster_refused = False
    except SettlementError:
        poster_refused = True
    # A forged-custodian demo over an unsigned pool, so one verifier checks only the proof.
    bare_pool = post_collateral_pool([a_contract("acme", "vendor", "y", 100.0)], fraction=0.3)
    forged = attest_custody("vendor", {"omnibus": 20.0}, custodian="custodian")
    forged.sign(custodian.contract_signer)
    from vincio.security.audit import HMACSigner

    forged.signatures[0].signature = HMACSigner("forger-key").sign(forged.content_hash)
    try:
        guard_collateral([bare_pool], custody=forged, verify_with=custodian.contract_signer)
        forged_refused = False
    except SettlementError:
        forged_refused = True
    print(
        f"3. Tampered reserve figure refused: {tamper_refused}; wrong-poster attestation "
        f"refused: {poster_refused}; forged custodian refused with the verifier: {forged_refused}."
    )

    # 4. Auditable & offline: the attestation and the guard are on the hash-chained audit logs
    #    and verify from the bytes alone.
    print(
        f"4. Offline-verifiable: proof verifies={proof.verify(custodian.contract_signer).valid}, "
        f"guard verifies={under.verify(acme.contract_signer).valid}; "
        f"{len(custodian.audit.query(action='custody_attestation'))} attestations + "
        f"{len(acme.audit.query(action='rehypothecation'))} guards on the chains, "
        f"books intact={acme.settlement_book.verify(acme.contract_signer).intact}."
    )


if __name__ == "__main__":
    main()
