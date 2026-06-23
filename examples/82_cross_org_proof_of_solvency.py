"""Cross-org custody liability attestation & proof-of-solvency.

A ``CustodyAttestation`` proves the capital a counterparty *holds*, so the rehypothecation
guard bounds its pledges against a **proven** reserve figure. But reserves are only one side
of the ledger: a counterparty solvent against *one* buyer's pledges may be deeply **under-water**
once *every* obligation it owes is counted. The guard sees the reserves and this buyer's
pledges, never the counterparty's other liabilities — so a counterparty could prove the same
reserves against many buyers while quietly insolvent across all of them. This example adds the
next rung the proof-of-reserves literature pairs with: a **proof-of-solvency** (reserves ≥
liabilities), folding a liability proof against the reserve proof into a solvency-adjusted held
figure the guard bounds pledges against.

Four steps, all offline and deterministic:

  1. An auditor attests the obligations a vendor owes (the liability side), and `prove_solvency`
     folds it against the proven reserves into a `SolvencyProof` — a bounded solvency margin
     (reserves − liabilities) the guard reads (`solvency=`) as the held figure, bounding pledges
     against capital **not already owed elsewhere**.
  2. When the proven liabilities exceed the proven reserves, the shortfall surfaces as a bounded,
     pinpointed **insolvency breach** — a counterparty that proves the same reserves against many
     buyers while insolvent across all of them is caught.
  3. The proof reads only signed, content-bound artifacts: a tampered liability figure, a
     custody/liability pair for **different posters**, and a tampered solvency proof are each
     **refused**.
  4. Every attestation and proof lands on the hash-chained audit log and verifies offline from
     the bytes alone — `app.attest_liabilities` / `app.prove_solvency` / `app.guard_collateral`.

Everything here is opt-in and additive; this is a library capability inside your process,
never a hosted solvency auditor, a proof-of-reserves auditor, or a trusted third party.
"""

from __future__ import annotations

from vincio import (
    ContextApp,
    attest_liabilities,
    guard_collateral,
    prove_solvency,
)
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider


def a_contract(buyer: str, seller: str, scope: str, price: float) -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope=scope, price_usd=price)
    ).seal()


def main() -> None:
    # An auditor org folds a vendor's reserves and liabilities; an acme buyer guards its pools.
    auditor = ContextApp(name="auditor", provider=MockProvider(default_text="ok"))
    auditor.use_settlement_book()
    acme = ContextApp(name="acme", provider=MockProvider(default_text="ok"))
    acme.use_settlement_book()

    # The vendor backs two contracts with one pool (pledges 30).
    c_a = a_contract("acme", "vendor", "transcribe batch A", price=100.0)
    c_b = a_contract("acme", "vendor", "transcribe batch B", price=200.0)
    pool = acme.post_collateral_pool([c_a, c_b], fraction=0.1)  # pledges 30

    # 1. The vendor proves $80 of reserves — but it owes $50 to other creditors. prove_solvency
    #    folds the two proofs into a solvency margin (80 − 50 = 30 free), and the guard bounds
    #    the $30 pledged against the $30 unencumbered capital, not the gross reserves.
    reserves = auditor.attest_custody("vendor", {"omnibus": 80.0})  # 80 held
    owed = auditor.attest_liabilities("vendor", {"globex": 35.0, "initech": 15.0})  # 50 owed
    proof = auditor.prove_solvency(reserves, owed)  # margin = 30 free
    covered = acme.guard_collateral([pool], solvency=proof)
    print(
        f"1. Reserves ${proof.reserves_usd:,.2f} − liabilities ${proof.liabilities_usd:,.2f} = "
        f"${proof.margin_usd:,.2f} free (solvent={proof.solvent}); the guard bounds "
        f"${covered.pledged_usd:,.2f} pledged against ${covered.held_usd:,.2f} unencumbered "
        f"(solvency_adjusted={covered.solvency_adjusted}, under_reserved={covered.under_reserved})."
    )

    # 2. The same $80 reserves, but $120 owed across every buyer: insolvent by $40. The shortfall
    #    is pinpointed, and the guard sees zero free capital behind the pledges.
    deep_owed = auditor.attest_liabilities("vendor", {"globex": 70.0, "initech": 50.0})  # 120
    insolvent = auditor.prove_solvency(reserves, deep_owed)
    under = acme.guard_collateral([pool], solvency=insolvent)
    breach = insolvent.breach
    print(
        f"2. Against ${insolvent.liabilities_usd:,.2f} owed, ${insolvent.reserves_usd:,.2f} "
        f"reserves are insolvent by ${breach.shortfall_usd:,.2f} — pinpointed against attestor "
        f"{breach.attestor!r}; the guard sees ${under.held_usd:,.2f} free "
        f"(insolvent={under.insolvent}); require_solvent() would raise."
    )

    # 3. The proof reads only signed, content-bound artifacts. A tampered liability figure, a
    #    custody/liability pair for different posters, and a tampered solvency proof are caught
    #    from the bytes alone.
    tampered = attest_liabilities("vendor", {"globex": 50.0})
    tampered.liabilities_usd = 1.0  # under-state the debt ...
    tampered.seal()  # ... and re-seal; the total no longer re-derives
    try:
        prove_solvency(reserves, tampered)
        tamper_refused = False
    except SettlementError:
        tamper_refused = True
    wrong = attest_liabilities("globex", 50.0)  # attests globex, not the reserves' vendor
    try:
        prove_solvency(reserves, wrong)
        poster_refused = False
    except SettlementError:
        poster_refused = True
    forged = prove_solvency(reserves, owed)
    forged.margin_usd = 9_999.0  # overstate the free capital ...
    forged.seal()  # ... and re-seal; the margin no longer re-derives
    try:
        guard_collateral([pool], solvency=forged)
        proof_refused = False
    except SettlementError:
        proof_refused = True
    print(
        f"3. Tampered liability figure refused: {tamper_refused}; wrong-poster pair refused: "
        f"{poster_refused}; tampered solvency proof refused at the guard: {proof_refused}."
    )

    # 4. Auditable & offline: the attestations and the proof are on the hash-chained audit logs
    #    and verify from the bytes alone.
    print(
        f"4. Offline-verifiable: proof verifies={proof.verify(auditor.contract_signer).valid}, "
        f"guard verifies={under.verify(acme.contract_signer).valid}; "
        f"{len(auditor.audit.query(action='liability_attestation'))} liability + "
        f"{len(auditor.audit.query(action='solvency_proof'))} solvency entries on the auditor's "
        f"chain, {len(acme.audit.query(action='rehypothecation'))} guards on acme's, "
        f"books intact={acme.settlement_book.verify(acme.contract_signer).intact}."
    )


if __name__ == "__main__":
    main()
