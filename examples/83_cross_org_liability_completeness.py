"""Cross-org liability inclusion proofs & completeness.

A ``SolvencyProof`` folds a proven liability total against the proven reserves, so the
rehypothecation guard bounds pledges against capital not already owed elsewhere. But the
liability *total* is still the attestor's single number: a counterparty could **under-state**
what it owes by quietly omitting a creditor and still attest a sound, re-deriving total over the
creditors it *did* list. This example adds the canonical second half of a proof-of-liabilities —
each creditor proves its own claim is **included** in the attested total, so the total is provably
**complete**, not merely internally consistent.

Four steps, all offline and deterministic:

  1. The attestation commits its line items into a Merkle root, and each creditor gets an
     offline-verifiable `InclusionProof` that its claim is a leaf of that root — a poster cannot
     drop a creditor without the omitted party detecting it.
  2. A creditor folds its own claim against the attestation with `check_completeness`, surfacing
     an **omission breach** when the attested liabilities exclude (or under-state) a claim it can
     prove, and raising the attested figure to a **completed** total.
  3. `prove_solvency(completeness=)` bounds the solvency margin by the *completed* liability
     total rather than the attestor's figure — tipping a counterparty that looked solvent on the
     attestor's number into a pinpointed insolvency.
  4. The inclusion and completeness proofs read only signed, content-bound artifacts: a tampered
     leaf, a dropped omission, and a completeness check for a different attestation are each
     **refused**, and the check lands on the hash-chained audit log.

Everything here is opt-in and additive; this is a library capability inside your process, never
a hosted attestation registry, a solvency auditor, or a trusted third party.
"""

from __future__ import annotations

from vincio import (
    ContextApp,
    InclusionProof,
    attest_liabilities,
    check_completeness,
    prove_solvency,
)
from vincio.core.errors import SettlementError
from vincio.providers import MockProvider


def main() -> None:
    # An auditor org attests a vendor's obligations; a globex creditor checks completeness.
    auditor = ContextApp(name="auditor", provider=MockProvider(default_text="ok"))
    auditor.use_settlement_book()
    globex = ContextApp(name="globex", provider=MockProvider(default_text="ok"))
    globex.use_settlement_book()

    # The auditor attests the vendor owes $60 to acme — but quietly omits the $40 owed to globex.
    owed = auditor.attest_liabilities("vendor", {"acme": 60.0})

    # 1. The attestation commits its line items into a Merkle root; acme gets an inclusion proof
    #    that its claim is a leaf of that root, verifiable against the signed attestation alone.
    acme_proof = auditor.inclusion_proof(owed, "acme")
    print(
        f"1. Attestation root {owed.liabilities_root[:12]}…; acme's inclusion proof verifies: "
        f"{acme_proof.verify(owed, auditor.contract_signer).valid}. globex, omitted, has no leaf "
        f"to prove — so it can detect the omission."
    )

    # 2. globex folds its own $40 claim against the attestation. Because the attestation never
    #    listed it, the completeness check pinpoints the omission and raises the completed total.
    check = globex.check_completeness(owed, {"globex": 40.0})
    breach = check.breaches[0]
    print(
        f"2. Completeness check ({check.status}): attested ${check.attested_usd:,.2f}, completed "
        f"${check.completed_usd:,.2f}. Omitted creditor {breach.creditor!r} can prove "
        f"${breach.understatement_usd:,.2f} owed (omitted={breach.omitted})."
    )

    # 3. The vendor proves $80 of reserves. On the attestor's $60 figure it looks solvent ($20
    #    free); folding the completeness check bounds the margin by the completed $100 — insolvent.
    reserves = auditor.attest_custody("vendor", {"omnibus": 80.0})
    naive = prove_solvency(reserves, owed)
    complete = prove_solvency(reserves, owed, completeness=check)
    print(
        f"3. On the attestor's figure: ${naive.reserves_usd:,.2f} − ${naive.liabilities_usd:,.2f} "
        f"= ${naive.margin_usd:,.2f} (solvent={naive.solvent}). Completeness-adjusted: "
        f"${complete.reserves_usd:,.2f} − ${complete.liabilities_usd:,.2f} = "
        f"${complete.margin_usd:,.2f} (solvent={complete.solvent}); the hidden $40 tips it into a "
        f"${complete.breach.shortfall_usd:,.2f} shortfall."
    )

    # 4. The proofs read only signed, content-bound artifacts. A tampered inclusion leaf, a
    #    completeness figure that under-states the omission, and a completeness check for a
    #    different attestation are each caught from the bytes alone.
    tampered_leaf = InclusionProof.from_wire(acme_proof.to_wire())
    tampered_leaf.amount_usd = 9_999.0  # claim a bigger debt than the attested leaf
    leaf_refused = not tampered_leaf.verify(owed).valid

    shrunk = check_completeness(owed, {"globex": 40.0})
    shrunk.completed_usd = 70.0  # pretend globex is owed only $10 more, not $40 ...
    shrunk.seal()  # ... and re-seal; the completed total no longer re-derives
    figure_refused = not shrunk.verify().completeness_sound

    other = attest_liabilities("vendor", {"acme": 60.0, "globex": 40.0})  # a different attestation
    wrong = check_completeness(other, {"globex": 50.0})
    try:
        prove_solvency(reserves, owed, completeness=wrong)
        wrong_refused = False
    except SettlementError:
        wrong_refused = True
    print(
        f"4. Tampered inclusion leaf refused: {leaf_refused}; under-stated completed total "
        f"caught: {figure_refused}; wrong-attestation completeness refused: {wrong_refused}; "
        f"{len(globex.audit.query(action='liability_completeness'))} completeness entr(y/ies) on "
        f"globex's chain, books intact={globex.settlement_book.verify(globex.contract_signer).intact}."
    )


if __name__ == "__main__":
    main()
