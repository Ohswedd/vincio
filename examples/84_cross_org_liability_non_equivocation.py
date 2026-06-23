"""Cross-org liability non-equivocation & root consistency.

A completeness check proves each creditor's claim is *included* in the attested liabilities. But a
counterparty issues its liability attestation **per relationship**: nothing in completeness stops
it presenting *different* liability roots — a smaller total — to different creditors, so each
creditor's `InclusionProof` verifies against the root *it* was shown while the counterparty
equivocates across the set. Completeness catches an omission only when the omitted creditor folds
its own claim; equivocation hides the omission by showing each creditor a root on which its own
claim *is* present. This example adds the cross-creditor check that catches it.

Four steps, all offline and deterministic:

  1. The vendor's auditor signs acme a root over ``{acme: 60}`` and globex a *different* root over
     ``{globex: 40}`` — a smaller total each, for the **same** instant. Each creditor's inclusion
     proof verifies against the root it was shown, so neither alone can tell.
  2. The creditors compare the signed `RootCommitment` each holds — the root and the ``as_of`` the
     attestor signed, **without** the line items — and find a conflict over the exchange.
  3. `check_root_consistency` folds the two conflicting attestations into a non-repudiable
     `EquivocationProof`, pinning the poster, the two signed roots, and the creditor each was
     shown — and dings the equivocating poster on the reputation path.
  4. The comparison and the proof read only signed, content-bound artifacts: a forged conflicting
     root is **refused** with the attestor's verifier and excluded from a scan (so it cannot
     manufacture a false accusation), an honest set showing every creditor the same root is
     consistent, and the check lands on the hash-chained audit log.

Everything here is opt-in and additive; this is a library capability inside your process, never a
hosted transparency log, an attestation registry, or a trusted third party.
"""

from __future__ import annotations

from datetime import UTC, datetime

from vincio import (
    ContextApp,
    EquivocationProof,
    attest_liabilities,
    check_root_consistency,
    prove_equivocation,
)
from vincio.core.errors import SettlementError
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner


def main() -> None:
    # One auditor attests the vendor's liabilities to each creditor; an as-of pins the snapshot.
    auditor = HMACSigner("auditor-key", key_id="auditor")
    as_of = datetime(2026, 1, 1, tzinfo=UTC)

    # 1. The vendor equivocates: it shows acme a root over {acme: 60} and globex a different root
    #    over {globex: 40} — each a smaller total, each for the *same* instant.
    to_acme = attest_liabilities("vendor", {"acme": 60.0}, attestor="auditor", as_of=as_of)
    to_acme.sign(auditor)
    to_globex = attest_liabilities("vendor", {"globex": 40.0}, attestor="auditor", as_of=as_of)
    to_globex.sign(auditor)
    print(
        f"1. acme was shown root {to_acme.liabilities_root[:12]}… (${to_acme.liabilities_usd:,.2f}); "
        f"globex was shown root {to_globex.liabilities_root[:12]}… (${to_globex.liabilities_usd:,.2f})"
        f" — each verifies its own inclusion proof, so neither alone can tell."
    )

    # 2. The creditors compare the signed root commitments — the root and as-of only, never the
    #    other creditor's line items — and find the vendor signed two roots for one instant.
    acme_commitment = to_acme.root_commitment()
    globex_commitment = to_globex.root_commitment()
    conflict = acme_commitment.conflicts_with(globex_commitment)
    print(
        f"2. Commitments compared over the exchange — conflict detected: {conflict}; both signed by "
        f"the attestor: {acme_commitment.verify(auditor).valid and globex_commitment.verify(auditor).valid}"
        f". The commitment leaks no line items: {'acme' not in globex_commitment.model_dump_json()}."
    )

    # 3. The two conflicting attestations fold into a non-repudiable EquivocationProof. Run it
    #    through a verifying app so the equivocation lands on the audit chain and dings reputation.
    app = ContextApp(name="auditor", provider=MockProvider(default_text="ok"))
    app.use_settlement_book(owner="auditor")
    app.use_reputation_ledger()
    report = app.check_root_consistency(
        [("acme", to_acme), ("globex", to_globex)], verify_with=auditor
    )
    proof = report.equivocations[0]
    print(
        f"3. Root-consistency check: consistent={report.consistent}; equivocating posters="
        f"{report.equivocating_posters}. Proof pins ${proof.first.liabilities_usd:,.2f} vs "
        f"${proof.second.liabilities_usd:,.2f} (gap ${proof.liabilities_gap_usd:,.2f}); vendor's "
        f"reputation weight is now {app.reputation_ledger.weight('vendor'):.3f} (< 1.0)."
    )

    # 4. The proof reads only signed, content-bound artifacts. A forged conflicting root (signed
    #    with the wrong key) is refused and excluded from a scan; an honest set showing every
    #    creditor the same root is consistent; and the equivocation is on the audit chain.
    forged = attest_liabilities("vendor", {"zeta": 5.0}, attestor="auditor", as_of=as_of)
    forged.sign(HMACSigner("forger-key", key_id="auditor"))  # not the attestor's key
    try:
        prove_equivocation(to_acme, forged, verifier=auditor)
        forged_refused = False
    except SettlementError:
        forged_refused = True
    forged_scan = check_root_consistency([to_acme, forged], verifier=auditor)
    honest_other = attest_liabilities("vendor", {"acme": 60.0}, attestor="auditor", as_of=as_of)
    honest_other.sign(auditor)
    honest = check_root_consistency([to_acme, honest_other], verifier=auditor)
    roundtrips = EquivocationProof.from_wire(proof.to_wire()).verify(auditor).valid
    print(
        f"4. Forged conflicting root refused: {forged_refused}; excluded from a scan so no false "
        f"accusation: {forged_scan.consistent}; honest same-root set consistent: {honest.consistent}; "
        f"proof verifies after a wire roundtrip: {roundtrips}; "
        f"{len(app.audit.query(action='liability_equivocation'))} equivocation entr(y/ies) on the "
        f"chain, intact={app.audit.verify_chain()}."
    )


if __name__ == "__main__":
    main()
