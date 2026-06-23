"""Cross-org liability history consistency & snapshot monotonicity.

Non-equivocation catches a counterparty signing *different* liability roots for the **same**
instant. But it is scoped to one `as_of`: a counterparty can still issue a *later* snapshot that
quietly **drops** a past obligation — a debt committed at `T` simply absent from the root it signs
at `T'` — and each snapshot is internally sound, so nothing yet ties one attestation to its
predecessor. Equivocation is conflict *across creditors*; this is consistency *across time*. This
example adds the hash-linked history and the monotone-consistency check that catches it.

Four steps, all offline and deterministic:

  1. The vendor's auditor issues three linked liability snapshots — each `attest_liabilities(...,
     prior=...)` commits to its predecessor's root, so the snapshots form a hash-linked sequence a
     creditor can walk, each `as_of` strictly succeeding the last.
  2. `check_history_consistency` walks the snapshots and confirms acme's $100 obligation **persists**
     across them — the history is monotone, the chain contiguous.
  3. The vendor then signs a later snapshot that drops acme to $30 with no settlement behind it. The
     walk pinpoints the unexplained $70 as a `MonotonicityBreach` and dings the vendor's reputation —
     a debt cannot silently vanish between snapshots.
  4. A signed, creditor-issued `Discharge` (acme releasing $70 it was paid) legitimately explains the
     drop, so the same later snapshot is monotone again — while a forged or out-of-window release
     does not, and the proof verifies from the bytes alone.

Everything here is opt-in and additive; this is a library capability inside your process, never a
hosted transparency log, a settlement registry, or a trusted third party.
"""

from __future__ import annotations

from datetime import UTC, datetime

from vincio import (
    ContextApp,
    HistoryConsistencyProof,
    attest_liabilities,
    check_history_consistency,
    discharge_liability,
)
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner


def main() -> None:
    # The cross-party signing convention: every party signs with the shared fabric verification
    # secret, distinguished only by key_id (its identity), so one verifier checks the attestor's and
    # the creditors' signatures alike.
    auditor = HMACSigner("fabric-secret", key_id="auditor")
    acme = HMACSigner("fabric-secret", key_id="acme")
    t1 = datetime(2026, 1, 1, tzinfo=UTC)
    t2 = datetime(2026, 2, 1, tzinfo=UTC)
    t3 = datetime(2026, 3, 1, tzinfo=UTC)

    # 1. Three linked snapshots — each commits to its predecessor's root, forming a hash-linked
    #    history. acme is owed $100 throughout; globex $40.
    s1 = attest_liabilities("vendor", {"acme": 100.0, "globex": 40.0}, attestor="auditor", as_of=t1)
    s1.sign(auditor)
    s2 = attest_liabilities(
        "vendor", {"acme": 100.0, "globex": 40.0}, attestor="auditor", as_of=t2, prior=s1
    ).sign(auditor)
    s3 = attest_liabilities(
        "vendor", {"acme": 100.0, "globex": 40.0}, attestor="auditor", as_of=t3, prior=s2
    ).sign(auditor)
    print(
        f"1. Three snapshots linked: s2→s1 {s2.prior_hash[:12]}…, s3→s2 {s3.prior_hash[:12]}… — "
        f"each as_of strictly succeeds the last."
    )

    # 2. The history is monotone and the chain contiguous: acme's $100 persisted across snapshots.
    report = check_history_consistency([s1, s2, s3], verifier=auditor)
    proof = report.proofs[0]
    print(
        f"2. History walk: consistent={report.consistent}, chain_linked={proof.chain_linked} over "
        f"{proof.snapshot_count} snapshot(s) — acme's obligation persisted."
    )

    # 3. The vendor signs a later snapshot dropping acme to $30 with nothing behind it. Run it
    #    through a verifying app so the inconsistent history lands on the audit chain and dings
    #    reputation.
    dropped = attest_liabilities(
        "vendor", {"acme": 30.0, "globex": 40.0}, attestor="auditor", as_of=t3, prior=s2
    ).sign(auditor)
    app = ContextApp(name="auditor", provider=MockProvider(default_text="ok"))
    app.use_settlement_book(owner="auditor")
    app.use_reputation_ledger()
    bad = app.check_history_consistency([s1, s2, dropped], verify_with=auditor)
    breach = bad.proofs[0].breaches[0]
    print(
        f"3. Silent drop: consistent={bad.consistent}; {breach.creditor} fell "
        f"${breach.prior_usd:,.0f}→${breach.next_usd:,.0f}, unexplained ${breach.unexplained_usd:,.0f}"
        f". vendor's reputation weight is now {app.reputation_ledger.weight('vendor'):.3f} (< 1.0)."
    )

    # 4. A signed, creditor-issued discharge legitimately explains the drop; a forged or
    #    out-of-window one does not. The proof reads only signed, content-bound artifacts.
    settled = discharge_liability("vendor", "acme", 70.0, as_of=t3).sign(acme)  # acme releases $70
    forged = discharge_liability("vendor", "acme", 70.0, as_of=t3).sign(
        HMACSigner("forger-secret", key_id="acme"),
        party="acme",  # not the fabric secret
    )
    ok = check_history_consistency([s1, s2, dropped], discharges=[settled], verifier=auditor)
    still_bad = check_history_consistency([s1, s2, dropped], discharges=[forged], verifier=auditor)
    roundtrips = HistoryConsistencyProof.from_wire(ok.proofs[0].to_wire()).verify(auditor).valid
    print(
        f"4. Discharge explains the drop: consistent={ok.consistent} (1 discharge embedded); forged "
        f"release ignored: consistent={still_bad.consistent}; proof verifies after a wire roundtrip: "
        f"{roundtrips}; {len(app.audit.query(action='liability_history'))} history entr(y/ies) on "
        f"the chain, intact={app.audit.verify_chain()}."
    )


if __name__ == "__main__":
    main()
