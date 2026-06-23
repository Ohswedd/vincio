"""Cross-org insolvency resolution & liability seniority waterfall.

A `SolvencyProof` *flags* an insolvency when a counterparty's proven liabilities exceed its proven
reserves — but when the reserves genuinely cannot cover every obligation, nothing yet says **which**
creditors the available capital pays, and in what order. Today an insolvency is flagged, not
resolved: every creditor is left to assume it is made whole. The rehypothecation guard already
apportions a scarce stake across beneficiaries pari-passu; the liability side needs the same, plus
the **seniority** real obligations carry. This example adds the signed seniority schedule and the
insolvency waterfall that resolves it into who-gets-what.

Four steps, all offline and deterministic:

  1. The vendor proves $60 in reserves and its auditor attests $100 owed — $50 to a senior `bank`,
     $30 to `acme`, $20 to `globex`. `prove_solvency` flags the $40 shortfall, but says nothing
     about who absorbs it.
  2. `build_seniority_schedule` ranks the obligations into priority tranches the bank signs off on —
     the bank senior (rank 0), acme and globex junior (rank 1) — a signed, non-repudiable artifact.
  3. `resolve_insolvency` distributes the $60 by seniority then pari-passu within a tranche: the
     bank is paid in full, and the remaining $10 splits across the $50 junior tranche at 20¢ on the
     dollar. The insolvency is *resolved* — each creditor's bounded recovery and the shortfall it
     bears are pinpointed, and the vendor's reputation is dinged for not making its creditors whole.
  4. The resolution verifies from the bytes alone — an over-stated recovery or a re-ordered tranche
     is refused — and binds the seniority schedule by hash, so a creditor cannot be quietly re-ranked
     away from the order it agreed to.

Everything here is opt-in and additive; this is a library capability inside your process, never a
hosted receiver, a bankruptcy court, or a trusted third party.
"""

from __future__ import annotations

from vincio import (
    ContextApp,
    InsolvencyResolution,
    attest_custody,
    attest_liabilities,
    build_seniority_schedule,
    prove_solvency,
    resolve_insolvency,
)
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner


def main() -> None:
    # The cross-party signing convention: every party signs with the shared fabric verification
    # secret, distinguished only by key_id (its identity), so one verifier checks the custodian's,
    # the auditor's, and the bank's signatures alike.
    custodian = HMACSigner("fabric-secret", key_id="custodian")
    auditor = HMACSigner("fabric-secret", key_id="auditor")
    bank = HMACSigner("fabric-secret", key_id="bank")

    # 1. The vendor proves $60 in reserves and its auditor attests $100 owed. prove_solvency flags
    #    the $40 shortfall — but cannot say which creditors the $60 pays, or in what order.
    reserves = attest_custody("vendor", {"omnibus": 60.0}, custodian="custodian").sign(custodian)
    owed = attest_liabilities(
        "vendor", {"bank": 50.0, "acme": 30.0, "globex": 20.0}, attestor="auditor"
    ).sign(auditor)
    proof = prove_solvency(reserves, owed, verifier=auditor)
    print(
        f"1. Solvency: reserves ${proof.reserves_usd:,.0f} − liabilities ${proof.liabilities_usd:,.0f}"
        f" = ${proof.margin_usd:,.0f} ({proof.status}). The $40 shortfall is flagged, not resolved."
    )

    # 2. The obligations are ranked into signed priority tranches — the bank senior (rank 0), acme
    #    and globex junior (rank 1). Position is priority; the bank signs the inter-creditor order.
    schedule = build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]]).sign(
        bank, party="bank"
    )
    print(
        f"2. Seniority schedule: rank 0 = bank (senior), rank 1 = acme, globex (junior) — signed by "
        f"{', '.join(schedule.signed_by)}, verifies={schedule.verify(bank).valid}."
    )

    # 3. Distribute the $60 by seniority then pari-passu. Run it through a verifying app so the
    #    resolution lands on the audit chain and the unmade-whole vendor's reputation is dinged.
    app = ContextApp(name="auditor", provider=MockProvider(default_text="ok"))
    app.use_settlement_book(owner="auditor")
    app.use_reputation_ledger()
    resolution = app.resolve_insolvency(reserves, owed, schedule, verify_with=auditor)
    print(
        f"3. Waterfall ({resolution.status}): distributed ${resolution.distributed_usd:,.0f} of "
        f"${resolution.liabilities_usd:,.0f} owed; {resolution.shortfall_bearers} bear "
        f"${resolution.shortfall_usd:,.0f}."
    )
    for r in sorted(resolution.recoveries, key=lambda r: (r.rank, r.creditor)):
        mark = "made whole" if r.made_whole else f"short ${r.shortfall_usd:,.0f}"
        print(
            f"     rank {r.rank} {r.creditor}: ${r.recovery_usd:,.0f} of ${r.claim_usd:,.0f} "
            f"({r.recovery_rate:.0%}) — {mark}"
        )
    print(f"     vendor's reputation weight is now {app.reputation_ledger.weight('vendor'):.3f}.")

    # 4. The resolution verifies from the bytes alone and binds the schedule by hash. Folded and
    #    signed with the shared fabric secret, one verifier checks the resolution and its embedded
    #    schedule alike. An over-stated recovery is refused even after re-sealing; a re-ranked
    #    schedule does not bind.
    clean = resolve_insolvency(reserves, owed, schedule, verifier=auditor).sign(
        auditor, party="auditor"
    )
    tampered = InsolvencyResolution.from_wire(clean.to_wire())
    tampered.recoveries[0].recovery_usd += 100.0
    tampered.seal()  # recompute the hash to match the lie
    reordered = build_seniority_schedule("vendor", [["bank", "acme"], ["globex"]])
    roundtrips = InsolvencyResolution.from_wire(clean.to_wire()).verify(auditor, schedule).valid
    print(
        f"4. Verifies after a wire roundtrip: {roundtrips}; over-stated recovery refused: "
        f"{not tampered.verify().distribution_sound}; bound to its schedule: "
        f"{clean.verify(auditor, schedule=schedule).valid}; a re-ranked schedule does not "
        f"bind: {not clean.verify(auditor, schedule=reordered).schedule_bound}; "
        f"{len(app.audit.query(action='insolvency_resolution'))} resolution(s) on the chain, "
        f"intact={app.audit.verify_chain()}."
    )


if __name__ == "__main__":
    main()
