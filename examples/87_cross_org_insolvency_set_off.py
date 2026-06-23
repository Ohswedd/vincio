"""Cross-org insolvency set-off & close-out netting.

The insolvency waterfall distributes a poster's reserves across the creditors it owes — but a
creditor is often *also* a debtor of the same counterparty, and the waterfall pays it on its
**gross** claim while it still owes the insolvent estate the other side. Real insolvency law resolves
this first with **set-off** (close-out netting): mutual obligations between the same two parties
collapse to a single net claim *before* any distribution, so a creditor that owes more than it is
owed is not paid at all, and one owed more recovers only its *net* position. This example adds the
signed set-off statement and the close-out pass that folds it into the waterfall.

Four steps, all offline and deterministic:

  1. The vendor's auditor attests $100 owed — $50 to a senior `bank`, $30 to `acme`, $20 to
     `globex` — and the vendor proves $60 in reserves. But `acme` owes the vendor $24 back across a
     settled contract, and `globex` owes it $25 — more than the $20 it is owed.
  2. Each pair signs a `SetOffStatement` of the obligations running *both ways*: acme's $30 claim
     nets to $6, and globex (in debit) nets to $0 — a mutually-agreed, content-bound close-out, not
     one side's assertion.
  3. `resolve_insolvency(set_off=…)` reduces each creditor to its net claim *before* distributing:
     the distributable estate shrinks from $100 to $56, globex recovers nothing, and the $60
     reserves now make every remaining creditor whole — the insolvency is netted away.
  4. The netted resolution verifies from the bytes alone — an inflated set-off is refused — binds
     the statements by hash, and a one-sided or over-stated close-out is refused at fold time.

Everything here is opt-in and additive; this is a library capability inside your process, never a
hosted clearing house, a bankruptcy court, or a trusted third party.
"""

from __future__ import annotations

from vincio import (
    ContextApp,
    InsolvencyResolution,
    SetOffStatement,
    attest_custody,
    attest_liabilities,
    build_set_off_statement,
    resolve_insolvency,
    set_off_from_records,
)
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.settlement.record import SettlementRecord


def main() -> None:
    # The cross-party signing convention: every party signs with the shared fabric verification
    # secret, distinguished only by key_id (its identity), so one verifier checks every signature.
    secret = "fabric-secret"
    custodian = HMACSigner(secret, key_id="custodian")
    auditor = HMACSigner(secret, key_id="auditor")
    vendor = HMACSigner(secret, key_id="vendor")
    acme = HMACSigner(secret, key_id="acme")
    globex = HMACSigner(secret, key_id="globex")
    verifier = HMACSigner(secret, key_id="any")

    # 1. The vendor proves $60 in reserves and its auditor attests $100 owed. acme owes the vendor
    #    $24 back (from a settled contract), and globex owes $25 — more than the $20 it is owed.
    reserves = attest_custody("vendor", {"omnibus": 60.0}, custodian="custodian").sign(custodian)
    owed = attest_liabilities(
        "vendor", {"bank": 50.0, "acme": 30.0, "globex": 20.0}, attestor="auditor"
    ).sign(auditor)
    acme_owes_back = SettlementRecord(
        contract_id="c-acme", buyer="acme", seller="vendor", amount_owed_usd=24.0
    )
    acme_owes_back.seal()
    print(
        f"1. Gross: vendor owes ${owed.liabilities_usd:,.0f} (bank $50, acme $30, globex $20), "
        f"holds ${reserves.reserves_usd:,.0f}. acme owes $24 back, globex owes $25 back."
    )

    # 2. Each pair co-signs a set-off statement of the obligations running both ways. acme's is
    #    derived straight from the existing artifacts (the attestation + the settled record); globex
    #    is in debit (owes more than it is owed), so it nets to zero.
    acme_set_off = set_off_from_records("vendor", "acme", owed, [acme_owes_back])
    acme_set_off.sign(vendor, party="vendor").sign(acme, party="acme")
    globex_set_off = build_set_off_statement("vendor", "globex", 20.0, 25.0)
    globex_set_off.sign(vendor, party="vendor").sign(globex, party="globex")
    print(
        f"2. Set-off: acme $30 owed − $24 owing → net ${acme_set_off.poster_net_claim_usd:,.0f} "
        f"(mutual={acme_set_off.mutual}); globex $20 owed − $25 owing → net "
        f"${globex_set_off.poster_net_claim_usd:,.0f} (in debit={globex_set_off.creditor_in_debit})."
    )

    # 3. Close-out net before the waterfall. Run it through a verifying app so the resolution lands
    #    on the audit chain. The distributable estate shrinks from $100 to $56, and the $60 reserves
    #    now make every remaining creditor whole — the insolvency is netted away.
    app = ContextApp(name="auditor", provider=MockProvider(default_text="ok"))
    app.use_settlement_book(owner="auditor")
    app.use_reputation_ledger()
    resolution = app.resolve_insolvency(reserves, owed, set_off=[acme_set_off, globex_set_off])
    print(
        f"3. Close-out ({resolution.status}): gross ${resolution.gross_liabilities_usd:,.0f} − "
        f"set-off ${resolution.set_off_usd:,.0f} = net ${resolution.liabilities_usd:,.0f}; "
        f"distributed ${resolution.distributed_usd:,.0f} of ${resolution.reserves_usd:,.0f}."
    )
    for r in sorted(resolution.recoveries, key=lambda r: r.creditor):
        note = f" (set off ${r.set_off_usd:,.0f} of ${r.gross_claim_usd:,.0f})" if r.set_off else ""
        mark = "made whole" if r.made_whole else f"short ${r.shortfall_usd:,.0f}"
        print(f"     {r.creditor}: ${r.recovery_usd:,.0f} of ${r.claim_usd:,.0f}{note} — {mark}")

    # 4. The netted resolution verifies from the bytes alone, binds the set-off statements by hash,
    #    and refuses an inflated set-off, a one-sided close-out, and an over-stated one at fold time.
    inflated = InsolvencyResolution.from_wire(resolution.to_wire())
    inflated.recovery_of("acme").set_off_usd = 28.0  # claim more was netted than really was
    inflated.seal()
    one_sided = build_set_off_statement("vendor", "acme", 30.0, 24.0)
    one_sided.sign(vendor, party="vendor")  # only the vendor signed
    try:
        resolve_insolvency(reserves, owed, set_off=[one_sided], verifier=verifier)
        one_sided_refused = False
    except Exception:
        one_sided_refused = True
    bound = resolution.verify(verifier, set_off=[acme_set_off, globex_set_off]).set_off_bound
    print(
        f"4. Verifies offline: {resolution.verify().valid}; binds its set-off statements: {bound}; "
        f"inflated set-off refused: {not inflated.verify().distribution_sound}; one-sided close-out "
        f"refused: {one_sided_refused}; "
        f"{len(app.audit.query(action='insolvency_resolution'))} resolution(s) on the chain, "
        f"intact={app.audit.verify_chain()}."
    )

    assert isinstance(acme_set_off, SetOffStatement)


if __name__ == "__main__":
    main()
