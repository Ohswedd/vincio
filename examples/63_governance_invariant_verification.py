"""Formal verification of governance invariants — proof, not just enforcement.

The platform already *enforces* its governance invariants at runtime — residency
refuses an out-of-region egress, provable erasure binds a signed proof to the
removed-id set, the budget caps spend, and the injection-containment gate stops an
untrusted-tainted argument reaching a side-effecting tool without a user-minted
capability. This example adds the rung beside that: a **machine-checkable proof
that those invariants hold across the whole input space, ahead of any run** —
checked by construction by a deterministic in-process verifier, not observed after
the fact.

Five steps, all offline and deterministic:

  1. Verify: prove all four governance invariants over their bounded, typed state
     spaces in one call; the verdict is a content-hashed report.
  2. A proof, not a sample: a holding invariant was checked at every point of its
     domain — `states_checked == domain_size`.
  3. Counterexample, not just a verdict: a fail-open residency posture yields the
     concrete, minimal state that violates in-jurisdiction egress.
  4. The verifier catches a real bug: a budget cap that ignores the projection
     admits an over-budget run — the verifier exhibits the witness.
  5. Auditable & offline: the verdict is a deterministic artifact on the
     hash-chained audit log, computed in-process with no external prover.

Everything here is opt-in and additive; `app.verify_governance()` reads the app's
own governance posture and never touches the network.
"""

from __future__ import annotations

import asyncio

from vincio import ContextApp, GovernanceVerifier, VincioConfig
from vincio.governance.verification import budget_invariant, residency_invariant
from vincio.providers import MockProvider


def _app(**governance) -> ContextApp:
    config = VincioConfig()
    config.observability.exporter = "memory"
    for key, value in governance.items():
        setattr(config.governance, key, value)
    return ContextApp(name="verify_demo", provider=MockProvider(), config=config)


async def main() -> None:
    print("Formal verification of governance invariants — proof, not just enforcement\n")

    # 1. Prove all four invariants in one call.
    print("1. Verify — prove containment, residency, budget, and erasure at once")
    app = _app()
    report = app.verify_governance()
    print(f"   held = {report.held}  |  digest = {report.content_sha256[:16]}…")
    for result in report.results:
        print(
            f"     {result.category:12} held={result.held}  "
            f"checked {result.states_checked}/{result.domain_size} states"
        )

    # 2. A holding invariant is a proof over its whole bounded domain, not a sample.
    print("\n2. A proof, not a sample — every state in the domain was checked")
    total = sum(r.states_checked for r in report.results)
    print(f"   {total} states checked across the four invariants; verdict reproduces = {report.verify()}")

    # 3. Counterexample, not just a verdict: a fail-open residency posture is caught.
    print("\n3. Counterexample — a fail-open residency posture is caught, with a witness")
    fail_open = GovernanceVerifier([residency_invariant(deny_on_unknown=False)]).verify(record=False)
    print(f"   held = {fail_open.held}")
    print(f"   {fail_open.counterexamples[0].render()}")

    # 4. The verifier catches a real bug: a budget cap that ignores the projection.
    print("\n4. Catches a real bug — a cap that checks only spend, not the projection")
    weak = GovernanceVerifier(
        [budget_invariant(admits=lambda spent, projected, limit: spent < limit)]
    ).verify(record=False)
    print(f"   held = {weak.held}")
    print(f"   {weak.counterexamples[0].render()}")

    # 5. Auditable & offline — the verdict is on the verifiable chain.
    print("\n5. Auditable — the verdict is a deterministic artifact on the chain")
    entries = [e for e in app.audit.entries if e.action == "governance_verification"]
    print(
        f"   {len(entries)} verification verdict(s) on the chain; "
        f"decision = {entries[-1].decision}; chain verifies = {app.audit.verify_chain()}"
    )

    # The same posture, misconfigured, is flagged through the app surface too.
    print("\n   (an app that turned off fail-closed residency is flagged through the app:)")
    misconfigured = _app(allowed_regions=["eu"], deny_on_unknown_region=False)
    bad = misconfigured.verify_governance()
    residency = next(r for r in bad.results if r.category == "residency")
    print(f"     residency held = {residency.held}  ->  {residency.counterexample.render()}")

    print("\nGovernance held by construction — and a regression is a debuggable witness.")


if __name__ == "__main__":
    asyncio.run(main())
