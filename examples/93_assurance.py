"""Continuous assurance cases & production certification.

Vincio already *produces* the evidence a production AI system is judged on — eval
gates, the governance verifier, reasoning certificates, the signed audit chain, and
SBOM / SLSA provenance. This example shows the capstone that ties them together:
assembling that evidence into one structured, machine-checkable argument that the
system is fit for purpose, and keeping it continuously valid as the system changes.
Everything is deterministic and offline (no model call).

  1. **A structured assurance case** — `app.assurance_case(...)` builds an argument
     tree: a top claim decomposed into sub-claims, each discharged by `Evidence` the
     platform already emits, bound by hash so the whole case verifies offline.
  2. **Continuous assurance** — `case.check()` re-derives the verdict from the bytes,
     pinpointing a missing, stale, or falsified piece, and `assurance_regression_gate`
     turns a falsified claim into a build failure.
  3. **Incident learning & certification** — a signed `Incident` makes the case demand
     fresh evidence before it re-validates, and `app.certify(...)` emits a portable,
     offline-verifiable certification report an auditor checks from the bytes.

This is a library capability inside your process — never a hosted certification service.
"""

from __future__ import annotations

from vincio import (
    Claim,
    ContextApp,
    Evidence,
    Incident,
    assurance_regression_gate,
)
from vincio.providers import MockProvider


def main() -> None:
    app = ContextApp(name="support-assistant", provider=MockProvider(default_text="ok"))

    # Build the assurance case from the evidence the platform emits for the current
    # state of the system. Each re-check on a change rebuilds the case the same way,
    # so the argument always reflects live evidence — never a stale snapshot.
    def build_case(answer: str):
        return app.assurance_case(
            "The support assistant is fit for production",
            context="EU deployment, tier-1 traffic",
            subclaims=[
                Claim(
                    id="governance",
                    statement="Governance controls (containment, residency, budget) hold",
                    evidence=[Evidence.from_governance(app.verify_governance())],
                ),
                Claim(
                    id="quality",
                    statement="Answers meet the quality bar",
                    evidence=[Evidence.from_gate(True, label="quality gate")],
                ),
                Claim(
                    id="reasoning",
                    statement="Numeric answers are certified",
                    evidence=[Evidence.from_certificate(app.verify_reasoning(answer).certificate)],
                ),
                Claim(
                    id="provenance",
                    statement="Decisions are attested on the audit chain",
                    evidence=[Evidence.from_audit(app.audit)],
                ),
            ],
        )

    case = build_case("2 + 2 = 4")

    print("== Assurance case ==")
    baseline = case.check()
    print(f"statement: {case.goal.statement}")
    print(f"holds={baseline.holds}  case.verify()={case.verify()}  signed={bool(case.signature)}")
    for status in baseline.root.children:
        print(f"  - {status.id}: holds={status.holds} discharged_by={status.discharged_by}")

    print("\n== Continuous assurance: a regression is caught ==")
    # A change ships that falsifies the reasoning claim (the answer no longer
    # certifies). Rebuild the case from the new evidence and gate against the prior.
    changed = build_case("2 + 2 = 5")  # a refuted answer
    after = changed.check()
    passed, reason = assurance_regression_gate(baseline, after)
    print(f"holds={after.holds}  falsified={after.falsified}")
    print(f"regression gate: passed={passed}  reason={reason}")

    print("\n== Incident & safety-case learning ==")
    # File a production incident against the reasoning claim on the healthy case.
    incident = Incident(
        id="inc-2026-06-001",
        description="A numeric answer regressed in production",
        falsified_claim="reasoning",
        required_evidence=["eval_gate"],
    ).seal()
    case.learn_from(incident)
    print(f"after learning, holds={case.check().holds} (the case now demands a fix proof)")
    remediation = case.goal.find("reasoning").subclaims[-1]
    case.discharge(remediation.id, Evidence.from_gate(True, label="post-fix gate"))
    print(f"after discharging the remediation, holds={case.check().holds}")

    print("\n== Certification ==")
    report = app.certify(case)
    print(f"certified={report.certified}  report.verify()={report.verify()}")
    print(f"residual_risks={report.residual_risks}")
    print(
        f"provenance: vincio {report.provenance.get('vincio_version')}, "
        f"sbom components={len(report.provenance.get('sbom', {}).get('components', []))}"
    )
    print(f"\nAudit chain: {len(app.audit.entries)} entries, verifies={app.audit.verify_chain()}")


if __name__ == "__main__":
    main()
