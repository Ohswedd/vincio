# Continuous assurance cases & production certification

Vincio already *produces* the evidence a production AI system is judged on, eval and
regression gates, the governance-invariant verifier, reasoning certificates and
runtime monitors, identity and delegation provenance, the signed audit chain, C2PA
media provenance, and SBOM / SLSA build attestations. The `vincio.assurance` module is
the capstone that ties them together: it **assembles that evidence into one
structured, machine-checkable argument** that the system is fit for purpose, and keeps
that argument **continuously valid as the system changes**. That is the assurance-case
discipline (GSN / CAE) the safety and regulatory frontier now demands.

Everything here is opt-in, additive, deterministic, and offline. It composes the
platform's existing verdicts, it does not introduce a new runtime, a hosted prover,
or a network dependency. A machine-checkable artifact, never a slide deck.

> **An argument, not an audit log.** An assurance case is a *tree of claims*, each
> discharged by a verdict the platform already emits and bound by hash. It is distinct
> from the audit chain (a flat, append-only record of decisions): the case argues
> *why* the system is fit, the chain records *what* it did. A claim can be discharged
> by an audit-chain segment, so the two compose.

## The argument tree

A `Claim` is a node: a statement, an optional decomposition into sub-claims, and the
`Evidence` that discharges it. The top `Claim` is the goal, *this app is fit for
purpose X under context Y*, and each leaf rests on evidence the platform emits.

```python
from vincio import Claim, ContextApp, Evidence
from vincio.providers import MockProvider

app = ContextApp(name="assistant", provider=MockProvider(default_text="ok"))

case = app.assurance_case(
    "The support assistant is fit for production",
    context="EU deployment, tier-1 traffic",
    subclaims=[
        Claim(id="governance", statement="Governance controls hold",
              evidence=[Evidence.from_governance(app.verify_governance())]),
        Claim(id="quality", statement="Answers meet the quality bar",
              evidence=[Evidence.from_gate(quality_gate_verdict)]),
        Claim(id="reasoning", statement="Numeric answers are certified",
              evidence=[Evidence.from_certificate(app.verify_reasoning("2 + 2 = 4").certificate)]),
        Claim(id="provenance", statement="The build is attested",
              evidence=[Evidence.from_audit(app.audit)]),
    ],
)

report = case.check()          # re-derive the verdict from the bytes
assert report.holds
assert report.verify()         # the report itself is content-bound
```

Each `Evidence` binder wraps a verdict the platform **already produces**, never
re-implementing the check:

| Binder | Artifact | Supports the claim when… |
|---|---|---|
| `Evidence.from_gate` | a `CanaryVerdict` / eval gate / bool | the gate **passed** |
| `Evidence.from_governance` | a `GovernanceVerifier` report | the report **held** and re-verifies |
| `Evidence.from_certificate` | a reasoning `Certificate` | the certificate is **verified** |
| `Evidence.from_audit` | an `AuditLog` segment | the hash-linked chain **verifies** |
| `Evidence.from_identity` | an identity / delegation verification | the chain is **valid** |
| `Evidence.from_sbom` | an `AIBOM` | the bill of materials is present and intact |
| `Evidence.asserted` | any external verdict | as captured (still bound by hash) |

Every piece is **bound by hash**, so the whole case verifies offline. A flipped
verdict, a tampered support, or an edited argument tree is caught from the bytes; a
missing or stale piece is **pinpointed**, not silently passed.

## Soundness: missing, stale, and falsified evidence

The whole value of an assurance case is that **no claim stands on missing or stale
evidence**. A leaf claim that lists a required evidence kind it does not have is
*undischarged*; a piece of evidence past its freshness horizon is *stale*; a verdict
that no longer supports the claim is *falsified*. Any of these invalidates the claim,
which propagates up the tree, and the failing path is reported.

```python
from datetime import timedelta
from vincio.core.utils import utcnow

# A proof carries a freshness horizon, a stale proof expires.
ev = Evidence.from_governance(app.verify_governance(), horizon_days=30,
                              recorded_at=utcnow() - timedelta(days=40))
assert not ev.holds()                       # intact and supportive, but expired

report = case.check()
report.missing      # ["claim_id:eval_gate", ...]  required but absent
report.stale        # ["claim_id:governance_proof", ...]  past the horizon
report.falsified    # ["claim_id:reasoning_certificate", ...]  verdict flipped
report.failing_claims
```

## Continuous assurance & the regression gate

The case is not a point-in-time audit; it is **re-checked on every change** (a model
swap, a prompt edit, a dependency bump, a new deployment). The same gate machinery
that blocks a quality regression blocks an *assurance* regression:
`assurance_regression_gate` fails the build when a claim that **held** before is no
longer discharged.

```python
from vincio import assurance_regression_gate

before = case.check()
# ... a change to the system; re-gather the evidence and re-check ...
after = case.check()

passed, reason = assurance_regression_gate(before, after)
if not passed:
    raise SystemExit(f"assurance regression: {reason}")   # CI-gated invariant
```

## Incidents & safety-case learning

When a production failure falsifies a claim, a signed `Incident` ties the observed
failure to the sub-claim it broke and the case **learns**: a remediation sub-claim is
added that *demands fresh evidence* before the case can re-validate, closing the loop
from a production incident back into a stronger safety argument.

```python
from vincio import Incident

incident = Incident(
    id="inc-2026-06-001",
    description="A numeric answer regressed in production",
    falsified_claim="reasoning",
    required_evidence=["eval_gate", "reasoning_certificate"],
).seal()

case.learn_from(incident)
assert not case.check().holds          # the case now demands the remediation evidence

# once the fix ships and is proven, discharge the remediation:
remediation = case.goal.find("reasoning").subclaims[-1]
case.discharge(remediation.id, Evidence.from_gate(post_fix_gate))
assert case.check().holds              # the argument is whole again, and stronger
```

## Certification

`app.certify(case)` emits a portable, offline-verifiable `CertificationReport`, the
case, its discharged evidence verdict, the residual risks, and the build provenance
(the `vincio` version and a CycloneDX AI-BOM / SLSA note). A downstream operator or
auditor checks it **from the bytes**:

```python
report = app.certify(case)
assert report.verify()          # recomputes the hash AND re-runs the case's check
assert report.certified         # the case holds
report.to_json()                # hand it to an auditor
```

`CertificationReport.verify()` recomputes the report hash, re-verifies the embedded
case and assurance report, and re-runs the evidence check, so a report claiming
`certified` over a case that does not hold is caught offline, and so is a tamper to
any underlying piece of evidence.

## Gotchas & best practice

- **Evidence binders bind a verdict, they never re-run the check.**
  `Evidence.from_gate`, `from_governance`, `from_certificate`, … wrap a verdict
  the platform already produced — so an `Evidence.asserted(...)` or a hand-passed
  bool is only as trustworthy as its source. Prefer a real platform verdict over
  an asserted one wherever the check exists.
- **A stale proof invalidates a claim even when it is intact and supportive.**
  Freshness is a first-class failure mode: a piece past its `horizon_days` shows
  up in `report.stale` and fails its claim. Set realistic horizons and re-gather
  evidence on each check, don't set a horizon so long it can never expire.
- **Missing ≠ passed.** A leaf that declares a required evidence *kind* it does
  not have is *undischarged* and reported in `report.missing`; the case does not
  silently skip it. Read `report.failing_claims` to see the exact failing path.
- **Re-check on every change.** `assurance_regression_gate(before, after)` is the
  CI invariant — run it on a model swap, a prompt edit, a dependency bump, or a
  new deployment so a claim that *held* can never silently stop holding.
- **`learn_from(incident)` deliberately breaks the case** until remediation ships:
  the added sub-claim *demands fresh evidence*, so `case.check().holds` stays
  `False` until you `discharge` the remediation with a post-fix verdict.
- **`CertificationReport.verify()` re-runs the case check.** A report claiming
  `certified` over a case that no longer holds is caught offline from the bytes —
  so a certificate is only as current as its last honest re-check.

## What it is not

The assurance case is a **library capability inside your process**. It never becomes a
hosted certification authority, a managed control plane, or a network service. The
evidence it binds is the evidence the platform already emits; the verdict it produces
is deterministic, offline, and reproducible from the bytes. With it, the platform is
**production-complete**: every subsystem composes into one continuously-verified safety
argument.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Example: 09_security_governance.py](../../examples/09_security_governance.py)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#governance)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
