# Formal verification of governance invariants

Vincio *enforces* its governance invariants at runtime: residency refuses an
out-of-region egress, provable erasure binds a signed proof to the removed-id set,
the budget caps spend, and the injection-containment gate stops an untrusted-tainted
argument reaching a side-effecting tool without a user-minted capability. Each
decision lands on the signed audit chain. What that leaves implicit is a question an
auditor actually asks: *do those controls hold for every input, or only the ones a
run happened to exercise?*

`app.verify_governance()` answers it with a **proof, not an observation**. A
deterministic, in-process verifier checks each governance invariant across its
*whole* bounded, typed state space, ahead of any run — so a `held=True` verdict is a
statement about every reachable state, not a sample. A violation yields a concrete,
minimal counterexample. The verdict is a content-hashed artifact on the hash-chained
audit log, computed in-process with no external prover service.

Everything here is opt-in and additive — without calling `verify_governance`, a run
behaves exactly as before.

## Why bounded model checking

Each governance control is a *deterministic, pure* decision function over a small,
well-typed state: a trust label, a capability's presence, a provider region, an
accrued budget, a removed-id set. Those alphabets are finite and small. So the
verifier does not sample or fuzz — it **enumerates the entire state space** and
checks the property at every point. Over the modeled domain the check is sound and
complete: if it holds, it is proven; if it fails, the failing state is exhibited.

An `Invariant` pairs a formal *specification* (the property the control must satisfy)
with the variables it quantifies over. Crucially, the predicate calls the **same
decision functions the runtime uses** — the containment gate is
`vincio.security.requires_authority`, the erasure binding is
`verify_erasure_proof` — so verifying the invariant verifies the shipped machinery,
not a re-implementation of it.

## The four invariants

| Invariant | Specification | Bound to |
|---|---|---|
| **Containment** | `untrusted ⇒ no unapproved capability`: an untrusted-tainted argument never reaches a side-effecting tool without a user-minted capability or an approval. | `requires_authority` (the `DualPlaneExecutor` gate) vs `ContainmentEvent.is_escalation` |
| **Residency** | An enforced residency policy admits egress only to a provider region whose jurisdiction is in the allowed set; an unknown region is refused. | `ResidencyPolicy.check` |
| **Budget** | A budget is a hard cap: an admitted run keeps the *projected* total under the limit, and a scope at its limit refuses every further run. | `within_budget` (the dollar / energy / carbon cap predicate) |
| **Erasure** | An erasure proof verifies if and only if its recorded removed-id set is intact; any added, dropped, or swapped id breaks verification. | `build_erasure_proof` / `verify_erasure_proof` |

## Verify in one call

```python
from vincio import ContextApp
from vincio.providers import MockProvider

app = ContextApp(name="app", provider=MockProvider())

report = app.verify_governance()
assert report.held                       # all four invariants proven
print(report.content_sha256)             # reproducible, content-hashed verdict

for result in report.results:
    print(result.category, result.held, result.states_checked, "/", result.domain_size)
    # containment True 48 / 48   <- checked at every point, not sampled
```

`verify_governance` reflects **this app's** posture: the residency invariant is built
from the app's configured `deny_on_unknown` setting, so an app that turned off
fail-closed residency is caught.

The verdict lands on the audit chain as a `governance_verification` decision
(`allow` when it holds, `deny` otherwise):

```python
entry = next(e for e in app.audit.entries if e.action == "governance_verification")
assert entry.decision == "allow"
assert entry.details["content_sha256"] == report.content_sha256
assert app.audit.verify_chain()
```

## A counterexample, not just a verdict

A failed property returns the concrete, delta-minimized state that violates it — the
input, the labels, the capability gap — so a governance regression is debuggable.
Here is the verifier catching a fail-open residency posture:

```python
from vincio.governance.verification import GovernanceVerifier, residency_invariant

report = GovernanceVerifier([residency_invariant(deny_on_unknown=False)]).verify(record=False)
assert not report.held
print(report.counterexamples[0].render())
# [residency_in_jurisdiction_egress] egress to region None (jurisdiction None) was
# admitted under an allowed set of 'eu' it is not part of | state: allowed='eu', region=None
```

The counterexample is **minimized**: each variable is relaxed back toward its benign
default while the violation persists, so the reported witness is the simplest one the
search exposes — no incidental noise.

The same machinery catches a real implementation bug. A budget cap that checks only
what is already spent (ignoring the projection) admits an over-budget run:

```python
from vincio.governance.verification import budget_invariant

weak = budget_invariant(admits=lambda spent, projected, limit: spent < limit)
report = GovernanceVerifier([weak]).verify(record=False)
assert not report.held
print(report.counterexamples[0].render())
# [budget_hard_cap_never_overspends] a run was admitted at spent=0 + projected=50
# = 50, which reaches or exceeds the limit 1 | state: limit=1.0, projected=50.0, spent=0.0
```

## Raise instead of return

For a CI gate or a startup assertion, raise on a violation:

```python
from vincio.core.errors import GovernanceVerificationError

try:
    app.verify_governance(raise_on_violation=True)
except GovernanceVerificationError as exc:
    for cx in exc.counterexamples:
        print(cx.render())
```

## Reproducible and tamper-evident

`VerificationReport.content_sha256` binds the verdict to the per-invariant results
(and any counterexamples) and excludes the timestamp, so two passes over the same
invariants produce the same digest. `report.verify()` recomputes it; editing a
recorded verdict breaks the binding:

```python
report = app.verify_governance(record=False)
assert report.verify()
report.results[0].held = not report.results[0].held
assert not report.verify()              # content binding catches the edit
```

## Custom invariants

The verifier is not limited to the four built-ins. An `Invariant` is a statement, a
tuple of `StateVariable`s, and a predicate over an assignment; pass any list to
`verify_governance` or `GovernanceVerifier`:

```python
from vincio.governance.verification import Invariant, StateVariable, GovernanceVerifier

my_invariant = Invariant(
    id="tenant_never_reads_other",
    statement="A tenant scope never admits another tenant's rows.",
    category="isolation",
    variables=(
        StateVariable("reader", ("acme", "globex")),
        StateVariable("owner", ("acme", "globex", "shared")),
    ),
    predicate=lambda s: s["owner"] in (s["reader"], "shared"),  # bind to your real check
)
report = GovernanceVerifier([my_invariant]).verify(record=False)
```

Order each variable's `values` from benign (index 0) to adversarial so
counterexample minimization produces the cleanest witness.

## What it does not do

The verifier proves the **modeled** properties over their **bounded** domains. The
domains are the controls' real, finite alphabets (three trust labels, the side-effect
classes, representative regions and budget points), so the proof is complete *for the
control as modeled* — it is not a whole-program proof of the Python implementation,
and it does not replace the runtime guards or the adversarial ContainmentBench
corpus. It is the rung beside them: a property checked by construction, ahead of any
run, with a debuggable witness when it fails.
