# Verified reasoning & neuro-symbolic certificates

Vincio scores answers with judges, oracles, and a governance verifier — but those
per-answer signals are *probabilistic*. For the classes of question where it is
possible, an answer can instead carry a **checkable certificate** a deterministic
verifier confirms independently of the model: arithmetic, units, dates, constraint
satisfaction, schema and citation consistency are all checkable. This turns *the
judge thinks it is right* into *here is a proof you can verify* — the output-side
analogue of the governance verifier's machine-checked invariants.

Everything here is opt-in, additive, deterministic, and offline. The deterministic
kernels are dependency-free; optional SMT / CAS backends sit behind
`pip install "vincio[verify]"`.

## Proof-carrying answers

`app.verify_reasoning(answer)` runs a set of deterministic kernels over an answer
and returns a `VerifiedAnswer` whose `Certificate` is **verified** (a kernel
recomputed a claim and it held), **refuted** (a recomputation disagreed — a *proof
the answer is wrong*), or **inapplicable** (no checkable claim of that kind).

```python
from vincio import ContextApp
from vincio.providers import MockProvider

app = ContextApp(name="solver", provider=MockProvider(default_text="ok"))

bad = app.verify_reasoning("The total is 2 + 2 = 5.")
assert not bad.holds and bad.refused          # refuse to emit a refuted answer
print(bad.certificate.render())

good = app.verify_reasoning("We compute 12 * 3 = 36 and 10% of 200 is 20.")
assert good.holds
```

A certificate is **content-bound**: `certificate.verify()` recomputes its hash and
re-derives the status from the recorded checks, so a verdict flipped to `verified`
after the fact is caught from the bytes alone — the same discipline the audit chain
and the cross-org settlement artifacts hold. Soundness is by construction: a kernel
emits `verified` only when it actually recomputed the claim and it matched, so a
wrong answer the relevant kernel can see is *refuted*, never silently passed.

### The kernels

| Kernel | Checks | Refutes |
|---|---|---|
| `ArithmeticVerifier` | `a op b = c`, `n% of m is k` | a recomputation that disagrees |
| `UnitVerifier` | `X u1 = Y u2` conversions | a wrong value **or** a dimensional mismatch (`5 km = 5000 kg`) |
| `TemporalVerifier` | date ordering, `from A to B is N days` | an off-by-one or reversed ordering against a real calendar |
| `ConstraintVerifier` | an assignment satisfies typed `Constraint`s | any violated constraint |
| `SchemaVerifier` | structural conformance to a JSON schema | a structural violation |
| `CitationVerifier` | every verifiable claim is entailed by cited evidence | a claim no evidence supports (with strict number checking) |

Ground the kernels through the call: `evidence=` for citation entailment, `schema=`
for structural conformance, `constraints=` for constraint satisfaction, `facts=` and
`now=` for cross-checks.

```python
from vincio.verify import Constraint

ctx_constraints = [Constraint.compare("x", "<=", 10), Constraint.compare("x", ">", 0)]
ok = app.verify_reasoning({"x": 7}, constraints=ctx_constraints)
assert ok.holds
```

### Refuse or repair

A refuted certificate refuses to emit by default. When the answer can be repaired,
pass a `regenerate` callable to drive the **bounded self-correction loop** — the
deterministic refutations become a critique, the callable produces a fresh answer,
and it is re-certified, up to `max_cycles`. This is the same refuse-or-repair
discipline structured output already uses, now over *reasoning* rather than
*structure*.

```python
fixed = app.verify_reasoning("2 + 2 = 5", regenerate=lambda ans, critique: "2 + 2 = 4")
assert fixed.holds and fixed.attempts == 2
```

Pass `raise_on_refute=True` to raise `CertificateRefutedError` instead of returning a
refused answer. Every verdict lands on the hash-chained audit log as a
`reasoning_verification` decision.

## Runtime verification & shielding

The certificate proves a *result*; its behavioural, online analogue is a property
over an agent's plan or tool trajectory, checked step-by-step as it runs. A
`BehaviorSpec` states the property as plain data — events that must **never** occur
(`forbid`), an ordering one event must precede another (`require_before` /
`precede`), and an invariant of every event (`invariant`).

```python
from vincio import BehaviorSpec, EventPattern, RuntimeMonitor, BehaviorEvent

cite_first = BehaviorSpec(name="cite-before-claim").precede(
    EventPattern(kind="retrieval"), EventPattern(kind="claim"),
    description="claimed before retrieving any evidence",
)
monitor = RuntimeMonitor(cite_first)               # app.behavior_monitor(cite_first)
verdict = monitor.check_trajectory([BehaviorEvent(kind="claim", name="x")])
assert not verdict.ok                              # a claim with no prior retrieval
```

A `Shield` wraps a monitor and **prevents** a violation: `block` refuses a violating
action, `repair` maps it through a callback to a safe alternative that is re-checked,
and `monitor` records without stopping. Installed on the tool runtime, the shield
makes an unsafe tool call structurally impossible — the per-step, online counterpart
of the rails and the ahead-of-run governance verifier.

```python
no_unapproved_write = BehaviorSpec(
    name="approval-before-write",
    forbid=[EventPattern(kind="tool_call",
                         where={"side_effects": "write", "approved": False})],
)
app.shield(no_unapproved_write, use=True)          # installs on app.tool_runtime
# An unapproved write tool now returns a denied result before it executes.
```

## Verified tool use & synthesized programs

A tool can declare a **contract** on its behaviour, not merely its schema: pre- and
post-conditions the runtime checks against the *actual* arguments and result. A
breach raises `ToolContractError` at the boundary — an out-of-contract result is
refused, never returned.

```python
from vincio import ToolContract

contract = (
    ToolContract()
    .requires_that("amount > 0", lambda args: args["amount"] > 0)
    .ensures_that("returns a charge id", lambda args, result: "id" in result)
)
app.add_tool(charge, side_effects="write", contract=contract)
```

`synthesize` brings proof-carrying code into the tool plane: a small, **verified**
data transform built from a whitelisted, deterministic op set (no `eval`, no I/O). It
runs on representative examples, checks its declared properties, and binds the
verdict into the same `Certificate` an answer carries — the properties are proven
before the program is allowed to run, and re-checked on every use.

```python
from vincio import ProgramSpec, ProgramOp, ProgramProperty

spec = ProgramSpec(
    name="line-total",
    ops=[ProgramOp(op="derive", field="total", expr="price * quantity")],
    properties=[
        ProgramProperty(kind="row_count", relation="preserved"),
        ProgramProperty(kind="field_nonnegative", field="total"),
    ],
)
program = app.synthesize_program(spec, examples=[{"price": 3.0, "quantity": 2}])
assert program.holds
program.run([{"price": 2.0, "quantity": 10}])      # re-checks properties at run time
```

## Optional SMT / CAS

The deterministic kernels are the default and need no extra. For the cases that
warrant a solver — proving a constraint system is *consistent* rather than that one
assignment happens to satisfy it, or checking an equality with **exact** rational
arithmetic — `vincio.verify.smt` provides `SmtConstraintVerifier` (Z3) and
`CasArithmeticVerifier` (SymPy) behind `pip install "vincio[verify]"`. They are
strictly opt-in: nothing on the offline path imports them.

## How it composes

`verify_reasoning` is a deterministic, offline check that folds into self-correction
and the rails; the shield is the per-step counterpart of `verify_governance`; and tool
contracts and synthesized programs extend the proof discipline into the tool plane.
Together they take the platform from one whose per-answer signals are *probabilistic*
to one that, where it is possible, emits **a proof you can check** — without a hosted
prover, always offline, always additive on the frozen surface.
