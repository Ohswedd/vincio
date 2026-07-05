# Verified reasoning & neuro-symbolic certificates

Vincio scores answers with judges, oracles, and a governance verifier, but those
per-answer signals are *probabilistic*. For the classes of question where it is
possible, an answer can instead carry a **checkable certificate** a deterministic
verifier confirms independently of the model: arithmetic, units, dates, constraint
satisfaction, schema and citation consistency are all checkable. This turns *the
judge thinks it is right* into *here is a proof you can verify*, the output-side
analogue of the governance verifier's machine-checked invariants.

Everything here is opt-in, additive, deterministic, and offline. The deterministic
kernels are dependency-free; optional SMT / CAS backends sit behind
`pip install "vincio[verify]"`.

## Proof-carrying answers

`app.verify_reasoning(answer)` runs a set of deterministic kernels over an answer
and returns a `VerifiedAnswer` whose `Certificate` is **verified** (a kernel
recomputed a claim and it held), **refuted** (a recomputation disagreed, a *proof
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
after the fact is caught from the bytes alone, the same discipline the audit chain
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
pass a `regenerate` callable to drive the **bounded self-correction loop**, the
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

## Statistical claims — forecasting & causal inference

A data answer concludes from numbers as well as retrieves them: a trend, a
correlation, a confidence interval, a forecast. Each of those carries a checkable
certificate the way an arithmetic claim does, recomputed from the **cited cells** —
not judged by a model. Pass the claims as `statistical_claims=`; the statistical
kernels are added to the default set automatically.

| Kernel | Claim | Refutes |
|---|---|---|
| `TrendVerifier` | `TrendClaim` — OLS slope / intercept / R² / direction over a series | a slope, intercept, R², or direction the data does not bear out |
| `CorrelationVerifier` | `CorrelationClaim` — Pearson `r` of two series, optionally causal | a wrong `r`, **or** a correlation stated as causation with no controls, **or** a controlled claim whose association collapses once the confounder is partialled out |
| `IntervalVerifier` | `IntervalClaim` — a confidence (`mean`) or regression `prediction` interval | a stated interval that is too tight or too wide |
| `ForecastVerifier` | `ForecastClaim` — a deterministic model's projection (`naive` / `mean` / `drift` / `linear` / `moving_average` / `ses`) | a projection the model does not produce |

A statistic is **bound to its cited cells**: a `CitedSeries` carries the
`CellRef`s its values came from (build one straight from a cell-cited
`QueryResult` with `CitedSeries.from_cells(result.citations(row, col))`), and a
value swapped after it was cited makes the series unbound and the kernel refuses —
so a smuggled number cannot pass.

```python
from vincio import CitedSeries, TrendClaim

series = CitedSeries(name="revenue", values=[12_000, 12_300, 12_650, 12_900, 13_300])
out = app.verify_reasoning(
    "Revenue is trending up about 320/month.",
    statistical_claims=[TrendClaim(series=series, slope=320.0, direction="increasing")],
)
assert out.holds                                    # recomputed slope ≈ 320
```

The headline is **causal soundness**. A causal claim must earn its warrant — a
declared randomized design, or declared controls *with* their series so the
**partial correlation** can be recomputed. Correlation stated as causation with no
warrant is refused, and a controlled claim whose association vanishes once the
confounder is partialled out is refuted, while a genuine driver that survives the
control is verified.

```python
from vincio import CorrelationClaim

# Ice-cream sales and drownings both rise with temperature (the confounder).
claim = CorrelationClaim(x=ice_cream, y=drownings, r=0.97, causal=True,
                         controls=["temperature"], control_series=[temperature])
verdict = app.verify_reasoning("Ice cream sales cause drownings.",
                               statistical_claims=[claim])
assert verdict.refused                              # partial r collapses ≈ 0
```

A refuted statistical claim drives the same self-correction loop: a `regenerate`
callback may repair it by returning a corrected `StatisticalClaim`, and the loop
re-grounds the context before re-certifying. The fully-offline
[`13_data_and_analytics`](../../examples/13_data_and_analytics.py) example walks
all four statistical kernels end to end (section 11).

## Runtime verification & shielding

The certificate proves a *result*; its behavioural, online analogue is a property
over an agent's plan or tool trajectory, checked step-by-step as it runs. A
`BehaviorSpec` states the property as plain data, events that must **never** occur
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
makes an unsafe tool call structurally impossible, the per-step, online counterpart
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
breach raises `ToolContractError` at the boundary, an out-of-contract result is
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
verdict into the same `Certificate` an answer carries; the properties are proven
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
warrant a solver, proving a constraint system is *consistent* rather than that one
assignment happens to satisfy it, or checking an equality with **exact** rational
arithmetic, `vincio.verify.smt` provides `SmtConstraintVerifier` (Z3),
`CasArithmeticVerifier` (SymPy), and `CasTrendVerifier` (SymPy — re-discharges an
OLS trend fit with exact rational arithmetic, no floating-point drift) behind
`pip install "vincio[verify]"`. They are strictly opt-in: nothing on the offline
path imports them.

## Gotchas

- **`inapplicable` is not a pass.** A kernel that finds no checkable claim of its
  kind returns `inapplicable`, so `holds=True` means *nothing checkable was
  refuted*, not *the whole answer is proven*. `verify_reasoning` certifies the
  parts a deterministic kernel can see (arithmetic, units, dates, citations,
  constraints, the statistical claims) — it is not a hallucination catch-all for
  free prose.
- **Ground the kernels or they stay inapplicable.** `CitationVerifier` needs
  `evidence=`, `SchemaVerifier` needs `schema=`, `ConstraintVerifier` needs
  `constraints=`, temporal cross-checks need `facts=`/`now=`. Omit the grounding
  input and the corresponding kernel simply has nothing to check.
- **A causal claim must earn its warrant.** Correlation stated as causation with
  no declared randomized design or controls-*with*-series is **refused**, and a
  controlled claim whose partial correlation collapses is **refuted** — passing a
  bare `causal=True` will never verify.
- **A cited statistic is bound to its cells.** Swap a `CitedSeries` value after it
  was cited and the series is unbound, so the kernel refuses — a smuggled number
  cannot pass. Build the series from the `QueryResult` with
  `CitedSeries.from_cells(...)` rather than by hand.
- **A refuted answer refuses to emit by default.** Pass `regenerate=` to drive the
  bounded self-correction loop, or `raise_on_refute=True` to get
  `CertificateRefutedError` — don't expect a refuted certificate to be returned as
  a normal answer.
- **SMT/CAS are strictly opt-in.** Nothing on the offline path imports Z3/SymPy;
  reach for `vincio[verify]` only when you need consistency of a whole constraint
  *system* or exact rational arithmetic.

## How it composes

`verify_reasoning` is a deterministic, offline check that folds into self-correction
and the rails; the shield is the per-step counterpart of `verify_governance`; and tool
contracts and synthesized programs extend the proof discipline into the tool plane.
Together they take the platform from one whose per-answer signals are *probabilistic*
to one that, where it is possible, emits **a proof you can check**, without a hosted
prover, always offline, always additive on the frozen surface.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Example: 09_security_governance.py](../../examples/09_security_governance.py)
- [Example: 13_data_and_analytics.py](../../examples/13_data_and_analytics.py)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#governance)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
