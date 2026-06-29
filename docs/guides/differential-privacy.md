# Differential-privacy memory & training

A federated round bounds a *single member's per-round influence* with clipping and
a Gaussian mechanism. But a single bounded round is not a guarantee about a
*subject* whose data is touched again and again, by every memory consolidation
that folds their episodes into a durable summary, by every federated contribution
that learns from their traffic. The missing rung is a **provable, composing,
per-subject privacy budget** over memory consolidation and the whole learning loop:
the privacy analogue of the cost report's dollar budget.

`vincio.governance.privacy` adds it. A Rényi / moments **accountant** tracks the
cumulative `(ε, δ)` a subject's data has spent; a per-subject **budget** gates a
learning step the way the cost report gates a dollar; and a per-subject **report**
makes the spent budget a mechanical, auditable number. Everything here is opt-in and
additive, with no accountant attached, consolidation and contributions are
unaccounted exactly as before.

## Why an accountant, not a per-round bound

Privacy loss **composes**. Each release of aggregated information about a subject,
a consolidated summary, a federated contribution, leaks a little, and those leaks
accumulate. Naively adding each step's `ε` over-counts badly. A Rényi-DP accountant
composes the per-step Rényi-divergence curves by simple addition and converts to
`(ε, δ)` once at the end, so a subject who is consolidated ten times pays far less
than ten times a single consolidation's `ε`.

```python
from vincio.governance.privacy import PrivacyAccountant, PrivacyMechanism

acc = PrivacyAccountant()
mech = PrivacyMechanism(noise_multiplier=4.0)   # σ relative to L2 sensitivity
for _ in range(4):
    acc.record("alice", mech)
acc.spent("alice")          # ≈ 2.53  (naive 4× would be ≈ 4.92)
```

`PrivacyMechanism` models one differentially-private release: a Gaussian
mechanism with a `noise_multiplier` (`z = σ / Δ`, larger is more private), an
optional Poisson `sample_rate` (`< 1` amplifies privacy), and a `steps` count. The
math is exposed directly, `gaussian_rdp` for the (sub-sampled) Gaussian RDP curve
and `rdp_to_epsilon` for the standard RDP→`(ε, δ)` conversion, but you rarely need
it; the accountant composes for you.

## A budget that refuses

A `PrivacyBudget` is a per-subject (or default) `(ε, δ)` ceiling with an
`on_breach` policy. `check` decides whether a proposed release fits; `charge` gates
**and** commits in one call, raising `PrivacyBudgetError` when the budget refuses.

```python
from vincio import PrivacyBudget, PrivacyBudgetError

acc = PrivacyAccountant(default_budget=PrivacyBudget(epsilon=2.0))
acc.set_budget(PrivacyBudget(subject_id="vip", epsilon=10.0))   # a generous per-subject budget

try:
    acc.charge("alice", mech, operation="consolidation")
except PrivacyBudgetError as exc:
    print(exc.code, exc.details["remaining_epsilon"])   # PRIVACY_BUDGET_EXCEEDED
```

Set `on_breach="downweight"` and an over-budget release is not refused but **clipped
harder**, its sensitivity (and therefore its privacy cost) scaled down to fit the
remaining budget, instead of being dropped. Budgets are per-subject and isolated:
spending one subject's budget never touches another's.

## Wired into the app

Attach an accountant to a `ContextApp` and the two learning paths gate
automatically:

```python
from vincio import ContextApp, PrivacyBudget, PrivacyMechanism

app = ContextApp(name="assistant")
app.use_privacy_accountant(
    default_budget=PrivacyBudget(epsilon=2.0),
    default_mechanism=PrivacyMechanism(noise_multiplier=4.0),
)
app.add_memory()
```

**Memory consolidation.** Consolidating a subject's episodes into a durable
summary is a release of aggregated information about that subject, so
`app.memory.consolidate(session_id, user_id="alice")` charges Alice's budget. A
consolidation that would exceed it is refused, the episodes simply stay in their
short-lived episodic form, and the `ConsolidationReport` carries `privacy_refused`
and the cumulative `privacy_epsilon`.

**Federated contributions.** When the federated `PrivacyConfig` configures the
Gaussian mechanism (`dp_epsilon` set), `app.contribute_federated(...)` /
`app.federated_improvement(...)` compose the *same* per-subject budget, so the
accountant spans memory and the learning loop. An over-budget contribution is
refused; a down-weighted one is released **more privately**, the Gaussian
mechanism's `ε` scaled down (more noise relative to sensitivity) by the same factor,
so the recorded spend and the geometry actually released agree.

`app.set_privacy_budget(subject_id="alice", epsilon=1.0)` is the one-liner for
setting a budget (it creates the accountant on first use).

## A report, alongside the cost report

The spent privacy budget is auditable, not asserted. `app.privacy_report()` rolls
up each subject's `ε` spent against its ceiling, with operation and refusal counts,
the privacy analogue of `app.cost_report()`:

```python
app.privacy_report().print_summary()
# Privacy report (δ=1e-05)
#   alice: ε=1.76753/2 (remaining 0.232472), ops=2, refusals=1
```

Every spend (`privacy_spend`) and every refusal (`privacy_refused`) lands on the
same hash-chained, verifiable audit log as consent grants and erasure proofs, so the
guarantee is checkable offline.

## What ships

| Symbol | Role |
|---|---|
| `PrivacyAccountant` | Composing per-subject RDP accountant, budget gate, ledger, and report |
| `PrivacyBudget` | A per-subject (or default) `(ε, δ)` ceiling with a refuse / down-weight policy |
| `PrivacyMechanism` | One Gaussian release (noise multiplier, sample rate, steps) |
| `PrivacySpend` / `PrivacyDecision` | A recorded spend; an explainable gate verdict |
| `PrivacyReport` | Per-subject `ε`-spent / `ε`-remaining roll-up |
| `gaussian_rdp` / `rdp_to_epsilon` | The accountant's math, exposed for direct use |
| `app.use_privacy_accountant` / `set_privacy_budget` / `privacy_report` | The app surface |

The `privacy` VincioBench family holds the composition, refusal, and auditability
SLOs. See [`60_differential_privacy_memory_training.py`](../../examples/08_optimization_self_improvement.py)
for a runnable, fully-offline walkthrough.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Example: 08_optimization_self_improvement.py](../../examples/08_optimization_self_improvement.py)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#governance)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
