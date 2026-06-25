# Guide: cross-org workflow choreography

Vincio already lets agents discover, negotiate, and contract across organizations
over the A2A fabric. This guide covers the next rung: the **durable work** they
coordinate, a long-running, compensating workflow that spans more than one
organization's agent fabric. Each org governs and audits its *own* steps on its
*own* hash-chained chain; there is no shared control plane, only a typed contract
and audited handoffs cross a trust boundary; and a failure on one side triggers
deterministic compensation across the whole choreography, so a half-completed
cross-org transaction unwinds cleanly. It is the choreography analogue of the
durable graphs that already run within one trust boundary.

This is additive (`vincio.choreography`); it changes nothing about how a single
agent or an in-process workflow runs, and runs fully offline against deterministic
local participants.

## A compensating cross-org saga

A `Saga` is an ordered list of steps. Each step dispatches a typed request to a
named **participant** (an org) and declares the **compensation** that undoes it.
The steps run in order; if any step fails, the already-completed steps are
compensated in reverse order, the saga pattern, now spanning trust boundaries.

```python
from vincio import ContextApp
from vincio.choreography import Saga

app = ContextApp(name="acme")

saga = (
    Saga(name="fulfil-order")
    .step("reserve", participant="warehouse", action="reserve", compensation="release")
    .step("charge",  participant="payments",  action="charge",  compensation="refund")
    .step("ship",    participant="warehouse", action="ship")
)
```

A participant is the binding to one org. The simplest is a dict of handler
callables run in-process (each handler takes the step's payload and returns its
output, or a `StepOutcome` to also declare delivered cost / latency / quality):

```python
participants = {
    "warehouse": {
        "reserve": lambda p: {"ticket": reserve_stock(p["sku"])},
        "release": lambda p: release_stock(p["forward_output"]["ticket"]),
        "ship":    lambda p: {"tracking": dispatch(p["sku"])},
    },
    "payments": {
        "charge": lambda p: {"receipt": charge_card(p["amount"])},
        "refund": lambda p: refund(p["forward_output"]["receipt"]),
    },
}

result = app.choreograph(saga, participants=participants, input={"sku": "A1", "amount": 4999})
print(result.status)            # "completed" | "compensated" | "failed" | "interrupted"
print(result.completed_steps)   # steps that ran forward
print(result.compensated_steps) # steps unwound on a failure (reverse order)
```

A later step can derive its payload from earlier steps' outputs with a `build`
callable, which receives a `SagaContext` (the run `input` and each completed step's
output):

```python
.step(
    "ship",
    participant="warehouse",
    action="ship",
    build=lambda ctx: {"ticket": ctx["reserve"]["ticket"], "to": ctx.input["address"]},
)
```

## Compensation unwinds a failure

A forward step **fails** when its participant returns `ok=False`, raises, or
breaches its step contract (below). A failure is terminal: the engine compensates
the completed steps in reverse order and stops; it never loops. A compensation
handler receives the forward step's recorded output under `forward_output`, so it
knows exactly what to undo.

```python
result = app.choreograph(saga, participants=participants)

if result.status == "compensated":
    # A step failed; every completed step was rolled back cleanly.
    print("rolled back:", result.compensated_steps)   # e.g. ["charge", "reserve"]
elif result.status == "failed":
    # A compensation itself failed, the residue needs operator attention.
    print("could not fully unwind")
```

A step with no `compensation` is simply skipped on rollback (nothing to undo). If a
compensation itself fails, the saga ends `failed` and the journal pinpoints the
outstanding compensations; pass `raise_on_compensation_failure=True` (on the
`Choreography` engine) to surface a `CompensationError` instead.

## Under a negotiated contract

A step can carry the `Contract` a `Negotiation` converged on. After the step runs,
the coordinator checks the delivered cost / latency / quality against the agreed
terms; a breach is a step failure and triggers compensation, so the contract is
enforced as part of the choreography, not just hoped for.

```python
deal = app.negotiate("transcribe batch", buyer=buyer, seller=seller)

saga.step(
    "transcribe",
    participant="vendor",
    action="transcribe",
    compensation="discard",
    contract=deal.contract,        # delivered work is checked against the agreed terms
)
```

The participant declares its delivered metrics by returning a `StepOutcome`:

```python
from vincio.choreography import StepOutcome

def transcribe(payload):
    out = do_work(payload)
    return StepOutcome(ok=True, output=out, cost_usd=0.08, latency_ms=2400, quality=0.95)
```

## Run-time discovery: bind a capability instead of an org

A step does not have to name its participant up front. Instead of
`participant="vendor"`, declare the **capability** the step needs and let the
engine resolve *who* runs it at dispatch time from a governed
[`AgentDirectory`](agent-fabric.md), so a choreography binds the best-available
counterparty for each step rather than a hard-coded org id. Discovery changes
*who* runs a step, never *how* it is governed: the resolved org runs under the
same allow-list, contract, per-org audit, compensation, and settlement a
statically-wired one does.

```python
from vincio.choreography import Saga

# Register candidate vendors in a governed (allow-listed, audited) directory.
directory = app.agent_directory(allow=["vendor-*"])
directory.register(vendor_a_card)   # an A2A Agent Card advertising "transcription"
directory.register(vendor_b_card)

saga = Saga(name="job").step(
    "transcribe",
    action="run",
    capability="transcription",     # <- resolved at dispatch time, not wired by id
    compensation="discard",
)

result = app.choreograph(
    saga,
    participants={"vendor-a": vendor_a, "vendor-b": vendor_b},
    directory=directory,            # the binder is built from this + reputation + settlement
)

binding = result.bindings["transcribe"]
print(binding.org)                  # the counterparty discovery chose
print([(c.org, c.score) for c in binding.candidates])  # the full ranked field, for audit
```

Among the candidates that advertise the capability **and** pass the directory's
allow-list **and** have a participant binding (so they are reachable), the binding
prefers the one whose reputation and prior settlement record best fit the step's
contract:

- **Reputation**: a candidate's [`ReputationLedger`](../guides/negotiation.md)
  standing (its no-regression / contract-fulfilment track record) weights its
  score, so a repeatedly-regressing vendor is discounted without being singled out.
- **Settlement fit**: its [`SettlementBook`](settlement.md) history weights the
  rest: the share of prior settlements it honoured, and how well its delivered
  cost sat under the step contract's agreed price.

Attach a reputation ledger (`app.use_reputation_ledger()`) and a settlement book
(`app.use_settlement_book()`) and they feed the ranking automatically; tune the
blend with `binding_weights=BindingWeights(...)`, or pass a prepared
`CapabilityBinder` as `binder=`. Ties break deterministically by org id, so a
binding is reproducible.

Every candidate's governed resolution is recorded on the audit chain
(`agent_resolve`), and the binding decision itself lands as a `choreography_bind`
entry, so discovery is as accountable as a statically-wired dispatch. A capability
that no allowed, reachable candidate advertises raises a `ChoreographyError` rather
than failing silently. A discovered step compensates at the org it was actually
bound to (recorded on the journal), and a resume re-binds only the steps that had
not yet run.

## Durable and resumable

The `SagaJournal` is checkpointed to the app's metadata store after **every** step,
so a saga survives a restart the way an in-process durable graph does. Rebuild the
same `Saga` and participants in code (only the journal is persisted) and resume by
`saga_id`, completed steps keep their outputs and are never re-run:

```python
# First process: run, or pause cooperatively after N steps.
result = app.choreograph(saga, participants=participants, saga_id="ord-42", interrupt_after=1)
assert result.status == "interrupted"

# A later process, after a restart, same store, fresh engine.
resumed = app.resume_choreography(saga, "ord-42", participants=participants)
assert resumed.status == "completed"
```

A saga interrupted mid-rollback finishes compensating on resume; a terminal saga
is returned unchanged, so resume is idempotent.

## Offline-verifiable

The journal is **hash-chained**: each record links to the previous one by a content
hash, so `verify()` recomputes the chain from the bytes alone and pinpoints any
record edited, inserted, or dropped, without the live coordinator.

```python
verdict = result.journal.verify()
assert verdict.intact            # the whole chain recomputes

# A tampered record is caught at its sequence number.
result.journal.records[0].output = {"forged": True}
assert not result.journal.verify().intact
```

Pass a `signer` to the `Choreography` engine to also sign each record for
third-party verification (`journal.verify(verifier)`), the same way the audit chain
signs its entries.

## Per-org governance over the A2A fabric

A participant can live in **another organization, reached over A2A**. The remote org
exposes its handlers as an A2A agent; the coordinator drives it with a
`RemoteParticipant`. Crucially, each side audits its own steps on its own chain,
the coordinator records the dispatched handoff, the participant records its
execution, so there is no shared control plane, only the typed handoff crossing
the boundary.

```python
from vincio.choreography import RemoteParticipant
from vincio.a2a import connect_a2a_in_process

# Vendor org: expose its capabilities over A2A (audited on the vendor's own chain).
server = vendor_app.serve_choreography(
    {"transcribe": transcribe, "discard": discard}, org_id="vendor"
)

# Coordinator org: reach it as a remote participant.
client = connect_a2a_in_process(server)          # or connect_a2a(url) over HTTP
remote_vendor = RemoteParticipant(client, org_id="vendor")

result = coordinator_app.choreograph(saga, participants={"vendor": remote_vendor})
```

The same saga runs identically against a local or a remote participant; the engine
drives both through the same `perform` / `compensate` protocol.

## What it is not

This is a library capability inside your process, not a hosted control plane. There
is no central orchestrator service, no shared state machine, no managed saga log, a
choreography is a coordinator dispatching typed requests under a contract, and the
journal is a hash-chained file you hold and verify yourself. Each organization
governs and audits the steps that cross into it. Everything that looks operational
is something you run.
