# Cross-org / federated analytics — the data plane across a trust boundary

The single-org [data engagement](data-engagement.md) threads the whole analytics
plane behind one governed call-path and seals it into a signed, data-bound
[`DataNarrative`](../../vincio/data/engagement.py). But an analytical question
often spans **more than one organization's** data — total revenue across a
partnership, a benchmark over a cohort of independent operators — and the answer
must be computed **without pooling the raw rows into a shared warehouse**.

A [`FederatedDataEngagement`](../../vincio/data/federated.py) (built by
[`app.federated_data_engagement`](../../vincio/core/app.py)) is that reach — the
analytics analogue of [federated self-improvement](../../vincio/optimize/federated.py)
and the data-plane twin of the cross-org
[`CrossOrgEngagement`](../../vincio/settlement/engagement.py). It runs one governed
metric across several organizations over the *existing* cross-org fabric:
negotiated as a [`Contract`](../../vincio/negotiation/contract.py), choreographed
as a [`Saga`](../../vincio/choreography/saga.py), reconciled into one signed,
offline-verifiable narrative. Everything here is deterministic, dependency-free,
and offline — never a hosted query federation, a shared warehouse, or a data
clean-room service.

## The federated query is negotiated, not pooled

A [`FederatedQuery`](../../vincio/data/federated.py) is the *shape* of the metric
run everywhere — the measures and dimensions, the source columns it touches, the
residency posture it must respect, the budget, and the `min_members` k-anonymity
contributor floor. Its digest is bound into a negotiated contract's hashed scope,
so the agreed query shape is tamper-evident.

```python
from vincio import ContextApp, FederatedQuery
from vincio.data import DerivedColumn, Dimension, Measure

# Each org has its own data, its own governed layer, its own audit chain.
def org(name, rows):
    app = ContextApp(name=name)
    app.register_dataset(rows, columns=["region", "price", "qty"], name="sales")
    app.semantic_layer(
        "sales",
        derived=[DerivedColumn(name="revenue", expression="price * qty")],
        measures=[Measure(name="total_revenue", agg="sum", expression="revenue")],
        dimensions=[Dimension(name="region")],
    )
    return app

coordinator = ContextApp(name="coordinator")   # holds the layer definitions, no data
query = FederatedQuery.of("total_revenue", table="sales", by=["region"], min_members=2)

fed = coordinator.federated_data_engagement(query=query)
fed.add_member("acme", org("acme", acme_rows), region="us-east-1")
fed.add_member("globex", org("globex", globex_rows), region="eu-west-1")

findings = fed.run()        # negotiate → dispatch (choreographed) → reconcile
narrative = fed.seal()      # signed, hash-chained FederatedNarrative
```

`run` is the convenience over the granular `negotiate` / `dispatch` / `reconcile`
verbs. Each org is a [`FederatedMember`](../../vincio/data/federated.py) binding an
org id to its *own* `ContextApp` — its catalog, its
[semantic layer](semantic-layer-and-governed-metrics.md), its consent ledger and
privacy accountant, its audit chain.

### How the shape is bound and dispatched

The query's `digest()` hashes its facts — the metrics, dimensions, filters, table,
`columns_touched`, `residency`, `purpose`, `budget_usd`, and `min_members` — and
its `scope()` renders the human-legible, digest-bound string the contract carries:

```
federated-metric:total_revenue@sales#<digest>
```

`negotiate` binds that scope into a signed `Contract` — the scope *is* part of the
contract's content hash, so the agreed query shape is tamper-evident — defaulting
the buyer/seller positions from the query's budget. `dispatch` then builds a
contract-governed `Saga` with one step per approved member; each step's handler
runs, **in that member's own app**, `member.app.query_metric(query.metric,
table=member.table)` and returns only the `MetricResult` wire plus its source
content hash — never a row. As each aggregate returns, the coordinator checks its
`layer_hash` against the reference layer's digest and **refuses a mismatch**, so
one metric means one thing federation-wide or the whole round fails.

## The raw rows never cross — only the aggregate does

`dispatch` builds a contract-governed `Saga` with one step per org. Each step runs
that org's **own** [`query_metric`](semantic-layer-and-governed-metrics.md)
*locally* and returns only the typed, aggregated, cell-cited
[`MetricResult`](semantic-layer-and-governed-metrics.md). The raw rows stay home —
they are never serialized into the dispatch, the journal, or the narrative; the
only thing that crosses a trust boundary is a group-by aggregate. Because the
metric runs through each org's semantic layer, the grouping attributes are the
analyst-sanctioned governed **dimensions**, never arbitrary raw identifiers, and
every org must compute the metric by the **same** layer definitions — an org whose
layer digest differs is refused, so the metric is computed one way everywhere or
not at all.

The members' aggregates are **reconciled** into one
[`FederatedFinding`](../../vincio/data/federated.py) per metric and group.
Reconciliation is *exact* for the partition-decomposable aggregations — `SUM` and
`COUNT` add across orgs, `MIN` / `MAX` take the extremum — so a federated total is
the true total, not an estimate.

```python
narrative.finding("total_revenue", region="NA").value   # the exact cross-org total
```

`AVG`, `COUNT_DISTINCT`, and ratio measures are *not* exactly decomposable across
partitions (an average needs the per-partition counts; a distinct count needs the
per-partition sets) and are refused at construction with guidance — federate the
decomposable components (e.g. a `SUM` and a `COUNT`) and combine the reconciled
values.

## Governance crosses the boundary intact

Every rail a local query passes applies at the boundary, exactly as it would
locally — audited on the coordinator's chain while each org's query plane audits
its read on its own chain:

- **Residency-aware egress refusal** (reusing
  [`ResidencyPolicy`](../../vincio/governance/residency.py)) — a member whose data
  region is outside the query's posture may not let even an aggregate cross;
  `dispatch` raises `ResidencyViolationError`.
- **Consent** — when a member has configured a [consent ledger](../guides/governance.md),
  it must permit the `ANALYTICS` purpose for its subject, or the contribution is
  refused.
- **The differential-privacy budget** — each contribution is charged against the
  member's [privacy accountant](../../vincio/governance/privacy.py) exactly as a
  federated training round is; a member over budget refuses before its aggregate
  runs.
- **The k-anonymity contributor floor** — a round with fewer than `min_members`
  eligible orgs is refused, so a single org's aggregate is never singled out.

The rails run **before** the saga, in order — residency, then consent (only when
the member configured a ledger), then the privacy charge (only when it has an
accountant) — refusing and auditing a non-compliant member rather than dropping it
silently. Only the surviving *approved* set is then checked against `min_members`.
Each per-member decision lands on the coordinator's chain under
`federated_query_governance`; the sealed engagement lands under
`federated_data_engagement`.

## The narrative verifies offline and is data-bound

`fed.seal()` mints a [`FederatedNarrative`](../../vincio/data/federated.py): an
ordered chain of `FederatedStage`s (negotiate → choreograph → per-org query →
reconcile), signed by the coordinator and landed on the audit chain (action
`federated_data_engagement`). The reconciled findings travel on the narrative, so
the federated answer is carried as data.

```python
v = fed.verify(coordinator.contract_signer)
v.valid          # the chain links, the head and content hash recompute, the sig checks
v.data_bound     # every org's aggregate re-executes against its content-hashed
                 # source AND every reconciled value re-derives from those aggregates
```

`FederatedNarrative.verify` recomputes the whole chain from the bytes alone, so a
re-ordered stage, an edited digest, a tampered head, or a forged signature is
caught; `FederatedDataEngagement.verify` additionally re-executes each member's
aggregate against its live source and re-derives every reconciled finding, so a
tamper to any org's source — or to the reconciliation — is caught even when the
chain itself is intact.

## Gotchas

- Only `SUM` / `COUNT` / `MIN` / `MAX` federate **exactly**; `validate_against`
  refuses an `AVG`, `COUNT_DISTINCT`, or ratio measure at negotiate time — federate
  the decomposable components (a `SUM` and a `COUNT`) and combine the reconciled
  values yourself.
- Every member must define the metric by **identical** layer definitions; a
  differing `layer_hash` fails the round (it is refused, never silently dropped).
- An empty `residency` posture leaves egress **ungated** — the policy is enforced
  only when `residency` names allowed regions; set them to actually gate a member's
  region.
- `verify(catalogs=...)` re-executes each member's aggregate against its **own**
  catalog by default; pass `catalogs` keyed by org only to override the source a
  member re-runs against.

## Held by the data-plane family

The [`data_plane`](../../benchmarks/vinciobench.py) VincioBench family holds the
federated reach under CI-gated budgets and [published SLOs](../reference/slo.md): a
**rows-never-cross** check (no raw source row crosses the boundary; only the
aggregated metric output is dispatched and sealed), a **federated-data-binding**
check (every reconciled finding re-executes against each org's content-hashed
source), and a **governance-preservation** check (residency, consent, the privacy
budget, and the contributor floor each refuse a non-compliant round).

## See also

- [The data engagement](data-engagement.md) — the single-org analytics capstone this extends.
- [The semantic layer and governed metrics](semantic-layer-and-governed-metrics.md) — the governed metric each org runs locally.
- [Real-time and streaming analytics](realtime-streaming-analytics.md) — the same governed metric over an unbounded event stream.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: Analyze data](../guides/analyze-data.md)
- [Example: 13_data_and_analytics.py](../../examples/13_data_and_analytics.py)
- [Concept: Context packets & long-horizon governance](context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
