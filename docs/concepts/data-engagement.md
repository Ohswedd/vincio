# The data engagement — the analytics plane as one system

Seven rungs built the data & analytics plane — [tabular evidence](tabular-evidence.md),
[profiling and quality rails](dataset-profiling.md),
[governed text-to-query](governed-text-to-query.md), the
[data-analysis agent](data-analysis-agent.md),
[charts and cited artifacts](charts-and-cited-artifacts.md),
[streaming](streaming-and-out-of-core.md), and the
[semantic layer](semantic-layer-and-governed-metrics.md) — each grounded, cited,
and offline-verifiable on its own. What was missing is not an eighth primitive but
the **whole**: nothing yet presented the plane as one coherent system, threaded a
real analysis from a raw table to a cited deliverable behind one governed call-path,
or proved that whole composition verifies offline.

A [`DataEngagement`](../../vincio/data/engagement.py) (built by
[`app.data_engagement`](../../vincio/core/app.py)) is that capstone — the analytics
analogue of the cross-org [`CrossOrgEngagement`](../../vincio/settlement/engagement.py).
It is **purely compositional**: every lifecycle method delegates to the *same*
`app.*` primitive a caller would use directly (each unchanged and still usable on
its own), captures the artifact it produced, and records it as a stage in a single
hash-linked narrative. Everything here is deterministic, dependency-free, and
offline — never a hosted query engine, a managed warehouse, or a notebook service.

## Thread the whole plane in one call-path

```python
from vincio import ContextApp
from vincio.data import DerivedColumn, Dimension, Measure

app = ContextApp(name="analyst")
eng = app.data_engagement(question="how does revenue break down by region?")
eng.register(rows, columns=["region", "price", "qty"], name="sales", source="crm-export")
app.semantic_layer(
    "sales",
    derived=[DerivedColumn(name="revenue", expression="price * qty")],
    measures=[Measure(name="total_revenue", agg="sum", expression="revenue")],
    dimensions=[Dimension(name="region")],
)

eng.profile()                                   # DatasetProfile
eng.sample(4)                                   # representative sample
eng.screen()                                    # DataQualityReport
eng.query("total qty by region")                # cell-cited QueryResult
eng.analyze("how does qty break down by region?")   # cited AnalysisResult
eng.chart(eng.result, title="Qty by region")    # content- and data-bound Chart
eng.query_metric("total_revenue", by=["region"])    # governed MetricResult
eng.cite(title="Revenue analysis")              # per-figure data-bound deliverable

narrative = eng.seal()                          # signed, hash-chained DataNarrative
```

Each lifecycle verb — `register`, `profile`, `sample`, `fit`, `screen`, `query`,
`analyze`, `chart`, `query_metric`, `cite` — calls the primitive documented in the
rungs above, stores the artifact on the engagement (`eng.result`, `eng.analysis`,
`eng.chart_`, `eng.metric`, …), returns it, and records it as a stage. The
primitives are unchanged: `app.query_data`, `app.analyze_data`, and the rest still
work exactly as they do on their own. The facade adds no new analytical logic; it
composes and *narrates* them.

## The narrative verifies offline

`eng.seal()` mints a [`DataNarrative`](../../vincio/data/engagement.py): an ordered
chain of `DataStage`s, each binding the stage's verb, the artifact's own published
commitment (a `result_hash` / `chart_hash` / `layer_hash`), and a deterministic
digest of its bytes into a link chaining to the previous one. It is signed by the
analyst and lands on the hash-chained audit log (action `data_engagement`).

```python
v = narrative.verify(app.contract_signer)
v.valid            # True — the chain links, the head and content hash recompute,
                   # the signature checks
v.broken_at        # the first failing stage, when a tamper is introduced
```

`DataNarrative.verify` recomputes the whole chain from the bytes alone, so a
re-ordered stage, an edited digest, a broken link, a tampered head, or a forged
signature is caught. It round-trips through `to_wire` / `from_wire` unchanged.

## Data-binding — every finding re-derives from the source

Beyond the cross-org engagement's digest-binding, a data engagement carries the
data plane's distinguishing guarantee. `eng.verify(verifier, catalog=...)`
re-digests every captured artifact *and* — given the live catalog (defaulting to
`app.data_catalog()`) — **re-executes every captured query, analysis, chart, and
metric against the content-hashed source** and confirms each re-derives from the
bytes:

```python
whole = eng.verify(app.contract_signer)
whole.digests_ok   # True — no captured artifact's bytes were edited
whole.data_bound   # True — every finding re-derives from the source it cites
whole.valid        # the conjunction
```

So a tampered *source* is caught even when the chain itself is intact — the
analytics analogue of a generated report's per-claim entailment, applied to the
whole engagement. The verifier is the query plane's offline re-execution, not a
model: a finding is carried only when its query, analysis, chart, or metric
re-derives.

## How a stage seals into the chain

Every lifecycle verb runs the same three steps: call the underlying `app.*`
primitive, stash the artifact on a named attribute (`eng.result`, `eng.chart_`,
`eng.metric`, …), and append a `DataStage` capturing four things — the verb, the
artifact's `kind` / `artifact_id`, the **published commitment** it already exposes
(a `result_hash`, `chart_hash`, or `layer_hash`, read off the artifact), and a
deterministic `digest` of the artifact's own `to_wire` / `model_dump` bytes. A
query / analysis / chart / metric stage additionally records a **binder** — a
`catalog -> bool` closure that re-runs the artifact's own `verify` against a live
catalog.

`seal()` walks the stages in order and stamps the hash chain: each stage's
`prev_hash` is the prior stage's `entry_hash`; its `entry_hash` hashes the link
facts (index, verb, kind, ids, digest, summary, prev — deliberately *not* the
timestamp); the narrative's `head_hash` is the last link; and `content_hash` binds
the analyst, dataset, question, and the ordered entry hashes. That single content
hash is what the analyst signs and what `verify` recomputes from the bytes alone.

The chain then yields two **independent** guarantees, surfaced as two flags on
`DataEngagementVerification`:

- **`digests_ok`** — re-digesting each captured artifact's bytes still matches the
  stage's bound `digest`. This catches an edit to a *captured result* (a number
  changed after the fact) with no catalog needed.
- **`data_bound`** — every stage's binder re-executed its query / analysis / chart
  / metric against the live, content-hashed source and re-derived. This catches a
  tamper to the *source itself*, even when the chain and every digest are intact.

`eng.cite(...)`'s own headline guarantee is **per-figure** data binding: it embeds
each chart/table as a `Figure` verified to re-derive from the source under a
default `CitationContract` that requires figure binding but not per-claim
`[E]`-marker coverage over the prose narrative (whose grounding is the cell refs
the figures carry). Pass a stricter contract when you also want per-claim
entailment.

**Gotchas.** Run the verbs in analysis order — each defaults its dataset/table to
the one `register(...)` opened, so register first. A one-shot `dataset=` on
`query` / `analyze` / `query_metric` (an unregistered table) records **no**
data-binder: that stage is digest-bound but not source-re-executed, because there
is nothing registered to re-run against. `verify(catalog=...)` defaults to
`app.data_catalog()`; when nothing is registered, `data_bound` stays `None`
(unchecked), not `False`. And re-sealing after more stages run mints a *fresh*
narrative — hold the object `seal()` returns rather than re-reading a stale one.

## Held by the conformance family

The [`data_analysis_conformance`](../../benchmarks/vinciobench.py) VincioBench family
holds the capstone under CI-gated budgets and published SLOs: an **end-to-end**
check (a complete engagement composes and seals into a narrative that verifies
offline), a **data-bound** check (every captured query, analysis, chart, and metric
re-derives from the content-hashed source), and a **tamper-evident** check (a
re-ordered stage, an edited digest, an edited underlying artifact, a tampered
source, and a forged signature are each caught). The facade adds no new
performance-sensitive path — it reuses the primitives, so its cost is their sum.

With this capstone the data & analytics plane is **feature-complete and frozen**
under the [stability policy](../reference/stability.md).

## Explore it interactively

The same governed engagement runs **interactively** in a notebook or REPL through the
[`vincio.notebook`](../../vincio/notebook.py) surface — no hosted notebook service.
`notebook_session(app, ...)` is a thin front over `app.data_engagement`: each verb
delegates to the *same* primitive, renders the cited artifact inline (call
`enable_rich_reprs()` and a `QueryResult`, an `AnalysisResult`, a `Chart`, and the
sealed `DataNarrative` display as cards with clickable cell citations), and threads it
into the engagement's narrative. Sealing `session.narrative` mints the same signed,
audited `DataNarrative` a script does, and `session.verify()` re-derives every inline
finding from the bytes — so an interactive exploration is reproducible and
offline-verifiable by construction:

```python
import vincio.notebook as nb

session = nb.notebook_session(app, question="how does revenue split by region?")
session.register(rows, columns=cols, name="sales")
session.query("total revenue by region")   # renders inline, cited
session.analyze("how does revenue break down by region?")
session.chart(session.result, title="Revenue by region")
session.cite(title="Revenue analysis")
session.verify()                            # data-bound, offline
```

See the fully-offline [`06_notebook_native_analysis.ipynb`](../../examples/notebooks/06_notebook_native_analysis.ipynb).

## See also

- [Cross-org federated analytics](federated-data-engagement.md) — the cross-organization extension of this capstone: one governed metric across orgs, with only aggregated, cited results crossing the trust boundary.
- [The semantic layer and governed metrics](semantic-layer-and-governed-metrics.md) — the governed metric the engagement threads.
- [Charts and cited analytical artifacts](charts-and-cited-artifacts.md) — the data-bound figures the `cite` stage embeds.
- [Governed text-to-query and cell-level provenance](governed-text-to-query.md) — the cell-level lineage every finding inherits.

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
