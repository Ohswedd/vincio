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

## See also

- [The semantic layer and governed metrics](semantic-layer-and-governed-metrics.md) — the governed metric the engagement threads.
- [Charts and cited analytical artifacts](charts-and-cited-artifacts.md) — the data-bound figures the `cite` stage embeds.
- [Governed text-to-query and cell-level provenance](governed-text-to-query.md) — the cell-level lineage every finding inherits.
