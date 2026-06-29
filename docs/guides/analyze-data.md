# Analyze data

Vincio treats structured data as a first-class evidence modality, not a string
you paste into a prompt. A typed, columnar `Dataset` is scored and budgeted by
the context compiler, queried through a read-only-verified query plane with
cell-level provenance, analyzed by a bounded agent, charted, governed by a
semantic layer, and finally threaded into one signed, offline-verifiable
narrative. Everything below runs fully offline on the deterministic mock
provider and the standard-library SQL engine — no warehouse, no network.

This guide is the task-oriented tour of the data & analytics plane. For the
model behind each step, follow the linked concept page.

## Register a dataset

```python
from vincio import ContextApp

app = ContextApp(name="analyst")
rows = [
    {"region": "NA", "product": "alpha", "price": 10.0, "qty": 3},
    {"region": "EU", "product": "alpha", "price": 8.0, "qty": 5},
    {"region": "NA", "product": "beta", "price": 12.0, "qty": 2},
]
app.register_dataset(rows, columns=["region", "product", "price", "qty"], name="sales")
```

A registered dataset is a typed, columnar `Dataset` rendered by a lossless,
header-once `DataEncoder` so a table costs a fraction of the tokens that pasting
JSON would, and `app.table_evidence(...)` offers it to the compiler as scored,
citable `TableEvidence`. See
[Tabular evidence and the compact data encoder](../concepts/tabular-evidence.md).

## Profile, sample, and screen before you spend tokens

```python
app.profile_dataset("sales")            # fixed-size, bounded-memory column profile
app.sample_dataset("sales", n=1000)     # reservoir / stratified sample
app.fit_dataset("sales", max_tokens=2000)  # faithful fit under a token budget
app.screen_data("sales")                # deterministic quality / PII screening
```

`profile_dataset`, `sample_dataset`, and `fit_dataset` make a table far larger
than the window into bounded, faithful evidence whose size is invariant to the
row count, and `screen_data` runs the deterministic
[quality rails](../concepts/dataset-profiling.md).

## Ask a question — governed text-to-query

```python
result = app.query_data("total qty by region", table="sales")
result.rows           # the answer
result.cite_refs      # e.g. ["sales#r0!qty", ...] — the exact source cells
result.verify(app.data_catalog())   # re-executes offline; catches a tampered source
```

Every generated query is grounded to the registered schema, **structurally
verified read-only** (a write, DDL, stacked statement, or injection is refused
before it runs), cost-bounded, and carries cell-level provenance. See
[Governed text-to-query and cell-level provenance](../concepts/governed-text-to-query.md).

## Let an agent do multi-step analysis

```python
analysis = app.analyze_data("how does revenue break down by region?", table="sales")
analysis.narrative          # a cited analytical narrative
analysis.verify(app.data_catalog())   # every finding re-derives from the bytes
```

`analyze_data` plans, queries through the read-only query plane, inspects, and
drills into the dominant group, producing a narrative whose every finding points
at the exact cells it rests on. See
[Data-analysis agent and multi-step EDA](../concepts/data-analysis-agent.md).

## Chart the result — content-bound and data-bound

```python
chart = app.generate_chart(result, title="Qty by region")
chart.spec                  # a portable Vega-Lite spec (matplotlib PNG via vincio[charts])
chart.verify(app.data_catalog())   # re-derives from the exact source cells
```

A chart is **content-bound** (a C2PA credential bound to its rendered bytes) and
**data-bound** (a back-reference to the source cells). See
[Charts and cited analytical artifacts](../concepts/charts-and-cited-artifacts.md).

## Govern the numbers with a semantic layer

```python
from vincio.data import DerivedColumn, Dimension, Measure

layer = app.semantic_layer(
    "sales",
    derived=[DerivedColumn(name="revenue", expression="price * qty")],
    measures=[Measure(name="total_revenue", agg="sum", expression="revenue")],
    dimensions=[Dimension(name="region")],
)
metric = app.query_metric("total_revenue", by=["region"])
metric.verify(layer, app.data_catalog())   # proves the number is the governed one
```

A `SemanticLayer` defines measures, dimensions, and derived columns once so a
question maps to a **governed metric** computed the same way however it is
phrased; `app.metric_lineage(...)` resolves a metric to its base columns and
source. See
[The semantic layer and governed metrics](../concepts/semantic-layer-and-governed-metrics.md).

## Process data far larger than memory

```python
stream = app.stream_dataset("huge.csv")          # lazy, re-iterable, schema-bearing
totals = app.aggregate_stream(stream, group_by="region", agg={"qty": "sum"})
app.map_stream(stream, transform, at_scale=True) # per-chunk, on the BatchRunner
```

`RowStream` iterates a source larger than memory in bounded chunks, and
`aggregate_stream`'s working set tracks the number of groups, not rows. See
[Streaming and out-of-core bulk processing](../concepts/streaming-and-out-of-core.md).

## Thread the whole plane into one signed narrative

```python
eng = app.data_engagement(question="how does revenue break down by region?")
eng.register(rows, columns=["region", "product", "price", "qty"], name="sales")
eng.profile(); eng.sample(1000); eng.screen()
eng.query("total qty by region")
eng.analyze("how does qty break down by region?")
eng.cite(title="Revenue analysis")
narrative = eng.seal()
narrative.verify(app.contract_signer)        # the chain recomputes from the bytes
eng.verify(app.contract_signer, catalog=app.data_catalog())  # every finding data-bound
```

`app.data_engagement` threads register → profile → … → cite behind one governed,
audited call-path and seals it into a hash-chained, signed `DataNarrative` that
verifies offline and is **data-bound** — the analytics analogue of the cross-org
engagement. See [The data engagement](../concepts/data-engagement.md).

## Run it

Examples [13](../../examples/13_tabular_evidence.py) through
[20](../../examples/20_data_engagement.py) are heavily-commented, fully-offline
programs for each step above, ending with the engagement capstone.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: Tabular evidence](../concepts/tabular-evidence.md)
- [Concept: Profiling, sampling & quality rails](../concepts/dataset-profiling.md)
- [Concept: Governed text-to-query](../concepts/governed-text-to-query.md)
- [Concept: Data-analysis agent](../concepts/data-analysis-agent.md)
- [Concept: Charts & cited artifacts](../concepts/charts-and-cited-artifacts.md)
- [Concept: Streaming & out-of-core](../concepts/streaming-and-out-of-core.md)
- [Concept: Semantic layer & governed metrics](../concepts/semantic-layer-and-governed-metrics.md)
- [Concept: Data engagement (the analytics capstone)](../concepts/data-engagement.md)
- [Guide: Generate documents & media (`vincio.generation`)](generate-documents.md)
- [Guide: Performance & streaming](performance.md)
- [Example: 13_tabular_evidence.py](../../examples/13_tabular_evidence.py)
- [Example: 14_dataset_profiling.py](../../examples/14_dataset_profiling.py)
- [Example: 15_governed_text_to_query.py](../../examples/15_governed_text_to_query.py)
- [Example: 16_data_analysis_agent.py](../../examples/16_data_analysis_agent.py)
- [Example: 17_charts_cited_artifacts.py](../../examples/17_charts_cited_artifacts.py)
- [Example: 18_streaming_out_of_core.py](../../examples/18_streaming_out_of_core.py)
- [Example: 19_semantic_layer_governed_metrics.py](../../examples/19_semantic_layer_governed_metrics.py)
- [Example: 20_data_engagement.py](../../examples/20_data_engagement.py)
- [Concept: Context packets & long-horizon governance](../concepts/context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
