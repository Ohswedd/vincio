# The semantic layer and governed metrics

An analyst's "revenue" is `price × qty` summed. If every question re-derives it,
two questions compute it two ways and the numbers drift. The semantic layer
defines the analytical vocabulary **once** — measures, dimensions, and derived
columns — so a natural-language question maps to a *governed metric* rather than a
raw column, and that metric compiles to SQL **one way everywhere**: the
single-source discipline the governance refactors already use, brought to the data
plane.

This rung reuses the [governed query plane](governed-text-to-query.md) rather than
growing a parallel stack: a metric compiles to a single read-only `SELECT` that is
re-screened read-only, dry-run-grounded, executed where the data lives, and
returned **cell-level cited** and offline-verifiable like any other query.
Everything here is deterministic, dependency-free, and offline.

## Define the vocabulary once

A [`SemanticLayer`](../../vincio/data/semantic.py) is built over one registered
table from three kinds of definition:

- a [`DerivedColumn`](../../vincio/data/semantic.py) — a row-level calculation
  declared once (`revenue = price * qty`) that measures and dimensions reference by
  name, and that other derived columns can compose with;
- a [`Dimension`](../../vincio/data/semantic.py) — a groupable attribute (a column,
  or an expression like a month bucket);
- a [`Measure`](../../vincio/data/semantic.py) — the **governed metric**: a `SUM` /
  `AVG` / `MIN` / `MAX` / `COUNT` / `COUNT_DISTINCT` over a column or derived
  column, an optional row filter, or a **ratio** of two other measures.

```python
from vincio import ContextApp
from vincio.data import DerivedColumn, Dimension, Measure

app = ContextApp(name="analytics")
app.register_dataset(rows, columns=["region", "product", "price", "qty"],
                     name="sales", source="crm-export")

layer = app.semantic_layer(
    "sales",
    derived=[DerivedColumn(name="revenue", expression="price * qty", unit="USD")],
    measures=[
        Measure(name="total_revenue", agg="sum", expression="revenue",
                synonyms=["revenue", "sales"]),
        Measure(name="orders", agg="count"),
        Measure(name="avg_order_value", numerator="total_revenue", denominator="orders"),
    ],
    dimensions=[Dimension(name="region", synonyms=["geography"])],
)
```

Names share one namespace and must be simple identifiers; a duplicate name, a
derived-column or ratio-measure cycle, a measure that declares neither an
aggregation nor a complete ratio, or an expression that could break out of its
clause (a stacked statement, unbalanced parentheses) is refused with a
`SemanticLayerError`. When the table is registered, `app.semantic_layer` also
dry-run-grounds every metric and dimension against it, so a measure that references
an unknown column is caught at definition, not at query time.

## A question maps to a governed metric

`app.query_metric` (and the free `query_metric` / `SemanticLayer.query`) resolves a
metric name, a list of names, a `MetricQuery`, or a **natural-language question** to
the governed measure, compiles it to one canonical `SELECT`, and runs it through
the governed query plane.

```python
result = app.query_metric("total revenue by region")
result.value(0)                          # the governed number
result.cite_refs(0)                      # the exact source cells it rests on
result.verify(layer, app.data_catalog()) # governed + re-derives from the bytes
```

A question grounds to a metric (or dimension) when its name, its space-separated
form, or one of its `synonyms` appears — the governed analogue of the offline
[`HeuristicQueryPlanner`](governed-text-to-query.md); a question that names no
defined metric grounds to nothing rather than guessing. The natural-language
question is **injection-screened** with the same detector the text rails use before
it grounds, and the compiled SQL is re-screened read-only, so a metric can never
smuggle a write past the read-only guard.

The whole point is the single source of truth: two phrasings of the same question
compile to byte-identical SQL and return equal results.

```python
a = app.query_metric("total revenue by region")
b = app.query_metric("sales by geography")
assert a.sql == b.sql and a.rows == b.rows
```

A ratio measure compiles with a zero-safe denominator
(`CAST(num AS REAL) / NULLIF(den, 0)`), so a division by zero yields `NULL` rather
than crashing.

## The result is provably the governed metric

A [`MetricResult`](../../vincio/data/semantic.py) wraps the cell-cited
[`QueryResult`](governed-text-to-query.md) with the `MetricQuery` it answered and
the layer's content hash. `MetricResult.verify` proves three things from the bytes
alone:

1. the layer's definitions still hash to the recorded `layer_hash` (the metric was
   not redefined);
2. the SQL that ran equals the layer's **canonical** compilation of the metric — so
   an ad-hoc query passed off as the governed metric is rejected;
3. the underlying query result re-executes and re-derives every cited cell from the
   hashed source — so a tampered source is caught.

```python
from vincio.data import query_dataset, MetricResult, MetricQuery

adhoc = query_dataset("SELECT region, SUM(price) AS total_revenue FROM sales GROUP BY region",
                      app.data_catalog())
forged = MetricResult(spec=MetricQuery(metrics=["total_revenue"], dimensions=["region"]),
                      result=adhoc, layer_hash=layer.digest())
assert forged.verify(layer, app.data_catalog()) is False   # not the governed compilation
```

## Column-level lineage reaches the dataset plane

A metric's provenance is a column-level fact. `app.metric_lineage` resolves the
derived-column graph and any ratio references down to the **base columns** the
metric ultimately rests on, plus the source the dataset was ingested under.

```python
lin = app.metric_lineage("total_revenue")
lin.base_columns   # ['price', 'qty']
lin.derived_via    # ['revenue']
lin.source         # 'crm-export'
```

That same link feeds the existing right-to-erasure machinery. When a dataset is
registered, it is recorded in the [lineage index](../../vincio/governance/lineage.py)
under its source with its columns; an `app.erase_source` sweep now reaches the
**dataset plane**, dropping the registered dataset (and any semantic layer over it)
alongside the source's documents, memories, and generated artifacts — and recording
exactly which dataset was removed in the signed, content-bound `ErasureProof`.

```python
result = app.erase_source("crm-export")
result.datasets_removed                      # 1
result.proof.removed_ids["datasets"]         # ['sales']
"sales" not in app.data_catalog().names      # True
```

So a metric's provenance and a subject's erasure both reach into the dataset plane,
on the same mechanical, auditable machinery a document's lineage and erasure already
ride.

## Guarantees

DataPlaneBench holds the rung under CI-gated budgets and published SLOs: a
**governed one-way** check (a metric compiles to one canonical SQL and returns the
same number however the question is phrased), a **metric-verifiable** check (the
result re-derives from the bytes while a forged ad-hoc result and a tampered source
both fail), and a **lineage-reaches-the-dataset-plane** check (a metric's lineage
resolves to its base columns and source, and an erasure sweep removes the dataset
and records it in the proof). Because a metric compiles to the same SQL a
hand-written query would, there is no new performance-sensitive path — the cost is
identical to the governed query it reuses.

## See also

- [Governed text-to-query and cell-level provenance](governed-text-to-query.md) — the query plane a metric compiles to and the cell-level lineage it inherits.
- [Tabular evidence and the compact encoder](tabular-evidence.md) — the typed, columnar `Dataset` a layer is defined over.
- [The data-analysis agent](data-analysis-agent.md) — the multi-step EDA that explores the same governed query plane.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: Analyze data](../guides/analyze-data.md)
- [Example: 19_semantic_layer_governed_metrics.py](../../examples/19_semantic_layer_governed_metrics.py)
- [Concept: Context packets & long-horizon governance](context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
