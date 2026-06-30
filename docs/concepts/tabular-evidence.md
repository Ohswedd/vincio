# Tabular evidence and the compact data encoder

Structured data is first-class evidence in Vincio — schema-bearing and columnar,
never flattened to prose or dumped as `json.dumps`. The `vincio.data` subpackage
gives you a typed `Dataset`, a deterministic `DataEncoder` that renders it
header-once in a compact, token-oriented form, and `TableEvidence` that drops a
dataset straight into the context compiler the way text, images, and tables
already flow.

This is the token-efficiency foundation of the data plane: the schema, types,
units, and null-handling are declared **once** in a one-line header, and the
cells follow as delimited rows, instead of repeating the keys and punctuation on
every row.

## A typed, columnar dataset

A `Dataset` carries a `DataSchema` — a `ColumnSchema` per column with a name, a
`DataType`, an optional unit, and a nullability flag — over cells stored
column-major. Build one from rows, records, columns, a legacy `TableData`, or a
compact encoding:

```python
from vincio.data import Dataset, ColumnSchema, DataType

ds = Dataset.from_records(
    [
        {"region": "NA", "revenue": 1200.50, "units": 5},
        {"region": "EU", "revenue": 980.00, "units": None},
    ],
    name="sales",
)

ds.column_names   # ['region', 'revenue', 'units']
ds.dtypes         # ['str', 'float', 'int']
ds.row_count      # 2
ds.column("region")          # ['NA', 'EU']
ds.records()                 # [{'region': 'NA', ...}, ...]
ds.head(1)                   # a new Dataset with the first row
```

Types are inferred from the values; pass an explicit schema when you want to fix
a column's type or attach a unit:

```python
ds = Dataset.from_rows(
    [["NA", 1200.50], ["EU", 980.00]],
    [ColumnSchema(name="region"),
     ColumnSchema(name="revenue", dtype=DataType.FLOAT, unit="USD")],
    name="sales",
)
```

## The compact encoding

`DataEncoder` (and `Dataset.encode()`) render a dataset header-once:

```python
print(ds.encode())
# sales{#2,region:str,revenue:float USD}
# NA,1200.5
# EU,980.0
```

The header inside the braces declares the row count (`#2`), then `name:type` per
column, a trailing `?` for a nullable column, and a space-separated unit after
the type (`float USD`). Each following line is one row, comma-delimited. A
**null** cell is an empty field; an **empty string** is the quoted `""`, so the
two are distinguishable.

The encoding is **lossless** — `decode` (or `Dataset.from_encoding`) reconstructs
the columns, types, and cells from the bytes alone, including the null/empty-string
distinction, leading zeros, and embedded delimiters, quotes, and newlines:

```python
from vincio.data import DataEncoder

encoder = DataEncoder()
text = encoder.encode(ds)
back = encoder.decode(text)
assert back.rows() == ds.rows()
```

The encoder also replaces `json.dumps` for arbitrary JSON-like values with a
compact, token-oriented rendering (`encoder.encode_value(obj)`), and reports the
columnar-accurate token cost of an encoding — the count of the tokens the model
actually receives, which fixes the generic per-cell heuristic:

```python
ds.token_cost()                       # exact tokens for this dataset
encoder.token_cost(ds, model="gpt-4o")
```

`DataEncoder` options control the rendering: `delimiter`, whether to include the
name, row count, types, or units, and `exemplars=k` to inject up to *k* example
values per column once as a leading `# ...` description line (`Dataset.exemplars(k)`
returns them directly). The description line is decode-safe — it never affects
the round-trip.

## Datasets as context evidence

`TableEvidence` projects a dataset into the evidence the context compiler scores,
deduplicates, budgets, orders, and cites. Build it directly, with
`Dataset.to_evidence(...)`, or with `app.table_evidence(...)`. The compiler
accepts a `Dataset` or a `TableEvidence` in its `evidence` list and projects it
to `modality="table"` evidence whose scorable text and prompt rendering are the
compact encoding and whose token cost is columnar-accurate. To attach it to a
full run, add it to `app.pending_evidence`:

```python
evidence = app.table_evidence(
    [{"region": "NA", "revenue": 1200.50}, {"region": "EU", "revenue": 980.00}],
    name="sales",
    caption="Quarterly sales by region",
)
app.pending_evidence.append(evidence.to_evidence_item())
result = await app.arun("Which region had the most revenue?")
```

`app.table_evidence` accepts a list of records, a list of rows (with `columns=`
or `schema=`), an existing `Dataset`, or a legacy `TableData`.

## On the path to the prompt

The compact encoding is also what the document path now uses: `TableData.to_text`
renders the schema-once form instead of a pipe-joined table, and `structure_data`
encodes nested JSON-like values compactly instead of `json.dumps(indent=2)`. A
table extracted from a CSV, an HTML page, or a spreadsheet therefore reaches the
model token-cheap and with its column types declared, while its string cells are
preserved exactly.

## What it is not

This is the encoding and evidence foundation. Profiling, representative sampling,
fit-in-window, and data-quality rails build directly on it — see
[Dataset profiling, sampling, and quality rails](dataset-profiling.md).
Text-to-query, the data-analysis agent, charts, and cell-level provenance are
later rungs of the data plane (see the [roadmap](../../ROADMAP.md)). Nothing here
calls a database or a network: `vincio.data` is deterministic, dependency-free,
and offline.

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
