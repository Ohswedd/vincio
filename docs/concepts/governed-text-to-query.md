# Governed text-to-query and cell-level provenance

A data analyst's question over a table should not pour every row into the prompt.
It should become a *query* — verified **before** it runs, executed **where the
data lives**, and answered with a citation to the **exact cells** the answer rests
on. This is the analyst rung of the data plane: it builds on the typed, columnar
[`Dataset`](tabular-evidence.md) and the [profiling and sampling](dataset-profiling.md)
rung beneath it.

Everything here is deterministic, dependency-free, and offline — the default
engine is the Python standard library's `sqlite3`, a real SQL engine.

## The pipeline

`query_dataset` (and the app-surface `app.query_data`) run one pipeline: ground →
verify → cost-bound → execute → cite.

```python
from vincio.data import query_dataset, DataCatalog, Dataset

catalog = DataCatalog.of(Dataset.from_records(sales, name="sales"), name="sales")
result = query_dataset("total revenue by region", catalog)

result.rows                              # [["APAC", 1500.25], ["EU", 980.0], ["NA", 1500.5]]
result.value(0, "sum_revenue")           # one cell of the answer
result.cite_refs(2, "sum_revenue")       # ['sales#r0!revenue', 'sales#r2!revenue']
result.verify(catalog)                   # True — re-derives from the bytes
```

The question is grounded to a read-only `SELECT` over the catalog's schema (the
offline `HeuristicQueryPlanner` handles counts and single-column / group-by
aggregates; pass explicit SQL for anything else, or wire a model planner through
`app.query_data`). You can also pass SQL directly:

```python
result = query_dataset(
    "SELECT region, revenue FROM sales WHERE revenue > 1000", catalog
)
```

## Verified before it runs

Verification is **structural** — never gated on model output:

- **Schema-grounded.** A query may reference only registered tables and the
  columns they declare. An unknown table is refused immediately; an unknown column
  is refused at the dry-run compile, before any row is read.
- **Read-only by default.** `is_read_only_sql` / `assert_read_only_sql` accept only
  a single statement with a `SELECT` / `WITH` head and no write, DDL, or stacked
  statement — checked after stripping comments and string literals, so a write
  keyword hidden in a quoted value or a comment cannot slip the guard. A breach
  raises `UnsafeQueryError`.
- **Injection-screened.** A natural-language question is run through the same
  injection detector the text rails use before it becomes a query.
- **Cost-bounded.** The query is compiled and its plan inspected without fetching;
  a `max_rows` ceiling bounds the result.

```python
from vincio.core.errors import UnsafeQueryError

for attempt in ["DROP TABLE sales", "UPDATE sales SET revenue = 0",
                "SELECT 1; DROP TABLE sales"]:
    try:
        query_dataset(attempt, catalog)
    except UnsafeQueryError:
        ...  # refused structurally, before execution
```

The same guarantee is available as a `ToolContract`: `build_query_contract()`
refuses a non-read-only query as a pre-condition and bounds the row count as a
post-condition, so a `query_data` tool **structurally** refuses a write when it
rides the permissioned, approval-gated, audited tool runtime.

## Executed where the data lives

The verified query runs on a pluggable `QueryEngine`. The default
`InProcessSqlEngine` builds an in-memory `sqlite3` database from the catalog and
executes the query **read-only**: it sets `PRAGMA query_only` and installs an
authorizer that denies every non-read action, so a write or DDL that somehow
passed the screen is still refused by the engine — *defense in depth beneath the
screen*. Rows go to the engine, not the prompt; only the result and the cells it
cites reach the model.

A pushdown engine can run the same verified SQL against a live source (DuckDB,
SQL, BigQuery, Snowflake) instead — the offline in-process engine is the default.

## Cell-level provenance

The answer is a schema-bearing `QueryResult` that cites the exact source cells it
rests on. A `CellCitation` renders the stable locator `table#r<row>!<column>`, and
`RowProvenance` records, per result row, the source cells it was derived from:

```python
result.coverage                          # LineageCoverage.CELL
result.citations(2, "sum_revenue")       # [CellCitation(table='sales', row=0, ...), ...]
result.cite_refs(2, "sum_revenue")       # ['sales#r0!revenue', 'sales#r2!revenue']
```

Lineage is **cell-exact** for the analyst's common shapes — a single-table
projection / filter (each result row maps to one source row) and a single-table
group-by aggregation (each result row maps to the source rows of its group). For a
shape outside that grammar (a multi-table join, a nested subquery), the
`LineageCoverage` is honestly reported as `result` rather than silently
downgraded — the result still re-derives from the hashed source.

`verify()` is the offline guarantee, the analytics analogue of a cited report's
per-claim entailment: it re-executes the query against a catalog, re-derives the
result and every cited cell from the bytes, and returns `False` on any divergence
— a tampered result, a tampered source, or a flipped cell.

```python
result.verify(catalog)                   # True
result.verify(tampered_catalog)          # False — a changed source cell is caught
```

## How cell-exact lineage is derived

The engine does not guess which rows an answer came from — it makes SQLite tell
it, by threading the table's implicit `rowid` (its 1-based insertion order):

- **Projection / filter.** The query is transparently rebuilt with the source
  rowid as a leading column (`SELECT sales.rowid AS __vrow__, …`), so every result
  row carries the exact source row it came from. Each output column then cites the
  source columns its expression references — a bare projection cites one cell, a
  derived blend (`revenue + tax`) cites both operands, a constant cites none.
- **Group-by aggregation.** A separate **witness query**
  (`SELECT <keys>, group_concat(rowid) … GROUP BY <keys>`) maps each group key to
  the source rowids that fell into it, and each result row is attributed to the
  rows of its group. The witness deliberately omits `HAVING` / `ORDER BY` /
  `LIMIT`: attribution is looked up by the group keys that actually appear in the
  result, so a filtered-out group is simply never asked about.

When the shape falls outside that grammar — a join, a subquery, a CTE, more than
one table, or a user column that shadows `rowid` / `oid` / `_rowid_` — the engine
degrades to `LineageCoverage.RESULT` rather than cite the wrong rows. The result
still re-derives from the content-hashed source on `verify()`; only the per-cell
attribution is withheld, and the coverage says so.

## The dataframe-op dialect

The same pipeline runs over a whitelisted, `eval`-free dataframe-op pipeline
(`select` / `filter` / `derive` / `rename`, reusing `vincio.verify.ProgramOp`).
These ops are read-only by construction and yield **exact per-cell lineage** with
no model in the loop — the deterministic path for a programmatic transform:

```python
from vincio.verify import ProgramOp

result = query_dataset(
    "NA line totals", catalog, dialect="dataframe",
    ops=[
        ProgramOp(op="derive", field="line_total", expr="revenue * units"),
        ProgramOp(op="filter", field="region", op_symbol="==", value="NA"),
        ProgramOp(op="select", fields=["product", "line_total"]),
    ],
)
result.cite_refs(0)   # the derived total cites both 'revenue' and 'units' of its source row
```

## The app surface

```python
app.register_dataset(sales, name="sales")          # into the app's DataCatalog
result = app.query_data("total revenue by region", table="sales")
result.verify(app.data_catalog())
```

Every query lands on the shared, hash-chained audit log: `data_register` for a
registration, `data_query` (with the lineage coverage and result hash) for a query,
and a `deny` entry pinpointing a refused unsafe query.

## Gotchas

- The offline `HeuristicQueryPlanner` grounds only the canonical shapes — counts,
  single-column aggregates, and single group-by aggregates. A question it cannot
  ground confidently returns `None` (raising `QueryError`) rather than guess: pass
  explicit SQL, or route through `app.query_data` where a model planner can propose
  SQL that is still verified read-only before it runs.
- The read-only guard is a **structural** check, not a parser: it strips comments,
  string literals, and quoted identifiers before scanning, so a write keyword
  inside a quoted value or a column named `"update"` is not misread — but it also
  refuses anything that is not a single `SELECT` / `WITH` statement, including a
  stacked `SELECT 1; DROP …`.
- `verify()` returns `False` on *any* divergence — a tampered result, a tampered
  source cell, or a source-hash mismatch — so treat it as a boolean gate, not a
  diff; use `cite_refs` / `citations` to see which cells an answer rests on.
- A result over `max_rows` raises rather than truncating silently, so a cited
  answer is never a partial one; tighten the query or raise the ceiling.

## What it is not

Governed text-to-query is the analyst rung of the data plane: it answers a
question over a registered dataset, read-only and cell-cited. It is **not** a
hosted query engine, a managed warehouse, or a notebook service — the in-process
`sqlite3` engine is the offline-first default and a pushdown engine runs the same
verified SQL where the data lives. The [multi-step data-analysis agent](data-analysis-agent.md)
builds on this rung, and [charts](charts-and-cited-artifacts.md) build on that.
Nothing here is gated on model output: grounding, the read-only guard, cost bounds,
and verification are enforced in code, deterministically and offline.

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
