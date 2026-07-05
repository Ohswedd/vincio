# Data-analysis agent and multi-step EDA

A real analytical question over a table is rarely answered by one query. An
analyst *explores*: they size the table up, summarize the measures, break a
measure down by a dimension, notice where it concentrates, and drill into the
part that dominates — then write up what they found, pointing at the figures
behind each statement. The data-analysis agent runs exactly that loop, composed
from the data plane's existing organs so every finding is grounded and cited *by
construction*. It is the analyst-agent rung of the data plane, built on the typed,
columnar [`Dataset`](tabular-evidence.md), the [profiling and sampling](dataset-profiling.md)
rung, and the [governed text-to-query](governed-text-to-query.md) rung beneath it.

Everything here is deterministic, dependency-free, and offline — the default
engine is the Python standard library's `sqlite3`, a real SQL engine, and the
"verifier" is not a model but the query plane's offline re-execution.

## The loop

`analyze_dataset` (and the app-surface `app.analyze_data`) run one bounded loop:
**plan → query → inspect → refine → synthesize**.

```python
from vincio.data import analyze_dataset, DataCatalog, Dataset

catalog = DataCatalog.of(Dataset.from_records(sales, name="sales"), name="sales")
analysis = analyze_dataset("how does revenue break down by region?", catalog)

print(analysis.narrative)        # the cited narrative, one finding per line
analysis.coverage                # LineageCoverage.CELL / RESULT (always stated)
analysis.cite_refs()             # every source cell the narrative rests on
analysis.verify(catalog)         # True — re-derives the whole analysis from the bytes
```

- **plan.** A deterministic plan is grounded against the table's schema: an
  overview (the table's shape), the objective itself (grounded by the same offline
  `HeuristicQueryPlanner` text-to-query uses), each measure's extreme and total, and
  a measure-by-dimension breakdown.
- **query.** Every step runs through the governed query plane: schema-grounded,
  **read-only-verified**, cost-bounded, executed where the data lives — never
  materialized into a prompt.
- **inspect.** Each result is inspected and turned into a finding that cites the
  exact source cells it rests on.
- **refine.** While refinement budget remains, the group that dominates a breakdown
  is drilled into for a narrower finding.
- **synthesize.** The findings become a cited analytical narrative that re-derives
  from the bytes.

## How the plan is built

The plan is generated deterministically from the table's schema, not by a model —
every query is a template filled from the columns:

- **Measures and dimensions are read off the schema.** A measure is a numeric
  column; a dimension is a low-cardinality non-measure column, ordered
  most-discriminating-first (fewest distinct groups). The objective itself is
  grounded by the same `HeuristicQueryPlanner` text-to-query uses, and marked the
  `primary` step.
- **Each measure gets an extreme and a total.** The extreme is a cell-cited
  `ORDER BY measure DESC LIMIT 1` projection (so the finding can name *where* the
  peak sits); the total is a `SUM` — honestly a result-level aggregate.
- **Each dimension gets a breakdown.** `SUM(measure) GROUP BY dimension ORDER BY …
  DESC`, cell-cited to the rows of each group, capped at `max_breakdowns`.
- **The dominant group is drilled.** While refinement budget remains, the top
  group of a breakdown is re-queried against a *second* dimension
  (`WHERE dimension = <dominant> GROUP BY <secondary>`) — the one narrowing move an
  analyst makes by hand.

Every generated query flows through `QueryPlan.for_sql` — grounded, read-only
verified, dry-run compiled — exactly as a hand-written one does, and a step that
cannot ground or execute is dropped (its query returns `None`) rather than sinking
the run. The narrative is then assembled by concatenating the executed steps'
cited findings in order, so a claim is in the narrative *only if* its query ran
and verified.

## Bounded by construction

The exploration is bounded by an explicit `AnalysisBudget` — there is no
open-ended search:

```python
from vincio.data import AnalysisBudget

analysis = analyze_dataset(
    "revenue by region", catalog,
    budget=AnalysisBudget(max_steps=5, max_refinements=1, max_rows=10_000),
)
```

`max_steps` caps the total number of queries (the plan is truncated to fit),
`max_refinements` caps how many dominant groups are drilled into, and `max_rows`
bounds each query's result. The agent reuses the *existing* governed query plane
for grounding and verification rather than growing a parallel search stack.

## The cited narrative

The result is an `AnalysisResult`: the objective, the executed `AnalysisStep`s
(each a verified, cell-cited query), the assembled narrative, and a content hash
binding them. Each step carries its finding and the exact source cells it rests
on:

```python
for step in analysis.steps:
    print(step.kind, step.finding, step.cite_refs)

analysis.primary_step()   # the step that answers the objective directly, if it grounded
analysis.answer()         # the headline answer (a scalar, or the result rows)
```

A finding produced by a single-table projection / filter or group-by aggregation
carries **cell-exact** lineage (`sales#r0!revenue`); a finding from a result-level
aggregate (a `COUNT(*)` or a whole-column `SUM`) is honestly reported at the
`result` level rather than silently downgraded. The narrative's coverage is the
weakest across its steps and is always stated.

## Verified, not asserted

The verifier is not a model — it is the query plane's deterministic, offline
re-execution, the analytics analogue of a cited report's per-claim entailment.
`AnalysisResult.verify` re-executes every step's query against a catalog,
re-derives the narrative and every cited cell from the bytes, and returns `False`
on any divergence — a tampered source, a flipped cell, or a tampered narrative:

```python
analysis.verify(catalog)            # True
analysis.verify(tampered_catalog)   # False — a changed source cell is caught
analysis.narrative += " (edited)"
analysis.verify(catalog)            # False — the narrative no longer matches its steps
```

Because the narrative is assembled deterministically from the verified findings,
it is grounded *by construction*: a finding can only enter the narrative if its
underlying query ran and verified.

## Executed where the data lives

Each step runs on a pluggable `QueryEngine`. The default `InProcessSqlEngine` runs
on the standard-library `sqlite3` engine, opened read-only, and derives cell-exact
lineage offline. For execution at scale, `DuckDbQueryEngine` (behind the
`vincio[data]` extra) runs the *same verified SQL* on DuckDB:

```python
from vincio.data import DuckDbQueryEngine

analysis = analyze_dataset("total revenue by region", catalog,
                           engine=DuckDbQueryEngine())
```

The accelerator reports `result`-level lineage (the result still re-derives from
the content-hashed source on `verify()`); the offline `sqlite3` engine remains the
path that derives per-cell citations. Coverage is always stated, never silently
downgraded.

## The app surface

```python
app.register_dataset(sales, name="sales")
analysis = app.analyze_data("how does revenue break down by region?", table="sales")
analysis.verify(app.data_catalog())
app.pending_evidence.append(analysis.to_evidence_item())   # carry it as cited evidence
```

`app.analyze_data` resolves the catalog (the app's registered datasets or a
one-shot `dataset=`), screens the objective with the same injection detector the
text rails use (a refused objective raises `UnsafeQueryError`), and audits the run
on the shared, hash-chained log (`data_analysis`, with the step count, cited-finding
count, lineage coverage, and result hash; a refused objective records a `deny`).
When a model is configured, the agent may ask it for additional analytical
follow-up questions — each still grounded and verified by the query plane, so the
model never produces a query that bypasses the screen; offline, or whenever the
model returns nothing groundable, the agent is byte-for-byte the deterministic
core.

## Gotchas

- The analysis's coverage is the **weakest across its steps**: one result-level
  step (a `SUM` total, a `COUNT(*)`) makes the whole narrative's coverage
  `RESULT`, even though the extreme and breakdown steps are cell-exact. Read each
  `step.coverage` for the per-finding truth.
- `max_steps` truncates the *plan* (the overview and the objective come first), so
  a tight budget keeps the sizing-up and the direct answer and drops the deeper
  breakdowns — it never runs an unbounded search.
- `propose_followups` lets a configured model suggest extra questions, but each is
  still grounded by the offline planner and verified by the query plane; a junk or
  empty model response degrades to no extra steps, and any provider failure is
  swallowed to `[]`.
- `DuckDbQueryEngine` accelerates execution but reports `RESULT`-level lineage;
  the offline `sqlite3` engine is the path that derives per-cell citations.

## What it is not

The data-analysis agent is the multi-step EDA rung of the data plane: it answers a
question over a registered dataset with a bounded, cited, offline-verifiable
analysis. It is **not** a notebook service, a hosted analytics dashboard, or an
unbounded autonomous agent — the loop is bounded by an explicit budget, the engine
is the offline `sqlite3` engine by default, and nothing is gated on model output:
grounding, the read-only guard, the budget, and verification are enforced in code,
deterministically and offline. [Charts and cited analytical
artifacts](charts-and-cited-artifacts.md) turn a cited result into a deliverable.

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
