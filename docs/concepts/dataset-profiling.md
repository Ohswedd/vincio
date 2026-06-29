# Dataset profiling, sampling, and quality rails

A table of ten million rows cannot enter a prompt, and truncating it to the first
thousand throws away everything the rest would have said. The data plane instead
**represents a dataset far larger than the window faithfully, under a fixed token
budget**: a deterministic column *profile* over every row, a *representative
sample* that stands in for the whole, and *data-quality rails* that screen the
input on the same deterministic path PII and injection detection ride for text.

Everything here is deterministic, dependency-free, and offline. It builds on the
typed, columnar [`Dataset`](tabular-evidence.md).

## Profiling

`profile_dataset` computes a fixed-size summary of a dataset in a single,
bounded-memory pass — its footprint depends on the number of columns, not the
number of rows, so a profile of ten million rows is the same size as a profile of
ten:

```python
from vincio.data import profile_dataset

profile = profile_dataset(dataset)            # or app.profile_dataset(records)
revenue = profile.column("revenue")
revenue.min, revenue.max, revenue.mean        # exact over every value
revenue.null_rate                             # exact
revenue.percentiles["p95"]                    # estimated from a bounded reservoir
revenue.histogram                             # population-scaled distribution
profile.column("region").distinct             # cardinality (exact up to a cap)
profile.column("region").top_values           # most frequent values
```

Exact figures — count, null rate, extrema, mean, standard deviation — accumulate
in constant space. Percentiles and histograms are estimated from a fixed-size
reservoir once a column grows past it (`column.estimated` flags this; small
columns are exact). Cardinality is exact up to a cap and a lower bound beyond it
(`distinct_is_lower_bound`).

The profile is itself first-class evidence: `profile.to_evidence_item()` renders
it as a compact stats table the context compiler scores, budgets, orders, and
cites. `profile_stream(rows, schema)` profiles a row iterator without
materializing it — the path for a source larger than memory.

## Representative sampling

A first-N cutoff is not a sample — it is the rows the source happened to return
first. `sample_dataset` draws a representative one instead:

```python
from vincio.data import sample_dataset, reservoir_sample

# Uniform, single pass, O(k) memory — order-independent, deterministic.
sample = sample_dataset(dataset, 1000, method="reservoir", seed=0)

# Proportional across a key column — a rare category keeps representation.
sample = sample_dataset(dataset, 1000, method="stratified", by="region", seed=0)

# Sample any iterable far larger than memory directly.
rows = reservoir_sample(huge_iterator, 1000, seed=0)
```

A sampled `Dataset` records how it was drawn in `metadata["sample"]`, so a
downstream reader knows it stands in for a larger whole. `reservoir`,
`stratified`, `systematic`, and `head` are the methods (see `SampleMethod`).

The SQL-family connectors take an opt-in `sample=` that reservoir-samples the
whole result set in one bounded pass instead of truncating at `max_rows` — a
representative sample replaces the order-biased prefix, with the default behavior
unchanged.

## Fitting a table into the window

`fit_to_window` combines the two into the headline guarantee — a dataset far
larger than the window represented under a fixed token budget:

```python
from vincio.data import fit_to_window, fit_stream

fit = fit_to_window(dataset, max_tokens=2000, method="reservoir", seed=0)
fit.within_budget                 # True — the combined encoding fits the budget
fit.to_evidence_items()           # [profile evidence, representative sample evidence]

# Single bounded pass over a source larger than memory.
fit = fit_stream(rows, schema, max_tokens=2000, seed=0)
```

The result is a full-fidelity column profile (computed over **all** rows) plus a
representative sample sized to whatever budget the profile leaves. Because the
profile is fixed-size and the sample is budget-bound, the representation stays
within the budget whether the table has ten thousand rows or ten million — and
`app.fit_dataset` does the same from the app surface, yielding cited table
evidence ready for `app.pending_evidence`.

## Data-quality rails

`DataQualityRails` screen a tabular input deterministically — no model judgment,
every finding explainable — for the failure modes structured data has:

```python
from vincio.data import DataQualityRails, ColumnConstraint, DataType

rails = DataQualityRails(
    [
        ColumnConstraint(column="id", dtype=DataType.INT, unique=True),
        ColumnConstraint(column="region", allowed_values=["NA", "EU", "APAC"]),
        ColumnConstraint(column="amount", min_value=0, max_value=10_000),
        ColumnConstraint(column="email", detectors=["pii"]),   # PII in a cell
    ],
    detect_anomalies=True,                                     # robust (MAD) outliers
)
report = rails.check(dataset)
report.allowed                 # False if any blocking rule fired
report.violations              # column, rule, count, examples
report.raise_for_status()      # raises DataQualityError when blocked
```

The constraints cover **schema violations** (wrong type, null in a non-nullable
column, a null rate above a ceiling), **constraint breaks** (range, allowed set,
required pattern, uniqueness, monotonicity), and **anomalies** (numeric outliers
via a robust median/MAD z-score). The very same security detectors ride this
path: a constraint may run the PII, secret, or injection detector over a column's
string cells, so a leaked email in a data table is caught exactly as it would be
in a prompt.

`DataQualityRails.from_dataset(ds)` derives a baseline that enforces the
dataset's own declared schema with zero configuration. `app.screen_data` runs the
rails and lands the decision on the shared, hash-chained audit log
(`data_quality`) like any other rail decision, optionally raising on a blocking
finding.

## What it is not

Profiling, sampling, fit-in-window, and quality rails are the representation rung
of the data plane. The analyst rung —
[governed text-to-query and cell-level provenance](governed-text-to-query.md) —
ships next on top of it; the data-analysis agent and charts are later rungs (see
the [roadmap](../../ROADMAP.md)). Nothing here calls a database or a network:
`vincio.data` is deterministic, dependency-free, and offline.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: Analyze data](../guides/analyze-data.md)
- [Example: 14_dataset_profiling.py](../../examples/14_dataset_profiling.py)
- [Concept: Context packets & long-horizon governance](context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
