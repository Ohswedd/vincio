# Charts and cited analytical artifacts

An analytical answer is not finished when the number is computed. It ships as a
*deliverable*: a figure a reader can trust and a report whose every claim and
every figure is grounded. The charts rung turns a cited
[query result](governed-text-to-query.md) into an **analytical artifact** that
carries the two guarantees a deliverable needs — the same provenance a generated
image carries, plus the cell-level lineage a cited answer carries.

Everything here is deterministic, dependency-free, and offline. The default
renderer emits a portable [Vega-Lite](https://vega.github.io/vega-lite/) v5 JSON
spec — no drawing library — and the optional `MatplotlibRenderer` (behind the
`vincio[charts]` extra) rasterizes the same spec to a PNG.

## Two guarantees

A `Chart` is both **content-bound** and **data-bound**:

- **Content-bound.** The rendered bytes carry a C2PA-style
  [`ProvenanceManifest`](../../vincio/governance/transparency.py) bound to them by
  SHA-256 — exactly the credential a generated image or audio clip carries. A chart
  is *data-driven* media (the IPTC `dataDrivenMedia` digital-source-type,
  `is_synthetic=False`): a faithful, deterministic rendering of real values, not
  model-synthesized content, and the credential says so honestly.
- **Data-bound.** The chart back-references the **exact source cells** it was built
  from (the result's per-row `RowProvenance`), and `verify(catalog)` re-executes the
  source query against the content-hashed source, confirms the plotted figure is a
  faithful projection of that verified result, and confirms the credential still
  binds the bytes.

```python
from vincio.data import generate_chart, query_dataset, DataCatalog, Dataset

catalog = DataCatalog.of(Dataset.from_records(sales, name="sales"), name="sales")
result = query_dataset(
    "SELECT region, SUM(revenue) AS s FROM sales GROUP BY region ORDER BY s DESC",
    catalog,
)

chart = generate_chart(result, title="Revenue by region")
chart.to_vega_lite()          # a portable Vega-Lite v5 spec with the data embedded
chart.cite_refs()             # ['sales#r0!revenue', 'sales#r2!revenue', ...]
chart.content_bound()         # True — the credential binds the rendered bytes
chart.verify(catalog)         # True — re-derives the figure from the bytes
```

`generate_chart` infers the encoding from the result's schema when you do not pin
it: a dimension on the x axis, a measure on the y axis, a second dimension as the
color series. A temporal x axis defaults to a line; everything else to a bar. Pin
the mark with `type=` (`bar` / `line` / `point` / `area` / `arc`) and the channels
with `x=` / `y=` / `color=`.

## What `verify` catches

`Chart.verify(catalog)` returns `False` on any divergence, so a figure can never
silently misrepresent its data:

- an **edited spec or edited bytes** — the chart hash no longer recomputes;
- a **stripped or mismatched credential** — the manifest no longer binds the bytes;
- a **tampered source** — the source query no longer re-executes to the same result;
- a **figure whose plotted values** are not a faithful projection of that verified
  result.

A chart built from a bare `Dataset` (rather than a cited `QueryResult`) is
content-bound but states `RESULT` coverage — there is no query to re-execute, and
the coverage is always stated, never silently downgraded.

## Cited reports extend to figures

The [cited-report builder](../../vincio/generation/report.py) already makes a report
**per-claim entailed**: every claim is cited and the cited evidence supports it.
For an analytical deliverable it also makes the report **per-figure data-bound**: a
`Figure` embeds a chart or a table, gets a `[F1]`-style marker the narrative can
reference, is rendered into the document, and — when a catalog is supplied — is
verified to re-derive from its source.

```python
from vincio.generation import CitedReportBuilder, CitationContract, Figure

report = await CitedReportBuilder().build_report(
    "NA leads revenue [F1]; the full split is in the table [F2].",
    evidence=[],
    figures=[
        Figure.from_chart(chart, caption="Revenue by region"),
        Figure.from_table(result, caption="Revenue table"),
    ],
    catalog=catalog,
    contract=CitationContract(require_figure_binding=True),
)

report.coverage.figure_binding_rate     # 1.0 — every figure re-derives from its source
[(f.marker, f.kind, f.data_bound) for f in report.figures]
```

`CitationContract(require_figure_binding=True)` makes per-figure data binding a
gate: if a figure does not re-derive from its source (a tampered source, say), the
build raises `CitationValidationError`, the per-figure analogue of the per-claim
entailment contract.

## On the app

`app.generate_chart` resolves the registered catalog, runs a natural-language
question or SQL string through the governed query plane first when you pass one,
and audits the result (`chart_generate`):

```python
app.register_dataset(sales, name="sales")
chart = app.generate_chart("revenue by region", table="sales", title="Revenue by region")
chart.verify(app.data_catalog())        # True

report = app.cited_report(
    "Revenue concentrates in NA and APAC [F1].",
    figures=[Figure.from_chart(chart)],
    contract=CitationContract(require_figure_binding=True),
)
```

`app.cited_report` (and `acited_report`) take `figures=` and an optional `catalog=`
(defaulting to the app's registered datasets), so the deliverable is grounded
end-to-end without leaving the governed runtime.

## Held by VincioBench

The `data_plane.charts` family gates three SLOs: a chart is **data-bound** (it
re-derives from its source and a tampered source is caught), **figure-cited** (it
cites the exact source cells, aggregates included), and **content-bound** (the C2PA
credential binds the bytes and an edited byte stream is caught). See the
[SLO reference](../reference/slo.md).
