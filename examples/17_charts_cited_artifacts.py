"""Charts & cited analytical artifacts.

An analytical answer is not finished when the number is computed — it ships as a
*deliverable*: a figure a reader can trust and a report whose every claim and
every figure is grounded. This program walks the data plane's artifact rung,
fully offline on the standard-library SQL engine and the dependency-free
Vega-Lite renderer:

  * `generate_chart` / `app.generate_chart` — turn a cited query result into a
    spec-driven chart (a portable Vega-Lite v5 spec by default);
  * content-bound — the chart carries a C2PA *data-driven* credential bound to
    its rendered bytes, exactly the provenance a generated image carries;
  * data-bound — the chart back-references the **exact source cells** it was
    built from (`sales#r0!revenue`), and `verify(catalog)` re-derives it offline;
  * the cited-report builder extends to figures — a `Figure` embeds a chart or a
    table into a report that is per-claim entailed *and* per-figure data-bound.

Everything below is opt-in and additive; none of it touches a database, a
network, or a drawing library. (`MatplotlibRenderer`, behind the `vincio[charts]`
extra, rasterizes the same spec to a PNG.)
"""

from __future__ import annotations

import asyncio

from _shared import example_provider

from vincio import CitedReportBuilder, ContextApp, Figure, generate_chart
from vincio.data import DataCatalog, Dataset, query_dataset
from vincio.generation.report import CitationContract


def banner(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


SALES = [
    {"region": "NA", "product": "alpha", "revenue": 1200.5},
    {"region": "EU", "product": "alpha", "revenue": 980.0},
    {"region": "NA", "product": "beta", "revenue": 300.0},
    {"region": "APAC", "product": "beta", "revenue": 1500.25},
]


def _catalog() -> DataCatalog:
    return DataCatalog.of(Dataset.from_records(SALES, name="sales"), name="sales")


# ---------------------------------------------------------------------------
# 1. A query result becomes a spec-driven chart.
# ---------------------------------------------------------------------------
def section_generate() -> None:
    banner("1. generate_chart — a cited result becomes a Vega-Lite spec")

    catalog = _catalog()
    result = query_dataset(
        "SELECT region, SUM(revenue) AS s FROM sales GROUP BY region ORDER BY s DESC", catalog
    )
    chart = generate_chart(result, title="Total revenue by region")

    spec = chart.to_vega_lite()
    print(f"   mark={chart.spec.mark}  renderer={chart.renderer}  media={chart.media_type}")
    print(f"   x={spec['encoding']['x']['field']}  y={spec['encoding']['y']['field']}")
    print(f"   plotted {chart.point_count} points")


# ---------------------------------------------------------------------------
# 2. The chart is content-bound: a C2PA credential binds its bytes.
# ---------------------------------------------------------------------------
def section_content_bound() -> None:
    banner("2. content-bound — a C2PA data-driven credential binds the bytes")

    catalog = _catalog()
    result = query_dataset("SELECT region, revenue FROM sales ORDER BY region", catalog)
    chart = generate_chart(result, title="Revenue")

    print(f"   is_synthetic={chart.manifest.is_synthetic}  (a chart is data-driven, not synthetic)")
    print(f"   credential binds the bytes: {chart.content_bound()}")
    edited = chart.data + b" "
    chart.data = edited
    print(f"   an edited byte stream is caught: {not chart.content_bound()}")


# ---------------------------------------------------------------------------
# 3. The chart is data-bound: it cites the exact cells and re-derives.
# ---------------------------------------------------------------------------
def section_data_bound() -> None:
    banner("3. data-bound — the figure cites its cells and re-derives from the bytes")

    catalog = _catalog()
    result = query_dataset(
        "SELECT region, SUM(revenue) AS s FROM sales GROUP BY region ORDER BY region", catalog
    )
    chart = generate_chart(result, title="Total revenue by region")

    print(f"   cites: {chart.cite_refs()[:4]} …")
    print(f"   coverage: {chart.coverage}")
    print(f"   verifies against the source: {chart.verify(catalog)}")

    tampered = DataCatalog.of(
        Dataset.from_records([{**r, "revenue": r["revenue"] + 1} for r in SALES], name="sales"),
        name="sales",
    )
    print(f"   a tampered source is caught: {not chart.verify(tampered)}")


# ---------------------------------------------------------------------------
# 4. The cited-report builder embeds data-bound figures.
# ---------------------------------------------------------------------------
async def section_report() -> None:
    banner("4. cited report — per-claim entailed AND per-figure data-bound")

    catalog = _catalog()
    result = query_dataset(
        "SELECT region, SUM(revenue) AS s FROM sales GROUP BY region ORDER BY s DESC", catalog
    )
    chart = generate_chart(result, title="Revenue by region")

    builder = CitedReportBuilder()
    report = await builder.build_report(
        "NA leads revenue [F1]; the full split is in the table [F2].",
        [],
        figures=[
            Figure.from_chart(chart, caption="Revenue by region"),
            Figure.from_table(result, caption="Revenue table"),
        ],
        catalog=catalog,
        contract=CitationContract(min_coverage=0.0, require_figure_binding=True),
    )
    print(f"   figures: {[(f.marker, f.kind, f.data_bound) for f in report.figures]}")
    print(f"   per-figure data-binding rate: {report.coverage.figure_binding_rate}")


# ---------------------------------------------------------------------------
# 5. The app surface: query, chart, audit.
# ---------------------------------------------------------------------------
async def section_app() -> None:
    banner("5. app.generate_chart (audited) over a registered dataset")

    provider, model = example_provider()
    app = ContextApp(name="charts", provider=provider, model=model)
    app.register_dataset(SALES, name="sales")

    chart = app.generate_chart(
        "SELECT region, SUM(revenue) AS s FROM sales GROUP BY region ORDER BY s DESC",
        table="sales",
        title="Revenue by region",
    )
    decision = next(e for e in app.audit.entries if e.action == "chart_generate")
    print(f"   chart_generate type={decision.details['chart_type']} points={decision.details['points']}")
    print(f"   chart verifies: {chart.verify(app.data_catalog())}")

    # Embed the chart in a downstream cited report; the registered catalog binds it.
    report = app.cited_report(
        "Revenue concentrates in NA and APAC [F1].",
        figures=[Figure.from_chart(chart, caption="Revenue by region")],
        contract=CitationContract(min_coverage=0.0, require_figure_binding=True),
    )
    print(f"   report rendered: {report.format}, {len(report.content)} bytes")


async def main() -> None:
    section_generate()
    section_content_bound()
    section_data_bound()
    await section_report()
    await section_app()
    print("\nDone — a result became a content-bound, data-bound, offline-verifiable artifact.")


if __name__ == "__main__":
    asyncio.run(main())
