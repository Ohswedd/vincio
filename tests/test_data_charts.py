"""Charts & cited analytical artifacts (data plane 4.5).

Covers the spec-driven chart core (encoding inference, the Vega-Lite spec, the
mark vocabulary), the C2PA data-driven credential bound to the rendered bytes,
the back-reference to the exact source cells, offline verification (a tampered
source, spec, or credential is caught), the ``app.generate_chart`` surface with
audit, the cited-report builder's per-figure data binding, and the optional
matplotlib renderer.
"""

from __future__ import annotations

import json

import pytest

from vincio import Chart, ChartType, ContextApp, Figure, generate_chart
from vincio.core.errors import ChartError, DataError
from vincio.data import (
    DataCatalog,
    Dataset,
    VegaLiteRenderer,
    analyze_dataset,
    query_dataset,
)
from vincio.generation.report import CitationContract, CitedReportBuilder
from vincio.governance.transparency import verify_manifest
from vincio.providers.mock import MockProvider

SALES_ROWS = [
    {"region": "NA", "product": "alpha", "revenue": 1200.5},
    {"region": "EU", "product": "alpha", "revenue": 980.0},
    {"region": "NA", "product": "beta", "revenue": 300.0},
    {"region": "APAC", "product": "beta", "revenue": 1500.25},
]


def _catalog() -> DataCatalog:
    return DataCatalog.of(Dataset.from_records(SALES_ROWS, name="sales"), name="sales")


def _tampered() -> DataCatalog:
    rows = [dict(r) for r in SALES_ROWS]
    rows[0]["revenue"] = 9999.0
    return DataCatalog.of(Dataset.from_records(rows, name="sales"), name="sales")


def _app() -> ContextApp:
    app = ContextApp(name="charts-test", provider=MockProvider(default_text="x"))
    app.register_dataset(SALES_ROWS, name="sales")
    return app


# --------------------------------------------------------------------------- #
# Spec & encoding inference                                                    #
# --------------------------------------------------------------------------- #


def test_encoding_inferred_dimension_x_measure_y():
    cat = _catalog()
    result = query_dataset("SELECT region, revenue FROM sales ORDER BY region", cat)
    chart = generate_chart(result, title="Revenue by region")
    assert chart.spec.encoding.x.field == "region"
    assert chart.spec.encoding.x.type == "nominal"
    assert chart.spec.encoding.y.field == "revenue"
    assert chart.spec.encoding.y.type == "quantitative"
    assert chart.spec.columns == ["region", "revenue"]
    assert chart.point_count == 4


def test_second_dimension_becomes_color_series():
    cat = _catalog()
    result = query_dataset(
        "SELECT region, product, revenue FROM sales ORDER BY region, product", cat
    )
    chart = generate_chart(result)
    assert chart.spec.encoding.color is not None
    assert chart.spec.encoding.color.field == "product"


def test_explicit_channels_and_color_pinned():
    cat = _catalog()
    result = query_dataset("SELECT region, product, revenue FROM sales", cat)
    chart = generate_chart(result, x="region", y="revenue", color="product", type="bar")
    assert [chart.spec.encoding.x.field, chart.spec.encoding.y.field] == ["region", "revenue"]
    assert chart.spec.encoding.color.field == "product"


def test_unknown_column_refused():
    cat = _catalog()
    result = query_dataset("SELECT region, revenue FROM sales", cat)
    with pytest.raises(ChartError):
        generate_chart(result, x="nope")
    # ChartError is a DataError so the data-plane catch-all still works.
    with pytest.raises(DataError):
        generate_chart(result, y="missing")


def test_no_measure_refused():
    cat = _catalog()
    result = query_dataset("SELECT DISTINCT region FROM sales", cat)
    with pytest.raises(ChartError):
        generate_chart(result)


def test_empty_result_refused():
    cat = _catalog()
    result = query_dataset("SELECT region, revenue FROM sales WHERE region = 'ZZ'", cat)
    with pytest.raises(ChartError):
        generate_chart(result)


def test_vega_lite_spec_is_valid_and_embeds_data():
    cat = _catalog()
    result = query_dataset("SELECT region, SUM(revenue) AS s FROM sales GROUP BY region", cat)
    chart = generate_chart(result, title="Total revenue")
    spec = chart.to_vega_lite()
    assert spec["$schema"].endswith("/v5.json")
    assert spec["mark"]["type"] == chart.spec.mark.value
    assert {"x", "y"} <= set(spec["encoding"])
    assert spec["data"]["values"] == chart.spec.values
    # Round-trips through deterministic JSON.
    assert json.loads(chart.spec.to_json()) == spec


def test_arc_uses_theta_and_color_channels():
    cat = _catalog()
    result = query_dataset("SELECT region, SUM(revenue) AS s FROM sales GROUP BY region", cat)
    chart = generate_chart(result, type="arc", infer_type=False)
    spec = chart.to_vega_lite()
    assert set(spec["encoding"]) == {"theta", "color"}


def test_temporal_x_defaults_to_line():
    import datetime

    ds = Dataset.from_records(
        [
            {"day": datetime.date(2026, 1, 1), "hits": 5},
            {"day": datetime.date(2026, 1, 2), "hits": 9},
        ],
        name="traffic",
    )
    chart = generate_chart(ds)
    assert chart.spec.encoding.x.type == "temporal"
    assert chart.spec.mark is ChartType.LINE


def test_unit_carried_into_axis_title():
    from vincio.data import DataType
    from vincio.data.core import ColumnSchema, DataSchema

    schema = DataSchema(
        columns=[
            ColumnSchema(name="region"),
            ColumnSchema(name="revenue", dtype=DataType.FLOAT, unit="USD"),
        ]
    )
    ds = Dataset.from_rows([["NA", 1200.5]], schema=schema, name="sales")
    chart = generate_chart(ds, x="region", y="revenue")
    assert chart.spec.encoding.y.axis_title() == "revenue (USD)"


# --------------------------------------------------------------------------- #
# Provenance: content-bound + data-bound                                       #
# --------------------------------------------------------------------------- #


def test_credential_binds_the_rendered_bytes():
    cat = _catalog()
    result = query_dataset("SELECT region, revenue FROM sales", cat)
    chart = generate_chart(result)
    assert chart.manifest is not None
    assert chart.manifest.is_synthetic is False  # a chart is data-driven, not synthetic
    assert chart.manifest.media_type == "application/vnd.vega-lite+json"
    assert chart.content_bound()
    assert verify_manifest(chart.manifest, chart.data)


def test_cite_refs_point_at_exact_source_cells():
    cat = _catalog()
    result = query_dataset("SELECT region, revenue FROM sales ORDER BY region", cat)
    chart = generate_chart(result)
    refs = chart.cite_refs()
    assert all("#r" in r and "!" in r for r in refs)
    assert any(r.startswith("sales#r") and r.endswith("!revenue") for r in refs)
    # de-duplicated and stable
    assert len(refs) == len(set(refs))


def test_verify_holds_and_catches_tampered_source():
    cat = _catalog()
    result = query_dataset(
        "SELECT region, SUM(revenue) AS s FROM sales GROUP BY region ORDER BY region", cat
    )
    chart = generate_chart(result, title="Total")
    assert chart.verify(cat) is True
    assert chart.verify(_tampered()) is False


def test_verify_catches_tampered_spec_values():
    cat = _catalog()
    result = query_dataset("SELECT region, revenue FROM sales", cat)
    chart = generate_chart(result)
    chart.spec.values[0]["revenue"] = 0.0
    assert chart.verify(cat) is False


def test_verify_catches_stripped_credential_bytes():
    cat = _catalog()
    result = query_dataset("SELECT region, revenue FROM sales", cat)
    chart = generate_chart(result)
    chart.data = chart.data + b" "  # edit the bytes the credential binds
    assert chart.content_bound() is False
    assert chart.verify(cat) is False


def test_bare_dataset_is_content_bound_but_states_result_coverage():
    ds = Dataset.from_records([{"k": "a", "v": 1}, {"k": "b", "v": 2}], name="kv")
    chart = generate_chart(ds)
    assert chart.source is None
    assert str(chart.coverage) == "result"
    assert chart.content_bound()
    # Nothing to re-execute, but the credential still binds and the hash holds.
    assert chart.verify(_catalog()) is True


def test_analysis_result_input_charts_primary_step():
    cat = _catalog()
    analysis = analyze_dataset("total revenue by region", cat)
    chart = generate_chart(analysis, title="From analysis")
    assert chart.point_count >= 1
    assert chart.verify(cat) is True


def test_save_writes_bytes_and_sidecar(tmp_path):
    cat = _catalog()
    result = query_dataset("SELECT region, revenue FROM sales", cat)
    chart = generate_chart(result)
    path = tmp_path / "chart.vl.json"
    chart.save(path)
    assert path.read_bytes() == chart.data
    assert (tmp_path / "chart.vl.json.c2pa.json").exists()


def test_to_evidence_item_carries_lineage():
    cat = _catalog()
    result = query_dataset("SELECT region, revenue FROM sales", cat)
    chart = generate_chart(result, title="Rev")
    item = chart.to_evidence_item()
    assert item.modality == "table"
    assert item.metadata["chart_type"] == chart.spec.mark.value
    assert item.metadata["result_hash"] == chart.result_hash
    assert item.metadata["cite_refs"] == chart.cite_refs()


def test_signed_credential_verifies_with_signer():
    from vincio.governance.transparency import HmacSigner

    cat = _catalog()
    result = query_dataset("SELECT region, revenue FROM sales", cat)
    signer = HmacSigner("s3cr3t", key_id="k1")
    chart = generate_chart(result, signer=signer)
    assert chart.manifest.signature is not None
    assert verify_manifest(chart.manifest, chart.data, signer=signer)
    # without the signer a signed credential is not reported valid
    assert verify_manifest(chart.manifest, chart.data) is False
    # the chart's own checks thread the signer through honestly
    assert chart.content_bound(signer=signer) is True
    assert chart.verify(cat, signer=signer) is True
    assert chart.content_bound() is False  # a signed credential needs its verifier
    assert chart.verify(cat) is False


# --------------------------------------------------------------------------- #
# App surface                                                                  #
# --------------------------------------------------------------------------- #


def test_app_generate_chart_from_question_is_audited():
    app = _app()
    chart = app.generate_chart(
        "SELECT region, SUM(revenue) AS s FROM sales GROUP BY region ORDER BY s DESC",
        table="sales",
        title="Revenue by region",
    )
    assert isinstance(chart, Chart)
    assert chart.verify(app.data_catalog())
    entry = next(e for e in app.audit.entries if e.action == "chart_generate")
    assert entry.details["chart_type"] == chart.spec.mark.value
    assert entry.details["points"] == chart.point_count
    assert entry.details["content_sha256"] == chart.manifest.content_sha256


def test_app_generate_chart_from_result_object():
    app = _app()
    result = app.query_data("SELECT region, revenue FROM sales", table="sales")
    chart = app.generate_chart(result, title="Rev")
    assert chart.verify(app.data_catalog())


def test_app_generate_chart_refuses_injection_in_question():
    from vincio.core.errors import UnsafeQueryError

    app = _app()
    with pytest.raises(UnsafeQueryError):
        app.generate_chart("DROP TABLE sales", table="sales")


# --------------------------------------------------------------------------- #
# Cited-report builder: per-figure data binding                               #
# --------------------------------------------------------------------------- #


async def test_cited_report_embeds_data_bound_figures():
    cat = _catalog()
    result = query_dataset(
        "SELECT region, SUM(revenue) AS s FROM sales GROUP BY region ORDER BY s DESC", cat
    )
    chart = generate_chart(result, title="Revenue by region")
    figures = [
        Figure.from_chart(chart, caption="Revenue by region"),
        Figure.from_table(result, caption="Revenue table"),
    ]
    report = await CitedReportBuilder().build_report(
        "NA leads revenue [F1]; the full split is in [F2].",
        [],
        figures=figures,
        catalog=cat,
    )
    assert [f.marker for f in report.figures] == ["F1", "F2"]
    assert all(f.data_bound for f in report.figures)
    assert report.coverage.figures == 2
    assert report.coverage.figure_binding_rate == 1.0
    assert not report.unresolved_markers
    content = report.render("markdown").content
    text = content.decode() if isinstance(content, bytes) else content
    assert "Figures" in text and "data-bound" in text


async def test_cited_report_figure_binding_contract_breaches_on_tamper():
    cat = _catalog()
    result = query_dataset("SELECT region, SUM(revenue) AS s FROM sales GROUP BY region", cat)
    chart = generate_chart(result, title="Revenue")
    contract = CitationContract(min_coverage=0.0, require_figure_binding=True)
    with pytest.raises(Exception) as excinfo:
        await CitedReportBuilder().build_report(
            "see [F1]",
            [],
            figures=[Figure.from_chart(chart)],
            catalog=_tampered(),
            contract=contract,
        )
    assert "figure" in str(excinfo.value).lower()


async def test_cited_report_without_catalog_reports_binding_unchecked():
    cat = _catalog()
    result = query_dataset("SELECT region, revenue FROM sales", cat)
    chart = generate_chart(result)
    report = await CitedReportBuilder().build_report(
        "figure [F1]", [], figures=[Figure.from_chart(chart)]
    )
    assert report.figures[0].data_bound is None
    assert report.coverage.figure_binding_rate is None


async def test_app_cited_report_resolves_registered_catalog_for_figures():
    app = _app()
    result = app.query_data("SELECT region, SUM(revenue) AS s FROM sales GROUP BY region", table="sales")
    chart = app.generate_chart(result, title="Rev")
    report = app.cited_report(
        "see [F1]",
        figures=[Figure.from_chart(chart)],
        contract=CitationContract(min_coverage=0.0, require_figure_binding=True),
    )
    # No exception means the figure was verified against the app's registered catalog.
    assert report is not None


# --------------------------------------------------------------------------- #
# Optional matplotlib renderer                                                 #
# --------------------------------------------------------------------------- #


def test_matplotlib_renderer_produces_png_with_embedded_credential():
    pytest.importorskip("matplotlib")
    from vincio.data import MatplotlibRenderer
    from vincio.governance.transparency import verify_embedded_manifest

    cat = _catalog()
    result = query_dataset("SELECT region, SUM(revenue) AS s FROM sales GROUP BY region", cat)
    chart = generate_chart(result, renderer=MatplotlibRenderer(), title="Revenue")
    assert chart.media_type == "image/png"
    assert chart.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert chart.content_bound()
    assert verify_embedded_manifest(chart.data)
    assert chart.verify(cat) is True


def test_default_renderer_is_vega_lite():
    assert VegaLiteRenderer().media_type == "application/vnd.vega-lite+json"
