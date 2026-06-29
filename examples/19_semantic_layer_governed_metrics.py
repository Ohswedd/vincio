"""The semantic layer and governed metrics over a registered dataset.

The data plane already grounds a question to a verified, cell-cited query. This
program walks the rung that defines the analytical vocabulary **once** — measures,
dimensions, and derived columns — so a natural-language question maps to a
*governed metric* rather than a raw column and is computed one way everywhere,
fully offline on the deterministic mock:

  * `SemanticLayer` — declare a `DerivedColumn` (`revenue = price × qty`) once, a
    `Measure` that aggregates it (`total_revenue = SUM(revenue)`), a ratio measure
    (`avg_order_value = revenue ÷ orders`), and a `Dimension` to break down by;
  * `app.query_metric` — resolve a question (or a metric name) to the governed
    measure, compile it to a single read-only `SELECT`, and run it through the same
    governed query plane `query_data` uses, so the answer cites the exact source
    cells and re-derives from the bytes;
  * `MetricResult.verify` — prove the answer *is* the governed metric: the layer's
    definitions are unchanged, the SQL that ran is the layer's canonical
    compilation (an ad-hoc number cannot pass as the governed one), and the result
    re-derives from the hashed source;
  * `app.metric_lineage` — a metric's column-level provenance, resolving the
    derived-column graph down to its base columns and the source the dataset was
    ingested under;
  * `app.erase_source` — a right-to-erasure sweep that now reaches the dataset
    plane, removing the registered dataset alongside the source's documents and
    memories and recording it in the signed erasure proof.

Everything below is opt-in and additive; none of it touches a database or a network.
"""

from __future__ import annotations

from _shared import example_provider

from vincio import ContextApp, SemanticLayer
from vincio.data import DerivedColumn, Dimension, Measure

SALES_ROWS = [
    {"region": "NA", "product": "alpha", "price": 10.0, "qty": 3, "cost": 4.0},
    {"region": "EU", "product": "alpha", "price": 20.0, "qty": 2, "cost": 9.0},
    {"region": "NA", "product": "beta", "price": 5.0, "qty": 10, "cost": 2.0},
    {"region": "APAC", "product": "beta", "price": 8.0, "qty": 5, "cost": 3.0},
]


def banner(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


def build_app() -> tuple[ContextApp, SemanticLayer]:
    app = ContextApp(name="semantic-layer-demo", provider=example_provider())
    # Register the dataset under a named source so a metric's provenance and a
    # subject's erasure both reach into the dataset plane.
    app.register_dataset(
        SALES_ROWS, columns=list(SALES_ROWS[0]), name="sales", source="crm-export"
    )
    layer = app.semantic_layer(
        "sales",
        derived=[
            DerivedColumn(name="revenue", expression="price * qty", unit="USD"),
            DerivedColumn(name="margin", expression="(price - cost) * qty", unit="USD"),
        ],
        measures=[
            Measure(name="total_revenue", agg="sum", expression="revenue", unit="USD",
                    synonyms=["revenue", "sales"]),
            Measure(name="total_margin", agg="sum", expression="margin", unit="USD"),
            Measure(name="orders", agg="count"),
            Measure(name="avg_order_value", numerator="total_revenue", denominator="orders",
                    unit="USD"),
        ],
        dimensions=[Dimension(name="region", synonyms=["geography"])],
    )
    return app, layer


def section_define(app: ContextApp, layer: SemanticLayer) -> None:
    banner("1. measures, dimensions, and derived columns defined once")
    print("   metrics:   ", layer.metric_names)
    print("   dimensions:", layer.dimension_names)
    print("   derived:   ", [d.name for d in layer.derived])
    print("   total_revenue compiles to:")
    print("     ", app.query_metric("total_revenue", by=["region"]).sql)


def section_governed(app: ContextApp) -> None:
    banner("2. a question maps to a governed metric, computed one way everywhere")
    by_region = app.query_metric("total revenue by region")
    phrased = app.query_metric("sales by geography")  # different words, same metric
    print("   'total revenue by region' ->", {r[0]: r[1] for r in by_region.rows})
    print("   'sales by geography'       ->", {r[0]: r[1] for r in phrased.rows})
    print("   same canonical SQL:", by_region.sql == phrased.sql)
    # The governed number cites the exact source cells it rests on.
    print("   NA rests on cells:", by_region.cite_refs(0))


def section_ratio(app: ContextApp) -> None:
    banner("3. a ratio metric, computed and zero-safe")
    aov = app.query_metric("avg_order_value")
    print("   average order value (revenue ÷ orders):", aov.value(0))


def section_verify(app: ContextApp, layer: SemanticLayer) -> None:
    banner("4. the answer is provably the governed metric")
    result = app.query_metric("total_revenue", by=["region"])
    print("   verify (governed + re-derives from the bytes):", result.verify(layer, app.data_catalog()))


def section_lineage(app: ContextApp) -> None:
    banner("5. column-level lineage reaches the source")
    lin = app.metric_lineage("total_revenue")
    print("   total_revenue base columns:", lin.base_columns)
    print("   derived via:               ", lin.derived_via)
    print("   ingested under source:     ", lin.source)


def section_erasure(app: ContextApp) -> None:
    banner("6. right-to-erasure reaches the dataset plane")
    print("   catalog before erase:", app.data_catalog().names)
    result = app.erase_source("crm-export")
    print("   datasets removed:", result.datasets_removed)
    print("   catalog after erase:", app.data_catalog().names)
    print("   recorded in the signed proof:", result.proof.removed_ids.get("datasets"))


def main() -> None:
    app, layer = build_app()
    section_define(app, layer)
    section_governed(app)
    section_ratio(app)
    section_verify(app, layer)
    section_lineage(app)
    section_erasure(app)
    print("\nDone — a metric was defined once, computed one way, cited, verified, and erasable.")


if __name__ == "__main__":
    main()
