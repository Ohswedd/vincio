"""The data engagement: the whole analytics plane threaded as one system.

The data plane shipped seven rungs of primitives — first-class tabular evidence and
the compact encoder, profiling / sampling / fit-in-window and the quality rails,
governed text-to-query with cell-level provenance, the multi-step analysis agent,
content- and data-bound charts, streaming, and the semantic layer's governed
metrics — each grounded, cited, and offline-verifiable on its own. This program
walks the **capstone** that composes them into one coherent system, fully offline on
the deterministic mock:

  * `app.data_engagement` — a purely-compositional facade that threads the pipeline
    behind one governed, audited call-path: register → profile → sample → screen →
    query → analyze → chart → governed metric → cite. Every lifecycle method
    delegates to the same `app.*` primitive a caller would use directly;
  * `DataNarrative` — the sealed, signed, hash-chained narrative of the whole
    engagement: `verify()` recomputes the chain from the bytes alone, so a re-ordered
    stage, an edited digest, or a forged signature is caught;
  * data-binding — `engagement.verify(catalog=...)` re-executes every captured query,
    analysis, chart, and metric against the content-hashed source and confirms each
    re-derives from the bytes (the data plane's distinguishing guarantee), so a
    tampered source is caught even when the chain itself is intact;
  * purely compositional — the same primitives stay usable on their own, byte-for-byte
    the same as when the facade calls them.

Everything below is opt-in and additive; none of it touches a database or a network.
"""

from __future__ import annotations

from _shared import example_provider

from vincio import ContextApp, DataNarrative
from vincio.data import DerivedColumn, Dimension, Measure
from vincio.security.audit import HMACSigner

SALES_ROWS = [
    {"region": "NA", "product": "alpha", "price": 10.0, "qty": 3},
    {"region": "EU", "product": "alpha", "price": 8.0, "qty": 5},
    {"region": "NA", "product": "beta", "price": 12.0, "qty": 2},
    {"region": "EU", "product": "beta", "price": 9.0, "qty": 4},
    {"region": "NA", "product": "alpha", "price": 11.0, "qty": 6},
]
COLS = ["region", "product", "price", "qty"]


def banner(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


def build_app() -> ContextApp:
    provider, _ = example_provider()
    return ContextApp(name="analyst", provider=provider)


def thread_engagement(app: ContextApp):
    banner("1. one call-path threads the whole plane")
    eng = app.data_engagement(question="how does revenue break down by region?")
    eng.register(SALES_ROWS, columns=COLS, name="sales", source="crm-export")
    layer = app.semantic_layer(
        "sales",
        derived=[DerivedColumn(name="revenue", expression="price * qty", unit="USD")],
        measures=[Measure(name="total_revenue", agg="sum", expression="revenue", unit="USD")],
        dimensions=[Dimension(name="region")],
    )
    eng.profile()
    eng.sample(4)
    eng.screen()
    eng.query("total qty by region")
    eng.analyze("how does qty break down by region?")
    eng.chart(eng.result, title="Qty by region")
    eng.query_metric("total_revenue", by=["region"])
    eng.cite(title="Revenue analysis")
    narrative = eng.seal()
    print("   stages threaded:", " → ".join(narrative.stage_names))
    print("   query result:   ", {r[0]: r[1] for r in eng.result.rows})
    print("   governed metric:", {r[0]: r[1] for r in eng.metric.rows})
    return eng, layer, narrative


def section_offline(app: ContextApp, narrative: DataNarrative) -> None:
    banner("2. the sealed narrative verifies offline, from the bytes alone")
    v = narrative.verify(app.contract_signer)
    print("   valid:", v.valid, "| chain intact:", v.intact, "| signed by:", v.signed_by)
    # Round-trips through the wire and still verifies.
    restored = DataNarrative.from_wire(narrative.to_wire())
    print("   round-trips and verifies:", restored.verify(app.contract_signer).valid)


def section_data_bound(app: ContextApp, eng) -> None:
    banner("3. every finding re-derives from the source it cites (data-bound)")
    whole = eng.verify(app.contract_signer)
    print(
        "   valid:",
        whole.valid,
        "| digests ok:",
        whole.digests_ok,
        "| data-bound:",
        whole.data_bound,
    )
    print("   the query's answer cites cells:", eng.result.cite_refs(0, eng.result.columns[-1]))


def section_tamper(app: ContextApp, eng, narrative: DataNarrative) -> None:
    banner("4. a tamper introduced anywhere is caught")
    # A re-ordered stage breaks the hash chain.
    reordered = DataNarrative.from_wire(narrative.to_wire())
    reordered.stages[1], reordered.stages[2] = reordered.stages[2], reordered.stages[1]
    print("   re-ordered stage caught:    ", not reordered.verify().valid)
    # A forged signature fails authentication.
    print(
        "   forged signature caught:    ", not narrative.verify(HMACSigner("x", key_id="x")).valid
    )
    # A tampered source breaks data-binding even with the chain intact.
    tampered = build_app()
    tampered.register_dataset(
        [{**r, "qty": r["qty"] + 100} for r in SALES_ROWS], columns=COLS, name="sales"
    )
    print(
        "   tampered source caught:     ",
        eng.verify(app.contract_signer, catalog=tampered.data_catalog()).data_bound is False,
    )


def section_compositional(app: ContextApp, layer) -> None:
    banner("5. purely compositional — the primitives stay usable on their own")
    direct = app.query_data("total qty by region", table="sales")
    print("   direct query verifies:      ", direct.verify(app.data_catalog()))
    metric = app.query_metric("total_revenue", by=["region"])
    print("   direct governed metric verifies:", metric.verify(layer, app.data_catalog()))
    print("   one continuous audit narrative:", app.audit.verify_chain())


def main() -> None:
    app = build_app()
    eng, layer, narrative = thread_engagement(app)
    section_offline(app, narrative)
    section_data_bound(app, eng)
    section_tamper(app, eng, narrative)
    section_compositional(app, layer)
    print(
        "\nDone — the whole analytics plane threaded into one signed, data-bound, offline-verifiable narrative."
    )


if __name__ == "__main__":
    main()
