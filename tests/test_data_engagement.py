"""The data-engagement lifecycle facade — composition, narrative, conformance.

The capstone that unifies the data & analytics plane (4.1–4.7) into one coherent,
conformance-proven system. These tests prove the facade is *purely compositional*
(it delegates to the same ``app.*`` primitives, which stay usable directly), that it
narrates the whole pipeline — register → profile → sample → fit → screen → query →
analyze → chart → governed metric → cite — into one hash-linked, signed,
offline-verifiable :class:`~vincio.data.DataNarrative`, that every analytical finding
is **data-bound** (re-derives from the content-hashed source), and that a tamper
introduced anywhere — a re-ordered stage, an edited digest, a forged signature, a
tampered source, or an edited underlying artifact — is caught from the bytes alone.
"""

from __future__ import annotations

import pytest

from vincio import (
    ContextApp,
    DataEngagement,
    DataNarrative,
    DataStage,
    VincioConfig,
)
from vincio.core.errors import DataError
from vincio.data import DerivedColumn, Dimension, Measure
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner

ROWS = [
    {"region": "NA", "product": "alpha", "price": 10.0, "qty": 3},
    {"region": "EU", "product": "alpha", "price": 8.0, "qty": 5},
    {"region": "NA", "product": "beta", "price": 12.0, "qty": 2},
    {"region": "EU", "product": "beta", "price": 9.0, "qty": 4},
    {"region": "NA", "product": "alpha", "price": 11.0, "qty": 6},
]
COLS = ["region", "product", "price", "qty"]


def _app(name: str = "analyst") -> ContextApp:
    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), config=cfg)


def _layer(app: ContextApp) -> object:
    return app.semantic_layer(
        "sales",
        derived=[DerivedColumn(name="revenue", expression="price * qty")],
        measures=[Measure(name="total_revenue", agg="sum", expression="revenue")],
        dimensions=[Dimension(name="region")],
    )


def _full_engagement(app: ContextApp) -> tuple[DataEngagement, object]:
    """Thread the whole pipeline and return the engagement plus the semantic layer."""
    eng = app.data_engagement(question="how does revenue break down by region?")
    eng.register(ROWS, columns=COLS, name="sales")
    layer = _layer(app)
    eng.profile()
    eng.sample(4)
    eng.fit(max_tokens=1500)
    eng.screen()
    eng.query("total qty by region")
    eng.analyze("how does qty break down by region?")
    eng.chart(eng.result, title="Qty by region")
    eng.query_metric("total_revenue", by=["region"])
    eng.cite(title="Revenue analysis")
    return eng, layer


# -- construction & setup -----------------------------------------------------


def test_data_engagement_factory_returns_facade():
    app = _app()
    eng = app.data_engagement(question="q", dataset="sales", analyst="me")
    assert isinstance(eng, DataEngagement)
    assert eng.analyst == "me"
    assert eng.table == "sales"
    assert eng.question == "q"


def test_analyst_defaults_to_app_name():
    eng = _app(name="dataco").data_engagement()
    assert eng.analyst == "dataco"


def test_register_sets_table_and_opens_narrative():
    app = _app()
    eng = app.data_engagement()
    table = eng.register(ROWS, columns=COLS, name="sales")
    assert table == "sales"
    assert eng.table == "sales"
    assert eng.dataset_obj is not None
    assert eng.stages[0].stage == "register"
    # the dataset is genuinely registered in the app's catalog (delegation, not a fork)
    assert "sales" in app.data_catalog()


# -- composition: the lifecycle threads every stage ---------------------------


def test_lifecycle_threads_every_stage_in_order():
    app = _app()
    eng, _ = _full_engagement(app)
    narrative = eng.seal()
    assert narrative.stage_names == [
        "register",
        "profile",
        "sample",
        "fit",
        "screen",
        "query",
        "analyze",
        "chart",
        "metric",
        "cite",
    ]
    assert len(narrative.stages) == 10


def test_facade_attributes_capture_each_artifact():
    app = _app()
    eng, _ = _full_engagement(app)
    assert eng.profile_ is not None
    assert eng.sample_ is not None
    assert eng.window is not None
    assert eng.quality is not None
    assert eng.result is not None
    assert eng.analysis is not None
    assert eng.chart_ is not None
    assert eng.metric is not None
    assert eng.report is not None


def test_stages_default_to_the_registered_dataset():
    app = _app()
    eng = app.data_engagement()
    eng.register(ROWS, columns=COLS, name="sales")
    # profile/sample/screen need no explicit data — they default to the registered one
    assert eng.profile().row_count == len(ROWS)
    assert eng.sample(2).row_count == 2
    assert eng.screen().allowed in (True, False)


def test_stage_without_a_dataset_raises():
    app = _app()
    eng = app.data_engagement()
    with pytest.raises(DataError):
        eng.profile()


# -- narrative: content-bound, signed, offline-verifiable ---------------------


def test_narrative_seals_signs_and_verifies_offline():
    app = _app()
    eng, _ = _full_engagement(app)
    narrative = eng.seal()
    v = narrative.verify(app.contract_signer)
    assert v.valid
    assert v.intact and v.head_ok and v.hash_ok and v.signatures_ok
    assert v.signed_by == ["analyst"]


def test_narrative_round_trips_through_wire():
    app = _app()
    eng, _ = _full_engagement(app)
    narrative = eng.seal()
    restored = DataNarrative.from_wire(narrative.to_wire())
    assert restored.verify(app.contract_signer).valid
    assert restored.stage_names == narrative.stage_names
    # stage wire round-trip too
    stage = DataStage.from_wire(narrative.stages[0].to_wire())
    assert stage.entry_hash == narrative.stages[0].entry_hash


def test_seal_lands_on_the_audit_chain():
    app = _app()
    eng, _ = _full_engagement(app)
    narrative = eng.seal()
    assert narrative.audit_id is not None
    assert len(app.audit.query(action="data_engagement")) == 1
    assert app.audit.verify_chain()


def test_seal_without_audit_does_not_record():
    app = _app()
    eng, _ = _full_engagement(app)
    narrative = eng.seal(record_audit=False)
    assert narrative.audit_id is None
    assert len(app.audit.query(action="data_engagement")) == 0


# -- data-binding: every finding re-derives from the source -------------------


def test_engagement_is_data_bound_against_the_live_catalog():
    app = _app()
    eng, _ = _full_engagement(app)
    v = eng.verify(app.contract_signer)
    assert v.valid
    assert v.digests_ok
    assert v.data_bound is True


def test_data_binding_unchecked_without_a_catalog():
    app = _app()
    eng, _ = _full_engagement(app)
    narrative = eng.seal()
    # the narrative's own verify is offline-only; data_bound stays None
    assert narrative.verify(app.contract_signer).data_bound is None


def test_tampered_source_breaks_data_binding():
    app = _app()
    eng, _ = _full_engagement(app)
    # a different catalog whose 'sales' table holds different bytes: the findings
    # no longer re-derive, so data-binding fails even though the chain is intact
    tampered = _app()
    tampered.register_dataset(
        [{**r, "qty": r["qty"] + 100} for r in ROWS], columns=COLS, name="sales"
    )
    v = eng.verify(app.contract_signer, catalog=tampered.data_catalog())
    assert v.intact and v.digests_ok  # the bytes of the artifacts are untouched
    assert v.data_bound is False  # but they no longer match the (tampered) source
    assert not v.valid


# -- tamper detection: a change anywhere is caught ----------------------------


def test_reordered_stage_breaks_the_chain():
    app = _app()
    eng, _ = _full_engagement(app)
    narrative = eng.seal()
    tampered = DataNarrative.from_wire(narrative.to_wire())
    tampered.stages[1], tampered.stages[2] = tampered.stages[2], tampered.stages[1]
    v = tampered.verify()
    assert not v.valid
    assert not v.intact


def test_edited_digest_is_caught_and_pinpointed():
    app = _app()
    eng, _ = _full_engagement(app)
    narrative = eng.seal()
    tampered = DataNarrative.from_wire(narrative.to_wire())
    tampered.stages[0].digest = "deadbeef"
    v = tampered.verify()
    assert not v.valid
    assert v.broken_at == 0


def test_forged_signature_fails_authentication():
    app = _app()
    eng, _ = _full_engagement(app)
    narrative = eng.seal()
    stranger = HMACSigner("stranger-secret", key_id="stranger")
    assert not narrative.verify(stranger).valid


def test_edited_underlying_artifact_fails_the_digest_check():
    app = _app()
    eng, _ = _full_engagement(app)
    eng.seal()
    # tamper a captured artifact after sealing: re-digesting catches it
    eng.result.result_hash = "deadbeef"
    v = eng.verify(app.contract_signer)
    assert not v.digests_ok
    assert not v.valid


def test_require_valid_raises_on_tamper():
    app = _app()
    eng, _ = _full_engagement(app)
    narrative = eng.seal()
    narrative.require_valid(app.contract_signer)  # ok
    narrative.stages[0].digest = "deadbeef"
    with pytest.raises(DataError):
        narrative.require_valid(app.contract_signer, artifacts=list(eng._artifacts))


# -- compositional: every primitive stays usable directly ---------------------


def test_facade_delegates_to_unchanged_primitives():
    app = _app()
    eng, _ = _full_engagement(app)
    eng.seal()
    # the same primitive, called directly, produces an equally-verifiable result
    direct = _app()
    direct.register_dataset(ROWS, columns=COLS, name="sales")
    result = direct.query_data("total qty by region", table="sales")
    assert result.verify(direct.data_catalog())
    # and the facade's own captured result verifies against its source
    assert eng.result.verify(app.data_catalog())


def test_metric_stage_proves_governed_compilation():
    app = _app()
    eng, layer = _full_engagement(app)
    # the captured metric is the governed one — re-derives AND is the layer's
    # canonical compilation, not an ad-hoc number
    assert eng.metric.verify(layer, app.data_catalog())


# -- one-shot and escape-hatch paths ------------------------------------------


def test_one_shot_dataset_path_without_registration():
    app = _app()
    eng = app.data_engagement(question="q")
    # query over an unregistered dataset passed inline
    result = eng.query("total qty by region", dataset=ROWS, columns=COLS, name="sales")
    assert result.row_count == 2
    narrative = eng.seal()
    # the chain still verifies offline; data-binding is skipped for a one-shot
    assert narrative.verify(app.contract_signer).valid


def test_record_stage_escape_hatch():
    app = _app()
    eng = app.data_engagement()
    eng.register(ROWS, columns=COLS, name="sales")
    profile = app.profile_dataset(ROWS, columns=COLS)
    eng.record_stage("custom_profile", profile, note="hand-built")
    narrative = eng.seal()
    assert "custom_profile" in narrative.stage_names
    assert narrative.verify(app.contract_signer).valid


def test_chart_without_a_prior_result_raises():
    app = _app()
    eng = app.data_engagement()
    eng.register(ROWS, columns=COLS, name="sales")
    with pytest.raises(DataError):
        eng.chart()


def test_print_summary_runs(capsys):
    app = _app()
    eng, _ = _full_engagement(app)
    narrative = eng.seal()
    narrative.print_summary()
    out = capsys.readouterr().out
    assert "Data engagement" in out
    assert "register" in out


def test_cite_deliverable_is_per_figure_data_bound():
    app = _app()
    eng = app.data_engagement(question="q")
    eng.register(ROWS, columns=COLS, name="sales")
    eng.query("total qty by region")
    eng.chart(eng.result, title="Qty by region")
    report = eng.cite(title="Deliverable")
    # the rendered deliverable embeds the chart as a data-bound figure
    assert report.metadata.get("figure_binding_rate") == 1.0
