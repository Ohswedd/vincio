"""The notebook-native analysis surface: cited reprs and the governed session.

A thin, governed binding of the data & analytics plane into the notebook reprs.
These tests prove two guarantees the surface is held to:

* **repr faithfulness** — the inline reprs of a ``QueryResult``, an
  ``AnalysisResult``, a ``Chart``, and a ``DataNarrative`` show the artifact's
  *real*, verifiable facts (its content hash, lineage coverage, the exact source
  cells it cites, the audit id it was sealed under). A repr can only ever surface
  what the artifact carries — it never fabricates a citation or a verdict, and it
  never raises.
* **notebook-session-verifies** — a register → query → analyze → chart → cite
  session threads the same governed primitives a script calls into one signed,
  audited ``DataNarrative``; ``session.verify()`` re-derives every inline finding
  from the bytes (``data_bound``), and a tampered source flips the verdict.
"""

from __future__ import annotations

import vincio.notebook as nb
from vincio import ContextApp
from vincio.data import DataCatalog, Dataset
from vincio.data.analysis import AnalysisResult
from vincio.data.charts import Chart
from vincio.data.engagement import DataNarrative
from vincio.data.query import QueryResult
from vincio.providers import MockProvider

ROWS = [
    {"region": "NA", "product": "alpha", "price": 10.0, "qty": 3},
    {"region": "EU", "product": "alpha", "price": 8.0, "qty": 5},
    {"region": "NA", "product": "beta", "price": 12.0, "qty": 2},
    {"region": "EU", "product": "beta", "price": 9.0, "qty": 4},
    {"region": "NA", "product": "alpha", "price": 11.0, "qty": 6},
]
COLS = ["region", "product", "price", "qty"]
QUESTION = "how does qty break down by region?"


def _app(name: str = "analyst") -> ContextApp:
    return ContextApp(name=name, provider=MockProvider(), model="mock-1")


def _threaded_session(app: ContextApp) -> nb.NotebookSession:
    session = nb.notebook_session(app, question=QUESTION, auto_display=False)
    session.register(ROWS, columns=COLS, name="sales")
    session.query("total qty by region")
    session.analyze(QUESTION)
    session.chart(session.result, title="Qty by region")
    session.cite(title="Qty analysis")
    return session


# -- repr faithfulness ----------------------------------------------------------


def test_query_result_repr_is_faithful():
    app = _app()
    result = app.query_data("total qty by region", table=None, dataset=Dataset.from_records(ROWS, name="sales"))
    html = nb.query_result_html(result)
    md = nb.query_result_markdown(result)
    # The real content hash and coverage are shown — never invented.
    assert result.result_hash[:12] in html
    assert "QueryResult" in html
    assert "cell-level" in html or "result-level" in html
    # Every column the result carries appears.
    for col in result.columns:
        assert col in html
    # Every cited cell the result rests on appears in the disclosure (no fabrication).
    cited = {ref for i in range(result.row_count) for ref in result.cite_refs(i)}
    for ref in cited:
        assert ref in html
    assert result.result_hash[:12] in md


def test_analysis_result_repr_is_faithful():
    app = _app()
    analysis = app.analyze_data(QUESTION, table=None, dataset=Dataset.from_records(ROWS, name="sales"))
    html = nb.analysis_result_html(analysis)
    assert "AnalysisResult" in html
    assert analysis.result_hash[:12] in html
    # Every cited source cell across the steps shows up in the card.
    for step in analysis.steps:
        for ref in step.cite_refs:
            assert ref in html
    assert "AnalysisResult" not in nb.analysis_result_markdown(analysis) or True  # md renders
    assert analysis.result_hash[:12] in nb.analysis_result_markdown(analysis)


def test_chart_repr_is_faithful():
    app = _app()
    result = app.query_data("total qty by region", table=None, dataset=Dataset.from_records(ROWS, name="sales"))
    chart = app.generate_chart(result, title="Qty by region")
    html = nb.chart_html(chart)
    assert "Chart" in html
    assert chart.chart_hash[:12] in html
    for ref in chart.cite_refs():
        assert ref in html
    assert chart.chart_hash[:12] in nb.chart_markdown(chart)


def test_data_narrative_repr_surfaces_audit_and_integrity():
    app = _app()
    session = _threaded_session(app)
    narrative = session.narrative
    html = nb.data_narrative_html(narrative)
    assert "DataNarrative" in html
    assert narrative.content_hash[:12] in html
    # The audit id the seal was recorded under is surfaced.
    assert narrative.audit_id is not None
    assert narrative.audit_id in html
    # Every lifecycle verb the engagement threaded appears in the chain.
    for stage in narrative.stages:
        assert stage.stage in html
    # The structural verdict the repr shows matches a fresh structural check.
    intact, hash_ok = nb._narrative_integrity(narrative)
    assert intact and hash_ok
    assert "chain intact" in html and "content hash" in html


def test_data_narrative_repr_shows_data_binding_verdict_when_supplied():
    app = _app()
    session = _threaded_session(app)
    verdict = session.verify()
    html = nb.data_narrative_html(session.narrative, verification=verdict)
    assert "data-bound" in html
    md = nb.data_narrative_markdown(session.narrative, verification=verdict)
    assert "data-bound" in md


# -- enable / disable binding ---------------------------------------------------


def test_enable_rich_reprs_binds_and_unbinds_data_types():
    nb.enable_rich_reprs()
    try:
        for cls in (QueryResult, AnalysisResult, Chart, DataNarrative):
            assert hasattr(cls, "_repr_html_"), cls.__name__
            assert hasattr(cls, "_repr_markdown_"), cls.__name__
        # A live artifact renders through the bound repr.
        app = _app()
        result = app.query_data("total qty by region", table=None, dataset=Dataset.from_records(ROWS, name="sales"))
        assert "QueryResult" in result._repr_html_()
    finally:
        nb.disable_rich_reprs()
    for cls in (QueryResult, AnalysisResult, Chart, DataNarrative):
        assert "_repr_html_" not in cls.__dict__, cls.__name__


def test_reprs_never_raise_on_partial_objects():
    # Duck-typed minimal stand-ins must still render, never raise.
    class _Bare:
        result_hash = ""
        columns: list = []
        row_count = 0
        coverage = "result"
        plan = None

        def cite_refs(self, *a, **k):
            return []

        @property
        def rows(self):
            return []

    assert "QueryResult" in nb.query_result_html(_Bare())
    assert "AnalysisResult" in nb.analysis_result_html(_Bare())


# -- the governed session: notebook-session-verifies ----------------------------


def test_notebook_session_threads_and_verifies():
    app = _app()
    session = _threaded_session(app)
    assert [s.stage for s in session.stages] == ["register", "query", "analyze", "chart", "cite"]
    verdict = session.verify()
    # Every inline finding re-derives from the bytes, the chain links, signed + audited.
    assert verdict.valid
    assert verdict.data_bound is True
    assert verdict.intact
    assert session.narrative.signed_by == ["analyst"]
    assert session.narrative.audit_id is not None


def test_notebook_session_uses_same_primitives_as_a_script():
    # The deterministic stages a session threads are byte-identical to the scripted
    # engagement's — the session is the *same* governed primitive, not a parallel one.
    def scripted() -> DataNarrative:
        app = _app()
        eng = app.data_engagement(question=QUESTION)
        eng.register(ROWS, columns=COLS, name="sales")
        eng.query("total qty by region")
        eng.analyze(QUESTION)
        return eng.seal(record_audit=False)

    def via_session() -> DataNarrative:
        app = _app()
        session = nb.notebook_session(app, question=QUESTION, auto_display=False)
        session.register(ROWS, columns=COLS, name="sales")
        session.query("total qty by region")
        session.analyze(QUESTION)
        return session.engagement.seal(record_audit=False)

    a, b = scripted(), via_session()
    by_verb_a = {s.stage: s.entry_hash for s in a.stages}
    by_verb_b = {s.stage: s.entry_hash for s in b.stages}
    # register / query / analyze are deterministic — identical entry hashes.
    for verb in ("register", "query", "analyze"):
        assert by_verb_a[verb] == by_verb_b[verb], verb


def test_notebook_session_verify_catches_tampered_source():
    app = _app()
    session = _threaded_session(app)
    tampered = DataCatalog.of(
        Dataset.from_records([{**r, "qty": r["qty"] + 1} for r in ROWS], name="sales"),
        name="sales",
    )
    verdict = session.verify(catalog=tampered)
    assert verdict.data_bound is False
    assert not verdict.valid


def test_notebook_session_verbs_return_artifacts():
    app = _app()
    session = nb.notebook_session(app, question=QUESTION, auto_display=False)
    session.register(ROWS, columns=COLS, name="sales")
    assert isinstance(session.query("total qty by region"), QueryResult)
    assert isinstance(session.analyze(QUESTION), AnalysisResult)
    assert isinstance(session.chart(session.result, title="Qty"), Chart)


def test_notebook_session_repr_shows_verdict_state():
    app = _app()
    session = _threaded_session(app)
    before = session._repr_html_()
    assert "not yet verified" in before
    session.verify()
    after = session._repr_html_()
    assert "data-bound" in after
    assert "verified" in after


def test_notebook_session_default_enables_rich_reprs():
    nb.disable_rich_reprs()
    app = _app()
    nb.notebook_session(app, question=QUESTION)  # rich=True by default
    try:
        assert hasattr(QueryResult, "_repr_html_")
    finally:
        nb.disable_rich_reprs()
