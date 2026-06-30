"""Notebook & REPL ergonomics: rich reprs for the objects you inspect most.

Call :func:`enable_rich_reprs` once (e.g. at the top of a notebook) to give
``RunResult``, ``Trace``, ``EvalReport``, ``MemoryItem``, and ``SearchHit`` —
and the data & analytics artifacts ``QueryResult``, ``AnalysisResult``,
``Chart``, and ``DataNarrative`` — HTML/Markdown reprs, so Jupyter renders them
as readable cards (with clickable cell citations, the data-binding verdict, and
the audit id) instead of giant ``repr`` blobs::

    import vincio.notebook as nb
    nb.enable_rich_reprs()
    app.run("...")          # now displays as a summary card

For an interactive analysis, :func:`notebook_session` threads the same governed
register → query → analyze → chart → cite pipeline a script runs into one signed,
audited :class:`~vincio.data.DataNarrative` — so a notebook exploration is
reproducible by construction, and ``session.verify()`` re-derives every inline
finding from the bytes::

    session = nb.notebook_session(app, question="how does revenue split by region?")
    session.register(rows, columns=cols, name="sales")
    session.query("total revenue by region")        # renders inline, cited
    session.analyze("how does revenue break down by region?")
    session.chart(title="Revenue by region")
    session.cite(title="Revenue analysis")
    session.verify()                                  # data-bound, offline

The pure ``*_html`` / ``*_markdown`` functions can also be called directly, and
:func:`display` works inside or outside IPython.
"""

from __future__ import annotations

import html
from typing import Any

__all__ = [
    "enable_rich_reprs",
    "disable_rich_reprs",
    "display",
    "run_result_markdown",
    "run_result_html",
    "trace_markdown",
    "trace_html",
    "eval_report_markdown",
    "eval_report_html",
    "memory_item_html",
    "search_hit_html",
    "query_result_html",
    "query_result_markdown",
    "analysis_result_html",
    "analysis_result_markdown",
    "chart_html",
    "chart_markdown",
    "data_narrative_html",
    "data_narrative_markdown",
    "notebook_session",
    "NotebookSession",
]


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _num(value: Any, spec: str = ".4f") -> str:
    """Format a number defensively — reprs must never raise on odd inputs."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value if value is not None else 0)
    return format(value, spec)


def _table(rows: list[tuple[str, Any]], *, title: str = "") -> str:
    head = f'<div style="font-weight:600;margin-bottom:2px">{_esc(title)}</div>' if title else ""
    body = "".join(
        f'<tr><td style="padding:1px 8px 1px 0;color:#666">{_esc(k)}</td>'
        f"<td style=\"padding:1px 0\"><code>{_esc(v)}</code></td></tr>"
        for k, v in rows
    )
    return f'{head}<table style="border-collapse:collapse;font-size:12px">{body}</table>'


def _truncate(text: Any, limit: int = 160) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "…"


# -- RunResult ------------------------------------------------------------------


def _run_rows(result: Any) -> list[tuple[str, Any]]:
    status = getattr(getattr(result, "status", None), "value", getattr(result, "status", ""))
    usage = getattr(result, "usage", None)
    tokens = (
        f"{getattr(usage, 'input_tokens', 0)} in / {getattr(usage, 'output_tokens', 0)} out"
        if usage is not None
        else "—"
    )
    rows = [
        ("status", status),
        ("output", _truncate(getattr(result, "output", ""))),
        ("cost_usd", f"${_num(getattr(result, 'cost_usd', 0.0), '.6f')}"),
        ("latency_ms", getattr(result, "latency_ms", 0)),
        ("tokens", tokens),
        ("citations", len(getattr(result, "citations", []) or [])),
        ("trace_id", getattr(result, "trace_id", "")),
    ]
    if getattr(result, "eval_scores", None):
        rows.append(("eval_scores", dict(result.eval_scores)))
    if getattr(result, "error", None):
        rows.append(("error", result.error))
    return rows


def run_result_html(result: Any) -> str:
    return _table(_run_rows(result), title="RunResult")


def run_result_markdown(result: Any) -> str:
    lines = ["| field | value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in _run_rows(result)]
    return "\n".join(lines)


# -- Trace ----------------------------------------------------------------------


def _trace_rows(trace: Any) -> list[tuple[str, Any]]:
    rows = [
        ("id", getattr(trace, "id", "")),
        ("app", getattr(trace, "app_name", "")),
        ("status", getattr(trace, "status", "")),
        ("duration_ms", getattr(trace, "duration_ms", 0)),
        ("spans", len(getattr(trace, "spans", []) or [])),
    ]
    if getattr(trace, "scores", None):
        rows.append(("scores", dict(trace.scores)))
    if getattr(trace, "feedback", None):
        rows.append(("feedback", len(trace.feedback)))
    return rows


def trace_html(trace: Any) -> str:
    return _table(_trace_rows(trace), title="Trace")


def trace_markdown(trace: Any) -> str:
    lines = ["| field | value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in _trace_rows(trace)]
    return "\n".join(lines)


# -- EvalReport -----------------------------------------------------------------


def _eval_metric_rows(report: Any) -> list[tuple[str, Any]]:
    try:
        summary = report.summary()
    except Exception:  # noqa: BLE001 - reprs must never raise
        summary = {}
    rows: list[tuple[str, Any]] = [
        ("name", getattr(report, "name", "")),
        ("dataset", getattr(report, "dataset", "")),
        ("cases", len(getattr(report, "cases", []) or [])),
    ]
    for metric, aggregates in summary.items():
        mean = aggregates.get("mean") if isinstance(aggregates, dict) else aggregates
        rows.append((metric, f"{mean:.4f}" if isinstance(mean, (int, float)) else mean))
    gates = getattr(report, "gates", None) or {}
    for gate, info in gates.items():
        passed = info.get("passed") if isinstance(info, dict) else info
        rows.append((f"gate:{gate}", "✓ pass" if passed else "✗ fail"))
    return rows


def eval_report_html(report: Any) -> str:
    return _table(_eval_metric_rows(report), title="EvalReport")


def eval_report_markdown(report: Any) -> str:
    lines = ["| field | value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in _eval_metric_rows(report)]
    return "\n".join(lines)


# -- MemoryItem / SearchHit -----------------------------------------------------


def memory_item_html(item: Any) -> str:
    scope = getattr(getattr(item, "scope", None), "value", getattr(item, "scope", ""))
    mtype = getattr(getattr(item, "type", None), "value", getattr(item, "type", ""))
    rows = [
        ("id", getattr(item, "id", "")),
        ("scope/type", f"{scope}/{mtype}"),
        ("confidence", _num(getattr(item, "confidence", 0.0), ".2f")),
        ("status", getattr(item, "status", "")),
        ("content", _truncate(getattr(item, "content", ""))),
    ]
    return _table(rows, title="MemoryItem")


def search_hit_html(hit: Any) -> str:
    chunk = getattr(hit, "chunk", None)
    rows = [
        ("score", _num(getattr(hit, "score", 0.0), ".4f")),
        ("source", getattr(hit, "source", "")),
        ("doc", getattr(chunk, "document_id", "")),
        ("text", _truncate(getattr(chunk, "text", ""))),
    ]
    return _table(rows, title="SearchHit")


# -- data & analytics artifacts -------------------------------------------------
#
# Inline reprs for the data plane's cited artifacts — a QueryResult, an
# AnalysisResult, a Chart, and the DataNarrative a session seals. They are pure
# and never raise: they read an artifact's already-computed, *verifiable* facts
# (its content hash, lineage coverage, the exact source cells it cites, the audit
# id it was recorded under) and present them; they never re-execute or re-derive
# (that is ``verify()``'s job, surfaced on a :class:`NotebookSession`). So the
# repr can only ever show what the artifact actually carries — it cannot fabricate
# a citation or a verdict.

_MAX_REPR_ROWS = 12
_MAX_CHIPS = 24

_TH = "text-align:left;padding:1px 10px 2px 0;color:#666;border-bottom:1px solid #ddd;font-weight:600"
_TD = "padding:1px 10px 1px 0;font-family:monospace;font-size:11px"


def _md_rows(rows: list[tuple[str, Any]]) -> str:
    lines = ["| field | value |", "| --- | --- |"]
    lines += [f"| {_esc_md(k)} | {_esc_md(v)} |" for k, v in rows]
    return "\n".join(lines)


def _esc_md(value: Any) -> str:
    """Escape pipes so a value never breaks a Markdown table."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _short(value: Any, width: int = 12) -> str:
    """A short, stable prefix of a content hash (``""`` stays ``""``)."""
    text = str(value or "")
    return text[:width] + "…" if len(text) > width else text


def _pill(state: bool | None, *, yes: str, no: str, unknown: str = "—") -> str:
    """A small coloured verdict pill — green for ``True``, red ``False``, grey ``None``."""
    if state is None:
        return f'<span style="color:#999">{_esc(unknown)}</span>'
    color = "#137333" if state else "#c5221f"
    glyph = "✓" if state else "✗"
    return f'<span style="color:{color};font-weight:600">{glyph} {_esc(yes if state else no)}</span>'


def _coverage_pill(coverage: Any) -> str:
    """A pill for lineage coverage — cell-level is green, result-level a neutral note."""
    cov = str(getattr(coverage, "value", coverage) or "")
    if cov == "cell":
        return _pill(True, yes="cell-level lineage", no="")
    return '<span style="color:#a06a00;font-weight:600">◑ result-level lineage</span>'


def _cell_tooltip(ref: str) -> str:
    """A human description of a ``table#r<row>!<column>`` cell locator (best-effort)."""
    try:
        table, rest = str(ref).split("#r", 1)
        row, column = rest.split("!", 1)
        return f"row {row}, column {column} of table {table}"
    except (ValueError, AttributeError):  # pragma: no cover - non-standard ref
        return str(ref)


def _cite_chip(ref: str) -> str:
    return (
        f'<code title="{_esc(_cell_tooltip(ref))}" style="background:#eef2ff;'
        "border:1px solid #c7d2fe;border-radius:3px;padding:0 4px;margin:1px;"
        f'display:inline-block;font-size:11px">{_esc(ref)}</code>'
    )


def _cites_disclosure(refs: list[str], *, label: str = "cited source cells") -> str:
    """A click-to-expand list of cell citations — the clickable provenance."""
    refs = list(dict.fromkeys(str(r) for r in refs if r))  # dedup, preserve order
    if not refs:
        return ""
    shown = refs[:_MAX_CHIPS]
    chips = "".join(_cite_chip(r) for r in shown)
    more = (
        f' <span style="color:#999;font-size:11px">+{len(refs) - len(shown)} more</span>'
        if len(refs) > len(shown)
        else ""
    )
    return (
        '<details style="margin-top:4px"><summary style="cursor:pointer;font-size:12px;'
        f'color:#444">🔗 {len(refs)} {_esc(label)}</summary>'
        f'<div style="margin:3px 0 0">{chips}{more}</div></details>'
    )


def _cites_md(refs: list[str]) -> str:
    refs = list(dict.fromkeys(str(r) for r in refs if r))
    return " ".join(f"`{r}`" for r in refs[:_MAX_CHIPS]) if refs else ""


# -- QueryResult ----------------------------------------------------------------


def _query_facts(result: Any) -> list[tuple[str, Any]]:
    plan = getattr(result, "plan", None)
    sql = (getattr(plan, "sql", "") or "").strip()
    rows: list[tuple[str, Any]] = []
    if sql:
        rows.append(("query", _truncate(" ".join(sql.split()), 140)))
    tables = ", ".join(getattr(plan, "tables", []) or [])
    if tables:
        rows.append(("tables", tables))
    cols = list(getattr(result, "columns", []) or [])
    rows.append(("rows × cols", f"{getattr(result, 'row_count', 0)} × {len(cols)}"))
    rows.append(("result_hash", _short(getattr(result, "result_hash", ""))))
    return rows


def query_result_html(result: Any) -> str:
    """Render a :class:`~vincio.data.QueryResult` as a cited result card.

    Shows the query, the result table (each cell tooltipped with the source cells
    it rests on), the lineage-coverage pill, the result hash, and a clickable
    disclosure of every cell citation the result carries."""
    columns = list(getattr(result, "columns", []) or [])
    try:
        all_rows = list(getattr(result, "rows", []) or [])
    except Exception:  # noqa: BLE001 - reprs must never raise
        all_rows = []
    shown = all_rows[:_MAX_REPR_ROWS]

    all_refs: list[str] = []
    body: list[str] = []
    for i, row in enumerate(shown):
        cells: list[str] = []
        for j, val in enumerate(row):
            try:
                refs = result.cite_refs(i, j)
            except Exception:  # noqa: BLE001
                refs = []
            all_refs.extend(refs)
            attr = f' title="{_esc("; ".join(refs))}"' if refs else ""
            cells.append(f'<td style="{_TD}"{attr}>{_esc(_truncate(val, 40))}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    extra = (
        f'<tr><td colspan="{max(len(columns), 1)}" style="color:#999;font-size:11px">'
        f"… {len(all_rows) - len(shown)} more rows</td></tr>"
        if len(all_rows) > len(shown)
        else ""
    )
    head = "".join(f'<th style="{_TH}">{_esc(c)}</th>' for c in columns)
    table = (
        '<table style="border-collapse:collapse;margin-top:4px">'
        f"<tr>{head}</tr>{''.join(body)}{extra}</table>"
        if columns
        else ""
    )
    title = (
        '<div style="font-weight:600;margin-bottom:2px">QueryResult &nbsp;'
        f"{_coverage_pill(getattr(result, 'coverage', None))}</div>"
    )
    return f"{title}{_table(_query_facts(result))}{table}{_cites_disclosure(all_refs)}"


def query_result_markdown(result: Any) -> str:
    refs: list[str] = []
    try:
        for i in range(min(getattr(result, "row_count", 0), _MAX_REPR_ROWS)):
            refs.extend(result.cite_refs(i))
    except Exception:  # noqa: BLE001
        refs = []
    facts = _query_facts(result)
    facts.append(("coverage", getattr(getattr(result, "coverage", ""), "value", "")))
    out = _md_rows(facts)
    cites = _cites_md(refs)
    return f"{out}\n\n**cited cells:** {cites}" if cites else out


# -- AnalysisResult -------------------------------------------------------------


def _analysis_facts(result: Any) -> list[tuple[str, Any]]:
    metrics = getattr(result, "metrics", {}) or {}
    rows: list[tuple[str, Any]] = [
        ("objective", _truncate(getattr(result, "objective", ""), 140)),
        ("table", getattr(result, "table", "")),
        ("steps", int(metrics.get("steps", len(getattr(result, "steps", []) or [])))),
        ("cited findings", int(metrics.get("cited_findings", 0))),
        ("citation coverage", _num(metrics.get("citation_coverage", 0.0), ".0%")),
        ("result_hash", _short(getattr(result, "result_hash", ""))),
    ]
    return rows


def analysis_result_html(result: Any) -> str:
    """Render an :class:`~vincio.data.AnalysisResult` as a cited-narrative card.

    Shows the objective, the per-step findings each with the exact source cells it
    cites (clickable), the lineage-coverage pill, the citation coverage, and the
    content hash binding the narrative to its steps."""
    steps = list(getattr(result, "steps", []) or [])
    items: list[str] = []
    for step in steps:
        finding = getattr(step, "finding", "")
        if not finding:
            continue
        kind = getattr(getattr(step, "kind", ""), "value", getattr(step, "kind", ""))
        star = "★ " if getattr(step, "primary", False) else ""
        refs = list(getattr(step, "cite_refs", []) or [])
        items.append(
            f'<li style="margin:3px 0"><span style="color:#888;font-size:11px">{star}{_esc(kind)}</span> '
            f"{_esc(finding)}{_cites_disclosure(refs)}</li>"
        )
    body = (
        f'<ul style="margin:4px 0 0;padding-left:18px;font-size:12px">{"".join(items)}</ul>'
        if items
        else ""
    )
    title = (
        '<div style="font-weight:600;margin-bottom:2px">AnalysisResult &nbsp;'
        f"{_coverage_pill(getattr(result, 'coverage', None))}</div>"
    )
    return f"{title}{_table(_analysis_facts(result))}{body}"


def analysis_result_markdown(result: Any) -> str:
    out = [_md_rows(_analysis_facts(result))]
    for step in getattr(result, "steps", []) or []:
        cited = getattr(step, "cited", "")
        if cited:
            out.append(f"- {_esc_md(cited)}")
    return "\n".join(out)


# -- Chart ----------------------------------------------------------------------


def _chart_facts(chart: Any) -> list[tuple[str, Any]]:
    spec = getattr(chart, "spec", None)
    mark = getattr(getattr(spec, "mark", ""), "value", getattr(spec, "mark", ""))
    rows: list[tuple[str, Any]] = [
        ("title", getattr(spec, "title", "") or "—"),
        ("type", mark),
        ("media_type", getattr(chart, "media_type", "")),
        ("points", getattr(chart, "point_count", 0)),
        ("chart_hash", _short(getattr(chart, "chart_hash", ""))),
    ]
    return rows


def chart_html(chart: Any) -> str:
    """Render a :class:`~vincio.data.Chart` as a content- and data-bound figure card.

    Shows the chart type and media type, whether a C2PA credential binds the rendered
    bytes (content-bound), the exact source cells the figure was built from (clickable),
    and the chart hash binding the spec, values, and source together."""
    try:
        refs = list(chart.cite_refs())
    except Exception:  # noqa: BLE001
        refs = []
    content_bound = getattr(chart, "manifest", None) is not None
    badge = _pill(content_bound or None, yes="content-bound (C2PA)", no="", unknown="unsigned bytes")
    title = f'<div style="font-weight:600;margin-bottom:2px">Chart &nbsp;{badge}</div>'
    return f"{title}{_table(_chart_facts(chart))}{_cites_disclosure(refs, label='source cells')}"


def chart_markdown(chart: Any) -> str:
    try:
        refs = list(chart.cite_refs())
    except Exception:  # noqa: BLE001
        refs = []
    out = _md_rows(_chart_facts(chart))
    cites = _cites_md(refs)
    return f"{out}\n\n**source cells:** {cites}" if cites else out


# -- DataNarrative --------------------------------------------------------------


def _narrative_integrity(narrative: Any) -> tuple[bool, bool]:
    """``(chain_intact, hash_ok)`` recomputed from the bytes — cheap, no catalog.

    This is the structural half of :meth:`~vincio.data.DataNarrative.verify` (the
    half that needs neither a verifier nor a catalog): every stage's link hash
    re-derives and chains, and the content hash recomputes. The data-binding half
    — re-executing each finding against the live source — is surfaced on a
    :class:`NotebookSession`, which holds the catalog."""
    try:
        stages = list(getattr(narrative, "stages", []) or [])
        prev = ""
        intact = True
        for i, stage in enumerate(stages):
            if (
                stage.index != i
                or stage.prev_hash != prev
                or stage.entry_hash != stage.compute_entry_hash()
            ):
                intact = False
                break
            prev = stage.entry_hash
        intact = intact and getattr(narrative, "head_hash", "") == prev
        hash_ok = narrative.compute_hash() == getattr(narrative, "content_hash", "")
        return intact, hash_ok
    except Exception:  # noqa: BLE001 - reprs must never raise
        return False, False


def _narrative_facts(narrative: Any) -> list[tuple[str, Any]]:
    signed = list(getattr(narrative, "signed_by", []) or [])
    rows: list[tuple[str, Any]] = [
        ("analyst", getattr(narrative, "analyst", "")),
        ("dataset", getattr(narrative, "dataset", "") or "—"),
        ("question", _truncate(getattr(narrative, "question", "") or "—", 120)),
        ("stages", len(getattr(narrative, "stages", []) or [])),
        ("content_hash", _short(getattr(narrative, "content_hash", ""))),
        ("signed by", ", ".join(signed) if signed else "—"),
        ("audit id", getattr(narrative, "audit_id", None) or "—"),
    ]
    return rows


def data_narrative_html(narrative: Any, *, verification: Any = None) -> str:
    """Render a :class:`~vincio.data.DataNarrative` as a sealed-engagement card.

    Shows the stage chain the engagement threaded, the structural offline verdict
    recomputed from the bytes (the chain links and the content hash re-derive), the
    signatures, and the **audit id** the seal was recorded under. When a
    :class:`~vincio.data.DataEngagementVerification` is supplied (e.g. by a verified
    :class:`NotebookSession`), the full data-binding verdict is shown too."""
    intact, hash_ok = _narrative_integrity(narrative)
    stages = list(getattr(narrative, "stages", []) or [])
    rows_html = "".join(
        f'<tr><td style="{_TD}">{i}</td><td style="{_TD}">{_esc(getattr(s, "stage", ""))}</td>'
        f'<td style="{_TD};color:#888">{_esc(getattr(s, "kind", ""))}</td>'
        f'<td style="{_TD};color:#888">{_esc(_short(getattr(s, "digest", ""), 10))}</td></tr>'
        for i, s in enumerate(stages)
    )
    chain = (
        f'<table style="border-collapse:collapse;margin-top:4px">'
        f'<tr><th style="{_TH}">#</th><th style="{_TH}">stage</th>'
        f'<th style="{_TH}">artifact</th><th style="{_TH}">digest</th></tr>{rows_html}</table>'
        if stages
        else ""
    )
    verdicts = [
        _pill(intact, yes="chain intact", no="chain broken"),
        _pill(hash_ok, yes="content hash", no="hash mismatch"),
    ]
    if verification is not None:
        verdicts.append(
            _pill(getattr(verification, "data_bound", None), yes="data-bound", no="not data-bound")
        )
        verdicts.append(_pill(getattr(verification, "valid", None), yes="verified", no="invalid"))
    badges = ' <span style="color:#ccc">·</span> '.join(verdicts)
    title = (
        '<div style="font-weight:600;margin-bottom:2px">DataNarrative &nbsp;'
        f'<span style="font-size:12px">{badges}</span></div>'
    )
    return f"{title}{_table(_narrative_facts(narrative))}{chain}"


def data_narrative_markdown(narrative: Any, *, verification: Any = None) -> str:
    intact, hash_ok = _narrative_integrity(narrative)
    facts = _narrative_facts(narrative)
    facts.append(("chain intact", "✓" if intact else "✗"))
    facts.append(("content hash", "✓" if hash_ok else "✗"))
    if verification is not None:
        facts.append(("data-bound", _verdict_glyph(getattr(verification, "data_bound", None))))
        facts.append(("valid", _verdict_glyph(getattr(verification, "valid", None))))
    stages = list(getattr(narrative, "stages", []) or [])
    chain = " → ".join(getattr(s, "stage", "") for s in stages)
    out = _md_rows(facts)
    return f"{out}\n\n**chain:** {chain}" if chain else out


def _verdict_glyph(state: bool | None) -> str:
    return "—" if state is None else ("✓" if state else "✗")


# -- install / display ----------------------------------------------------------

_RICH_FLAG = "_vincio_rich_reprs"


def _repr_bindings() -> list[tuple[Any, Any, Any]]:
    """The ``(class, html_fn, markdown_fn)`` triples :func:`enable_rich_reprs` installs.

    The core result types are always bound; the data & analytics artifacts are added
    when the (dependency-free) data plane imports — so a notebook gets cited, inline
    reprs for a ``QueryResult``, an ``AnalysisResult``, a ``Chart``, and a
    ``DataNarrative`` too."""
    from .core.types import MemoryItem, RunResult
    from .evals.reports import EvalReport
    from .observability.spans import Trace
    from .retrieval.indexes import SearchHit

    bindings: list[tuple[Any, Any, Any]] = [
        (RunResult, run_result_html, run_result_markdown),
        (Trace, trace_html, trace_markdown),
        (EvalReport, eval_report_html, eval_report_markdown),
        (MemoryItem, memory_item_html, None),
        (SearchHit, search_hit_html, None),
    ]
    try:
        from .data.analysis import AnalysisResult
        from .data.charts import Chart
        from .data.engagement import DataNarrative
        from .data.query import QueryResult

        bindings += [
            (QueryResult, query_result_html, query_result_markdown),
            (AnalysisResult, analysis_result_html, analysis_result_markdown),
            (Chart, chart_html, chart_markdown),
            (DataNarrative, data_narrative_html, data_narrative_markdown),
        ]
    except ImportError:  # pragma: no cover - the data plane ships with the core
        pass
    return bindings


def enable_rich_reprs() -> None:
    """Attach ``_repr_html_`` / ``_repr_markdown_`` to the core and data-plane types."""
    for cls, html_fn, md_fn in _repr_bindings():
        cls._repr_html_ = (lambda fn: lambda self: fn(self))(html_fn)  # type: ignore[attr-defined]
        if md_fn is not None:
            cls._repr_markdown_ = (lambda fn: lambda self: fn(self))(md_fn)  # type: ignore[attr-defined]
        setattr(cls, _RICH_FLAG, True)


def disable_rich_reprs() -> None:
    """Remove the reprs installed by :func:`enable_rich_reprs`."""
    for cls, _html_fn, _md_fn in _repr_bindings():
        if getattr(cls, _RICH_FLAG, False):
            for attr in ("_repr_html_", "_repr_markdown_", _RICH_FLAG):
                if attr in cls.__dict__:
                    delattr(cls, attr)


def display(obj: Any) -> Any:
    """Display *obj* richly in IPython, or print a readable summary elsewhere."""
    try:
        from IPython.display import display as ipy_display

        return ipy_display(obj)
    except ImportError:
        if hasattr(obj, "_repr_markdown_"):
            print(obj._repr_markdown_())
        else:
            print(obj)
        return None


# -- notebook-native analysis session -------------------------------------------


def _session_facts(session: NotebookSession) -> list[tuple[str, Any]]:
    eng = session.engagement
    stages = list(getattr(eng, "stages", []) or [])
    chain = " → ".join(getattr(s, "stage", "") for s in stages) or "—"
    rows: list[tuple[str, Any]] = [
        ("dataset", getattr(eng, "table", "") or "—"),
        ("question", _truncate(getattr(eng, "question", "") or "—", 120)),
        ("analyst", getattr(eng, "analyst", "")),
        ("stages", chain),
    ]
    return rows


def _session_html(session: NotebookSession) -> str:
    v = session.verification
    if v is not None:
        badges = ' <span style="color:#ccc">·</span> '.join(
            [
                _pill(getattr(v, "valid", None), yes="verified", no="invalid"),
                _pill(getattr(v, "data_bound", None), yes="data-bound", no="not data-bound"),
                _pill(getattr(v, "intact", None), yes="chain intact", no="chain broken"),
            ]
        )
        audit = getattr(getattr(session.engagement, "narrative", None), "audit_id", None)
        tail = (
            f'<div style="font-size:11px;color:#888;margin-top:3px">audit id: '
            f"<code>{_esc(audit)}</code></div>"
            if audit
            else ""
        )
    else:
        badges = (
            '<span style="color:#a06a00">◷ not yet verified — call '
            "<code>session.verify()</code></span>"
        )
        tail = ""
    title = (
        '<div style="font-weight:600;margin-bottom:2px">NotebookSession &nbsp;'
        f'<span style="font-size:12px">{badges}</span></div>'
    )
    return f"{title}{_table(_session_facts(session))}{tail}"


def _session_markdown(session: NotebookSession) -> str:
    facts = _session_facts(session)
    v = session.verification
    if v is not None:
        facts.append(("valid", _verdict_glyph(getattr(v, "valid", None))))
        facts.append(("data-bound", _verdict_glyph(getattr(v, "data_bound", None))))
        audit = getattr(getattr(session.engagement, "narrative", None), "audit_id", None)
        if audit:
            facts.append(("audit id", audit))
    else:
        facts.append(("verified", "call session.verify()"))
    return _md_rows(facts)


class NotebookSession:
    """An interactive, governed data-analysis session for notebooks and REPLs.

    A thin front over :meth:`~vincio.core.app.ContextApp.data_engagement`. Each verb
    — :meth:`register`, :meth:`profile`, :meth:`sample`, :meth:`screen`,
    :meth:`query`, :meth:`analyze`, :meth:`chart`, :meth:`query_metric`,
    :meth:`cite` — delegates to the *same* governed primitive a script calls, renders
    the artifact it produced inline (when ``auto_display`` and IPython are present),
    and threads it into the engagement's hash-linked narrative. Sealing
    :attr:`narrative` or calling :meth:`verify` yields the *identical* signed
    :class:`~vincio.data.DataNarrative` the scripted pipeline produces — so a notebook
    exploration is reproducible and offline-verifiable by construction, and
    :meth:`verify` re-derives every inline finding from the bytes.

    The session adds **no** analytical capability of its own; it is an interactive
    front for the governed one. Build it with :func:`notebook_session`.
    """

    def __init__(self, engagement: Any, *, auto_display: bool = True) -> None:
        self._eng = engagement
        self.auto_display = auto_display
        self.verification: Any = None

    # -- accessors ------------------------------------------------------------

    @property
    def engagement(self) -> Any:
        """The underlying :class:`~vincio.data.DataEngagement` this session narrates."""
        return self._eng

    @property
    def app(self) -> Any:
        """The :class:`~vincio.core.app.ContextApp` the session runs on."""
        return self._eng.app

    @property
    def stages(self) -> Any:
        """The lifecycle stages threaded so far (a live, copied view)."""
        return self._eng.stages

    @property
    def result(self) -> Any:
        """The last :class:`~vincio.data.QueryResult` produced, if any."""
        return self._eng.result

    @property
    def analysis(self) -> Any:
        """The last :class:`~vincio.data.AnalysisResult` produced, if any."""
        return self._eng.analysis

    @property
    def chart_(self) -> Any:
        """The last :class:`~vincio.data.Chart` produced, if any."""
        return self._eng.chart_

    # -- lifecycle verbs (delegate, capture, display) -------------------------

    def _step(self, artifact: Any) -> Any:
        """Invalidate the cached verdict and render the artifact inline."""
        self.verification = None
        if self.auto_display and artifact is not None:
            try:
                display(artifact)
            except Exception:  # noqa: BLE001 - display is best-effort, never fatal
                pass
        return artifact

    def register(self, data: Any, **kwargs: Any) -> Any:
        """Register a dataset into the app catalog and record it as a stage."""
        return self._step(self._eng.register(data, **kwargs))

    def profile(self, data: Any | None = None, **kwargs: Any) -> Any:
        """Profile the dataset (bounded-memory) and record it as a stage."""
        return self._step(self._eng.profile(data, **kwargs))

    def sample(self, n: int, data: Any | None = None, **kwargs: Any) -> Any:
        """Draw a representative sample and record it as a stage."""
        return self._step(self._eng.sample(n, data, **kwargs))

    def screen(self, data: Any | None = None, **kwargs: Any) -> Any:
        """Run the data-quality / safety rails and record it as a stage."""
        return self._step(self._eng.screen(data, **kwargs))

    def query(self, request: str, **kwargs: Any) -> Any:
        """Run a governed, read-only-verified query and record it as a stage."""
        return self._step(self._eng.query(request, **kwargs))

    def analyze(self, objective: str, **kwargs: Any) -> Any:
        """Run the bounded multi-step analysis agent and record it as a stage."""
        return self._step(self._eng.analyze(objective, **kwargs))

    def chart(self, result: Any | None = None, **kwargs: Any) -> Any:
        """Render a content- and data-bound chart and record it as a stage."""
        return self._step(self._eng.chart(result, **kwargs))

    def query_metric(self, request: Any, **kwargs: Any) -> Any:
        """Compute a governed metric through the semantic layer and record it as a stage."""
        return self._step(self._eng.query_metric(request, **kwargs))

    def cite(self, answer: Any | None = None, **kwargs: Any) -> Any:
        """Assemble the findings into a cited, per-figure data-bound deliverable."""
        return self._step(self._eng.cite(answer, **kwargs))

    # -- seal / verify --------------------------------------------------------

    def seal(self, **kwargs: Any) -> Any:
        """Mint the signed, audited :class:`~vincio.data.DataNarrative` of the session."""
        return self._eng.seal(**kwargs)

    @property
    def narrative(self) -> Any:
        """The sealed narrative (sealing on first access), identical to the script path."""
        if self._eng.narrative is None:
            self._eng.seal()
        return self._eng.narrative

    def verify(self, catalog: Any | None = None, *, verifier: Any | None = None) -> Any:
        """Verify the whole session offline — chain, digests, signatures, **and** data-binding.

        Seals a signed, audited narrative if one is not current, then re-derives every
        inline finding (query, analysis, chart, metric) against the live source — so the
        returned :class:`~vincio.data.DataEngagementVerification` is ``data_bound`` only
        when every finding reproduces from the bytes. The verdict is cached for the
        session's repr. ``catalog`` defaults to the app's registered catalog; pass an
        explicit ``verifier`` to authenticate the analyst's signature with a different key.
        """
        if verifier is None:
            verifier = getattr(self.app, "contract_signer", None)
        if self._eng.narrative is None:
            self._eng.seal(sign=True, record_audit=True)
        self.verification = self._eng.verify(verifier, catalog=catalog)
        return self.verification

    # -- reprs ----------------------------------------------------------------

    def _repr_html_(self) -> str:
        return _session_html(self)

    def _repr_markdown_(self) -> str:
        return _session_markdown(self)


def notebook_session(
    app: Any,
    *,
    dataset: str = "",
    question: str = "",
    analyst: str | None = None,
    auto_display: bool = True,
    rich: bool = True,
) -> NotebookSession:
    """Open a governed, notebook-native analysis session over *app*.

    Returns a :class:`NotebookSession` — a thin interactive front over
    :meth:`~vincio.core.app.ContextApp.data_engagement` that threads
    register → query → analyze → chart → cite into the *same* signed, audited
    :class:`~vincio.data.DataNarrative` a script produces, rendering each cited
    artifact inline as you go. With ``rich`` (the default), inline reprs are enabled
    so a ``QueryResult``, an ``AnalysisResult``, a ``Chart``, and the sealed
    ``DataNarrative`` display as cards with clickable cell citations::

        session = notebook_session(app, question="how does revenue split by region?")
        session.register(rows, columns=cols, name="sales")
        session.query("total revenue by region")
        session.analyze("how does revenue break down by region?")
        session.chart(title="Revenue by region")
        session.cite(title="Revenue analysis")
        session.verify()        # data-bound, offline — every finding re-derives

    The session adds no analytical capability; it is the interactive front for the
    governed one, so an exploration is reproducible and offline-verifiable by
    construction.
    """
    if rich:
        enable_rich_reprs()
    engagement = app.data_engagement(dataset=dataset, question=question, analyst=analyst)
    return NotebookSession(engagement, auto_display=auto_display)
