"""Notebook & REPL ergonomics: rich reprs for the objects you inspect most.

Call :func:`enable_rich_reprs` once (e.g. at the top of a notebook) to give
``RunResult``, ``Trace``, ``EvalReport``, ``MemoryItem``, and ``SearchHit``
HTML/Markdown reprs, so Jupyter renders them as readable tables instead of
giant ``repr`` blobs::

    import vincio.notebook as nb
    nb.enable_rich_reprs()
    app.run("...")          # now displays as a summary card

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


# -- install / display ----------------------------------------------------------

_RICH_FLAG = "_vincio_rich_reprs"


def enable_rich_reprs() -> None:
    """Attach ``_repr_html_`` / ``_repr_markdown_`` to the core result types."""
    from .core.types import MemoryItem, RunResult
    from .evals.reports import EvalReport
    from .observability.spans import Trace
    from .retrieval.indexes import SearchHit

    bindings = [
        (RunResult, run_result_html, run_result_markdown),
        (Trace, trace_html, trace_markdown),
        (EvalReport, eval_report_html, eval_report_markdown),
        (MemoryItem, memory_item_html, None),
        (SearchHit, search_hit_html, None),
    ]
    for cls, html_fn, md_fn in bindings:
        cls._repr_html_ = (lambda fn: lambda self: fn(self))(html_fn)
        if md_fn is not None:
            cls._repr_markdown_ = (lambda fn: lambda self: fn(self))(md_fn)
        setattr(cls, _RICH_FLAG, True)


def disable_rich_reprs() -> None:
    """Remove the reprs installed by :func:`enable_rich_reprs`."""
    from .core.types import MemoryItem, RunResult
    from .evals.reports import EvalReport
    from .observability.spans import Trace
    from .retrieval.indexes import SearchHit

    for cls in (RunResult, Trace, EvalReport, MemoryItem, SearchHit):
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
