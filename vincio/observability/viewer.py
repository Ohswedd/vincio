"""Local trace viewer: terminal rendering and static HTML export.

No server, no account, no dependency: ``render_trace_text`` gives a TUI-style
tree for the terminal, ``trace_to_html`` writes a single self-contained HTML
file for a trace or a whole session, and ``trace_diff_html`` renders two
traces side by side with the structural diff highlighted.
"""

from __future__ import annotations

import html
import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from ..core.utils import to_jsonable, utcnow
from .sessions import Session
from .spans import Span, Trace
from .traces import trace_diff

if TYPE_CHECKING:
    from .store import IndexedTraceStore

__all__ = [
    "render_trace_text",
    "render_session_text",
    "trace_to_html",
    "session_to_html",
    "trace_diff_html",
    # 2.1: served observability plane
    "ViewerApp",
    "serve_viewer",
]

_STATUS_GLYPH = {"ok": "✓", "error": "✗", "running": "…"}

_INTERESTING_ATTRIBUTES = (
    "model", "tokens", "input_tokens", "output_tokens", "cost_usd", "cacheability",
    "evidence", "valid", "task_type", "tool", "cached", "finish", "ttft_ms",
)


# -- terminal rendering ------------------------------------------------------


def _format_attributes(span: Span) -> str:
    parts = [
        f"{key}={span.attributes[key]}"
        for key in _INTERESTING_ATTRIBUTES
        if key in span.attributes
    ]
    parts.extend(f"score:{name}={value:g}" for name, value in span.scores.items())
    return f"  [{', '.join(parts)}]" if parts else ""


def render_trace_text(trace: Trace, *, show_attributes: bool = True) -> str:
    """Render one trace as an indented tree for the terminal."""
    lines = [
        f"trace {trace.id}  app={trace.app_name or '-'}  status={trace.status}  "
        f"duration={trace.duration_ms}ms"
    ]
    context = [
        f"run={trace.run_id}" if trace.run_id else "",
        f"session={trace.session_id}" if trace.session_id else "",
        f"user={trace.user_id}" if trace.user_id else "",
    ]
    context_line = "  ".join(c for c in context if c)
    if context_line:
        lines.append(f"  {context_line}")
    if trace.scores:
        lines.append("  scores: " + ", ".join(f"{k}={v:g}" for k, v in sorted(trace.scores.items())))
    for item in trace.feedback:
        score = f" score={item.score:g}" if item.score is not None else ""
        comment = f' "{item.comment}"' if item.comment else ""
        lines.append(f"  feedback: {item.key}{score}{comment}")

    children: dict[str | None, list[Span]] = {}
    for span in trace.spans:
        children.setdefault(span.parent_id, []).append(span)

    def walk(parent_id: str | None, prefix: str) -> None:
        siblings = sorted(children.get(parent_id, []), key=lambda s: s.start_time)
        for index, span in enumerate(siblings):
            last = index == len(siblings) - 1
            connector = "└─" if last else "├─"
            glyph = _STATUS_GLYPH.get(span.status, "?")
            attributes = _format_attributes(span) if show_attributes else ""
            error = f"  error={span.error}" if span.error else ""
            lines.append(
                f"{prefix}{connector} {glyph} {span.type}:{span.name} "
                f"({span.duration_ms}ms){attributes}{error}"
            )
            walk(span.id, prefix + ("   " if last else "│  "))

    walk(None, "  ")
    return "\n".join(lines)


def render_session_text(session: Session) -> str:
    """Render a session: aggregate header plus each trace tree."""
    summary = session.summary()
    lines = [
        f"session {session.id}  app={session.app_name or '-'}  runs={summary['runs']}  "
        f"duration={summary['duration_ms']}ms  error_rate={summary['error_rate']:g}",
    ]
    if summary["scores"]:
        lines.append("  scores: " + ", ".join(f"{k}={v:g}" for k, v in summary["scores"].items()))
    if summary["feedback_count"]:
        mean = summary["mean_feedback"]
        lines.append(
            f"  feedback: {summary['feedback_count']} item(s)"
            + (f", mean={mean:g}" if mean is not None else "")
        )
    for trace in session.traces:
        lines.append("")
        lines.append(render_trace_text(trace))
    return "\n".join(lines)


# -- static HTML export ------------------------------------------------------

_CSS = """
body{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;
margin:24px;color:#1a1a2e;background:#fafafa}
h1{font-size:16px}h2{font-size:14px;margin-top:28px}
.meta{color:#555;margin:2px 0}
.badge{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;margin-right:6px}
.badge.ok{background:#d7f5dd;color:#11691f}.badge.error{background:#fde0e0;color:#a11212}
.badge.running{background:#fff3cd;color:#7a5d00}.badge.score{background:#e3ecfd;color:#1c4587}
.badge.feedback{background:#f3e3fd;color:#6a1c87}
details{margin:2px 0 2px 18px;border-left:1px solid #ddd;padding-left:10px}
summary{cursor:pointer;padding:2px 0}
summary:hover{background:#f0f0f5}
.bar{display:inline-block;height:8px;background:#7aa7e8;border-radius:3px;vertical-align:middle;
margin-left:8px;min-width:2px}
.dur{color:#777;font-size:11px}
table.attrs{border-collapse:collapse;margin:6px 0 6px 12px;font-size:12px}
table.attrs td{border:1px solid #e3e3ea;padding:2px 8px;vertical-align:top;max-width:640px;
overflow-wrap:anywhere}
.col{display:inline-block;vertical-align:top;width:48%;margin-right:1%}
.diffadd{background:#d7f5dd}.diffdel{background:#fde0e0}.diffchg{background:#fff3cd}
""".strip()


def _html_page(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><h1>{html.escape(title)}</h1>{body}</body></html>"
    )


def _attributes_table(data: dict[str, Any]) -> str:
    if not data:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td>"
        f"<td>{html.escape(json.dumps(value, default=str) if not isinstance(value, str) else value)}</td></tr>"
        for key, value in data.items()
    )
    return f"<table class='attrs'>{rows}</table>"


def _span_html(span: Span, children: dict[str | None, list[Span]], total_ms: int) -> str:
    width = min(100.0, 100.0 * span.duration_ms / total_ms) if total_ms else 0.0
    badges = f"<span class='badge {span.status}'>{span.status}</span>"
    badges += "".join(
        f"<span class='badge score'>{html.escape(name)}={value:g}</span>"
        for name, value in span.scores.items()
    )
    body = _attributes_table(span.attributes)
    if span.events:
        body += _attributes_table(
            {f"event:{e.name}": e.attributes for e in span.events}
        )
    if span.error:
        body += f"<p class='meta'>error: {html.escape(span.error)}</p>"
    body += "".join(
        _span_html(child, children, total_ms)
        for child in sorted(children.get(span.id, []), key=lambda s: s.start_time)
    )
    return (
        f"<details open><summary>{badges}<b>{html.escape(span.type)}:{html.escape(span.name)}</b> "
        f"<span class='dur'>{span.duration_ms}ms</span>"
        f"<span class='bar' style='width:{width:.1f}px'></span></summary>{body}</details>"
    )


def _trace_body(trace: Trace) -> str:
    meta = [
        f"status: {trace.status}",
        f"duration: {trace.duration_ms}ms",
        f"run: {trace.run_id or '-'}",
        f"session: {trace.session_id or '-'}",
        f"user: {trace.user_id or '-'}",
        f"spans: {len(trace.spans)}",
    ]
    body = f"<p class='meta'>{' · '.join(html.escape(m) for m in meta)}</p>"
    badges = "".join(
        f"<span class='badge score'>{html.escape(name)}={value:g}</span>"
        for name, value in sorted(trace.scores.items())
    )
    badges += "".join(
        "<span class='badge feedback'>"
        + html.escape(item.key + (f"={item.score:g}" if item.score is not None else ""))
        + (f": {html.escape(item.comment)}" if item.comment else "")
        + "</span>"
        for item in trace.feedback
    )
    if badges:
        body += f"<p>{badges}</p>"
    if trace.attributes:
        body += _attributes_table(trace.attributes)
    children: dict[str | None, list[Span]] = {}
    for span in trace.spans:
        children.setdefault(span.parent_id, []).append(span)
    total_ms = trace.duration_ms or max((s.duration_ms for s in trace.spans), default=0)
    body += "".join(
        _span_html(span, children, total_ms)
        for span in sorted(children.get(None, []), key=lambda s: s.start_time)
    )
    return body


def trace_to_html(trace: Trace, *, title: str | None = None) -> str:
    """A self-contained static HTML page for one trace (no server needed)."""
    return _html_page(title or f"Vincio trace {trace.id}", _trace_body(trace))


def session_to_html(session: Session, *, title: str | None = None) -> str:
    """A self-contained static HTML page for a whole session."""
    summary = session.summary()
    body = f"<p class='meta'>{html.escape(json.dumps(summary, default=str))}</p>"
    for trace in session.traces:
        body += f"<h2>trace {html.escape(trace.id)}</h2>{_trace_body(trace)}"
    return _html_page(title or f"Vincio session {session.id}", body)


def trace_diff_html(a: Trace, b: Trace, *, title: str | None = None) -> str:
    """Two traces side by side with the structural diff highlighted."""
    diff = trace_diff(a, b)
    rows = ""
    for name in diff["spans_only_in_a"]:
        rows += f"<tr class='diffdel'><td>{html.escape(name)}</td><td>only in A</td></tr>"
    for name in diff["spans_only_in_b"]:
        rows += f"<tr class='diffadd'><td>{html.escape(name)}</td><td>only in B</td></tr>"
    for change in diff["status_changes"]:
        rows += (
            f"<tr class='diffchg'><td>{html.escape(change['span'])}</td>"
            f"<td>status {html.escape(change['a'])} → {html.escape(change['b'])}</td></tr>"
        )
    for change in diff["duration_changes"]:
        rows += (
            f"<tr class='diffchg'><td>{html.escape(change['span'])}</td>"
            f"<td>{change['a_ms']}ms → {change['b_ms']}ms ({change['delta_ms']:+d}ms)</td></tr>"
        )
    summary = (
        f"<p class='meta'>A={html.escape(a.id)} ({diff['duration_ms']['a']}ms, {diff['status']['a']}) · "
        f"B={html.escape(b.id)} ({diff['duration_ms']['b']}ms, {diff['status']['b']})</p>"
    )
    table = f"<table class='attrs'>{rows}</table>" if rows else "<p class='meta'>no structural changes</p>"
    body = (
        summary + "<h2>Changes</h2>" + table
        + f"<div class='col'><h2>Trace A</h2>{_trace_body(a)}</div>"
        + f"<div class='col'><h2>Trace B</h2>{_trace_body(b)}</div>"
    )
    return _html_page(title or f"Vincio trace diff {a.id} vs {b.id}", body)


# ---------------------------------------------------------------------------
# Served observability plane (2.1)
# ---------------------------------------------------------------------------

_WINDOWS = {"1h": timedelta(hours=1), "24h": timedelta(hours=24), "7d": timedelta(days=7)}


def _since_from(window: str | None):
    delta = _WINDOWS.get(window or "")
    return (utcnow() - delta) if delta else None


def _slice_rows(slices: list[Any]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(str(s.key))}</td><td>${s.cost_usd:.6f}</td>"
        f"<td>{s.calls}</td><td>{s.errors}</td></tr>"
        for s in slices
    )
    return rows or "<tr><td colspan='4' class='meta'>no data</td></tr>"


def _dashboard_html(store: IndexedTraceStore, *, window: str | None) -> str:
    since = _since_from(window)
    stats = store.stats()
    latency = store.percentiles("latency", since=since)
    cost = store.percentiles("cost", since=since)
    by_tenant = store.cost_by_dimension("tenant", since=since)
    by_model = store.cost_by_dimension("model", since=since)
    recent = store.tail(25)
    win_links = " · ".join(
        f"<a href='/?window={w}'>{w}</a>" for w in ("1h", "24h", "7d", "all")
    ).replace("window=all", "")
    head = (
        f"<p class='meta'>traces={stats['traces']} · errors={stats['errors']} "
        f"· error_rate={stats['error_rate']:.1%} · total_cost=${stats['total_cost_usd']:.6f} "
        f"· window: {win_links}</p>"
    )
    perf = (
        "<h2>Latency (ms)</h2>"
        f"<table class='attrs'><tr><th>p50</th><th>p95</th><th>p99</th><th>max</th><th>n</th></tr>"
        f"<tr><td>{latency.p50:g}</td><td>{latency.p95:g}</td><td>{latency.p99:g}</td>"
        f"<td>{latency.max:g}</td><td>{latency.count}</td></tr></table>"
        "<h2>Cost per run (usd)</h2>"
        f"<table class='attrs'><tr><th>p50</th><th>p95</th><th>p99</th><th>max</th></tr>"
        f"<tr><td>${cost.p50:.6f}</td><td>${cost.p95:.6f}</td><td>${cost.p99:.6f}</td>"
        f"<td>${cost.max:.6f}</td></tr></table>"
    )
    dims = (
        "<div class='col'><h2>Cost by tenant</h2>"
        "<table class='attrs'><tr><th>tenant</th><th>cost</th><th>calls</th><th>errors</th></tr>"
        f"{_slice_rows(by_tenant)}</table></div>"
        "<div class='col'><h2>Cost by model</h2>"
        "<table class='attrs'><tr><th>model</th><th>cost</th><th>calls</th><th>errors</th></tr>"
        f"{_slice_rows(by_model)}</table></div>"
    )
    tail_rows = "".join(
        f"<tr class='{'diffdel' if t.status == 'error' else ''}'>"
        f"<td><a href='/trace?id={html.escape(t.id)}'>{html.escape(t.id)}</a></td>"
        f"<td>{html.escape(t.status)}</td><td>{t.duration_ms}ms</td>"
        f"<td>{html.escape(t.tenant_id or '∅')}</td></tr>"
        for t in recent
    ) or "<tr><td colspan='4' class='meta'>no traces</td></tr>"
    tail = (
        "<h2>Live tail</h2>"
        "<table class='attrs'><tr><th>trace</th><th>status</th><th>duration</th><th>tenant</th></tr>"
        f"{tail_rows}</table>"
    )
    return _html_page("Vincio observability", head + perf + dims + tail)


class ViewerApp:
    """Request handler for the served observability plane, decoupled from sockets.

    Holds an :class:`~vincio.observability.store.IndexedTraceStore` and answers
    ``(path, params)`` with ``(status, content_type, body)`` — so the HTTP
    routes are exercised directly in tests without binding a port. Routes:
    ``/`` (dashboard), ``/trace?id=`` (single-trace HTML), ``/healthz``,
    ``/api/stats``, ``/api/traces``, ``/api/trace?id=``, ``/api/rollup``.
    """

    def __init__(self, store: IndexedTraceStore) -> None:
        self.store = store

    def handle(self, path: str, params: dict[str, str] | None = None) -> tuple[int, str, str]:
        params = params or {}
        window = params.get("window")
        since = _since_from(window)
        if path in ("/", "/dashboard"):
            return 200, "text/html; charset=utf-8", _dashboard_html(self.store, window=window)
        if path == "/healthz":
            return 200, "text/plain; charset=utf-8", "ok"
        if path == "/api/stats":
            payload = {
                "stats": self.store.stats(),
                "latency": self.store.percentiles("latency", since=since).model_dump(),
                "cost": self.store.percentiles("cost", since=since).model_dump(),
                "cost_by_tenant": [s.model_dump() for s in self.store.cost_by_dimension("tenant", since=since)],
                "cost_by_model": [s.model_dump() for s in self.store.cost_by_dimension("model", since=since)],
            }
            return 200, "application/json", json.dumps(to_jsonable(payload))
        if path == "/api/traces":
            limit = int(params.get("limit", "50"))
            traces = self.store.query(
                tenant_id=params.get("tenant"),
                model=params.get("model"),
                status=params.get("status"),
                session_id=params.get("session"),
                since=since,
                limit=limit,
            )
            body = [
                {
                    "id": t.id, "status": t.status, "duration_ms": t.duration_ms,
                    "tenant_id": t.tenant_id, "session_id": t.session_id,
                }
                for t in traces
            ]
            return 200, "application/json", json.dumps(to_jsonable(body))
        if path in ("/trace", "/api/trace"):
            trace_id = params.get("id", "")
            trace = self.store.get(trace_id)
            if trace is None:
                return 404, "text/plain; charset=utf-8", f"no trace {trace_id!r}"
            if path == "/api/trace":
                return 200, "application/json", json.dumps(to_jsonable(trace.model_dump(mode="json")))
            return 200, "text/html; charset=utf-8", trace_to_html(trace)
        if path == "/api/rollup":
            bucket = params.get("bucket", "1h")
            series = self.store.rollup(
                bucket, dimension=params.get("dimension", "global"),
                key=params.get("key", "∅"), since=since,
            )
            return 200, "application/json", json.dumps(to_jsonable([b.model_dump() for b in series]))
        return 404, "text/plain; charset=utf-8", "not found"


def serve_viewer(
    store: IndexedTraceStore,
    *,
    host: str = "127.0.0.1",
    port: int = 8043,
) -> Any:
    """Start the served observability plane over ``store`` (opt-in, self-hosted).

    Returns the running ``ThreadingHTTPServer`` (started on a daemon thread);
    call ``.shutdown()`` to stop it. Zero new dependency — the standard-library
    ``http.server`` — and it never leaves your infrastructure. The
    zero-dependency static export (:func:`trace_to_html`) stays the default for
    one-off shares; this plane is for a dashboard you keep watching.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    app = ViewerApp(store)

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            parsed = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            status, content_type, body = app.handle(parsed.path, params)
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args: Any) -> None:  # silence default stderr logging
            return

    server = ThreadingHTTPServer((host, port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
