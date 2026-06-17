"""Vincio observability: traces, spans, sessions, exporters, costs, viewer."""

from .costs import CostTracker, ModelPrice, PriceTable, default_price_table
from .exporters import (
    Alert,
    AlertSink,
    ConsoleExporter,
    InMemoryExporter,
    JSONLExporter,
    MemoryAlertSink,
    MultiExporter,
    NullExporter,
    PagerDutyAlertSink,
    PrometheusExporter,
    SlackAlertSink,
    TailSamplingExporter,
    TraceExporter,
    WebhookAlertSink,
)
from .finops import (
    AlertManager,
    AlertRule,
    BudgetDecision,
    BudgetManager,
    CostBudget,
    CostEvent,
    CostLedger,
    CostReport,
    CostRow,
)
from .redaction import ContentCapturePolicy
from .sessions import Session, record_feedback, sessions_from_traces
from .spans import Feedback, Span, SpanType, Trace, TraceEvent
from .store import CostSlice, IndexedTraceStore, Percentiles, RollupBucket
from .traces import Tracer, trace_diff, trace_replay_plan
from .viewer import (
    ViewerApp,
    render_session_text,
    render_trace_text,
    serve_viewer,
    trace_diff_html,
    trace_to_html,
)

__all__ = [
    "CostTracker",
    "ModelPrice",
    "PriceTable",
    "default_price_table",
    "CostEvent",
    "CostRow",
    "CostReport",
    "CostLedger",
    "CostBudget",
    "BudgetDecision",
    "BudgetManager",
    "ConsoleExporter",
    "InMemoryExporter",
    "JSONLExporter",
    "MultiExporter",
    "NullExporter",
    "TraceExporter",
    "Feedback",
    "Session",
    "record_feedback",
    "sessions_from_traces",
    "Span",
    "SpanType",
    "Trace",
    "TraceEvent",
    "Tracer",
    "trace_diff",
    "trace_replay_plan",
    "render_trace_text",
    "render_session_text",
    "trace_to_html",
    "trace_diff_html",
    # 2.1: served observability & alerting plane
    "IndexedTraceStore",
    "Percentiles",
    "RollupBucket",
    "CostSlice",
    "TailSamplingExporter",
    "Alert",
    "AlertSink",
    "MemoryAlertSink",
    "WebhookAlertSink",
    "SlackAlertSink",
    "PagerDutyAlertSink",
    "PrometheusExporter",
    "AlertRule",
    "AlertManager",
    "ContentCapturePolicy",
    "ViewerApp",
    "serve_viewer",
]


def build_exporter(kind: str, traces_dir: str = ".vincio/traces") -> TraceExporter:
    """Factory used by config: jsonl | memory | console | otel | none."""
    if kind == "jsonl":
        return JSONLExporter(traces_dir)
    if kind == "memory":
        return InMemoryExporter()
    if kind == "console":
        return ConsoleExporter()
    if kind == "otel":
        from .otel import OTelExporter

        return OTelExporter()
    if kind in ("none", "null"):
        return NullExporter()
    raise ValueError(f"unknown exporter kind: {kind!r}")
