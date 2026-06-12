"""Vincio observability: traces, spans, exporters, costs."""

from .costs import CostTracker, ModelPrice, PriceTable, default_price_table
from .exporters import (
    ConsoleExporter,
    InMemoryExporter,
    JSONLExporter,
    MultiExporter,
    NullExporter,
    TraceExporter,
)
from .spans import Span, SpanType, Trace, TraceEvent
from .traces import Tracer, trace_diff, trace_replay_plan

__all__ = [
    "CostTracker",
    "ModelPrice",
    "PriceTable",
    "default_price_table",
    "ConsoleExporter",
    "InMemoryExporter",
    "JSONLExporter",
    "MultiExporter",
    "NullExporter",
    "TraceExporter",
    "Span",
    "SpanType",
    "Trace",
    "TraceEvent",
    "Tracer",
    "trace_diff",
    "trace_replay_plan",
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
