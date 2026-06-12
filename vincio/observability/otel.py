"""OpenTelemetry export. Requires ``pip install "vincio[otel]"``.

Model and tool spans are emitted with OpenTelemetry **GenAI semantic
conventions** (``gen_ai.*`` attributes, ``chat {model}`` /
``execute_tool {tool}`` span names) so GenAI-aware backends render them
natively; every span also keeps the full ``vincio.*`` attribute set.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.errors import ConfigError
from .spans import Span, Trace

__all__ = ["OTelExporter"]

logger = logging.getLogger("vincio.observability.otel")


def _genai_span(span: Span) -> tuple[str, dict[str, Any]]:
    """Span name + gen_ai.* attributes per the OTel GenAI semantic conventions."""
    if span.type == "model_call":
        model = str(span.attributes.get("model") or "unknown")
        attributes: dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.system": "vincio",
            "gen_ai.request.model": model,
        }
        if span.attributes.get("output_tokens") is not None:
            attributes["gen_ai.usage.output_tokens"] = int(span.attributes["output_tokens"])
        if span.attributes.get("input_tokens") is not None:
            attributes["gen_ai.usage.input_tokens"] = int(span.attributes["input_tokens"])
        if span.attributes.get("finish"):
            attributes["gen_ai.response.finish_reasons"] = [str(span.attributes["finish"])]
        return f"chat {model}", attributes
    if span.type == "tool_call":
        tool = str(span.attributes.get("tool") or span.name)
        return f"execute_tool {tool}", {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool,
        }
    return f"{span.type}:{span.name}", {}


class OTelExporter:
    """Re-emits Vincio traces through an OpenTelemetry TracerProvider.

    Spans keep their original timestamps and hierarchy, so any OTLP-compatible
    backend (Jaeger, Datadog, Honeycomb, Grafana Tempo, collector) receives a
    faithful copy of the Vincio trace.
    """

    def __init__(self, tracer_provider=None, service_name: str = "vincio") -> None:
        try:
            from opentelemetry import trace as otel_trace
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ConfigError(
                'OpenTelemetry export requires: pip install "vincio[otel]"'
            ) from exc
        self._otel_trace = otel_trace
        if tracer_provider is None:
            tracer_provider = otel_trace.get_tracer_provider()
        self._tracer = tracer_provider.get_tracer(service_name)

    def export(self, trace: Trace) -> None:
        try:
            self._export(trace)
        except Exception:  # noqa: BLE001 - exporters must not break runs
            logger.exception("OTel export failed for %s", trace.id)

    def _export(self, trace: Trace) -> None:
        from opentelemetry.trace import Status, StatusCode

        def ns(dt) -> int:
            return int(dt.timestamp() * 1_000_000_000)

        root_attributes: dict[str, Any] = {
            "vincio.trace_id": trace.id,
            "vincio.run_id": trace.run_id or "",
            "vincio.app": trace.app_name,
            **{f"vincio.attr.{k}": str(v) for k, v in trace.attributes.items()},
        }
        if trace.session_id:
            root_attributes["gen_ai.conversation.id"] = trace.session_id
            root_attributes["vincio.session_id"] = trace.session_id
        if trace.thread_id:
            root_attributes["vincio.thread_id"] = trace.thread_id
        for score_name, score in trace.scores.items():
            root_attributes[f"vincio.score.{score_name}"] = score
        root = self._tracer.start_span(
            name=f"vincio.run:{trace.app_name or 'app'}",
            start_time=ns(trace.start_time),
            attributes=root_attributes,
        )
        otel_ids: dict[str | None, object] = {None: root}
        # Parents appear before children in span list order; sort by start time
        # and resolve hierarchy via recorded parent ids.
        from opentelemetry import context as otel_context
        from opentelemetry.trace import set_span_in_context

        for span in sorted(trace.spans, key=lambda s: s.start_time):
            parent = otel_ids.get(span.parent_id, root)
            ctx = set_span_in_context(parent, otel_context.get_current())
            span_name, genai_attributes = _genai_span(span)
            otel_span = self._tracer.start_span(
                name=span_name,
                context=ctx,
                start_time=ns(span.start_time),
                attributes={
                    "vincio.span_id": span.id,
                    **genai_attributes,
                    **{f"vincio.attr.{k}": str(v) for k, v in span.attributes.items()},
                    **{f"vincio.score.{k}": v for k, v in span.scores.items()},
                },
            )
            for event in span.events:
                otel_span.add_event(
                    event.name,
                    attributes={k: str(v) for k, v in event.attributes.items()},
                    timestamp=ns(event.timestamp),
                )
            if span.status == "error":
                otel_span.set_status(Status(StatusCode.ERROR, span.error or ""))
            otel_ids[span.id] = otel_span
            otel_span.end(end_time=ns(span.end_time or span.start_time))
        if trace.status == "error":
            root.set_status(Status(StatusCode.ERROR, str(trace.attributes.get("error", ""))))
        root.end(end_time=ns(trace.end_time or trace.start_time))
