"""OpenTelemetry export. Requires ``pip install "vincio[otel]"``."""

from __future__ import annotations

import logging

from ..core.errors import ConfigError
from .spans import Trace

__all__ = ["OTelExporter"]

logger = logging.getLogger("vincio.observability.otel")


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

        root = self._tracer.start_span(
            name=f"vincio.run:{trace.app_name or 'app'}",
            start_time=ns(trace.start_time),
            attributes={
                "vincio.trace_id": trace.id,
                "vincio.run_id": trace.run_id or "",
                "vincio.app": trace.app_name,
                **{f"vincio.attr.{k}": str(v) for k, v in trace.attributes.items()},
            },
        )
        otel_ids: dict[str | None, object] = {None: root}
        # Parents appear before children in span list order; sort by start time
        # and resolve hierarchy via recorded parent ids.
        from opentelemetry import context as otel_context
        from opentelemetry.trace import set_span_in_context

        for span in sorted(trace.spans, key=lambda s: s.start_time):
            parent = otel_ids.get(span.parent_id, root)
            ctx = set_span_in_context(parent, otel_context.get_current())
            otel_span = self._tracer.start_span(
                name=f"{span.type}:{span.name}",
                context=ctx,
                start_time=ns(span.start_time),
                attributes={
                    "vincio.span_id": span.id,
                    **{f"vincio.attr.{k}": str(v) for k, v in span.attributes.items()},
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
