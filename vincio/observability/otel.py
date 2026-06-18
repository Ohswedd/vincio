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


# Span types that represent invoking an agent (crew member, graph node, a
# composed step, or a single agent run). These map to the OTel
# GenAI **agentic** convention: an ``invoke_agent`` operation with
# ``gen_ai.agent.*`` attributes, so agent activity is first-class telemetry.
_AGENT_SPAN_TYPES = frozenset({"agent", "crew", "crew_agent", "graph_node", "compose_node"})


def _genai_span(span: Span) -> tuple[str, dict[str, Any]]:
    """Span name + gen_ai.* attributes per the OTel GenAI semantic conventions
    (including the agentic conventions for agent/tool spans)."""
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
        if span.attributes.get("cost_usd") is not None:
            # Cost is a first-class telemetry signal, not just a vincio.attr.
            attributes["gen_ai.usage.cost"] = float(span.attributes["cost_usd"])
        if span.attributes.get("finish"):
            attributes["gen_ai.response.finish_reasons"] = [str(span.attributes["finish"])]
        return f"chat {model}", attributes
    if span.type == "tool_call":
        tool = str(span.attributes.get("tool") or span.name)
        return f"execute_tool {tool}", {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.system": "vincio",
            "gen_ai.tool.name": tool,
        }
    if span.type in _AGENT_SPAN_TYPES:
        agent_name = str(
            span.attributes.get("agent")
            or span.attributes.get("role")
            or span.attributes.get("node")
            or span.name
        )
        agent_attrs: dict[str, Any] = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.system": "vincio",
            "gen_ai.agent.name": agent_name,
        }
        agent_id = span.attributes.get("agent_id") or span.attributes.get("id")
        if agent_id is not None:
            agent_attrs["gen_ai.agent.id"] = str(agent_id)
        return f"invoke_agent {agent_name}", agent_attrs
    return f"{span.type}:{span.name}", {}


class OTelExporter:
    """Re-emits Vincio traces through an OpenTelemetry TracerProvider.

    Spans keep their original timestamps and hierarchy, so any OTLP-compatible
    backend (Jaeger, Datadog, Honeycomb, Grafana Tempo, collector) receives a
    faithful copy of the Vincio trace.
    """

    def __init__(
        self,
        tracer_provider=None,
        service_name: str = "vincio",
        *,
        meter_provider=None,
        content_policy: Any = None,
    ) -> None:
        try:
            from opentelemetry import trace as otel_trace
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ConfigError(
                'OpenTelemetry export requires: pip install "vincio[otel]"'
            ) from exc
        from .redaction import ContentCapturePolicy

        # prompt/completion content is gated + redacted at the export
        # boundary. Default policy captures nothing — structural telemetry
        # (model/tokens/cost/scores) still exports; raw content does not.
        self.content_policy = content_policy or ContentCapturePolicy()
        self._otel_trace = otel_trace
        if tracer_provider is None:
            tracer_provider = otel_trace.get_tracer_provider()
        self._tracer = tracer_provider.get_tracer(service_name)
        # unified telemetry: the same trace is fanned out to spans *and* to
        # OTel metric histograms (token usage, operation duration, cost), so the
        # standard GenAI metrics are emitted once from one source. Metrics are
        # best-effort: if the metrics API is unavailable, spans still export.
        self._token_hist = self._duration_hist = self._cost_hist = None
        try:  # pragma: no cover - exercised only with the otel SDK installed
            from opentelemetry import metrics as otel_metrics

            meter = (meter_provider or otel_metrics.get_meter_provider()).get_meter(service_name)
            self._token_hist = meter.create_histogram(
                "gen_ai.client.token.usage", unit="token", description="GenAI token usage"
            )
            self._duration_hist = meter.create_histogram(
                "gen_ai.client.operation.duration", unit="s", description="GenAI operation duration"
            )
            self._cost_hist = meter.create_histogram(
                "gen_ai.client.operation.cost", unit="usd", description="GenAI operation cost"
            )
        except Exception:  # noqa: BLE001 - metrics are optional
            logger.debug("OTel metrics unavailable; exporting spans only", exc_info=True)

    def _record_metrics(self, span: Span, genai_attributes: dict[str, Any]) -> None:
        """Emit GenAI metric histograms for a span (best-effort)."""
        operation = genai_attributes.get("gen_ai.operation.name")
        if operation is None:
            return
        base = {"gen_ai.operation.name": operation, "gen_ai.system": "vincio"}
        model = genai_attributes.get("gen_ai.request.model")
        if model:
            base["gen_ai.request.model"] = model
        if self._duration_hist is not None and span.end_time is not None:
            duration_s = max(0.0, (span.end_time - span.start_time).total_seconds())
            self._duration_hist.record(duration_s, attributes=base)
        if self._token_hist is not None:
            input_tokens = genai_attributes.get("gen_ai.usage.input_tokens")
            output_tokens = genai_attributes.get("gen_ai.usage.output_tokens")
            if input_tokens is not None:
                self._token_hist.record(int(input_tokens), attributes={**base, "gen_ai.token.type": "input"})
            if output_tokens is not None:
                self._token_hist.record(int(output_tokens), attributes={**base, "gen_ai.token.type": "output"})
        cost = genai_attributes.get("gen_ai.usage.cost")
        if self._cost_hist is not None and cost is not None:
            self._cost_hist.record(float(cost), attributes=base)

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
            **{
                f"vincio.attr.{k}": str(v)
                for k, v in self.content_policy.scrub_attributes(trace.attributes).items()
            },
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
                    **{
                        f"vincio.attr.{k}": str(v)
                        for k, v in self.content_policy.scrub_attributes(span.attributes).items()
                    },
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
            self._record_metrics(span, genai_attributes)
        if trace.status == "error":
            root.set_status(Status(StatusCode.ERROR, str(trace.attributes.get("error", ""))))
        root.end(end_time=ns(trace.end_time or trace.start_time))
