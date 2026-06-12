"""Tracer: span lifecycle management with automatic nesting.

Usage::

    tracer = Tracer(app_name="contract_review", exporter=JSONLExporter())
    with tracer.trace(run_id="run_1") as trace:
        with tracer.span("retrieval", type="retrieval") as span:
            span.set(query="termination clauses", top_k=8)

Nesting uses :mod:`contextvars`, so it is correct under asyncio concurrency.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .exporters import NullExporter, TraceExporter
from .spans import Span, SpanType, Trace

__all__ = ["Tracer", "trace_diff", "trace_replay_plan"]

_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "vincio_current_trace", default=None
)
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "vincio_current_span", default=None
)


class Tracer:
    def __init__(
        self,
        app_name: str = "",
        exporter: TraceExporter | None = None,
        *,
        sample_rate: float = 1.0,
    ) -> None:
        self.app_name = app_name
        self.exporter: TraceExporter = exporter or NullExporter()
        self.sample_rate = sample_rate
        self._sample_counter = 0

    # -- trace lifecycle -----------------------------------------------------

    def _should_sample(self) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        # Deterministic sampling: keep every Nth trace.
        self._sample_counter += 1
        period = max(1, round(1.0 / self.sample_rate))
        return self._sample_counter % period == 0

    @contextmanager
    def trace(
        self,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        parent_id: str | None = None,
        **attributes: Any,
    ) -> Iterator[Trace]:
        trace = Trace(
            app_name=self.app_name,
            run_id=run_id,
            session_id=session_id,
            thread_id=thread_id,
            user_id=user_id,
            tenant_id=tenant_id,
            parent_id=parent_id,
            attributes=dict(attributes),
        )
        sampled = self._should_sample()
        trace_token = _current_trace.set(trace)
        span_token = _current_span.set(None)
        try:
            yield trace
            trace.status = "ok" if trace.status == "running" else trace.status
        except BaseException as exc:
            trace.status = "error"
            trace.attributes["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            from ..core.utils import utcnow

            trace.end_time = utcnow()
            _current_span.reset(span_token)
            _current_trace.reset(trace_token)
            if sampled:
                self.exporter.export(trace)

    @contextmanager
    def span(self, name: str, *, type: SpanType = "custom", **attributes: Any) -> Iterator[Span]:
        trace = _current_trace.get()
        parent = _current_span.get()
        span = Span(
            name=name,
            type=type,
            trace_id=trace.id if trace else "",
            parent_id=parent.id if parent else None,
            attributes=dict(attributes),
        )
        if trace is not None:
            trace.spans.append(span)
        token = _current_span.set(span)
        try:
            yield span
            span.end(status="ok")
        except BaseException as exc:
            span.end(status="error", error=f"{type_name(exc)}: {exc}")
            raise
        finally:
            _current_span.reset(token)

    # -- accessors -------------------------------------------------------------

    @property
    def current_trace(self) -> Trace | None:
        return _current_trace.get()

    @property
    def current_span(self) -> Span | None:
        return _current_span.get()

    def event(self, name: str, **attributes: Any) -> None:
        """Attach an event to the current span (no-op outside a span)."""
        span = _current_span.get()
        if span is not None:
            span.add_event(name, **attributes)


def type_name(exc: BaseException) -> str:
    return type(exc).__name__


# ---------------------------------------------------------------------------
# Trace tooling for the CLI
# ---------------------------------------------------------------------------


def trace_diff(a: Trace, b: Trace) -> dict[str, Any]:
    """Structural diff between two traces: span names/types/status/durations."""

    def signature(trace: Trace) -> list[tuple[str, str]]:
        return [(s.type, s.name) for s in trace.spans]

    sig_a, sig_b = signature(a), signature(b)
    only_a = [f"{t}:{n}" for t, n in sig_a if (t, n) not in sig_b]
    only_b = [f"{t}:{n}" for t, n in sig_b if (t, n) not in sig_a]
    common = [(t, n) for t, n in sig_a if (t, n) in sig_b]

    duration_changes: list[dict[str, Any]] = []
    status_changes: list[dict[str, Any]] = []
    for span_type, name in common:
        sa = next(s for s in a.spans if (s.type, s.name) == (span_type, name))
        sb = next(s for s in b.spans if (s.type, s.name) == (span_type, name))
        if sa.status != sb.status:
            status_changes.append(
                {"span": f"{span_type}:{name}", "a": sa.status, "b": sb.status}
            )
        if sa.duration_ms or sb.duration_ms:
            delta = sb.duration_ms - sa.duration_ms
            if abs(delta) > 0:
                duration_changes.append(
                    {
                        "span": f"{span_type}:{name}",
                        "a_ms": sa.duration_ms,
                        "b_ms": sb.duration_ms,
                        "delta_ms": delta,
                    }
                )
    return {
        "trace_a": a.id,
        "trace_b": b.id,
        "status": {"a": a.status, "b": b.status},
        "duration_ms": {"a": a.duration_ms, "b": b.duration_ms},
        "spans_only_in_a": only_a,
        "spans_only_in_b": only_b,
        "status_changes": status_changes,
        "duration_changes": duration_changes,
    }


def trace_replay_plan(trace: Trace) -> dict[str, Any]:
    """Extract the inputs needed to replay a trace deterministically."""
    model_calls = [
        {
            "span_id": s.id,
            "model": s.attributes.get("model"),
            "request_hash": s.attributes.get("request_hash"),
            "messages": s.attributes.get("messages"),
            "response_text": s.attributes.get("response_text"),
        }
        for s in trace.spans
        if s.type == "model_call"
    ]
    tool_calls = [
        {
            "span_id": s.id,
            "tool": s.attributes.get("tool"),
            "arguments": s.attributes.get("arguments"),
            "output": s.attributes.get("output"),
        }
        for s in trace.spans
        if s.type == "tool_call"
    ]
    return {
        "trace_id": trace.id,
        "app_name": trace.app_name,
        "run_id": trace.run_id,
        "input": trace.attributes.get("input"),
        "model_calls": model_calls,
        "tool_calls": tool_calls,
    }
