"""Span and trace data models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.utils import new_id, utcnow

__all__ = ["SpanType", "TraceEvent", "Feedback", "Span", "Trace"]

SpanType = Literal[
    "run",
    "input",
    "memory",
    "retrieval",
    "context_compile",
    "prompt_render",
    "model_call",
    "tool_call",
    "agent_step",
    "workflow_step",
    "crew",
    "crew_agent",
    "graph_node",
    "compose_node",
    "output_validation",
    "eval",
    "memory_write",
    "cache",
    "security",
    "custom",
]


class TraceEvent(BaseModel):
    name: str
    timestamp: datetime = Field(default_factory=utcnow)
    attributes: dict[str, Any] = Field(default_factory=dict)


class Feedback(BaseModel):
    """User or system feedback attached to a trace (LangSmith-style).

    ``key`` names the feedback channel ("user_rating", "thumbs", "correction"),
    ``score`` is a normalized value (e.g. 0..1 or -1/1 for thumbs).
    """

    key: str = "user_rating"
    score: float | None = None
    comment: str = ""
    user_id: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)
    attributes: dict[str, Any] = Field(default_factory=dict)


class Span(BaseModel):
    id: str = Field(default_factory=lambda: new_id("span"))
    trace_id: str = ""
    parent_id: str | None = None
    name: str
    type: SpanType = "custom"
    start_time: datetime = Field(default_factory=utcnow)
    end_time: datetime | None = None
    status: Literal["running", "ok", "error"] = "running"
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[TraceEvent] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> int:
        if self.end_time is None:
            return 0
        return int((self.end_time - self.start_time).total_seconds() * 1000)

    def set(self, **attributes: Any) -> Span:
        self.attributes.update(attributes)
        return self

    def add_score(self, name: str, value: float) -> Span:
        """Attach an eval/quality score to this span (e.g. a metric result)."""
        self.scores[name] = float(value)
        return self

    def add_event(self, name: str, **attributes: Any) -> TraceEvent:
        event = TraceEvent(name=name, attributes=attributes)
        self.events.append(event)
        return event

    def end(self, *, status: Literal["ok", "error"] = "ok", error: str | None = None) -> Span:
        self.end_time = utcnow()
        self.status = status
        self.error = error
        return self


class Trace(BaseModel):
    id: str = Field(default_factory=lambda: new_id("trace"))
    app_name: str = ""
    run_id: str | None = None
    session_id: str | None = None
    thread_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    parent_id: str | None = None
    start_time: datetime = Field(default_factory=utcnow)
    end_time: datetime | None = None
    status: Literal["running", "ok", "error"] = "running"
    spans: list[Span] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, float] = Field(default_factory=dict)
    feedback: list[Feedback] = Field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        if self.end_time is None:
            return 0
        return int((self.end_time - self.start_time).total_seconds() * 1000)

    def add_score(self, name: str, value: float) -> Trace:
        """Attach an eval/quality score to the whole trace."""
        self.scores[name] = float(value)
        return self

    def add_feedback(
        self,
        *,
        key: str = "user_rating",
        score: float | None = None,
        comment: str = "",
        user_id: str | None = None,
        **attributes: Any,
    ) -> Feedback:
        """Record user/system feedback on this trace."""
        item = Feedback(
            key=key, score=score, comment=comment, user_id=user_id, attributes=dict(attributes)
        )
        self.feedback.append(item)
        return item

    def span_tree(self) -> list[dict[str, Any]]:
        """Spans nested by parent for display/debugging."""
        children: dict[str | None, list[Span]] = {}
        for span in self.spans:
            children.setdefault(span.parent_id, []).append(span)

        def build(parent_id: str | None) -> list[dict[str, Any]]:
            return [
                {
                    "id": s.id,
                    "name": s.name,
                    "type": s.type,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "attributes": s.attributes,
                    "children": build(s.id),
                }
                for s in sorted(children.get(parent_id, []), key=lambda s: s.start_time)
            ]

        return build(None)
