"""Sessions: group traces by ``session_id`` into reviewable units.

A session is a derived view over exported traces — there is no separate
session store to keep in sync. Build sessions from any list of traces
(``InMemoryExporter.traces``, ``JSONLExporter.load_all()``) and get per-session
aggregates: run count, duration, cost, error rate, scores, and feedback.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .spans import Feedback, Trace

__all__ = ["Session", "sessions_from_traces", "record_feedback"]


class Session(BaseModel):
    """A grouped view of one session's traces (threaded runs)."""

    id: str
    app_name: str = ""
    user_id: str | None = None
    tenant_id: str | None = None
    traces: list[Trace] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def trace_ids(self) -> list[str]:
        return [t.id for t in self.traces]

    @property
    def start_time(self) -> datetime | None:
        return min((t.start_time for t in self.traces), default=None)

    @property
    def end_time(self) -> datetime | None:
        ends = [t.end_time for t in self.traces if t.end_time is not None]
        return max(ends, default=None)

    @property
    def duration_ms(self) -> int:
        if self.start_time is None or self.end_time is None:
            return 0
        return int((self.end_time - self.start_time).total_seconds() * 1000)

    @property
    def feedback(self) -> list[Feedback]:
        return [item for trace in self.traces for item in trace.feedback]

    @property
    def error_rate(self) -> float:
        if not self.traces:
            return 0.0
        return sum(1 for t in self.traces if t.status == "error") / len(self.traces)

    def mean_score(self, name: str) -> float | None:
        values = [t.scores[name] for t in self.traces if name in t.scores]
        return sum(values) / len(values) if values else None

    def mean_feedback(self, key: str = "user_rating") -> float | None:
        values = [f.score for f in self.feedback if f.key == key and f.score is not None]
        return sum(values) / len(values) if values else None

    def summary(self) -> dict[str, Any]:
        score_names = {name for trace in self.traces for name in trace.scores}
        return {
            "session_id": self.id,
            "app_name": self.app_name,
            "user_id": self.user_id,
            "runs": len(self.traces),
            "duration_ms": self.duration_ms,
            "error_rate": round(self.error_rate, 4),
            "scores": {name: round(self.mean_score(name) or 0.0, 4) for name in sorted(score_names)},
            "feedback_count": len(self.feedback),
            "mean_feedback": self.mean_feedback(),
        }


def sessions_from_traces(traces: list[Trace]) -> list[Session]:
    """Group traces into sessions by ``session_id``; traces without one are
    skipped. Duplicate trace ids collapse to the latest record (JSONL
    re-exports act as updates). Sessions are ordered by first trace start."""
    latest = {trace.id: trace for trace in traces}
    grouped: dict[str, list[Trace]] = {}
    for trace in latest.values():
        if trace.session_id:
            grouped.setdefault(trace.session_id, []).append(trace)
    sessions = []
    for session_id, items in grouped.items():
        items.sort(key=lambda t: t.start_time)
        first = items[0]
        sessions.append(
            Session(
                id=session_id,
                app_name=first.app_name,
                user_id=first.user_id,
                tenant_id=first.tenant_id,
                traces=items,
            )
        )
    sessions.sort(key=lambda s: s.start_time or datetime.min)
    return sessions


def record_feedback(
    trace: Trace,
    *,
    key: str = "user_rating",
    score: float | None = None,
    comment: str = "",
    user_id: str | None = None,
    exporter: Any = None,
) -> Feedback:
    """Attach feedback to a trace and (optionally) re-export it so the
    persisted copy carries the feedback. ``JSONLExporter`` resolves loads to
    the latest record per trace id, so re-exporting acts as an update."""
    item = trace.add_feedback(key=key, score=score, comment=comment, user_id=user_id)
    if exporter is not None:
        exporter.export(trace)
    return item
