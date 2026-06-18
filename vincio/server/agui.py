"""Generative UI / AG-UI streaming protocol.

The field streams *structured UI events* to interactive frontends. **AG-UI** (and
the overlapping MCP-UI surface) standardize that as a small set of typed events —
run lifecycle, streaming text messages, tool calls, and state snapshots/deltas.
This module translates a Vincio run's native stream
(:class:`~vincio.core.types.RunStreamEvent`, and the agent/crew
:class:`~vincio.agents.executor.AgentEvent` / :class:`~vincio.agents.crew.CrewEvent`)
into AG-UI events, so a run drives an interactive UI **over the same SSE/astream
path** — inheriting the run's provenance, budget metering, and audit. It is one
streamed run, not a bolt-on UI layer.

The translators are pure and dependency-free (the server wires them onto SSE);
``AGUIEvent.to_sse()`` renders the canonical ``data: {json}\\n\\n`` framing with
the AG-UI camelCase wire shape.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from ..core.types import new_id

__all__ = ["AGUIEventType", "AGUIEvent", "run_stream_to_agui", "agent_stream_to_agui", "agui_sse"]


class AGUIEventType:
    """The AG-UI event-type constants this implementation emits."""

    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"
    CUSTOM = "CUSTOM"


# Python field -> AG-UI camelCase wire key. Only set fields are emitted.
_WIRE_KEYS = {
    "message_id": "messageId",
    "role": "role",
    "delta": "delta",
    "tool_call_id": "toolCallId",
    "tool_call_name": "toolCallName",
    "parent_message_id": "parentMessageId",
    "content": "content",
    "snapshot": "snapshot",
    "step_name": "stepName",
    "thread_id": "threadId",
    "run_id": "runId",
    "result": "result",
    "message": "message",
    "name": "name",
    "value": "value",
}


class AGUIEvent(BaseModel):
    """One AG-UI event. Fields are populated per ``type``; ``to_wire`` emits the
    canonical camelCase shape, omitting unset fields."""

    type: str
    message_id: str | None = None
    role: str | None = None
    delta: str | None = None
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    parent_message_id: str | None = None
    content: str | None = None
    snapshot: Any = None
    step_name: str | None = None
    thread_id: str | None = None
    run_id: str | None = None
    result: Any = None
    message: str | None = None
    name: str | None = None
    value: Any = None

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"type": self.type}
        for attr, key in _WIRE_KEYS.items():
            val = getattr(self, attr)
            if val is not None:
                wire[key] = val
        return wire

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.to_wire(), default=str)}\n\n"


def agui_sse(event: AGUIEvent) -> str:
    """Render an AG-UI event as one SSE frame."""
    return event.to_sse()


class _MessageTracker:
    """Tracks the single open assistant text message across a stream."""

    def __init__(self) -> None:
        self.message_id: str | None = None

    def start_if_needed(self) -> list[AGUIEvent]:
        if self.message_id is None:
            self.message_id = new_id("msg")
            return [AGUIEvent(type=AGUIEventType.TEXT_MESSAGE_START, message_id=self.message_id, role="assistant")]
        return []

    def end_if_open(self) -> list[AGUIEvent]:
        if self.message_id is not None:
            mid = self.message_id
            self.message_id = None
            return [AGUIEvent(type=AGUIEventType.TEXT_MESSAGE_END, message_id=mid)]
        return []


async def run_stream_to_agui(
    events: AsyncIterator[Any], *, thread_id: str | None = None, run_id: str | None = None
) -> AsyncIterator[AGUIEvent]:
    """Translate a :class:`RunStreamEvent` stream into AG-UI events."""
    thread_id = thread_id or new_id("thread")
    run_id = run_id or new_id("run")
    yield AGUIEvent(type=AGUIEventType.RUN_STARTED, thread_id=thread_id, run_id=run_id)

    message = _MessageTracker()
    tool_seq = 0
    async for event in events:
        etype = getattr(event, "type", None)
        if etype == "text_delta" and getattr(event, "text", None):
            for ev in message.start_if_needed():
                yield ev
            yield AGUIEvent(
                type=AGUIEventType.TEXT_MESSAGE_CONTENT, message_id=message.message_id, delta=event.text
            )
        elif etype == "tool_call":
            tool_seq += 1
            tool_id = f"tool_{tool_seq}"
            yield AGUIEvent(
                type=AGUIEventType.TOOL_CALL_START,
                tool_call_id=tool_id,
                tool_call_name=getattr(event, "tool_name", None),
                parent_message_id=message.message_id,
            )
            yield AGUIEvent(type=AGUIEventType.TOOL_CALL_END, tool_call_id=tool_id)
        elif etype == "tool_result":
            tool_id = f"tool_{tool_seq}" if tool_seq else "tool_0"
            result = getattr(event, "tool_result", None)
            content = ""
            if result is not None:
                content = json.dumps(getattr(result, "output", None), default=str)
            yield AGUIEvent(type=AGUIEventType.TOOL_CALL_RESULT, tool_call_id=tool_id, content=content)
        elif etype == "partial_output" and getattr(event, "partial_output", None) is not None:
            yield AGUIEvent(type=AGUIEventType.STATE_SNAPSHOT, snapshot=event.partial_output)
        elif etype == "stage" and getattr(event, "stage", None):
            yield AGUIEvent(type=AGUIEventType.STEP_FINISHED, step_name=event.stage)
        elif etype == "done":
            for ev in message.end_if_open():
                yield ev
            result = getattr(event, "result", None)
            snapshot = result.model_dump(mode="json", exclude={"evidence", "raw_text"}) if result is not None else None
            if snapshot is not None:
                yield AGUIEvent(type=AGUIEventType.STATE_SNAPSHOT, snapshot=snapshot)
            yield AGUIEvent(
                type=AGUIEventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id, result=snapshot
            )
        elif etype == "error":
            for ev in message.end_if_open():
                yield ev
            yield AGUIEvent(type=AGUIEventType.RUN_ERROR, message=getattr(event, "error", "error"))


async def agent_stream_to_agui(
    events: AsyncIterator[Any], *, thread_id: str | None = None, run_id: str | None = None
) -> AsyncIterator[AGUIEvent]:
    """Translate an agent/crew event stream (:class:`AgentEvent` / :class:`CrewEvent`)
    into AG-UI events. Member-tagged crew events carry the member as ``stepName``
    on tool calls so a UI can group activity by agent."""
    thread_id = thread_id or new_id("thread")
    run_id = run_id or new_id("run")
    message = _MessageTracker()
    tool_seq = 0
    started = False
    async for event in events:
        etype = getattr(event, "type", None)
        if etype == "run_start":
            started = True
            yield AGUIEvent(type=AGUIEventType.RUN_STARTED, thread_id=thread_id, run_id=run_id)
        elif etype in ("step_start", "member_start", "delegation"):
            yield AGUIEvent(
                type=AGUIEventType.STEP_STARTED,
                step_name=getattr(event, "step", "") or getattr(event, "member", ""),
            )
        elif etype in ("step_end", "member_end"):
            yield AGUIEvent(
                type=AGUIEventType.STEP_FINISHED,
                step_name=getattr(event, "step", "") or getattr(event, "member", ""),
            )
        elif etype == "text_delta" and getattr(event, "text", None):
            for ev in message.start_if_needed():
                yield ev
            yield AGUIEvent(
                type=AGUIEventType.TEXT_MESSAGE_CONTENT, message_id=message.message_id, delta=event.text
            )
        elif etype == "tool_call":
            tool_seq += 1
            tool_id = f"tool_{tool_seq}"
            yield AGUIEvent(
                type=AGUIEventType.TOOL_CALL_START,
                tool_call_id=tool_id,
                tool_call_name=getattr(event, "tool_name", None),
                step_name=getattr(event, "member", None) or None,
            )
            args = getattr(event, "arguments", None)
            if args:
                yield AGUIEvent(type=AGUIEventType.TOOL_CALL_ARGS, tool_call_id=tool_id, delta=json.dumps(args, default=str))
            yield AGUIEvent(type=AGUIEventType.TOOL_CALL_END, tool_call_id=tool_id)
        elif etype == "tool_result":
            tool_id = f"tool_{tool_seq}" if tool_seq else "tool_0"
            yield AGUIEvent(
                type=AGUIEventType.TOOL_CALL_RESULT,
                tool_call_id=tool_id,
                content=json.dumps(getattr(event, "result", None), default=str),
            )
        elif etype == "done":
            if not started:
                yield AGUIEvent(type=AGUIEventType.RUN_STARTED, thread_id=thread_id, run_id=run_id)
            for ev in message.end_if_open():
                yield ev
            yield AGUIEvent(type=AGUIEventType.STATE_SNAPSHOT, snapshot=getattr(event, "payload", None))
            yield AGUIEvent(
                type=AGUIEventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id,
                result=getattr(event, "result", None),
            )
        elif etype == "error":
            for ev in message.end_if_open():
                yield ev
            yield AGUIEvent(
                type=AGUIEventType.RUN_ERROR, message=getattr(event, "error", None) or getattr(event, "status", "error")
            )
