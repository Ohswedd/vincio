"""Tests for AG-UI streaming + agent/crew astream + MCP UI resources (2.2)."""

from __future__ import annotations

import json

import pytest

from vincio.agents import AgentExecutor, Crew
from vincio.agents.planner import Planner
from vincio.core.types import Budget, RunResult, RunStreamEvent, ToolResult
from vincio.mcp import MCPUIResource, build_app_server, connect_in_process
from vincio.providers.mock import MockProvider
from vincio.server.agui import (
    AGUIEvent,
    AGUIEventType,
    agent_stream_to_agui,
    run_stream_to_agui,
)
from vincio.tools.registry import ToolRegistry
from vincio.tools.runtime import ToolRuntime


async def _aiter(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_run_stream_to_agui_maps_text_and_lifecycle():
    events = [
        RunStreamEvent(type="text_delta", text="Hello "),
        RunStreamEvent(type="text_delta", text="world"),
        RunStreamEvent(type="done", result=RunResult(output="Hello world")),
    ]
    out = [e async for e in run_stream_to_agui(_aiter(events))]
    types = [e.type for e in out]
    assert types[0] == AGUIEventType.RUN_STARTED
    assert AGUIEventType.TEXT_MESSAGE_START in types
    assert types.count(AGUIEventType.TEXT_MESSAGE_CONTENT) == 2
    assert AGUIEventType.TEXT_MESSAGE_END in types
    assert types[-1] == AGUIEventType.RUN_FINISHED
    # Text deltas precede the END marker.
    assert types.index(AGUIEventType.TEXT_MESSAGE_END) > types.index(AGUIEventType.TEXT_MESSAGE_START)


@pytest.mark.asyncio
async def test_run_stream_to_agui_tool_events():
    events = [
        RunStreamEvent(type="tool_call", tool_name="search"),
        RunStreamEvent(type="tool_result", tool_result=ToolResult(call_id="c", tool_name="search", output={"hits": 3})),
        RunStreamEvent(type="done", result=RunResult(output="ok")),
    ]
    out = [e async for e in run_stream_to_agui(_aiter(events))]
    types = [e.type for e in out]
    assert AGUIEventType.TOOL_CALL_START in types
    assert AGUIEventType.TOOL_CALL_RESULT in types
    result_ev = next(e for e in out if e.type == AGUIEventType.TOOL_CALL_RESULT)
    assert "hits" in (result_ev.content or "")


@pytest.mark.asyncio
async def test_run_stream_to_agui_error():
    events = [RunStreamEvent(type="error", error="boom")]
    out = [e async for e in run_stream_to_agui(_aiter(events))]
    assert out[-1].type == AGUIEventType.RUN_ERROR
    assert out[-1].message == "boom"


def test_agui_event_wire_is_camelcase():
    ev = AGUIEvent(type=AGUIEventType.TEXT_MESSAGE_START, message_id="m1", role="assistant")
    wire = ev.to_wire()
    assert wire == {"type": "TEXT_MESSAGE_START", "messageId": "m1", "role": "assistant"}
    sse = ev.to_sse()
    assert sse.startswith("data: ") and sse.endswith("\n\n")
    assert json.loads(sse[6:].strip())["messageId"] == "m1"


@pytest.mark.asyncio
async def test_executor_astream_to_agui_end_to_end():
    reg = ToolRegistry()

    @reg.register()
    def probe(q: str) -> dict:
        """Probe."""
        return {"q": q}

    looping = MockProvider(responder=lambda req: {"tool_call": {"name": "probe", "arguments": {"q": "x"}}})
    ex = AgentExecutor(
        looping, model="mock-1", planner=Planner(mode="react"),
        tool_runtime=ToolRuntime(reg, cache_enabled=False), tool_specs=reg.specs(),
    )
    stream = ex.astream("do work", budget=Budget(max_steps=3, max_tool_calls=2))
    out = [e async for e in agent_stream_to_agui(stream)]
    types = [e.type for e in out]
    assert types[0] == AGUIEventType.RUN_STARTED
    assert AGUIEventType.TOOL_CALL_START in types
    assert AGUIEventType.TOOL_CALL_ARGS in types
    assert types[-1] == AGUIEventType.RUN_FINISHED


@pytest.mark.asyncio
async def test_crew_astream_forwards_member_events():
    good = MockProvider(default_text="member done")
    crew = Crew("team")
    for name in ("alpha", "beta"):
        crew.add(name, AgentExecutor(good, model="mock-1", planner=Planner(mode="static")))
    events = [e async for e in crew.astream("objective")]
    types = [e.type for e in events]
    assert types[0] == "run_start" and types[-1] == "done"
    assert types.count("member_start") == 2
    assert "text_delta" in types  # member text forwarded as a crew event
    # done payload reconstructs a CrewResult shape.
    assert events[-1].payload["status"] in ("succeeded", "failed")


@pytest.mark.asyncio
async def test_crew_astream_to_agui():
    good = MockProvider(default_text="done")
    crew = Crew("team")
    crew.add("alpha", AgentExecutor(good, model="mock-1", planner=Planner(mode="static")))
    out = [e async for e in agent_stream_to_agui(crew.astream("objective"))]
    types = [e.type for e in out]
    assert AGUIEventType.STEP_STARTED in types
    assert types[-1] == AGUIEventType.RUN_FINISHED


@pytest.mark.asyncio
async def test_mcp_ui_resource_served():
    from vincio import ContextApp

    app = ContextApp(name="ui", provider=MockProvider(), model="mock-1")
    ui = MCPUIResource.from_html("ui://dashboard", "<h1>Hi</h1>", name="dashboard")
    server = build_app_server(app, ui_resources=[ui])
    client = connect_in_process(server)
    await client.initialize()
    resources = await client.list_resources()
    uris = {r.uri for r in resources}
    assert "ui://dashboard" in uris
    body = await client.read_resource("ui://dashboard")
    assert "<h1>Hi</h1>" in body


@pytest.mark.asyncio
async def test_mcp_ui_resource_from_agui_snapshot():
    from vincio import ContextApp

    app = ContextApp(name="ui2", provider=MockProvider(), model="mock-1")
    snapshot = [AGUIEvent(type=AGUIEventType.STATE_SNAPSHOT, snapshot={"count": 1})]
    ui = MCPUIResource.from_agui("ui://state", snapshot, name="state")
    assert ui.mime_type == "application/vnd.ag-ui+json"
    server = build_app_server(app, expose_resources=False, ui_resources=[ui])
    client = connect_in_process(server)
    await client.initialize()
    assert "ui://state" in {r.uri for r in await client.list_resources()}
