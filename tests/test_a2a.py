"""A2A (1.1): agent card, task lifecycle, crew/graph exposure, remote delegate."""

from __future__ import annotations

import httpx
import pytest

from vincio import ContextApp
from vincio.a2a import (
    A2AError,
    RemoteA2AAgent,
    connect_a2a,
    connect_a2a_in_process,
    static_token_validator,
)
from vincio.a2a.server import app_a2a_server, graph_a2a_server
from vincio.agents import interrupt
from vincio.providers import MockProvider


def _app(text: str = "the answer is 4") -> ContextApp:
    return ContextApp(name="helper", provider=MockProvider(default_text=text), model="mock-1")


@pytest.mark.asyncio
async def test_agent_card_served():
    server = app_a2a_server(_app(), name="helper")
    client = connect_a2a_in_process(server)
    card = await client.agent_card()
    assert card.name == "helper"
    assert card.protocol_version == "0.3.0"
    assert card.capabilities.get("streaming") is True


@pytest.mark.asyncio
async def test_task_lifecycle_completed():
    client = connect_a2a_in_process(app_a2a_server(_app(), name="helper"))
    task = await client.send("what is 2+2?")
    assert task.status.state == "completed"
    assert "answer" in task.status.message.text
    # tasks/get returns the stored task.
    fetched = await client.get_task(task.id)
    assert fetched.status.state == "completed"


@pytest.mark.asyncio
async def test_task_cancel():
    client = connect_a2a_in_process(app_a2a_server(_app(), name="helper"))
    task = await client.send("hi")
    canceled = await client.cancel(task.id)
    assert canceled.status.state == "canceled"


@pytest.mark.asyncio
async def test_crew_exposed_over_a2a():
    app = _app()
    crew = app.crew(members=[{"name": "researcher", "goal": "find numbers", "keywords": ["find"]}])
    server = app.serve_a2a(crew, name="research_crew")
    assert [s["name"] for s in server.agent_card()["skills"]] == ["researcher"]
    client = connect_a2a_in_process(server)
    task = await client.send("summarize refunds")
    assert task.status.state == "completed"
    # serving records an inbound audit entry.
    assert any(e.action == "a2a_serve" for e in app.audit.entries)


@pytest.mark.asyncio
async def test_graph_hitl_input_required_then_resume():
    app = _app()
    g = app.graph("approval")
    g.add_node("ask", lambda s: {"output": interrupt(s, "approve?")})
    g.add_node("finish", lambda s: {"output": "approved: " + str(s.get("output"))})
    g.add_edge("ask", "finish")
    server = graph_a2a_server(g.compile(), name="approval", tracer=app.tracer)
    client = connect_a2a_in_process(server)
    task = await client.send("start")
    assert task.status.state == "input-required"
    assert "approve?" in task.status.message.text
    resumed = await client.send("yes", task_id=task.id)
    assert resumed.status.state == "completed"
    assert "approved: yes" in resumed.status.message.text


@pytest.mark.asyncio
async def test_remote_agent_as_crew_delegate():
    remote = RemoteA2AAgent(connect_a2a_in_process(app_a2a_server(_app(), name="delegate")))
    state = await remote.run("compute 2+2")
    assert state.termination_reason == "objective_complete"
    assert "answer" in str(state.final_answer)
    assert state.working_memory["a2a_task_state"] == "completed"


@pytest.mark.asyncio
async def test_token_validation():
    server = app_a2a_server(_app(), name="secure", token_validator=static_token_validator({"tok"}))
    ok = connect_a2a_in_process(server, auth="Bearer tok")
    assert (await ok.send("hi")).status.state == "completed"
    bad = connect_a2a_in_process(server, auth="Bearer nope")
    with pytest.raises(A2AError) as exc:
        await bad.send("hi")
    assert exc.value.code == -32001


@pytest.mark.asyncio
async def test_http_transport_card_and_send():
    server = app_a2a_server(_app("remote answer"), name="remote", url="https://agent.example")

    async def dispatch(request: httpx.Request) -> httpx.Response:
        import json

        if request.url.path == "/.well-known/agent.json":
            return httpx.Response(200, json=server.agent_card())
        body = json.loads(request.content)
        response = await server.handle(body)
        return httpx.Response(200, json=response)

    client = connect_a2a(
        "https://agent.example/rpc",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(dispatch)),
    )
    card = await client.agent_card()
    assert card.name == "remote"
    task = await client.send("hello")
    assert task.status.state == "completed"
    assert "remote answer" in task.status.message.text
