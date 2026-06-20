"""MCP client + server, offline via the in-process transport + httpx mocks."""

from __future__ import annotations

import httpx
import pytest

from vincio import ContextApp
from vincio.core.types import ToolCall
from vincio.mcp import (
    MCPError,
    MCPServer,
    StreamableHTTPTransport,
    connect_in_process,
    static_token_validator,
)
from vincio.mcp.client import MCPClient
from vincio.providers import MockProvider


def _calc_server() -> MCPServer:
    def list_tools():
        return [
            {
                "name": "add",
                "description": "add two integers",
                "inputSchema": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            }
        ]

    async def call_tool(name, args):
        if name == "add":
            return {"text": str(args["a"] + args["b"])}
        return {"_mcp_error": True, "text": "unknown"}

    def list_resources():
        return [{"uri": "vincio://doc/1", "name": "policy", "mimeType": "text/plain"}]

    async def read_resource(uri):
        return {"uri": uri, "mimeType": "text/plain", "text": "Refunds within 30 days."}

    return MCPServer(
        name="calc",
        list_tools=list_tools,
        call_tool=call_tool,
        list_resources=list_resources,
        read_resource=read_resource,
    )


@pytest.mark.asyncio
async def test_initialize_and_capabilities():
    client = connect_in_process(_calc_server())
    result = await client.initialize()
    assert result["serverInfo"]["name"] == "calc"
    assert "tools" in result["capabilities"]
    assert "resources" in result["capabilities"]


@pytest.mark.asyncio
async def test_list_and_call_tool():
    client = connect_in_process(_calc_server())
    tools = await client.list_tools()
    assert [t.name for t in tools] == ["add"]
    assert tools[0].input_schema["required"] == ["a", "b"]
    assert await client.call_tool("add", {"a": 2, "b": 3}) == "5"


@pytest.mark.asyncio
async def test_resource_read():
    client = connect_in_process(_calc_server())
    resources = await client.list_resources()
    assert resources[0].uri == "vincio://doc/1"
    assert "30 days" in await client.read_resource("vincio://doc/1")


@pytest.mark.asyncio
async def test_tool_error_raises():
    client = connect_in_process(_calc_server())
    with pytest.raises(MCPError):
        await client.call_tool("nope", {})


@pytest.mark.asyncio
async def test_server_initiated_sampling_routes_to_provider():
    srv = MCPServer(name="s")
    srv._list_tools = lambda: [{"name": "ask", "description": "", "inputSchema": {"type": "object"}}]

    async def call_tool(name, args):
        reply = await srv.request_client(
            "sampling/createMessage",
            {"messages": [{"role": "user", "content": {"type": "text", "text": "q"}}]},
        )
        return {"text": reply["content"]["text"]}

    srv._call_tool = call_tool
    client = connect_in_process(srv, sampling_provider=MockProvider(default_text="sampled!"))
    assert await client.call_tool("ask", {}) == "sampled!"


@pytest.mark.asyncio
async def test_server_initiated_elicitation_routes_to_callback():
    srv = MCPServer(name="s")
    srv._list_tools = lambda: []

    async def call_tool(name, args):
        res = await srv.request_client("elicitation/create", {"message": "email?", "requestedSchema": {}})
        return {"text": str(res)}

    srv._call_tool = call_tool
    client = connect_in_process(srv, elicitation_callback=lambda msg, schema: {"email": "a@b.com"})
    assert "a@b.com" in await client.call_tool("anything", {})


@pytest.mark.asyncio
async def test_token_validation_rejects_bad_token():
    srv = MCPServer(name="secure", list_tools=lambda: [], token_validator=static_token_validator({"ok"}))
    good = connect_in_process(srv, auth="Bearer ok")
    assert await good.list_tools() == []
    bad = connect_in_process(srv, auth="Bearer no")
    with pytest.raises(MCPError) as exc:
        await bad.list_tools()
    assert exc.value.code == -32001


# -- ContextApp integration ---------------------------------------------------


@pytest.mark.asyncio
async def test_register_into_app_through_tool_runtime():
    app = ContextApp(name="consumer", provider=MockProvider(), model="mock-1")
    client = connect_in_process(_calc_server(), name="calc")
    manifest = await client.register_into(app)
    assert manifest["tools"] == ["add"]
    assert "calc.add" in app.enabled_tools
    # The MCP tool runs through the *existing* permissioned tool runtime.
    result = await app.tool_runtime.execute(ToolCall(tool_name="calc.add", arguments={"a": 4, "b": 5}))
    assert result.status == "ok"
    assert result.output == "9"
    # MCP resources become evidence with provenance.
    assert any(ev.metadata.get("origin") == "mcp:calc" for ev in app.pending_evidence)


@pytest.mark.asyncio
async def test_app_serve_mcp_round_trip():
    app = ContextApp(name="kb", provider=MockProvider(), model="mock-1")

    def shout(text: str) -> str:
        return text.upper()

    app.add_tool(shout, description="uppercase the text")
    server = app.serve_mcp()
    client = connect_in_process(server)
    tools = await client.list_tools()
    assert "shout" in [t.name for t in tools]
    assert await client.call_tool("shout", {"text": "hi"}) == "HI"
    # serving records an inbound audit entry
    assert any(e.action == "mcp_serve" for e in app.audit.entries)


@pytest.mark.asyncio
async def test_add_mcp_server_in_process():
    consumer = ContextApp(name="c", provider=MockProvider(), model="mock-1")
    consumer.add_mcp_server("calc", server=_calc_server())
    assert "calc.add" in consumer.enabled_tools
    assert "calc" in consumer.mcp_clients


# -- HTTP transport (offline via MockTransport) -------------------------------


@pytest.mark.asyncio
async def test_streamable_http_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        method = body["method"]
        msg_id = body.get("id")
        if msg_id is None:  # a notification (e.g. notifications/initialized)
            return httpx.Response(202, text="")
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "remote"}, "capabilities": {"tools": {}}}
        elif method == "tools/list":
            result = {"tools": [{"name": "ping", "description": "", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": "pong"}], "isError": False}
        else:
            result = {}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": msg_id, "result": result})

    transport = StreamableHTTPTransport(
        "https://mcp.example/rpc", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    client = MCPClient(transport)
    await client.initialize()
    assert [t.name for t in await client.list_tools()] == ["ping"]
    assert await client.call_tool("ping", {}) == "pong"
    await client.aclose()


@pytest.mark.asyncio
async def test_http_transport_maps_401_to_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    transport = StreamableHTTPTransport(
        "https://mcp.example/rpc", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(MCPError):
        await transport.request("tools/list")


# -- MCP-server marketplace bridge ---------------------------------------------


def _weather_provider_app() -> ContextApp:
    app = ContextApp(name="weather_provider", provider=MockProvider(), model="mock-1")

    @app.tool_registry.register(name="get_weather")
    def get_weather(city: str) -> dict:
        """Look up the weather."""
        return {"city": city, "temp": 72}

    app.enabled_tools.append("get_weather")
    return app


class TestMarketplaceBridge:
    def test_discover_govern_connect_in_one_call(self):
        from vincio.mcp import build_app_server
        from vincio.registry import MCPRegistryClient, MCPServerRecord

        server = build_app_server(_weather_provider_app())
        consumer = ContextApp(name="consumer", provider=MockProvider(), model="mock-1")
        registry = MCPRegistryClient(
            catalog=[
                MCPServerRecord(name="weather", url="https://weather.example/mcp", description="weather"),
                MCPServerRecord(name="evil-server", url="https://evil.example/mcp"),
            ]
        )

        consumer.add_mcp_from_registry("weather", registry=registry, server=server, allow=["weather"])

        # The server's tools landed in the permissioned runtime, namespaced.
        assert "weather.get_weather" in consumer.enabled_tools
        assert "weather" in consumer.mcp_clients
        # The governed resolution is an audited access decision.
        decisions = consumer.audit.query(action="agent_resolve")
        assert any(d.decision == "allow" and d.resource == "weather" for d in decisions)

    def test_unlisted_server_is_denied(self):
        from vincio.core.errors import AccessDeniedError
        from vincio.mcp import build_app_server
        from vincio.registry import MCPRegistryClient, MCPServerRecord

        server = build_app_server(_weather_provider_app())
        consumer = ContextApp(name="consumer2", provider=MockProvider(), model="mock-1")
        registry = MCPRegistryClient(catalog=[MCPServerRecord(name="evil-server", url="https://evil/mcp")])
        with pytest.raises(AccessDeniedError):
            consumer.add_mcp_from_registry("evil-server", registry=registry, server=server, allow=["weather"])
        # Denial is audited.
        decisions = consumer.audit.query(action="agent_resolve")
        assert any(d.decision == "deny" and d.resource == "evil-server" for d in decisions)
