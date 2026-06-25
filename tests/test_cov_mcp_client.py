"""Real-behavior coverage for the MCP *client* (vincio.mcp.client).

Everything is offline: a deterministic in-process :class:`MCPServer`, a tiny
scripted :class:`MCPTransport` for protocol corners the base server does not
dispatch (long-running Tasks, UI embeds), and a real ``MockProvider`` for the
server-initiated sampling path. No unittest.mock, no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from vincio import ContextApp
from vincio.mcp import MCPServer, connect_in_process
from vincio.mcp.apps import ElicitationGate, ElicitationPolicy
from vincio.mcp.client import MCPClient, _content_text
from vincio.mcp.protocol import INVALID_PARAMS, MCPError
from vincio.mcp.transport import MCPTransport
from vincio.providers import MockProvider


# ---------------------------------------------------------------------------
# A scripted transport: answers ``request`` from a callable, so we can drive
# protocol corners (tasks, UI content blocks) the bare MCPServer doesn't dispatch.
# ---------------------------------------------------------------------------
class ScriptedTransport(MCPTransport):
    def __init__(self, responder: Any) -> None:
        self._responder = responder
        self.notified: list[str] = []
        self.requests: list[tuple[str, dict[str, Any]]] = []

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self.requests.append((method, params or {}))
        return self._responder(method, params or {})

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self.notified.append(method)


def _init_result() -> dict[str, Any]:
    return {
        "protocolVersion": "2025-06-18",
        "serverInfo": {"name": "scripted"},
        "capabilities": {"tools": {}},
    }


# ---------------------------------------------------------------------------
# A richer in-process server with prompts + UI resources for the real path.
# ---------------------------------------------------------------------------
def _full_server() -> MCPServer:
    def list_tools():
        return [{"name": "echo", "description": "echo", "inputSchema": {"type": "object"}}]

    async def call_tool(name, args):
        return {"text": str(args.get("v", ""))}

    def list_resources():
        return [
            {"uri": "vincio://doc/1", "name": "doc", "mimeType": "text/plain"},
            {"uri": "ui://panel", "name": "panel", "mimeType": "text/html"},
        ]

    async def read_resource(uri):
        if uri == "ui://panel":
            return {"uri": uri, "mimeType": "text/html", "text": "<b>hi</b>"}
        return {"uri": uri, "mimeType": "text/plain", "text": "doc body"}

    def list_prompts():
        return [
            {
                "name": "greet",
                "description": "say hi",
                "arguments": [{"name": "who", "required": True}],
            }
        ]

    async def get_prompt(name, args):
        return {
            "messages": [
                {"role": "system", "content": {"type": "text", "text": "You are kind."}},
                {"role": "user", "content": {"type": "text", "text": f"hi {args.get('who', '')}"}},
            ]
        }

    return MCPServer(
        name="full",
        list_tools=list_tools,
        call_tool=call_tool,
        list_resources=list_resources,
        read_resource=read_resource,
        list_prompts=list_prompts,
        get_prompt=get_prompt,
    )


# -- discovery: prompts, ui resources ---------------------------------------


@pytest.mark.asyncio
async def test_list_prompts_maps_arguments():
    client = connect_in_process(_full_server())
    prompts = await client.list_prompts()
    assert [p.name for p in prompts] == ["greet"]
    assert prompts[0].description == "say hi"
    assert prompts[0].arguments == [{"name": "who", "required": True}]


@pytest.mark.asyncio
async def test_list_ui_resources_filters_to_ui_only():
    client = connect_in_process(_full_server())
    all_res = await client.list_resources()
    ui = await client.list_ui_resources()
    # list_resources returns both; list_ui_resources keeps only the ui:// one.
    assert {r.uri for r in all_res} == {"vincio://doc/1", "ui://panel"}
    assert [r.uri for r in ui] == ["ui://panel"]
    assert ui[0].mime_type == "text/html"


@pytest.mark.asyncio
async def test_get_prompt_returns_raw_messages():
    client = connect_in_process(_full_server())
    messages = await client.get_prompt("greet", {"who": "Sam"})
    assert messages[0]["role"] == "system"
    assert messages[1]["content"]["text"] == "hi Sam"


@pytest.mark.asyncio
async def test_import_prompt_concatenates_text_into_spec():
    client = connect_in_process(_full_server())
    spec = await client.import_prompt("greet", {"who": "Ada"})
    assert spec.name == "greet"
    # Both message texts are joined; non-dict content would be skipped.
    assert spec.objective == "You are kind.\n\nhi Ada"


@pytest.mark.asyncio
async def test_import_prompt_falls_back_to_name_when_empty():
    # A server whose prompt has no dict-content messages -> objective falls back to name.
    def list_prompts():
        return [{"name": "blank"}]

    async def get_prompt(name, args):
        return {"messages": [{"role": "user", "content": "plain-string-not-dict"}]}

    server = MCPServer(name="p", list_prompts=list_prompts, get_prompt=get_prompt)
    client = connect_in_process(server)
    spec = await client.import_prompt("blank")
    assert spec.objective == "blank"


# -- ensure_initialized only initializes once -------------------------------


@pytest.mark.asyncio
async def test_initialize_runs_once_via_ensure():
    inits = {"n": 0}

    def responder(method, params):
        if method == "initialize":
            inits["n"] += 1
            return _init_result()
        if method == "tools/list":
            return {"tools": []}
        if method == "resources/list":
            return {"resources": []}
        return {}

    transport = ScriptedTransport(responder)
    client = MCPClient(transport)
    await client.list_tools()
    await client.list_resources()
    # First discovery call triggers exactly one initialize; second reuses it.
    assert inits["n"] == 1
    assert transport.notified == ["notifications/initialized"]


# -- call_tool: long-running Tasks polling ----------------------------------


@pytest.mark.asyncio
async def test_call_tool_awaits_task_to_completion():
    polls = {"n": 0}

    def responder(method, params):
        if method == "initialize":
            return _init_result()
        if method == "tools/call":
            return {"taskId": "T1"}
        if method == "tasks/get":
            polls["n"] += 1
            if polls["n"] >= 2:
                return {"status": "completed", "result": {"content": [{"type": "text", "text": "done"}]}}
            return {"status": "working"}
        return {}

    client = MCPClient(ScriptedTransport(responder))
    # initial_delay tiny so the one backoff sleep is instant.
    out = await client.call_tool("slow", {})
    assert out == "done"
    assert polls["n"] == 2


@pytest.mark.asyncio
async def test_call_tool_raises_on_failed_task():
    def responder(method, params):
        if method == "initialize":
            return _init_result()
        if method == "tools/call":
            # nested-task shape: result.task.taskId
            return {"task": {"taskId": "T2"}}
        if method == "tasks/get":
            return {"status": "failed", "error": "boom in worker"}
        return {}

    client = MCPClient(ScriptedTransport(responder))
    with pytest.raises(MCPError, match="boom in worker"):
        await client.call_tool("slow", {})


@pytest.mark.asyncio
async def test_call_tool_does_not_await_task_when_disabled():
    def responder(method, params):
        if method == "initialize":
            return _init_result()
        if method == "tools/call":
            # Returns a taskId but ALSO text; with await_result=False we keep text.
            return {"taskId": "T3", "content": [{"type": "text", "text": "queued"}]}
        if method == "tasks/get":
            raise AssertionError("tasks/get must not be polled when await_result=False")
        return {}

    client = MCPClient(ScriptedTransport(responder))
    assert await client.call_tool("slow", {}, await_result=False) == "queued"


@pytest.mark.asyncio
async def test_await_task_deadline_returns_failed():
    def responder(method, params):
        if method == "initialize":
            return _init_result()
        if method == "tasks/get":
            return {"status": "working"}  # never finishes
        return {}

    client = MCPClient(ScriptedTransport(responder))
    task = await client._await_task("STUCK", deadline_s=0.0, initial_delay_s=0.0)
    assert task.status == "failed"
    assert "deadline" in (task.error or "")
    assert "working" in (task.error or "")


@pytest.mark.asyncio
async def test_call_tool_raises_on_iserror_result():
    def responder(method, params):
        if method == "initialize":
            return _init_result()
        if method == "tools/call":
            return {"isError": True, "content": [{"type": "text", "text": "tool blew up"}]}
        return {}

    client = MCPClient(ScriptedTransport(responder))
    with pytest.raises(MCPError, match="tool blew up"):
        await client.call_tool("bad", {})


@pytest.mark.asyncio
async def test_call_tool_iserror_without_text_uses_default_message():
    def responder(method, params):
        if method == "initialize":
            return _init_result()
        if method == "tools/call":
            return {"isError": True, "content": []}
        return {}

    client = MCPClient(ScriptedTransport(responder))
    with pytest.raises(MCPError, match="returned an error"):
        await client.call_tool("bad", {})


# -- call_tool_ui: embedded UI resource blocks ------------------------------


@pytest.mark.asyncio
async def test_call_tool_ui_extracts_ui_resource_block():
    def responder(method, params):
        if method == "initialize":
            return _init_result()
        if method == "tools/call":
            return {
                "content": [
                    {"type": "text", "text": "rendered"},
                    {
                        "type": "resource",
                        "resource": {
                            "uri": "ui://card",
                            "mimeType": "text/html",
                            "text": "<div>card</div>",
                        },
                    },
                    {"type": "resource", "resource": {"uri": "vincio://plain", "mimeType": "text/plain"}},
                ]
            }
        return {}

    client = MCPClient(ScriptedTransport(responder))
    text, ui = await client.call_tool_ui("render", {})
    assert text == "rendered"
    # Only the ui:// resource block is surfaced; the plain one is filtered out.
    assert len(ui) == 1
    assert ui[0].uri == "ui://card"
    assert ui[0].text == "<div>card</div>"


@pytest.mark.asyncio
async def test_call_tool_ui_raises_on_iserror():
    def responder(method, params):
        if method == "initialize":
            return _init_result()
        if method == "tools/call":
            return {"isError": True, "content": [{"type": "text", "text": "ui failed"}]}
        return {}

    client = MCPClient(ScriptedTransport(responder))
    with pytest.raises(MCPError, match="ui failed"):
        await client.call_tool_ui("render", {})


# -- server-initiated requests ----------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_server_request_raises_invalid_params():
    client = MCPClient(ScriptedTransport(lambda m, p: {}))
    with pytest.raises(MCPError, match="unsupported server request") as exc:
        await client._handle_server_request("roots/list", {})
    assert exc.value.code == INVALID_PARAMS


@pytest.mark.asyncio
async def test_handle_server_request_routes_elicitation():
    # The dispatch entry for elicitation/create (no gate, no callback -> decline).
    client = MCPClient(ScriptedTransport(lambda m, p: {}))
    out = await client._handle_server_request("elicitation/create", {"message": "x"})
    assert out == {"action": "decline"}


@pytest.mark.asyncio
async def test_handle_server_request_routes_sampling():
    client = MCPClient(
        ScriptedTransport(lambda m, p: {}),
        sampling_provider=MockProvider(default_text="hi"),
    )
    out = await client._handle_server_request(
        "sampling/createMessage",
        {"messages": [{"role": "user", "content": {"type": "text", "text": "q"}}]},
    )
    assert out["content"]["text"] == "hi"


@pytest.mark.asyncio
async def test_sampling_without_provider_raises():
    client = MCPClient(ScriptedTransport(lambda m, p: {}))
    with pytest.raises(MCPError, match="no provider is configured"):
        await client._handle_server_request("sampling/createMessage", {"messages": []})


@pytest.mark.asyncio
async def test_sampling_routes_system_and_messages_to_provider():
    seen: dict[str, Any] = {}

    def responder(request):
        seen["model"] = request.model
        seen["roles"] = [m.role for m in request.messages]
        seen["max"] = request.max_output_tokens
        return "ANSWER"

    client = MCPClient(
        ScriptedTransport(lambda m, p: {}),
        sampling_provider=MockProvider(responder=responder),
        sampling_model="custom-model",
    )
    result = await client._sample(
        {
            "systemPrompt": "be terse",
            "messages": [
                {"role": "user", "content": {"type": "text", "text": "q1"}},
                {"role": "tool", "content": {"type": "text", "text": "q2"}},  # unknown role -> user
            ],
            "maxTokens": 50,
        }
    )
    assert result["role"] == "assistant"
    assert result["content"] == {"type": "text", "text": "ANSWER"}
    assert result["stopReason"] == "endTurn"
    # system prompt prepended, unknown role coerced to user.
    assert seen["roles"] == ["system", "user", "user"]
    assert seen["model"] == "custom-model"
    assert seen["max"] == 50


@pytest.mark.asyncio
async def test_sampling_without_system_prompt_omits_system_message():
    seen: dict[str, Any] = {}

    def responder(request):
        seen["roles"] = [m.role for m in request.messages]
        return "x"

    client = MCPClient(
        ScriptedTransport(lambda m, p: {}),
        sampling_provider=MockProvider(responder=responder),
    )
    await client._sample(
        {"messages": [{"role": "assistant", "content": {"type": "text", "text": "prior"}}]}
    )
    # No systemPrompt key -> no system message prepended; default model used.
    assert seen["roles"] == ["assistant"]


# -- elicitation: callback fallback variants --------------------------------


@pytest.mark.asyncio
async def test_elicit_no_callback_declines():
    client = MCPClient(ScriptedTransport(lambda m, p: {}))
    assert await client._elicit({"message": "x"}) == {"action": "decline"}


@pytest.mark.asyncio
async def test_elicit_callback_returning_none_declines():
    client = MCPClient(ScriptedTransport(lambda m, p: {}), elicitation_callback=lambda msg, s: None)
    assert await client._elicit({"message": "x"}) == {"action": "decline"}


@pytest.mark.asyncio
async def test_elicit_callback_returning_true_accepts_empty():
    client = MCPClient(ScriptedTransport(lambda m, p: {}), elicitation_callback=lambda msg, s: True)
    assert await client._elicit({"message": "x"}) == {"action": "accept", "content": {}}


@pytest.mark.asyncio
async def test_elicit_callback_returning_dict_accepts_value():
    client = MCPClient(
        ScriptedTransport(lambda m, p: {}),
        elicitation_callback=lambda msg, s: {"email": "a@b.com"},
    )
    assert await client._elicit({"message": "email?"}) == {
        "action": "accept",
        "content": {"email": "a@b.com"},
    }


@pytest.mark.asyncio
async def test_elicit_awaits_async_callback():
    async def collector(msg, schema):
        return {"k": "v"}

    client = MCPClient(ScriptedTransport(lambda m, p: {}), elicitation_callback=collector)
    assert await client._elicit({"message": "x"}) == {"action": "accept", "content": {"k": "v"}}


@pytest.mark.asyncio
async def test_elicit_gate_takes_precedence_over_callback():
    # The governed gate path: it accepts and returns a wire response, and the
    # raw callback (which would also accept) is bypassed.
    gate = ElicitationGate(
        collector=lambda msg, schema: {"name": "Ada"},
        policy=ElicitationPolicy(screen_rails=False),
    )
    client = MCPClient(
        ScriptedTransport(lambda m, p: {}),
        name="srv",
        elicitation_gate=gate,
        elicitation_callback=lambda msg, s: {"never": "used"},
    )
    out = await client._elicit({"message": "name?", "requestedSchema": {}})
    assert out["action"] == "accept"
    assert out["content"] == {"name": "Ada"}


# -- register_into: prompts branch + idempotent enable ----------------------


@pytest.mark.asyncio
async def test_register_into_includes_prompts_when_requested():
    app = ContextApp(name="consumer", provider=MockProvider(), model="mock-1")
    client = connect_in_process(_full_server(), name="full")
    manifest = await client.register_into(app, prompts=True)
    assert manifest["tools"] == ["echo"]
    assert manifest["prompts"] == ["greet"]
    # ui:// resource is also imported as evidence (register_into reads all resources).
    origins = [ev.metadata.get("origin") for ev in app.pending_evidence]
    assert origins.count("mcp:full") == 2


@pytest.mark.asyncio
async def test_register_into_tools_only_skips_resources_and_prompts():
    app = ContextApp(name="c2", provider=MockProvider(), model="mock-1")
    client = connect_in_process(_full_server(), name="full")
    manifest = await client.register_into(app, resources=False, prompts=False)
    assert manifest["tools"] == ["echo"]
    assert manifest["resources"] == []
    assert manifest["prompts"] == []
    assert app.pending_evidence == []


@pytest.mark.asyncio
async def test_register_tool_is_idempotent_in_enabled_tools():
    app = ContextApp(name="c3", provider=MockProvider(), model="mock-1")
    client = connect_in_process(_full_server(), name="full")
    await client.register_into(app, resources=False)
    # Register the same server again; enabled_tools must not double up.
    await client.register_into(app, resources=False)
    assert app.enabled_tools.count("full.echo") == 1


@pytest.mark.asyncio
async def test_register_into_resources_only_skips_tools():
    app = ContextApp(name="c4", provider=MockProvider(), model="mock-1")
    client = connect_in_process(_full_server(), name="full")
    manifest = await client.register_into(app, tools=False)
    assert manifest["tools"] == []
    assert "full.echo" not in app.enabled_tools
    assert manifest["resources"] == ["vincio://doc/1", "ui://panel"]


@pytest.mark.asyncio
async def test_registered_tool_handler_bridges_to_call_tool():
    app = ContextApp(name="c5", provider=MockProvider(), model="mock-1")
    client = connect_in_process(_full_server(), name="full")
    await client.register_into(app, resources=False)
    # Drive the registered handler through the real tool runtime (echo returns "v").
    from vincio.core.types import ToolCall

    result = await app.tool_runtime.execute(
        ToolCall(tool_name="full.echo", arguments={"v": "ping"})
    )
    assert result.status == "ok"
    assert result.output == "ping"


@pytest.mark.asyncio
async def test_aclose_closes_transport():
    closed = {"n": 0}

    class Closing(ScriptedTransport):
        async def aclose(self) -> None:
            closed["n"] += 1

    client = MCPClient(Closing(lambda m, p: {}))
    await client.aclose()
    assert closed["n"] == 1


# -- _content_text helper ----------------------------------------------------


def test_content_text_joins_only_text_blocks():
    result = {
        "content": [
            {"type": "text", "text": "a"},
            {"type": "image", "data": "..."},  # not text -> skipped
            {"type": "text", "text": "b"},
            "not-a-dict",  # skipped
        ]
    }
    assert _content_text(result) == "a\nb"


def test_content_text_empty_when_no_content():
    assert _content_text({}) == ""
