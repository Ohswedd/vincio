"""Real-behavior coverage for vincio.mcp.server.

Drives MCPServer.handle through every JSON-RPC branch, build_app_server's
tool/resource/prompt wiring through a real ContextApp + MockProvider, the
MCPUIResource / embedded-resource path, and serve_stdio over an in-memory
reader/writer. No mocks, no patching — real objects and exact assertions.
"""

from __future__ import annotations

import io
import json

import pytest

from vincio import ContextApp
from vincio.core.types import EvidenceItem
from vincio.mcp.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    MCPError,
)
from vincio.mcp.server import (
    MCPServer,
    MCPUIResource,
    _app_resources,
    _as_ui_resource,
    _stringify,
    build_app_server,
    serve_stdio,
)
from vincio.providers import MockProvider


def _req(msg_id, method, params=None):
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


# -- MCPUIResource ------------------------------------------------------------


def test_ui_resource_from_html_descriptor_and_content():
    res = MCPUIResource.from_html("ui://panel", "<b>hi</b>", description="a panel")
    # name defaults to the uri when omitted.
    assert res.name == "ui://panel"
    assert res.mime_type == "text/html"
    assert res.descriptor() == {
        "uri": "ui://panel",
        "name": "ui://panel",
        "description": "a panel",
        "mimeType": "text/html",
    }
    assert res.content() == {"uri": "ui://panel", "mimeType": "text/html", "text": "<b>hi</b>"}


def test_ui_resource_from_agui_serializes_events_as_json():
    class _Evt:
        def to_wire(self):
            return {"type": "TEXT", "value": "x"}

    res = MCPUIResource.from_agui("ui://stream", [_Evt(), {"raw": 1}], name="stream")
    assert res.mime_type == "application/vnd.ag-ui+json"
    assert json.loads(res.text) == [{"type": "TEXT", "value": "x"}, {"raw": 1}]
    assert res.name == "stream"


# -- handle(): envelope + auth gating -----------------------------------------


@pytest.mark.asyncio
async def test_non_jsonrpc_message_is_invalid_request():
    srv = MCPServer(name="s")
    resp = await srv.handle({"id": 1, "method": "ping"})
    assert resp["error"]["code"] == INVALID_REQUEST
    assert resp["error"]["message"] == "not a JSON-RPC 2.0 message"
    assert resp["id"] == 1


@pytest.mark.asyncio
async def test_notification_returns_none():
    srv = MCPServer(name="s")
    # No id => notification => no response.
    assert await srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


@pytest.mark.asyncio
async def test_unknown_method_returns_method_not_found():
    srv = MCPServer(name="s")
    resp = await srv.handle(_req(7, "does/not/exist"))
    assert resp["error"]["code"] == METHOD_NOT_FOUND
    assert "does/not/exist" in resp["error"]["message"]


@pytest.mark.asyncio
async def test_ping_is_unauthenticated_and_returns_empty():
    # A token validator that always rejects must NOT run for ping.
    def reject(_auth):
        raise MCPError("nope", code=-32001)

    srv = MCPServer(name="s", token_validator=reject)
    resp = await srv.handle(_req(1, "ping"))
    assert resp == {"jsonrpc": "2.0", "id": 1, "result": {}}


@pytest.mark.asyncio
async def test_initialize_negotiates_requested_version():
    srv = MCPServer(name="srv", version="9.9.9", list_tools=lambda: [])
    resp = await srv.handle(_req(2, "initialize", {"protocolVersion": "2024-11-05"}))
    result = resp["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert srv.negotiated_version == "2024-11-05"
    assert result["serverInfo"] == {"name": "srv", "version": "9.9.9"}
    # Only tools capability advertised (no resources / prompts wired).
    assert result["capabilities"] == {"tools": {"listChanged": False}}


@pytest.mark.asyncio
async def test_initialize_unsupported_version_falls_back_to_latest():
    srv = MCPServer(name="srv")
    resp = await srv.handle(_req(2, "initialize", {"protocolVersion": "1999-01-01"}))
    assert resp["result"]["protocolVersion"] == "2025-06-18"
    # No providers wired => empty capabilities.
    assert resp["result"]["capabilities"] == {}


@pytest.mark.asyncio
async def test_capabilities_lists_all_three_when_wired():
    srv = MCPServer(
        name="s",
        list_tools=lambda: [],
        list_resources=lambda: [],
        list_prompts=lambda: [],
    )
    caps = srv.capabilities()
    assert caps == {
        "tools": {"listChanged": False},
        "resources": {"listChanged": False, "subscribe": False},
        "prompts": {"listChanged": False},
    }


@pytest.mark.asyncio
async def test_async_token_validator_is_awaited_and_rejection_propagates():
    seen = {}

    async def validate(auth):
        seen["auth"] = auth
        if auth != "Bearer ok":
            raise MCPError("bad token", code=-32001, data={"hint": "use ok"})
        return {"sub": "user"}

    srv = MCPServer(name="s", list_tools=lambda: [], token_validator=validate)
    # Good token: passes through to the gated method.
    ok = await srv.handle(_req(1, "tools/list"), auth="Bearer ok")
    assert ok["result"] == {"tools": []}
    assert seen["auth"] == "Bearer ok"
    # Bad token: error carries the validator's code AND its data.
    bad = await srv.handle(_req(2, "tools/list"), auth="Bearer no")
    assert bad["error"]["code"] == -32001
    assert bad["error"]["message"] == "bad token"
    assert bad["error"]["data"] == {"hint": "use ok"}


@pytest.mark.asyncio
async def test_sync_token_validator_success_passes_through():
    # A synchronous validator that returns a (non-awaitable) identity dict must
    # not be awaited; the gated call proceeds.
    calls = []

    def validate(auth):
        calls.append(auth)
        return {"sub": "ok"}

    srv = MCPServer(name="s", list_tools=lambda: [], token_validator=validate)
    resp = await srv.handle(_req(1, "tools/list"), auth="Bearer whatever")
    assert resp["result"] == {"tools": []}
    assert calls == ["Bearer whatever"]


@pytest.mark.asyncio
async def test_dispatch_internal_error_is_wrapped():
    # A non-MCPError raised inside a provider is mapped to INTERNAL_ERROR.
    def boom():
        raise ValueError("kaboom")

    srv = MCPServer(name="s", list_tools=boom)
    resp = await srv.handle(_req(1, "tools/list"))
    assert resp["error"]["code"] == INTERNAL_ERROR
    assert resp["error"]["message"] == "ValueError: kaboom"


# -- tools/list + tools/call --------------------------------------------------


@pytest.mark.asyncio
async def test_tools_list_returns_provider_output():
    spec = {"name": "echo", "description": "", "inputSchema": {"type": "object"}}
    srv = MCPServer(name="s", list_tools=lambda: [spec])
    resp = await srv.handle(_req(1, "tools/list"))
    assert resp["result"] == {"tools": [spec]}


@pytest.mark.asyncio
async def test_tools_list_empty_when_no_provider():
    srv = MCPServer(name="s")
    resp = await srv.handle(_req(1, "tools/list"))
    assert resp["result"] == {"tools": []}


@pytest.mark.asyncio
async def test_tools_call_without_provider_is_method_not_found():
    srv = MCPServer(name="s")
    resp = await srv.handle(_req(1, "tools/call", {"name": "x"}))
    assert resp["error"]["code"] == METHOD_NOT_FOUND
    assert resp["error"]["message"] == "tools not supported"


@pytest.mark.asyncio
async def test_tools_call_missing_name_is_invalid_params():
    async def call(name, args):
        return {"text": "never"}

    srv = MCPServer(name="s", call_tool=call)
    resp = await srv.handle(_req(1, "tools/call", {"arguments": {}}))
    assert resp["error"]["code"] == INVALID_PARAMS
    assert resp["error"]["message"] == "tools/call requires 'name'"


@pytest.mark.asyncio
async def test_tools_call_text_dict_result():
    async def call(name, args):
        return {"text": args["a"] + args["b"]}

    srv = MCPServer(name="s", call_tool=call)
    resp = await srv.handle(_req(1, "tools/call", {"name": "cat", "arguments": {"a": "x", "b": "y"}}))
    assert resp["result"] == {"content": [{"type": "text", "text": "xy"}], "isError": False}


@pytest.mark.asyncio
async def test_tools_call_error_dict_sets_iserror():
    async def call(name, args):
        return {"_mcp_error": True, "text": "boom"}

    srv = MCPServer(name="s", call_tool=call)
    resp = await srv.handle(_req(1, "tools/call", {"name": "x"}))
    assert resp["result"]["isError"] is True
    assert resp["result"]["content"] == [{"type": "text", "text": "boom"}]


@pytest.mark.asyncio
async def test_tools_call_non_dict_result_is_stringified_json():
    async def call(name, args):
        return {"answer": 42}  # dict without "text" => stringified

    srv = MCPServer(name="s", call_tool=call)
    resp = await srv.handle(_req(1, "tools/call", {"name": "x"}))
    assert resp["result"]["content"] == [{"type": "text", "text": '{"answer": 42}'}]
    assert resp["result"]["isError"] is False


@pytest.mark.asyncio
async def test_tools_call_with_ui_resource_dict_embeds_resource():
    ui = MCPUIResource.from_html("ui://card", "<div/>")

    async def call(name, args):
        return {"text": "rendered", "ui": ui}

    srv = MCPServer(name="s", call_tool=call)
    resp = await srv.handle(_req(1, "tools/call", {"name": "x"}))
    content = resp["result"]["content"]
    assert content[0] == {"type": "text", "text": "rendered"}
    assert content[1] == {
        "type": "resource",
        "resource": {"uri": "ui://card", "mimeType": "text/html", "text": "<div/>"},
    }


@pytest.mark.asyncio
async def test_tools_call_returning_bare_ui_resource_embeds_it():
    async def call(name, args):
        return MCPUIResource.from_html("ui://bare", "<p/>")

    srv = MCPServer(name="s", call_tool=call)
    resp = await srv.handle(_req(1, "tools/call", {"name": "x"}))
    content = resp["result"]["content"]
    # text falls back to stringified MCPUIResource; the embedded resource is present.
    assert content[1]["resource"]["uri"] == "ui://bare"
    assert content[1]["type"] == "resource"


# -- resources ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_resources_list_empty_without_provider():
    srv = MCPServer(name="s")
    resp = await srv.handle(_req(1, "resources/list"))
    assert resp["result"] == {"resources": []}


@pytest.mark.asyncio
async def test_resources_read_without_provider_is_method_not_found():
    srv = MCPServer(name="s")
    resp = await srv.handle(_req(1, "resources/read", {"uri": "x://1"}))
    assert resp["error"]["code"] == METHOD_NOT_FOUND
    assert resp["error"]["message"] == "resources not supported"


@pytest.mark.asyncio
async def test_resources_read_missing_uri_is_invalid_params():
    async def read(uri):
        return {"uri": uri}

    srv = MCPServer(name="s", read_resource=read)
    resp = await srv.handle(_req(1, "resources/read", {}))
    assert resp["error"]["code"] == INVALID_PARAMS
    assert resp["error"]["message"] == "resources/read requires 'uri'"


@pytest.mark.asyncio
async def test_resources_read_wraps_in_contents_list():
    async def read(uri):
        return {"uri": uri, "mimeType": "text/plain", "text": "body"}

    srv = MCPServer(name="s", read_resource=read)
    resp = await srv.handle(_req(1, "resources/read", {"uri": "x://1"}))
    assert resp["result"] == {"contents": [{"uri": "x://1", "mimeType": "text/plain", "text": "body"}]}


# -- prompts ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompts_list_empty_without_provider():
    srv = MCPServer(name="s")
    resp = await srv.handle(_req(1, "prompts/list"))
    assert resp["result"] == {"prompts": []}


@pytest.mark.asyncio
async def test_prompts_get_without_provider_is_method_not_found():
    srv = MCPServer(name="s")
    resp = await srv.handle(_req(1, "prompts/get", {"name": "p"}))
    assert resp["error"]["code"] == METHOD_NOT_FOUND
    assert resp["error"]["message"] == "prompts not supported"


@pytest.mark.asyncio
async def test_prompts_get_missing_name_is_invalid_params():
    async def get(name, args):
        return {"messages": []}

    srv = MCPServer(name="s", get_prompt=get)
    resp = await srv.handle(_req(1, "prompts/get", {}))
    assert resp["error"]["code"] == INVALID_PARAMS
    assert resp["error"]["message"] == "prompts/get requires 'name'"


@pytest.mark.asyncio
async def test_prompts_get_passes_name_and_arguments():
    seen = {}

    async def get(name, args):
        seen["name"] = name
        seen["args"] = args
        return {"description": "d", "messages": []}

    srv = MCPServer(name="s", get_prompt=get)
    resp = await srv.handle(_req(1, "prompts/get", {"name": "p", "arguments": {"task": "do it"}}))
    assert resp["result"] == {"description": "d", "messages": []}
    assert seen == {"name": "p", "args": {"task": "do it"}}


# -- elicit() -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_elicit_without_transport_raises():
    srv = MCPServer(name="s")
    with pytest.raises(MCPError, match="no bidirectional transport"):
        await srv.elicit("email?")


@pytest.mark.asyncio
async def test_elicit_routes_to_request_client_with_schema():
    srv = MCPServer(name="s")
    captured = {}

    async def request_client(method, params):
        captured["method"] = method
        captured["params"] = params
        return {"action": "accept", "content": {"email": "a@b.com"}}

    srv.request_client = request_client
    schema = {"type": "object", "properties": {"email": {"type": "string"}}}
    out = await srv.elicit("email?", schema=schema)
    assert out == {"action": "accept", "content": {"email": "a@b.com"}}
    assert captured["method"] == "elicitation/create"
    assert captured["params"] == {"message": "email?", "requestedSchema": schema}


@pytest.mark.asyncio
async def test_elicit_defaults_schema_to_empty_dict():
    srv = MCPServer(name="s")
    captured = {}

    async def request_client(method, params):
        captured["params"] = params
        return {"action": "decline"}

    srv.request_client = request_client
    await srv.elicit("name?")
    assert captured["params"]["requestedSchema"] == {}


# -- module helpers -----------------------------------------------------------


def test_stringify_passthrough_for_strings():
    assert _stringify("already text") == "already text"


def test_stringify_json_encodes_objects():
    assert _stringify({"b": 2, "a": 1}) == '{"b": 2, "a": 1}'
    assert _stringify([1, 2]) == "[1, 2]"


def test_as_ui_resource_passthrough_for_instance():
    res = MCPUIResource.from_html("ui://x", "<x/>")
    assert _as_ui_resource(res) is res


def test_as_ui_resource_reconstructs_from_dict():
    rebuilt = _as_ui_resource(
        {"uri": "ui://panel", "name": "panel", "description": "d", "mime_type": "text/html", "text": "<p/>"}
    )
    assert isinstance(rebuilt, MCPUIResource)
    assert rebuilt.uri == "ui://panel"
    assert rebuilt.text == "<p/>"
    assert rebuilt.description == "d"


def test_as_ui_resource_accepts_camelcase_mime_key():
    rebuilt = _as_ui_resource({"uri": "any://1", "mimeType": "application/vnd.ag-ui+json", "text": "[]"})
    assert isinstance(rebuilt, MCPUIResource)
    assert rebuilt.mime_type == "application/vnd.ag-ui+json"
    # name falls back to the uri.
    assert rebuilt.name == "any://1"


def test_as_ui_resource_rejects_non_ui_dict():
    # A plain text resource (not ui:// and not a UI MIME) is not reconstructed.
    assert _as_ui_resource({"uri": "vincio://evidence/1", "mime_type": "text/plain"}) is None


def test_as_ui_resource_rejects_dict_without_uri():
    assert _as_ui_resource({"mime_type": "text/html", "text": "<p/>"}) is None


def test_as_ui_resource_rejects_non_dict_non_resource():
    assert _as_ui_resource("just a string") is None
    assert _as_ui_resource(42) is None


# -- build_app_server: ContextApp integration ---------------------------------


def _app_with_tool_and_evidence() -> ContextApp:
    app = ContextApp(name="kb", provider=MockProvider(), model="mock-1")

    def greet(name: str) -> str:
        return f"hello {name}"

    app.add_tool(greet, description="greet someone")
    app.pending_evidence.append(
        EvidenceItem(id="ev-1", source_id="DOC1", text="Refunds within 30 days.")
    )
    return app


@pytest.mark.asyncio
async def test_build_app_server_lists_tools_with_schema():
    server = build_app_server(_app_with_tool_and_evidence())
    resp = await server.handle(_req(1, "tools/list"))
    tools = {t["name"]: t for t in resp["result"]["tools"]}
    assert "greet" in tools
    assert tools["greet"]["description"] == "greet someone"
    assert tools["greet"]["inputSchema"]["type"] == "object"
    assert "name" in tools["greet"]["inputSchema"]["properties"]


@pytest.mark.asyncio
async def test_build_app_server_call_tool_runs_through_runtime_and_audits():
    app = _app_with_tool_and_evidence()
    server = build_app_server(app)
    resp = await server.handle(_req(1, "tools/call", {"name": "greet", "arguments": {"name": "Ada"}}))
    assert resp["result"]["content"] == [{"type": "text", "text": "hello Ada"}]
    assert resp["result"]["isError"] is False
    # An inbound serve event was audited.
    assert any(e.action == "mcp_serve" and e.resource == "greet" for e in app.audit.entries)


@pytest.mark.asyncio
async def test_build_app_server_tool_returning_ui_resource_embeds_it():
    app = ContextApp(name="ui-app", provider=MockProvider(), model="mock-1")

    def render() -> MCPUIResource:
        return MCPUIResource.from_html("ui://widget", "<widget/>", description="a widget")

    app.add_tool(render, description="render a widget")
    server = build_app_server(app)
    resp = await server.handle(_req(1, "tools/call", {"name": "render", "arguments": {}}))
    content = resp["result"]["content"]
    # The runtime serialized the MCPUIResource to a dict; the server reconstructs
    # it and surfaces text (description) plus the embedded ui:// resource.
    assert content[0] == {"type": "text", "text": "a widget"}
    assert content[1]["type"] == "resource"
    assert content[1]["resource"]["uri"] == "ui://widget"
    assert content[1]["resource"]["mimeType"] == "text/html"


@pytest.mark.asyncio
async def test_build_app_server_unknown_tool_returns_mcp_error():
    app = _app_with_tool_and_evidence()
    server = build_app_server(app)
    resp = await server.handle(_req(1, "tools/call", {"name": "nope", "arguments": {}}))
    assert resp["result"]["isError"] is True
    assert resp["result"]["content"][0]["text"] == "unknown tool 'nope'"


@pytest.mark.asyncio
async def test_build_app_server_tool_runtime_failure_surfaces_error():
    app = ContextApp(name="boom", provider=MockProvider(), model="mock-1")

    def explode() -> str:
        raise RuntimeError("tool blew up")

    app.add_tool(explode, description="explodes")
    server = build_app_server(app)
    resp = await server.handle(_req(1, "tools/call", {"name": "explode", "arguments": {}}))
    assert resp["result"]["isError"] is True
    assert resp["result"]["content"][0]["text"]  # carries the error text


@pytest.mark.asyncio
async def test_build_app_server_lists_evidence_resources():
    server = build_app_server(_app_with_tool_and_evidence())
    resp = await server.handle(_req(1, "resources/list"))
    uris = {r["uri"]: r for r in resp["result"]["resources"]}
    assert "vincio://evidence/ev-1" in uris
    r = uris["vincio://evidence/ev-1"]
    assert r["name"] == "DOC1"
    assert r["description"] == "Refunds within 30 days."
    assert r["mimeType"] == "text/plain"


@pytest.mark.asyncio
async def test_build_app_server_reads_evidence_resource():
    server = build_app_server(_app_with_tool_and_evidence())
    resp = await server.handle(_req(1, "resources/read", {"uri": "vincio://evidence/ev-1"}))
    assert resp["result"]["contents"] == [
        {"uri": "vincio://evidence/ev-1", "mimeType": "text/plain", "text": "Refunds within 30 days."}
    ]


@pytest.mark.asyncio
async def test_build_app_server_read_unknown_resource_raises_invalid_params():
    server = build_app_server(_app_with_tool_and_evidence())
    resp = await server.handle(_req(1, "resources/read", {"uri": "vincio://evidence/missing"}))
    assert resp["error"]["code"] == INVALID_PARAMS
    assert "missing" in resp["error"]["message"]


@pytest.mark.asyncio
async def test_build_app_server_with_ui_resources_serves_and_reads_them():
    app = _app_with_tool_and_evidence()
    ui = MCPUIResource.from_html("ui://dash", "<dashboard/>", name="dash")
    server = build_app_server(app, ui_resources=[ui])
    lst = await server.handle(_req(1, "resources/list"))
    uris = [r["uri"] for r in lst["result"]["resources"]]
    assert "ui://dash" in uris
    # The ui:// resource is read back from the ui_map (not the evidence path).
    rd = await server.handle(_req(2, "resources/read", {"uri": "ui://dash"}))
    assert rd["result"]["contents"][0] == {
        "uri": "ui://dash",
        "mimeType": "text/html",
        "text": "<dashboard/>",
    }


@pytest.mark.asyncio
async def test_build_app_server_ui_resources_force_resource_serving_when_disabled():
    # expose_resources=False but ui_resources present => resources still served.
    app = _app_with_tool_and_evidence()
    ui = MCPUIResource.from_html("ui://only", "<x/>")
    server = build_app_server(app, expose_resources=False, ui_resources=[ui])
    resp = await server.handle(_req(1, "resources/list"))
    uris = [r["uri"] for r in resp["result"]["resources"]]
    # The presence of a ui_resource forces resource serving back on; the ui
    # resource is listed first, and evidence follows once serving is enabled.
    assert uris[0] == "ui://only"
    assert "vincio://evidence/ev-1" in uris


@pytest.mark.asyncio
async def test_build_app_server_expose_resources_false_disables_resources():
    app = _app_with_tool_and_evidence()
    server = build_app_server(app, expose_resources=False)
    resp = await server.handle(_req(1, "resources/list"))
    # No provider wired => method-not-found path is not hit for list, returns [].
    assert resp["result"] == {"resources": []}
    # And resources/read is unsupported.
    read = await server.handle(_req(2, "resources/read", {"uri": "x"}))
    assert read["error"]["code"] == METHOD_NOT_FOUND


@pytest.mark.asyncio
async def test_build_app_server_lists_prompt_from_spec():
    app = _app_with_tool_and_evidence()
    app.configure(objective="answer support questions")
    app.prompt_spec = app.prompt_spec.model_copy(update={"name": "kb-prompt"})
    server = build_app_server(app)
    resp = await server.handle(_req(1, "prompts/list"))
    prompts = resp["result"]["prompts"]
    assert prompts[0]["name"] == "kb-prompt"
    assert prompts[0]["description"] == "answer support questions"
    assert prompts[0]["arguments"] == [
        {"name": "task", "description": "the user task", "required": True}
    ]


@pytest.mark.asyncio
async def test_build_app_server_get_prompt_compiles_messages():
    app = _app_with_tool_and_evidence()
    app.configure(objective="be helpful")
    server = build_app_server(app)
    resp = await server.handle(_req(1, "prompts/get", {"name": "kb", "arguments": {"task": "reset password"}}))
    result = resp["result"]
    assert result["description"] == "be helpful"
    # Compiled messages carry text content blocks; 'developer' role is downgraded to 'user'.
    assert result["messages"]
    roles = {m["role"] for m in result["messages"]}
    assert "developer" not in roles
    assert all(m["content"]["type"] == "text" for m in result["messages"])


@pytest.mark.asyncio
async def test_build_app_server_expose_prompts_false_disables_prompts():
    app = _app_with_tool_and_evidence()
    server = build_app_server(app, expose_prompts=False)
    lst = await server.handle(_req(1, "prompts/list"))
    assert lst["result"] == {"prompts": []}
    get = await server.handle(_req(2, "prompts/get", {"name": "x"}))
    assert get["error"]["code"] == METHOD_NOT_FOUND


@pytest.mark.asyncio
async def test_build_app_server_name_override():
    app = _app_with_tool_and_evidence()
    server = build_app_server(app, name="custom-name")
    assert server.name == "custom-name"
    # Default uses the app name.
    assert build_app_server(app).name == "kb"


# -- _app_resources: dedup across pending + ingested --------------------------


def test_app_resources_dedups_by_id_keeping_order():
    app = ContextApp(name="r", provider=MockProvider(), model="mock-1")
    e1 = EvidenceItem(id="a", source_id="S1", text="one")
    e2 = EvidenceItem(id="b", source_id="S2", text="two")
    dup = EvidenceItem(id="a", source_id="S1-dup", text="one-again")
    app.pending_evidence.extend([e1, e2])
    app._ingested_files["/file.txt"] = [dup, EvidenceItem(id="c", source_id="S3", text="three")]
    out = _app_resources(app)
    ids = [ev.id for ev in out]
    assert ids == ["a", "b", "c"]
    # The first 'a' wins (pending evidence), the ingested dup is dropped.
    assert out[0].source_id == "S1"


# -- serve_stdio: newline-delimited JSON-RPC over in-memory streams -----------


class _LineReader:
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        return self._lines.pop(0) if self._lines else ""


@pytest.mark.asyncio
async def test_serve_stdio_processes_requests_and_writes_responses():
    srv = MCPServer(name="stdio", list_tools=lambda: [{"name": "t", "description": "", "inputSchema": {}}])
    reader = _LineReader(
        [
            json.dumps(_req(1, "tools/list")) + "\n",
            "\n",  # blank line is skipped
            "{ not json }\n",  # JSONDecodeError is swallowed
            json.dumps({"jsonrpc": "2.0", "method": "notifications/x"}) + "\n",  # no response
            json.dumps(_req(2, "ping")) + "\n",
            "",  # EOF
        ]
    )
    writer = io.StringIO()
    await serve_stdio(srv, reader=reader, writer=writer)
    lines = [json.loads(line) for line in writer.getvalue().splitlines()]
    # Only the two requests with ids produced responses, in order.
    assert [m["id"] for m in lines] == [1, 2]
    assert lines[0]["result"] == {"tools": [{"name": "t", "description": "", "inputSchema": {}}]}
    assert lines[1]["result"] == {}


@pytest.mark.asyncio
async def test_serve_stdio_stops_at_immediate_eof():
    srv = MCPServer(name="stdio")
    writer = io.StringIO()
    await serve_stdio(srv, reader=_LineReader([""]), writer=writer)
    assert writer.getvalue() == ""
