"""Real-behavior coverage for vincio.mcp.transport.

All offline & deterministic: InProcessTransport routes to a real MCPServer,
StreamableHTTPTransport runs over httpx.MockTransport, and StdioTransport
spawns a tiny line-delimited JSON-RPC echo server subprocess (pure stdlib,
no network). No unittest.mock anywhere.
"""

from __future__ import annotations

import json
import sys
import textwrap

import httpx
import pytest

from vincio.mcp.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    MCPError,
    jsonrpc_response,
)
from vincio.mcp.server import MCPServer
from vincio.mcp.transport import (
    InProcessTransport,
    StdioTransport,
    StreamableHTTPTransport,
)

# -- A real, dependency-free MCPServer to route in-process requests at ---------


def _calc_server(*, token_validator=None) -> MCPServer:
    async def call_tool(name: str, args: dict) -> dict:
        if name == "add":
            return {"text": str(args["a"] + args["b"])}
        if name == "boom":
            raise MCPError("tool exploded", code=INVALID_PARAMS)
        if name == "ask":
            # Server initiates a request back to the client mid-call.
            assert server.request_client is not None
            answer = await server.request_client("elicitation/create", {"message": "name?"})
            return {"text": answer["content"]["value"]}
        raise MCPError("no such tool", code=-32601)

    server = MCPServer(
        list_tools=lambda: [{"name": "add", "description": "", "inputSchema": {}}],
        call_tool=call_tool,
        token_validator=token_validator,
    )
    return server


# -- InProcessTransport --------------------------------------------------------


async def test_inprocess_request_returns_result():
    t = InProcessTransport(_calc_server())
    out = await t.request("tools/call", {"name": "add", "arguments": {"a": 2, "b": 3}})
    assert out == {"content": [{"type": "text", "text": "5"}], "isError": False}


async def test_inprocess_counter_increments_per_request():
    t = InProcessTransport(_calc_server())
    await t.request("ping")
    await t.request("ping")
    # Two requests -> the internal id counter advanced to 2.
    assert t._counter == 2


async def test_inprocess_error_response_raises_mcperror_with_code():
    t = InProcessTransport(_calc_server())
    with pytest.raises(MCPError, match="tool exploded") as exc:
        await t.request("tools/call", {"name": "boom", "arguments": {}})
    assert exc.value.code == INVALID_PARAMS


async def test_inprocess_notification_returns_none():
    # Notifications go through server.handle with no id -> None response.
    t = InProcessTransport(_calc_server())
    assert await t.notify("notifications/initialized", {"x": 1}) is None


async def test_inprocess_request_none_when_server_returns_none():
    # handle() returns None for a message with no id; force that path by
    # sending a notification-shaped dispatch through a server that returns None.
    class NoneServer:
        request_client = None

        async def handle(self, message, *, auth=None):
            return None

    t = InProcessTransport(NoneServer())
    assert await t.request("ping") is None


async def test_inprocess_auth_is_forwarded_to_server():
    seen: list[str | None] = []

    def validator(auth):
        seen.append(auth)
        if auth != "secret":
            raise MCPError("unauthorized", code=-32001)

    t = InProcessTransport(_calc_server(token_validator=validator), auth="secret")
    await t.request("tools/list")
    assert seen == ["secret"]


async def test_inprocess_bad_auth_raises():
    def validator(auth):
        raise MCPError("nope", code=-32001)

    t = InProcessTransport(_calc_server(token_validator=validator), auth="wrong")
    with pytest.raises(MCPError, match="nope"):
        await t.request("tools/list")


async def test_inprocess_server_to_client_without_handler_raises():
    server = _calc_server()
    t = InProcessTransport(server)
    # No on_server_request handler registered -> server-initiated call errors,
    # which surfaces through tools/call as an MCPError.
    with pytest.raises(MCPError, match="no client handler for server request"):
        await t.request("tools/call", {"name": "ask", "arguments": {}})


async def test_inprocess_server_to_client_invokes_handler():
    server = _calc_server()
    t = InProcessTransport(server)

    async def on_request(method: str, params: dict):
        assert method == "elicitation/create"
        return {"content": {"value": "Ada"}}

    t.on_server_request = on_request
    out = await t.request("tools/call", {"name": "ask", "arguments": {}})
    assert out["content"][0]["text"] == "Ada"


async def test_inprocess_aclose_is_noop():
    t = InProcessTransport(_calc_server())
    assert await t.aclose() is None


# -- StreamableHTTPTransport ---------------------------------------------------


def _http(handler, **kw) -> StreamableHTTPTransport:
    return StreamableHTTPTransport(
        "https://mcp.example/rpc",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kw,
    )


async def test_http_tracks_session_id_and_resends_it():
    seen_session: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_session.append(request.headers.get("Mcp-Session-Id"))
        body = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"Mcp-Session-Id": "sess-42"},
            json=jsonrpc_response(body["id"], {"ok": True}),
        )

    t = _http(handler)
    await t.request("initialize")
    await t.request("tools/list")
    # First request had no session, second carried the captured id.
    assert seen_session == [None, "sess-42"]
    assert t._session_id == "sess-42"
    await t.aclose()


async def test_http_stateless_never_sends_session_id():
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Mcp-Session-Id"))
        body = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"Mcp-Session-Id": "sess-99"},
            json=jsonrpc_response(body["id"], {}),
        )

    t = _http(handler, stateless=True)
    await t.request("initialize")
    await t.request("tools/list")
    # Stateless mode ignores the server's session id entirely.
    assert seen == [None, None]
    assert t._session_id is None
    await t.aclose()


async def test_http_401_maps_to_unauthorized():
    t = _http(lambda r: httpx.Response(401, text="nope"))
    with pytest.raises(MCPError, match="unauthorized") as exc:
        await t.request("tools/list")
    assert exc.value.code == -32001
    assert exc.value.data == {"status": 401}


async def test_http_500_maps_to_internal_error_with_truncated_text():
    big = "x" * 500
    t = _http(lambda r: httpx.Response(500, text=big))
    with pytest.raises(MCPError, match=r"HTTP 500") as exc:
        await t.request("tools/list")
    assert exc.value.code == INTERNAL_ERROR
    # Body is truncated to 200 chars in the message.
    assert ("x" * 200) in str(exc.value)
    assert ("x" * 201) not in str(exc.value)


async def test_http_transport_error_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    t = _http(handler)
    with pytest.raises(MCPError, match="transport error: refused") as exc:
        await t.request("ping")
    assert exc.value.code == INTERNAL_ERROR


async def test_http_jsonrpc_error_body_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "error": {"code": -32601, "message": "method gone", "data": {"why": "x"}},
            },
        )

    t = _http(handler)
    with pytest.raises(MCPError, match="method gone") as exc:
        await t.request("tools/list")
    assert exc.value.code == -32601
    assert exc.value.data == {"why": "x"}


async def test_http_202_empty_body_returns_none():
    t = _http(lambda r: httpx.Response(202, text=""))
    assert await t.request("ping") is None


async def test_http_notify_sends_no_session_when_unset():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["session"] = request.headers.get("Mcp-Session-Id")
        captured["body"] = json.loads(request.content)
        return httpx.Response(202, text="")

    t = _http(handler)
    await t.notify("notifications/initialized", {"a": 1})
    assert "Mcp-Session-Id" not in captured  # only set key exists
    assert captured["session"] is None
    assert captured["body"] == {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {"a": 1},
    }


async def test_http_notify_resends_session_in_sticky_mode():
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Mcp-Session-Id"))
        body = json.loads(request.content)
        if body.get("id") is None:
            return httpx.Response(202, text="")
        return httpx.Response(
            200, headers={"Mcp-Session-Id": "s1"}, json=jsonrpc_response(body["id"], {})
        )

    t = _http(handler)
    await t.request("initialize")  # establishes session s1
    await t.notify("notifications/initialized")
    assert seen == [None, "s1"]


async def test_http_default_headers_set():
    t = _http(lambda r: httpx.Response(202, text=""))
    assert t.headers["Content-Type"] == "application/json"
    assert "text/event-stream" in t.headers["Accept"]


async def test_http_custom_headers_merge():
    t = StreamableHTTPTransport(
        "https://x/rpc",
        headers={"Authorization": "Bearer abc"},
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(202))),
    )
    assert t.headers["Authorization"] == "Bearer abc"
    assert t.headers["Content-Type"] == "application/json"


async def test_http_ensure_client_lazily_builds_real_client():
    # No injected client -> _ensure_client constructs a real httpx.AsyncClient.
    t = StreamableHTTPTransport("https://unused.example/rpc")
    assert t._client is None
    assert t._owns_client is True
    client = t._ensure_client()
    assert isinstance(client, httpx.AsyncClient)
    # A second call reuses the same instance.
    assert t._ensure_client() is client
    await t.aclose()  # owns the client -> closes it
    assert client.is_closed


async def test_http_aclose_does_not_close_injected_client():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(202)))
    t = StreamableHTTPTransport("https://x/rpc", client=client)
    assert t._owns_client is False
    await t.aclose()
    assert client.is_closed is False
    await client.aclose()


# -- _decode_http_body: SSE + edge cases ---------------------------------------


async def test_http_sse_body_decoded():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sse = (
            "event: message\n"
            f"data: {json.dumps(jsonrpc_response(body['id'], {'sse': True}))}\n"
            "\n"
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)

    t = _http(handler)
    assert await t.request("ping") == {"sse": True}


async def test_http_sse_skips_done_and_blank_then_returns_message():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sse = (
            "data: [DONE]\n"
            "data: \n"
            f"data: {json.dumps(jsonrpc_response(body['id'], {'final': 1}))}\n"
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)

    t = _http(handler)
    assert await t.request("ping") == {"final": 1}


async def test_http_sse_malformed_data_skipped_then_none():
    def handler(request: httpx.Request) -> httpx.Response:
        sse = "data: {not json}\ndata: also-bad\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)

    t = _http(handler)
    # No parseable data lines -> decode returns None -> request returns None.
    assert await t.request("ping") is None


async def test_http_sse_no_data_lines_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text="event: ping\n\n"
        )

    t = _http(handler)
    assert await t.request("ping") is None


async def test_http_whitespace_json_body_returns_none():
    t = _http(lambda r: httpx.Response(200, text="   \n  "))
    assert await t.request("ping") is None


# -- StdioTransport: real subprocess, line-delimited JSON-RPC ------------------

# A tiny stdlib-only MCP-ish server. It echoes requests, supports a tool that
# triggers a server-initiated request back to the client, and exits on a
# "shutdown" method so we exercise the read-loop EOF path too.
_STDIO_SERVER = textwrap.dedent(
    """
    import json, sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = msg.get("method")
        mid = msg.get("id")
        if mid is None:  # notification: no reply
            if method == "shutdown":
                break
            continue
        if method == "echo":
            out = {"jsonrpc": "2.0", "id": mid, "result": {"got": msg.get("params")}}
        elif method == "boom":
            out = {"jsonrpc": "2.0", "id": mid,
                   "error": {"code": -32000, "message": "server boom", "data": {"k": 1}}}
        elif method == "needclient":
            # server-initiated request to the client, then wait for its reply
            req = {"jsonrpc": "2.0", "id": 9001, "method": "ping_client", "params": {}}
            sys.stdout.write(json.dumps(req) + "\\n"); sys.stdout.flush()
            reply = json.loads(sys.stdin.readline())
            out = {"jsonrpc": "2.0", "id": mid, "result": {"client_said": reply.get("result")}}
        else:
            out = {"jsonrpc": "2.0", "id": mid, "result": {}}
        sys.stdout.write(json.dumps(out) + "\\n"); sys.stdout.flush()
    """
)


def _stdio_transport() -> StdioTransport:
    return StdioTransport([sys.executable, "-c", _STDIO_SERVER])


async def test_stdio_request_roundtrip_and_reuses_process():
    t = _stdio_transport()
    try:
        r1 = await t.request("echo", {"v": 1})
        assert r1 == {"got": {"v": 1}}
        proc1 = t._proc
        r2 = await t.request("echo", {"v": 2})
        assert r2 == {"got": {"v": 2}}
        # _ensure reused the already-running process; counter advanced to 2.
        assert t._proc is proc1
        assert t._counter == 2
    finally:
        await t.aclose()


async def test_stdio_error_response_raises_with_code_and_data():
    t = _stdio_transport()
    try:
        with pytest.raises(MCPError, match="server boom") as exc:
            await t.request("boom")
        assert exc.value.code == -32000
        assert exc.value.data == {"k": 1}
    finally:
        await t.aclose()


async def test_stdio_unknown_method_returns_empty_result():
    t = _stdio_transport()
    try:
        assert await t.request("whatever") == {}
    finally:
        await t.aclose()


async def test_stdio_server_initiated_request_answered_by_handler():
    t = _stdio_transport()

    async def on_request(method: str, params: dict):
        assert method == "ping_client"
        return {"pong": True}

    t.on_server_request = on_request
    try:
        out = await t.request("needclient")
        assert out == {"client_said": {"pong": True}}
    finally:
        await t.aclose()


async def test_stdio_server_initiated_request_without_handler_errors_back():
    # No on_server_request -> _answer_server_request replies with an MCPError;
    # the server records it as the client's reply.
    t = _stdio_transport()
    try:
        out = await t.request("needclient")
        # The client returned a jsonrpc error to the server; server echoed its
        # 'result' field (absent) as None.
        assert out == {"client_said": None}
    finally:
        await t.aclose()


async def test_stdio_notify_writes_without_awaiting_reply():
    t = _stdio_transport()
    try:
        # A notification has no id; the server consumes it and sends nothing.
        assert await t.notify("noted", {"a": 1}) is None
        # The transport is still usable for a subsequent request.
        assert await t.request("echo", {"after": "notify"}) == {"got": {"after": "notify"}}
    finally:
        await t.aclose()


async def test_stdio_aclose_terminates_process():
    t = _stdio_transport()
    await t.request("echo", {})
    proc = t._proc
    await t.aclose()
    # Process was terminated and reaped; reader task cancelled.
    assert proc.returncode is not None
    assert t._reader_task.cancelled() or t._reader_task.done()


async def test_stdio_aclose_idempotent_when_never_started():
    t = _stdio_transport()
    # Never sent a request -> no proc, no reader task. aclose must be safe.
    assert t._proc is None
    assert await t.aclose() is None


async def test_stdio_read_loop_ignores_malformed_lines():
    # Server that emits a junk line before the valid reply; the read loop must
    # skip the undecodable line (json.JSONDecodeError branch) and still resolve.
    server = textwrap.dedent(
        """
        import json, sys
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            mid = msg.get("id")
            sys.stdout.write("<<not json>>\\n"); sys.stdout.flush()
            sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":mid,"result":{"ok":1}}) + "\\n")
            sys.stdout.flush()
        """
    )
    t = StdioTransport([sys.executable, "-c", server])
    try:
        assert await t.request("echo", {}) == {"ok": 1}
    finally:
        await t.aclose()
