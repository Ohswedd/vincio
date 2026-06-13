"""MCP server: expose tools / resources / prompts over JSON-RPC.

:class:`MCPServer` is transport-agnostic — it consumes one JSON-RPC message and
produces one response (or ``None`` for notifications). :func:`build_app_server`
wires a :class:`~vincio.core.app.ContextApp` as a server: its registered tools
become MCP tools, its evidence/sources become MCP resources, and its prompt
spec becomes an MCP prompt — with the deterministic policy engine and the
hash-chained audit log enforced on every inbound call.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PROTOCOL_VERSION,
    MCPError,
    jsonrpc_error,
    jsonrpc_response,
    text_content,
)

__all__ = ["MCPServer", "build_app_server", "serve_stdio"]

# A token validator returns a principal-ish identity dict, or raises MCPError.
TokenValidator = Callable[[str | None], Awaitable[dict[str, Any]] | dict[str, Any]]


class MCPServer:
    """A minimal, transport-agnostic MCP server.

    Hand it provider callables for tools/resources/prompts. ``handle`` accepts a
    decoded JSON-RPC message plus an optional bearer ``auth`` string and returns
    the response message (``None`` for notifications).
    """

    def __init__(
        self,
        *,
        name: str = "vincio",
        version: str = "1.1.0",
        list_tools: Callable[[], list[dict[str, Any]]] | None = None,
        call_tool: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
        list_resources: Callable[[], list[dict[str, Any]]] | None = None,
        read_resource: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
        list_prompts: Callable[[], list[dict[str, Any]]] | None = None,
        get_prompt: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
        token_validator: TokenValidator | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self._list_tools = list_tools
        self._call_tool = call_tool
        self._list_resources = list_resources
        self._read_resource = read_resource
        self._list_prompts = list_prompts
        self._get_prompt = get_prompt
        self._token_validator = token_validator
        # Set by a bidirectional transport so handlers can initiate requests
        # back to the client (sampling/createMessage, elicitation/create).
        self.request_client: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None

    def capabilities(self) -> dict[str, Any]:
        caps: dict[str, Any] = {}
        if self._list_tools is not None:
            caps["tools"] = {"listChanged": False}
        if self._list_resources is not None:
            caps["resources"] = {"listChanged": False, "subscribe": False}
        if self._list_prompts is not None:
            caps["prompts"] = {"listChanged": False}
        return caps

    async def _validate(self, auth: str | None) -> None:
        if self._token_validator is None:
            return
        result = self._token_validator(auth)
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]

    async def handle(self, message: dict[str, Any], *, auth: str | None = None) -> dict[str, Any] | None:
        """Dispatch one JSON-RPC message; returns a response (or None for notifications)."""
        if message.get("jsonrpc") != "2.0":
            return jsonrpc_error(message.get("id"), INVALID_REQUEST, "not a JSON-RPC 2.0 message")
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}
        # Notifications (no id) get no response.
        if msg_id is None:
            return None
        # ``initialize`` and ``ping`` are unauthenticated; everything else is gated.
        if method not in ("initialize", "ping"):
            try:
                await self._validate(auth)
            except MCPError as exc:
                return jsonrpc_error(msg_id, exc.code, exc.message, exc.data)
        try:
            result = await self._dispatch(method, params)
        except MCPError as exc:
            return jsonrpc_error(msg_id, exc.code, exc.message, exc.data)
        except Exception as exc:  # pragma: no cover - defensive
            return jsonrpc_error(msg_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        if result is _METHOD_NOT_FOUND:
            return jsonrpc_error(msg_id, METHOD_NOT_FOUND, f"unknown method {method!r}")
        return jsonrpc_response(msg_id, result)

    async def _dispatch(self, method: str | None, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {"name": self.name, "version": self.version},
                "capabilities": self.capabilities(),
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self._list_tools() if self._list_tools else []}
        if method == "tools/call":
            if self._call_tool is None:
                raise MCPError("tools not supported", code=METHOD_NOT_FOUND)
            name = params.get("name")
            if not name:
                raise MCPError("tools/call requires 'name'", code=INVALID_PARAMS)
            output = await self._call_tool(name, params.get("arguments") or {})
            is_error = isinstance(output, dict) and output.get("_mcp_error") is True
            text = output.get("text") if isinstance(output, dict) and "text" in output else _stringify(output)
            return {"content": [text_content(text)], "isError": is_error}
        if method == "resources/list":
            return {"resources": self._list_resources() if self._list_resources else []}
        if method == "resources/read":
            if self._read_resource is None:
                raise MCPError("resources not supported", code=METHOD_NOT_FOUND)
            uri = params.get("uri")
            if not uri:
                raise MCPError("resources/read requires 'uri'", code=INVALID_PARAMS)
            return {"contents": [await self._read_resource(uri)]}
        if method == "prompts/list":
            return {"prompts": self._list_prompts() if self._list_prompts else []}
        if method == "prompts/get":
            if self._get_prompt is None:
                raise MCPError("prompts not supported", code=METHOD_NOT_FOUND)
            name = params.get("name")
            if not name:
                raise MCPError("prompts/get requires 'name'", code=INVALID_PARAMS)
            return await self._get_prompt(name, params.get("arguments") or {})
        return _METHOD_NOT_FOUND


_METHOD_NOT_FOUND = object()


def _stringify(value: Any) -> str:
    import json

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)


def build_app_server(
    app: Any,
    *,
    name: str | None = None,
    expose_resources: bool = True,
    expose_prompts: bool = True,
    token_validator: TokenValidator | None = None,
) -> MCPServer:
    """Expose a :class:`ContextApp` as an MCP server.

    Registered tools become MCP tools (run through the permissioned, sandboxed,
    audited tool runtime); the app's evidence/sources become MCP resources; the
    prompt spec becomes an MCP prompt. Every inbound ``tools/call`` is recorded
    on the hash-chained audit log.
    """
    from ..core.types import ToolCall

    def list_tools() -> list[dict[str, Any]]:
        names = app.enabled_tools or app.tool_registry.names
        out: list[dict[str, Any]] = []
        for spec in app.tool_registry.specs(names):
            out.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "inputSchema": spec.input_schema
                    or {"type": "object", "properties": {}},
                }
            )
        return out

    async def call_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
        app.audit.record(
            "mcp_serve",
            resource=tool_name,
            decision="received",
            details={"transport": "mcp", "direction": "inbound"},
        )
        if tool_name not in app.tool_registry:
            return {"_mcp_error": True, "text": f"unknown tool {tool_name!r}"}
        call = ToolCall(tool_name=tool_name, arguments=arguments, requested_by="user")
        result = await app.tool_runtime.execute(call)
        if result.status != "ok":
            return {"_mcp_error": True, "text": result.error or f"tool {result.status}"}
        return {"text": _stringify(result.output)}

    def list_resources() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ev in _app_resources(app):
            out.append(
                {
                    "uri": f"vincio://evidence/{ev.id}",
                    "name": ev.source_id,
                    "description": (ev.text or "")[:80],
                    "mimeType": "text/plain",
                }
            )
        return out

    async def read_resource(uri: str) -> dict[str, Any]:
        ev_id = uri.rsplit("/", 1)[-1]
        for ev in _app_resources(app):
            if ev.id == ev_id:
                return {"uri": uri, "mimeType": "text/plain", "text": ev.text or ""}
        raise MCPError(f"unknown resource {uri!r}", code=INVALID_PARAMS)

    def list_prompts() -> list[dict[str, Any]]:
        spec = app.prompt_spec
        return [
            {
                "name": spec.name or app.name,
                "description": spec.objective or "Vincio app prompt",
                "arguments": [{"name": "task", "description": "the user task", "required": True}],
            }
        ]

    async def get_prompt(prompt_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        spec = app.prompt_spec
        compiled = app.prompt_compiler.compile(
            spec, user_task=str(arguments.get("task", "")), variables=app.prompt_variables
        )
        messages = [
            {"role": m.role if m.role != "developer" else "user", "content": text_content(m.text)}
            for m in compiled.messages
            if m.text
        ]
        return {"description": spec.objective or "", "messages": messages}

    return MCPServer(
        name=name or app.name,
        list_tools=list_tools,
        call_tool=call_tool,
        list_resources=list_resources if expose_resources else None,
        read_resource=read_resource if expose_resources else None,
        list_prompts=list_prompts if expose_prompts else None,
        get_prompt=get_prompt if expose_prompts else None,
        token_validator=token_validator,
    )


async def serve_stdio(server: MCPServer, *, reader: Any = None, writer: Any = None) -> None:
    """Serve an :class:`MCPServer` over newline-delimited JSON-RPC on stdio.

    Reads requests from ``reader`` (default: stdin) and writes responses to
    ``writer`` (default: stdout). Runs until EOF. Used by ``vincio mcp serve``.
    """
    import asyncio
    import json
    import sys

    out = writer or sys.stdout
    loop = asyncio.get_event_loop()

    def _readline() -> str:
        return (reader or sys.stdin).readline()

    while True:
        line = await loop.run_in_executor(None, _readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = await server.handle(message)
        if response is not None:
            out.write(json.dumps(response) + "\n")
            out.flush()


def _app_resources(app: Any) -> list[Any]:
    items = list(app.pending_evidence)
    for evs in app._ingested_files.values():
        items.extend(evs)
    # De-duplicate by id, keep order.
    seen: set[str] = set()
    out = []
    for ev in items:
        if ev.id not in seen:
            seen.add(ev.id)
            out.append(ev)
    return out
