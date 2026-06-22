"""MCP client: consume external MCP servers as tools, resources, and prompts.

The client negotiates capabilities, surfaces ``tools`` / ``resources`` /
``prompts``, and answers server-initiated ``sampling`` (routed to a model
provider) and ``elicitation`` (routed to a human-gate callback) requests. It is
transport-agnostic — pair it with an in-process, stdio, or Streamable HTTP
:class:`~vincio.mcp.transport.MCPTransport`.

``register_into(app)`` wires a connected server into a
:class:`~vincio.core.app.ContextApp`: MCP tools register through the *existing*
permissioned, sandboxed, audited tool runtime; MCP resources become evidence
with ``origin: mcp:<server>`` provenance; MCP prompts import as ``PromptSpec``.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from ..core.types import EvidenceItem, Message, ModelRequest, TrustLevel

if TYPE_CHECKING:
    from .server import MCPUIResource
from .protocol import (
    INVALID_PARAMS,
    PROTOCOL_VERSION,
    MCPError,
    MCPPromptInfo,
    MCPResourceInfo,
    MCPTask,
    MCPToolInfo,
)
from .transport import MCPTransport

__all__ = ["MCPClient"]


class MCPClient:
    """A connected MCP client bound to one server transport."""

    def __init__(
        self,
        transport: MCPTransport,
        *,
        name: str = "mcp",
        sampling_provider: Any | None = None,
        sampling_model: str | None = None,
        elicitation_callback: Any | None = None,
        elicitation_gate: Any | None = None,
    ) -> None:
        self.transport = transport
        self.name = name
        self.server_info: dict[str, Any] = {}
        self.server_capabilities: dict[str, Any] = {}
        self.negotiated_version: str | None = None
        self.sampling_provider = sampling_provider
        self.sampling_model = sampling_model
        # ``elicitation_gate`` (an :class:`~vincio.mcp.apps.ElicitationGate`)
        # governs a server's mid-call input request through the approval + rail
        # machinery and takes precedence; ``elicitation_callback`` is the raw,
        # ungoverned fallback kept for backward compatibility.
        self.elicitation_callback = elicitation_callback
        self.elicitation_gate = elicitation_gate
        self._initialized = False
        transport.on_server_request = self._handle_server_request

    # -- lifecycle -------------------------------------------------------------

    async def initialize(self) -> dict[str, Any]:
        result = await self.transport.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "clientInfo": {"name": "vincio", "version": "1.1.0"},
                "capabilities": {"sampling": {}, "elicitation": {}},
            },
        )
        self.server_info = (result or {}).get("serverInfo", {})
        self.server_capabilities = (result or {}).get("capabilities", {})
        self.negotiated_version = (result or {}).get("protocolVersion")
        await self.transport.notify("notifications/initialized")
        self._initialized = True
        return result or {}

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def aclose(self) -> None:
        await self.transport.aclose()

    # -- discovery -------------------------------------------------------------

    async def list_tools(self) -> list[MCPToolInfo]:
        await self._ensure_initialized()
        result = await self.transport.request("tools/list")
        return [
            MCPToolInfo(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema") or {},
            )
            for t in (result or {}).get("tools", [])
        ]

    async def list_resources(self) -> list[MCPResourceInfo]:
        await self._ensure_initialized()
        result = await self.transport.request("resources/list")
        return [
            MCPResourceInfo(
                uri=r["uri"],
                name=r.get("name", ""),
                description=r.get("description", ""),
                mime_type=r.get("mimeType", "text/plain"),
            )
            for r in (result or {}).get("resources", [])
        ]

    async def list_ui_resources(self) -> list[MCPResourceInfo]:
        """The server's server-rendered UI resources (MCP Apps: ``ui://`` / UI MIME).

        A subset of :meth:`list_resources` filtered to renderable UI, for the
        :class:`~vincio.mcp.apps.MCPAppBridge` to surface through the AG-UI channel.
        """
        from .apps import is_ui_resource

        return [r for r in await self.list_resources() if is_ui_resource(r.uri, r.mime_type)]

    async def list_prompts(self) -> list[MCPPromptInfo]:
        await self._ensure_initialized()
        result = await self.transport.request("prompts/list")
        return [
            MCPPromptInfo(
                name=p["name"],
                description=p.get("description", ""),
                arguments=p.get("arguments") or [],
            )
            for p in (result or {}).get("prompts", [])
        ]

    # -- invocation ------------------------------------------------------------

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, *, await_result: bool = True
    ) -> str:
        await self._ensure_initialized()
        result = await self.transport.request(
            "tools/call", {"name": name, "arguments": arguments or {}}
        )
        result = result or {}
        # Long-running Tasks primitive: poll/await to completion.
        task_id = result.get("taskId") or (result.get("task") or {}).get("taskId")
        if task_id and await_result:
            task = await self._await_task(task_id)
            if task.status == "failed":
                raise MCPError(task.error or "task failed")
            result = task.result or {}
        if result.get("isError"):
            raise MCPError(_content_text(result) or f"tool {name!r} returned an error")
        return _content_text(result)

    async def call_tool_ui(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> tuple[str, list[MCPUIResource]]:
        """Call a tool and return its text plus any embedded UI resources.

        MCP Apps return server-rendered UI as an embedded ``resource`` content
        block on the tool result; this surfaces both the text and those UI
        resources (for the :class:`~vincio.mcp.apps.MCPAppBridge`), where
        :meth:`call_tool` returns the text alone.
        """
        from .apps import is_ui_resource
        from .server import MCPUIResource

        await self._ensure_initialized()
        result = await self.transport.request(
            "tools/call", {"name": name, "arguments": arguments or {}}
        )
        result = result or {}
        if result.get("isError"):
            raise MCPError(_content_text(result) or f"tool {name!r} returned an error")
        ui: list[MCPUIResource] = []
        for block in result.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "resource":
                res = block.get("resource") or {}
                if is_ui_resource(res.get("uri", ""), res.get("mimeType", "")):
                    ui.append(
                        MCPUIResource(
                            uri=res.get("uri", ""),
                            name=res.get("uri", ""),
                            mime_type=res.get("mimeType", "text/html"),
                            text=res.get("text", ""),
                        )
                    )
        return _content_text(result), ui

    async def read_resource(self, uri: str) -> str:
        await self._ensure_initialized()
        result = await self.transport.request("resources/read", {"uri": uri})
        contents = (result or {}).get("contents") or []
        return "\n\n".join(c.get("text", "") for c in contents if c.get("text"))

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        await self._ensure_initialized()
        result = await self.transport.request(
            "prompts/get", {"name": name, "arguments": arguments or {}}
        )
        return (result or {}).get("messages", [])

    async def _await_task(
        self,
        task_id: str,
        *,
        deadline_s: float = 30.0,
        initial_delay_s: float = 0.05,
        max_delay_s: float = 2.0,
    ) -> MCPTask:
        """Poll a long-running task to a terminal state with exponential backoff
        and a wall-clock deadline — no busy-loop. The first poll runs immediately
        (a fast task never sleeps); subsequent polls back off up to ``max_delay_s``
        until the task finishes or the deadline passes."""
        start = time.monotonic()
        delay = initial_delay_s
        while True:
            result = await self.transport.request("tasks/get", {"taskId": task_id})
            task = MCPTask(
                task_id=task_id,
                status=(result or {}).get("status", "working"),
                result=(result or {}).get("result"),
                error=(result or {}).get("error"),
            )
            if task.status in ("completed", "failed"):
                return task
            if time.monotonic() - start >= deadline_s:
                return MCPTask(
                    task_id=task_id, status="failed",
                    error=f"task poll deadline ({deadline_s}s) exceeded while status={task.status!r}",
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, max_delay_s)

    # -- server-initiated requests (sampling / elicitation) --------------------

    async def _handle_server_request(self, method: str, params: dict[str, Any]) -> Any:
        if method == "sampling/createMessage":
            return await self._sample(params)
        if method == "elicitation/create":
            return await self._elicit(params)
        raise MCPError(f"unsupported server request {method!r}", code=INVALID_PARAMS)

    async def _sample(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.sampling_provider is None:
            raise MCPError("server requested sampling but no provider is configured")
        messages: list[Message] = []
        system = params.get("systemPrompt")
        if system:
            messages.append(Message(role="system", content=str(system)))
        for m in params.get("messages") or []:
            content = m.get("content") or {}
            text = content.get("text") if isinstance(content, dict) else str(content)
            role = m.get("role", "user")
            messages.append(Message(role=role if role in ("user", "assistant") else "user", content=text or ""))
        request = ModelRequest(
            model=self.sampling_model or "mock-1",
            messages=messages,
            max_output_tokens=params.get("maxTokens") or 1024,
            temperature=params.get("temperature"),
        )
        response = await self.sampling_provider.generate(request)
        return {
            "role": "assistant",
            "content": {"type": "text", "text": response.text},
            "model": response.model,
            "stopReason": "endTurn",
        }

    async def _elicit(self, params: dict[str, Any]) -> dict[str, Any]:
        # The governed path: an ElicitationGate runs the approval + rail machinery
        # and taints the accepted value untrusted.
        if self.elicitation_gate is not None:
            from .apps import ElicitationRequest

            request = ElicitationRequest.from_params(params, server=self.name)
            decision = await self.elicitation_gate.decide(request)
            return decision.to_wire()
        # The raw, ungoverned fallback (no rail screening / taint).
        message = params.get("message", "")
        schema = params.get("requestedSchema")
        if self.elicitation_callback is None:
            return {"action": "decline"}
        result = self.elicitation_callback(message, schema)
        if hasattr(result, "__await__"):
            result = await result
        if result is None or result is False:
            return {"action": "decline"}
        if result is True:
            return {"action": "accept", "content": {}}
        return {"action": "accept", "content": result}

    # -- integration with a ContextApp -----------------------------------------

    async def register_into(
        self,
        app: Any,
        *,
        tools: bool = True,
        resources: bool = True,
        prompts: bool = False,
        permissions: list[str] | None = None,
        side_effects: str = "external",
    ) -> dict[str, list[str]]:
        """Register this server's tools/resources/prompts into ``app``.

        Returns a manifest: ``{"tools": [...], "resources": [...], "prompts": [...]}``.
        """
        await self._ensure_initialized()
        registered: dict[str, list[str]] = {"tools": [], "resources": [], "prompts": []}
        # Default to no extra RBAC scope (like native add_tool); MCP tools still
        # run through the full permissioned/sandboxed/audited runtime. Pass
        # ``permissions=["mcp:<server>"]`` to additionally gate them behind a scope.
        perms = permissions if permissions is not None else []
        if tools:
            for tool in await self.list_tools():
                self._register_tool(app, tool, perms, side_effects)
                registered["tools"].append(tool.name)
        if resources:
            for resource in await self.list_resources():
                text = await self.read_resource(resource.uri)
                app.pending_evidence.append(
                    EvidenceItem(
                        source_id=resource.uri,
                        source_type="document",
                        text=text,
                        trust_level=TrustLevel.UNTRUSTED_EXTERNAL,
                        relevance=0.5,
                        provenance=0.8,
                        metadata={"origin": f"mcp:{self.name}", "mime_type": resource.mime_type},
                    )
                )
                registered["resources"].append(resource.uri)
        if prompts:
            for prompt in await self.list_prompts():
                registered["prompts"].append(prompt.name)
        return registered

    def _register_tool(
        self, app: Any, tool: MCPToolInfo, permissions: list[str], side_effects: str
    ) -> None:
        client = self
        tool_name = tool.name

        async def handler(**kwargs: Any) -> str:
            # MCP tools run through the *existing* permissioned/sandboxed/audited
            # tool runtime — this handler is only the transport bridge.
            return await client.call_tool(tool_name, kwargs)

        # Namespace by server to avoid collisions with native tools.
        registered_name = f"{self.name}.{tool_name}"
        app.tool_registry.register(
            handler,
            name=registered_name,
            description=tool.description or f"MCP tool {tool_name!r} from {self.name!r}",
            input_schema=tool.input_schema or {"type": "object", "properties": {}},
            permissions=permissions,
            side_effects=side_effects,
        )
        if registered_name not in app.enabled_tools:
            app.enabled_tools.append(registered_name)

    async def import_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Import an MCP prompt as a :class:`PromptSpec`."""
        from ..prompts.templates import PromptSpec

        messages = await self.get_prompt(name, arguments)
        objective = "\n\n".join(
            (m.get("content") or {}).get("text", "")
            for m in messages
            if isinstance(m.get("content"), dict)
        )
        return PromptSpec(name=name, objective=objective.strip() or name)


def _content_text(result: dict[str, Any]) -> str:
    blocks = result.get("content") or []
    return "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
