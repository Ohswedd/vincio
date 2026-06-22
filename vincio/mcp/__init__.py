"""Vincio Model Context Protocol (MCP) support — client *and* server.

Consume MCP servers as tools/resources/prompts, and expose a Vincio app as an
MCP server, all over stdio, Streamable HTTP, or an in-process transport (for
offline tests). MCP tools run through the same permissioned, sandboxed, audited,
budgeted runtime as native tools; MCP resources become cited evidence.

    from vincio.mcp import connect_stdio
    client = connect_stdio(["python", "weather_server.py"])
    await client.register_into(app)        # tools + resources land in the app

    server = build_app_server(app)          # expose the app as an MCP server
"""

from __future__ import annotations

from typing import Any

from .apps import (
    ElicitationAction,
    ElicitationDecision,
    ElicitationGate,
    ElicitationPolicy,
    ElicitationRequest,
    ElicitationResponse,
    MCPAppBridge,
    MCPUIRender,
    is_ui_resource,
)
from .client import MCPClient
from .oauth import bearer_headers, pkce_pair, static_token_validator
from .protocol import (
    PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    MCPError,
    MCPPromptInfo,
    MCPResourceInfo,
    MCPTask,
    MCPToolInfo,
    negotiate_version,
)
from .server import MCPServer, MCPUIResource, build_app_server, serve_stdio
from .transport import (
    InProcessTransport,
    MCPTransport,
    StdioTransport,
    StreamableHTTPTransport,
)

__all__ = [
    "MCPClient",
    "MCPServer",
    "MCPUIResource",
    "build_app_server",
    "serve_stdio",
    "MCPTransport",
    "InProcessTransport",
    "StdioTransport",
    "StreamableHTTPTransport",
    "MCPError",
    "MCPToolInfo",
    "MCPResourceInfo",
    "MCPPromptInfo",
    "MCPTask",
    "PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "negotiate_version",
    "ElicitationAction",
    "ElicitationRequest",
    "ElicitationResponse",
    "ElicitationPolicy",
    "ElicitationDecision",
    "ElicitationGate",
    "MCPUIRender",
    "MCPAppBridge",
    "is_ui_resource",
    "pkce_pair",
    "bearer_headers",
    "static_token_validator",
    "connect_stdio",
    "connect_http",
    "connect_in_process",
]


def connect_stdio(command: list[str], **client_kwargs: Any) -> MCPClient:
    """Connect to an MCP server launched as a subprocess (stdio transport)."""
    return MCPClient(StdioTransport(command), **client_kwargs)


def connect_http(url: str, *, headers: dict[str, str] | None = None, **client_kwargs: Any) -> MCPClient:
    """Connect to an MCP server over Streamable HTTP."""
    client = client_kwargs.pop("http_client", None)
    return MCPClient(
        StreamableHTTPTransport(url, headers=headers, client=client), **client_kwargs
    )


def connect_in_process(server: MCPServer, *, auth: str | None = None, **client_kwargs: Any) -> MCPClient:
    """Connect to an in-process :class:`MCPServer` (offline tests, local serving)."""
    return MCPClient(InProcessTransport(server, auth=auth), **client_kwargs)
