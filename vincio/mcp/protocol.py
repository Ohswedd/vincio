"""Model Context Protocol (MCP) wire types: JSON-RPC 2.0 + MCP primitives.

This module is transport-agnostic and dependency-free. The same JSON-RPC
envelopes are carried over stdio, Streamable HTTP, or the in-process transport
used for offline tests.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import VincioError

__all__ = [
    "PROTOCOL_VERSION",
    "MCPError",
    "MCPToolInfo",
    "MCPResourceInfo",
    "MCPPromptInfo",
    "MCPTask",
    "jsonrpc_request",
    "jsonrpc_response",
    "jsonrpc_error",
    "text_content",
    "PARSE_ERROR",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
]

# The stable MCP revision Vincio implements. (The in-flight 2025-11-25 spec and
# MCP Apps are tracked under the roadmap's "Exploring" section.)
PROTOCOL_VERSION = "2025-06-18"

# JSON-RPC 2.0 error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class MCPError(VincioError):
    """An MCP protocol or transport error. ``code`` is the JSON-RPC error code."""

    # JSON-RPC codes are numeric; this intentionally narrows the str base ``code``.
    code: int = INTERNAL_ERROR  # type: ignore[assignment]

    def __init__(self, message: str, *, code: int = INTERNAL_ERROR, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class MCPToolInfo(BaseModel):
    """A tool advertised by an MCP server (``tools/list``)."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPResourceInfo(BaseModel):
    """A resource advertised by an MCP server (``resources/list``)."""

    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = "text/plain"


class MCPPromptInfo(BaseModel):
    """A prompt advertised by an MCP server (``prompts/list``)."""

    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = Field(default_factory=list)


class MCPTask(BaseModel):
    """A long-running task handle (poll via ``tasks/get`` / await)."""

    task_id: str
    status: str = "working"  # working | completed | failed
    result: Any = None
    error: str | None = None


# -- JSON-RPC envelope helpers ------------------------------------------------


def jsonrpc_request(id: Any, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def jsonrpc_response(id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def jsonrpc_error(id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id, "error": err}


def text_content(text: str) -> dict[str, Any]:
    """A single MCP text content block."""
    return {"type": "text", "text": text}
