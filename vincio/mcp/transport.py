"""MCP transports: in-process, stdio, and Streamable HTTP.

All three carry the same JSON-RPC envelopes. ``InProcessTransport`` routes
directly to an :class:`~vincio.mcp.server.MCPServer` in the same process — the
established Vincio pattern for fully offline, deterministic protocol tests and
for serving a local app to a local client. ``StdioTransport`` spawns a server
subprocess; ``StreamableHTTPTransport`` POSTs JSON-RPC over httpx (with an SSE
read path), and accepts an injected client so it is testable with
``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from .protocol import INTERNAL_ERROR, MCPError, jsonrpc_request

__all__ = ["MCPTransport", "InProcessTransport", "StdioTransport", "StreamableHTTPTransport"]

# Handler the client registers to answer server-initiated requests
# (sampling/createMessage, elicitation/create).
ServerRequestHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]


class MCPTransport(ABC):
    """Bidirectional JSON-RPC transport."""

    on_server_request: ServerRequestHandler | None = None

    @abstractmethod
    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a request and await its result (raises MCPError on error)."""

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:  # noqa: B027
        """Send a notification (no response expected). Optional; default no-op."""

    async def aclose(self) -> None:
        return None


class InProcessTransport(MCPTransport):
    """Route requests directly to an in-process :class:`MCPServer`."""

    def __init__(self, server: Any, *, auth: str | None = None) -> None:
        self.server = server
        self.auth = auth
        self._counter = 0
        # Let the server initiate requests back to the client (sampling/elicitation).
        server.request_client = self._server_to_client

    async def _server_to_client(self, method: str, params: dict[str, Any]) -> Any:
        if self.on_server_request is None:
            raise MCPError(f"no client handler for server request {method!r}")
        return await self.on_server_request(method, params)

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._counter += 1
        response = await self.server.handle(
            jsonrpc_request(self._counter, method, params), auth=self.auth
        )
        if response is None:
            return None
        if "error" in response:
            err = response["error"]
            raise MCPError(err["message"], code=err["code"], data=err.get("data"))
        return response.get("result")

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self.server.handle({"jsonrpc": "2.0", "method": method, "params": params or {}})


class StdioTransport(MCPTransport):
    """Spawn an MCP server subprocess and talk newline-delimited JSON-RPC."""

    def __init__(self, command: list[str], *, env: dict[str, str] | None = None, cwd: str | None = None) -> None:
        self.command = command
        self.env = env
        self.cwd = cwd
        self._proc: asyncio.subprocess.Process | None = None
        self._counter = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def _ensure(self) -> asyncio.subprocess.Process:
        if self._proc is not None and self._proc.returncode is None:
            return self._proc
        self._proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
            cwd=self.cwd,
        )
        self._reader_task = asyncio.ensure_future(self._read_loop(self._proc))
        return self._proc

    async def _read_loop(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                message = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            await self._on_message(message)

    async def _on_message(self, message: dict[str, Any]) -> None:
        if "method" in message and "id" in message:
            # Server-initiated request: answer it.
            await self._answer_server_request(message)
            return
        msg_id = message.get("id")
        future = self._pending.pop(msg_id, None) if msg_id is not None else None
        if future is not None and not future.done():
            future.set_result(message)

    async def _answer_server_request(self, message: dict[str, Any]) -> None:
        from .protocol import jsonrpc_error, jsonrpc_response

        assert self._proc is not None and self._proc.stdin is not None
        try:
            if self.on_server_request is None:
                raise MCPError(f"no client handler for {message.get('method')!r}")
            result = await self.on_server_request(message["method"], message.get("params") or {})
            reply = jsonrpc_response(message["id"], result)
        except MCPError as exc:
            reply = jsonrpc_error(message["id"], exc.code, exc.message)
        except Exception as exc:  # pragma: no cover - defensive
            reply = jsonrpc_error(message["id"], INTERNAL_ERROR, str(exc))
        self._proc.stdin.write((json.dumps(reply) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        proc = await self._ensure()
        assert proc.stdin is not None
        async with self._lock:
            self._counter += 1
            msg_id = self._counter
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = future
        proc.stdin.write((json.dumps(jsonrpc_request(msg_id, method, params)) + "\n").encode("utf-8"))
        await proc.stdin.drain()
        response = await future
        if "error" in response:
            err = response["error"]
            raise MCPError(err["message"], code=err["code"], data=err.get("data"))
        return response.get("result")

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        proc = await self._ensure()
        assert proc.stdin is not None
        proc.stdin.write(
            (json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}}) + "\n").encode("utf-8")
        )
        await proc.stdin.drain()

    async def aclose(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:  # pragma: no cover - defensive
                self._proc.kill()


class StreamableHTTPTransport(MCPTransport):
    """POST JSON-RPC to a Streamable HTTP MCP endpoint (legacy SSE accepted)."""

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        client: Any | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.url = url
        self.headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if headers:
            self.headers.update(headers)
        self._client = client
        self._owns_client = client is None
        self.timeout_s = timeout_s
        self._counter = 0
        self._session_id: str | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        import httpx

        client = self._ensure_client()
        self._counter += 1
        headers = dict(self.headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        try:
            resp = await client.post(
                self.url, json=jsonrpc_request(self._counter, method, params), headers=headers
            )
        except httpx.HTTPError as exc:
            raise MCPError(f"transport error: {exc}", code=INTERNAL_ERROR) from exc
        if resp.status_code == 401:
            raise MCPError("unauthorized", code=-32001, data={"status": 401})
        if resp.status_code >= 400:
            raise MCPError(f"HTTP {resp.status_code}: {resp.text[:200]}", code=INTERNAL_ERROR)
        session = resp.headers.get("Mcp-Session-Id")
        if session:
            self._session_id = session
        message = _decode_http_body(resp)
        if message is None:
            return None
        if "error" in message:
            err = message["error"]
            raise MCPError(err["message"], code=err["code"], data=err.get("data"))
        return message.get("result")

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        client = self._ensure_client()
        headers = dict(self.headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        await client.post(
            self.url, json={"jsonrpc": "2.0", "method": method, "params": params or {}}, headers=headers
        )

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()


def _decode_http_body(resp: Any) -> dict[str, Any] | None:
    """Decode a JSON or SSE Streamable-HTTP response body into one JSON-RPC message."""
    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        for raw in resp.text.splitlines():
            raw = raw.strip()
            if raw.startswith("data:"):
                payload = raw[len("data:") :].strip()
                if payload and payload != "[DONE]":
                    try:
                        return json.loads(payload)
                    except json.JSONDecodeError:
                        continue
        return None
    if not resp.text.strip():
        return None
    return resp.json()
