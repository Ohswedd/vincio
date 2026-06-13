"""A2A transports: in-process (offline tests) and HTTP (JSON-RPC + card fetch)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .protocol import AGENT_CARD_PATH, A2AError, jsonrpc_request

__all__ = ["A2ATransport", "InProcessA2ATransport", "HTTPA2ATransport"]


class A2ATransport(ABC):
    @abstractmethod
    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a JSON-RPC request and await its result."""

    @abstractmethod
    async def fetch_agent_card(self) -> dict[str, Any]:
        """Fetch the Agent Card document."""

    async def aclose(self) -> None:
        return None


class InProcessA2ATransport(A2ATransport):
    """Route directly to an in-process :class:`A2AServer`."""

    def __init__(self, server: Any, *, auth: str | None = None) -> None:
        self.server = server
        self.auth = auth
        self._counter = 0

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._counter += 1
        response = await self.server.handle(
            jsonrpc_request(self._counter, method, params), auth=self.auth
        )
        if response is None:
            return None
        if "error" in response:
            err = response["error"]
            raise A2AError(err["message"], code=err["code"], data=err.get("data"))
        return response.get("result")

    async def fetch_agent_card(self) -> dict[str, Any]:
        return self.server.agent_card()


class HTTPA2ATransport(A2ATransport):
    """Talk JSON-RPC over HTTP to a remote A2A agent."""

    def __init__(
        self,
        url: str,
        *,
        card_url: str | None = None,
        headers: dict[str, str] | None = None,
        client: Any | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.url = url
        self.card_url = card_url or _card_url_for(url)
        self.headers = {"Content-Type": "application/json"}
        if headers:
            self.headers.update(headers)
        self._client = client
        self._owns_client = client is None
        self.timeout_s = timeout_s
        self._counter = 0

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        import httpx

        client = self._ensure_client()
        self._counter += 1
        try:
            resp = await client.post(
                self.url, json=jsonrpc_request(self._counter, method, params), headers=self.headers
            )
        except httpx.HTTPError as exc:
            raise A2AError(f"transport error: {exc}") from exc
        if resp.status_code == 401:
            raise A2AError("unauthorized", code=-32001, data={"status": 401})
        if resp.status_code >= 400:
            raise A2AError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        message = resp.json()
        if "error" in message:
            err = message["error"]
            raise A2AError(err["message"], code=err["code"], data=err.get("data"))
        return message.get("result")

    async def fetch_agent_card(self) -> dict[str, Any]:
        client = self._ensure_client()
        resp = await client.get(self.card_url, headers={"Accept": "application/json"})
        if resp.status_code >= 400:
            raise A2AError(f"agent card fetch failed: HTTP {resp.status_code}")
        return resp.json()

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()


def _card_url_for(url: str) -> str:
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{AGENT_CARD_PATH}"
