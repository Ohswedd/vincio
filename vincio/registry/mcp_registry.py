"""MCP Registry discovery client (2.2).

Resolve MCP servers from the official **MCP Registry** so they can be discovered
and reached under the same :class:`~vincio.security.access.AllowListGate` as A2A
and ACP agents. Reads the registry over REST (``GET {base}/v0/servers``) or an
injected in-process catalog (offline). Discovered servers normalize to
:class:`~vincio.registry.directory.AgentRecord`\\ s with ``protocol="mcp"``, so the
directory governs MCP-server reachability exactly like agent reachability.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..a2a.protocol import AgentCard, AgentSkill
from ..core.errors import VincioError
from ..stability import experimental

__all__ = ["MCPServerRecord", "MCPRegistryClient"]

# The official registry; overridable for self-hosted/mirror deployments.
DEFAULT_MCP_REGISTRY = "https://registry.modelcontextprotocol.io"


class MCPServerRecord(BaseModel):
    """A normalized MCP server entry from the registry."""

    name: str
    description: str = ""
    version: str = ""
    url: str = ""  # remote (Streamable HTTP) endpoint, when published
    command: list[str] = Field(default_factory=list)  # stdio launch hint, when published
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_registry(cls, item: dict[str, Any]) -> MCPServerRecord:
        remotes = item.get("remotes") or []
        url = ""
        for remote in remotes:
            if isinstance(remote, dict) and remote.get("url"):
                url = str(remote["url"])
                break
        command: list[str] = []
        packages = item.get("packages") or []
        if packages and isinstance(packages[0], dict):
            pkg = packages[0]
            registry_name = pkg.get("registry_name") or pkg.get("registryType") or ""
            ident = pkg.get("name") or pkg.get("identifier") or ""
            if registry_name and ident:
                command = [str(registry_name), str(ident)]
        return cls(
            name=str(item.get("name", "")),
            description=str(item.get("description", "")),
            version=str((item.get("version_detail") or {}).get("version") or item.get("version") or ""),
            url=url,
            command=command,
            metadata={"registry_id": item.get("id")},
        )

    def to_agent_card(self) -> AgentCard:
        return AgentCard(
            name=self.name,
            description=self.description,
            url=self.url,
            version=self.version or "1.0.0",
            skills=[AgentSkill(id="mcp", name="mcp", description="MCP server", tags=["mcp"])],
        )


@experimental(since="2.2")
class MCPRegistryClient:
    """Discover MCP servers from the official registry (or a local catalog)."""

    def __init__(
        self,
        base_url: str = DEFAULT_MCP_REGISTRY,
        *,
        headers: dict[str, str] | None = None,
        http_client: Any | None = None,
        catalog: list[MCPServerRecord] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self._client = http_client
        self.catalog = list(catalog) if catalog is not None else None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(headers=self.headers, timeout=30.0)
        return self._client

    async def list_servers(self, *, query: str | None = None, limit: int = 100) -> list[MCPServerRecord]:
        if self.catalog is not None:
            servers = list(self.catalog)
        else:
            client = self._ensure_client()
            resp = await client.get(f"{self.base_url}/v0/servers", headers=self.headers)
            resp.raise_for_status()
            payload = resp.json()
            raw = payload.get("servers", payload) if isinstance(payload, dict) else payload
            servers = [MCPServerRecord.from_registry(item) for item in raw]
        if query:
            q = query.lower()
            servers = [s for s in servers if q in (s.name + " " + s.description).lower()]
        return servers[:limit]

    async def get_server(self, name: str) -> MCPServerRecord | None:
        for server in await self.list_servers(limit=10_000):
            if server.name == name:
                return server
        return None

    async def register_into_directory(self, directory: Any, *, query: str | None = None) -> list[str]:
        """Discover MCP servers and register them as ``AgentRecord``\\ s (protocol=mcp)."""
        registered: list[str] = []
        for server in await self.list_servers(query=query):
            if not server.name:
                continue
            directory.register(server.to_agent_card(), url=server.url, protocol="mcp")
            registered.append(server.name)
        return registered

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "aclose"):
            await self._client.aclose()

    @staticmethod
    def _require_httpx() -> None:  # pragma: no cover - import guard
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise VincioError("MCPRegistryClient HTTP mode requires httpx") from exc
