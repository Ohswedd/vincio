"""AGNTCY / ACP (Agent Connect Protocol) adapter (2.2).

AGNTCY's Agent Connect Protocol is the **REST-native** interop camp (vs A2A's
JSON-RPC). This adapter lets Vincio span both: it models an ACP **agent
manifest**, maps it to/from the A2A :class:`~vincio.a2a.protocol.AgentCard` so one
:class:`~vincio.registry.directory.AgentDirectory` indexes agents from either
camp, and provides a discovery :class:`ACPClient` that reads an ACP agent
directory over REST (or an injected in-process catalog, for offline use).

The mapping is intentionally lossless on the discovery surface — name,
description, version, URL, and skills/capabilities — so an ACP-discovered agent
and an A2A-discovered agent are the same kind of :class:`AgentRecord` once in the
directory, and both pass the same allow-list gate.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..a2a.protocol import AgentCard, AgentSkill
from ..core.errors import VincioError
from ..stability import experimental

__all__ = ["ACPAgentManifest", "ACPClient", "acp_to_agent_card", "agent_card_to_acp"]


class ACPAgentManifest(BaseModel):
    """An AGNTCY/ACP agent record (the REST-native discovery document)."""

    id: str
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    url: str = ""
    capabilities: list[str] = Field(default_factory=list)
    skills: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def acp_to_agent_card(manifest: ACPAgentManifest) -> AgentCard:
    """Map an ACP manifest onto an A2A :class:`AgentCard`."""
    skills = [
        AgentSkill(
            id=str(s.get("id") or s.get("name") or f"skill_{i}"),
            name=str(s.get("name") or s.get("id") or f"skill_{i}"),
            description=str(s.get("description", "")),
            tags=list(s.get("tags", [])),
        )
        for i, s in enumerate(manifest.skills)
    ]
    # Capabilities with no skill object still become advertised skills, so
    # capability discovery works uniformly across camps.
    known = {s.id for s in skills}
    for cap in manifest.capabilities:
        if cap not in known:
            skills.append(AgentSkill(id=cap, name=cap, tags=[cap]))
    return AgentCard(
        name=manifest.name or manifest.id,
        description=manifest.description,
        url=manifest.url,
        version=manifest.version,
        skills=skills,
    )


def agent_card_to_acp(card: AgentCard, *, agent_id: str | None = None) -> ACPAgentManifest:
    """Map an A2A :class:`AgentCard` onto an ACP manifest."""
    return ACPAgentManifest(
        id=agent_id or card.name,
        name=card.name,
        description=card.description,
        version=card.version,
        url=card.url,
        capabilities=sorted({t for s in card.skills for t in s.tags}) or [s.id for s in card.skills],
        skills=[s.model_dump() for s in card.skills],
    )


@experimental(since="2.2")
class ACPClient:
    """Discover agents from an AGNTCY/ACP registry over REST (or a local catalog).

    Pass ``base_url`` to read a live ACP directory (``GET {base}/agents`` /
    ``GET {base}/agents/{id}``), or ``catalog`` (a list of manifests) to resolve
    fully offline — the same in-process/HTTP duality the MCP and A2A clients use.
    """

    def __init__(
        self,
        base_url: str = "",
        *,
        headers: dict[str, str] | None = None,
        http_client: Any | None = None,
        catalog: list[ACPAgentManifest] | None = None,
        agents_path: str = "/agents",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self._client = http_client
        self.catalog = list(catalog) if catalog is not None else None
        self.agents_path = agents_path

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(headers=self.headers, timeout=30.0)
        return self._client

    async def list_agents(self, *, query: str | None = None) -> list[ACPAgentManifest]:
        if self.catalog is not None:
            manifests = list(self.catalog)
        else:
            if not self.base_url:
                raise VincioError("ACPClient requires base_url or catalog")
            client = self._ensure_client()
            resp = await client.get(f"{self.base_url}{self.agents_path}", headers=self.headers)
            resp.raise_for_status()
            payload = resp.json()
            raw = payload.get("agents", payload) if isinstance(payload, dict) else payload
            manifests = [ACPAgentManifest.model_validate(item) for item in raw]
        if query:
            q = query.lower()
            manifests = [
                m
                for m in manifests
                if q in (m.name + " " + m.description + " " + " ".join(m.capabilities)).lower()
            ]
        return manifests

    async def get_agent(self, agent_id: str) -> ACPAgentManifest | None:
        if self.catalog is not None:
            return next((m for m in self.catalog if m.id == agent_id), None)
        client = self._ensure_client()
        resp = await client.get(f"{self.base_url}{self.agents_path}/{agent_id}", headers=self.headers)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return ACPAgentManifest.model_validate(resp.json())

    async def register_into_directory(self, directory: Any, *, query: str | None = None) -> list[str]:
        """Discover ACP agents and register them as ``AgentRecord``\\ s (protocol=acp)."""
        registered: list[str] = []
        for manifest in await self.list_agents(query=query):
            card = acp_to_agent_card(manifest)
            directory.register(card, url=manifest.url, protocol="acp")
            registered.append(card.name)
        return registered

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "aclose"):
            await self._client.aclose()
