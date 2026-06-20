"""Vincio agent registry & discovery — the governed agent fabric.

Point-to-point A2A/MCP delegation becomes a *discoverable, governed* fabric: an
:class:`AgentDirectory` indexes agents by capability over the existing A2A
**Agent Card**, every resolution passes an :class:`~vincio.security.access.AllowListGate`
and is recorded as an access decision on the hash-chained audit log, and Vincio
spans both interop camps — A2A and **AGNTCY / ACP** (REST-native Agent Connect
Protocol) — plus the official **MCP Registry** for server discovery.

    directory = AgentDirectory(allow_list=AllowListGate(allow=["researcher"]), audit=app.audit)
    directory.register(card, url="https://researcher.example")
    record = directory.resolve("researcher")          # governed + audited

    # discover from an AGNTCY/ACP registry or the MCP registry, under the gate
    await ACPClient(catalog=manifests).register_into_directory(directory)
    await MCPRegistryClient(catalog=servers).register_into_directory(directory)

A :class:`CommunityRegistry` applies the same governance to opt-in domain
**packs** and ``SKILL.md`` **skill** bundles: a signed, content-bound index whose
resolutions pass the same allow-list gate and land on the audit chain.
"""

from __future__ import annotations

from .acp import ACPAgentManifest, ACPClient, acp_to_agent_card, agent_card_to_acp
from .community import BundleKind, BundleRecord, BundleResolution, CommunityRegistry
from .directory import AgentDirectory, AgentRecord, AgentResolution
from .mcp_registry import MCPRegistryClient, MCPServerRecord

__all__ = [
    "AgentDirectory",
    "AgentRecord",
    "AgentResolution",
    "ACPClient",
    "ACPAgentManifest",
    "acp_to_agent_card",
    "agent_card_to_acp",
    "MCPRegistryClient",
    "MCPServerRecord",
    "CommunityRegistry",
    "BundleRecord",
    "BundleResolution",
    "BundleKind",
]
