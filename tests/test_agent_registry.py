"""Tests for the agent registry / discovery fabric + allow-list."""

from __future__ import annotations

import pytest

from vincio.a2a.protocol import AgentCard, AgentSkill
from vincio.core.errors import AccessDeniedError
from vincio.registry import (
    ACPAgentManifest,
    ACPClient,
    AgentDirectory,
    MCPRegistryClient,
    MCPServerRecord,
    acp_to_agent_card,
    agent_card_to_acp,
)
from vincio.registry.mcp_registry import DEFAULT_MCP_REGISTRY
from vincio.security.access import AllowListGate
from vincio.security.audit import AuditLog


def _card(name: str, *, tags: list[str]) -> AgentCard:
    return AgentCard(
        name=name,
        description=f"{name} agent",
        url=f"https://{name}.example",
        skills=[AgentSkill(id=f"{name}-skill", name=name, description="does work", tags=tags)],
    )


def test_allow_list_gate_fails_closed():
    gate = AllowListGate(allow=["researcher", "*.trusted.example"])
    assert gate.allows("researcher") is True
    assert gate.allows("api.trusted.example") is True
    assert gate.allows("malicious") is False  # not allow-listed → denied


def test_allow_list_deny_takes_precedence():
    gate = AllowListGate(allow=["*"], deny=["evil*"])
    assert gate.allows("good-agent") is True
    assert gate.allows("evil-agent") is False


def test_directory_find_by_capability_and_tag():
    directory = AgentDirectory()
    directory.register(_card("researcher", tags=["research", "web"]))
    directory.register(_card("coder", tags=["code", "python"]))
    assert [r.name for r in directory.find(tag="research")] == ["researcher"]
    assert [r.name for r in directory.find(query="python")] == ["coder"]
    assert {r.name for r in directory.find()} == {"researcher", "coder"}


def test_resolution_is_governed_and_audited():
    audit = AuditLog(directory=None)
    gate = AllowListGate(allow=["researcher"])
    directory = AgentDirectory(allow_list=gate, audit=audit)
    directory.register(_card("researcher", tags=["research"]))
    directory.register(_card("coder", tags=["code"]))

    record = directory.resolve("researcher")
    assert record.name == "researcher"

    # A non-allow-listed agent is denied — even though it is in the directory.
    with pytest.raises(AccessDeniedError):
        directory.resolve("coder")

    decisions = audit.query(action="agent_resolve")
    by_resource = {e.resource: e.decision for e in decisions}
    assert by_resource["researcher"] == "allow"
    assert by_resource["coder"] == "deny"


def test_resolve_unknown_agent_denied():
    directory = AgentDirectory(allow_list=AllowListGate(allow=["*"]))
    res = directory.try_resolve("ghost")
    assert res.allowed is False
    assert "not in directory" in res.decision.reason


def test_no_gate_resolves_but_unknown_still_fails():
    directory = AgentDirectory()
    directory.register(_card("a", tags=["x"]))
    assert directory.resolve("a").name == "a"
    with pytest.raises(AccessDeniedError):
        directory.resolve("missing")


def test_acp_roundtrip_mapping():
    manifest = ACPAgentManifest(
        id="planner-1",
        name="planner",
        description="plans tasks",
        url="https://planner.example",
        capabilities=["planning", "decomposition"],
        skills=[{"id": "plan", "name": "plan", "tags": ["planning"]}],
    )
    card = acp_to_agent_card(manifest)
    assert card.name == "planner"
    assert "planning" in {t for s in card.skills for t in s.tags}
    back = agent_card_to_acp(card, agent_id="planner-1")
    assert back.id == "planner-1"
    assert "planning" in back.capabilities


@pytest.mark.asyncio
async def test_acp_client_offline_catalog_into_directory():
    catalog = [
        ACPAgentManifest(id="a1", name="acp-agent", capabilities=["summarize"], url="https://a1.example"),
    ]
    directory = AgentDirectory(allow_list=AllowListGate(allow=["acp-agent"]))
    names = await ACPClient(catalog=catalog).register_into_directory(directory)
    assert names == ["acp-agent"]
    record = directory.resolve("acp-agent")
    assert record.protocol == "acp"
    assert "summarize" in record.capabilities


@pytest.mark.asyncio
async def test_mcp_registry_offline_catalog_into_directory():
    catalog = [
        MCPServerRecord(name="filesystem", description="fs server", url="https://fs.example/mcp"),
        MCPServerRecord(name="github", description="gh server", command=["npx", "gh-mcp"]),
    ]
    directory = AgentDirectory(allow_list=AllowListGate(allow=["filesystem"]))
    names = await MCPRegistryClient(catalog=catalog).register_into_directory(directory)
    assert set(names) == {"filesystem", "github"}
    assert directory.resolve("filesystem").protocol == "mcp"
    with pytest.raises(AccessDeniedError):
        directory.resolve("github")  # discovered but not allow-listed


def test_mcp_registry_record_from_registry_payload():
    item = {
        "id": "io.example/fs",
        "name": "filesystem",
        "description": "file ops",
        "version_detail": {"version": "1.2.3"},
        "remotes": [{"url": "https://fs.example/mcp"}],
        "packages": [{"registry_name": "npm", "name": "@example/fs"}],
    }
    record = MCPServerRecord.from_registry(item)
    assert record.name == "filesystem"
    assert record.version == "1.2.3"
    assert record.url == "https://fs.example/mcp"
    assert record.command == ["npm", "@example/fs"]


def test_default_registry_constant():
    assert DEFAULT_MCP_REGISTRY.startswith("https://")


def test_app_facade_builds_governed_directory():
    from vincio import ContextApp
    from vincio.providers.mock import MockProvider

    app = ContextApp(name="fabric", provider=MockProvider(), model="mock-1")
    directory = app.agent_directory(allow=["researcher"])
    directory.register(_card("researcher", tags=["research"]))
    assert directory.resolve("researcher").name == "researcher"
    # Resolution recorded on the app's audit chain.
    assert any(e.action == "agent_resolve" for e in app.audit.query(action="agent_resolve"))
