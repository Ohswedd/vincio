"""Vincio A2A (Agent-to-Agent) support — server *and* client.

Expose a Vincio crew, graph, or app as an A2A agent (Agent Card +
JSON-RPC task lifecycle), and reach remote A2A agents — including as bounded,
traced delegates inside a local crew. Works in-process (offline tests) or over
HTTP.

    server = crew_a2a_server(crew, name="researcher")
    client = connect_a2a_in_process(server)
    task = await client.send("Summarize Q3 refunds")

    crew.add(role, RemoteA2AAgent(client))   # remote agent as a local delegate
"""

from __future__ import annotations

from typing import Any

from .client import A2AClient, RemoteA2AAgent
from .protocol import (
    A2A_PROTOCOL_VERSION,
    AGENT_CARD_PATH,
    A2AArtifact,
    A2AError,
    A2AMessage,
    A2APart,
    A2ATask,
    A2ATaskStatus,
    AgentCard,
    AgentSkill,
    static_token_validator,
    text_message,
)
from .server import A2AServer, app_a2a_server, crew_a2a_server, graph_a2a_server
from .transport import A2ATransport, HTTPA2ATransport, InProcessA2ATransport

__all__ = [
    "A2AServer",
    "A2AClient",
    "RemoteA2AAgent",
    "AgentCard",
    "AgentSkill",
    "A2AMessage",
    "A2APart",
    "A2AArtifact",
    "A2ATask",
    "A2ATaskStatus",
    "A2AError",
    "A2ATransport",
    "InProcessA2ATransport",
    "HTTPA2ATransport",
    "crew_a2a_server",
    "graph_a2a_server",
    "app_a2a_server",
    "text_message",
    "static_token_validator",
    "A2A_PROTOCOL_VERSION",
    "AGENT_CARD_PATH",
    "connect_a2a",
    "connect_a2a_in_process",
]


def connect_a2a(
    url: str, *, headers: dict[str, str] | None = None, http_client: Any | None = None
) -> A2AClient:
    """Connect to a remote A2A agent over HTTP."""
    return A2AClient(HTTPA2ATransport(url, headers=headers, client=http_client))


def connect_a2a_in_process(server: A2AServer, *, auth: str | None = None) -> A2AClient:
    """Connect to an in-process :class:`A2AServer` (offline tests, local use)."""
    return A2AClient(InProcessA2ATransport(server, auth=auth))
