"""A2A (Agent-to-Agent) wire types: Agent Card, Task lifecycle, messages.

A2A is JSON-RPC 2.0 over HTTP with an Agent Card discovery document at
``/.well-known/agent.json``. The Task lifecycle is
``submitted → working → input-required → completed/failed`` (also ``canceled`` /
``rejected``). This module is transport-agnostic and dependency-free.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import VincioError
from ..core.types import new_id

__all__ = [
    "A2A_PROTOCOL_VERSION",
    "AGENT_CARD_PATH",
    "A2AError",
    "AgentSkill",
    "AgentCard",
    "A2APart",
    "A2AMessage",
    "A2ATaskStatus",
    "A2AArtifact",
    "A2ATask",
    "TaskState",
    "text_message",
    "text_part",
    "jsonrpc_request",
    "jsonrpc_response",
    "jsonrpc_error",
    "static_token_validator",
    "UNAUTHORIZED",
]

# JSON-RPC code for an unauthorized request (maps to HTTP 401).
UNAUTHORIZED = -32001

A2A_PROTOCOL_VERSION = "0.3.0"
AGENT_CARD_PATH = "/.well-known/agent.json"

TaskState = Literal[
    "submitted", "working", "input-required", "completed", "canceled", "failed", "rejected"
]


class A2AError(VincioError):
    """An A2A protocol or transport error (``code`` is the JSON-RPC error code)."""

    def __init__(self, message: str, *, code: int = -32603, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class AgentSkill(BaseModel):
    """A capability advertised on the Agent Card."""

    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class AgentCard(BaseModel):
    """The discovery document served at ``/.well-known/agent.json``."""

    name: str
    description: str = ""
    url: str = ""
    version: str = "1.1.0"
    protocol_version: str = A2A_PROTOCOL_VERSION
    capabilities: dict[str, Any] = Field(default_factory=lambda: {"streaming": True})
    default_input_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    default_output_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    skills: list[AgentSkill] = Field(default_factory=list)
    security_schemes: dict[str, Any] = Field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        wire = {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities,
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "skills": [s.model_dump() for s in self.skills],
        }
        if self.security_schemes:
            wire["securitySchemes"] = self.security_schemes
        return wire


class A2APart(BaseModel):
    kind: Literal["text", "data"] = "text"
    text: str = ""
    data: dict[str, Any] | None = None


class A2AMessage(BaseModel):
    role: Literal["user", "agent"] = "user"
    parts: list[A2APart] = Field(default_factory=list)
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    task_id: str | None = None
    context_id: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.parts if p.kind == "text" and p.text)

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "role": self.role,
            "parts": [
                {"kind": "text", "text": p.text} if p.kind == "text" else {"kind": "data", "data": p.data}
                for p in self.parts
            ],
            "messageId": self.message_id,
            "kind": "message",
        }
        if self.task_id:
            wire["taskId"] = self.task_id
        if self.context_id:
            wire["contextId"] = self.context_id
        return wire

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> A2AMessage:
        parts = [
            A2APart(kind="text", text=p.get("text", ""))
            if p.get("kind", "text") == "text"
            else A2APart(kind="data", data=p.get("data"))
            for p in data.get("parts") or []
        ]
        return cls(
            role=data.get("role", "user"),
            parts=parts,
            message_id=data.get("messageId") or new_id("msg"),
            task_id=data.get("taskId"),
            context_id=data.get("contextId"),
        )


class A2ATaskStatus(BaseModel):
    state: TaskState = "submitted"
    message: A2AMessage | None = None


class A2AArtifact(BaseModel):
    artifact_id: str = Field(default_factory=lambda: new_id("artifact"))
    name: str = "result"
    parts: list[A2APart] = Field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        return {
            "artifactId": self.artifact_id,
            "name": self.name,
            "parts": [{"kind": "text", "text": p.text} for p in self.parts],
        }


class A2ATask(BaseModel):
    id: str = Field(default_factory=lambda: new_id("task"))
    context_id: str = Field(default_factory=lambda: new_id("ctx"))
    status: A2ATaskStatus = Field(default_factory=A2ATaskStatus)
    artifacts: list[A2AArtifact] = Field(default_factory=list)
    history: list[A2AMessage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        status: dict[str, Any] = {"state": self.status.state}
        if self.status.message is not None:
            status["message"] = self.status.message.to_wire()
        return {
            "id": self.id,
            "contextId": self.context_id,
            "kind": "task",
            "status": status,
            "artifacts": [a.to_wire() for a in self.artifacts],
            "history": [m.to_wire() for m in self.history],
            "metadata": self.metadata,
        }


def text_part(text: str) -> A2APart:
    return A2APart(kind="text", text=text)


def text_message(text: str, *, role: str = "user") -> A2AMessage:
    return A2AMessage(role=role, parts=[text_part(text)])  # type: ignore[arg-type]


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


def static_token_validator(tokens: set[str] | list[str]):
    """A resource-server bearer-token validator accepting a fixed allow-list.

    Suitable for tests and simple deployments (swap in JWT/introspection or mTLS
    for production). Raises :class:`A2AError` (code 401) on failure.
    """
    allow = set(tokens)

    def validate(auth: str | None) -> dict[str, Any]:
        if not auth:
            raise A2AError("missing bearer token", code=UNAUTHORIZED, data={"status": 401})
        token = auth.split(" ", 1)[1] if auth.lower().startswith("bearer ") else auth
        if token not in allow:
            raise A2AError("invalid bearer token", code=UNAUTHORIZED, data={"status": 401})
        return {"token": token[:6] + "…"}

    return validate
