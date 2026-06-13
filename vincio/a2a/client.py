"""A2A client + a remote-agent adapter usable as a local crew delegate."""

from __future__ import annotations

from typing import Any

from .protocol import (
    A2AError,
    A2AMessage,
    A2APart,
    A2ATask,
    A2ATaskStatus,
    AgentCard,
    AgentSkill,
    text_message,
)
from .transport import A2ATransport

__all__ = ["A2AClient", "RemoteA2AAgent"]


class A2AClient:
    """A connected A2A client bound to one remote agent transport."""

    def __init__(self, transport: A2ATransport) -> None:
        self.transport = transport
        self._card: AgentCard | None = None

    async def agent_card(self) -> AgentCard:
        if self._card is None:
            wire = await self.transport.fetch_agent_card()
            self._card = AgentCard(
                name=wire.get("name", "agent"),
                description=wire.get("description", ""),
                url=wire.get("url", ""),
                version=wire.get("version", "1.1.0"),
                protocol_version=wire.get("protocolVersion", "0.3.0"),
                capabilities=wire.get("capabilities", {}),
                default_input_modes=wire.get("defaultInputModes", ["text/plain"]),
                default_output_modes=wire.get("defaultOutputModes", ["text/plain"]),
                skills=[AgentSkill(**s) for s in wire.get("skills", [])],
                security_schemes=wire.get("securitySchemes", {}),
            )
        return self._card

    async def send(self, message: str | A2AMessage, *, task_id: str | None = None) -> A2ATask:
        """Send a message; returns the resulting :class:`A2ATask`."""
        msg = text_message(message) if isinstance(message, str) else message
        if task_id is not None:
            msg.task_id = task_id
        result = await self.transport.request("message/send", {"message": msg.to_wire()})
        return _task_from_wire(result or {})

    async def get_task(self, task_id: str) -> A2ATask:
        result = await self.transport.request("tasks/get", {"id": task_id})
        return _task_from_wire(result or {})

    async def cancel(self, task_id: str) -> A2ATask:
        result = await self.transport.request("tasks/cancel", {"id": task_id})
        return _task_from_wire(result or {})

    async def aclose(self) -> None:
        await self.transport.aclose()


def _task_from_wire(wire: dict[str, Any]) -> A2ATask:
    status_wire = wire.get("status") or {}
    status = A2ATaskStatus(
        state=status_wire.get("state", "submitted"),
        message=A2AMessage.from_wire(status_wire["message"]) if status_wire.get("message") else None,
    )
    artifacts = []
    for a in wire.get("artifacts") or []:
        artifacts.append(
            {
                "artifact_id": a.get("artifactId", ""),
                "name": a.get("name", "result"),
                "parts": [A2APart(kind="text", text=p.get("text", "")) for p in a.get("parts") or []],
            }
        )
    task = A2ATask(
        id=wire.get("id", ""),
        context_id=wire.get("contextId", ""),
        status=status,
        metadata=wire.get("metadata", {}),
    )
    from .protocol import A2AArtifact

    task.artifacts = [A2AArtifact(**a) for a in artifacts]
    return task


def _task_output(task: A2ATask) -> str:
    for artifact in task.artifacts:
        text = "\n".join(p.text for p in artifact.parts if p.kind == "text")
        if text:
            return text
    if task.status.message is not None:
        return task.status.message.text
    return ""


class RemoteA2AAgent:
    """A remote A2A agent that plugs into a local crew as a bounded delegate.

    Implements the :class:`~vincio.agents.executor.AgentExecutor` ``run``
    contract, so ``crew.add(role, RemoteA2AAgent(...))`` delegates a member's
    work to another vendor's agent over A2A — while the crew keeps its budget,
    termination, and tracing guarantees around the call.
    """

    def __init__(self, client: A2AClient, *, name: str = "remote") -> None:
        self.client = client
        self.name = name

    async def run(
        self,
        objective: Any,
        *,
        budget: Any | None = None,
        initial_evidence: Any | None = None,
    ) -> Any:
        from ..agents.state import AgentError, AgentState
        from ..core.types import Objective

        obj = objective if isinstance(objective, Objective) else Objective(text=str(objective))
        state = AgentState(objective=obj, budget=budget or _default_budget())
        if initial_evidence:
            state.evidence.extend(initial_evidence)
        try:
            task = await self.client.send(obj.text)
        except A2AError as exc:
            state.terminated = True
            state.termination_reason = "unrecoverable_error"
            state.errors.append(AgentError(message=f"A2A delegate failed: {exc}", recoverable=False))
            return state
        output = _task_output(task)
        state.final_answer = output
        state.raw_answer_text = output
        state.working_memory["a2a_task_id"] = task.id
        state.working_memory["a2a_task_state"] = task.status.state
        state.terminated = True
        if task.status.state == "completed":
            state.termination_reason = "objective_complete"
        elif task.status.state == "input-required":
            state.termination_reason = "approval_required"
        else:
            state.termination_reason = "unrecoverable_error"
            state.errors.append(
                AgentError(message=f"A2A delegate ended in {task.status.state}", recoverable=False)
            )
        return state


def _default_budget() -> Any:
    from ..core.types import Budget

    return Budget()
