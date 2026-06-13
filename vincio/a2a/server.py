"""A2A server: task lifecycle over JSON-RPC, plus crew/graph/app builders.

:class:`A2AServer` is transport-agnostic — it consumes a JSON-RPC message and
produces a response. The executor it wraps is bounded and traced: crews enforce
per-member budgets and termination guarantees, graphs checkpoint and surface
human-in-the-loop interrupts as the ``input-required`` task state — guarantees a
raw A2A SDK does not provide.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from .protocol import (
    A2AArtifact,
    A2AError,
    A2AMessage,
    A2ATask,
    A2ATaskStatus,
    AgentCard,
    AgentSkill,
    jsonrpc_error,
    jsonrpc_response,
    text_message,
    text_part,
)

__all__ = [
    "A2AServer",
    "crew_a2a_server",
    "graph_a2a_server",
    "app_a2a_server",
]

# An executor maps (input text, task) -> a result dict:
#   {"state": "completed", "output": "..."}
#   {"state": "input-required", "prompt": "..."}
#   {"state": "failed", "error": "..."}
Executor = Callable[[str, A2ATask], Awaitable[dict[str, Any]]]
TokenValidator = Callable[[str | None], Awaitable[Any] | Any]

INVALID_PARAMS = -32602
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603
UNAUTHORIZED = -32001


class A2AServer:
    """Serves one agent (a crew, graph, or app) over the A2A task protocol."""

    def __init__(
        self,
        card: AgentCard,
        executor: Executor,
        *,
        tracer: Any | None = None,
        token_validator: TokenValidator | None = None,
        audit: Any | None = None,
    ) -> None:
        self.card = card
        self.executor = executor
        self.tracer = tracer
        self.token_validator = token_validator
        self.audit = audit
        self.tasks: dict[str, A2ATask] = {}

    def agent_card(self) -> dict[str, Any]:
        return self.card.to_wire()

    async def _validate(self, auth: str | None) -> None:
        if self.token_validator is None:
            return
        result = self.token_validator(auth)
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]

    async def handle(self, message: dict[str, Any], *, auth: str | None = None) -> dict[str, Any] | None:
        if message.get("jsonrpc") != "2.0":
            return jsonrpc_error(message.get("id"), -32600, "not a JSON-RPC 2.0 message")
        msg_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}
        if msg_id is None:
            return None
        try:
            await self._validate(auth)
        except Exception as exc:
            # Any validator (A2A, MCP, or custom) maps to a clean 401-style error.
            code = getattr(exc, "code", UNAUTHORIZED)
            data = getattr(exc, "data", None) or {"status": 401}
            return jsonrpc_error(msg_id, code, str(exc), data)
        try:
            if method == "message/send":
                result = (await self._message_send(params)).to_wire()
            elif method == "tasks/get":
                result = self._get_task(params).to_wire()
            elif method == "tasks/cancel":
                result = self._cancel_task(params).to_wire()
            else:
                return jsonrpc_error(msg_id, METHOD_NOT_FOUND, f"unknown method {method!r}")
        except A2AError as exc:
            return jsonrpc_error(msg_id, exc.code, exc.message, exc.data)
        except Exception as exc:  # pragma: no cover - defensive
            return jsonrpc_error(msg_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        return jsonrpc_response(msg_id, result)

    def _get_task(self, params: dict[str, Any]) -> A2ATask:
        task_id = params.get("id")
        task = self.tasks.get(task_id)
        if task is None:
            raise A2AError(f"unknown task {task_id!r}", code=INVALID_PARAMS)
        return task

    def _cancel_task(self, params: dict[str, Any]) -> A2ATask:
        task = self._get_task(params)
        task.status = A2ATaskStatus(state="canceled")
        return task

    async def _message_send(self, params: dict[str, Any]) -> A2ATask:
        message = A2AMessage.from_wire(params.get("message") or {})
        if not message.parts:
            raise A2AError("message/send requires a message with parts", code=INVALID_PARAMS)
        task = self.tasks.get(message.task_id) if message.task_id else None
        if task is None:
            task = A2ATask()
            self.tasks[task.id] = task
        task.history.append(message)
        task.status = A2ATaskStatus(state="working")
        if self.audit is not None:
            self.audit.record(
                "a2a_serve",
                resource=self.card.name,
                decision="working",
                details={"task_id": task.id, "direction": "inbound"},
            )
        result = await self._run(message.text, task)
        self._apply_result(task, result)
        return task

    async def _run(self, text: str, task: A2ATask) -> dict[str, Any]:
        if self.tracer is not None:
            with self.tracer.span("a2a_task", type="custom") as span:
                span.set(agent=self.card.name, task_id=task.id)
                result = await self.executor(text, task)
                span.set(state=result.get("state", "completed"))
                return result
        return await self.executor(text, task)

    def _apply_result(self, task: A2ATask, result: dict[str, Any]) -> None:
        state = result.get("state", "completed")
        if state == "input-required":
            task.status = A2ATaskStatus(
                state="input-required",
                message=text_message(str(result.get("prompt", "")), role="agent"),
            )
        elif state == "failed":
            task.status = A2ATaskStatus(
                state="failed", message=text_message(str(result.get("error", "")), role="agent")
            )
        else:
            output = result.get("output", "")
            text = output if isinstance(output, str) else _stringify(output)
            task.artifacts.append(A2AArtifact(parts=[text_part(text)]))
            task.status = A2ATaskStatus(state="completed", message=text_message(text, role="agent"))

    async def stream_send(self, params: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE-style status-update events for ``message/stream``."""
        message = A2AMessage.from_wire(params.get("message") or {})
        task = A2ATask()
        self.tasks[task.id] = task
        task.history.append(message)
        yield {"kind": "status-update", "taskId": task.id, "status": {"state": "working"}, "final": False}
        result = await self._run(message.text, task)
        self._apply_result(task, result)
        yield {
            "kind": "status-update",
            "taskId": task.id,
            "status": task.status.model_dump(mode="json"),
            "final": True,
        }


def _stringify(value: Any) -> str:
    import json

    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):  # pragma: no cover
        return str(value)


# -- builders -----------------------------------------------------------------


def _crew_skills(crew: Any) -> list[AgentSkill]:
    skills: list[AgentSkill] = []
    members = getattr(crew, "_members", {})
    for name, member in members.items():
        role = getattr(member, "role", None)
        skills.append(
            AgentSkill(
                id=name,
                name=name,
                description=getattr(role, "goal", "") or getattr(role, "description", ""),
                tags=list(getattr(role, "keywords", []) or []),
            )
        )
    return skills


def crew_a2a_server(
    crew: Any,
    *,
    name: str | None = None,
    url: str = "",
    description: str = "",
    tracer: Any | None = None,
    token_validator: TokenValidator | None = None,
    audit: Any | None = None,
    budget: Any | None = None,
) -> A2AServer:
    """Expose a :class:`Crew` over A2A — bounded and traced by construction."""
    card = AgentCard(
        name=name or getattr(crew, "name", "crew"),
        description=description or "A Vincio crew exposed over A2A.",
        url=url,
        skills=_crew_skills(crew),
    )

    async def executor(text: str, task: A2ATask) -> dict[str, Any]:
        result = await crew.arun(text, budget=budget)
        if result.status == "failed":
            return {"state": "failed", "error": "crew failed"}
        output = result.output
        if output is None and result.reports:
            output = result.reports[-1].output
        return {"state": "completed", "output": output if output is not None else ""}

    return A2AServer(card, executor, tracer=tracer or getattr(crew, "tracer", None), token_validator=token_validator, audit=audit)


def graph_a2a_server(
    compiled_graph: Any,
    *,
    name: str,
    url: str = "",
    description: str = "",
    input_key: str = "input",
    output_key: str = "output",
    tracer: Any | None = None,
    token_validator: TokenValidator | None = None,
    audit: Any | None = None,
) -> A2AServer:
    """Expose a compiled :class:`StateGraph` over A2A.

    Graph human-in-the-loop interrupts surface as the ``input-required`` task
    state; a follow-up ``message/send`` carrying the same ``taskId`` resumes the
    checkpointed thread with the provided answer.
    """
    card = AgentCard(
        name=name,
        description=description or "A Vincio durable state graph exposed over A2A.",
        url=url,
    )

    async def executor(text: str, task: A2ATask) -> dict[str, Any]:
        thread_id = task.metadata.get("thread_id")
        if thread_id is not None:
            # Resume an interrupted thread with the caller's answer.
            graph_result = await compiled_graph.aresume(thread_id, value=text)
        else:
            graph_result = await compiled_graph.ainvoke({input_key: text})
            task.metadata["thread_id"] = graph_result.thread_id
        if graph_result.status == "interrupted":
            return {"state": "input-required", "prompt": _stringify(graph_result.interrupt_payload)}
        if graph_result.status == "max_steps":
            return {"state": "failed", "error": "graph exceeded max_steps"}
        return {"state": "completed", "output": graph_result.state.get(output_key, graph_result.state)}

    return A2AServer(card, executor, tracer=tracer, token_validator=token_validator, audit=audit)


def app_a2a_server(
    app: Any,
    *,
    name: str | None = None,
    url: str = "",
    description: str = "",
    token_validator: TokenValidator | None = None,
) -> A2AServer:
    """Expose a :class:`ContextApp` itself (``app.arun``) over A2A."""
    card = AgentCard(
        name=name or app.name,
        description=description or (app.objective.text if app.objective else "A Vincio app over A2A."),
        url=url,
    )

    async def executor(text: str, task: A2ATask) -> dict[str, Any]:
        result = await app.arun(text)
        return {"state": "completed", "output": result.output}

    return A2AServer(card, executor, tracer=app.tracer, token_validator=token_validator, audit=app.audit)
