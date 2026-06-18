"""Agent executor: bounded DAG execution and ReAct loop.

The executor never loops uncontrolled: every model/tool call charges the
budget, and termination conditions are checked before each step.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import ToolApprovalRequiredError, VincioError
from ..core.types import (
    Budget,
    EvidenceItem,
    Message,
    ModelRequest,
    Objective,
    ToolCall,
    ToolSpec,
)
from ..observability.costs import CostTracker
from ..observability.finops import CostLedger
from ..observability.traces import Tracer
from ..output.validators import OutputValidator
from ..providers.base import ModelProvider
from ..security.access import Principal
from ..tools.runtime import ToolRuntime
from .planner import Planner
from .state import AgentError, AgentState, AgentStep

__all__ = ["AgentExecutor", "AgentEvent"]

RetrieveFn = Callable[[str], Awaitable[list[EvidenceItem]]]
HumanGate = Callable[[str, dict[str, Any]], Awaitable[bool]]


class AgentEvent(BaseModel):
    """A streamed event from :meth:`AgentExecutor.astream`.

    Mirrors the flat, ``type``-discriminated event surface ``graph`` and
    ``compose`` already expose: step lifecycle, real text deltas, tool activity,
    and a terminal ``done`` carrying the final :class:`AgentState`.
    """

    type: Literal[
        "run_start", "step_start", "step_end", "text_delta", "tool_call", "tool_result", "done", "error"
    ]
    step: str = ""
    step_type: str = ""
    instruction: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    text: str | None = None
    status: str = ""
    result: Any = None
    error: str | None = None
    usage: dict[str, float] = Field(default_factory=dict)
    payload: Any = None  # final AgentState (model_dump) on "done"

_TOOL_ARGS_SCHEMA = {
    "type": "object",
    "properties": {
        "calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object", "additionalProperties": True},
                },
                "required": ["tool_name", "arguments"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["calls"],
    "additionalProperties": False,
}

_VALIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "revised_answer": {"type": "string"},
    },
    "required": ["passed", "issues", "revised_answer"],
    "additionalProperties": False,
}


class AgentExecutor:
    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str,
        planner: Planner | None = None,
        tool_runtime: ToolRuntime | None = None,
        tool_specs: list[ToolSpec] | None = None,
        retrieve_fn: RetrieveFn | None = None,
        output_validator: OutputValidator | None = None,
        principal: Principal | None = None,
        tracer: Tracer | None = None,
        cost_tracker: CostTracker | None = None,
        cost_ledger: CostLedger | None = None,
        human_gate: HumanGate | None = None,
        system_prompt: str = "",
        max_validation_rounds: int = 2,
        max_context_tokens: int = 6000,
        max_parallel_steps: int = 8,
        max_replans: int = 2,
    ) -> None:
        self.provider = provider
        self.model = model
        self.planner = planner or Planner(mode="static")
        self.tools = tool_runtime
        self.tool_specs = tool_specs or []
        self.retrieve_fn = retrieve_fn
        self.output_validator = output_validator
        self.principal = principal or Principal()
        self.tracer = tracer or Tracer()
        self.costs = cost_tracker or CostTracker()
        # Cost attribution: when an app wires its ledger here, every agent
        # model call is attributed by tenant/user/feature/run, exactly like a
        # ContextApp run. ``attribution`` is set per run by the caller.
        self.cost_ledger = cost_ledger
        self.attribution: dict[str, Any] = {}
        self.human_gate = human_gate
        self.system_prompt = system_prompt
        self.max_validation_rounds = max_validation_rounds
        # In-loop compaction: old tool/observation turns are folded into a
        # rolling summary once the working context exceeds this token budget,
        # replacing fixed slicing. Level-parallelism and plan-and-execute bounds.
        from .compaction import ContextCompactor

        self.compactor = ContextCompactor(max_tokens=max_context_tokens)
        self.max_context_tokens = max_context_tokens
        self.max_parallel_steps = max(1, max_parallel_steps)
        self.max_replans = max_replans
        # Streaming sink: set by ``astream`` so the step/tool/text handlers
        # emit events live. ``None`` on the default ``run`` path — zero overhead.
        self._event_sink: Callable[[AgentEvent], None] | None = None

    # -- streaming emission ----------------------------------------------------

    def _emit(self, event: AgentEvent) -> None:
        if self._event_sink is not None:
            self._event_sink(event)

    # -- budget / termination -------------------------------------------------------

    def _check_termination(self, state: AgentState) -> bool:
        breaches = state.usage.exceeds(state.budget)
        if breaches:
            state.terminated = True
            state.termination_reason = "budget_exhausted"
            state.errors.append(AgentError(message=f"budget exhausted: {breaches}", recoverable=False))
            return True
        if state.usage.steps >= state.budget.max_steps:
            state.terminated = True
            state.termination_reason = "max_steps"
            return True
        return False

    async def _stream_model(self, request: ModelRequest) -> Any:
        """Drive ``provider.stream`` and emit **genuine provider token deltas** as
        they arrive. The final :class:`ModelResponse` is taken from the
        stream's terminal ``done`` event (a thin generate fallback if a provider
        omits it), so the call returns the identical result the non-streaming
        path would — only the text now arrives live, token by token."""
        response: Any = None
        async for event in self.provider.stream(request):
            if event.type == "text_delta" and event.text:
                self._emit(AgentEvent(type="text_delta", text=event.text, step="answer"))
            elif event.type == "done" and event.response is not None:
                response = event.response
        if response is None:  # pragma: no cover - provider without a done.response
            response = await self.provider.generate(request)
        return response

    async def _model_call(
        self, state: AgentState, messages: list[Message], *, stream_text: bool = False, **kwargs: Any
    ):
        request = ModelRequest(model=self.model, messages=messages, **kwargs)
        # When streaming a run, route answer-producing free-text calls through the
        # provider's real token stream; structured (schema) calls stay on generate.
        if stream_text and self._event_sink is not None and "output_schema" not in kwargs:
            response = await self._stream_model(request)
        else:
            response = await self.provider.generate(request)
        cost = self.costs.record_model_call(self.model, response.usage)
        spent = cost if cost else response.cost_usd
        state.usage.input_tokens += response.usage.input_tokens
        state.usage.output_tokens += response.usage.output_tokens
        state.usage.cost_usd += spent
        state.usage.latency_ms += response.latency_ms
        if self.cost_ledger is not None:
            self.cost_ledger.record_model_call(
                model=self.model,
                usage=response.usage,
                cost_usd=spent,
                provider=response.provider or "",
                tenant_id=self.attribution.get("tenant_id"),
                user_id=self.attribution.get("user_id"),
                feature=self.attribution.get("feature"),
                run_id=self.attribution.get("run_id"),
            )
        return response

    # -- context rendering ----------------------------------------------------------------

    def _context_messages(self, state: AgentState, instruction: str) -> list[Message]:
        parts: list[str] = [f"Objective: {state.objective.text}"]
        # In-loop compaction: keep recent evidence/observations verbatim within a
        # token budget and fold older ones into a rolling summary, instead
        # of an arbitrary fixed slice. Split the budget across the two sections.
        section_budget = max(512, self.max_context_tokens // 2)
        if state.evidence:
            blocks = [f"[{e.citation_ref}] {e.text}" for e in state.evidence if e.text]
            summary, kept = self.compactor.compact_blocks(blocks, budget=section_budget)
            evidence_text = ""
            if summary:
                evidence_text += f"(earlier evidence summarized) {summary}\n"
            evidence_text += "\n".join(kept)
            parts.append(f"Evidence:\n{evidence_text}")
        if state.tool_results:
            blocks = [
                f"{r.tool_name} ({r.status}): {json.dumps(r.output, default=str)[:800]}"
                for r in state.tool_results
            ]
            summary, kept = self.compactor.compact_blocks(blocks, budget=section_budget)
            tool_text = ""
            if summary:
                tool_text += f"(earlier tool results summarized) {summary}\n"
            tool_text += "\n".join(kept)
            parts.append(f"Tool results:\n{tool_text}")
        if state.working_memory:
            memory_lines = "\n".join(f"{k}: {str(v)[:400]}" for k, v in state.working_memory.items())
            parts.append(f"Working memory:\n{memory_lines}")
        parts.append(f"Current step: {instruction}")
        messages = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt, cache_hint=True))
        messages.append(Message(role="user", content="\n\n".join(parts)))
        return messages

    # -- step handlers ------------------------------------------------------------------------

    async def _run_step(self, state: AgentState, step: AgentStep) -> None:
        step.status = "running"
        step.attempts += 1
        started = time.monotonic()
        self._emit(
            AgentEvent(
                type="step_start",
                step=step.name or step.type,
                step_type=step.type,
                instruction=step.instruction[:300],
            )
        )
        try:
            handler = {
                "retrieve": self._step_retrieve,
                "think": self._step_think,
                "tool": self._step_tool,
                "validate": self._step_validate,
                "ask_human": self._step_ask_human,
                "finalize": self._step_finalize,
            }[step.type]
            with self.tracer.span(step.name or step.type, type="agent_step") as span:
                span.set(step_type=step.type, instruction=step.instruction[:300])
                await handler(state, step)
                span.set(status=step.status)
            if step.status == "running":
                step.status = "done"
        except ToolApprovalRequiredError as exc:
            step.status = "failed"
            step.error = str(exc)
            state.terminated = True
            state.termination_reason = "approval_required"
            state.errors.append(AgentError(step_id=step.id, message=str(exc), recoverable=False))
        except VincioError as exc:
            step.status = "failed"
            step.error = exc.message
            state.errors.append(AgentError(step_id=step.id, message=exc.message))
        except Exception as exc:  # noqa: BLE001
            step.status = "failed"
            step.error = f"{type(exc).__name__}: {exc}"
            state.errors.append(AgentError(step_id=step.id, message=step.error))
        finally:
            step.duration_ms = int((time.monotonic() - started) * 1000)
            state.usage.steps += 1
            self._emit(
                AgentEvent(
                    type="step_end",
                    step=step.name or step.type,
                    step_type=step.type,
                    status=step.status,
                    error=step.error,
                )
            )

    async def _step_retrieve(self, state: AgentState, step: AgentStep) -> None:
        if self.retrieve_fn is None:
            step.status = "skipped"
            step.error = "no retrieval configured"
            return
        query = step.instruction or state.objective.text
        evidence = await self.retrieve_fn(query)
        existing_ids = {e.id for e in state.evidence}
        new_items = [e for e in evidence if e.id not in existing_ids]
        state.evidence.extend(new_items)
        step.result = [e.id for e in new_items]

    async def _step_think(self, state: AgentState, step: AgentStep) -> None:
        if step.name == "plan_tools" and self.tools is not None and self.tool_specs:
            await self._plan_and_call_tools(state, step)
            return
        response = await self._model_call(
            state, self._context_messages(state, step.instruction), stream_text=True
        )
        step.result = response.text
        state.working_memory[step.name or step.id] = response.text

    async def _plan_and_call_tools(self, state: AgentState, step: AgentStep) -> None:
        tool_lines = "\n".join(
            f"- {t.name}: {t.description} | input schema: {json.dumps(t.input_schema)[:400]}"
            for t in self.tool_specs
        )
        messages = self._context_messages(
            state,
            f"{step.instruction}\nAvailable tools:\n{tool_lines}\n"
            "Output the tool calls needed (empty list if none).",
        )
        response = await self._model_call(
            state, messages, output_schema=_TOOL_ARGS_SCHEMA, output_schema_name="tool_calls"
        )
        payload = response.structured or {}
        calls = payload.get("calls", [])
        results = []
        known_tools = {t.name for t in self.tool_specs}
        for call_spec in calls[: state.budget.max_tool_calls]:
            if state.usage.tool_calls >= state.budget.max_tool_calls:
                break
            tool_name = call_spec.get("tool_name", "")
            if tool_name not in known_tools:
                results.append({"tool": tool_name, "status": "error", "error": "unknown tool"})
                continue
            self._emit(
                AgentEvent(type="tool_call", tool_name=tool_name, arguments=dict(call_spec.get("arguments", {})))
            )
            try:
                result = await self.tools.execute(  # type: ignore[union-attr]
                    ToolCall(tool_name=tool_name, arguments=call_spec.get("arguments", {})),
                    principal=self.principal,
                )
            except ToolApprovalRequiredError:
                raise
            except VincioError as exc:
                from ..core.types import ToolResult

                result = ToolResult(call_id="", tool_name=tool_name, status="error", error=exc.message)
            state.usage.tool_calls += 1
            state.tool_results.append(result)
            self._emit(
                AgentEvent(
                    type="tool_result",
                    tool_name=result.tool_name,
                    status=result.status,
                    result=result.output if result.status == "ok" else result.error,
                )
            )
            results.append({"tool": result.tool_name, "status": result.status})
        step.result = results

    async def _step_tool(self, state: AgentState, step: AgentStep) -> None:
        if self.tools is None or not step.tool_name:
            step.status = "skipped"
            step.error = "no tool runtime or tool name"
            return
        arguments = step.tool_arguments
        if not arguments:
            spec = next((t for t in self.tool_specs if t.name == step.tool_name), None)
            schema_hint = json.dumps(spec.input_schema)[:600] if spec else "{}"
            response = await self._model_call(
                state,
                self._context_messages(
                    state,
                    f"Produce arguments for tool {step.tool_name!r} to: {step.instruction}\n"
                    f"Input schema: {schema_hint}",
                ),
                output_schema={
                    "type": "object",
                    "properties": {"arguments": {"type": "object", "additionalProperties": True}},
                    "required": ["arguments"],
                    "additionalProperties": False,
                },
                output_schema_name="tool_arguments",
            )
            arguments = (response.structured or {}).get("arguments", {})
        self._emit(AgentEvent(type="tool_call", tool_name=step.tool_name, arguments=dict(arguments)))
        result = await self.tools.execute(
            ToolCall(tool_name=step.tool_name, arguments=arguments), principal=self.principal
        )
        state.usage.tool_calls += 1
        state.tool_results.append(result)
        self._emit(
            AgentEvent(
                type="tool_result",
                tool_name=step.tool_name,
                status=result.status,
                result=result.output if result.status == "ok" else result.error,
            )
        )
        step.result = result.output if result.status == "ok" else {"status": result.status, "error": result.error}
        if result.status not in ("ok",):
            step.metadata["tool_status"] = result.status

    async def _step_validate(self, state: AgentState, step: AgentStep) -> None:
        draft = state.working_memory.get("analyze") or state.raw_answer_text or ""
        if not draft:
            done_thoughts = [s.result for s in state.steps if s.type == "think" and s.result]
            draft = str(done_thoughts[-1]) if done_thoughts else ""
        response = await self._model_call(
            state,
            self._context_messages(
                state,
                "Critique the draft answer below against the objective and the evidence. "
                "Set passed=false if claims are unsupported, the objective is unmet, or "
                "citations are missing where required; provide a corrected revised_answer.\n\n"
                f"Draft:\n{draft}",
            ),
            output_schema=_VALIDATE_SCHEMA,
            output_schema_name="validation",
        )
        payload = response.structured or {"passed": True, "issues": [], "revised_answer": draft}
        step.result = payload
        if not payload.get("passed", True) and payload.get("revised_answer"):
            state.working_memory["analyze"] = payload["revised_answer"]
        state.working_memory["validation"] = payload

    async def _step_ask_human(self, state: AgentState, step: AgentStep) -> None:
        if self.human_gate is None:
            state.terminated = True
            state.termination_reason = "approval_required"
            step.status = "failed"
            step.error = "human approval required but no gate configured"
            return
        approved = await self.human_gate(step.instruction, {"objective": state.objective.text})
        step.result = {"approved": approved}
        if not approved:
            state.terminated = True
            state.termination_reason = "approval_required"

    async def _step_finalize(self, state: AgentState, step: AgentStep) -> None:
        draft = state.working_memory.get("analyze")
        instruction = step.instruction or "Produce the final answer."
        if draft:
            instruction += f"\nDraft to finalize:\n{draft}"
        kwargs: dict[str, Any] = {}
        contract = self.output_validator.contract if self.output_validator else None
        if contract is not None and contract.schema_def is not None:
            kwargs["output_schema"] = contract.schema_def
            kwargs["output_schema_name"] = contract.schema_name
        response = await self._model_call(
            state, self._context_messages(state, instruction), stream_text=True, **kwargs
        )
        state.raw_answer_text = response.text
        if self.output_validator is not None:
            report = await self.output_validator.validate(
                response.text,
                structured=response.structured,
                evidence_ids={e.citation_ref for e in state.evidence} | {e.id for e in state.evidence},
            )
            step.result = report.model_dump(mode="json", exclude={"output"})
            if report.valid:
                state.final_answer = report.output
                state.termination_reason = "validation_passed"
            else:
                state.final_answer = response.structured or response.text
                state.errors.append(
                    AgentError(step_id=step.id, message=f"output validation failed: {report.errors}")
                )
                state.termination_reason = "objective_complete"
        else:
            state.final_answer = response.structured or response.text
            state.termination_reason = "objective_complete"
        state.terminated = True

    # -- DAG execution -------------------------------------------------------------------------

    async def run(
        self,
        objective: Objective | str,
        *,
        budget: Budget | None = None,
        initial_evidence: list[EvidenceItem] | None = None,
        attribution: dict[str, Any] | None = None,
    ) -> AgentState:
        objective = Objective(text=objective) if isinstance(objective, str) else objective
        state = AgentState(objective=objective, budget=budget or Budget())
        # Cost-attribution dimensions for this run's model calls; default
        # the run id to the agent state's id so events are always grouped.
        self.attribution = {"run_id": state.id, **(attribution or {})}
        if initial_evidence:
            state.evidence.extend(initial_evidence)

        if self.planner.mode == "react":
            return await self._run_react(state)

        dag = await self.planner.plan(
            objective,
            has_retrieval=self.retrieve_fn is not None,
            tools=self.tool_specs,
        )
        state.steps = list(dag.steps.values())
        await self._execute_dag(state, dag)
        # plan-and-execute: if the run finished unsatisfactorily, replan
        # corrective steps and execute them, bounded by ``max_replans`` + budget.
        if self.planner.mode == "plan_and_execute":
            await self._replan_loop(state, objective)
        if not state.terminated:
            state.terminated = True
            state.termination_reason = state.termination_reason or (
                "unrecoverable_error" if state.errors and state.final_answer is None else "objective_complete"
            )
        return state

    async def astream(
        self,
        objective: Objective | str,
        *,
        budget: Budget | None = None,
        initial_evidence: list[EvidenceItem] | None = None,
        attribution: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream the run as :class:`AgentEvent`\\ s.

        Yields ``run_start``, then live ``step_start`` / ``step_end``,
        ``tool_call`` / ``tool_result``, and real ``text_delta`` events, ending
        with a terminal ``done`` carrying the final :class:`AgentState` (or
        ``error``). Matches the streaming surface ``graph`` and ``compose`` expose
        and feeds the AG-UI protocol over the server's SSE path.
        """
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        previous_sink = self._event_sink
        self._event_sink = queue.put_nowait
        objective_text = objective if isinstance(objective, str) else objective.text
        self._emit(AgentEvent(type="run_start", instruction=objective_text[:300]))

        async def _drive() -> None:
            try:
                state = await self.run(
                    objective,
                    budget=budget,
                    initial_evidence=initial_evidence,
                    attribution=attribution,
                )
                queue.put_nowait(
                    AgentEvent(
                        type="done",
                        status=state.termination_reason or "",
                        result=state.final_answer,
                        usage={
                            "steps": float(state.usage.steps),
                            "tool_calls": float(state.usage.tool_calls),
                            "cost_usd": float(state.usage.cost_usd),
                        },
                        payload=state.model_dump(mode="json"),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - surface as a terminal event
                queue.put_nowait(AgentEvent(type="error", error=f"{type(exc).__name__}: {exc}"))
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(_drive())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            self._event_sink = previous_sink
            await task

    async def _run_step_with_retry(self, state: AgentState, step: AgentStep) -> None:
        await self._run_step(state, step)
        # Retry failed recoverable steps once.
        if (
            step.status == "failed"
            and step.attempts <= state.budget.max_retries
            and step.type in ("retrieve", "think")
        ):
            state.usage.retries += 1
            step.status = "pending"
            await self._run_step(state, step)

    async def _execute_dag(self, state: AgentState, dag: Any) -> None:
        """Run the DAG level by level, executing each level's independent steps
        concurrently — bounded by ``max_parallel_steps`` and the budget."""
        while not dag.complete and not state.terminated:
            if self._check_termination(state):
                break
            ready = dag.ready_steps()
            if not ready:
                break
            # Steps in one topological level share no dependencies, so running
            # them concurrently is safe; asyncio keeps state mutation serialized.
            semaphore = asyncio.Semaphore(self.max_parallel_steps)
            await asyncio.gather(
                *[self._run_step_bounded(state, step, semaphore) for step in ready]
            )

    async def _run_step_bounded(
        self, state: AgentState, step: AgentStep, semaphore: asyncio.Semaphore
    ) -> None:
        async with semaphore:
            if state.terminated or self._check_termination(state):
                return
            await self._run_step_with_retry(state, step)

    def _needs_replan(self, state: AgentState) -> bool:
        if state.termination_reason == "approval_required":
            return False
        validation = state.working_memory.get("validation")
        if isinstance(validation, dict) and validation.get("passed") is False:
            return True
        return state.final_answer is None

    async def _replan_loop(self, state: AgentState, objective: Objective) -> None:
        replans = 0
        while replans < self.max_replans and self._needs_replan(state):
            if state.usage.steps >= state.budget.max_steps or state.usage.exceeds(state.budget):
                break
            extra = await self.planner.replan(objective, state=state, tools=self.tool_specs)
            if extra is None or not extra.steps:
                break
            # Reopen the run for the corrective sub-plan, then execute it.
            state.terminated = False
            state.steps.extend(extra.steps.values())
            await self._execute_dag(state, extra)
            replans += 1
        state.working_memory["_replans"] = replans

    # -- ReAct loop -------------------------------------------------------------------

    async def _run_react(self, state: AgentState) -> AgentState:
        messages: list[Message] = []
        system = self.system_prompt or (
            "Solve the objective step by step. Use the available tools when they "
            "help. When you have enough information, give the final answer."
        )
        messages.append(Message(role="system", content=system, cache_hint=True))
        if state.evidence:
            evidence_lines = "\n".join(f"[{e.citation_ref}] {e.text}" for e in state.evidence[:20])
            messages.append(Message(role="user", content=f"Evidence:\n{evidence_lines}\n\nObjective: {state.objective.text}"))
        else:
            messages.append(Message(role="user", content=f"Objective: {state.objective.text}"))

        while not state.terminated:
            if self._check_termination(state):
                break
            # In-loop compaction: fold older turns into a rolling summary once the
            # transcript exceeds the token budget, so the loop never overflows.
            messages = self.compactor.compact_messages(messages, budget=self.max_context_tokens)
            response = await self._model_call(state, messages, tools=self.tool_specs, stream_text=True)
            state.usage.steps += 1
            step = AgentStep(type="think", name=f"react_{len(state.steps)}", instruction="react iteration")
            step.status = "done"
            step.result = response.text or [tc.name for tc in response.tool_calls]
            state.steps.append(step)
            if response.tool_calls and self.tools is not None:
                messages.append(
                    Message(role="assistant", content=response.text or "", tool_calls=response.tool_calls)
                )
                for tool_call in response.tool_calls:
                    if state.usage.tool_calls >= state.budget.max_tool_calls:
                        state.terminated = True
                        state.termination_reason = "budget_exhausted"
                        break
                    self._emit(
                        AgentEvent(type="tool_call", tool_name=tool_call.name, arguments=dict(tool_call.arguments))
                    )
                    try:
                        result = await self.tools.execute(
                            ToolCall(tool_name=tool_call.name, arguments=tool_call.arguments),
                            principal=self.principal,
                        )
                    except VincioError as exc:
                        from ..core.types import ToolResult

                        result = ToolResult(
                            call_id=tool_call.id, tool_name=tool_call.name, status="error", error=exc.message
                        )
                    state.usage.tool_calls += 1
                    state.tool_results.append(result)
                    self._emit(
                        AgentEvent(
                            type="tool_result",
                            tool_name=result.tool_name,
                            status=result.status,
                            result=result.output if result.status == "ok" else result.error,
                        )
                    )
                    messages.append(
                        Message(
                            role="tool",
                            content=json.dumps(result.output, default=str) if result.status == "ok" else f"error: {result.error}",
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        )
                    )
                continue
            # No tool calls: this is the final answer (already streamed live above).
            state.raw_answer_text = response.text
            if self.output_validator is not None:
                report = await self.output_validator.validate(
                    response.text,
                    structured=response.structured,
                    evidence_ids={e.citation_ref for e in state.evidence} | {e.id for e in state.evidence},
                )
                state.final_answer = report.output if report.valid else (response.structured or response.text)
                state.termination_reason = "validation_passed" if report.valid else "objective_complete"
            else:
                state.final_answer = response.structured or response.text
                state.termination_reason = "objective_complete"
            state.terminated = True
        return state
