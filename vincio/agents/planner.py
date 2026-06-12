"""Agent planners: direct, static DAG, dynamic (LLM) DAG, ReAct.

Planners produce a :class:`StepDAG`; the executor runs it bounded by budget
and validation. ReAct is plan-free (the executor loops); the planner still
declares it so the mode is explicit and traceable.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from ..core.types import Message, ModelRequest, Objective, TaskType, ToolSpec
from ..providers.base import ModelProvider
from .dag import StepDAG
from .state import AgentStep

__all__ = ["PlanningMode", "Planner"]

PlanningMode = Literal["direct", "static", "dynamic", "react", "plan_and_execute"]

_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["retrieve", "think", "tool", "validate", "finalize"],
                    },
                    "instruction": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "type", "instruction", "tool_name", "depends_on"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}


class Planner:
    def __init__(
        self,
        *,
        mode: PlanningMode = "static",
        provider: ModelProvider | None = None,
        model: str | None = None,
        max_steps: int = 12,
    ) -> None:
        self.mode = mode
        self.provider = provider
        self.model = model
        self.max_steps = max_steps

    # -- static plans by task shape -------------------------------------------------

    def _static_plan(
        self,
        objective: Objective,
        *,
        has_retrieval: bool,
        tools: list[ToolSpec],
    ) -> StepDAG:
        dag = StepDAG()
        last_ids: list[str] = []
        if has_retrieval:
            retrieve = dag.add(
                AgentStep(type="retrieve", name="retrieve", instruction=objective.text)
            )
            last_ids = [retrieve.id]
        if tools and objective.task_type in (TaskType.TOOL_ACTION, TaskType.AGENT_WORKFLOW, TaskType.DATA_ANALYSIS):
            think = dag.add(
                AgentStep(
                    type="think",
                    name="plan_tools",
                    instruction="Decide which tools to call and with what arguments to satisfy the objective.",
                ),
                depends_on=last_ids,
            )
            last_ids = [think.id]
        analyze = dag.add(
            AgentStep(
                type="think",
                name="analyze",
                instruction="Analyze all gathered context and produce a draft answer with citations.",
            ),
            depends_on=last_ids,
        )
        validate = dag.add(
            AgentStep(
                type="validate",
                name="validate",
                instruction="Check the draft against the objective, evidence, and output contract.",
            ),
            depends_on=[analyze.id],
        )
        dag.add(
            AgentStep(type="finalize", name="finalize", instruction="Produce the final answer."),
            depends_on=[validate.id],
        )
        return dag

    def _direct_plan(self, objective: Objective) -> StepDAG:
        dag = StepDAG()
        dag.add(AgentStep(type="finalize", name="answer", instruction=objective.text))
        return dag

    # -- dynamic (LLM) plans -------------------------------------------------------------

    async def _dynamic_plan(
        self,
        objective: Objective,
        *,
        has_retrieval: bool,
        tools: list[ToolSpec],
    ) -> StepDAG:
        if self.provider is None or self.model is None:
            return self._static_plan(objective, has_retrieval=has_retrieval, tools=tools)
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools) or "(none)"
        request = ModelRequest(
            model=self.model,
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are an agent planner. Produce a minimal DAG of steps to "
                        "accomplish the objective. Step types: retrieve (search the "
                        "knowledge base), think (reason over gathered context), tool "
                        "(call one of the available tools; set tool_name), validate, "
                        "finalize (exactly one, last). Use depends_on with step names. "
                        f"At most {self.max_steps} steps. For steps without a tool, set tool_name to ''."
                    ),
                ),
                Message(
                    role="user",
                    content=f"Objective: {objective.text}\nTask type: {objective.task_type.value}\nAvailable tools:\n{tool_lines}",
                ),
            ],
            output_schema=_PLAN_SCHEMA,
            output_schema_name="agent_plan",
            temperature=0.0,
        )
        try:
            response = await self.provider.generate(request)
            payload = response.structured or json.loads(response.text)
            return self._dag_from_payload(payload, tools)
        except Exception:  # noqa: BLE001 - fall back to a safe static plan
            return self._static_plan(objective, has_retrieval=has_retrieval, tools=tools)

    def _dag_from_payload(self, payload: dict[str, Any], tools: list[ToolSpec]) -> StepDAG:
        dag = StepDAG()
        name_to_id: dict[str, str] = {}
        tool_names = {t.name for t in tools}
        raw_steps = payload.get("steps", [])[: self.max_steps]
        has_finalize = any(s.get("type") == "finalize" for s in raw_steps)
        for raw in raw_steps:
            step_type = raw.get("type", "think")
            tool_name = (raw.get("tool_name") or "").strip() or None
            if step_type == "tool" and tool_name not in tool_names:
                step_type = "think"  # unknown tool: degrade to reasoning
                tool_name = None
            step = AgentStep(
                type=step_type,
                name=raw.get("name", step_type),
                instruction=raw.get("instruction", ""),
                tool_name=tool_name,
            )
            depends = [name_to_id[d] for d in raw.get("depends_on", []) if d in name_to_id]
            dag.add(step, depends_on=depends)
            name_to_id[step.name] = step.id
        if not has_finalize:
            terminal_ids = [
                step_id for step_id in dag.steps if not dag.edges.get(step_id)
            ]
            dag.add(
                AgentStep(type="finalize", name="finalize", instruction="Produce the final answer."),
                depends_on=terminal_ids,
            )
        return dag

    # -- entry ----------------------------------------------------------------------------

    async def plan(
        self,
        objective: Objective,
        *,
        has_retrieval: bool = False,
        tools: list[ToolSpec] | None = None,
    ) -> StepDAG:
        tools = tools or []
        if self.mode == "direct":
            return self._direct_plan(objective)
        if self.mode in ("dynamic", "plan_and_execute"):
            return await self._dynamic_plan(objective, has_retrieval=has_retrieval, tools=tools)
        if self.mode == "react":
            # ReAct is executor-driven; an empty DAG signals loop mode.
            return StepDAG()
        return self._static_plan(objective, has_retrieval=has_retrieval, tools=tools)
