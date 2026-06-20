"""Agent planners: direct, static DAG, dynamic (LLM) DAG, hierarchical, ReAct.

Planners produce a :class:`StepDAG`; the executor runs it bounded by budget
and validation. ReAct is plan-free (the executor loops); the planner still
declares it so the mode is explicit and traceable.

The ``hierarchical`` mode runs an HTN decomposition (see
:mod:`vincio.agents.hierarchical`): a goal is decomposed into a sub-goal tree and
each leaf is bound to a bounded tool / retrieval / reasoning / finalize step. It
is just another mode, so it composes with plan repair and cost-aware selection.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from ..core.types import Message, ModelRequest, Objective, TaskType, ToolSpec
from ..providers.base import ModelProvider
from .dag import StepDAG
from .hierarchical import (
    HTNDomain,
    HTNOperator,
    HTNPlanNode,
    MethodOrdering,
    dag_from_plan_node,
)
from .state import AgentStep

__all__ = ["PlanningMode", "Planner"]

PlanningMode = Literal[
    "direct", "static", "dynamic", "react", "plan_and_execute", "hierarchical"
]

_HTN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subgoals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "method": {"type": "string", "enum": ["sequence", "parallel"]},
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
                            },
                            "required": ["name", "type", "instruction", "tool_name"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["name", "method", "steps"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subgoals"],
    "additionalProperties": False,
}

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
        domain: HTNDomain | None = None,
        root_task: str = "root",
    ) -> None:
        self.mode = mode
        self.provider = provider
        self.model = model
        self.max_steps = max_steps
        # HTN knowledge for the ``hierarchical`` mode. When set, decomposition is
        # deterministic; otherwise the planner asks the model (or, offline,
        # degrades to a static plan). ``root_task`` names the domain's entry task.
        self.domain = domain
        self.root_task = root_task
        # The most recent resolved sub-goal tree (``hierarchical`` mode only),
        # exposed for tracing / inspection; ``None`` until the first plan.
        self.last_plan_tree: HTNPlanNode | None = None

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

    # -- hierarchical (HTN) plans --------------------------------------------------------

    async def _hierarchical_plan(
        self,
        objective: Objective,
        *,
        has_retrieval: bool,
        tools: list[ToolSpec],
    ) -> StepDAG:
        """Decompose the goal into a sub-goal tree and bind each leaf to a step.

        A supplied :class:`HTNDomain` decomposes deterministically; otherwise the
        model proposes a two-level decomposition; offline with no domain the
        planner degrades to a safe static plan. The resolved tree is stashed on
        :attr:`last_plan_tree` for tracing.
        """
        tool_names = {t.name for t in tools}
        if self.domain is not None:
            context = {
                "task_type": objective.task_type.value,
                "has_retrieval": has_retrieval,
                "has_tools": bool(tools),
            }
            root_task = str(objective.metadata.get("root_task") or self.root_task)
            root = self.domain.decompose(root_task, context=context)
            self.last_plan_tree = root
            return dag_from_plan_node(root, available_tools=tool_names)
        if self.provider is not None and self.model is not None:
            dag = await self._llm_hierarchical_plan(objective, has_retrieval, tools)
            if dag is not None:
                return dag
        self.last_plan_tree = None
        return self._static_plan(objective, has_retrieval=has_retrieval, tools=tools)

    async def _llm_hierarchical_plan(
        self, objective: Objective, has_retrieval: bool, tools: list[ToolSpec]
    ) -> StepDAG | None:
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools) or "(none)"
        request = ModelRequest(
            model=self.model or "",
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are a hierarchical (HTN) agent planner. Decompose the goal "
                        "into a small ordered list of sub-goals; for each sub-goal emit the "
                        "concrete steps that accomplish it. Sub-goals run in order; a "
                        "sub-goal's steps run in 'sequence' or in 'parallel'. Step types: "
                        "retrieve, think, tool (set tool_name to an available tool), "
                        "validate, finalize. For non-tool steps set tool_name to ''. Keep "
                        f"the whole plan to at most {self.max_steps} steps."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"Goal: {objective.text}\nTask type: {objective.task_type.value}\n"
                        f"Retrieval available: {has_retrieval}\nAvailable tools:\n{tool_lines}"
                    ),
                ),
            ],
            output_schema=_HTN_SCHEMA,
            output_schema_name="htn_plan",
            temperature=0.0,
        )
        if self.provider is None:  # pragma: no cover - guarded by the caller
            return None
        try:
            response = await self.provider.generate(request)
            payload = response.structured or json.loads(response.text)
        except Exception:  # noqa: BLE001 - fall back to the static plan
            return None
        return self._dag_from_htn_payload(payload, tools)

    def _dag_from_htn_payload(
        self, payload: dict[str, Any], tools: list[ToolSpec]
    ) -> StepDAG | None:
        subgoals = payload.get("subgoals") or []
        if not subgoals:
            return None
        tool_names = {t.name for t in tools}
        children: list[HTNPlanNode] = []
        total_steps = 0
        for raw_goal in subgoals:
            ordering: MethodOrdering = (
                "parallel" if raw_goal.get("method") == "parallel" else "sequence"
            )
            leaves: list[HTNPlanNode] = []
            for raw_step in raw_goal.get("steps", []):
                if total_steps >= self.max_steps:
                    break
                step_type = raw_step.get("type", "think")
                tool_name = (raw_step.get("tool_name") or "").strip() or None
                if step_type == "tool" and tool_name not in tool_names:
                    step_type, tool_name = "think", None
                leaves.append(
                    HTNPlanNode(
                        task=raw_step.get("name", step_type),
                        operator=HTNOperator(
                            name=raw_step.get("name", step_type),
                            step_type=step_type,
                            tool_name=tool_name,
                            instruction=raw_step.get("instruction", ""),
                        ),
                    )
                )
                total_steps += 1
            if leaves:
                children.append(
                    HTNPlanNode(task=raw_goal.get("name", "subgoal"), ordering=ordering, children=leaves)
                )
        if not children:
            return None
        root = HTNPlanNode(task="root", ordering="sequence", children=children)
        self.last_plan_tree = root
        return dag_from_plan_node(root, available_tools=tool_names)

    # -- replanning (plan-and-execute) ----------------------------------------------------

    async def replan(
        self,
        objective: Objective,
        *,
        state: Any,
        tools: list[ToolSpec] | None = None,
    ) -> StepDAG | None:
        """Propose corrective steps after an unsatisfactory execution.

        Given the progress so far (draft answer, validation feedback, gathered
        evidence), returns a small sub-DAG of additional steps — or ``None`` when
        no further work is warranted. With a provider the proposal is model-driven
        and validated against the plan schema; offline a deterministic corrective
        plan (revise → finalize) keeps the loop reproducible.
        """
        tools = tools or []
        validation = state.working_memory.get("validation") if state.working_memory else None
        issues = validation.get("issues", []) if isinstance(validation, dict) else []
        if self.provider is not None and self.model is not None:
            dag = await self._llm_replan(objective, state, tools, issues)
            if dag is not None:
                return dag
        return self._heuristic_replan(objective, issues)

    def _heuristic_replan(self, objective: Objective, issues: list[str]) -> StepDAG | None:
        dag = StepDAG()
        focus = "; ".join(str(i) for i in issues[:3]) or "the unmet objective"
        revise = dag.add(
            AgentStep(
                type="think",
                name="revise",
                instruction=f"Revise the draft to address: {focus}. Stay grounded in the evidence.",
            )
        )
        dag.add(
            AgentStep(type="finalize", name="refinalize", instruction="Produce the corrected final answer."),
            depends_on=[revise.id],
        )
        return dag

    async def _llm_replan(
        self, objective: Objective, state: Any, tools: list[ToolSpec], issues: list[str]
    ) -> StepDAG | None:
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools) or "(none)"
        draft = state.working_memory.get("analyze", "") if state.working_memory else ""
        request = ModelRequest(
            model=self.model or "",
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are replanning an agent run that did not satisfy the objective. "
                        "Propose only the additional steps needed to fix it (retrieve / think / "
                        "tool / validate / finalize, exactly one finalize last). Return an empty "
                        "list if the objective is already met. Use depends_on with step names."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"Objective: {objective.text}\nKnown issues: {issues}\n"
                        f"Draft: {str(draft)[:800]}\nTools:\n{tool_lines}"
                    ),
                ),
            ],
            output_schema=_PLAN_SCHEMA,
            output_schema_name="replan",
            temperature=0.0,
        )
        try:
            response = await self.provider.generate(request)  # type: ignore[union-attr]
            payload = response.structured or json.loads(response.text)
        except Exception:  # noqa: BLE001 - fall back to the deterministic corrective plan
            return None
        if not payload.get("steps"):
            return None
        return self._dag_from_payload(payload, tools)

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
        if self.mode == "hierarchical":
            return await self._hierarchical_plan(
                objective, has_retrieval=has_retrieval, tools=tools
            )
        if self.mode in ("dynamic", "plan_and_execute"):
            return await self._dynamic_plan(objective, has_retrieval=has_retrieval, tools=tools)
        if self.mode == "react":
            # ReAct is executor-driven; an empty DAG signals loop mode.
            return StepDAG()
        return self._static_plan(objective, has_retrieval=has_retrieval, tools=tools)
