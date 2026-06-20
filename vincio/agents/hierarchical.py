"""Hierarchical task-network (HTN) planning (agents/hierarchical).

An HTN planner decomposes a goal into a tree of sub-goals and binds each *leaf*
to a bounded action — a tool call, a retrieval, a reasoning turn, or the final
answer. Unlike a flat planner that emits one list of steps, an HTN planner works
top-down: a compound task is expanded by a **method** into ordered subtasks, and
the recursion bottoms out at **operators** that name a concrete step.

The output is a plain :class:`~vincio.agents.dag.StepDAG`, so the existing
:class:`~vincio.agents.executor.AgentExecutor` runs an HTN plan with no special
casing — hierarchical planning composes with the direct / static / dynamic /
ReAct / plan-and-execute modes rather than replacing them, and a plan it
produces is repaired in place by the same plan-repair pass.

Two ways to decompose:

* **knowledge-driven** — supply an :class:`HTNDomain` of operators and methods;
  decomposition is deterministic (offline-safe, reproducible), choosing the
  first method whose precondition matches the planning context; and
* **model-driven** — with a provider, the planner asks the model for a two-level
  goal → sub-goal → step decomposition validated against a schema, falling back
  to a safe static plan when no domain and no model are available.

The resolved sub-goal tree (:class:`HTNPlanNode`) is returned alongside the DAG
for tracing and inspection, so a hierarchical plan is observable, not opaque.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .dag import StepDAG
from .state import AgentStep, AgentStepType

__all__ = [
    "HTNOperator",
    "HTNMethod",
    "HTNDomain",
    "HTNPlanNode",
    "MethodOrdering",
    "dag_from_plan_node",
]

# Ordering of a method's subtasks: ``sequence`` chains them (each depends on the
# previous), ``parallel`` leaves them independent so the executor's level
# scheduler runs them concurrently.
MethodOrdering = Literal["sequence", "parallel"]


class HTNOperator(BaseModel):
    """A primitive task bound to one concrete, bounded action.

    ``step_type`` is the executor step the leaf becomes (``retrieve`` / ``think``
    / ``tool`` / ``validate`` / ``finalize``); ``tool_name`` binds a ``tool``
    operator to a registered tool. An operator whose ``tool_name`` is not
    available at plan time degrades to a reasoning (``think``) step, exactly as
    the dynamic planner degrades an unknown tool — the plan never references a
    tool the runtime cannot honor.
    """

    name: str
    step_type: AgentStepType = "think"
    tool_name: str | None = None
    instruction: str = ""
    # Alternative tools a ``tool`` leaf may re-bind to if its primary fails — the
    # explicit substitution set the plan-repair pass prefers before degrading to
    # a reasoning step.
    fallbacks: list[str] = Field(default_factory=list)


class HTNMethod(BaseModel):
    """A decomposition of one compound task into ordered subtasks.

    ``subtasks`` names operators or other compound tasks. ``when`` is an optional
    precondition matched (by equality) against the planning context — the first
    method whose precondition holds is chosen, so a domain can offer alternative
    decompositions guarded by task shape (e.g. retrieval availability).
    """

    task: str
    name: str = ""
    subtasks: list[str] = Field(default_factory=list)
    ordering: MethodOrdering = "sequence"
    when: dict[str, Any] = Field(default_factory=dict)

    def applies(self, context: dict[str, Any]) -> bool:
        return all(context.get(key) == value for key, value in self.when.items())


class HTNPlanNode(BaseModel):
    """A node of the resolved sub-goal tree.

    A leaf carries a bound :class:`HTNOperator`; a compound node carries the
    chosen method's ``ordering`` and its expanded ``children``. Returned for
    tracing and inspection — the executable form is the :class:`StepDAG` built by
    :func:`dag_from_plan_node`.
    """

    task: str
    ordering: MethodOrdering = "sequence"
    operator: HTNOperator | None = None
    children: list[HTNPlanNode] = Field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.operator is not None

    def leaves(self) -> list[HTNOperator]:
        if self.operator is not None:
            return [self.operator]
        out: list[HTNOperator] = []
        for child in self.children:
            out.extend(child.leaves())
        return out


class HTNDomain(BaseModel):
    """A library of operators and methods the planner decomposes against.

    Build one with :meth:`operator` / :meth:`method` (chainable), then call
    :meth:`decompose` with the root task. Decomposition is deterministic and
    cycle-guarded (``max_depth`` bounds runaway recursion), so the same domain
    and context always yield the same sub-goal tree.
    """

    operators: dict[str, HTNOperator] = Field(default_factory=dict)
    methods: dict[str, list[HTNMethod]] = Field(default_factory=dict)
    max_depth: int = 32

    def operator(
        self,
        name: str,
        *,
        step_type: AgentStepType = "think",
        tool_name: str | None = None,
        instruction: str = "",
        fallbacks: list[str] | None = None,
    ) -> HTNDomain:
        self.operators[name] = HTNOperator(
            name=name,
            step_type=step_type,
            tool_name=tool_name,
            instruction=instruction,
            fallbacks=fallbacks or [],
        )
        return self

    def method(
        self,
        task: str,
        subtasks: list[str],
        *,
        name: str = "",
        ordering: MethodOrdering = "sequence",
        when: dict[str, Any] | None = None,
    ) -> HTNDomain:
        self.methods.setdefault(task, []).append(
            HTNMethod(
                task=task, name=name or task, subtasks=subtasks, ordering=ordering, when=when or {}
            )
        )
        return self

    def decompose(
        self, task: str, *, context: dict[str, Any] | None = None
    ) -> HTNPlanNode:
        """Expand ``task`` into a resolved sub-goal tree.

        An operator name resolves to a leaf; a compound name resolves through the
        first applicable method. An unknown task with no method and no operator is
        treated as a reasoning leaf so a partially-specified domain still plans.
        """
        return self._decompose(task, context or {}, depth=0, seen=())

    def _decompose(
        self, task: str, context: dict[str, Any], *, depth: int, seen: tuple[str, ...]
    ) -> HTNPlanNode:
        if task in self.operators:
            return HTNPlanNode(task=task, operator=self.operators[task])
        candidates = self.methods.get(task, [])
        chosen = next((m for m in candidates if m.applies(context)), None)
        if chosen is None or depth >= self.max_depth or task in seen:
            # No applicable method (or recursion guard tripped): bottom out as a
            # reasoning leaf so the plan is always executable.
            return HTNPlanNode(
                task=task,
                operator=HTNOperator(name=task, step_type="think", instruction=task),
            )
        children = [
            self._decompose(sub, context, depth=depth + 1, seen=(*seen, task))
            for sub in chosen.subtasks
        ]
        return HTNPlanNode(task=task, ordering=chosen.ordering, children=children)


def dag_from_plan_node(
    root: HTNPlanNode, *, available_tools: set[str] | None = None
) -> StepDAG:
    """Flatten a resolved sub-goal tree into an executable :class:`StepDAG`.

    Sequence methods chain their children (each depends on the previous child's
    exit steps); parallel methods leave siblings independent so they land on one
    topological level. A terminal ``finalize`` step is appended when the plan
    does not already end in one, so the executor always has a single answer step.
    """
    tools = available_tools or set()
    dag = StepDAG()

    def emit(node: HTNPlanNode, upstream: list[str]) -> list[str]:
        if node.is_leaf:
            op = node.operator
            assert op is not None
            step_type: AgentStepType = op.step_type
            tool_name = op.tool_name
            if step_type == "tool" and (tool_name is None or tool_name not in tools):
                step_type, tool_name = "think", None  # degrade an unbound tool
            metadata: dict[str, Any] = {}
            fallbacks = [f for f in op.fallbacks if f in tools]
            if step_type == "tool" and fallbacks:
                metadata["fallback_tools"] = fallbacks
            step = AgentStep(
                type=step_type,
                name=op.name,
                instruction=op.instruction or node.task,
                tool_name=tool_name,
                metadata=metadata,
            )
            dag.add(step, depends_on=upstream)
            return [step.id]
        if node.ordering == "parallel":
            exits: list[str] = []
            for child in node.children:
                exits.extend(emit(child, upstream))
            return exits or upstream
        current = upstream
        for child in node.children:
            current = emit(child, current)
        return current

    exits = emit(root, [])
    has_finalize = any(step.type == "finalize" for step in dag.steps.values())
    if not has_finalize:
        dag.add(
            AgentStep(type="finalize", name="finalize", instruction="Produce the final answer."),
            depends_on=exits,
        )
    return dag


# Pydantic v2 needs the forward reference in ``children`` resolved once the class
# body is complete.
HTNPlanNode.model_rebuild()
