"""In-place plan repair and replanning (agents/repair).

When a step of a running plan fails — a tool errors, validation surfaces a
contradiction, or the budget is about to blow — restarting the whole run throws
away every step that already succeeded. The :class:`PlanRepairer` instead edits
the *remaining* plan in place and lets the executor continue:

* **rebind** — a failed tool step is re-bound to an alternative tool (an explicit
  ``fallback_tools`` set, then a name-overlap sibling in the toolset);
* **substitute** — a failed tool with no alternative degrades to a reasoning
  step, so the run keeps moving rather than dead-ending;
* **reorder** — a validation contradiction inserts a corrective retrieval +
  re-analysis ahead of the finalize, instead of finalizing an unsupported draft;
* **drop** — under budget pressure the remaining optional steps are dropped and
  the finalize is rewired to run directly, returning the best answer reachable
  inside the budget rather than exhausting it.

Each repair is bounded (``max_repairs`` total, and a step is repaired at most
once per kind) and returned as a :class:`PlanRepair` record, which the executor
stamps onto the agent state, the trace, and the event bus — so a repair is a
first-class, auditable trajectory event, never a silent retry.
"""

from __future__ import annotations

from ..core.types import Budget, BudgetUsage, ToolSpec
from .dag import StepDAG
from .state import AgentState, AgentStep, PlanRepair, RepairAction, RepairTrigger

__all__ = ["RepairTrigger", "RepairAction", "PlanRepair", "PlanRepairer"]

# Tokens that carry no signal when matching one tool name against another.
_STOPWORDS = frozenset({"the", "a", "an", "of", "and", "to", "for", "tool", "api", "v1", "v2"})


def _tokens(name: str) -> set[str]:
    raw = name.replace("-", "_").replace(".", "_").replace(" ", "_").lower().split("_")
    return {t for t in raw if t and t not in _STOPWORDS}


class PlanRepairer:
    """Repairs a running :class:`StepDAG` in place. Deterministic and offline.

    The repairer never invents facts and never relaxes a budget — it only
    re-binds, substitutes, reorders, or prunes steps the plan already declared,
    keeping the run inside its original guarantees.
    """

    def __init__(
        self, *, max_repairs: int = 3, budget_shock_fraction: float = 0.75
    ) -> None:
        self.max_repairs = max(0, max_repairs)
        # Fraction of the cost budget that, once spent with non-finalize work
        # still pending, counts as a budget shock worth pruning toward an answer.
        self.budget_shock_fraction = budget_shock_fraction

    # -- failure repair ---------------------------------------------------------

    def repair_failure(
        self,
        state: AgentState,
        dag: StepDAG,
        step: AgentStep,
        *,
        tools: list[ToolSpec],
    ) -> PlanRepair | None:
        """Repair a freshly-failed ``step``; mutate the DAG and return the record.

        ``None`` means no repair was warranted (the executor's normal
        upstream-skip cascade then applies).
        """
        if step.status != "failed" or self._repairs_exhausted(state):
            return None
        if step.type == "tool":
            # A tool may re-bind through its untried alternatives (bounded by the
            # tried set) and finally substitute to reasoning, so a chain of failing
            # backups still recovers rather than dead-ending.
            return self._repair_tool(state, dag, step, tools)
        if step.type == "validate" and not self._already_repaired(state, step):
            # A contradiction inserts one corrective re-analysis, not a new one on
            # every re-critique.
            return self._repair_contradiction(state, dag, step)
        return None

    def _repair_tool(
        self, state: AgentState, dag: StepDAG, step: AgentStep, tools: list[ToolSpec]
    ) -> PlanRepair:
        tried = set(step.metadata.get("tried_tools", []))
        tried.add(step.tool_name or "")
        alternative = self._alternative_tool(step, tools, tried)
        if alternative is not None:
            repair = PlanRepair(
                action="rebind",
                trigger="tool_failure",
                step_id=step.id,
                step_name=step.name,
                detail=f"tool {step.tool_name!r} failed; re-bound to {alternative!r}",
                from_binding=step.tool_name,
                to_binding=alternative,
            )
            step.metadata["tried_tools"] = sorted(tried)
            step.tool_name = alternative
            step.tool_arguments = {}  # re-derive arguments for the new tool
        else:
            repair = PlanRepair(
                action="substitute",
                trigger="tool_failure",
                step_id=step.id,
                step_name=step.name,
                detail=f"no alternative tool for {step.tool_name!r}; reasoning instead",
                from_binding=step.tool_name,
                to_binding="think",
            )
            step.metadata["tried_tools"] = sorted(tried)
            step.metadata["substituted_from_tool"] = step.tool_name
            step.type = "think"
            step.tool_name = None
            step.instruction = (
                f"The tool {repair.from_binding!r} is unavailable. Reason about the "
                f"objective from the evidence already gathered to: {step.instruction}"
            )
        self._reset(step)
        self._mark_repaired(state, step, repair)
        return repair

    def _alternative_tool(
        self, step: AgentStep, tools: list[ToolSpec], tried: set[str]
    ) -> str | None:
        available = {t.name for t in tools}
        # 1. explicit fallbacks declared on the step (HTN operator / caller).
        for fallback in step.metadata.get("fallback_tools", []):
            if fallback in available and fallback not in tried:
                return str(fallback)
        # 2. a name-overlap sibling in the toolset (deterministic: most shared
        #    tokens first, then registry order).
        target = _tokens(step.tool_name or step.name)
        scored: list[tuple[int, int, str]] = []
        for index, spec in enumerate(tools):
            if spec.name in tried:
                continue
            overlap = len(target & _tokens(spec.name))
            if overlap:
                scored.append((-overlap, index, spec.name))
        scored.sort()
        return scored[0][2] if scored else None

    def _repair_contradiction(
        self, state: AgentState, dag: StepDAG, step: AgentStep
    ) -> PlanRepair | None:
        validation = state.working_memory.get("validation")
        issues = validation.get("issues", []) if isinstance(validation, dict) else []
        focus = "; ".join(str(i) for i in issues[:3]) or "the unmet objective"
        # Reorder the tail: insert a corrective re-analysis the finalize depends on.
        revise = AgentStep(
            type="think",
            name=f"{step.name}_revise",
            instruction=f"Revise the draft to resolve: {focus}. Stay grounded in the evidence.",
        )
        dag.add(revise, depends_on=[step.id])
        rewired: list[str] = []
        for other in dag.steps.values():
            if other.type == "finalize" and other.status == "pending" and step.id in other.input_refs:
                other.input_refs = [*[r for r in other.input_refs if r != step.id], revise.id]
                dag.edges.setdefault(revise.id, []).append(other.id)
                rewired.append(other.id)
        repair = PlanRepair(
            action="reorder",
            trigger="contradiction",
            step_id=step.id,
            step_name=step.name,
            detail=f"validation contradiction; inserted corrective re-analysis before {len(rewired)} finalize step(s)",
            added_steps=[revise.id],
        )
        step.status = "done"  # the critique itself succeeded; the fix is the new step
        self._mark_repaired(state, step, repair)
        return repair

    # -- budget-shock repair ----------------------------------------------------

    def repair_budget_shock(
        self, state: AgentState, dag: StepDAG, budget: Budget
    ) -> PlanRepair | None:
        """Prune the remaining optional plan toward the answer under budget pressure.

        Triggered between levels when spend has crossed
        ``budget_shock_fraction`` of the cost budget while non-finalize work is
        still pending. Drops the pending optional steps and rewires each pending
        finalize to run directly on what has already completed.
        """
        if self._repairs_exhausted(state) or state.working_memory.get("_budget_shock_repaired"):
            return None
        if not self._under_budget_shock(state.usage, budget):
            return None
        pending = [s for s in dag.steps.values() if s.status == "pending"]
        finalize = [s for s in pending if s.type == "finalize"]
        droppable = [s for s in pending if s.type != "finalize"]
        if not finalize or not droppable:
            return None
        done_ids = [s.id for s in dag.steps.values() if s.status == "done"]
        for step in droppable:
            step.status = "skipped"
            step.error = "dropped under budget pressure"
        for step in finalize:
            step.input_refs = list(done_ids)
        repair = PlanRepair(
            action="drop",
            trigger="budget_shock",
            detail=(
                f"spend ${state.usage.cost_usd:.4f} crossed "
                f"{self.budget_shock_fraction:.0%} of ${budget.max_cost_usd:.4f}; "
                f"dropped {len(droppable)} optional step(s) to finalize directly"
            ),
            dropped_steps=[s.id for s in droppable],
        )
        state.working_memory["_budget_shock_repaired"] = True
        state.repairs.append(repair)
        return repair

    def _under_budget_shock(self, usage: BudgetUsage, budget: Budget) -> bool:
        if budget.max_cost_usd <= 0:
            return False
        return usage.cost_usd >= self.budget_shock_fraction * budget.max_cost_usd

    # -- bookkeeping ------------------------------------------------------------

    def _repairs_exhausted(self, state: AgentState) -> bool:
        return len(state.repairs) >= self.max_repairs

    @staticmethod
    def _already_repaired(state: AgentState, step: AgentStep) -> bool:
        kinds = {(r.step_id, r.trigger) for r in state.repairs}
        trigger = "tool_failure" if step.type == "tool" else "contradiction"
        return (step.id, trigger) in kinds

    @staticmethod
    def _reset(step: AgentStep) -> None:
        step.status = "pending"
        step.error = None
        step.result = None

    @staticmethod
    def _mark_repaired(state: AgentState, step: AgentStep, repair: PlanRepair) -> None:
        state.repairs.append(repair)
