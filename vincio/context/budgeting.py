"""Token budget allocation.

Splits the input-token budget across context blocks. The default allocation
follows a fixed table; adaptive allocation reshapes it per task type
(classification needs no evidence; document QA is evidence-heavy). Blocks whose
size is already known — instructions, schema, the user task — are charged at
cost, and the entire remaining budget is distributed across the flexible blocks
(evidence, memory, tool results) proportionally to their fractions, so tokens a
fixed block does not need flow to evidence and memory rather than going unused.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.types import TaskType

__all__ = ["BlockBudget", "BudgetAllocation", "BudgetAllocator", "DEFAULT_ALLOCATION"]

# Default allocation (fractions of the input budget).
DEFAULT_ALLOCATION: dict[str, float] = {
    "instructions": 0.12,
    "examples": 0.08,
    "user_task": 0.05,
    "evidence": 0.45,
    "memory": 0.10,
    "tool_results": 0.10,
    "schema": 0.10,
}

# Task-specific reallocation (adaptive allocation).
TASK_ALLOCATIONS: dict[TaskType, dict[str, float]] = {
    TaskType.CLASSIFICATION: {
        "instructions": 0.30,
        "examples": 0.35,
        "user_task": 0.15,
        "evidence": 0.0,
        "memory": 0.05,
        "tool_results": 0.0,
        "schema": 0.15,
    },
    TaskType.EXTRACTION: {
        "instructions": 0.15,
        "examples": 0.10,
        "user_task": 0.30,
        "evidence": 0.20,
        "memory": 0.0,
        "tool_results": 0.05,
        "schema": 0.20,
    },
    TaskType.DOCUMENT_QA: {
        "instructions": 0.08,
        "examples": 0.02,
        "user_task": 0.05,
        "evidence": 0.70,
        "memory": 0.05,
        "tool_results": 0.05,
        "schema": 0.05,
    },
    TaskType.DOCUMENT_COMPARISON: {
        "instructions": 0.08,
        "examples": 0.02,
        "user_task": 0.05,
        "evidence": 0.70,
        "memory": 0.03,
        "tool_results": 0.05,
        "schema": 0.07,
    },
    TaskType.SUMMARIZATION: {
        "instructions": 0.10,
        "examples": 0.03,
        "user_task": 0.04,
        "evidence": 0.75,
        "memory": 0.03,
        "tool_results": 0.0,
        "schema": 0.05,
    },
    TaskType.TOOL_ACTION: {
        "instructions": 0.20,
        "examples": 0.05,
        "user_task": 0.15,
        "evidence": 0.10,
        "memory": 0.10,
        "tool_results": 0.30,
        "schema": 0.10,
    },
    TaskType.AGENT_WORKFLOW: {
        "instructions": 0.15,
        "examples": 0.05,
        "user_task": 0.10,
        "evidence": 0.30,
        "memory": 0.10,
        "tool_results": 0.20,
        "schema": 0.10,
    },
    TaskType.CODING: {
        "instructions": 0.12,
        "examples": 0.10,
        "user_task": 0.15,
        "evidence": 0.45,
        "memory": 0.05,
        "tool_results": 0.08,
        "schema": 0.05,
    },
}


class BlockBudget(BaseModel):
    block: str
    fraction: float
    tokens: int
    used_tokens: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.tokens - self.used_tokens)


class BudgetAllocation(BaseModel):
    total_tokens: int
    blocks: dict[str, BlockBudget] = Field(default_factory=dict)

    def block(self, name: str) -> BlockBudget:
        return self.blocks[name]

    def report(self) -> dict[str, dict[str, int | float]]:
        return {
            name: {
                "fraction": round(b.fraction, 4),
                "budget_tokens": b.tokens,
                "used_tokens": b.used_tokens,
                "remaining": b.remaining,
            }
            for name, b in self.blocks.items()
        }

    @property
    def used_total(self) -> int:
        return sum(b.used_tokens for b in self.blocks.values())


class BudgetAllocator:
    def __init__(
        self,
        allocation: dict[str, float] | None = None,
        *,
        learned: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.base_allocation = allocation or dict(DEFAULT_ALLOCATION)
        # Learned per-task tables: tuned from eval outcomes by
        # vincio.optimize.BudgetLearner, keyed by TaskType value. A learned
        # table overrides the fixed TASK_ALLOCATIONS entry for its task.
        self.learned: dict[str, dict[str, float]] = {
            str(key): dict(value) for key, value in (learned or {}).items()
        }

    def allocation_for(self, task_type: TaskType) -> dict[str, float]:
        fractions = self.learned.get(task_type.value)
        if fractions is None:
            fractions = TASK_ALLOCATIONS.get(task_type, self.base_allocation)
        fractions = dict(fractions)
        total = sum(fractions.values())
        if total <= 0:
            fractions = dict(DEFAULT_ALLOCATION)
            total = sum(fractions.values())
        return {k: v / total for k, v in fractions.items()}

    def allocate(
        self,
        total_tokens: int,
        *,
        task_type: TaskType = TaskType.GENERAL,
        fixed_costs: dict[str, int] | None = None,
        reserve_tokens: int = 0,
    ) -> BudgetAllocation:
        """Allocate *total_tokens* across blocks.

        ``fixed_costs`` are token counts for blocks whose size is already
        known (instructions, schema, user task): those are charged at cost,
        and the remainder is distributed over the flexible blocks
        proportionally to their configured fractions.

        ``reserve_tokens`` is headroom held back from the flexible blocks
        for the model's response and tool-loop turns, so the allocator accounts
        for the *full* window (input + output + tool loop) instead of input
        only. Defaults to 0 (no reservation), so it is fully additive.
        """
        fractions = self.allocation_for(task_type)
        fixed_costs = fixed_costs or {}
        allocation = BudgetAllocation(total_tokens=total_tokens)

        fixed_total = sum(fixed_costs.values())
        remaining = max(0, total_tokens - fixed_total - max(0, reserve_tokens))
        flexible = {k: v for k, v in fractions.items() if k not in fixed_costs}
        flexible_total = sum(flexible.values()) or 1.0

        # Every fraction block, plus any fixed-cost block not in the fraction
        # table (e.g. the pinned "anchor" reservation), gets a truthful line.
        names = list(fractions) + [k for k in fixed_costs if k not in fractions]
        for name in names:
            if name in fixed_costs:
                allocation.blocks[name] = BlockBudget(
                    block=name,
                    fraction=fixed_costs[name] / total_tokens if total_tokens else 0.0,
                    tokens=fixed_costs[name],
                    used_tokens=fixed_costs[name],
                )
            else:
                share = flexible[name] / flexible_total
                allocation.blocks[name] = BlockBudget(
                    block=name, fraction=share, tokens=int(remaining * share)
                )
        return allocation
