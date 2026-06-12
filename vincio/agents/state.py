"""Agent state models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.types import Budget, BudgetUsage, EvidenceItem, Objective, ToolResult
from ..core.utils import new_id

__all__ = ["AgentStepType", "AgentStep", "AgentError", "AgentState", "TerminationReason"]

AgentStepType = Literal["retrieve", "think", "tool", "validate", "ask_human", "finalize"]

TerminationReason = Literal[
    "objective_complete",
    "validation_passed",
    "max_steps",
    "budget_exhausted",
    "safety_violation",
    "approval_required",
    "unrecoverable_error",
]


class AgentStep(BaseModel):
    id: str = Field(default_factory=lambda: new_id("step"))
    type: AgentStepType
    name: str = ""
    instruction: str = ""  # what this step should accomplish
    input_refs: list[str] = Field(default_factory=list)  # upstream step ids
    output_refs: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"
    result: Any = None
    error: str | None = None
    attempts: int = 0
    duration_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentError(BaseModel):
    step_id: str | None = None
    message: str
    recoverable: bool = True


class AgentState(BaseModel):
    """Full execution state of one agent run."""

    id: str = Field(default_factory=lambda: new_id("agent_run"))
    objective: Objective
    steps: list[AgentStep] = Field(default_factory=list)
    working_memory: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    errors: list[AgentError] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    final_answer: Any = None
    raw_answer_text: str = ""
    terminated: bool = False
    termination_reason: TerminationReason | None = None

    def step_by_id(self, step_id: str) -> AgentStep | None:
        return next((s for s in self.steps if s.id == step_id), None)

    def upstream_results(self, step: AgentStep) -> list[Any]:
        return [
            upstream.result
            for ref in step.input_refs
            if (upstream := self.step_by_id(ref)) is not None and upstream.status == "done"
        ]

    def metrics(self) -> dict[str, Any]:
        """Agent eval metrics."""
        done = [s for s in self.steps if s.status == "done"]
        failed = [s for s in self.steps if s.status == "failed"]
        tool_steps = [s for s in self.steps if s.type == "tool"]
        return {
            "success": self.termination_reason in ("objective_complete", "validation_passed"),
            "steps_total": len(self.steps),
            "steps_done": len(done),
            "steps_failed": len(failed),
            "tool_calls": len(tool_steps),
            "tool_errors": sum(1 for s in tool_steps if s.status == "failed"),
            "termination_reason": self.termination_reason,
            "cost_usd": self.usage.cost_usd,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "errors": len(self.errors),
        }
