"""Agent trajectory model and adapters.

A :class:`Trajectory` is the provider-neutral, eval-ready view of *how* a run
reached its answer — the objective, the ordered steps (with tool calls), the
outcome, and the resource usage. Crews, ``StateGraph`` runs, and raw traces are
all projected onto the same shape, so the trajectory metrics in
:mod:`vincio.evals.metrics` (``tool_call_accuracy``, ``goal_accuracy``,
``plan_adherence`` …) score any of them without re-instrumentation.

This module deliberately imports nothing heavy at module load (only pydantic and
``core.types``); the ``from_*`` adapters read their sources by attribute, so
``metrics.py`` can depend on ``Trajectory`` without a circular import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..agents.crew import CrewResult
    from ..agents.state import AgentState
    from ..observability.spans import Trace

__all__ = [
    "TrajectoryStep",
    "Trajectory",
    "trajectory_from_agent_state",
    "trajectory_from_crew_result",
    "trajectory_from_trace",
    "TRAJECTORY_METRICS",
]

# Outcome statuses that count as a successfully completed step or run.
_OK_STATUSES = {"done", "ok", "succeeded", "validation_passed", "objective_complete"}
# Run terminations that count as goal success.
_SUCCESS_TERMINATIONS = {"objective_complete", "validation_passed"}


class TrajectoryStep(BaseModel):
    """One step of an agent run, normalized across sources."""

    type: str = "step"  # retrieve | think | tool | validate | finalize | agent_step | ...
    name: str = ""
    instruction: str = ""
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    status: str = "done"
    error: str | None = None

    @property
    def is_tool(self) -> bool:
        return self.type in ("tool", "tool_call") or self.tool_name is not None

    @property
    def ok(self) -> bool:
        return self.status in _OK_STATUSES

    @property
    def text(self) -> str:
        """All free text on the step (used by topic_adherence)."""
        parts = [self.name, self.instruction, self.tool_name or ""]
        if self.tool_arguments:
            parts.append(" ".join(f"{k} {v}" for k, v in self.tool_arguments.items()))
        return " ".join(p for p in parts if p).strip()


class Trajectory(BaseModel):
    """Provider-neutral view of one agent run, ready for trajectory metrics."""

    objective: str = ""
    steps: list[TrajectoryStep] = Field(default_factory=list)
    final_answer: Any = None
    raw_text: str = ""
    terminated: bool = False
    termination_reason: str | None = None
    success: bool = False
    source: str = ""  # agent_state | crew | trace
    usage: dict[str, float] = Field(default_factory=dict)

    def tool_calls(self) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.is_tool]

    def tool_names(self) -> list[str]:
        return [s.tool_name or s.name for s in self.tool_calls()]

    # -- adapters (also exposed as module functions) -------------------------

    @classmethod
    def from_agent_state(cls, state: AgentState) -> Trajectory:
        return trajectory_from_agent_state(state)

    @classmethod
    def from_crew_result(cls, result: CrewResult) -> Trajectory:
        return trajectory_from_crew_result(result)

    @classmethod
    def from_trace(cls, trace: Trace) -> Trajectory:
        return trajectory_from_trace(trace)


def trajectory_from_agent_state(state: AgentState) -> Trajectory:
    """Project a :class:`~vincio.agents.state.AgentState` onto a Trajectory.

    Planner/graph agents carry the tool on the step itself; a ReAct loop records
    its tool invocations in ``tool_results`` while its steps are reasoning turns.
    Both are normalized to the same ordered tool-use trajectory."""
    if any(step.tool_name for step in state.steps):
        steps = [
            TrajectoryStep(
                type=step.type,
                name=step.name,
                instruction=step.instruction,
                tool_name=step.tool_name,
                tool_arguments=dict(step.tool_arguments or {}),
                status=step.status,
                error=step.error,
            )
            for step in state.steps
        ]
    else:
        steps = [
            TrajectoryStep(
                type="tool",
                name=result.tool_name,
                tool_name=result.tool_name,
                status="done" if result.status == "ok" else result.status,
                error=result.error,
            )
            for result in state.tool_results
        ]
        if state.terminated and (state.final_answer is not None or state.raw_answer_text):
            steps.append(TrajectoryStep(type="finalize", name="finalize", status="done"))
    # In-place plan repairs are first-class trajectory events: surface each as a
    # step so trajectory metrics and human review see how the plan recovered.
    for repair in state.repairs:
        steps.append(
            TrajectoryStep(
                type="plan_repair",
                name=repair.step_name or repair.action,
                instruction=repair.detail,
                status=repair.action,
            )
        )
    reason = state.termination_reason
    usage = state.usage
    return Trajectory(
        objective=state.objective.text if state.objective else "",
        steps=steps,
        final_answer=state.final_answer,
        raw_text=state.raw_answer_text or "",
        terminated=state.terminated,
        termination_reason=reason,
        success=reason in _SUCCESS_TERMINATIONS,
        source="agent_state",
        usage={
            "steps": float(len(steps)),
            "reasoning_steps": float(len(state.steps)),
            "tool_calls": float(sum(1 for s in steps if s.is_tool)),
            "input_tokens": float(usage.input_tokens),
            "output_tokens": float(usage.output_tokens),
            "cost_usd": float(usage.cost_usd),
        },
    )


def trajectory_from_crew_result(result: CrewResult) -> Trajectory:
    """Project a :class:`~vincio.agents.crew.CrewResult` onto a Trajectory.

    Each member run and each delegation becomes a step, so tool/plan metrics see
    the crew as one ordered trajectory.
    """
    steps: list[TrajectoryStep] = []
    for record in result.delegations:
        steps.append(
            TrajectoryStep(
                type="delegate",
                name=f"{record.from_agent}->{record.to_agent}",
                instruction=record.task,
                status="done",
            )
        )
    for report in result.reports:
        ok = bool(report.metrics.get("success")) or report.termination_reason in _SUCCESS_TERMINATIONS
        steps.append(
            TrajectoryStep(
                type="agent_step",
                name=report.role,
                instruction=report.task,
                status="done" if ok else "failed",
            )
        )
    usage = result.usage
    return Trajectory(
        objective="",
        steps=steps,
        final_answer=result.output,
        raw_text=str(result.output or ""),
        terminated=result.status != "max_rounds",
        termination_reason=result.status,
        success=result.status == "succeeded",
        source="crew",
        usage={
            "steps": float(len(steps)),
            "tool_calls": float(usage.tool_calls),
            "input_tokens": float(usage.input_tokens),
            "output_tokens": float(usage.output_tokens),
            "cost_usd": float(usage.cost_usd),
            "rounds": float(result.rounds),
        },
    )


def trajectory_from_trace(trace: Trace) -> Trajectory:
    """Project a captured :class:`~vincio.observability.spans.Trace` onto a
    Trajectory by reading its ``agent_step`` / ``tool_call`` spans."""
    steps: list[TrajectoryStep] = []
    tool_calls = 0
    for span in sorted(trace.spans, key=lambda s: s.start_time):
        if span.type == "tool_call":
            tool_calls += 1
            steps.append(
                TrajectoryStep(
                    type="tool",
                    name=span.name,
                    tool_name=span.attributes.get("tool") or span.name,
                    tool_arguments=dict(span.attributes.get("arguments") or {}),
                    status="ok" if span.status == "ok" else span.status,
                    error=span.error,
                )
            )
        elif span.type in ("agent_step", "crew_agent", "graph_node"):
            steps.append(
                TrajectoryStep(
                    type="agent_step",
                    name=span.name,
                    instruction=str(span.attributes.get("instruction", "")),
                    status="ok" if span.status == "ok" else span.status,
                    error=span.error,
                )
            )
    return Trajectory(
        objective=str(trace.attributes.get("input", "")),
        steps=steps,
        final_answer=trace.attributes.get("output"),
        raw_text=str(trace.attributes.get("output") or ""),
        terminated=trace.status != "running",
        termination_reason=trace.status,
        success=trace.status == "ok",
        source="trace",
        usage={
            "steps": float(len(steps)),
            "tool_calls": float(tool_calls),
            "cost_usd": float(trace.attributes.get("cost_usd", 0.0) or 0.0),
        },
    )


# The metric names that read a Trajectory rather than just the final output.
# Reports use this to show output-only and trajectory evaluation side by side.
TRAJECTORY_METRICS: frozenset[str] = frozenset(
    {
        "tool_call_accuracy",
        "tool_call_f1",
        "goal_accuracy",
        "plan_adherence",
        "plan_quality",
        "step_efficiency",
        "topic_adherence",
    }
)
