"""Multi-agent crews (agents/crew).

A :class:`Crew` binds named roles to bounded :class:`AgentExecutor`s and runs
them as a team over a shared :class:`Blackboard`. Three processes:

- ``sequential`` — members run in order; each sees everything posted so far.
- ``parallel`` — members run concurrently on their tasks, then post results.
- ``hierarchical`` — a manager decomposes the objective, delegates tasks to
  members (LLM-planned with a deterministic offline fallback), reviews the
  board, and either finishes or delegates follow-ups, bounded by
  ``max_rounds``.

Termination is guaranteed by construction: every member runs under a scaled
share of the crew budget, the crew checks its own budget before every
delegation, and hierarchical review is capped at ``max_rounds``. Every member
run emits a ``crew_agent`` span and an eval-ready ``AgentState.metrics()``
report, so crews are traced and scoreable like any other Vincio run.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.concurrency import gather_bounded
from ..core.errors import AgentEngineError
from ..core.types import Budget, BudgetUsage, Message, ModelRequest, Objective
from ..observability.traces import Tracer
from ..providers.base import ModelProvider, run_sync
from .blackboard import Blackboard
from .executor import AgentExecutor
from .state import AgentState

__all__ = ["AgentRole", "DelegationRecord", "CrewMemberReport", "CrewResult", "Crew"]

CrewProcess = Literal["sequential", "parallel", "hierarchical"]

_ASSIGN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "task": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["agent", "task", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assignments"],
    "additionalProperties": False,
}

_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "done": {"type": "boolean"},
        "final_answer": {"type": "string"},
        "follow_ups": _ASSIGN_SCHEMA["properties"]["assignments"],
    },
    "required": ["done", "final_answer", "follow_ups"],
    "additionalProperties": False,
}


class AgentRole(BaseModel):
    """A named role in a crew: who the agent is and what share it gets."""

    name: str
    description: str = ""
    goal: str = ""
    keywords: list[str] = Field(default_factory=list)
    budget_fraction: float | None = None  # share of the crew budget; default equal split


class DelegationRecord(BaseModel):
    from_agent: str = "manager"
    to_agent: str
    task: str
    reason: str = ""
    round: int = 0


class CrewMemberReport(BaseModel):
    role: str
    task: str
    answer: Any = None
    termination_reason: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class CrewResult(BaseModel):
    crew: str
    process: str
    status: Literal["succeeded", "budget_exhausted", "max_rounds", "failed"]
    output: Any = None
    reports: list[CrewMemberReport] = Field(default_factory=list)
    delegations: list[DelegationRecord] = Field(default_factory=list)
    blackboard: dict[str, Any] = Field(default_factory=dict)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    rounds: int = 0

    def metrics(self) -> dict[str, Any]:
        """Crew eval metrics, aggregated from per-member agent metrics."""
        return {
            "success": self.status == "succeeded",
            "status": self.status,
            "members_run": len(self.reports),
            "members_succeeded": sum(1 for r in self.reports if r.metrics.get("success")),
            "delegations": len(self.delegations),
            "rounds": self.rounds,
            "cost_usd": self.usage.cost_usd,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "tool_calls": self.usage.tool_calls,
        }


class _Member(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    role: AgentRole
    executor: Any  # AgentExecutor


class Crew:
    def __init__(
        self,
        name: str = "crew",
        *,
        process: CrewProcess = "sequential",
        blackboard: Blackboard | None = None,
        tracer: Tracer | None = None,
        manager_provider: ModelProvider | None = None,
        manager_model: str | None = None,
        max_rounds: int = 4,
        concurrency: int = 4,
    ) -> None:
        self.name = name
        self.process = process
        self.blackboard = blackboard or Blackboard()
        self.tracer = tracer or Tracer()
        self.manager_provider = manager_provider
        self.manager_model = manager_model
        self.max_rounds = max_rounds
        self.concurrency = concurrency
        self._members: dict[str, _Member] = {}

    def add(
        self,
        role: AgentRole | str,
        executor: AgentExecutor,
        *,
        description: str = "",
        goal: str = "",
        keywords: list[str] | None = None,
        budget_fraction: float | None = None,
    ) -> Crew:
        """Register a member; chains for fluent construction."""
        if isinstance(role, str):
            role = AgentRole(
                name=role,
                description=description,
                goal=goal,
                keywords=keywords or [],
                budget_fraction=budget_fraction,
            )
        if role.name in self._members:
            raise AgentEngineError(f"duplicate crew member {role.name!r}")
        self._members[role.name] = _Member(role=role, executor=executor)
        return self

    @property
    def names(self) -> list[str]:
        return list(self._members)

    # -- member execution -------------------------------------------------------

    def _member_budget(self, role: AgentRole, budget: Budget) -> Budget:
        fraction = role.budget_fraction or (1.0 / max(1, len(self._members)))
        return budget.scaled(fraction)

    def _member_objective(self, role: AgentRole, task: str) -> str:
        parts = [task]
        if role.goal:
            parts.append(f"Your goal as {role.name}: {role.goal}")
        board = self.blackboard.as_context()
        if board:
            parts.append(f"Shared blackboard (findings from your team so far):\n{board}")
        return "\n\n".join(parts)

    async def _run_member(
        self, member: _Member, task: str, budget: Budget, usage: BudgetUsage
    ) -> CrewMemberReport:
        role = member.role
        with self.tracer.span(role.name, type="crew_agent") as span:
            span.set(crew=self.name, task=task[:300])
            state: AgentState = await member.executor.run(
                self._member_objective(role, task), budget=self._member_budget(role, budget)
            )
            span.set(termination_reason=state.termination_reason)
        usage.add(state.usage)
        answer = state.final_answer if state.final_answer is not None else state.raw_answer_text
        self.blackboard.post(role.name, answer, author=role.name, task=task)
        return CrewMemberReport(
            role=role.name,
            task=task,
            answer=answer,
            termination_reason=state.termination_reason,
            metrics=state.metrics(),
        )

    # -- manager (hierarchical) ---------------------------------------------------

    def _select(self, task: str) -> str:
        """Deterministic fallback: pick the member best matching the task."""
        from ..context.scoring import lexical_similarity

        text = task.lower()
        best_name, best_score = next(iter(self._members)), -1.0
        for name, member in self._members.items():
            role = member.role
            score = sum(1.0 for kw in role.keywords if kw.lower() in text)
            score += lexical_similarity(f"{role.description} {role.goal}", text)
            if score > best_score:
                best_name, best_score = name, score
        return best_name

    async def _manager_call(
        self, instruction: str, schema: dict[str, Any], usage: BudgetUsage
    ) -> dict[str, Any] | None:
        if self.manager_provider is None or self.manager_model is None:
            return None
        roster = "\n".join(
            f"- {m.role.name}: {m.role.description or m.role.goal}" for m in self._members.values()
        )
        board = self.blackboard.as_context() or "(empty)"
        request = ModelRequest(
            model=self.manager_model,
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are the manager of an agent crew. Delegate tasks to the "
                        "members best suited for them and decide when the objective is met. "
                        f"Members:\n{roster}"
                    ),
                ),
                Message(role="user", content=f"{instruction}\n\nBlackboard:\n{board}"),
            ],
            output_schema=schema,
            output_schema_name="crew_manager",
            temperature=0.0,
        )
        try:
            with self.tracer.span("manager", type="crew_agent") as span:
                span.set(crew=self.name)
                response = await self.manager_provider.generate(request)
            usage.input_tokens += response.usage.input_tokens
            usage.output_tokens += response.usage.output_tokens
            usage.cost_usd += response.cost_usd
            return response.structured or json.loads(response.text)
        except Exception:  # noqa: BLE001 - manager degrades to the heuristic
            return None

    def _valid_assignments(
        self, payload: dict[str, Any] | None, key: str
    ) -> list[tuple[str, str, str]]:
        if not payload:
            return []
        out: list[tuple[str, str, str]] = []
        for raw in payload.get(key, []):
            agent = raw.get("agent", "")
            if agent in self._members and raw.get("task"):
                out.append((agent, raw["task"], raw.get("reason", "")))
        return out

    # -- processes -------------------------------------------------------------------

    def _tasks_for(self, objective: str, tasks: list[str] | dict[str, str] | None) -> list[tuple[_Member, str]]:
        members = list(self._members.values())
        if tasks is None:
            return [(m, objective) for m in members]
        if isinstance(tasks, dict):
            unknown = set(tasks) - set(self._members)
            if unknown:
                raise AgentEngineError(f"tasks reference unknown members: {sorted(unknown)}")
            return [(self._members[name], task) for name, task in tasks.items()]
        if len(tasks) != len(members):
            raise AgentEngineError(
                f"got {len(tasks)} tasks for {len(members)} members; pass a dict to assign by name"
            )
        return list(zip(members, tasks, strict=True))

    async def arun(
        self,
        objective: Objective | str,
        *,
        tasks: list[str] | dict[str, str] | None = None,
        budget: Budget | None = None,
    ) -> CrewResult:
        """Run the crew on ``objective``; ``tasks`` optionally assigns per-member work."""
        if not self._members:
            raise AgentEngineError(f"crew {self.name!r} has no members")
        objective_text = objective.text if isinstance(objective, Objective) else objective
        budget = budget or Budget()
        usage = BudgetUsage()
        reports: list[CrewMemberReport] = []
        delegations: list[DelegationRecord] = []
        status: str = "succeeded"
        rounds = 0
        output: Any = None

        with self.tracer.span(self.name, type="crew") as span:
            span.set(process=self.process, members=len(self._members))
            if self.process == "parallel":
                pairs = self._tasks_for(objective_text, tasks)
                reports = await gather_bounded(
                    [self._run_member(m, t, budget, usage) for m, t in pairs],
                    limit=self.concurrency,
                )
                rounds = 1
                output = {r.role: r.answer for r in reports}
            elif self.process == "sequential":
                rounds = 1
                for member, task in self._tasks_for(objective_text, tasks):
                    if usage.exceeds(budget):
                        status = "budget_exhausted"
                        break
                    reports.append(await self._run_member(member, task, budget, usage))
                output = reports[-1].answer if reports else None
            else:  # hierarchical
                instruction = (
                    f"Objective: {objective_text}\n"
                    "Decompose it into assignments for your members (one per task)."
                )
                payload = await self._manager_call(instruction, _ASSIGN_SCHEMA, usage)
                assignments = self._valid_assignments(payload, "assignments") or [
                    (self._select(objective_text), objective_text, "best keyword/description match")
                ]
                while rounds < self.max_rounds:
                    rounds += 1
                    for agent, task, reason in assignments:
                        if usage.exceeds(budget):
                            status = "budget_exhausted"
                            break
                        delegations.append(
                            DelegationRecord(to_agent=agent, task=task, reason=reason, round=rounds)
                        )
                        reports.append(
                            await self._run_member(self._members[agent], task, budget, usage)
                        )
                    if status != "succeeded":
                        break
                    review = await self._manager_call(
                        f"Objective: {objective_text}\n"
                        "Review the blackboard. If the objective is met, set done=true and "
                        "write the final answer. Otherwise delegate follow_ups.",
                        _REVIEW_SCHEMA,
                        usage,
                    )
                    follow_ups = self._valid_assignments(review, "follow_ups")
                    if review is None or review.get("done") or not follow_ups:
                        output = (review or {}).get("final_answer") or (
                            reports[-1].answer if reports else None
                        )
                        break
                    assignments = follow_ups
                else:
                    status = "max_rounds"
                if output is None and reports:
                    output = reports[-1].answer
            if status == "succeeded" and not reports:
                status = "failed"
            span.set(status=status, rounds=rounds, members_run=len(reports))

        return CrewResult(
            crew=self.name,
            process=self.process,
            status=status,  # type: ignore[arg-type]
            output=output,
            reports=reports,
            delegations=delegations,
            blackboard=self.blackboard.snapshot(),
            usage=usage,
            rounds=rounds,
        )

    def run(
        self,
        objective: Objective | str,
        *,
        tasks: list[str] | dict[str, str] | None = None,
        budget: Budget | None = None,
    ) -> CrewResult:
        return run_sync(self.arun(objective, tasks=tasks, budget=budget))
