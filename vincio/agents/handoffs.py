"""Agent handoffs (agents/handoffs).

A :class:`HandoffRouter` holds named agents (executors) and routes an
objective to the right one; agents can hand work to each other with bounded
depth. Handoffs are recorded for tracing and agent evals.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import AgentEngineError
from ..core.types import Budget, Objective
from .executor import AgentExecutor
from .state import AgentState

__all__ = ["HandoffRecord", "HandoffRouter"]


class HandoffRecord(BaseModel):
    from_agent: str
    to_agent: str
    reason: str = ""
    objective: str = ""


class RegisteredAgent(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str
    executor: Any  # AgentExecutor
    keywords: list[str] = Field(default_factory=list)


class HandoffRouter:
    def __init__(self, *, max_handoffs: int = 3) -> None:
        self._agents: dict[str, RegisteredAgent] = {}
        self.max_handoffs = max_handoffs
        self.handoffs: list[HandoffRecord] = []

    def register(
        self,
        name: str,
        executor: AgentExecutor,
        *,
        description: str = "",
        keywords: list[str] | None = None,
    ) -> None:
        self._agents[name] = RegisteredAgent(
            name=name, description=description, executor=executor, keywords=keywords or []
        )

    @property
    def names(self) -> list[str]:
        return sorted(self._agents)

    def select(self, objective: Objective | str) -> str:
        """Pick the agent whose keywords/description best match the objective."""
        if not self._agents:
            raise AgentEngineError("no agents registered")
        text = (objective.text if isinstance(objective, Objective) else objective).lower()
        best_name, best_score = next(iter(self._agents)), -1.0
        for name, agent in self._agents.items():
            score = sum(1.0 for kw in agent.keywords if kw.lower() in text)
            from ..context.scoring import lexical_similarity

            score += lexical_similarity(agent.description, text)
            if score > best_score:
                best_name, best_score = name, score
        return best_name

    async def run(
        self,
        objective: Objective | str,
        *,
        agent: str | None = None,
        budget: Budget | None = None,
        _depth: int = 0,
    ) -> AgentState:
        if _depth > self.max_handoffs:
            raise AgentEngineError(f"handoff depth exceeded ({self.max_handoffs})")
        name = agent or self.select(objective)
        if name not in self._agents:
            raise AgentEngineError(f"unknown agent {name!r}; known: {self.names}")
        state = await self._agents[name].executor.run(objective, budget=budget)

        # An agent can request a handoff by setting working_memory["handoff_to"].
        target = state.working_memory.get("handoff_to")
        if isinstance(target, str) and target in self._agents and target != name:
            reason = str(state.working_memory.get("handoff_reason", ""))
            objective_text = (
                objective.text if isinstance(objective, Objective) else objective
            )
            self.handoffs.append(
                HandoffRecord(from_agent=name, to_agent=target, reason=reason, objective=objective_text)
            )
            follow_up = state.working_memory.get("handoff_objective") or objective_text
            handed = await self.run(
                str(follow_up), agent=target, budget=budget, _depth=_depth + 1
            )
            # Merge evidence/results so the final answer keeps full provenance.
            handed.evidence = state.evidence + [
                e for e in handed.evidence if e.id not in {x.id for x in state.evidence}
            ]
            handed.tool_results = state.tool_results + handed.tool_results
            return handed
        return state
