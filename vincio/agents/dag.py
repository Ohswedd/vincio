"""Bounded step DAG: validated, acyclic, dynamically extensible."""

from __future__ import annotations

from collections import defaultdict, deque

from pydantic import BaseModel, Field

from ..core.errors import AgentEngineError
from .state import AgentStep

__all__ = ["StepDAG"]


class StepDAG(BaseModel):
    steps: dict[str, AgentStep] = Field(default_factory=dict)
    edges: dict[str, list[str]] = Field(default_factory=dict)  # step id -> dependents

    def add(self, step: AgentStep, *, depends_on: list[str] | None = None) -> AgentStep:
        if step.id in self.steps:
            raise AgentEngineError(f"duplicate step id {step.id}")
        for dep in depends_on or []:
            if dep not in self.steps:
                raise AgentEngineError(f"unknown dependency {dep!r} for step {step.name or step.id!r}")
        self.steps[step.id] = step
        step.input_refs = list(depends_on or [])
        self.edges.setdefault(step.id, [])
        for dep in step.input_refs:
            self.edges.setdefault(dep, []).append(step.id)
        self._check_acyclic()
        return step

    def _check_acyclic(self) -> None:
        in_degree = {step_id: 0 for step_id in self.steps}
        for targets in self.edges.values():
            for target in targets:
                in_degree[target] += 1
        queue = deque([s for s, d in in_degree.items() if d == 0])
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for target in self.edges.get(current, []):
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)
        if visited != len(self.steps):
            raise AgentEngineError("step graph contains a cycle")

    def topological_levels(self) -> list[list[AgentStep]]:
        """Steps grouped into parallel-executable levels."""
        in_degree: dict[str, int] = defaultdict(int)
        for targets in self.edges.values():
            for target in targets:
                in_degree[target] += 1
        current = [s for s in self.steps.values() if in_degree[s.id] == 0]
        levels: list[list[AgentStep]] = []
        seen: set[str] = set()
        while current:
            levels.append(current)
            seen.update(s.id for s in current)
            next_level: list[AgentStep] = []
            for step in current:
                for target_id in self.edges.get(step.id, []):
                    in_degree[target_id] -= 1
                    if in_degree[target_id] == 0 and target_id not in seen:
                        next_level.append(self.steps[target_id])
            current = next_level
        return levels

    def ready_steps(self) -> list[AgentStep]:
        """Pending steps whose dependencies are all done."""
        ready = []
        for step in self.steps.values():
            if step.status != "pending":
                continue
            deps = [self.steps[ref] for ref in step.input_refs if ref in self.steps]
            if all(d.status == "done" for d in deps):
                ready.append(step)
            elif any(d.status in ("failed", "skipped") for d in deps):
                step.status = "skipped"
                step.error = "upstream step failed"
        return ready

    @property
    def complete(self) -> bool:
        return all(s.status in ("done", "failed", "skipped") for s in self.steps.values())
