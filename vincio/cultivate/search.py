"""Deterministic, offline test-time search over an environment.

The *attempt* step of ``app.cultivate`` searches for a procedure that solves a
curriculum task, **composing the existing skill library** as macro-actions: a
bounded beam search over the task's primitive action space plus one macro per
active learned skill, scored by the environment's own reward and the
task-success oracle's end-state verdict. Everything is deterministic (no
randomness, no network) so a cultivation run is reproducible and CI-golden.

The same machinery measures **capability**: :func:`library_capability` is the
fraction of a held-out task set the library can already solve *by retrieving and
applying an existing skill* — the monotonic quantity the no-regression promotion
gate protects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..evals.environment import (
    EnvAction,
    EnvironmentSimulator,
    TaskVerification,
    scripted_policy,
)
from ..evals.trajectory import Trajectory
from .skill import LearnedSkillLibrary, SkillStep

__all__ = ["Solution", "SkillSearch", "library_capability"]


class Solution(BaseModel):
    """The outcome of searching for (or retrieving) a procedure for a task."""

    solved: bool = False
    steps: list[SkillStep] = Field(default_factory=list)
    actions: list[EnvAction] = Field(default_factory=list)  # flattened primitives
    used_skills: list[str] = Field(default_factory=list)
    n_steps: int = 0
    reward: float = 0.0
    verification: TaskVerification | None = None
    trajectory: Trajectory | None = None


def _flatten(steps: list[SkillStep], library: LearnedSkillLibrary) -> list[EnvAction]:
    actions: list[EnvAction] = []
    for step in steps:
        if step.action is not None:
            actions.append(step.action)
        elif step.skill is not None:
            actions.extend(library.compose(step.skill))
    return actions


class SkillSearch:
    """Bounded, deterministic beam search that composes the skill library."""

    def __init__(self, *, beam_width: int = 4, max_depth: int = 8) -> None:
        self.beam_width = max(1, beam_width)
        self.max_depth = max(1, max_depth)

    # -- low-level rollout --------------------------------------------------

    def _rollout(
        self, task: CurriculumTask, steps: list[SkillStep], library: LearnedSkillLibrary
    ) -> tuple[TaskVerification, float, list[EnvAction]]:
        actions = _flatten(steps, library)
        env = task.make_env()
        env.reset()
        reward = 0.0
        for action in actions:
            result = env.step(action)
            reward += result.reward
            if result.done:
                break
        return env.verify(), reward, actions

    def _finalize(
        self,
        task: CurriculumTask,
        steps: list[SkillStep],
        actions: list[EnvAction],
        verification: TaskVerification,
        reward: float,
        library: LearnedSkillLibrary,
    ) -> Solution:
        env = task.make_env()
        trajectory = EnvironmentSimulator().run(env, scripted_policy(list(actions)))
        return Solution(
            solved=verification.passed,
            steps=steps,
            actions=actions,
            used_skills=[s.skill for s in steps if s.skill is not None],
            n_steps=len(actions),
            reward=round(reward, 6),
            verification=verification,
            trajectory=trajectory.trajectory,
        )

    # -- search -------------------------------------------------------------

    def solve(self, task: CurriculumTask, library: LearnedSkillLibrary) -> Solution:
        """Search for the shortest procedure that satisfies the task oracle.

        Moves are the task's primitive actions plus one macro per active learned
        skill, so a solution may *call existing skills*. Returns the first passing
        procedure at the smallest depth (ties broken deterministically by move
        order), or an unsolved :class:`Solution` if none is found within the bound.
        """
        env0 = task.make_env()
        env0.reset()
        space = task.resolved_action_space(env0)
        moves: list[SkillStep] = [SkillStep(action=a) for a in space]
        moves += [SkillStep(skill=name) for name in library.names]

        # The empty procedure may already satisfy the oracle.
        ver, reward, actions = self._rollout(task, [], library)
        if ver.passed:
            return self._finalize(task, [], actions, ver, reward, library)
        if not moves:
            return Solution(solved=False, verification=ver)

        depth_limit = min(task.max_steps or self.max_depth, self.max_depth)
        beam: list[list[SkillStep]] = [[]]
        for _ in range(depth_limit):
            scored: list[tuple[list[SkillStep], float, int]] = []
            seen: set[tuple[Any, ...]] = set()
            for path in beam:
                for move in moves:
                    candidate = [*path, move]
                    ver, reward, actions = self._rollout(task, candidate, library)
                    key = tuple(_action_key(a) for a in actions)
                    if key in seen:
                        continue
                    seen.add(key)
                    if ver.passed:
                        return self._finalize(task, candidate, actions, ver, reward, library)
                    scored.append((candidate, reward, len(actions)))
            if not scored:
                break
            scored.sort(key=lambda t: (-t[1], t[2]))
            beam = [path for path, _, _ in scored[: self.beam_width]]
        return Solution(solved=False)

    def solve_with_library(self, task: CurriculumTask, library: LearnedSkillLibrary) -> Solution:
        """Solve a task by **retrieving and applying** an existing skill — no search.

        This is the competence/capability primitive: it asks whether the library
        *already* knows how to solve the task. It tries the empty procedure, then
        each active skill (most-relevant first) applied as the whole procedure, and
        returns the first that satisfies the oracle.
        """
        ver, reward, actions = self._rollout(task, [], library)
        if ver.passed:
            return self._finalize(task, [], actions, ver, reward, library)
        for skill, _score in library.relevant(
            task.objective, threshold=0.0, limit=max(1, len(library))
        ):
            steps = [SkillStep(skill=skill.name)]
            ver, reward, actions = self._rollout(task, steps, library)
            if ver.passed:
                return self._finalize(task, steps, actions, ver, reward, library)
        return Solution(solved=False)


def _action_key(action: EnvAction) -> tuple[Any, ...]:
    return (action.kind, action.tool, tuple(sorted(action.arguments.items())), action.text)


def library_capability(
    library: LearnedSkillLibrary,
    tasks: list[CurriculumTask],
    *,
    search: SkillSearch | None = None,
) -> float:
    """Fraction of *tasks* the library solves by applying an existing skill.

    The monotonic quantity the promotion gate protects: adding a verified skill can
    only solve more tasks, and a demotion is refused when it would lower this.
    """
    if not tasks:
        return 1.0
    searcher = search or SkillSearch()
    solved = sum(1 for task in tasks if searcher.solve_with_library(task, library).solved)
    return solved / len(tasks)


if TYPE_CHECKING:  # forward references for type checkers only — avoids an import cycle
    from .curriculum import CurriculumTask
