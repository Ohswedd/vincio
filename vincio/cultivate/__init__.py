"""Autonomous skill acquisition & open-ended curriculum.

The closed self-improvement loop (trace → dataset → eval → optimize → promote),
RLVR, and the distillation flywheel all make an agent *better at known tasks*.
This module is the apex of that arc: **open-ended capability growth**. An agent
proposes its own tasks at the edge of its competence, distills successful
trajectories into a reusable, versioned :class:`LearnedSkill` library, and
bootstraps — under the *same* no-regression gate a promotion already clears — so
growth is safe and reversible rather than unbounded drift.

* :class:`LearnedSkill` / :class:`LearnedSkillLibrary` — verified,
  content-addressed, versioned, composable procedures distilled from winning
  trajectories, retrieved like memory and tools, deduplicated and composed (a new
  skill can call existing ones), demoted when they stop paying their way.
* :class:`AutoCurriculum` — proposes the next task at the frontier of current
  competence, with the rails and the governance verifier **gating every proposed
  objective**, so it never proposes an unsafe or out-of-policy task.
* :class:`Cultivator` (``app.cultivate``) — runs propose → attempt (test-time
  search + the skill library) → verify → distill → promote, held by a
  capability-monotonicity property and a stay-in-policy guarantee.

Everything here is opt-in, additive, deterministic, and offline; it composes the
existing eval harness, optimizer gates, rails, and governance verifier rather
than introducing a new runtime.

    from vincio.cultivate import AutoCurriculum, CurriculumTask, LearnedSkillLibrary
    result = app.cultivate(AutoCurriculum(tasks), library=LearnedSkillLibrary())
"""

from __future__ import annotations

from typing import Any

from .cultivator import CultivationResult, Cultivator, CycleReport
from .curriculum import AutoCurriculum, CurriculumProposal, CurriculumTask, FrontierEstimate
from .search import SkillSearch, Solution, library_capability
from .skill import LearnedSkill, LearnedSkillLibrary, SkillProvenance, SkillStep

__all__ = [
    "LearnedSkill",
    "LearnedSkillLibrary",
    "SkillProvenance",
    "SkillStep",
    "CurriculumTask",
    "FrontierEstimate",
    "CurriculumProposal",
    "AutoCurriculum",
    "SkillSearch",
    "Solution",
    "library_capability",
    "Cultivator",
    "CultivationResult",
    "CycleReport",
    "cultivate",
]


def cultivate(
    curriculum: AutoCurriculum | list[CurriculumTask],
    *,
    app: Any | None = None,
    library: LearnedSkillLibrary | None = None,
    held_out: list[CurriculumTask] | None = None,
    cycles: int = 3,
    rails: Any | None = None,
    governance: Any | None = None,
    search: SkillSearch | None = None,
    min_capability_gain: float = 0.0,
    prune: bool = True,
    record: bool = True,
) -> CultivationResult:
    """Run the open-ended cultivation loop and return its sealed result.

    The free-function form of :meth:`vincio.ContextApp.cultivate` for use without
    an app (the rails / governance gates are then only applied if passed
    explicitly). See :class:`Cultivator` for the full parameter semantics.
    """
    cultivator = Cultivator(
        app,
        curriculum=curriculum,
        library=library,
        held_out=held_out,
        rails=rails,
        governance=governance,
        search=search,
        min_capability_gain=min_capability_gain,
        prune=prune,
        record=record,
    )
    return cultivator.run(cycles=cycles)
