"""The open-ended cultivation loop: propose → attempt → verify → distill → promote.

``app.cultivate`` runs this loop across cycles so capability **compounds** safely.
Each cycle asks the :class:`~vincio.cultivate.AutoCurriculum` for the frontier
tasks that pass the rails and the governance gate, *attempts* each with the
library-composing :class:`~vincio.cultivate.SkillSearch`, *verifies* the result
against the task-success oracle, *distills* a winning trajectory into a
:class:`~vincio.cultivate.LearnedSkill`, and *promotes* it only through the same
**no-regression gate** a prompt deploy clears — capability on a held-out frontier
set must not fall. A skill that no longer pays its way is demoted, never silently
kept. The whole run is content-bound: :meth:`CultivationResult.verify` re-derives
the monotonicity and stay-in-policy verdicts from the bytes alone.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import CultivationError
from ..core.utils import slugify, stable_hash
from ..optimize.self_improvement import CanaryVerdict
from ..optimize.trajectory_opt import no_regression_gate
from .curriculum import AutoCurriculum, CurriculumProposal, CurriculumTask
from .search import SkillSearch, Solution, library_capability
from .skill import LearnedSkill, LearnedSkillLibrary, SkillProvenance

__all__ = ["CycleReport", "CultivationResult", "Cultivator"]


class CycleReport(BaseModel):
    """What one cultivation cycle proposed, learned, promoted, and demoted."""

    cycle: str
    proposal: CurriculumProposal
    attempted: list[str] = Field(default_factory=list)
    solved: list[str] = Field(default_factory=list)
    promoted: list[str] = Field(default_factory=list)
    refused_promotions: list[dict[str, Any]] = Field(default_factory=list)
    demoted: list[dict[str, Any]] = Field(default_factory=list)
    capability_before: float = 0.0
    capability_after: float = 0.0
    verdict: CanaryVerdict = Field(default_factory=lambda: CanaryVerdict(passed=True))

    @property
    def monotonic(self) -> bool:
        return self.verdict.passed


class CultivationResult(BaseModel):
    """The content-bound, offline-verifiable outcome of a cultivation run."""

    cycles: list[CycleReport] = Field(default_factory=list)
    skills_promoted: int = 0
    skills_refused: int = 0
    tasks_refused: int = 0
    capability_before: float = 0.0
    capability_after: float = 0.0
    monotonic: bool = True
    stayed_in_policy: bool = True
    result_hash: str = ""
    library: Any = Field(default=None, exclude=True, repr=False)

    def _facts(self) -> dict[str, Any]:
        return {
            "cycles": [
                {
                    "cycle": c.cycle,
                    "proposed": c.proposal.proposed,
                    "promoted": c.promoted,
                    "demoted": [d.get("name") for d in c.demoted],
                    "capability_before": round(c.capability_before, 6),
                    "capability_after": round(c.capability_after, 6),
                    "monotonic": c.monotonic,
                    "stayed_in_policy": c.proposal.stayed_in_policy,
                }
                for c in self.cycles
            ],
            "capability_before": round(self.capability_before, 6),
            "capability_after": round(self.capability_after, 6),
        }

    def seal(self) -> CultivationResult:
        self.result_hash = stable_hash(self._facts(), length=32)
        return self

    def verify(self) -> bool:
        """Recompute the hash and re-derive monotonicity and stay-in-policy.

        Catches a tampered run — a capability number edited upward, a refused
        objective relabelled as promoted — from the bytes alone.
        """
        if self.result_hash != stable_hash(self._facts(), length=32):
            return False
        monotonic = self.capability_after >= self.capability_before - 1e-9 and all(
            c.monotonic for c in self.cycles
        )
        stayed = all(c.proposal.stayed_in_policy for c in self.cycles)
        return monotonic == self.monotonic and stayed == self.stayed_in_policy


class Cultivator:
    """Drive the cultivation loop over a :class:`LearnedSkillLibrary`."""

    def __init__(
        self,
        app: Any | None = None,
        *,
        curriculum: AutoCurriculum | list[CurriculumTask],
        library: LearnedSkillLibrary | None = None,
        held_out: list[CurriculumTask] | None = None,
        rails: Any | None = None,
        governance: Any | None = None,
        search: SkillSearch | None = None,
        min_capability_gain: float = 0.0,
        tolerance: float = 1e-9,
        prune: bool = True,
        record: bool = True,
    ) -> None:
        self.app = app
        self.search = search or SkillSearch()
        if isinstance(curriculum, AutoCurriculum):
            self.curriculum = curriculum
        else:
            self.curriculum = AutoCurriculum(
                list(curriculum), rails=rails, governance=governance, search=self.search
            )
        self.library = library if library is not None else LearnedSkillLibrary()
        # Capability is measured over a held-out set that includes the frontier
        # tasks; defaulting to the curriculum's own tasks makes a learned skill's
        # contribution observable (it solves at least its own task).
        self.held_out = list(held_out) if held_out is not None else list(self.curriculum.tasks)
        self.min_capability_gain = max(0.0, min_capability_gain)
        self.tolerance = tolerance
        self.prune = prune
        self.record = record

    # -- distillation -------------------------------------------------------

    def _distill(self, task: CurriculumTask, solution: Solution, cycle: str) -> LearnedSkill:
        env = task.make_env()
        verifier = list(getattr(env.task, "checks", []))
        keywords = task.keywords or sorted(
            {t for t in slugify(task.objective).split("-") if len(t) > 2}
        )
        # Name by the objective so the skill is descriptive and retrievable; fall
        # back to the task id when the objective has no slug-able content.
        slug = slugify(task.objective)
        name = f"skill-{slug if slug != 'item' else slugify(task.id)}"
        return LearnedSkill(
            name=name,
            description=f"Procedure for: {task.objective}",
            objective=task.objective,
            precondition=list(task.precondition),
            steps=list(solution.steps),
            verifier=verifier,
            keywords=keywords,
            provenance=SkillProvenance(
                cycle=cycle,
                objective=task.objective,
                reward=solution.reward,
                attempts=1,
                verified=True,
                winning_steps=solution.n_steps,
                trajectory_hash=stable_hash(
                    [
                        {"tool": a.tool, "kind": a.kind, "args": a.arguments}
                        for a in solution.actions
                    ],
                    length=16,
                ),
                source_app=getattr(self.app, "name", ""),
            ),
        ).seal()

    # -- gates --------------------------------------------------------------

    def _promote(self, skill: LearnedSkill) -> CanaryVerdict:
        cap_before = library_capability(self.library, self.held_out, search=self.search)
        trial = self.library.clone()
        trial.add(skill.model_copy(deep=True))
        cap_after = library_capability(trial, self.held_out, search=self.search)
        passed, reason = no_regression_gate(
            cap_before,
            cap_after,
            0.0,
            kl_max=float("inf"),
            min_improvement=self.min_capability_gain,
            tol=self.tolerance,
        )
        return CanaryVerdict(
            passed=passed,
            metric="capability",
            baseline=round(cap_before, 6),
            candidate=round(cap_after, 6),
            delta=round(cap_after - cap_before, 6),
            samples=len(self.held_out),
            reason=reason,
        )

    def _prune(self) -> list[dict[str, Any]]:
        """Demote any active skill whose removal does not lower capability."""
        demoted: list[dict[str, Any]] = []
        base = library_capability(self.library, self.held_out, search=self.search)
        for name in list(self.library.names):
            dependents = [
                s.name for s in self.library.skills if s.name != name and name in s.requires
            ]
            if dependents:
                continue
            trial = self.library.clone()
            try:
                trial.demote(name)
            except CultivationError:
                continue
            cap = library_capability(trial, self.held_out, search=self.search)
            if cap >= base - self.tolerance:
                self.library.demote(name, reason="zero marginal capability")
                demoted.append(
                    {"name": name, "reason": f"zero marginal capability ({cap:.4f} ≥ {base:.4f})"}
                )
        return demoted

    # -- the loop -----------------------------------------------------------

    def _cycle(self, index: int) -> CycleReport:
        cycle_id = f"cycle-{index}"
        cap_before = library_capability(self.library, self.held_out, search=self.search)
        proposal = self.curriculum.propose(self.library, app=self.app)

        attempted: list[str] = []
        solved: list[str] = []
        promoted: list[str] = []
        refused_promotions: list[dict[str, Any]] = []

        for task_id in proposal.proposed:
            task = self.curriculum.task(task_id)
            if task is None:
                continue
            attempted.append(task_id)
            solution = self.search.solve(task, self.library)  # attempt (+ skill reuse)
            if (
                not solution.solved
                or solution.verification is None
                or not solution.verification.passed
            ):
                continue  # verify failed — no skill is distilled from an unverified attempt
            solved.append(task_id)
            skill = self._distill(task, solution, cycle_id)  # distill
            verdict = self._promote(skill)  # promote through the no-regression gate
            if verdict.passed:
                skill.provenance.promotion = verdict.model_dump(mode="json")
                stored = self.library.add(skill)
                promoted.append(stored.name)
            else:
                refused_promotions.append(
                    {
                        "task_id": task_id,
                        "skill": skill.name,
                        "reason": verdict.reason,
                        "delta": verdict.delta,
                    }
                )

        demoted = self._prune() if self.prune else []
        cap_after = library_capability(self.library, self.held_out, search=self.search)
        passed, reason = no_regression_gate(
            cap_before, cap_after, 0.0, kl_max=float("inf"), min_improvement=0.0, tol=self.tolerance
        )
        verdict = CanaryVerdict(
            passed=passed,
            metric="capability",
            baseline=round(cap_before, 6),
            candidate=round(cap_after, 6),
            delta=round(cap_after - cap_before, 6),
            samples=len(self.held_out),
            reason=reason,
        )
        return CycleReport(
            cycle=cycle_id,
            proposal=proposal,
            attempted=attempted,
            solved=solved,
            promoted=promoted,
            refused_promotions=refused_promotions,
            demoted=demoted,
            capability_before=round(cap_before, 6),
            capability_after=round(cap_after, 6),
            verdict=verdict,
        )

    def run(self, *, cycles: int = 3) -> CultivationResult:
        """Run the open-ended loop for ``cycles`` cycles and seal the result."""
        if cycles < 1:
            raise CultivationError("cultivate requires at least one cycle")
        capability_before = library_capability(self.library, self.held_out, search=self.search)
        reports: list[CycleReport] = []
        for index in range(cycles):
            reports.append(self._cycle(index))
        capability_after = library_capability(self.library, self.held_out, search=self.search)

        monotonic = capability_after >= capability_before - self.tolerance and all(
            r.monotonic for r in reports
        )
        stayed = all(r.proposal.stayed_in_policy for r in reports)
        result = CultivationResult(
            cycles=reports,
            skills_promoted=sum(len(r.promoted) for r in reports),
            skills_refused=sum(len(r.refused_promotions) for r in reports),
            tasks_refused=sum(len(r.proposal.refused) for r in reports),
            capability_before=round(capability_before, 6),
            capability_after=round(capability_after, 6),
            monotonic=monotonic,
            stayed_in_policy=stayed,
            library=self.library,
        ).seal()
        self._record(result)
        return result

    def _record(self, result: CultivationResult) -> None:
        app = self.app
        if not self.record or app is None:
            return
        audit = getattr(app, "audit", None)
        if audit is not None:
            audit.record(
                "skill_cultivation",
                decision="allow" if (result.monotonic and result.stayed_in_policy) else "deny",
                resource=getattr(app, "name", ""),
                details={
                    "cycles": len(result.cycles),
                    "skills_promoted": result.skills_promoted,
                    "skills_refused": result.skills_refused,
                    "tasks_refused": result.tasks_refused,
                    "capability_before": result.capability_before,
                    "capability_after": result.capability_after,
                    "monotonic": result.monotonic,
                    "stayed_in_policy": result.stayed_in_policy,
                    "library_hash": self.library.library_hash,
                    "result_hash": result.result_hash,
                },
            )
        events = getattr(app, "events", None)
        if events is not None:
            events.emit(
                "cultivation.completed",
                {
                    "skills_promoted": result.skills_promoted,
                    "capability_delta": round(
                        result.capability_after - result.capability_before, 6
                    ),
                    "monotonic": result.monotonic,
                    "stayed_in_policy": result.stayed_in_policy,
                },
            )
