"""Self-proposed, bounded curriculum.

An :class:`AutoCurriculum` proposes the next task at the **frontier of current
competence** — a task the library cannot yet solve by retrieving an existing
skill, but that a bounded search *can* solve, so the agent is stretched without
being set an impossible objective. Every proposed objective is gated **before**
it is ever attempted: its instruction is screened by the programmable
:class:`~vincio.security.rails.RailEngine`, and the
:class:`~vincio.governance.verification.GovernanceVerifier` must prove the app's
controls (containment, residency, budget, erasure) still hold. A blocked or
out-of-policy objective is **pinpointed and refused**, never run — that refusal
is the stay-in-policy guarantee, recomputable from the proposal's bytes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import CultivationError
from ..core.utils import stable_hash
from ..evals.environment import EnvAction, Environment, StateCheck
from .search import SkillSearch
from .skill import LearnedSkillLibrary

__all__ = [
    "CurriculumTask",
    "FrontierEstimate",
    "CurriculumProposal",
    "AutoCurriculum",
]


class CurriculumTask(BaseModel):
    """A candidate objective: a deterministic environment plus its success oracle.

    The :attr:`environment` factory builds a fresh, deterministic
    :class:`~vincio.evals.environment.Environment` per rollout (it carries the
    task's own end-state oracle), and is excluded from the serialized identity so
    two tasks are equal by their stable fields. :attr:`action_space` bounds the
    primitive moves the searcher may try (defaulting to the argument-free tools
    the environment advertises), keeping search finite and reproducible.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    objective: str
    environment: Callable[[], Environment] | None = Field(default=None, exclude=True, repr=False)
    action_space: list[EnvAction] = Field(default_factory=list)
    precondition: list[StateCheck] = Field(default_factory=list)
    difficulty: float = 0.0
    max_steps: int | None = None
    keywords: list[str] = Field(default_factory=list)

    def make_env(self) -> Environment:
        if self.environment is None:
            raise CultivationError(
                f"curriculum task {self.id!r} has no environment factory to attempt"
            )
        return self.environment()

    def resolved_action_space(self, env: Environment) -> list[EnvAction]:
        """The declared action space, or argument-free tool actions from the env."""
        if self.action_space:
            return list(self.action_space)
        obs = env.observe()
        return [EnvAction(kind="tool", tool=tool) for tool in obs.available_tools]


class FrontierEstimate(BaseModel):
    """Where a task sits relative to current competence."""

    task_id: str
    competent: bool  # the library already solves it by applying a known skill
    reachable: bool  # a bounded search can solve it
    in_frontier: bool  # reachable and not yet competent — the learnable edge
    difficulty: float = 0.0
    model_confidence: float = 1.0  # world-model calibration weight, if supplied
    reason: str = ""


class CurriculumProposal(BaseModel):
    """A content-bound, offline-verifiable curriculum round.

    :attr:`proposed` are the frontier task ids the cultivator may attempt, in
    rank order; :attr:`refused` pinpoints every objective the rails or the
    governance verifier rejected (with the stage and reason). :meth:`verify`
    recomputes the proposal hash from the bytes, so a tampered proposal — an
    objective slipped into ``proposed`` after a refusal — is caught.
    """

    proposed: list[str] = Field(default_factory=list)
    refused: list[dict[str, Any]] = Field(default_factory=list)
    estimates: list[FrontierEstimate] = Field(default_factory=list)
    governance_held: bool = True
    proposal_hash: str = ""

    def _facts(self) -> dict[str, Any]:
        return {
            "proposed": self.proposed,
            "refused": [
                {"task_id": r.get("task_id"), "stage": r.get("stage"), "reason": r.get("reason")}
                for r in self.refused
            ],
            "governance_held": self.governance_held,
            "estimates": [e.model_dump(mode="json") for e in self.estimates],
        }

    def seal(self) -> CurriculumProposal:
        self.proposal_hash = stable_hash(self._facts(), length=32)
        return self

    def verify(self) -> bool:
        """Recompute the hash and re-check that no refused objective was proposed."""
        if self.proposal_hash != stable_hash(self._facts(), length=32):
            return False
        refused_ids = {r.get("task_id") for r in self.refused}
        return not (set(self.proposed) & refused_ids)

    @property
    def stayed_in_policy(self) -> bool:
        """No proposed task was refused — the property the safety SLO asserts."""
        return self.verify()


class AutoCurriculum:
    """Propose the next frontier tasks, gated by rails and the governance verifier."""

    def __init__(
        self,
        tasks: list[CurriculumTask],
        *,
        rails: Any | None = None,
        governance: Any | None = None,
        world_model: Any | None = None,
        search: SkillSearch | None = None,
        max_tasks: int = 4,
    ) -> None:
        self.tasks = list(tasks)
        self.rails = rails
        self.governance = governance
        self.world_model = world_model
        self.search = search or SkillSearch()
        self.max_tasks = max(1, max_tasks)

    def task(self, task_id: str) -> CurriculumTask | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    # -- gates --------------------------------------------------------------

    def _governance_holds(self, app: Any | None) -> tuple[bool, str]:
        verifier = self.governance
        try:
            if verifier is not None:
                report = verifier.verify()
            elif app is not None and hasattr(app, "verify_governance"):
                report = app.verify_governance(record=False)
            else:
                return True, "no governance verifier configured"
        except Exception as exc:  # noqa: BLE001 - a verifier failure fails safe, surfaced in the result
            return False, f"governance verification error: {exc}"
        held = bool(getattr(report, "held", False))
        return held, "governance invariants hold" if held else "governance invariants violated"

    def _rails_block(self, objective: str, app: Any | None) -> str | None:
        engine = self.rails
        if engine is None and app is not None:
            engine = getattr(app, "rail_engine", None)
        if engine is None:
            return None
        check = engine.check(objective, direction="input")
        if check.allowed:
            return None
        return "; ".join(v.message for v in check.violations) or "blocked by rails"

    # -- frontier estimation ------------------------------------------------

    def _estimate(self, task: CurriculumTask, library: LearnedSkillLibrary) -> FrontierEstimate:
        competent = self.search.solve_with_library(task, library).solved
        reachable = competent or self.search.solve(task, library).solved
        confidence = self._model_confidence(task)
        in_frontier = reachable and not competent
        if competent:
            reason = "already mastered by an existing skill"
        elif not reachable:
            reason = "beyond reach of bounded search — not yet learnable"
        else:
            reason = "at the frontier: solvable by search, not yet by the library"
        return FrontierEstimate(
            task_id=task.id,
            competent=competent,
            reachable=reachable,
            in_frontier=in_frontier,
            difficulty=task.difficulty,
            model_confidence=confidence,
            reason=reason,
        )

    def _model_confidence(self, task: CurriculumTask) -> float:
        model = self.world_model
        if model is None:
            return 1.0
        calibration = getattr(model, "calibration", None)
        weight = getattr(calibration, "weight", None)
        if weight is None and getattr(model, "trusted", False):
            weight = 1.0
        return float(weight) if weight is not None else 0.5

    # -- propose ------------------------------------------------------------

    def propose(
        self, library: LearnedSkillLibrary, *, app: Any | None = None
    ) -> CurriculumProposal:
        """Rank the frontier tasks that pass the rails and the governance gate."""
        held, gov_reason = self._governance_holds(app)
        refused: list[dict[str, Any]] = []
        estimates: list[FrontierEstimate] = []

        if not held:
            # The whole round is refused: the app is not in a safe state to grow.
            for task in self.tasks:
                refused.append({"task_id": task.id, "stage": "governance", "reason": gov_reason})
            return CurriculumProposal(refused=refused, governance_held=False).seal()

        admitted: list[FrontierEstimate] = []
        for task in self.tasks:
            blocked = self._rails_block(task.objective, app)
            if blocked is not None:
                refused.append({"task_id": task.id, "stage": "rails", "reason": blocked})
                continue
            est = self._estimate(task, library)
            estimates.append(est)
            if est.in_frontier:
                admitted.append(est)
            else:
                refused.append({"task_id": task.id, "stage": "frontier", "reason": est.reason})

        # Rank: most model-confidence first, then hardest, then deterministic by id.
        admitted.sort(key=lambda e: (-e.model_confidence, -e.difficulty, e.task_id))
        proposed = [e.task_id for e in admitted[: self.max_tasks]]
        return CurriculumProposal(
            proposed=proposed,
            refused=refused,
            estimates=estimates,
            governance_held=True,
        ).seal()
