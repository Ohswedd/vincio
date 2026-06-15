"""Optimizer-judge calibration (1.4): closing the loop on the loop.

The optimizer trusts an LLM judge to score candidates. 1.2 made that judge earn
its gating weight by clearing a Cohen's-κ bar against human labels. 1.4 goes one
step further: it **tunes the judge's own evaluation procedure** — reflectively
proposing alternative evaluation steps — and adopts the procedure that best
agrees with people, gated on a real κ gain. The judge that gates the optimizer
is itself optimized, against κ-validated human labels.

Deterministic and gated: candidate step-sets are proposed in a fixed order,
each is scored against the same labelled samples, and a new procedure is adopted
only when its κ strictly improves on the incumbent's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..evals.annotation import cohens_kappa
from ..evals.datasets import EvalCase
from ..evals.metrics import RunOutput
from ..providers.base import run_sync

if TYPE_CHECKING:
    from ..evals.judges import GEvalJudge

__all__ = [
    "JudgeStepProposal",
    "JudgeStepReflector",
    "JudgeCalibrationResult",
    "JudgeCalibrator",
]

# A labelled sample: the case, the system output that was judged, and the human
# score for it (on the judge's 0..1 scale).
LabeledSample = tuple[EvalCase, RunOutput, float]


class JudgeStepProposal(BaseModel):
    name: str
    steps: list[str]
    rationale: str = ""


class JudgeStepReflector:
    """Deterministic proposer of alternative judge evaluation procedures.

    Each proposal is the incumbent's steps plus one targeted check that a low
    judge↔human agreement suggests is missing — grounding, numeric accuracy, or
    strictness. Offline and reproducible; subclass to plug in a model-backed
    proposer.
    """

    EXTRA_STEPS: list[tuple[str, str]] = [
        (
            "grounding",
            "Penalize any claim in the output that the provided context does not support.",
        ),
        (
            "accuracy",
            "Check every number, name, and entity in the output against the reference answer.",
        ),
        (
            "strictness",
            "Award the top score only when the output fully and correctly satisfies the criteria; "
            "deduct for partial or padded answers.",
        ),
        (
            "completeness",
            "Verify the output addresses every part of the task, not just the easiest part.",
        ),
    ]

    def propose(self, baseline_steps: list[str], *, budget: int) -> list[JudgeStepProposal]:
        proposals: list[JudgeStepProposal] = []
        for name, step in self.EXTRA_STEPS:
            if any(step == existing for existing in baseline_steps):
                continue
            proposals.append(
                JudgeStepProposal(
                    name=name,
                    steps=[*baseline_steps, step],
                    rationale=f"add a {name} check",
                )
            )
            if len(proposals) >= budget:
                break
        return proposals


class JudgeCalibrationResult(BaseModel):
    adopted: bool = False
    reason: str = ""
    kappa_before: float = 0.0
    kappa_after: float = 0.0
    gating_weight_before: float = 0.0
    gating_weight_after: float = 0.0
    steps_before: list[str] = Field(default_factory=list)
    steps_after: list[str] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)


class JudgeCalibrator:
    """Tune a :class:`~vincio.evals.judges.GEvalJudge`'s evaluation steps to
    maximize agreement with human labels, then leave the judge calibrated.

    The calibrator scores the judge over labelled samples with each proposed
    step-set, computes Cohen's κ against the human scores, and installs the
    best-agreeing procedure on the judge — but only when it strictly beats the
    incumbent. On adoption the judge's remembered κ (and thus its
    :meth:`~vincio.evals.judges.GEvalJudge.gating_weight`) reflects the new,
    higher agreement.
    """

    def __init__(
        self,
        judge: GEvalJudge,
        *,
        reflector: JudgeStepReflector | None = None,
        kappa_bins: int = 2,
        trust_threshold: float = 0.6,
        min_kappa_gain: float = 1e-9,
    ) -> None:
        self.judge = judge
        self.reflector = reflector or JudgeStepReflector()
        self.kappa_bins = kappa_bins
        self.trust_threshold = trust_threshold
        self.min_kappa_gain = min_kappa_gain

    async def _kappa_for_steps(self, steps: list[str], samples: list[LabeledSample]) -> float:
        original = self.judge.steps
        self.judge.steps = list(steps)
        try:
            pairs: list[tuple[float, float]] = []
            for case, output, human in samples:
                result = await self.judge.score(case, output)
                pairs.append((float(result.value), float(human)))
        finally:
            self.judge.steps = original
        if len(pairs) < 2:
            return 0.0
        return cohens_kappa(pairs, bins=self.kappa_bins)

    async def acalibrate(
        self, samples: list[LabeledSample], *, budget: int = 4
    ) -> JudgeCalibrationResult:
        if len(samples) < 2:
            raise ValueError("judge calibration requires at least 2 labelled samples")
        baseline_steps = await self.judge._ensure_steps()
        baseline_steps = list(baseline_steps)
        kappa_before = await self._kappa_for_steps(baseline_steps, samples)

        result = JudgeCalibrationResult(
            kappa_before=round(kappa_before, 4),
            steps_before=baseline_steps,
            steps_after=baseline_steps,
            kappa_after=round(kappa_before, 4),
            gating_weight_before=1.0 if kappa_before >= self.trust_threshold else 0.0,
        )

        best_steps = baseline_steps
        best_kappa = kappa_before
        for proposal in self.reflector.propose(baseline_steps, budget=budget):
            kappa = await self._kappa_for_steps(proposal.steps, samples)
            result.history.append(
                {"name": proposal.name, "kappa": round(kappa, 4), "rationale": proposal.rationale}
            )
            if kappa > best_kappa + self.min_kappa_gain:
                best_kappa = kappa
                best_steps = proposal.steps

        result.kappa_after = round(best_kappa, 4)
        result.gating_weight_after = 1.0 if best_kappa >= self.trust_threshold else 0.0
        if best_steps is not baseline_steps and best_kappa > kappa_before + self.min_kappa_gain:
            self.judge.steps = list(best_steps)
            self.judge._kappa = best_kappa
            result.adopted = True
            result.steps_after = list(best_steps)
            result.reason = (
                f"adopted judge procedure (+{best_kappa - kappa_before:.3f} κ → {best_kappa:.3f}); "
                f"gating weight {result.gating_weight_before:.0f} → {result.gating_weight_after:.0f}"
            )
        else:
            # Keep the incumbent but remember its agreement for gating.
            self.judge._kappa = kappa_before
            result.reason = f"no procedure beat the incumbent (κ {kappa_before:.3f}); kept it"
        return result

    def calibrate(self, samples: list[LabeledSample], *, budget: int = 4) -> JudgeCalibrationResult:
        return run_sync(self.acalibrate(samples, budget=budget))
