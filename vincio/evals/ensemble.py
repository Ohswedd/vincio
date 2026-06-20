"""Judge ensembles with disagreement detection.

A single LLM judge is a point estimate with unknown error. A **panel** of judges
turns that point estimate into a distribution: the panel's aggregate is the
score, and the *spread* across judges is a first-class **uncertainty signal**.
When the panel agrees, the score is trustworthy; when it disagrees, the case is
flagged uncertain so a CI gate can route it to human review instead of acting on
a coin-flip.

The ensemble is held to the same standard as any single judge that gates CI: it
earns gating weight only once :meth:`JudgeEnsemble.calibrate` shows it agrees
with human labels above a **Cohen's κ** threshold (the same κ-validated bar the
:class:`~vincio.evals.annotation.AnnotationQueue` and :class:`GEvalJudge` use).
So a panel can disagree internally *and* still be calibrated as a whole — the two
are independent: disagreement measures within-panel spread on one case;
calibration measures the panel-vs-human agreement across a labeled set.

Everything is offline and deterministic against the mock provider; the panel
runs its judges concurrently.
"""

from __future__ import annotations

import asyncio
import statistics

from pydantic import BaseModel, Field

from .datasets import EvalCase
from .judges import Judge
from .metrics import MetricResult, RunOutput

__all__ = [
    "judge_disagreement",
    "EnsembleVerdict",
    "JudgeEnsemble",
]


def judge_disagreement(scores: list[float]) -> dict[str, float]:
    """Quantify how much a panel of judges disagrees on one case.

    Returns the population ``stdev``, the ``range`` (max − min), the mean
    absolute deviation (``mad``), and the largest pairwise gap (``max_gap``) —
    all in the judges' [0, 1] score units. With fewer than two scores there is
    no disagreement to measure, so every figure is ``0.0``.
    """
    if len(scores) < 2:
        return {"stdev": 0.0, "range": 0.0, "mad": 0.0, "max_gap": 0.0}
    mean = sum(scores) / len(scores)
    span = max(scores) - min(scores)
    return {
        "stdev": round(statistics.pstdev(scores), 4),
        "range": round(span, 4),
        "mad": round(sum(abs(s - mean) for s in scores) / len(scores), 4),
        "max_gap": round(span, 4),
    }


class EnsembleVerdict(BaseModel):
    """The panel's verdict on one case: the aggregate score, the per-judge
    scores, and the disagreement signal that drives the ``uncertain`` flag."""

    value: float
    aggregate: str = "mean"
    scores: dict[str, float] = Field(default_factory=dict)
    disagreement: dict[str, float] = Field(default_factory=dict)
    uncertain: bool = False
    n_judges: int = 0
    calibrated: bool = False

    @property
    def spread(self) -> float:
        """The panel's score range (max − min) — the headline disagreement."""
        return self.disagreement.get("range", 0.0)


class JudgeEnsemble(Judge):
    """A panel of judges scored together, with disagreement surfaced as
    uncertainty and the panel as a whole calibrated against human labels.

    Each judge in ``judges`` scores the case; the panel aggregates them by
    ``aggregate`` — ``"mean"``, ``"median"``, or ``"trimmed_mean"`` (drop the
    extreme high and low before averaging, robust to one rogue judge). The score
    spread is measured by :func:`judge_disagreement`; when it exceeds
    ``disagreement_threshold`` the verdict is flagged ``uncertain`` so a gate can
    abstain rather than act on a split panel.

    :meth:`calibrate` fits a linear correction from ``(ensemble, human)`` pairs
    and records the Cohen's κ between the panel and human raters;
    :meth:`gating_weight` returns ``1.0`` only once that κ clears its threshold,
    so an uncalibrated panel cannot gate CI on its own — exactly the bar a single
    :class:`GEvalJudge` is held to.
    """

    def __init__(
        self,
        judges: list[Judge] | list[tuple[Judge, float]],
        *,
        aggregate: str = "mean",
        disagreement_threshold: float = 0.2,
        name: str = "judge_ensemble",
    ) -> None:
        if not judges:
            raise ValueError("JudgeEnsemble requires at least one judge")
        if aggregate not in ("mean", "median", "trimmed_mean"):
            raise ValueError(f"unknown aggregate {aggregate!r}; expected mean | median | trimmed_mean")
        # Accept bare judges or (judge, weight) pairs; weights apply to the mean.
        self.judges: list[Judge] = []
        self.weights: list[float] = []
        for entry in judges:
            if isinstance(entry, tuple):
                judge, weight = entry
            else:
                judge, weight = entry, 1.0
            self.judges.append(judge)
            self.weights.append(float(weight))
        self.aggregate = aggregate
        self.disagreement_threshold = disagreement_threshold
        self.name = name
        self._calibration: tuple[float, float] | None = None  # (scale, offset)
        self._kappa: float | None = None

    def _combine(self, scores: list[float]) -> float:
        if self.aggregate == "median":
            return statistics.median(scores)
        if self.aggregate == "trimmed_mean" and len(scores) >= 3:
            ordered = sorted(scores)[1:-1]  # drop one extreme each side
            return sum(ordered) / len(ordered)
        total_weight = sum(self.weights) or 1.0
        return sum(s * w for s, w in zip(scores, self.weights, strict=True)) / total_weight

    async def averdict(self, case: EvalCase, output: RunOutput) -> EnsembleVerdict:
        """Run the full panel and return the rich :class:`EnsembleVerdict`."""
        results = await asyncio.gather(*(judge.score(case, output) for judge in self.judges))
        scores: dict[str, float] = {}
        for index, (judge, result) in enumerate(zip(self.judges, results, strict=True)):
            # Disambiguate identically-named judges in the panel.
            label = judge.name if judge.name not in scores else f"{judge.name}#{index}"
            scores[label] = result.value
        raw_values = [r.value for r in results]
        value = self._combine(raw_values)
        if self._calibration is not None:
            scale, offset = self._calibration
            value = max(0.0, min(1.0, scale * value + offset))
        disagreement = judge_disagreement(raw_values)
        return EnsembleVerdict(
            value=round(value, 4),
            aggregate=self.aggregate,
            scores={k: round(v, 4) for k, v in scores.items()},
            disagreement=disagreement,
            uncertain=disagreement["range"] > self.disagreement_threshold,
            n_judges=len(self.judges),
            calibrated=self._calibration is not None,
        )

    async def score(self, case: EvalCase, output: RunOutput) -> MetricResult:
        verdict = await self.averdict(case, output)
        return MetricResult(
            name=self.name,
            value=verdict.value,
            details={
                "scores": verdict.scores,
                "disagreement": verdict.disagreement,
                "uncertain": verdict.uncertain,
                "aggregate": verdict.aggregate,
                "n_judges": verdict.n_judges,
                "calibrated": verdict.calibrated,
            },
        )

    def calibrate(self, pairs: list[tuple[float, float]], *, kappa_bins: int = 2) -> dict[str, float]:
        """Fit ``human ≈ scale · ensemble + offset`` from ``(ensemble, human)``
        pairs and record the panel-vs-human Cohen's κ.

        Stores the linear correction (applied to future verdicts) and the κ, and
        returns the fit (scale, offset, Pearson r), the κ, and ``n``. The κ is
        what :meth:`gating_weight` consults to decide whether the panel has earned
        CI-gating weight — the panel is trusted to gate exactly when it has
        demonstrably agreed with people.
        """
        if len(pairs) < 2:
            raise ValueError("calibration requires at least 2 (ensemble, human) pairs")
        xs = [float(e) for e, _ in pairs]
        ys = [float(h) for _, h in pairs]
        n = len(pairs)
        mean_x, mean_y = sum(xs) / n, sum(ys) / n
        var_x = sum((x - mean_x) ** 2 for x in xs)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        scale = cov / var_x if var_x else 1.0
        offset = mean_y - scale * mean_x
        var_y = sum((y - mean_y) ** 2 for y in ys)
        pearson = cov / ((var_x * var_y) ** 0.5) if var_x and var_y else 0.0
        self._calibration = (scale, offset)
        from .annotation import cohens_kappa

        self._kappa = cohens_kappa(pairs, bins=kappa_bins)
        return {
            "scale": round(scale, 4), "offset": round(offset, 4),
            "pearson_r": round(pearson, 4), "cohens_kappa": self._kappa, "n": n,
        }

    def gating_weight(self, *, threshold: float = 0.6) -> float:
        """``1.0`` once the panel's calibrated κ clears ``threshold``, else
        ``0.0`` — so an uncalibrated or low-agreement panel cannot gate CI."""
        if self._kappa is None:
            return 0.0
        return 1.0 if self._kappa >= threshold else 0.0

    @property
    def kappa(self) -> float | None:
        """The panel-vs-human Cohen's κ from the last :meth:`calibrate`, if any."""
        return self._kappa
