"""Adaptive eval sampling: spend the budget where the variance is.

A CI gate over a noisy metric (an LLM judge, a sampled-temperature run) usually
samples every case the same number of times. That wastes budget: cases whose
score is far from the gate threshold, or barely varies, are already decided after
one look, while a handful of high-variance cases near the threshold decide the
whole verdict. **Adaptive sampling** concentrates the budget on exactly those
cases.

The sampler estimates the gate aggregate (a mean) and its confidence interval,
then allocates each next sample to the case that most reduces the aggregate's
variance — the Neyman-optimal rule, here run sequentially: pull the case with the
largest ``s_i² / (n_i·(n_i+1))``. It **stops early** the moment the confidence
interval clears the threshold on one side, because the verdict can no longer
change. The promise it keeps is the one the docs gate: *the same verdict as the
exhaustive run, for fewer samples* — never a different verdict, just a cheaper
path to it.

The sampler is provider-agnostic: you supply a ``sample(case) -> float`` callable
(sync or async) that draws one noisy observation; the allocation logic is fully
deterministic given that callable, so a seeded sampler makes the whole run
reproducible.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "CaseEstimate",
    "AdaptiveSamplingResult",
    "AdaptiveSampler",
]

# A sampler draws one noisy observation of a case's metric. May be sync or async.
Sample = Callable[[Any], "float | Awaitable[float]"]

_GATE_RE = re.compile(r"^\s*(>=|<=|>|<)\s*([\d.]+)\s*$")
# Confidence → two-sided normal z. A small table keeps the module dependency-free.
_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.975: 2.2414, 0.99: 2.5758}


class CaseEstimate(BaseModel):
    """The running estimate for one case: its sample count, mean, and variance."""

    case_id: str
    n: int = 0
    mean: float = 0.0
    variance: float = 0.0

    # Welford accumulators (not serialized as semantic state, just bookkeeping).
    m2: float = Field(default=0.0, exclude=True)

    def observe(self, value: float) -> None:
        """Fold one observation into the running mean/variance (Welford)."""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.variance = self.m2 / (self.n - 1) if self.n > 1 else 0.0

    def marginal_reduction(self) -> float:
        """How much one more sample would cut this case's variance term in the
        aggregate — ``s² / (n·(n+1))``. Zero variance ⇒ nothing to gain."""
        if self.n < 1:
            return float("inf")
        return self.variance / (self.n * (self.n + 1))


class AdaptiveSamplingResult(BaseModel):
    """The outcome of an adaptive run: the estimate, its CI, the verdict, and how
    the budget was spent across cases."""

    metric: str
    expression: str
    estimate: float
    ci_low: float
    ci_high: float
    verdict: str  # "pass" | "fail" | "uncertain"
    decided: bool  # the CI cleared the threshold on one side
    samples_used: int
    budget: int
    n_cases: int
    confidence: float
    allocations: dict[str, int] = Field(default_factory=dict)
    estimates: list[CaseEstimate] = Field(default_factory=list)

    @property
    def savings(self) -> float:
        """Fraction of an exhaustive (full-budget) run this verdict cost."""
        return round(1.0 - self.samples_used / self.budget, 4) if self.budget else 0.0


def _decision(op: str, threshold: float, ci_low: float, ci_high: float) -> str | None:
    """The decided verdict if the CI clears ``threshold`` on one side, else None."""
    if op in (">=", ">"):
        if ci_low >= threshold:
            return "pass"
        if ci_high < threshold:
            return "fail"
    else:  # "<=", "<"
        if ci_high <= threshold:
            return "pass"
        if ci_low > threshold:
            return "fail"
    return None


class AdaptiveSampler:
    """Decide a mean-aggregate gate with the fewest samples by allocating the
    budget to the highest-variance cases and stopping as soon as the verdict is
    certain.

    ``cases`` is any list of case objects; ``sample(case)`` draws one noisy
    observation of the metric. ``gate`` is a mean-gate expression
    (``">= 0.9"``). Each case is seeded with ``seed_samples`` observations (so its
    variance is estimable), then the remaining ``budget`` is allocated greedily by
    variance reduction; the run halts early once the ``confidence`` interval for
    the aggregate clears the threshold.
    """

    def __init__(
        self,
        cases: list[Any],
        sample: Sample,
        *,
        gate: str,
        metric: str = "score",
        budget: int,
        seed_samples: int = 2,
        confidence: float = 0.95,
        weights: dict[str, float] | None = None,
    ) -> None:
        if not cases:
            raise ValueError("AdaptiveSampler requires at least one case")
        match = _GATE_RE.match(gate)
        if not match:
            raise ValueError(
                f"invalid gate expression {gate!r}; adaptive sampling supports mean "
                f"gates of the form '>= 0.9' / '<= 0.2' / '>' / '<'"
            )
        self.cases = cases
        self.sample = sample
        self.metric = metric
        self.op = match.group(1)
        self.threshold = float(match.group(2))
        self.gate = gate
        self.seed_samples = max(2, seed_samples)
        self.confidence = confidence
        self.z = _Z.get(round(confidence, 3), 1.96)
        self.weights = weights or {}
        self._ids = [str(getattr(c, "id", index)) for index, c in enumerate(cases)]
        n_seed = self.seed_samples * len(cases)
        if budget < n_seed:
            raise ValueError(
                f"budget {budget} is below the seed cost {n_seed} "
                f"({self.seed_samples} samples × {len(cases)} cases)"
            )
        self.budget = budget

    async def _draw(self, case: Any) -> float:
        value = self.sample(case)
        if hasattr(value, "__await__"):
            value = await value  # type: ignore[assignment]
        return float(value)

    def _aggregate(self, estimates: list[CaseEstimate]) -> tuple[float, float]:
        """Weighted aggregate mean and its standard error."""
        weights = [self.weights.get(cid, 1.0) for cid in self._ids]
        total_w = sum(weights) or 1.0
        estimate = sum(w * e.mean for w, e in zip(weights, estimates, strict=True)) / total_w
        # Var(Σ wᵢ μ̂ᵢ / Σw) = Σ wᵢ² · sᵢ²/nᵢ / (Σw)²  (independent cases).
        var = sum(
            (w / total_w) ** 2 * (e.variance / e.n if e.n else 0.0)
            for w, e in zip(weights, estimates, strict=True)
        )
        return estimate, var**0.5

    async def run(self) -> AdaptiveSamplingResult:
        """Run the adaptive allocation and return the verdict."""
        estimates = [CaseEstimate(case_id=cid) for cid in self._ids]
        by_id = dict(zip(self._ids, estimates, strict=True))
        used = 0

        # Seed: every case gets ``seed_samples`` observations so its variance is
        # estimable before any case is starved of budget.
        for _ in range(self.seed_samples):
            draws = await asyncio.gather(*(self._draw(case) for case in self.cases))
            for cid, value in zip(self._ids, draws, strict=True):
                by_id[cid].observe(value)
            used += len(self.cases)

        estimate, se = self._aggregate(estimates)
        decided = _decision(self.op, self.threshold, estimate - self.z * se, estimate + self.z * se)

        # Allocate the remaining budget greedily by variance reduction, stopping
        # the instant the verdict is certain.
        while decided is None and used < self.budget:
            target = max(estimates, key=lambda e: e.marginal_reduction())
            if target.marginal_reduction() == 0.0:
                # No case carries any variance — more sampling cannot move the CI.
                break
            value = await self._draw(self.cases[self._ids.index(target.case_id)])
            target.observe(value)
            used += 1
            estimate, se = self._aggregate(estimates)
            decided = _decision(self.op, self.threshold, estimate - self.z * se, estimate + self.z * se)

        estimate, se = self._aggregate(estimates)
        ci_low, ci_high = estimate - self.z * se, estimate + self.z * se
        verdict = decided or "uncertain"
        return AdaptiveSamplingResult(
            metric=self.metric,
            expression=self.gate,
            estimate=round(estimate, 6),
            ci_low=round(ci_low, 6),
            ci_high=round(ci_high, 6),
            verdict=verdict,
            decided=decided is not None,
            samples_used=used,
            budget=self.budget,
            n_cases=len(self.cases),
            confidence=self.confidence,
            allocations={e.case_id: e.n for e in estimates},
            estimates=[e.model_copy() for e in estimates],
        )
