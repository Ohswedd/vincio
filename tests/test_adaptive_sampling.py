"""Tests for adaptive eval sampling."""

from __future__ import annotations

import random

import pytest

from vincio.evals import AdaptiveSampler, AdaptiveSamplingResult, CaseEstimate


class _Case:
    """A case with a known true mean/sd, used to drive a seeded sampler."""

    def __init__(self, id: str, mean: float, sd: float) -> None:
        self.id = id
        self.mean = mean
        self.sd = sd


def _sampler(seed: int):
    rng = random.Random(seed)

    def sample(case: _Case) -> float:
        return max(0.0, min(1.0, rng.gauss(case.mean, case.sd)))

    return sample


async def _full_run_verdict(cases: list[_Case], gate: str, seed: int, budget: int) -> str:
    """The verdict from spending the *entire* budget uniformly (no early stop):
    seed every case with budget/N samples each — the exhaustive baseline the
    adaptive run must agree with."""
    per_case = budget // len(cases)
    # With seed_samples = budget/N, the seed phase exhausts the budget, so this
    # is the uniform full-budget run.
    sampler = AdaptiveSampler(cases, _sampler(seed), gate=gate, budget=budget,
                              seed_samples=per_case)
    return (await sampler.run()).verdict


@pytest.mark.asyncio
async def test_clear_pass_decides_early():
    cases = [_Case(f"c{i}", 0.95, 0.02) for i in range(5)]
    result = await AdaptiveSampler(cases, _sampler(1), gate=">= 0.8", budget=200).run()
    assert isinstance(result, AdaptiveSamplingResult)
    assert result.verdict == "pass"
    assert result.decided is True
    assert result.samples_used < result.budget  # converged for less than the full budget
    assert result.savings > 0.0


@pytest.mark.asyncio
async def test_clear_fail_decides_early():
    cases = [_Case(f"c{i}", 0.4, 0.05) for i in range(5)]
    result = await AdaptiveSampler(cases, _sampler(2), gate=">= 0.8", budget=200).run()
    assert result.verdict == "fail"
    assert result.decided is True
    assert result.samples_used < result.budget


@pytest.mark.asyncio
async def test_budget_concentrates_on_the_high_variance_case():
    # Three near-deterministic high scorers and one very noisy low scorer keep the
    # CI straddling the threshold, so the budget must flow to the noisy case.
    cases = [
        _Case("a", 0.95, 0.01),
        _Case("b", 0.96, 0.01),
        _Case("noisy", 0.55, 0.4),
        _Case("c", 0.95, 0.01),
    ]
    result = await AdaptiveSampler(cases, _sampler(3), gate=">= 0.8", budget=400,
                                   seed_samples=2).run()
    noisy = result.allocations["noisy"]
    quiet = max(result.allocations[k] for k in result.allocations if k != "noisy")
    assert noisy > quiet  # variance-targeted allocation


@pytest.mark.asyncio
async def test_adaptive_preserves_the_exhaustive_verdict_for_less_cost():
    # The promise: same verdict as the full run, fewer samples. True mean ~0.9.
    cases = [
        _Case("a", 0.97, 0.02),
        _Case("b", 0.96, 0.03),
        _Case("c", 0.95, 0.02),
        _Case("noisy", 0.82, 0.2),
        _Case("d", 0.94, 0.04),
    ]
    budget = 250
    adaptive = await AdaptiveSampler(cases, _sampler(11), gate=">= 0.8", budget=budget).run()
    full_verdict = await _full_run_verdict(cases, ">= 0.8", seed=11, budget=budget)
    assert adaptive.decided is True
    assert adaptive.verdict == full_verdict == "pass"  # verdict preserved
    assert adaptive.samples_used < budget  # for strictly less cost


@pytest.mark.asyncio
async def test_le_gate_direction():
    # A "lower is better" style gate: hallucination rate <= 0.2, true mean ~0.05.
    cases = [_Case(f"c{i}", 0.05, 0.02) for i in range(4)]
    result = await AdaptiveSampler(cases, _sampler(5), gate="<= 0.2", budget=200).run()
    assert result.verdict == "pass"
    cases_bad = [_Case(f"c{i}", 0.6, 0.05) for i in range(4)]
    bad = await AdaptiveSampler(cases_bad, _sampler(6), gate="<= 0.2", budget=200).run()
    assert bad.verdict == "fail"


@pytest.mark.asyncio
async def test_deterministic_cases_decide_after_seed():
    # Zero-variance cases carry no uncertainty; the verdict is fixed after seeding
    # and no extra budget is spent.
    cases = [_Case(f"c{i}", 0.95, 0.0) for i in range(3)]
    result = await AdaptiveSampler(cases, _sampler(9), gate=">= 0.8", budget=100,
                                   seed_samples=2).run()
    assert result.verdict == "pass"
    assert result.samples_used == 6  # 2 seeds × 3 cases, nothing more


def test_case_estimate_welford():
    est = CaseEstimate(case_id="c")
    for v in (0.2, 0.4, 0.6):
        est.observe(v)
    assert est.n == 3
    assert est.mean == pytest.approx(0.4, abs=1e-9)
    assert est.variance == pytest.approx(0.04, abs=1e-9)  # sample variance of {.2,.4,.6}


def test_invalid_gate_rejected():
    with pytest.raises(ValueError):
        AdaptiveSampler([_Case("c", 0.5, 0.1)], _sampler(1), gate="== 0.5", budget=10)


def test_budget_below_seed_cost_rejected():
    cases = [_Case(f"c{i}", 0.5, 0.1) for i in range(5)]
    with pytest.raises(ValueError):
        AdaptiveSampler(cases, _sampler(1), gate=">= 0.5", budget=5, seed_samples=2)


def test_empty_cases_rejected():
    with pytest.raises(ValueError):
        AdaptiveSampler([], _sampler(1), gate=">= 0.5", budget=10)
