"""Tests for judge ensembles with disagreement detection."""

from __future__ import annotations

import pytest

from vincio.evals import EnsembleVerdict, JudgeEnsemble, judge_disagreement
from vincio.evals.datasets import EvalCase
from vincio.evals.judges import Judge, ModelJudge
from vincio.evals.metrics import MetricResult, RunOutput
from vincio.providers import MockProvider


class _Fixed(Judge):
    """A deterministic judge returning a constant score."""

    def __init__(self, value: float, name: str) -> None:
        self.value = value
        self.name = name

    async def score(self, case: EvalCase, output: RunOutput) -> MetricResult:
        return MetricResult(name=self.name, value=self.value)


CASE = EvalCase(id="c", input="q", expected="a")
OUT = RunOutput(output="a")


def test_disagreement_zero_when_unanimous():
    assert judge_disagreement([0.8, 0.8, 0.8])["range"] == 0.0
    assert judge_disagreement([0.5])["range"] == 0.0  # single judge


def test_disagreement_scales_with_spread():
    spread = judge_disagreement([0.1, 0.9, 0.5])
    assert spread["range"] == 0.8
    assert spread["max_gap"] == 0.8
    assert spread["stdev"] > 0.0


@pytest.mark.asyncio
async def test_panel_agreement_is_not_uncertain():
    ensemble = JudgeEnsemble([_Fixed(0.9, "a"), _Fixed(0.92, "b"), _Fixed(0.88, "c")])
    verdict = await ensemble.averdict(CASE, OUT)
    assert isinstance(verdict, EnsembleVerdict)
    assert verdict.uncertain is False
    assert 0.88 <= verdict.value <= 0.92
    assert verdict.n_judges == 3


@pytest.mark.asyncio
async def test_panel_disagreement_flags_uncertain():
    ensemble = JudgeEnsemble(
        [_Fixed(0.1, "a"), _Fixed(0.9, "b"), _Fixed(0.5, "c")],
        disagreement_threshold=0.2,
    )
    verdict = await ensemble.averdict(CASE, OUT)
    assert verdict.uncertain is True
    assert verdict.spread == 0.8


@pytest.mark.asyncio
async def test_trimmed_mean_is_robust_to_an_outlier():
    # One rogue judge at 0.0; trimmed mean drops the extreme low and high.
    ensemble = JudgeEnsemble(
        [_Fixed(0.0, "rogue"), _Fixed(0.9, "b"), _Fixed(0.92, "c"), _Fixed(0.94, "d")],
        aggregate="trimmed_mean",
    )
    verdict = await ensemble.averdict(CASE, OUT)
    # 0.0 and 0.94 trimmed -> mean(0.9, 0.92) = 0.91, not dragged down by the rogue.
    assert verdict.value == pytest.approx(0.91, abs=1e-3)


@pytest.mark.asyncio
async def test_median_aggregate():
    ensemble = JudgeEnsemble([_Fixed(0.2, "a"), _Fixed(0.8, "b"), _Fixed(0.9, "c")],
                             aggregate="median")
    verdict = await ensemble.averdict(CASE, OUT)
    assert verdict.value == 0.8


@pytest.mark.asyncio
async def test_score_returns_metric_result_with_details():
    ensemble = JudgeEnsemble([_Fixed(0.7, "a"), _Fixed(0.9, "b")])
    result = await ensemble.score(CASE, OUT)
    assert result.name == "judge_ensemble"
    assert "disagreement" in result.details
    assert set(result.details["scores"]) == {"a", "b"}


def test_calibration_and_gating_weight():
    ensemble = JudgeEnsemble([_Fixed(0.7, "a"), _Fixed(0.9, "b")])
    # Uncalibrated panels cannot gate CI.
    assert ensemble.gating_weight() == 0.0
    fit = ensemble.calibrate(
        [(0.9, 1.0), (0.5, 0.6), (0.2, 0.1), (0.95, 0.9), (0.3, 0.25), (0.8, 0.85)]
    )
    assert fit["n"] == 6
    assert fit["cohens_kappa"] >= 0.6
    assert ensemble.kappa == fit["cohens_kappa"]
    assert ensemble.gating_weight(threshold=0.6) == 1.0


def test_low_agreement_panel_does_not_earn_gating_weight():
    ensemble = JudgeEnsemble([_Fixed(0.5, "a")])
    # Judge and human systematically disagree -> low κ -> no gating weight.
    ensemble.calibrate([(0.9, 0.1), (0.1, 0.9), (0.8, 0.2), (0.2, 0.8)])
    assert ensemble.gating_weight(threshold=0.6) == 0.0


def test_calibration_requires_two_pairs():
    ensemble = JudgeEnsemble([_Fixed(0.5, "a")])
    with pytest.raises(ValueError):
        ensemble.calibrate([(0.5, 0.5)])


def test_empty_panel_rejected():
    with pytest.raises(ValueError):
        JudgeEnsemble([])


def test_unknown_aggregate_rejected():
    with pytest.raises(ValueError):
        JudgeEnsemble([_Fixed(0.5, "a")], aggregate="geometric")


@pytest.mark.asyncio
async def test_weighted_panel_mean():
    ensemble = JudgeEnsemble([(_Fixed(0.0, "a"), 1.0), (_Fixed(1.0, "b"), 3.0)])
    verdict = await ensemble.averdict(CASE, OUT)
    # (0*1 + 1*3) / 4 = 0.75
    assert verdict.value == 0.75


@pytest.mark.asyncio
async def test_integration_with_model_judges():
    # A real panel of ModelJudges over a deterministic mock, scoring agreement.
    def responder(request):
        return {"score": 0.9, "reasoning": "good", "failures": []}

    provider = MockProvider(responder=responder)
    panel = JudgeEnsemble(
        [ModelJudge(provider, model="mock-1", name=f"j{i}") for i in range(3)]
    )
    verdict = await panel.averdict(CASE, OUT)
    assert verdict.uncertain is False
    assert verdict.value == pytest.approx(0.9, abs=1e-3)
