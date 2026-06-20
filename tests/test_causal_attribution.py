"""Tests for causal regression attribution via counterfactual replay."""

from __future__ import annotations

import pytest

from vincio import ContextApp
from vincio.evals import (
    AttributionFactor,
    CausalAttributor,
    attribute_regression,
)
from vincio.evals.datasets import Dataset, EvalCase
from vincio.prompts.templates import PromptSpec
from vincio.providers import MockProvider

ANSWER = "The capital of France is Paris."


def _dataset(n: int = 5) -> Dataset:
    return Dataset(
        name="capitals",
        cases=[
            EvalCase(id=f"c{i}", input="What is the capital of France?", expected=ANSWER)
            for i in range(n)
        ],
    )


def _app(responder) -> ContextApp:
    return ContextApp(name="attr", provider=MockProvider(responder=responder), model="gpt-5.2")


@pytest.mark.asyncio
async def test_attributes_regression_to_the_model_swap():
    # Only the model swap breaks the answer; an inert second factor must get ~0.
    def responder(request):
        return ANSWER if request.model == "gpt-5.2" else "totally unrelated text"

    app = _app(responder)
    factors = [
        AttributionFactor.model("model", baseline="gpt-5.2", candidate="gpt-5.2-nano"),
        AttributionFactor.attr("inert", "name", baseline="attr", candidate="attr-b"),
    ]
    report = await CausalAttributor(app, _dataset(), factors=factors, metric="lexical_overlap").attribute()

    assert report.regressed is True
    assert report.total_delta < 0
    assert report.dominant_factor == "model"
    assert report.explained is True  # Shapley efficiency: contributions sum to the delta
    contributions = {c.factor: c.contribution for c in report.contributions}
    assert contributions["model"] < -0.5  # the model owns the regression
    assert abs(contributions["inert"]) < 1e-6  # the inert factor owns none
    assert report.concentration == 1.0
    assert report.coalitions == 4  # 2**2 coalitions


@pytest.mark.asyncio
async def test_app_state_is_restored_after_attribution():
    def responder(request):
        return ANSWER if request.model == "gpt-5.2" else "wrong"

    app = _app(responder)
    factors = [AttributionFactor.model("model", baseline="gpt-5.2", candidate="gpt-5.2-nano")]
    await CausalAttributor(app, _dataset(3), factors=factors, metric="lexical_overlap").attribute()
    assert app.model == "gpt-5.2"  # the attributor leaves the app pristine


@pytest.mark.asyncio
async def test_shapley_splits_an_interaction_between_two_factors():
    # The answer is correct ONLY when both factors are at baseline; flipping
    # either one to candidate breaks it. The regression is a shared interaction,
    # so Shapley should split the blame roughly evenly (efficiency still holds).
    def responder(request):
        ok = request.model == "gpt-5.2" and "strict" in (request.messages[0].content or "")
        return ANSWER if ok else "wrong"

    app = _app(responder)
    factors = [
        AttributionFactor.model("model", baseline="gpt-5.2", candidate="gpt-5.2-nano"),
        AttributionFactor.prompt(
            "prompt",
            baseline=PromptSpec(name="p", role="assistant", objective="answer strictly"),
            candidate=PromptSpec(name="p", role="assistant", objective="answer loosely"),
        ),
    ]
    report = await CausalAttributor(app, _dataset(4), factors=factors, metric="lexical_overlap").attribute()
    assert report.regressed is True
    assert report.explained is True
    contribs = {c.factor: c.contribution for c in report.contributions}
    # Both factors share the blame; neither is a clean zero.
    assert contribs["model"] < 0 and contribs["prompt"] < 0


@pytest.mark.asyncio
async def test_attribute_regression_convenience():
    def responder(request):
        return ANSWER if request.model == "gpt-5.2" else "wrong"

    app = _app(responder)
    report = await attribute_regression(
        app, _dataset(3),
        [AttributionFactor.model("model", baseline="gpt-5.2", candidate="gpt-5.2-nano")],
        metric="lexical_overlap",
    )
    assert report.dominant_factor == "model"
    assert report.summary()["regressed"] is True


def test_requires_factors():
    app = _app(lambda r: ANSWER)
    with pytest.raises(ValueError):
        CausalAttributor(app, _dataset(), factors=[], metric="lexical_overlap")


def test_duplicate_factor_names_rejected():
    app = _app(lambda r: ANSWER)
    with pytest.raises(ValueError):
        CausalAttributor(
            app, _dataset(),
            factors=[
                AttributionFactor.model("dup", baseline="a", candidate="b"),
                AttributionFactor.model("dup", baseline="c", candidate="d"),
            ],
            metric="lexical_overlap",
        )
