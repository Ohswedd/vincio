"""Causal regression attribution by counterfactual replay.

A regression gate that only reports *that* a score dropped leaves the hard
question open: **which change caused it?** A release usually moves several things
at once — the prompt, the retriever, the model, the budget — and the headline
delta is their tangled sum. This module untangles it by **counterfactual
replay**: it re-evaluates the dataset under every combination of baseline and
candidate components, then assigns each component its **Shapley value** — its
average marginal contribution to the metric delta across all orderings.

Shapley attribution is the principled choice here, not one-at-a-time ablation:
it is the unique credit assignment that is *efficient* (the contributions sum
exactly to the total delta — every point of the regression is accounted for),
*symmetric* (two components with identical effect get identical credit), and
handles **interactions** (a regression that only appears when the new prompt
meets the new retriever is shared fairly between them, not double-counted). With
``k`` components the replay runs ``2**k`` evaluations — small and offline for the
handful of components a release changes.

The result names the dominant cause and how concentrated the blame is, so a
failing gate produces *"the retrieval change owns 0.18 of the 0.21 drop"* rather
than a bare red ✗. Everything runs against the deterministic mock provider.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

from pydantic import BaseModel, Field

from ..core.shapley import shapley_from_cache
from .metrics import LOWER_IS_BETTER
from .reports import GateSpec

__all__ = [
    "AttributionFactor",
    "FactorContribution",
    "AttributionReport",
    "CausalAttributor",
    "attribute_regression",
]

# A factor applier mutates the app in place to install one variant of one
# component (e.g. set ``app.model``, swap ``app.prompt_spec``, rebind a
# retriever, tighten a budget).
Apply = Callable[[Any], None]


class AttributionFactor(BaseModel):
    """One changed component, with the appliers that install its baseline and
    candidate variants on the app.

    Build one directly with two appliers, or via the conveniences
    :meth:`model` (swap ``app.model``), :meth:`prompt` (swap ``app.prompt_spec``),
    and :meth:`attr` (set any app attribute). Each applier takes the app and
    mutates it in place; the attributor always installs one variant of *every*
    factor before each evaluation, so a coalition is fully specified and never
    leaks state between replays.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str
    baseline: Apply
    candidate: Apply

    @classmethod
    def model(cls, name: str, *, baseline: str, candidate: str) -> AttributionFactor:
        """A model swap: install ``baseline`` / ``candidate`` on ``app.model``."""
        def set_model(value: str) -> Apply:
            def apply(app: Any) -> None:
                app.model = value
            return apply

        return cls(name=name, baseline=set_model(baseline), candidate=set_model(candidate))

    @classmethod
    def prompt(cls, name: str, *, baseline: Any, candidate: Any) -> AttributionFactor:
        """A prompt swap: install ``baseline`` / ``candidate`` on
        ``app.prompt_spec``."""
        def set_prompt(value: Any) -> Apply:
            def apply(app: Any) -> None:
                app.prompt_spec = value
            return apply

        return cls(name=name, baseline=set_prompt(baseline), candidate=set_prompt(candidate))

    @classmethod
    def attr(cls, name: str, attribute: str, *, baseline: Any, candidate: Any) -> AttributionFactor:
        """A generic component swap: install ``baseline`` / ``candidate`` on
        ``app.<attribute>`` (e.g. a retriever, a budget, a tool runtime)."""
        def set_attr(value: Any) -> Apply:
            def apply(app: Any) -> None:
                setattr(app, attribute, value)
            return apply

        return cls(name=name, baseline=set_attr(baseline), candidate=set_attr(candidate))


class FactorContribution(BaseModel):
    """One factor's signed contribution to the metric delta (its Shapley value)."""

    factor: str
    contribution: float  # signed, in the metric's own units (candidate − baseline)
    share: float  # |contribution| / Σ|contributions|
    regressive: bool  # whether this contribution worsens the metric


class AttributionReport(BaseModel):
    """The counterfactual attribution of a metric delta to its components."""

    metric: str
    aggregate: str = "mean"
    baseline_value: float
    candidate_value: float
    total_delta: float  # candidate − baseline
    regressed: bool  # whether the total delta is a regression for this metric
    contributions: list[FactorContribution] = Field(default_factory=list)
    dominant_factor: str | None = None
    concentration: float = 0.0  # top |contribution| / Σ|contributions|
    explained: bool = True  # Σ contributions reconstructs total_delta (efficiency)
    coalitions: int = 0

    def summary(self) -> dict[str, Any]:
        """A compact, human-facing summary of the attribution."""
        return {
            "metric": self.metric,
            "total_delta": self.total_delta,
            "regressed": self.regressed,
            "dominant_factor": self.dominant_factor,
            "concentration": self.concentration,
            "contributions": {c.factor: c.contribution for c in self.contributions},
        }


def _is_regression(metric: str, delta: float) -> bool:
    """Whether a candidate-minus-baseline ``delta`` worsens ``metric``."""
    if abs(delta) < 1e-12:
        return False
    return delta > 0 if metric in LOWER_IS_BETTER else delta < 0


class CausalAttributor:
    """Attribute a metric delta to the components a release changed, by Shapley
    counterfactual replay over the dataset.

    Assemble from a :class:`~vincio.core.app.ContextApp` (or any object exposing
    ``eval_target``) and the list of changed :class:`AttributionFactor`s, then
    call :meth:`attribute`. Each of the ``2**k`` coalitions installs one variant
    of every factor and evaluates ``metric`` on the dataset; the per-coalition
    means feed an exact Shapley decomposition whose contributions sum to the
    total delta.
    """

    def __init__(
        self,
        app: Any,
        dataset: Any,
        *,
        factors: list[AttributionFactor],
        metric: str = "lexical_overlap",
        aggregate: str = "mean",
        repeats: int = 1,
        concurrency: int = 8,
    ) -> None:
        if not factors:
            raise ValueError("CausalAttributor requires at least one factor")
        names = [f.name for f in factors]
        if len(set(names)) != len(names):
            raise ValueError(f"factor names must be unique; got {names}")
        self.app = app
        self.dataset = dataset
        self.factors = factors
        self.metric = metric
        self.aggregate = aggregate
        self.repeats = max(1, repeats)
        self.concurrency = max(1, concurrency)

    async def _evaluate(self, coalition: frozenset[str]) -> float:
        """Mean ``metric`` over the dataset with the factors in ``coalition`` set
        to their candidate variant and the rest to baseline."""
        from .runners import EvalRunner

        for factor in self.factors:
            applier = factor.candidate if factor.name in coalition else factor.baseline
            applier(self.app)
        runner = EvalRunner(
            self.app, metrics=[self.metric], repeats=self.repeats, concurrency=self.concurrency
        )
        report = await runner.arun(self.dataset, name=f"attribution[{','.join(sorted(coalition))}]")
        values = report.metric_values(self.metric)
        if not values:
            raise ValueError(f"metric {self.metric!r} produced no values during attribution")
        _, value = GateSpec(metric=self.metric, expression=">= 0", aggregate=self.aggregate).check(values)
        return value

    async def attribute(self) -> AttributionReport:
        """Run the counterfactual replay and return the Shapley attribution."""
        names = [f.name for f in self.factors]
        k = len(names)

        # Save/restore the app's headline knobs so attribution leaves it pristine.
        saved_model = getattr(self.app, "model", None)
        had_prompt = hasattr(self.app, "prompt_spec")
        saved_prompt = getattr(self.app, "prompt_spec", None)

        cache: dict[frozenset[str], float] = {}
        try:
            # Evaluate every coalition once (memoized by membership).
            for size in range(k + 1):
                for subset in combinations(names, size):
                    key = frozenset(subset)
                    if key not in cache:
                        cache[key] = await self._evaluate(key)
        finally:
            # Leave the app pristine: reset every factor to its baseline variant
            # (so arbitrary attribute factors are restored), then re-pin the exact
            # model / prompt the app entered with.
            for factor in self.factors:
                factor.baseline(self.app)
            if saved_model is not None:
                self.app.model = saved_model
            if had_prompt:
                self.app.prompt_spec = saved_prompt

        baseline_value = cache[frozenset()]
        candidate_value = cache[frozenset(names)]
        total_delta = candidate_value - baseline_value

        # Exact Shapley value per factor — the shared credit-assignment kernel:
        # each factor's average marginal contribution over all coalitions of the
        # others, weighted by coalition size. Built from the coalition cache this
        # replay already populated.
        contributions = shapley_from_cache(names, cache)

        total_abs = sum(abs(v) for v in contributions.values()) or 1.0
        rows = [
            FactorContribution(
                factor=name,
                contribution=round(value, 6),
                share=round(abs(value) / total_abs, 4),
                regressive=_is_regression(self.metric, value),
            )
            for name, value in contributions.items()
        ]
        # Rank by how much each factor worsens the metric (most regressive first).
        sign = 1.0 if self.metric in LOWER_IS_BETTER else -1.0
        rows.sort(key=lambda r: sign * r.contribution, reverse=True)

        regressive_rows = [r for r in rows if r.regressive]
        dominant = regressive_rows[0].factor if regressive_rows else (rows[0].factor if rows else None)
        concentration = max((r.share for r in rows), default=0.0)
        reconstructed = sum(contributions.values())

        return AttributionReport(
            metric=self.metric,
            aggregate=self.aggregate,
            baseline_value=round(baseline_value, 6),
            candidate_value=round(candidate_value, 6),
            total_delta=round(total_delta, 6),
            regressed=_is_regression(self.metric, total_delta),
            contributions=rows,
            dominant_factor=dominant,
            concentration=round(concentration, 4),
            explained=abs(reconstructed - total_delta) < 1e-6,
            coalitions=len(cache),
        )


async def attribute_regression(
    app: Any,
    dataset: Any,
    factors: list[AttributionFactor],
    *,
    metric: str = "lexical_overlap",
    aggregate: str = "mean",
    repeats: int = 1,
) -> AttributionReport:
    """Attribute a metric regression to the changed ``factors`` by Shapley
    counterfactual replay, the convenience entry point behind a failing gate."""
    return await CausalAttributor(
        app, dataset, factors=factors, metric=metric, aggregate=aggregate, repeats=repeats
    ).attribute()
