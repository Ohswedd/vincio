"""The evaluation & quality frontier: measure more, explain regressions, spend less.

Four capabilities, all offline on the deterministic mock:

  1. More benchmark adapters: AgentBench, ToolBench, LiveCodeBench, and MMLU-Pro
     join SWE-bench / τ-bench / GAIA / WebArena / BFCL behind one BenchmarkAdapter
     contract — nine in all — each scored by its own *verifiable* scorer and
     pinned by a task-set hash.
  2. Judge ensembles with disagreement detection: a panel of judges scored
     together, where the spread across judges is surfaced as an uncertainty
     signal, and the panel earns CI-gating weight only once its Cohen's κ against
     human labels clears the bar.
  3. Causal regression attribution: when a gate regresses, Shapley counterfactual
     replay attributes the drop to the component that caused it (prompt /
     retrieval / model / budget) instead of merely reporting the score.
  4. Adaptive eval sampling: spend the eval budget where the variance is, reaching
     the same gate verdict as the exhaustive run for far fewer samples.

Everything here is opt-in and additive; nothing below is required to run Vincio.
"""

from __future__ import annotations

import asyncio
import random

from _shared import example_provider

from vincio import ContextApp
from vincio.evals import (
    AdaptiveSampler,
    AgentBenchAdapter,
    AttributionFactor,
    BenchmarkTask,
    CausalAttributor,
    JudgeEnsemble,
    LiveCodeBenchAdapter,
    MMLUProAdapter,
    ToolBenchAdapter,
    attribute_regression,
    available_benchmarks,
    judge_disagreement,
)
from vincio.evals.datasets import Dataset, EvalCase
from vincio.evals.judges import Judge
from vincio.evals.metrics import MetricResult, RunOutput


async def benchmark_adapters() -> None:
    print("1. More benchmark adapters")
    print(f"   registry ({len(available_benchmarks())}): {', '.join(available_benchmarks())}")

    # AgentBench: a knowledge-graph query scored by order-independent set match.
    ab = AgentBenchAdapter()
    kg = BenchmarkTask(id="kg", gold={"match": "set_match", "value": ["spain", "germany", "italy"]})
    print(f"   AgentBench (KG set match):   {(await ab.score(kg, ['Italy', 'Germany', 'Spain'])).success}")

    # ToolBench: a solution path that terminates with an answer using valid APIs.
    tb = ToolBenchAdapter()
    task = BenchmarkTask(id="t", inputs={"available_apis": ["search"]}, gold={"final_answer": "done"})
    path = [
        {"name": "search", "arguments": {"q": "x"}},
        {"name": "Finish", "arguments": {"return_type": "give_answer", "final_answer": "done"}},
    ]
    print(f"   ToolBench (solvable path):   {(await tb.score(task, path)).success}")

    # LiveCodeBench: a solution passes iff every test goes green.
    lcb = LiveCodeBenchAdapter()
    code_task = BenchmarkTask(id="l", gold={"public": ["p0"], "hidden": ["h0"]})
    results = {"results": {"p0": "passed", "h0": "passed"}}
    print(f"   LiveCodeBench (all tests):   {(await lcb.score(code_task, results)).success}")

    # MMLU-Pro: extract the predicted A–J letter and match the gold.
    mmlu = MMLUProAdapter()
    q = BenchmarkTask(id="m", gold="C")
    print(f"   MMLU-Pro (letter extract):   {(await mmlu.score(q, 'After analysis, the answer is (C).')).success}")


class _Fixed(Judge):
    """A deterministic judge returning a constant score."""

    def __init__(self, value: float, name: str) -> None:
        self.value = value
        self.name = name

    async def score(self, case: EvalCase, output: RunOutput) -> MetricResult:
        return MetricResult(name=self.name, value=self.value)


async def judge_ensembles() -> None:
    print("\n2. Judge ensembles with disagreement detection")
    case, out = EvalCase(id="c", input="q", expected="a"), RunOutput(output="a")

    agree = await JudgeEnsemble([_Fixed(0.9, "a"), _Fixed(0.92, "b"), _Fixed(0.88, "c")]).averdict(case, out)
    print(f"   unanimous panel: value={agree.value} uncertain={agree.uncertain} spread={agree.spread}")

    split = await JudgeEnsemble(
        [_Fixed(0.1, "a"), _Fixed(0.9, "b"), _Fixed(0.5, "c")], disagreement_threshold=0.2
    ).averdict(case, out)
    print(f"   split panel:     value={split.value} uncertain={split.uncertain} spread={split.spread}")
    print(f"   disagreement of the split panel: {judge_disagreement([0.1, 0.9, 0.5])}")

    panel = JudgeEnsemble([_Fixed(0.7, "a"), _Fixed(0.9, "b")])
    print(f"   gating weight before calibration: {panel.gating_weight()}")
    fit = panel.calibrate([(0.9, 1.0), (0.5, 0.6), (0.2, 0.1), (0.95, 0.9), (0.3, 0.25), (0.8, 0.85)])
    print(f"   panel-vs-human κ={fit['cohens_kappa']} → gating weight={panel.gating_weight(threshold=0.6)}")


async def causal_attribution() -> None:
    print("\n3. Causal regression attribution")
    # Only the model swap breaks the answer; an inert factor must get ~0 blame.
    def responder(request):
        return "The capital of France is Paris." if request.model == "good" else "unrelated text"

    provider, _ = example_provider(default_responder=responder)
    app = ContextApp(name="frontier", provider=provider, model="good")
    dataset = Dataset(
        name="capitals",
        cases=[
            EvalCase(id=f"c{i}", input="What is the capital of France?",
                     expected="The capital of France is Paris.")
            for i in range(5)
        ],
    )
    report = await attribute_regression(
        app, dataset,
        factors=[
            AttributionFactor.model("model", baseline="good", candidate="bad"),
            AttributionFactor.attr("inert", "name", baseline="frontier", candidate="frontier-b"),
        ],
        metric="lexical_overlap",
    )
    print(f"   total delta={report.total_delta} regressed={report.regressed} explained={report.explained}")
    print(f"   dominant cause: {report.dominant_factor} (concentration={report.concentration})")
    for c in report.contributions:
        print(f"     {c.factor:8s} Shapley contribution={c.contribution:+.3f} regressive={c.regressive}")


async def adaptive_sampling() -> None:
    print("\n4. Adaptive eval sampling")

    class _Case:
        def __init__(self, cid: str, mean: float, sd: float) -> None:
            self.id, self.mean, self.sd = cid, mean, sd

    # Three near-deterministic high scorers and one noisy case near the threshold.
    cases = [_Case("a", 0.97, 0.02), _Case("b", 0.96, 0.03), _Case("c", 0.95, 0.02),
             _Case("noisy", 0.82, 0.25), _Case("d", 0.94, 0.04)]

    def make_sample(seed: int):
        rng = random.Random(seed)
        return lambda c: max(0.0, min(1.0, rng.gauss(c.mean, c.sd)))

    budget = 250
    adaptive = await AdaptiveSampler(cases, make_sample(11), gate=">= 0.8", budget=budget).run()
    full = await AdaptiveSampler(
        cases, make_sample(11), gate=">= 0.8", budget=budget, seed_samples=budget // len(cases)
    ).run()
    print(f"   gate '>= 0.8': verdict={adaptive.verdict} decided={adaptive.decided}")
    print(f"   adaptive spent {adaptive.samples_used} samples vs {full.samples_used} exhaustive "
          f"(same verdict: {adaptive.verdict == full.verdict})")
    print(f"   budget landed on: {adaptive.allocations}")


async def main() -> None:
    await benchmark_adapters()
    await judge_ensembles()
    await causal_attribution()
    await adaptive_sampling()


if __name__ == "__main__":
    asyncio.run(main())
