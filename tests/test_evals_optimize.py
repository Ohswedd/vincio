"""Eval metrics + optimization unit tests (eval metrics)."""

import json

import pytest

from vincio.core.types import EvidenceItem
from vincio.evals import (
    Dataset,
    EvalCase,
    EvalReport,
    EvalRunner,
    HybridJudge,
    ModelJudge,
    RunOutput,
    evaluate_gates,
)
from vincio.evals.judges import DeterministicJudge
from vincio.evals.metrics import (
    METRICS,
    citation_accuracy,
    classification_accuracy,
    exact_match,
    extraction_f1,
    groundedness,
    mrr,
    recall_at_k,
)
from vincio.evals.reports import CaseResult
from vincio.optimize import (
    Candidate,
    EpsilonGreedyBandit,
    FitnessWeights,
    RoutingPolicy,
    analyze_prompt_cacheability,
    estimate_difficulty,
    evolution_loop,
    fitness,
)
from vincio.providers import MockProvider


def make_output(text="Refunds are allowed within 30 days. [E1]", **kw) -> RunOutput:
    return RunOutput(
        output=text,
        raw_text=text,
        evidence=[
            EvidenceItem(id="E1", source_id="D1", text="Refunds are allowed within 30 days for Pro plans.")
        ],
        citations=["E1"],
        cost_usd=0.001,
        latency_ms=100,
        schema_valid=True,
        **kw,
    )


class TestMetrics:
    def test_exact_match(self):
        case = EvalCase(id="c", input="q", expected="Refunds are allowed within 30 days. [E1]")
        assert exact_match(case, make_output()).value == 1.0

    def test_classification_accuracy(self):
        case = EvalCase(id="c", input="q", expected={"label": "billing"})
        run = RunOutput(output={"label": "Billing"})
        assert classification_accuracy(case, run).value == 1.0

    def test_extraction_f1(self):
        case = EvalCase(id="c", input="q", expected=["INV-1", "INV-2"])
        run = RunOutput(output=["INV-1", "INV-3"])
        result = extraction_f1(case, run)
        assert result.value == 0.5

    def test_groundedness(self):
        case = EvalCase(id="c", input="q")
        grounded = groundedness(case, make_output())
        hallucinated = groundedness(
            case,
            make_output(text="The refund window is 90 days and includes free pizza for everyone."),
        )
        assert grounded.value == 1.0
        assert hallucinated.value < 1.0

    def test_citation_accuracy(self):
        case = EvalCase(id="c", input="q")
        good = citation_accuracy(case, make_output())
        run = make_output()
        run.citations = ["E1", "FAKE"]
        mixed = citation_accuracy(case, run)
        assert good.value == 1.0 and mixed.value == 0.5

    def test_retrieval_metrics(self):
        case = EvalCase(id="c", input="q", rubric={"relevant_ids": ["E1"]})
        assert recall_at_k(case, make_output()).value == 1.0
        assert mrr(case, make_output()).value == 1.0

    def test_registry_complete(self):
        expected = {
            "exact_match", "semantic_similarity", "classification_accuracy", "extraction_f1",
            "schema_validity", "groundedness", "unsupported_claim_rate", "citation_accuracy",
            "citation_recall", "context_precision", "context_recall", "cost", "latency",
            "recall_at_k", "precision_at_k", "mrr", "ndcg",
        }
        assert expected <= set(METRICS)


class TestRunnerAndGates:
    @pytest.fixture()
    def dataset(self):
        return Dataset(
            name="t",
            cases=[
                EvalCase(id="c1", input="Can I get a refund?", expected="Refunds are allowed within 30 days."),
                EvalCase(id="c2", input="What is the window?", expected="30 days"),
            ],
        )

    @pytest.mark.asyncio
    async def test_runner(self, dataset):
        async def target(case):
            return make_output()

        runner = EvalRunner(target, metrics=["groundedness", "citation_accuracy", "cost"])
        report = await runner.arun(dataset)
        assert report.summary()["groundedness"]["mean"] == 1.0
        assert len(report.cases) == 2

    @pytest.mark.asyncio
    async def test_errored_case_recorded(self, dataset):
        async def target(case):
            if case.id == "c2":
                raise RuntimeError("boom")
            return make_output()

        report = await EvalRunner(target, metrics=["cost"]).arun(dataset)
        assert len(report.failures()) == 1

    def test_gates(self):
        report = EvalReport(
            cases=[CaseResult(case_id="c", metrics={"groundedness": 0.8, "latency": 100.0})]
        )
        outcomes = evaluate_gates(report, {"groundedness": ">= 0.9", "p95_latency": "<= 500"})
        assert not outcomes["groundedness"]["passed"]
        assert outcomes["p95_latency"]["passed"]

    def test_baseline_diff(self):
        baseline = EvalReport(cases=[CaseResult(case_id="c", metrics={"accuracy": 0.9})])
        current = EvalReport(cases=[CaseResult(case_id="c", metrics={"accuracy": 0.7})])
        diff = current.diff(baseline)
        assert diff["metrics"]["accuracy"]["delta"] == pytest.approx(-0.2)
        assert diff["regressed_cases"]

    @pytest.mark.asyncio
    async def test_model_judge_with_mock(self):
        provider = MockProvider(
            responder=lambda req: json.dumps({"score": 0.8, "reasoning": "ok", "failures": []})
        )
        judge = ModelJudge(provider, model="mock-1", samples=3)
        result = await judge.score(EvalCase(id="c", input="q"), make_output())
        assert result.value == pytest.approx(0.8)
        assert result.details["samples"] == 3

    @pytest.mark.asyncio
    async def test_hybrid_judge(self):
        deterministic = DeterministicJudge(groundedness, name="g")
        provider = MockProvider(
            responder=lambda req: json.dumps({"score": 0.5, "reasoning": "", "failures": []})
        )
        hybrid = HybridJudge([(deterministic, 0.5), (ModelJudge(provider, model="m"), 0.5)])
        result = await hybrid.score(EvalCase(id="c", input="q"), make_output())
        assert result.value == pytest.approx(0.75)


class TestOptimize:
    def make_report(self, quality, schema=1.0):
        return EvalReport(
            cases=[
                CaseResult(
                    case_id=f"c{i}",
                    metrics={
                        "semantic_similarity": quality,
                        "schema_validity": schema,
                        "cost": 0.001,
                        "latency": 100.0,
                    },
                )
                for i in range(6)
            ]
        )

    def test_fitness(self):
        good = fitness(self.make_report(0.9))
        bad = fitness(self.make_report(0.2))
        assert good > bad

    @pytest.mark.asyncio
    async def test_evolution_promotes_winner(self):
        dataset = Dataset(cases=[EvalCase(id=f"c{i}", input="q") for i in range(6)])

        async def evaluate(candidate, ds):
            quality = {"baseline": 0.6, "good": 0.85, "bad": 0.3}[candidate.name]
            return self.make_report(quality)

        result = await evolution_loop(
            [Candidate(name="good"), Candidate(name="bad")],
            evaluate,
            dataset,
            baseline=Candidate(name="baseline"),
            top_n=1,
        )
        assert result.promoted and result.best.name == "good"

    @pytest.mark.asyncio
    async def test_safety_gate_blocks_schema_regression(self):
        dataset = Dataset(cases=[EvalCase(id=f"c{i}", input="q") for i in range(6)])

        async def evaluate(candidate, ds):
            if candidate.name == "baseline":
                return self.make_report(0.6, schema=1.0)
            return self.make_report(0.99, schema=0.5)  # better quality, broken schema

        result = await evolution_loop(
            [Candidate(name="unsafe")],
            evaluate,
            dataset,
            baseline=Candidate(name="baseline"),
            top_n=1,
            weights=FitnessWeights(schema_validity=0.0),  # fitness ignores schema...
        )
        assert not result.promoted  # ...but the safety gate still blocks it
        assert "schema_validity regressed" in result.reason

    @pytest.mark.asyncio
    async def test_small_dataset_refused(self):
        result = await evolution_loop(
            [Candidate(name="x")],
            lambda c, d: None,  # never called
            Dataset(cases=[EvalCase(id="c1", input="q")]),
            baseline=Candidate(name="baseline"),
        )
        assert not result.promoted and "too small" in result.reason

    def test_routing(self):
        policy = RoutingPolicy(cheap_model="c", default_model="d", strong_model="s")
        assert policy.route(difficulty=0.1) == "c"
        assert policy.route(difficulty=0.5) == "d"
        assert policy.route(difficulty=0.9) == "s"
        assert policy.route(difficulty=0.1, risk="high") == "s"
        assert estimate_difficulty("why does this fail? explain step by step") > estimate_difficulty("classify: bug")

    def test_bandit(self):
        bandit = EpsilonGreedyBandit(["a", "b"], epsilon=0.0, seed=1)
        bandit.update("a", 0.2)
        bandit.update("b", 0.9)
        assert bandit.select() == "b"

    def test_cache_tuning(self):
        from vincio.prompts import PromptCompiler, PromptSpec

        compiled = PromptCompiler().compile(
            PromptSpec(name="x", role="engine run_abcdef1234 at 2026-01-01T10:00", objective="answer"),
            user_task="question",
        )
        report = analyze_prompt_cacheability(compiled)
        codes = {a.code for a in report.advice}
        assert "CACHE001" in codes or "CACHE002" in codes
