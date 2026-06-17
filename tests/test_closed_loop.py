"""0.8 closed-loop ecosystem tests: improvement loop, Pareto optimization,
guided search strategies, learned budgeting, retrieval feedback, auto-memory."""

import re

import pytest

from vincio import ContextApp, TaskType
from vincio.context.budgeting import BudgetAllocator
from vincio.core.types import Chunk, EvidenceItem
from vincio.evals import Dataset, EvalCase, EvalReport
from vincio.evals.reports import CaseResult
from vincio.memory.facts import extract_grounded_facts
from vincio.optimize import (
    BudgetLearner,
    Candidate,
    ContextOptimizer,
    HillClimbSearch,
    ImprovementLoop,
    LearnedAllocations,
    ObjectiveSpec,
    ParetoFrontier,
    ParetoPoint,
    RelevanceRecord,
    RetrievalFeedback,
    dominates,
    guided_search,
    pareto_loop,
    recommend_chunking,
    records_from_dataset,
    records_from_report,
)
from vincio.providers import MockProvider
from vincio.retrieval.engine import RetrievalEngine
from vincio.retrieval.indexes import BM25Index


def report_for(metrics_per_case: list[dict[str, float]], *, name: str = "r") -> EvalReport:
    return EvalReport(
        name=name,
        dataset="d",
        cases=[
            CaseResult(case_id=f"c{i}", metrics=dict(metrics))
            for i, metrics in enumerate(metrics_per_case)
        ],
    )


def quality_report(quality: float, *, cost: float = 0.001, n: int = 4) -> EvalReport:
    return report_for(
        [
            {
                "lexical_overlap": quality,
                "groundedness": quality,
                "cost": cost,
                "latency": 100.0,
            }
        ]
        * n
    )


QA_DATASET = Dataset(
    name="refunds",
    cases=[
        EvalCase(id=f"c{i}", input=q, expected="The refund window for the Pro plan is 30 days.")
        for i, q in enumerate(
            [
                "What is the refund window for the Pro plan?",
                "How long do Pro customers have to request a refund?",
                "Within how many days can a Pro plan be refunded?",
                "Pro plan refund period?",
            ]
        )
    ],
)


# ---------------------------------------------------------------------------
# Pareto multi-objective optimization
# ---------------------------------------------------------------------------


class TestPareto:
    OBJECTIVES = [
        ObjectiveSpec(name="accuracy", metric="lexical_overlap"),
        ObjectiveSpec(name="cost", metric="cost", direction="min"),
    ]

    def test_dominates(self):
        better = {"accuracy": 0.9, "cost": 0.001}
        worse = {"accuracy": 0.8, "cost": 0.002}
        tradeoff = {"accuracy": 0.95, "cost": 0.01}
        assert dominates(better, worse, self.OBJECTIVES)
        assert not dominates(worse, better, self.OBJECTIVES)
        assert not dominates(better, tradeoff, self.OBJECTIVES)
        assert not dominates(tradeoff, better, self.OBJECTIVES)

    def test_frontier_and_knee(self):
        points = [
            ParetoPoint(name="premium", objectives={"accuracy": 0.95, "cost": 0.02}),
            ParetoPoint(name="balanced", objectives={"accuracy": 0.9, "cost": 0.004}),
            ParetoPoint(name="cheap", objectives={"accuracy": 0.6, "cost": 0.001}),
            ParetoPoint(name="dominated", objectives={"accuracy": 0.5, "cost": 0.01}),
        ]
        frontier = ParetoFrontier.build(points, specs=self.OBJECTIVES)
        names = {p.name for p in frontier.front}
        assert names == {"premium", "balanced", "cheap"}
        assert frontier.knee().name == "balanced"

    def test_select_constraints_and_prefer(self):
        points = [
            ParetoPoint(name="premium", objectives={"accuracy": 0.95, "cost": 0.02}),
            ParetoPoint(name="cheap", objectives={"accuracy": 0.6, "cost": 0.001}),
        ]
        frontier = ParetoFrontier.build(points, specs=self.OBJECTIVES)
        assert frontier.select(constraints={"cost": 0.005}).name == "cheap"
        assert frontier.select(prefer="accuracy").name == "premium"
        assert frontier.select(constraints={"accuracy": 0.99}) is None

    @pytest.mark.asyncio
    async def test_pareto_loop_promotes_frontier_point(self):
        qualities = {"baseline": 0.6, "good": 0.9, "bad": 0.3}

        async def evaluate(candidate, ds):
            return quality_report(qualities[candidate.name])

        result = await pareto_loop(
            [Candidate(name="good"), Candidate(name="bad")],
            evaluate,
            QA_DATASET,
            baseline=Candidate(name="baseline"),
            objectives=self.OBJECTIVES,
            subset_size=4,
            top_n=2,
        )
        assert result.promoted
        assert result.best.name == "good"
        assert "frontier" in result.reason
        assert any(p.name == "baseline" for p in result.frontier.points)

    @pytest.mark.asyncio
    async def test_pareto_loop_respects_constraints(self):
        async def evaluate(candidate, ds):
            return quality_report(0.9, cost=0.05)  # better but expensive

        result = await pareto_loop(
            [Candidate(name="pricey")],
            evaluate,
            QA_DATASET,
            baseline=Candidate(name="baseline"),
            objectives=self.OBJECTIVES,
            constraints={"cost": 0.01},
            subset_size=4,
        )
        # The expensive candidate and the baseline share cost 0.05 here, so
        # nothing satisfies the constraint.
        assert not result.promoted

    @pytest.mark.asyncio
    async def test_small_dataset_refused(self):
        async def evaluate(candidate, ds):  # pragma: no cover - never called
            raise AssertionError("should not evaluate")

        result = await pareto_loop(
            [Candidate(name="x")],
            evaluate,
            Dataset(name="tiny", cases=[EvalCase(id="c", input="q")]),
            baseline=Candidate(name="baseline"),
        )
        assert not result.promoted
        assert "too small" in result.reason


# ---------------------------------------------------------------------------
# Guided search strategies
# ---------------------------------------------------------------------------


SPACE = {"top_k": [4, 8, 12], "reranker": ["heuristic", None], "ordering": ["relevance", "boundary_sandwich"]}


class TestStrategies:
    def test_hill_climb_neighbors_of_best(self):
        strategy = HillClimbSearch(SPACE, seed=7)
        first = strategy.propose([], n=3)
        assert len(first) == 3
        best = {"top_k": 12, "reranker": "heuristic", "ordering": "relevance"}
        history = [(config, 0.1) for config in first] + [(best, 0.9)]
        batch = strategy.propose(history, n=3)
        assert batch
        for config in batch:
            differing = [k for k in SPACE if config.get(k) != best[k]]
            # Single-knob mutations of the incumbent (random top-up allowed
            # only when the neighborhood is exhausted, not the case here).
            assert len(differing) <= 2

    def test_strategies_deterministic_under_seed(self):
        a = HillClimbSearch(SPACE, seed=11).propose([], n=4)
        b = HillClimbSearch(SPACE, seed=11).propose([], n=4)
        assert a == b

    @pytest.mark.asyncio
    async def test_guided_search_bounded_by_budget(self):
        evaluated = []

        async def evaluate(config):
            evaluated.append(config)
            return float(config["top_k"])  # bigger top_k scores better

        history = await guided_search(SPACE, evaluate, strategy="hill_climb", budget=7, seed=7)
        assert len(history) == len(evaluated) == 7
        best_config, best_score = max(history, key=lambda entry: entry[1])
        assert best_config["top_k"] == 12 and best_score == 12.0

    @pytest.mark.asyncio
    async def test_anneal_search_runs(self):
        async def evaluate(config):
            return float(config["top_k"])

        history = await guided_search(SPACE, evaluate, strategy="anneal", budget=6, seed=3)
        assert len(history) == 6

    @pytest.mark.asyncio
    async def test_context_optimizer_guided_strategy(self):
        subset_evals = []

        async def evaluate_config(config, ds):
            subset_evals.append(config)
            quality = 0.5 + 0.04 * config["top_k"] / 12 + (0.2 if config["reranker"] else 0.0)
            return quality_report(min(1.0, quality), n=len(ds))

        optimizer = ContextOptimizer(evaluate_config)
        result = await optimizer.optimize(
            QA_DATASET,
            budget=6,
            subset_size=4,
            top_n=2,
            strategy="hill_climb",
            baseline_config={"top_k": 4, "reranker": None},
        )
        assert result.best is not None
        assert result.best.subset_fitness is not None  # pre-scored by the strategy
        assert result.promoted
        assert result.best.params["reranker"] == "heuristic"


# ---------------------------------------------------------------------------
# Learned context budgeting
# ---------------------------------------------------------------------------


class TestLearnedBudgeting:
    @pytest.mark.asyncio
    async def test_budget_learner_promotes_better_allocation(self):
        async def evaluate_allocation(fractions, ds):
            # Reward giving evidence more budget — simulates a QA workload.
            return quality_report(min(1.0, 0.4 + fractions.get("evidence", 0.0)), n=len(ds))

        learner = BudgetLearner(evaluate_allocation)
        result, learned = await learner.learn(
            QA_DATASET, task_type=TaskType.GENERAL, candidates=8, subset_size=4, seed=3
        )
        assert result.promoted and learned is not None
        table = learned.get(TaskType.GENERAL)
        assert table is not None
        baseline = BudgetAllocator().allocation_for(TaskType.GENERAL)
        assert table["evidence"] > baseline["evidence"]
        assert abs(sum(table.values()) - 1.0) < 1e-6

    def test_learned_allocations_roundtrip(self, tmp_path):
        learned = LearnedAllocations()
        learned.set(TaskType.DOCUMENT_QA, {"evidence": 0.8, "instructions": 0.2})
        path = learned.save(tmp_path / "budgets.json")
        loaded = LearnedAllocations.load(path)
        table = loaded.get("document_qa")
        assert table == {"evidence": 0.8, "instructions": 0.2}

    def test_allocator_prefers_learned_table(self):
        allocator = BudgetAllocator(
            learned={"document_qa": {"evidence": 0.9, "instructions": 0.1}}
        )
        learned = allocator.allocation_for(TaskType.DOCUMENT_QA)
        assert learned["evidence"] == pytest.approx(0.9)
        # Other tasks keep the fixed tables.
        fixed = allocator.allocation_for(TaskType.CLASSIFICATION)
        assert fixed["examples"] > fixed["evidence"]

    def test_app_use_learned_budgets(self, offline_config, tmp_cwd):
        app = ContextApp(name="t", provider=MockProvider(), model="mock-1", config=offline_config)
        app.use_learned_budgets({"document_qa": {"evidence": 0.85, "instructions": 0.15}})
        table = app.context_compiler.allocator.allocation_for(TaskType.DOCUMENT_QA)
        assert table["evidence"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Retrieval feedback
# ---------------------------------------------------------------------------


def make_chunk(document_id: str, index: int, text: str) -> Chunk:
    return Chunk(document_id=document_id, index=index, text=text, token_count=len(text.split()))


class TestRetrievalFeedback:
    @pytest.fixture()
    def labelled_dataset(self):
        return Dataset(
            name="labels",
            cases=[
                EvalCase(
                    id="c1",
                    input="What is the refund window for the Pro plan?",
                    rubric={"relevant_ids": ["D1:C0"]},
                ),
                EvalCase(id="c2", input="unlabelled case"),
            ],
        )

    def test_records_from_dataset(self, labelled_dataset):
        records = records_from_dataset(labelled_dataset)
        assert len(records) == 1
        assert records[0].relevant_ids == ["D1:C0"]

    def test_records_from_report(self, labelled_dataset):
        report = EvalReport(
            name="r",
            dataset="labels",
            cases=[CaseResult(case_id="c1", metrics={"recall_at_k": 0.5, "mrr": 0.5})],
        )
        records = records_from_report(report, labelled_dataset)
        assert len(records) == 1
        assert records[0].observed["recall_at_k"] == 0.5

    @pytest.mark.asyncio
    async def test_tune_index_weights_downweights_noisy_index(self):
        good = BM25Index()
        await good.add(
            [
                make_chunk("D1", 0, "Customers on the Pro plan may request refunds within 30 days."),
                make_chunk("D1", 1, "The subscription renews automatically before renewal."),
            ]
        )
        junk = BM25Index()
        await junk.add(
            [
                make_chunk("J1", 0, "Refund window pro plan marketing newsletter signup."),
                make_chunk("J2", 0, "Pro plan refund window stickers and merchandise."),
            ]
        )
        engine = RetrievalEngine([good, junk], index_weights=[1.0, 2.0], reranker=None)
        records = [
            RelevanceRecord(
                query="What is the refund window for the Pro plan?",
                relevant_ids=["D1:C0"],
            )
        ]
        feedback = RetrievalFeedback(engine, records, top_k=2)
        result = await feedback.tune_index_weights()
        assert result.applied
        assert result.tuned_score > result.baseline_score
        # The relative weight of the noisy index dropped.
        assert engine.index_weights[1] / engine.index_weights[0] < 2.0

    @pytest.mark.asyncio
    async def test_tuning_is_gated_when_nothing_improves(self):
        index = BM25Index()
        await index.add(
            [make_chunk("D1", 0, "Customers on the Pro plan may request refunds within 30 days.")]
        )
        engine = RetrievalEngine([index], reranker=None)
        records = [
            RelevanceRecord(query="refund window pro plan", relevant_ids=["D1:C0"])
        ]
        feedback = RetrievalFeedback(engine, records, top_k=2)
        result = await feedback.tune_index_weights()
        assert not result.applied
        assert engine.index_weights == [1.0]
        assert "unchanged" in result.reason

    def test_recommend_chunking(self):
        reports = {
            "recursive:400/50": report_for([{"recall_at_k": 0.5}] * 4),
            "sentence_window:200/0": report_for([{"recall_at_k": 0.9}] * 4),
        }
        recommendation = recommend_chunking(reports, baseline="recursive:400/50")
        assert recommendation.changed
        assert recommendation.recommended == "sentence_window:200/0"
        assert recommendation.improvement == pytest.approx(0.4)
        conservative = recommend_chunking(
            reports, baseline="recursive:400/50", min_improvement=0.5
        )
        assert not conservative.changed


# ---------------------------------------------------------------------------
# Auto-memory from runs
# ---------------------------------------------------------------------------


class TestAutoMemory:
    EVIDENCE = [
        EvidenceItem(
            id="D1:C0",
            source_id="D1",
            text="Customers on the Pro plan may request refunds within 30 days.",
            provenance=0.9,
        )
    ]

    def test_extract_grounded_facts(self):
        text = (
            "The refund window for the Pro plan is 30 days. [D1:C0] "
            "Our mascot is a purple axolotl named Gerald."
        )
        facts = extract_grounded_facts(text, self.EVIDENCE)
        assert len(facts) == 1
        assert "30 days" in facts[0].content
        assert "[D1:C0]" not in facts[0].content  # citation markers stripped
        assert facts[0].support >= 0.5
        assert facts[0].evidence_ids == ["D1:C0"]

    def test_extract_requires_evidence_support(self):
        assert extract_grounded_facts("The refund window is 30 days.", []) == []
        ungrounded = extract_grounded_facts(
            "The capital of France has 50 bridges and 12 airports total.", self.EVIDENCE
        )
        assert ungrounded == []

    def test_write_back_facts_become_candidates(self):
        from vincio.memory import MemoryEngine
        from vincio.memory.facts import GroundedFact
        from vincio.memory.stores import InMemoryMemoryStore

        engine = MemoryEngine(InMemoryMemoryStore())
        written = engine.write_back(
            facts=[
                GroundedFact(
                    content="The Pro plan refund window is 30 days.",
                    support=0.8,
                    evidence_ids=["D1:C0"],
                )
            ],
            owner_id="u1",
        )
        assert len(written) == 1
        item = written[0]
        assert item.status == "candidate"
        assert item.metadata["origin"] == "run_fact"
        assert item.confidence == pytest.approx(min(0.8, 0.35 + 0.45 * 0.8))

    def test_run_writes_grounded_facts(
        self, sample_docs_dir, citing_mock_provider, offline_config, tmp_cwd
    ):
        offline_config.memory.write_back = ["facts"]
        app = ContextApp(
            name="facts_app", provider=citing_mock_provider, model="mock-1", config=offline_config
        )
        app.add_source("docs", path=str(sample_docs_dir))
        app.add_memory()
        result = app.run("What is the refund window for the Pro plan?", user_id="u1")
        assert result.status.value == "succeeded"
        items = [
            item
            for item in app.memory.store.all_items(statuses=("candidate",))
            if item.metadata.get("origin") == "run_fact"
        ]
        assert items, "expected at least one grounded fact written back"
        assert any("30 days" in item.content for item in items)
        assert all(item.metadata.get("support", 0) >= 0.5 for item in items)


# ---------------------------------------------------------------------------
# The improvement loop
# ---------------------------------------------------------------------------


def format_sensitive_provider():
    """Answers correctly only for XML-rendered prompts — gives the optimizer
    a real signal so a variant can beat the baseline deterministically."""

    def responder(request):
        text = "\n".join(m.text for m in request.messages)
        if "</" not in text:
            return "I cannot answer that."
        match = re.search(r"\[([\w.:-]+:C\d+)\]", text)
        ref = match.group(1) if match else "E1"
        return f"The refund window for the Pro plan is 30 days. [{ref}]"

    return MockProvider(responder=responder)


class TestImprovementLoop:
    @pytest.fixture()
    def loop_app(self, sample_docs_dir, offline_config, tmp_cwd):
        app = ContextApp(
            name="loop_app",
            provider=format_sensitive_provider(),
            model="mock-1",
            config=offline_config,
        )
        app.add_source("docs", path=str(sample_docs_dir))
        return app

    def test_loop_promotes_and_records_everything(self, loop_app, tmp_cwd):
        loop = ImprovementLoop(
            loop_app, metrics=["lexical_overlap", "cost", "latency"], experiment="exp_loop"
        )
        result = loop.run(dataset=QA_DATASET, max_variants=6, subset_size=4)

        assert result.promoted
        assert result.dataset_fingerprint
        assert result.promoted_ref is not None
        # The registry holds the promoted version, tagged and eval-linked.
        version = loop.registry.get("loop_app", tag="production")
        assert version.ref == result.promoted_ref
        assert version.eval_runs and version.eval_runs[0]["dataset"] == "refunds"
        # The winner is applied to the live app.
        assert loop_app.prompt_spec.spec_hash == version.spec.spec_hash
        # Both baseline and winner are logged to the experiment tracker.
        variants = {run.variant for run in loop.tracker.runs("exp_loop")}
        assert "baseline" in variants and len(variants) >= 2
        # Every stage left a step record.
        assert [s["stage"] for s in result.steps][-1] == "promote"

    def test_loop_dry_run_changes_nothing(self, loop_app, tmp_cwd):
        original_hash = loop_app.prompt_spec.spec_hash
        loop = ImprovementLoop(loop_app, metrics=["lexical_overlap", "cost", "latency"])
        result = loop.run(dataset=QA_DATASET, max_variants=6, subset_size=4, dry_run=True)
        assert not result.promoted
        assert result.reason.startswith("dry run")
        assert loop_app.prompt_spec.spec_hash == original_hash
        assert loop.registry.names() == []

    def test_loop_refuses_small_dataset(self, loop_app, tmp_cwd):
        tiny = Dataset(name="tiny", cases=[EvalCase(id="c", input="q")])
        loop = ImprovementLoop(loop_app, metrics=["lexical_overlap"])
        result = loop.run(dataset=tiny)
        assert not result.promoted
        assert "too small" in result.reason

    def test_capture_and_curate_from_traces(self, loop_app, tmp_cwd):
        from vincio.observability.sessions import record_feedback

        loop_app.run("What is the refund window for the Pro plan?", user_id="u1")
        loop_app.run("How long is the Pro refund period?", user_id="u1")
        exporter = loop_app.tracer.exporter
        for trace in exporter.traces:
            record_feedback(trace, score=1.0 if "window" in trace.attributes["input"] else 0.2)

        loop = ImprovementLoop(loop_app)
        traces = loop.capture()
        assert len(traces) == 2
        dataset = loop.curate(traces, min_feedback_score=0.5)
        assert len(dataset) == 1
        case = dataset.cases[0]
        assert "refund window" in case.input_text
        assert case.metadata["trace_id"]
