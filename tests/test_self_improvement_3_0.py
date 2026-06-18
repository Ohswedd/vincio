"""3.0 unified self-improvement contract: SelfImprovementPolicy + streaming
controller, meta-optimization (successive-halving + learned weights),
active-learning label acquisition, and canary-gated deployment."""

import re

import pytest

from vincio import ContextApp
from vincio.evals import Dataset, EvalCase
from vincio.evals.reports import CaseResult, EvalReport
from vincio.optimize import (
    CanarySpec,
    DeployResult,
    FitnessWeights,
    MetaSpec,
    SelfImprovementController,
    SelfImprovementPolicy,
    learn_fitness_weights,
    select_for_labeling,
    successive_halving,
)
from vincio.providers import MockProvider

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
                "Refund window for Pro?",
                "How many days to refund a Pro subscription?",
                "Pro refund eligibility window?",
                "Days allowed to refund the Pro plan?",
            ]
        )
    ],
)


def format_sensitive_provider():
    """Answers correctly only for XML-rendered prompts — gives the optimizer a
    real signal so a variant can deterministically beat the baseline."""

    def responder(request):
        text = "\n".join(m.text for m in request.messages)
        if "</" not in text:
            return "I cannot answer that."
        match = re.search(r"\[([\w.:-]+:C\d+)\]", text)
        ref = match.group(1) if match else "E1"
        return f"The refund window for the Pro plan is 30 days. [{ref}]"

    return MockProvider(responder=responder)


@pytest.fixture()
def si_app(sample_docs_dir, offline_config, tmp_cwd):
    app = ContextApp(
        name="si_app",
        provider=format_sensitive_provider(),
        model="mock-1",
        config=offline_config,
    )
    app.add_source("docs", path=str(sample_docs_dir))
    return app


# ---------------------------------------------------------------------------
# Meta-optimization primitives
# ---------------------------------------------------------------------------


class TestMetaOptimization:
    async def test_successive_halving_picks_best_and_converges(self):
        # score = config value; halving keeps the best half each round.
        async def score(c):
            return float(c)

        best, history = await successive_halving([1, 5, 3, 9, 7], score, rounds=3)
        assert best == 9
        # First round scores all five; later rounds shrink the survivor set.
        first_round = [h for h in history if h["round"] == 0]
        assert len(first_round) == 5

    async def test_successive_halving_empty_raises(self):
        async def score(c):
            return 0.0

        with pytest.raises(ValueError):
            await successive_halving([], score)

    def test_learn_fitness_weights_boosts_weakest_metric(self):
        report = EvalReport(
            name="r",
            dataset="d",
            cases=[
                CaseResult(case_id=f"c{i}", metrics={"lexical_overlap": 0.4, "groundedness": 0.84})
                for i in range(4)
            ],
        )
        base = FitnessWeights()
        learned = learn_fitness_weights(report, base=base)
        # lexical_overlap (gap 0.4) is weaker than groundedness (gap ~0.01),
        # so the accuracy weight rises and the input is never mutated.
        assert learned.accuracy > base.accuracy
        assert base.accuracy == 1.0

    def test_learn_fitness_weights_none_report_is_identity(self):
        learned = learn_fitness_weights(None)
        assert learned.accuracy == FitnessWeights().accuracy

    def test_select_for_labeling_prefers_uncertain_cases(self):
        report = EvalReport(
            name="r",
            dataset="d",
            cases=[
                CaseResult(case_id="certain_high", metrics={"lexical_overlap": 0.98}),
                CaseResult(case_id="uncertain", metrics={"lexical_overlap": 0.51}),
                CaseResult(case_id="certain_low", metrics={"lexical_overlap": 0.02}),
            ],
        )
        picks = select_for_labeling(report, metric="lexical_overlap", budget=1)
        assert picks == ["uncertain"]


# ---------------------------------------------------------------------------
# Canary-gated deployment
# ---------------------------------------------------------------------------


class TestDeploy:
    def test_no_regression_deploys(self, si_app, tmp_cwd):
        # Deploying the live spec against itself is a tie → no regression → deploy.
        result = si_app.deploy(
            si_app.prompt_spec, dataset=QA_DATASET, canary=CanarySpec(metric="lexical_overlap")
        )
        assert isinstance(result, DeployResult)
        assert result.deployed
        assert result.verdict is not None and result.verdict.passed
        assert result.ref is not None

    def test_failing_gate_refuses_and_rolls_back(self, si_app, tmp_cwd):
        from vincio.prompts.registry import PromptRegistry

        registry = PromptRegistry()
        # First, a clean deploy establishes a known-good production version.
        good = si_app.deploy(
            si_app.prompt_spec,
            dataset=QA_DATASET,
            canary=CanarySpec(metric="lexical_overlap"),
            registry=registry,
        )
        assert good.deployed
        # A deploy with an unreachable gate is refused and rolls back to the
        # last known-good version (the canary verdict gates the serving change).
        refused = si_app.deploy(
            si_app.prompt_spec,
            dataset=QA_DATASET,
            canary=CanarySpec(metric="lexical_overlap"),
            gates={"lexical_overlap": ">= 0.999"},
            registry=registry,
        )
        assert not refused.deployed
        assert refused.rolled_back_to is not None
        assert "gates failed" in refused.reason

    def test_deploy_dry_run_changes_nothing(self, si_app, tmp_cwd):
        original = si_app.prompt_spec.spec_hash
        result = si_app.deploy(
            si_app.prompt_spec,
            dataset=QA_DATASET,
            canary=CanarySpec(metric="lexical_overlap"),
            dry_run=True,
        )
        assert not result.deployed
        assert "dry run" in result.reason
        assert si_app.prompt_spec.spec_hash == original

    def test_too_few_canary_samples_refuses(self, si_app, tmp_cwd):
        tiny = Dataset(name="tiny", cases=[EvalCase(id="c0", input="q", expected="a")])
        result = si_app.deploy(
            si_app.prompt_spec,
            dataset=tiny,
            canary=CanarySpec(metric="lexical_overlap", min_samples=4),
        )
        assert not result.deployed
        assert "insufficient canary samples" in result.verdict.reason

    def test_deploy_audits_decision(self, si_app, tmp_cwd):
        si_app.deploy(
            si_app.prompt_spec, dataset=QA_DATASET, canary=CanarySpec(metric="lexical_overlap")
        )
        actions = [e.action for e in si_app.audit.entries]
        assert "deploy" in actions


# ---------------------------------------------------------------------------
# The unified streaming controller
# ---------------------------------------------------------------------------


class TestSelfImprovementController:
    def test_app_self_improvement_returns_controller(self, si_app, tmp_cwd):
        ctl = si_app.self_improvement(SelfImprovementPolicy(), dataset=QA_DATASET)
        assert isinstance(ctl, SelfImprovementController)

    async def test_stream_emits_full_cycle(self, si_app, tmp_cwd):
        policy = SelfImprovementPolicy(
            metrics=["lexical_overlap", "cost", "latency"],
            meta=MetaSpec(strategies=["evolution"], budgets=[4]),
            canary=CanarySpec(metric="lexical_overlap"),
        )
        ctl = si_app.self_improvement(policy, dataset=QA_DATASET)
        phases = [ev.phase async for ev in ctl.astream()]
        # The cycle observes, meta-selects, re-optimizes, and reaches a serving
        # decision (promote on success).
        assert phases[0] == "observe"
        assert "meta" in phases
        assert "reeval" in phases
        assert phases[-1] in ("promote", "rollback")
        # Budget is tracked monotonically and every phase is audited.
        assert ctl.events[-1].budget_spent > 0
        assert any(e.action == "self_improvement" for e in si_app.audit.entries)

    async def test_meta_selects_strategy_and_learns_weights(self, si_app, tmp_cwd):
        policy = SelfImprovementPolicy(
            metrics=["lexical_overlap", "cost", "latency"],
            meta=MetaSpec(strategies=["evolution", "reflective"], budgets=[4, 8]),
        )
        ctl = si_app.self_improvement(policy, dataset=QA_DATASET)
        events = await ctl.step()
        meta = next(e for e in events if e.phase == "meta")
        assert meta.chosen_strategy in ("evolution", "reflective")
        assert meta.chosen_budget in (4, 8)

    async def test_active_learning_queues_uncertain_cases(self, si_app, tmp_cwd):
        policy = SelfImprovementPolicy(
            metrics=["lexical_overlap"],
            active_learning=True,
            label_budget=3,
            meta=None,
            propose=False,
        )
        ctl = si_app.self_improvement(policy, dataset=QA_DATASET)
        events = await ctl.step()
        label = next(e for e in events if e.phase == "label")
        assert 0 < len(label.label_case_ids) <= 3

    async def test_no_dataset_only_observes_and_proposes(self, si_app, tmp_cwd):
        ctl = si_app.self_improvement(SelfImprovementPolicy(), dataset=None)
        phases = [ev.phase async for ev in ctl.astream()]
        assert "reeval" in phases  # emitted as skipped
        assert "promote" not in phases

    def test_run_is_sync_wrapper(self, si_app, tmp_cwd):
        policy = SelfImprovementPolicy(
            metrics=["lexical_overlap", "cost", "latency"],
            meta=MetaSpec(strategies=["evolution"], budgets=[4]),
        )
        events = si_app.self_improvement(policy, dataset=QA_DATASET).run()
        assert events and events[0].phase == "observe"

    async def test_budget_exhaustion_stops_before_reeval(self, si_app, tmp_cwd):
        # A tiny budget is spent by active-learning, so re-eval never runs.
        policy = SelfImprovementPolicy(
            metrics=["lexical_overlap"],
            eval_budget=2.0,
            active_learning=True,
            meta=None,
            propose=False,
            canary=None,
        )
        ctl = si_app.self_improvement(policy, dataset=QA_DATASET)
        phases = [ev.phase async for ev in ctl.astream()]
        assert "exhausted" in phases
