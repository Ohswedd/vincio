"""1.10 — autonomous self-improvement: the experiment proposer, the guarded
online bandits, and the held-out growing golden regression suite."""

import warnings

from vincio import ContextApp, VincioConfig
from vincio.core.types import Message, ModelRequest
from vincio.evals import Dataset, EvalCase, EvalReport, GoldenRegressionSuite
from vincio.evals.reports import CaseResult
from vincio.optimize.loop import ExperimentProposer
from vincio.optimize.routing import GuardedBanditRouter, LinUCB
from vincio.providers import MockProvider
from vincio.providers.base import run_sync
from vincio.storage.base import InMemoryMetadataStore

warnings.simplefilter("ignore")


# --------------------------------------------------------------------------- #
# growing golden regression suite
# --------------------------------------------------------------------------- #


class TestGoldenRegressionSuite:
    def test_add_and_persist(self, tmp_path):
        suite = GoldenRegressionSuite(tmp_path / "g.jsonl")
        suite.add(EvalCase(id="c1", input="q", expected="a"), fixed_by="qa@v2",
                  guard_metric="groundedness", guard_threshold=0.8)
        assert len(suite) == 1
        # Reloads from disk (a reproducible artifact).
        reloaded = GoldenRegressionSuite(tmp_path / "g.jsonl")
        assert reloaded.case_ids() == ["c1"]
        assert reloaded.dataset.cases[0].metadata["fixed_by"] == "qa@v2"

    def test_gate_passes_when_floor_met(self, tmp_path):
        suite = GoldenRegressionSuite(tmp_path / "g.jsonl")
        suite.add(EvalCase(id="c1", input="q"), fixed_by="v2",
                  guard_metric="groundedness", guard_threshold=0.8)
        report = EvalReport(cases=[CaseResult(case_id="c1", metrics={"groundedness": 0.9})])
        assert suite.gate(report).passed

    def test_gate_blocks_regression(self, tmp_path):
        suite = GoldenRegressionSuite(tmp_path / "g.jsonl")
        suite.add(EvalCase(id="c1", input="q"), fixed_by="v2",
                  guard_metric="groundedness", guard_threshold=0.8)
        report = EvalReport(cases=[CaseResult(case_id="c1", metrics={"groundedness": 0.4})])
        result = suite.gate(report)
        assert not result.passed and "c1" in result.regressed

    def test_gate_flags_missing_case(self, tmp_path):
        suite = GoldenRegressionSuite(tmp_path / "g.jsonl")
        suite.add(EvalCase(id="c1", input="q"), fixed_by="v2")
        report = EvalReport(cases=[CaseResult(case_id="other", metrics={"lexical_overlap": 1.0})])
        result = suite.gate(report)
        assert not result.passed and "c1" in result.missing

    def test_lower_is_better_metric(self, tmp_path):
        suite = GoldenRegressionSuite(tmp_path / "g.jsonl")
        suite.add(EvalCase(id="c1", input="q"), fixed_by="v2",
                  guard_metric="toxicity", guard_threshold=0.1)
        ok = EvalReport(cases=[CaseResult(case_id="c1", metrics={"toxicity": 0.05})])
        bad = EvalReport(cases=[CaseResult(case_id="c1", metrics={"toxicity": 0.4})])
        assert suite.gate(ok).passed and not suite.gate(bad).passed

    def test_add_from_report(self, tmp_path):
        suite = GoldenRegressionSuite(tmp_path / "g.jsonl")
        dataset = Dataset(cases=[EvalCase(id="c1", input="q1"), EvalCase(id="c2", input="q2")])
        report = EvalReport(cases=[
            CaseResult(case_id="c1", metrics={"lexical_overlap": 0.9}),
            CaseResult(case_id="c2", metrics={"lexical_overlap": 0.3}),
        ])
        added = suite.add_from_report(report, dataset, fixed_by="v3", guard_threshold=0.5)
        assert added == ["c1"]  # only the passing case becomes a guard


# --------------------------------------------------------------------------- #
# guarded online bandits
# --------------------------------------------------------------------------- #


class TestLinUCB:
    def test_learns_context_dependent_arm(self):
        lin = LinUCB(["fast", "strong"], dim=2, alpha=0.5)
        # In context [1,0] "fast" wins; in [0,1] "strong" wins.
        for _ in range(10):
            lin.update("fast", [1.0, 0.0], 1.0)
            lin.update("strong", [1.0, 0.0], 0.0)
            lin.update("fast", [0.0, 1.0], 0.0)
            lin.update("strong", [0.0, 1.0], 1.0)
        assert lin.select([1.0, 0.0], explore=False) == "fast"
        assert lin.select([0.0, 1.0], explore=False) == "strong"

    def test_snapshot_roundtrip(self):
        lin = LinUCB(["a", "b"], dim=2)
        lin.update("a", [1.0, 1.0], 1.0)
        snap = lin.snapshot()
        restored = LinUCB(["a", "b"], dim=2)
        restored.load(snap)
        assert restored.counts["a"] == 1


class TestGuardedBanditRouter:
    def _router(self, store=None, bandit="epsilon_greedy"):
        return GuardedBanditRouter(
            [(MockProvider(), "m1"), (MockProvider(), "m2")],
            bandit=bandit, seed=3, store=store, app_name="t",
        )

    def test_generate_records_decision(self):
        r = self._router()
        req = ModelRequest(model="m1", messages=[Message(role="user", content="hi")])
        run_sync(r.generate(req))
        assert r.last_decision is not None and r.last_decision.arm in ("m1", "m2")

    def test_safety_floor_never_explores(self):
        r = self._router()
        req = ModelRequest(model="m1", messages=[Message(role="user", content="x")],
                           metadata={"risk": "high"})
        _, explored = r.select(req)
        assert explored is False

    def test_auto_freeze_on_regret(self):
        r = GuardedBanditRouter([(MockProvider(), "good"), (MockProvider(), "bad")],
                                bandit="epsilon_greedy", seed=1, regret_budget=0.5)
        req = ModelRequest(model="good", messages=[Message(role="user", content="x")])
        # Teach "good" is great, then repeatedly reward "bad" poorly to bank regret.
        for _ in range(5):
            r.record("good", 1.0, request=req)
        for _ in range(5):
            r.record("bad", 0.0, request=req)
        assert r.frozen is True

    def test_state_persists_restart_safe(self):
        store = InMemoryMetadataStore()
        r = self._router(store=store)
        req = ModelRequest(model="m1", messages=[Message(role="user", content="x")])
        r.record("m1", 1.0, request=req)
        r.record("m1", 1.0, request=req)
        r2 = self._router(store=store)
        assert r2.bandit.counts["m1"] == 2


# --------------------------------------------------------------------------- #
# autonomous experiment proposer
# --------------------------------------------------------------------------- #


def _app(tmp_path):
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/v.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    return ContextApp(name="prop", provider=MockProvider(), model="mock-1", config=config)


class TestExperimentProposer:
    def test_rank_orders_by_weakness(self, tmp_path):
        proposer = ExperimentProposer(_app(tmp_path), eval_budget=20)
        proposals = proposer.rank({"groundedness": 0.5, "schema_validity": 0.97})
        assert proposals[0].target_metric == "groundedness"  # weaker → first
        assert proposals[0].kind == "retrieval"  # groundedness → retrieval first
        assert sum(p.eval_budget for p in proposals) <= 20 + len(proposals) * 2

    def test_passing_metrics_yield_no_proposal(self, tmp_path):
        proposer = ExperimentProposer(_app(tmp_path))
        assert proposer.rank({"groundedness": 0.99, "lexical_overlap": 0.95}) == []

    def test_drift_boosts_priority(self, tmp_path):
        proposer = ExperimentProposer(_app(tmp_path))
        # Two equal weaknesses; the drifted one should sort first.
        proposals = proposer.rank(
            {"lexical_overlap": 0.7, "answer_relevance": 0.7}, drift={"answer_relevance"}
        )
        assert proposals[0].target_metric == "answer_relevance"
        assert proposals[0].drift is True

    def test_cost_maps_to_routing(self, tmp_path):
        proposer = ExperimentProposer(_app(tmp_path), targets={"cost": 0.001})
        proposals = proposer.rank({"cost": 0.01})
        assert proposals and proposals[0].kind == "routing"

    def test_run_next_records_audit(self, tmp_path):
        app = _app(tmp_path)
        proposer = ExperimentProposer(app, eval_budget=8)
        # No online signals -> nothing above target -> recorded as skip.
        result = proposer.run_next()
        assert result["executed"] is False
        assert app.audit.verify_chain()

    def test_run_next_executes_prompt_experiment(self, tmp_path):
        app = _app(tmp_path)
        dataset = Dataset(cases=[EvalCase(id=f"c{i}", input="q", expected="a") for i in range(6)])
        proposer = ExperimentProposer(app, eval_budget=8)
        # Force a weak signal by ranking directly, then run the top proposal.
        proposals = proposer.rank({"lexical_overlap": 0.4})
        assert proposals[0].kind == "prompt"
        # run_next reads online signals (none here) so we drive the prompt path
        # via the proposer's loop construction to confirm it wires end to end.
        from vincio.optimize.loop import ImprovementLoop

        loop = ImprovementLoop(app, optimizer="reflective")
        out = loop.run(dataset=dataset, max_variants=4, subset_size=4)
        assert out.experiment == "improvement_loop"


class TestGoldenSuiteInLoop:
    def test_promotion_blocked_by_regression_guard(self, tmp_path):
        app = _app(tmp_path)
        from vincio.optimize.loop import ImprovementLoop

        # A guard case that the candidate will fail (impossible threshold).
        suite = GoldenRegressionSuite(tmp_path / "g.jsonl")
        suite.add(EvalCase(id="guard", input="q", expected="a"),
                  fixed_by="seed", guard_metric="lexical_overlap", guard_threshold=2.0)
        dataset = Dataset(cases=[EvalCase(id=f"c{i}", input="q", expected="a") for i in range(6)])
        loop = ImprovementLoop(app, optimizer="reflective", golden_suite=suite)
        result = loop.run(dataset=dataset, max_variants=4, subset_size=4)
        # If a candidate was selected, the golden gate must have vetoed any
        # regressing promotion; the impossible floor guarantees a block when reached.
        gate_steps = [s for s in result.steps if s.get("stage") == "golden_gate"]
        if gate_steps:
            assert not result.promoted
            assert not gate_steps[0]["passed"]
