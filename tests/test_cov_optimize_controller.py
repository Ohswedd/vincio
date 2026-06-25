"""Real-behavior coverage for the continuous improvement controller.

Targets the uncovered paths of ``vincio.optimize.controller``: seeding
baselines from online series, the event-bus filter branches, the gated
``_act`` fan-out (severe rollback / re-eval clear / confirmed-regression
rollback / re-optimization), and the ``_rollback`` / ``_last_known_good`` /
state-persistence edge cases. Everything runs offline against the
deterministic ``MockProvider`` and an in-memory/sqlite store -- no mocks.
"""

import warnings

from vincio import ContextApp, VincioConfig
from vincio.evals import Dataset, EvalCase
from vincio.evals.metrics import RunOutput
from vincio.optimize.controller import (
    ContinuousImprovementController,
    ControllerDecision,
)
from vincio.prompts.registry import PromptRegistry
from vincio.providers import MockProvider

warnings.simplefilter("ignore")  # experimental-API warnings are not under test here


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _app(tmp_path):
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/vincio.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    return ContextApp(name="ctl", provider=MockProvider(), model="mock-1", config=config)


def _golden(n=4):
    return Dataset(
        name="held_out",
        cases=[
            EvalCase(id=f"g{i}", input="what is the refund window?", expected="30 days")
            for i in range(n)
        ],
    )


def _registry_with_head(app, tmp_path):
    """A registry with a production-tagged good version below a regressed head."""
    registry = PromptRegistry(directory=str(tmp_path / "prompts"))
    good = registry.push(app.prompt_spec, tags=["production"])
    registry.push(app.prompt_spec.model_copy(update={"objective": "regressed head"}))
    app.prompt_spec = registry.get(app.prompt_spec.name).spec
    return registry, good


# --------------------------------------------------------------------------- #
# construction defaults
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_metrics_default_to_lexical_overlap_when_no_evaluators(self, tmp_path):
        app = _app(tmp_path)
        assert app.online_evaluators == []
        ctl = ContinuousImprovementController(app)
        assert ctl.metrics == ["lexical_overlap"]

    def test_metrics_default_from_registered_online_evaluators(self, tmp_path):
        app = _app(tmp_path)
        app.add_online_evaluator("groundedness", sample_rate=1.0)
        app.add_online_evaluator("lexical_overlap", sample_rate=1.0)
        ctl = ContinuousImprovementController(app)
        assert ctl.metrics == ["groundedness", "lexical_overlap"]

    def test_prompt_name_defaults_to_app_spec_name(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app)
        assert ctl.prompt_name == app.prompt_spec.name

    def test_sustain_floored_at_one(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, sustain=0)
        assert ctl.sustain == 1

    def test_default_registry_constructed_when_absent(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app)
        assert isinstance(ctl.registry, PromptRegistry)


# --------------------------------------------------------------------------- #
# seed_from_online
# --------------------------------------------------------------------------- #


class TestSeedFromOnline:
    def test_seeds_baseline_from_online_series(self, tmp_path):
        app = _app(tmp_path)
        app.add_online_evaluator("lexical_overlap", sample_rate=1.0)
        ev = app.online_evaluators[0]
        out = RunOutput(raw_text="30 days", metadata={"input": "q", "expected": "30 days"})
        for _ in range(5):
            ev.observe(out, run_id="r")
        ctl = ContinuousImprovementController(app, metrics=["lexical_overlap"])
        # No baseline before seeding.
        assert "lexical_overlap" not in ctl.monitor._cusum
        returned = ctl.seed_from_online(window=10)
        # Fluent: returns self, and a CUSUM detector now exists for the metric.
        assert returned is ctl
        assert "lexical_overlap" in ctl.monitor._cusum

    def test_seed_skips_evaluators_not_in_watched_metrics(self, tmp_path):
        app = _app(tmp_path)
        app.add_online_evaluator("groundedness", sample_rate=1.0)
        ev = app.online_evaluators[0]
        out = RunOutput(raw_text="x", metadata={"input": "q", "expected": "y"})
        for _ in range(5):
            ev.observe(out, run_id="r")
        # Controller only watches a DIFFERENT metric -> groundedness is skipped.
        ctl = ContinuousImprovementController(app, metrics=["lexical_overlap"])
        ctl.seed_from_online()
        assert "groundedness" not in ctl.monitor._cusum

    def test_seed_skips_when_fewer_than_two_samples(self, tmp_path):
        app = _app(tmp_path)
        app.add_online_evaluator("lexical_overlap", sample_rate=1.0)
        ev = app.online_evaluators[0]
        out = RunOutput(raw_text="x", metadata={"input": "q", "expected": "y"})
        ev.observe(out, run_id="r")  # a single sample -> < 2, no baseline set
        ctl = ContinuousImprovementController(app, metrics=["lexical_overlap"])
        ctl.seed_from_online()
        assert "lexical_overlap" not in ctl.monitor._cusum


# --------------------------------------------------------------------------- #
# event-handler filter branches
# --------------------------------------------------------------------------- #


class TestEventHandlers:
    def test_online_eval_ignores_unwatched_metric(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, metrics=["groundedness"])
        ctl.set_baseline("groundedness", [0.9, 0.9, 0.9, 0.9])
        before = ctl.monitor._cusum["groundedness"].n
        ctl._on_online_eval(_event("eval.online", {"metric": "other", "value": 0.1}))
        # Unwatched metric -> not streamed into the watched detector.
        assert ctl.monitor._cusum["groundedness"].n == before

    def test_online_eval_ignores_missing_value(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, metrics=["groundedness"])
        ctl.set_baseline("groundedness", [0.9, 0.9, 0.9, 0.9])
        before = ctl.monitor._cusum["groundedness"].n
        ctl._on_online_eval(_event("eval.online", {"metric": "groundedness"}))
        assert ctl.monitor._cusum["groundedness"].n == before

    def test_online_eval_streams_watched_value(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, metrics=["groundedness"])
        ctl.set_baseline("groundedness", [0.9, 0.9, 0.9, 0.9])
        before = ctl.monitor._cusum["groundedness"].n
        ctl._on_online_eval(_event("eval.online", {"metric": "groundedness", "value": 0.3}))
        assert ctl.monitor._cusum["groundedness"].n == before + 1

    def test_on_drift_ignores_unwatched_metric(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, metrics=["groundedness"], sustain=1)
        ctl._on_drift(_event("drift.detected", {"metric": "unwatched", "method": "ks"}))
        # The handler returned before evaluate(); no decision recorded.
        assert ctl.decisions == []

    def test_on_drift_dispatches_watched_metric(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, metrics=["groundedness"], sustain=5)
        ctl._on_drift(_event("drift.detected", {"metric": "groundedness", "method": "ks"}))
        assert len(ctl.decisions) == 1
        assert ctl.decisions[0].metric == "groundedness"

    def test_on_drift_empty_metric_falls_back_to_first_watched(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, metrics=["m1", "m2"], sustain=5)
        ctl._on_drift(_event("drift.detected", {"method": "cusum"}))
        assert ctl.decisions[-1].metric == "m1"


def _event(name, payload):
    from vincio.core.events import Event

    return Event(name=name, payload=payload)


# --------------------------------------------------------------------------- #
# evaluate() debounce / budget gates
# --------------------------------------------------------------------------- #


class TestEvaluateGates:
    def test_observing_below_sustain(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, metrics=["m"], sustain=3)
        d = ctl.evaluate("m", {"method": "cusum"})
        assert d.action == "observing"
        assert d.reason == "1/3 sustained signals; observing"

    def test_debounced_within_cooldown(self, tmp_path):
        app = _app(tmp_path)
        clock = {"t": 1000.0}
        registry, _ = _registry_with_head(app, tmp_path)
        ctl = ContinuousImprovementController(
            app, metrics=["safety"], sustain=1, cooldown_s=300.0,
            registry=registry, prompt_name=app.prompt_spec.name, clock=lambda: clock["t"],
        )
        first = ctl.evaluate("safety", {"method": "cusum"})
        assert first.action == "rolled_back"
        clock["t"] += 42.0  # still inside the 300s cooldown
        second = ctl.evaluate("safety", {"method": "cusum"})
        assert second.action == "debounced"
        assert "42s since last action" in second.reason

    def test_budget_exhausted_after_sustain(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(
            app, metrics=["m"], sustain=1, eval_budget=0.0,
        )
        d = ctl.evaluate("m", {"method": "cusum"})
        assert d.action == "budget_exhausted"
        assert "budget 0 spent" in d.reason


# --------------------------------------------------------------------------- #
# _confirm_regression branches
# --------------------------------------------------------------------------- #


class TestConfirmRegression:
    def test_no_floor_treats_drift_as_authoritative(self, tmp_path):
        app = _app(tmp_path)
        registry, good = _registry_with_head(app, tmp_path)
        # No quality_floor entry for the metric -> _confirm_regression returns True.
        ctl = ContinuousImprovementController(
            app, metrics=["groundedness"], sustain=1, golden=_golden(),
            reoptimize=False, registry=registry, prompt_name=app.prompt_spec.name,
        )
        d = ctl.evaluate("groundedness", {"method": "ks"})
        assert d.confirmed is True
        assert d.action == "rolled_back"
        assert d.budget_spent == float(len(_golden()))  # one golden replay charged

    def test_lower_is_better_below_floor_clears(self, tmp_path):
        app = _app(tmp_path)
        registry = PromptRegistry(directory=str(tmp_path / "prompts"))
        registry.push(app.prompt_spec, tags=["production"])
        registry.push(app.prompt_spec.model_copy(update={"objective": "head"}))
        # cost is lower-is-better; measured cost (0.0) is below the floor
        # -> NOT a regression -> the alarm is cleared. (line: return value > floor)
        ctl = ContinuousImprovementController(
            app, metrics=["cost"], sustain=1, golden=_golden(),
            quality_floor={"cost": 1e9}, reoptimize=False, registry=registry,
        )
        d = ctl.evaluate("cost", {"method": "ks"})
        assert d.confirmed is False
        assert d.action == "reeval_clear"

    def test_higher_is_better_above_floor_clears(self, tmp_path):
        app = _app(tmp_path)
        registry = PromptRegistry(directory=str(tmp_path / "prompts"))
        registry.push(app.prompt_spec, tags=["production"])
        registry.push(app.prompt_spec.model_copy(update={"objective": "head"}))
        # groundedness is higher-is-better; measured 0.0 is NOT below a negative
        # floor (0.0 < -1.0 is False) -> no regression -> cleared. (return value < floor)
        ctl = ContinuousImprovementController(
            app, metrics=["groundedness"], sustain=1, golden=_golden(),
            quality_floor={"groundedness": -1.0}, reoptimize=False, registry=registry,
        )
        d = ctl.evaluate("groundedness", {"method": "ks"})
        assert d.confirmed is False
        assert d.action == "reeval_clear"

    def test_confirm_regression_returns_true_when_golden_is_none(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, metrics=["groundedness"], golden=None)
        # Defensive guard: with no golden set, confirmation defaults to True and
        # charges nothing to the budget.
        assert ctl._confirm_regression("groundedness") is True
        assert ctl._budget_spent == 0.0

    def test_lower_is_better_above_floor_confirms(self, tmp_path):
        app = _app(tmp_path)
        registry, good = _registry_with_head(app, tmp_path)
        # cost lower-is-better; measured cost 0.0 exceeds a negative floor
        # (0.0 > -1.0) -> regression confirmed -> rollback (reoptimize off).
        ctl = ContinuousImprovementController(
            app, metrics=["cost"], sustain=1, golden=_golden(),
            quality_floor={"cost": -1.0}, reoptimize=False, registry=registry,
            prompt_name=app.prompt_spec.name,
        )
        d = ctl.evaluate("cost", {"method": "ks"})
        assert d.confirmed is True
        assert d.action == "rolled_back"


# --------------------------------------------------------------------------- #
# _reoptimize
# --------------------------------------------------------------------------- #


class TestReoptimize:
    def test_reoptimize_skipped_on_insufficient_budget(self, tmp_path):
        app = _app(tmp_path)
        registry, good = _registry_with_head(app, tmp_path)
        # eval_budget tiny: one golden replay (cost 4) leaves < 2 rollouts,
        # so _reoptimize returns False and the controller falls through to rollback.
        ctl = ContinuousImprovementController(
            app, metrics=["groundedness"], sustain=1, golden=_golden(),
            reoptimize=True, eval_budget=5.0, registry=registry,
            prompt_name=app.prompt_spec.name,
        )
        d = ctl.evaluate("groundedness", {"method": "ks"})
        assert d.details.get("reoptimize_skipped") == "insufficient budget"
        assert d.action == "rolled_back"  # fell through after failed re-opt

    def test_reoptimize_runs_records_loop_reason(self, tmp_path):
        app = _app(tmp_path)
        registry, good = _registry_with_head(app, tmp_path)
        ctl = ContinuousImprovementController(
            app, metrics=["lexical_overlap"], sustain=1, golden=_golden(),
            reoptimize=True, eval_budget=48.0, registry=registry,
            prompt_name=app.prompt_spec.name,
        )
        d = ctl.evaluate("lexical_overlap", {"method": "ks"})
        # A real ImprovementLoop ran; either it promoted (reoptimized) or it could
        # not recover and we fell through to rollback. Either way the loop's
        # reason is recorded and the action is one of the two reopt outcomes.
        assert "reoptimize_reason" in d.details
        assert d.action in {"reoptimized", "rolled_back", "no_action"}
        # The re-optimization charged the remaining budget as rollouts.
        assert ctl._budget_spent > len(_golden())

    def test_reoptimization_promotes_a_winner(self, tmp_path):
        # The ImprovementLoop's significance gate is stochastic against the
        # MockProvider, so retry across fresh controllers until one promotes and
        # assert on the promote path (decision wiring + restored baseline).
        promoted = None
        for attempt in range(16):
            sub = tmp_path / f"a{attempt}"
            sub.mkdir()
            app = _app(sub)
            registry, _ = _registry_with_head(app, sub)
            ctl = ContinuousImprovementController(
                app, metrics=["lexical_overlap"], sustain=1, golden=_golden(),
                reoptimize=True, eval_budget=48.0, registry=registry,
                prompt_name=app.prompt_spec.name,
            )
            d = ctl.evaluate("lexical_overlap", {"method": "ks"})
            if d.action == "reoptimized":
                promoted = d
                break
        assert promoted is not None, "loop never promoted across 16 attempts"
        assert promoted.promoted_ref is not None
        assert promoted.rolled_back_to is None  # promotion, not a rollback
        # The decision reason names the re-optimized prompt and the loop's reason.
        assert ctl.prompt_name in promoted.reason
        assert promoted.details["reoptimize_reason"] in promoted.reason


# --------------------------------------------------------------------------- #
# _rollback and _last_known_good
# --------------------------------------------------------------------------- #


class TestRollback:
    def test_no_registry_history_is_no_action(self, tmp_path):
        app = _app(tmp_path)
        # Empty registry: registry.versions(name) raises PromptError -> caught,
        # decision becomes no_action rather than crashing.
        ctl = ContinuousImprovementController(
            app, metrics=["safety"], sustain=1,
            registry=PromptRegistry(directory=str(tmp_path / "empty")),
            prompt_name="does-not-exist",
        )
        d = ctl.evaluate("safety", {"method": "cusum"})
        assert d.action == "no_action"
        assert "no registry history" in d.reason

    def test_single_version_has_no_known_good(self, tmp_path):
        app = _app(tmp_path)
        registry = PromptRegistry(directory=str(tmp_path / "prompts"))
        registry.push(app.prompt_spec)  # exactly one version -> < 2 -> None
        ctl = ContinuousImprovementController(
            app, metrics=["safety"], sustain=1, registry=registry,
            prompt_name=app.prompt_spec.name,
        )
        d = ctl.evaluate("safety", {"method": "cusum"})
        assert d.action == "no_action"
        assert "no earlier known-good" in d.reason

    def test_known_good_falls_back_to_prior_version_without_production_tag(self, tmp_path):
        app = _app(tmp_path)
        registry = PromptRegistry(directory=str(tmp_path / "prompts"))
        prior = registry.push(app.prompt_spec)  # no production tag
        registry.push(app.prompt_spec.model_copy(update={"objective": "head"}))
        app.prompt_spec = registry.get(app.prompt_spec.name).spec
        ctl = ContinuousImprovementController(
            app, metrics=["safety"], sustain=1, registry=registry,
            prompt_name=app.prompt_spec.name,
        )
        d = ctl.evaluate("safety", {"method": "cusum"})
        # No production tag anywhere -> rolls back to the immediately prior version.
        assert d.action == "rolled_back"
        assert d.rolled_back_to == prior.ref

    def test_non_severe_no_golden_rolls_back_directly(self, tmp_path):
        app = _app(tmp_path)
        registry, good = _registry_with_head(app, tmp_path)
        # Non-safety metric, no golden set, reoptimize off: skips both the
        # re-eval and re-optimize stages and rolls back straight away.
        ctl = ContinuousImprovementController(
            app, metrics=["groundedness"], sustain=1, golden=None,
            reoptimize=False, registry=registry, prompt_name=app.prompt_spec.name,
        )
        d = ctl.evaluate("groundedness", {"method": "ks"})
        assert d.confirmed is None  # no re-eval ran
        assert d.action == "rolled_back"
        assert d.rolled_back_to == good.ref

    def test_rollback_emits_event_and_restores_spec(self, tmp_path):
        app = _app(tmp_path)
        registry, good = _registry_with_head(app, tmp_path)
        seen = []
        app.events.subscribe("improvement.rolled_back", lambda e: seen.append(e.payload))
        ctl = ContinuousImprovementController(
            app, metrics=["safety"], sustain=1, registry=registry,
            prompt_name=app.prompt_spec.name,
        )
        d = ctl.evaluate("safety", {"method": "cusum"})
        assert d.action == "rolled_back" and d.rolled_back_to == good.ref
        # The live app prompt spec was reverted off the regressed head.
        assert "regressed" not in app.prompt_spec.objective
        # The rollback event carried the restored ref.
        assert seen and seen[0]["restored"] == good.ref
        assert seen[0]["metric"] == "safety"
        assert "safety regression" in d.reason


# --------------------------------------------------------------------------- #
# helpers and state persistence
# --------------------------------------------------------------------------- #


class TestHelpersAndState:
    def test_remaining_rollouts_never_negative(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, eval_budget=10.0)
        ctl._budget_spent = 25.0  # overspent
        assert ctl._remaining_rollouts() == 0

    def test_remaining_rollouts_truncates_to_int(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, eval_budget=10.5)
        ctl._budget_spent = 3.0
        assert ctl._remaining_rollouts() == 7  # int(7.5)

    def test_save_and_load_state_no_op_without_store(self, tmp_path):
        app = _app(tmp_path)
        # A store-less app (e.g. ephemeral deployment): both save_state and
        # load_state must short-circuit instead of dereferencing a None store.
        app.store = None
        ctl = ContinuousImprovementController(app, metrics=["groundedness"], sustain=5)
        # With no store, evaluate() still works; save/load are pure no-ops.
        ctl.evaluate("groundedness", {"method": "cusum"})
        ctl.evaluate("groundedness", {"method": "cusum"})
        assert ctl._sustain["groundedness"] == 2
        # A fresh controller cannot resume (nothing was persisted).
        ctl2 = ContinuousImprovementController(app, metrics=["groundedness"], sustain=5)
        assert ctl2._sustain.get("groundedness", 0) == 0

    def test_state_persists_and_restores_budget_and_sustain(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, metrics=["groundedness"], sustain=5)
        ctl._budget_spent = 12.0
        ctl.evaluate("groundedness", {"method": "cusum"})  # save_state writes budget+sustain
        ctl2 = ContinuousImprovementController(app, metrics=["groundedness"], sustain=5)
        assert ctl2._budget_spent == 12.0
        assert ctl2._sustain.get("groundedness") == 1

    def test_record_emits_decision_event_and_audits(self, tmp_path):
        app = _app(tmp_path)
        seen = []
        app.events.subscribe("improvement.decision", lambda e: seen.append(e.payload))
        ctl = ContinuousImprovementController(app, metrics=["groundedness"], sustain=5)
        d = ctl.evaluate("groundedness", {"method": "cusum"})
        assert seen and seen[0]["action"] == d.action == "observing"
        # observing -> audit recorded as a skip, chain stays verifiable.
        assert app.audit.verify_chain()


# --------------------------------------------------------------------------- #
# attach / detach lifecycle
# --------------------------------------------------------------------------- #


class TestAttachDetach:
    def test_detach_unsubscribes_all_handlers(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, metrics=["groundedness"], sustain=1)
        ctl.attach()
        assert len(ctl._unsubscribe) == 2
        ctl.detach()
        assert ctl._unsubscribe == []
        # After detach, an online eval event is no longer streamed.
        ctl.set_baseline("groundedness", [0.9, 0.9, 0.9, 0.9])
        before = ctl.monitor._cusum["groundedness"].n
        app.events.emit("eval.online", {"metric": "groundedness", "value": 0.2})
        assert ctl.monitor._cusum["groundedness"].n == before


# --------------------------------------------------------------------------- #
# ControllerDecision model defaults
# --------------------------------------------------------------------------- #


def test_controller_decision_defaults():
    d = ControllerDecision()
    assert d.trigger == "drift"
    assert d.action == "observing"
    assert d.confirmed is None
    assert d.sustain_count == 0
    assert d.details == {}
