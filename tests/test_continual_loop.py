"""1.10 — the continual loop: drift detectors, online state persistence, and the
online improvement controller (gated re-eval / re-optimization / rollback)."""

import warnings

import pytest

from vincio import ContextApp, VincioConfig
from vincio.evals import Dataset, EvalCase
from vincio.evals.drift import (
    CUSUMDetector,
    DriftMonitor,
    ks_drift,
    ks_statistic,
    psi,
    rbf_mmd2,
)
from vincio.evals.online import OnlineEvaluator
from vincio.optimize.controller import ContinuousImprovementController
from vincio.prompts.registry import PromptRegistry
from vincio.providers import MockProvider
from vincio.storage.base import InMemoryMetadataStore

warnings.simplefilter("ignore")  # experimental-API warnings are not under test here


# --------------------------------------------------------------------------- #
# distributional drift statistics
# --------------------------------------------------------------------------- #


class TestDriftStatistics:
    def test_ks_zero_for_identical(self):
        assert ks_statistic([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 0.0

    def test_ks_one_for_disjoint(self):
        assert ks_statistic([1, 2, 3], [10, 11, 12]) == pytest.approx(1.0)

    def test_ks_drift_verdict(self):
        d, p, drifted = ks_drift(list(range(20)), [x + 50 for x in range(20)])
        assert drifted and d == pytest.approx(1.0) and p < 0.05

    def test_ks_no_drift_same_distribution(self):
        _, _, drifted = ks_drift(list(range(20)), list(range(20)))
        assert not drifted

    def test_psi_zero_for_identical(self):
        assert psi(list(range(20)), list(range(20))) == pytest.approx(0.0, abs=1e-9)

    def test_psi_large_for_shift(self):
        assert psi(list(range(20)), [x + 100 for x in range(20)]) > 0.25

    def test_mmd_zero_for_identical(self):
        assert rbf_mmd2([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(0.0, abs=1e-9)

    def test_mmd_positive_for_shift(self):
        assert rbf_mmd2([1, 2, 3, 4], [20, 21, 22, 23]) > 0.0

    def test_mmd_accepts_vectors(self):
        a = [[0.0, 0.0], [0.1, 0.1]]
        b = [[5.0, 5.0], [5.1, 5.1]]
        assert rbf_mmd2(a, b) > 0.0


class TestCUSUM:
    def test_stable_stream_does_not_fire(self):
        c = CUSUMDetector(target=0.9, sigma=0.05, slack=0.5, threshold=4.0)
        fired = [c.observe(v) for v in [0.9, 0.91, 0.89, 0.9, 0.9, 0.88, 0.9]]
        assert not any(fired)
        assert c.changepoints == 0

    def test_sustained_drop_fires_downward(self):
        c = CUSUMDetector(target=0.9, sigma=0.05, slack=0.5, threshold=3.0)
        fired = [c.observe(v) for v in [0.6, 0.55, 0.5, 0.5, 0.5, 0.5]]
        assert any(fired)
        assert c.last_direction == "down"

    def test_state_roundtrip(self):
        c = CUSUMDetector(target=0.5, sigma=0.1)
        for v in [0.4, 0.45, 0.42]:
            c.observe(v)
        state = c.state()
        restored = CUSUMDetector(target=0.5, sigma=0.1)
        restored.load(state)
        assert restored.s_hi == c.s_hi and restored.s_lo == c.s_lo and restored.n == c.n


class TestDriftMonitorStreaming:
    def test_observe_score_emits_cusum_event(self):
        events = []
        bus_seen = []

        class _Bus:
            def emit(self, name, payload=None, **kw):
                bus_seen.append(name)
                events.append((name, payload))

        store = InMemoryMetadataStore()
        mon = DriftMonitor(bus=_Bus(), store=store, app_name="t", cusum_threshold=3.0)
        mon.set_score_baseline("groundedness", [0.9, 0.91, 0.89, 0.9, 0.9])
        # Streaming declining scores should eventually fire a cusum drift event.
        reports = [mon.observe_score("groundedness", v) for v in [0.5, 0.45, 0.4, 0.4, 0.4]]
        assert any(r is not None and r.method == "cusum" for r in reports)
        assert "drift.detected" in bus_seen
        # CUSUM state persisted to the store (restart-safe).
        rows = store.query("drift_state", where={"app_id": "t"})
        assert rows and rows[0]["metric"] == "groundedness"

    def test_load_state_restores_accumulators(self):
        store = InMemoryMetadataStore()
        mon = DriftMonitor(store=store, app_name="t")
        mon.set_score_baseline("m", [1.0, 1.0, 1.0, 1.0])
        mon.observe_score("m", 0.5)
        saved = mon._cusum["m"].state()
        mon2 = DriftMonitor(store=store, app_name="t")
        mon2.set_score_baseline("m", [1.0, 1.0, 1.0, 1.0])
        assert mon2.load_state() == 1
        assert mon2._cusum["m"].n == saved["n"]

    def test_check_distribution_methods(self):
        mon = DriftMonitor(app_name="t")
        base = list(range(30))
        mon.set_distribution_baseline("inputs", [float(x) for x in base])
        ks = mon.check_distribution("inputs", [x + 100 for x in base], method="ks")
        assert ks.method == "ks" and ks.drifted
        psi_r = mon.check_distribution("inputs", [x + 100 for x in base], method="psi")
        assert psi_r.method == "psi" and psi_r.drifted


# --------------------------------------------------------------------------- #
# online evaluator persistence
# --------------------------------------------------------------------------- #


class TestOnlineEvaluatorState:
    def test_counter_persists_across_restart(self):
        store = InMemoryMetadataStore()
        ev = OnlineEvaluator("groundedness", sample_rate=0.5, store=store, app_name="a")
        from vincio.evals.metrics import RunOutput

        out = RunOutput(raw_text="x", metadata={"input": "q"})
        for _ in range(6):
            ev.observe(out, run_id="r")
        first_counter = ev._counter
        assert first_counter == 6
        # A fresh evaluator (simulating a restart) resumes the counter.
        ev2 = OnlineEvaluator("groundedness", sample_rate=0.5, store=store, app_name="a")
        assert ev2._counter == first_counter

    def test_observed_total_aggregates_workers(self):
        store = InMemoryMetadataStore()
        from vincio.evals.metrics import RunOutput

        out = RunOutput(raw_text="x", metadata={"input": "q"})
        # Fractional sampling increments the persisted per-worker counter.
        w1 = OnlineEvaluator("groundedness", sample_rate=0.5, store=store, app_name="a", worker_id="w1")
        w2 = OnlineEvaluator("groundedness", sample_rate=0.5, store=store, app_name="a", worker_id="w2")
        for _ in range(4):
            w1.observe(out, run_id="r")
        for _ in range(6):
            w2.observe(out, run_id="r")
        # observed_total aggregates both workers' counters off the shared store.
        assert w1.observed_total() == 10
        # The online series itself also aggregates across workers (one app series).
        assert len(w1.series()) == len(w2.series())


# --------------------------------------------------------------------------- #
# the continuous improvement controller
# --------------------------------------------------------------------------- #


def _app(tmp_path):
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/vincio.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    app = ContextApp(name="ctl", provider=MockProvider(), model="mock-1", config=config)
    return app


def _golden():
    return Dataset(
        name="held_out",
        cases=[EvalCase(id=f"g{i}", input="what is the refund window?", expected="30 days")
               for i in range(4)],
    )


class TestController:
    def test_observing_until_sustain_reached(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, metrics=["groundedness"], sustain=3)
        d1 = ctl.evaluate("groundedness", {"method": "cusum"})
        assert d1.action == "observing" and d1.sustain_count == 1
        d2 = ctl.evaluate("groundedness", {"method": "cusum"})
        assert d2.action == "observing" and d2.sustain_count == 2

    def test_budget_exhaustion_blocks_action(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(
            app, metrics=["groundedness"], sustain=1, eval_budget=0.0
        )
        d = ctl.evaluate("groundedness", {"method": "cusum"})
        assert d.action == "budget_exhausted"

    def test_cooldown_debounces(self, tmp_path):
        app = _app(tmp_path)
        clock = {"t": 1000.0}
        registry = PromptRegistry(directory=str(tmp_path / "prompts"))
        registry.push(app.prompt_spec, tags=["production"])
        registry.push(app.prompt_spec.model_copy(update={"objective": "v2 changed"}))
        app.prompt_spec = registry.get(app.prompt_spec.name).spec
        ctl = ContinuousImprovementController(
            app, metrics=["safety"], sustain=1, cooldown_s=300.0,
            registry=registry, prompt_name=app.prompt_spec.name, clock=lambda: clock["t"],
        )
        first = ctl.evaluate("safety", {"method": "cusum"})
        assert first.action == "rolled_back"
        clock["t"] += 10.0  # still inside cooldown
        second = ctl.evaluate("safety", {"method": "cusum"})
        assert second.action == "debounced"

    def test_safety_regression_rolls_back(self, tmp_path):
        app = _app(tmp_path)
        registry = PromptRegistry(directory=str(tmp_path / "prompts"))
        good = registry.push(app.prompt_spec, tags=["production"])
        registry.push(app.prompt_spec.model_copy(update={"objective": "regressed"}))
        app.prompt_spec = registry.get(app.prompt_spec.name).spec
        ctl = ContinuousImprovementController(
            app, metrics=["safety"], sustain=1, registry=registry,
            prompt_name=app.prompt_spec.name,
        )
        d = ctl.evaluate("safety", {"method": "cusum"})
        assert d.action == "rolled_back"
        assert d.rolled_back_to == good.ref
        assert "regressed" not in app.prompt_spec.objective
        # The decision is on the audit chain.
        assert app.audit.verify_chain()

    def test_reeval_clears_false_alarm(self, tmp_path):
        app = _app(tmp_path)
        registry = PromptRegistry(directory=str(tmp_path / "prompts"))
        registry.push(app.prompt_spec, tags=["production"])
        registry.push(app.prompt_spec.model_copy(update={"objective": "head"}))
        # quality_floor=0.0 means any non-negative score clears the alarm.
        ctl = ContinuousImprovementController(
            app, metrics=["groundedness"], sustain=1, golden=_golden(),
            quality_floor={"groundedness": 0.0}, reoptimize=False, registry=registry,
        )
        d = ctl.evaluate("groundedness", {"method": "ks"})
        assert d.action == "reeval_clear" and d.confirmed is False

    def test_confirmed_regression_without_reopt_rolls_back(self, tmp_path):
        app = _app(tmp_path)
        registry = PromptRegistry(directory=str(tmp_path / "prompts"))
        good = registry.push(app.prompt_spec, tags=["production"])
        registry.push(app.prompt_spec.model_copy(update={"objective": "head"}))
        app.prompt_spec = registry.get(app.prompt_spec.name).spec
        # quality_floor very high -> eval value always below it -> confirmed regression.
        ctl = ContinuousImprovementController(
            app, metrics=["groundedness"], sustain=1, golden=_golden(),
            quality_floor={"groundedness": 5.0}, reoptimize=False, registry=registry,
            prompt_name=app.prompt_spec.name,
        )
        d = ctl.evaluate("groundedness", {"method": "ks"})
        assert d.confirmed is True
        assert d.action == "rolled_back" and d.rolled_back_to == good.ref

    def test_state_persists_restart_safe(self, tmp_path):
        app = _app(tmp_path)
        ctl = ContinuousImprovementController(app, metrics=["groundedness"], sustain=5)
        ctl.evaluate("groundedness", {"method": "cusum"})
        ctl.evaluate("groundedness", {"method": "cusum"})
        # A fresh controller on the same store resumes the sustain count.
        ctl2 = ContinuousImprovementController(app, metrics=["groundedness"], sustain=5)
        assert ctl2._sustain.get("groundedness") == 2

    def test_event_wiring_streams_scores(self, tmp_path):
        app = _app(tmp_path)
        registry = PromptRegistry(directory=str(tmp_path / "prompts"))
        registry.push(app.prompt_spec, tags=["production"])
        registry.push(app.prompt_spec.model_copy(update={"objective": "head"}))
        app.prompt_spec = registry.get(app.prompt_spec.name).spec
        ctl = ContinuousImprovementController(
            app, metrics=["groundedness"], sustain=1, registry=registry,
            prompt_name=app.prompt_spec.name, golden=_golden(),
            quality_floor={"groundedness": 5.0}, reoptimize=False,
        )
        ctl.set_baseline("groundedness", [0.9, 0.91, 0.9, 0.9, 0.9]).attach()
        # Drive declining online scores through the bus; CUSUM should fire and act.
        for v in [0.4, 0.35, 0.3, 0.3, 0.3, 0.3]:
            app.events.emit("eval.online", {"metric": "groundedness", "value": v})
        actions = [d.action for d in ctl.decisions]
        assert "rolled_back" in actions
        ctl.detach()
