"""Tests for world-model / simulation-based planning (vincio.agents.world_model).

All deterministic and offline: a learned dynamics model is fit from recorded
reset/step transitions of the in-process reference environments, then planned
against. No model provider is involved.
"""

from __future__ import annotations

import pytest

import vincio
from vincio.agents.world_model import (
    CalibrationReport,
    ModelPredictivePlanner,
    MPCResult,
    PredictedStep,
    Transition,
    WorldModel,
    record_transitions,
    task_goal_value,
)
from vincio.core.errors import AgentEngineError
from vincio.evals.environment import (
    EnvAction,
    build_counter_environment,
    build_retail_environment,
    build_vault_environment,
)


def act(tool: str, **kwargs) -> EnvAction:
    return EnvAction(kind="tool", tool=tool, arguments=kwargs)


def retail_transitions() -> list[Transition]:
    explore = [
        [act("refund_order", order_id="O1002")],
        [act("cancel_order", order_id="O1002"), act("refund_order", order_id="O1002")],
        [act("cancel_order", order_id="O1002")],
        [act("refund_order", order_id="O1001")],
        [act("update_address", order_id="O1002", address="9 New Rd")],
        [act("get_order", order_id="O1002")],
    ]
    return record_transitions(build_retail_environment("cancel_refund"), explore)


def vault_transitions() -> list[Transition]:
    explore = [
        [act("advance"), act("advance"), act("advance"), act("open_vault")],
        [act("open_vault")],
        [act("advance"), act("open_vault")],
        [act("advance"), act("advance"), act("open_vault")],
        [act("shortcut")],
        [act("shortcut"), act("open_vault")],
        [act("advance")],
        [act("advance"), act("advance")],
        [act("advance"), act("advance"), act("advance")],
        [act("shortcut"), act("advance")],
    ]
    return record_transitions(build_vault_environment(), explore)


# ---------------------------------------------------------------------------
# Recording experience
# ---------------------------------------------------------------------------


def test_record_transitions_snapshots_are_independent():
    trs = record_transitions(
        build_retail_environment("cancel_refund"),
        [[act("cancel_order", order_id="O1002")]],
    )
    assert len(trs) == 1
    tr = trs[0]
    # The before/after snapshots must not share the live, mutated state dict.
    assert tr.observation.state["orders"]["O1002"]["status"] == "processing"
    assert tr.next_observation.state["orders"]["O1002"]["status"] == "cancelled"
    assert tr.ok and tr.reward == 1.0


def test_record_transitions_skips_non_tool_actions():
    trs = record_transitions(
        build_counter_environment(),
        [[EnvAction(kind="message", text="hello"), act("increment")]],
    )
    assert len(trs) == 1
    assert trs[0].action.tool == "increment"


def test_record_transitions_can_drop_failures():
    explore = [[act("refund_order", order_id="O1002")]]  # refund on processing fails
    kept = record_transitions(build_retail_environment(), explore, include_failures=False)
    assert kept == []
    with_failures = record_transitions(build_retail_environment(), explore)
    assert len(with_failures) == 1 and not with_failures[0].ok


# ---------------------------------------------------------------------------
# Learned dynamics
# ---------------------------------------------------------------------------


def test_unconditional_constant_effect():
    model = WorldModel(retail_transitions())
    obs = build_retail_environment("cancel_refund").observe()
    pred = model.predict(obs, act("cancel_order", order_id="O1002"))
    assert pred.known and pred.ok and pred.confidence == 1.0
    assert pred.observation.state["orders"]["O1002"]["status"] == "cancelled"
    # The source observation is not mutated by a prediction.
    assert obs.state["orders"]["O1002"]["status"] == "processing"


def test_argument_value_effect():
    model = WorldModel(retail_transitions())
    obs = build_retail_environment("update_shipping").observe()
    pred = model.predict(obs, act("update_address", order_id="O1002", address="9 New Rd"))
    assert pred.observation.state["orders"]["O1002"]["address"] == "9 New Rd"


def test_numeric_step_effect_learned():
    # increment learns new = old + 1 (an "add" template), generalizing past values.
    trs = record_transitions(
        build_counter_environment(target=3),
        [[act("increment"), act("increment"), act("increment")]],
    )
    model = WorldModel(trs)
    obs = build_counter_environment().observe()
    obs.state["count"] = 5
    pred = model.predict(obs, act("increment"))
    assert pred.observation.state["count"] == 6


def test_precondition_is_learned():
    model = WorldModel(retail_transitions())
    base = build_retail_environment("cancel_refund").observe()
    # Refund on a processing order is predicted to fail (no state change).
    fail = model.predict(base, act("refund_order", order_id="O1002"))
    assert fail.known and not fail.ok
    assert fail.observation.state["orders"]["O1002"]["refunded"] is False
    # After cancelling, the same refund is predicted to succeed.
    after_cancel = model.predict(base, act("cancel_order", order_id="O1002")).observation
    ok = model.predict(after_cancel, act("refund_order", order_id="O1002"))
    assert ok.ok and ok.observation.state["orders"]["O1002"]["refunded"] is True


def test_argument_generalization_to_unseen_entity():
    # ``cancel`` is only ever recorded on O1002; the model generalizes it to O1001.
    model = WorldModel(retail_transitions())
    obs = build_retail_environment("cancel_refund").observe()
    pred = model.predict(obs, act("cancel_order", order_id="O1001"))
    assert pred.observation.state["orders"]["O1001"]["status"] == "cancelled"


def test_unseen_signature_is_identity_with_zero_confidence():
    model = WorldModel(retail_transitions())
    obs = build_retail_environment("cancel_refund").observe()
    pred = model.predict(obs, act("teleport_order", order_id="O1002"))
    assert not pred.known and pred.confidence == 0.0
    assert pred.observation.state == obs.state  # identity prediction


def test_non_tool_action_does_not_mutate():
    model = WorldModel(retail_transitions())
    obs = build_retail_environment("cancel_refund").observe()
    finish = model.predict(obs, EnvAction(kind="finish", text="done"))
    assert finish.done and finish.observation.state == obs.state


def test_imagine_rolls_a_plan_forward():
    model = WorldModel(retail_transitions())
    obs = build_retail_environment("cancel_refund").observe()
    steps = model.imagine(
        obs,
        [act("cancel_order", order_id="O1002"), act("refund_order", order_id="O1002")],
    )
    assert len(steps) == 2
    assert all(isinstance(s, PredictedStep) for s in steps)
    final = steps[-1].observation.state["orders"]["O1002"]
    assert final["status"] == "cancelled" and final["refunded"] is True


def test_vocabulary_is_the_seen_action_set():
    model = WorldModel(vault_transitions())
    tools = {a.tool for a in model.vocabulary()}
    assert tools == {"advance", "shortcut", "open_vault"}


# ---------------------------------------------------------------------------
# Calibration gate
# ---------------------------------------------------------------------------


def test_calibration_trusts_an_accurate_model():
    trs = retail_transitions()
    model = WorldModel(trs)
    report = model.calibrate(trs)
    assert isinstance(report, CalibrationReport)
    assert report.trusted and report.state_accuracy == 1.0 and report.reward_mae == 0.0
    assert report.weight > 0.0 and model.trusted


def test_calibration_distrusts_an_inaccurate_model():
    # A model fit on the retail world does not predict the vault world.
    model = WorldModel(retail_transitions())
    report = model.calibrate(vault_transitions())
    assert not report.trusted and report.state_accuracy < 0.9 and not model.trusted


def test_calibration_on_no_transitions_is_untrusted():
    model = WorldModel(retail_transitions())
    report = model.calibrate([])
    assert report.n == 0 and not report.trusted


# ---------------------------------------------------------------------------
# Model-predictive planning
# ---------------------------------------------------------------------------


def test_mpc_plans_the_retail_task():
    model = WorldModel(retail_transitions())
    model.calibrate(retail_transitions())
    result = ModelPredictivePlanner(model, horizon=3, beam_width=16).plan(
        build_retail_environment("cancel_refund")
    )
    assert isinstance(result, MPCResult)
    assert result.success and result.real_steps == 2
    assert [a.tool for a in result.committed] == ["cancel_order", "refund_order"]
    assert result.calibrated and result.planning_weight > 0.0
    assert len(result.steps) == 2 and result.steps[0].action.tool == "cancel_order"


def test_planning_beats_reactive_at_a_fixed_budget():
    model = WorldModel(vault_transitions())
    model.calibrate(vault_transitions())
    budget = 6
    reactive = ModelPredictivePlanner(
        model, horizon=1, beam_width=64, max_real_steps=budget
    ).plan(build_vault_environment())
    planned = ModelPredictivePlanner(
        model, horizon=5, beam_width=64, max_real_steps=budget
    ).plan(build_vault_environment())
    # The reactive (one-step) planner takes the locally-attractive shortcut and is
    # trapped; the imagined-rollout planner avoids it and opens the vault.
    assert not reactive.success
    assert planned.success and planned.real_steps <= budget
    assert "shortcut" not in {a.tool for a in planned.committed}


def test_planner_refuses_an_uncalibrated_model():
    model = WorldModel(vault_transitions())  # fit but not calibrated
    with pytest.raises(AgentEngineError, match="not calibrated"):
        ModelPredictivePlanner(model, horizon=5).plan(build_vault_environment())


def test_planner_can_opt_out_of_the_calibration_gate():
    model = WorldModel(retail_transitions())  # not calibrated
    result = ModelPredictivePlanner(
        model, horizon=3, beam_width=16, require_calibrated=False
    ).plan(build_retail_environment("cancel_refund"))
    assert result.success and not result.calibrated


def test_planner_accepts_an_explicit_action_set():
    model = WorldModel(retail_transitions())
    model.calibrate(retail_transitions())
    actions = [
        act("cancel_order", order_id="O1002"),
        act("refund_order", order_id="O1002"),
    ]
    result = ModelPredictivePlanner(model, actions=actions, horizon=3, beam_width=8).plan(
        build_retail_environment("cancel_refund")
    )
    assert result.success


def test_task_goal_value_counts_satisfied_checks():
    env = build_vault_environment()
    value = task_goal_value(env.task.checks)
    assert value(env.observe()) == 0.0  # nothing satisfied at the start
    env.step(act("shortcut"))  # progress reaches the threshold (one of two checks)
    assert value(env.observe()) == 0.5


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_exports():
    for name in (
        "WorldModel",
        "Transition",
        "PredictedStep",
        "CalibrationReport",
        "ModelPredictivePlanner",
        "MPCStep",
        "MPCResult",
        "record_transitions",
        "task_goal_value",
    ):
        assert name in vincio.__all__
        assert hasattr(vincio, name)
