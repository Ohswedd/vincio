"""Tests for the stateful-environment eval harness (2.2)."""

from __future__ import annotations

import pytest

from vincio.evals import (
    EnvAction,
    Environment,
    EnvironmentSimulator,
    TaskVerification,
    make_counter_environment,
    make_retail_environment,
    scripted_policy,
    task_success,
)
from vincio.evals.datasets import EvalCase
from vincio.evals.metrics import METRICS, RunOutput


def test_reference_env_satisfies_protocol():
    env = make_retail_environment("cancel_refund")
    assert isinstance(env, Environment)


def test_retail_success_oracle_verifies_end_state():
    env = make_retail_environment("cancel_refund")
    policy = scripted_policy(
        [
            EnvAction(tool="get_order", arguments={"order_id": "O1002"}),
            EnvAction(tool="cancel_order", arguments={"order_id": "O1002"}),
            EnvAction(tool="refund_order", arguments={"order_id": "O1002"}),
            EnvAction(kind="finish", text="done: cancelled and refunded"),
        ]
    )
    result = EnvironmentSimulator().run(env, policy)
    assert result.success is True
    assert task_success(result) is True
    assert result.verification.score == 1.0
    assert result.trajectory.success is True
    # The end-state checks, not turn-by-turn plausibility, decide success.
    assert env.state["orders"]["O1002"]["status"] == "cancelled"
    assert env.state["orders"]["O1002"]["refunded"] is True
    assert env.state["orders"]["O1001"]["status"] == "delivered"  # untouched


def test_retail_wrong_order_of_operations_fails():
    # Refunding before cancelling is a policy violation: the refund tool rejects
    # a non-cancelled/processing order, so the end state never satisfies the oracle.
    env = make_retail_environment("cancel_refund")
    policy = scripted_policy(
        [
            EnvAction(tool="refund_order", arguments={"order_id": "O1002"}),
            EnvAction(kind="finish"),
        ]
    )
    result = EnvironmentSimulator().run(env, policy)
    assert result.success is False
    assert env.state["orders"]["O1002"]["refunded"] is False
    assert any(not c.passed for c in result.verification.checks)


def test_partial_completion_scores_between_zero_and_one():
    env = make_retail_environment("cancel_refund")
    policy = scripted_policy(
        [EnvAction(tool="cancel_order", arguments={"order_id": "O1002"}), EnvAction(kind="finish")]
    )
    result = EnvironmentSimulator().run(env, policy)
    assert result.success is False  # refund missing
    assert 0.0 < result.verification.score < 1.0


def test_unknown_tool_is_reported_not_raised():
    env = make_retail_environment("cancel_refund")
    step = env.step(EnvAction(tool="nope", arguments={}))
    assert step.ok is False
    assert "unknown tool" in (step.error or "")


def test_determinism_same_policy_same_verification():
    def build():
        env = make_retail_environment("cancel_refund")
        policy = scripted_policy(
            [
                EnvAction(tool="cancel_order", arguments={"order_id": "O1002"}),
                EnvAction(tool="refund_order", arguments={"order_id": "O1002"}),
                EnvAction(kind="finish"),
            ]
        )
        return EnvironmentSimulator().run(env, policy)

    a, b = build(), build()
    assert a.verification.model_dump() == b.verification.model_dump()
    assert [s.tool_name for s in a.trajectory.steps] == [s.tool_name for s in b.trajectory.steps]


def test_reset_is_total_between_runs():
    env = make_retail_environment("cancel_refund")
    env.step(EnvAction(tool="cancel_order", arguments={"order_id": "O1002"}))
    assert env.state["orders"]["O1002"]["status"] == "cancelled"
    obs = env.reset()
    assert env.state["orders"]["O1002"]["status"] == "processing"
    assert obs.step == 0


def test_counter_env_reaches_target():
    env = make_counter_environment(target=3)
    policy = scripted_policy([EnvAction(tool="increment") for _ in range(3)])
    result = EnvironmentSimulator().run(env, policy)
    assert result.success is True
    assert env.state["count"] == 3


def test_trajectory_scored_by_existing_metrics():
    # The harness projects onto a Trajectory, so goal/tool metrics score it.
    env = make_retail_environment("cancel_refund")
    policy = scripted_policy(
        [
            EnvAction(tool="cancel_order", arguments={"order_id": "O1002"}),
            EnvAction(tool="refund_order", arguments={"order_id": "O1002"}),
            EnvAction(kind="finish"),
        ]
    )
    result = EnvironmentSimulator().run(env, policy)
    run = RunOutput(output=result.trajectory.final_answer, trajectory=result.trajectory)
    case = EvalCase(id="t", input=env.task.instruction)
    goal = METRICS["goal_accuracy"](case, run)
    assert goal.value == 1.0


@pytest.mark.asyncio
async def test_async_policy_supported():
    env = make_counter_environment(target=2)

    async def policy(obs):
        if obs.state.get("count", 0) < 2:
            return EnvAction(tool="increment")
        return EnvAction(kind="finish")

    result = await EnvironmentSimulator().arun(env, policy, max_steps=5)
    assert result.success is True


def test_verify_returns_verification_model():
    env = make_retail_environment("update_shipping")
    policy = scripted_policy(
        [EnvAction(tool="update_address", arguments={"order_id": "O1002", "address": "9 New Rd"})]
    )
    result = EnvironmentSimulator().run(env, policy)
    assert isinstance(result.verification, TaskVerification)
    assert result.success is True
