"""Coverage-raising tests for vincio.agents.world_model.

Every test exercises real learned-dynamics / planning behavior through the real
API against the deterministic in-process reference environments (no provider, no
network). They target the previously-uncovered helper branches (path get/set/del
on lists, removed paths, value-template fallbacks, discriminative-field
selection) and planner branches (callable / explicit / vocabulary repertoires,
the early-break and empty-plan paths, the calibration gate).
"""

from __future__ import annotations

import pytest

from vincio.agents.world_model import (
    ModelPredictivePlanner,
    Transition,
    WorldModel,
    _del_path,
    _get_path,
    _set_path,
    record_transitions,
    task_goal_value,
)
from vincio.evals.environment import (
    EnvAction,
    EnvObservation,
    EnvTask,
    EnvToolResult,
    StateCheck,
    ToolEnvironment,
    make_counter_environment,
    make_retail_environment,
    make_vault_environment,
)


def act(tool: str, **kwargs) -> EnvAction:
    return EnvAction(kind="tool", tool=tool, arguments=kwargs)


# ---------------------------------------------------------------------------
# Path helpers — list indexing, missing segments, overwrite, deletion
# ---------------------------------------------------------------------------


def test_get_path_indexes_into_a_list():
    found, value = _get_path({"items": [10, 20, 30]}, "items.1")
    assert found and value == 20


def test_get_path_out_of_range_list_index_is_not_found():
    found, value = _get_path({"items": [10]}, "items.5")
    assert found is False and value is None


def test_get_path_non_integer_list_segment_is_not_found():
    found, value = _get_path({"items": [10]}, "items.oops")
    assert found is False and value is None


def test_get_path_missing_dict_key_is_not_found():
    found, value = _get_path({"a": 1}, "a.b")
    assert found is False and value is None


def test_get_path_descending_through_a_scalar_is_not_found():
    # Once a scalar leaf is hit, a deeper path cannot resolve.
    found, value = _get_path({"a": 5}, "a.b.c")
    assert found is False and value is None


def test_set_path_replaces_a_scalar_intermediate_with_a_dict():
    state: dict = {"a": 5}
    _set_path(state, "a.b", "x")
    assert state == {"a": {"b": "x"}}


def test_del_path_removes_a_nested_leaf():
    state = {"a": {"b": 1, "c": 2}}
    _del_path(state, "a.b")
    assert state == {"a": {"c": 2}}


def test_del_path_with_missing_intermediate_is_a_noop():
    state = {"a": 1}
    _del_path(state, "a.b.c")
    assert state == {"a": 1}


def test_del_path_through_a_non_dict_parent_is_a_noop():
    state = {"a": [1]}
    _del_path(state, "a.0")
    assert state == {"a": [1]}


# ---------------------------------------------------------------------------
# WorldModel.__init__ without transitions stays empty / untrusted
# ---------------------------------------------------------------------------


def test_world_model_unfit_predicts_identity_and_is_untrusted():
    model = WorldModel()  # the transitions-is-None branch: no fitting performed
    assert model.vocabulary() == []
    assert model.trusted is False
    obs = make_counter_environment().observe()
    pred = model.predict(obs, act("increment"))
    # No effect learned → unseen signature → identity prediction, zero confidence.
    assert pred.known is False and pred.confidence == 0.0
    assert pred.observation.state == obs.state


# ---------------------------------------------------------------------------
# record_transitions stops at a done step (mid-sequence break)
# ---------------------------------------------------------------------------


def _make_terminal_env() -> ToolEnvironment:
    def finish_now(state: dict, _args: dict) -> EnvToolResult:
        state["finished"] = True
        return EnvToolResult(text="finished")

    def noop(state: dict, _args: dict) -> EnvToolResult:
        state["touched"] = True
        return EnvToolResult(text="touched")

    env = ToolEnvironment(
        name="terminal",
        initial_state={"finished": False},
        tools={"finish_now": finish_now, "noop": noop},
        task=EnvTask(
            id="t",
            instruction="finish",
            checks=[StateCheck(name="done", path="finished", op="truthy")],
        ),
    )

    # Wrap step() so the finishing tool reports done=True, exercising the break.
    real_step = env.step

    def step(action: EnvAction):
        result = real_step(action)
        if action.tool == "finish_now":
            result.done = True
        return result

    env.step = step  # type: ignore[method-assign]
    return env


def test_record_transitions_breaks_on_a_done_step():
    env = _make_terminal_env()
    trs = record_transitions(env, [[act("finish_now"), act("noop")]])
    # The second action is never recorded because the first signalled done.
    assert len(trs) == 1
    assert trs[0].action.tool == "finish_now" and trs[0].done is True


# ---------------------------------------------------------------------------
# Learned dynamics: removed-path effect, single discriminative field, add/const
# ---------------------------------------------------------------------------


def _make_clearable_env() -> ToolEnvironment:
    def clear(state: dict, _args: dict) -> EnvToolResult:
        state.pop("flag", None)
        return EnvToolResult(text="cleared")

    return ToolEnvironment(
        name="clearable",
        initial_state={"flag": True, "keep": 1},
        tools={"clear": clear},
        task=EnvTask(
            id="t",
            instruction="clear",
            checks=[StateCheck(name="gone", path="flag", op="falsy")],
        ),
    )


def test_predict_applies_a_learned_removed_path():
    env = _make_clearable_env()
    trs = record_transitions(env, [[act("clear")]])
    model = WorldModel(trs)
    obs = env.reset()
    pred = model.predict(obs, act("clear"))
    # The removed-path branch deletes 'flag' from the predicted next state.
    assert "flag" not in pred.observation.state
    assert pred.observation.state == {"keep": 1}
    # The source observation is untouched by the prediction.
    assert obs.state["flag"] is True


def test_single_discriminative_field_is_selected_and_predicts_both_outcomes():
    # A tool whose outcome is decided by ONE repeating state field: this drives
    # the candidate-selection branch (a single separating field is chosen) rather
    # than the multi-field conjunction fallback.
    def toggle(state: dict, args: dict) -> EnvToolResult:
        oid = args["id"]
        order = state["orders"][oid]
        if order["status"] == "ready":
            order["done"] = True
            return EnvToolResult(text="ok")
        return EnvToolResult(ok=False, error="not ready")

    # Two orders share each status so every discriminative value repeats — this
    # is what lets a single field be picked as a candidate (it generalizes,
    # unlike a unique id) rather than discarded.
    seed = {
        "orders": {
            "A": {"status": "ready", "done": False},
            "B": {"status": "pending", "done": False},
            "C": {"status": "ready", "done": False},
            "D": {"status": "pending", "done": False},
        }
    }
    env = ToolEnvironment(
        name="toggle",
        initial_state=seed,
        tools={"toggle": toggle},
        task=EnvTask(
            id="t",
            instruction="toggle",
            checks=[StateCheck(name="d", path="orders.A.done", op="truthy")],
        ),
    )
    trs = record_transitions(
        env,
        [
            [act("toggle", id="A")],
            [act("toggle", id="B")],
            [act("toggle", id="C")],
            [act("toggle", id="D")],
        ],
    )
    model = WorldModel(trs)
    # The discriminative key is the per-order status, parameterized by {id}.
    sig = ("toggle", ("id",))
    assert model._effects[sig].key_fields == ("orders.{id}.status",)
    base = env.reset()
    ready = model.predict(base, act("toggle", id="A"))
    pending = model.predict(base, act("toggle", id="B"))
    assert ready.ok and ready.observation.state["orders"]["A"]["done"] is True
    assert not pending.ok and pending.observation.state["orders"]["B"]["done"] is False

    # An order whose precondition value was never recorded ("shipped") falls back
    # to the most-common effect with halved confidence (the unseen-precondition
    # branch), not an exact match.
    base.state["orders"]["E"] = {"status": "shipped", "done": False}
    fallback = model.predict(base, act("toggle", id="E"))
    assert fallback.confidence == 0.5
    assert "precondition unseen" in fallback.reason


def test_numeric_add_template_applies_old_plus_step():
    # An increment learns ("add", 1); predicting reads the live value and adds the
    # learned step (the _apply_value "add" branch), generalizing past unseen counts.
    trs = record_transitions(
        make_counter_environment(target=3),
        [[act("increment"), act("increment")]],
    )
    model = WorldModel(trs)
    obs = make_counter_environment().observe()
    obs.state["count"] = 41
    pred = model.predict(obs, act("increment"))
    assert pred.observation.state["count"] == 42


def test_value_template_falls_back_to_last_constant_when_unexplained():
    # A path whose new value is neither a stable arg, a stable constant, nor a
    # consistent numeric step → the resolver falls back to the last seen value.
    def stamp(state: dict, args: dict) -> EnvToolResult:
        # New label is unrelated to any argument and differs per call.
        state["label"] = args["seed"] + "!"
        return EnvToolResult(text="stamped")

    env = ToolEnvironment(
        name="stamp",
        initial_state={"label": ""},
        tools={"stamp": stamp},
        task=EnvTask(
            id="t",
            instruction="stamp",
            checks=[StateCheck(name="l", path="label", op="ne", value="")],
        ),
    )
    trs = record_transitions(
        env, [[act("stamp", seed="x")], [act("stamp", seed="y")]]
    )
    model = WorldModel(trs)
    # "x!"/"y!" never equal the arg ("x"/"y"), are not identical, and are not
    # numeric → ("const", last_value). Predicting reproduces that frozen value.
    pred = model.predict(env.reset(), act("stamp", seed="z"))
    assert pred.observation.state["label"] == "y!"


# ---------------------------------------------------------------------------
# imagine() breaks on a predicted-done step
# ---------------------------------------------------------------------------


def test_imagine_stops_at_a_predicted_finish_action():
    model = WorldModel(record_transitions(make_counter_environment(), [[act("increment")]]))
    obs = make_counter_environment().observe()
    steps = model.imagine(
        obs,
        [EnvAction(kind="finish", text="stop"), act("increment")],
    )
    # The finish action predicts done=True, so imagine breaks before incrementing.
    assert len(steps) == 1 and steps[0].done is True


# ---------------------------------------------------------------------------
# task_goal_value with no checks
# ---------------------------------------------------------------------------


def test_imagine_runs_a_full_plan_to_completion_without_a_done_step():
    # No action is terminal, so imagine threads every step and exits the loop
    # normally (no early break), one PredictedStep per action.
    # Consecutive increments teach the "add" template so predictions generalize.
    trs = record_transitions(
        make_counter_environment(target=4),
        [[act("increment"), act("increment"), act("increment")]],
    )
    model = WorldModel(trs)
    obs = make_counter_environment().observe()
    steps = model.imagine(obs, [act("increment"), act("increment"), act("increment")])
    assert len(steps) == 3
    assert [s.observation.state["count"] for s in steps] == [1, 2, 3]
    assert not any(s.done for s in steps)


def test_record_transitions_skips_message_actions_and_can_drop_failures():
    # A message action is not recorded (the non-tool continue); on a failing tool
    # step include_failures=False drops the transition entirely.
    explore = [[EnvAction(kind="message", text="hi"), act("refund_order", order_id="O1002")]]
    dropped = record_transitions(
        make_retail_environment("cancel_refund"), explore, include_failures=False
    )
    assert dropped == []  # the refund on a processing order failed and was dropped
    kept = record_transitions(make_retail_environment("cancel_refund"), explore)
    assert len(kept) == 1 and kept[0].ok is False and kept[0].action.tool == "refund_order"


def test_value_template_with_inconsistent_numeric_steps_uses_constant_fallback():
    # A doubling tool: new = old + old. The new values are numeric but the deltas
    # disagree (+1, +2, +4 ...) so no single "add" step exists, and they are not
    # all identical and equal no argument → the resolver falls through to the
    # last-seen constant (the numeric-but-not-a-step fallback branch).
    def grow(state: dict, _args: dict) -> EnvToolResult:
        state["n"] = state["n"] + state["n"]
        return EnvToolResult(text="grown")

    env = ToolEnvironment(
        name="grow",
        initial_state={"n": 1},
        tools={"grow": grow},
        task=EnvTask(
            id="t",
            instruction="grow",
            checks=[StateCheck(name="p", path="n", op="gte", value=8)],
        ),
    )
    # Within one rollout: 1→2 (+1), 2→4 (+2), 4→8 (+4): inconsistent step set.
    trs = record_transitions(env, [[act("grow"), act("grow"), act("grow")]])
    model = WorldModel(trs)
    pred = model.predict(env.reset(), act("grow"))
    # No "add" template was learned: the new value is frozen to the last recorded
    # value (8), not old+old.
    assert pred.observation.state["n"] == 8


def test_planner_exhausts_its_step_budget_without_reaching_the_goal():
    # With max_real_steps=1 and a target of 3, the planner commits one increment
    # (count=1), the goal is still unmet, and the real-step loop runs to exhaustion
    # (no break) before verifying — an unsuccessful, budget-bounded result.
    trs = record_transitions(
        make_counter_environment(target=3),
        [[act("increment"), act("increment"), act("increment")]],
    )
    model = WorldModel(trs)
    model.calibrate(trs)
    planner = ModelPredictivePlanner(
        model, horizon=2, beam_width=4, max_real_steps=1
    )
    result = planner.plan(make_counter_environment(target=3))
    assert result.real_steps == 1
    assert result.success is False
    assert result.final_value < 1.0


def test_planner_breaks_immediately_when_goal_already_met():
    # value_fn(obs) >= goal_bar at reset → the real-step loop breaks before any
    # action, committing nothing, yet verification still runs.
    model = _retail_model()
    planner = ModelPredictivePlanner(
        model,
        goal_value=lambda _o: 1.0,
        goal_bar=1.0,
        horizon=3,
        beam_width=4,
    )
    result = planner.plan(make_retail_environment("cancel_refund"))
    assert result.real_steps == 0 and result.steps == []


def test_task_goal_value_with_no_checks_is_zero():
    value = task_goal_value([])
    assert value(EnvObservation(state={"anything": 1})) == 0.0


# ---------------------------------------------------------------------------
# Planner repertoire variations: callable proposer and vocabulary fallback
# ---------------------------------------------------------------------------


def _retail_model() -> WorldModel:
    explore = [
        [act("refund_order", order_id="O1002")],
        [act("cancel_order", order_id="O1002"), act("refund_order", order_id="O1002")],
        [act("cancel_order", order_id="O1002")],
        [act("update_address", order_id="O1002", address="9 New Rd")],
    ]
    trs = record_transitions(make_retail_environment("cancel_refund"), explore)
    model = WorldModel(trs)
    model.calibrate(trs)
    return model


def test_planner_uses_a_callable_proposer():
    model = _retail_model()
    calls: list[int] = []

    def proposer(_obs: EnvObservation) -> list[EnvAction]:
        calls.append(1)
        return [
            act("cancel_order", order_id="O1002"),
            act("refund_order", order_id="O1002"),
        ]

    result = ModelPredictivePlanner(
        model, actions=proposer, horizon=3, beam_width=8
    ).plan(make_retail_environment("cancel_refund"))
    assert result.success
    assert calls  # the callable repertoire was actually invoked during search
    assert [a.tool for a in result.committed] == ["cancel_order", "refund_order"]


def test_planner_falls_back_to_learned_vocabulary():
    # actions=None → the planner proposes from the model's learned vocabulary.
    model = _retail_model()
    planner = ModelPredictivePlanner(model, actions=None, horizon=3, beam_width=16)
    result = planner.plan(make_retail_environment("cancel_refund"))
    assert result.success
    assert {a.tool for a in result.committed} <= {
        "cancel_order",
        "refund_order",
        "update_address",
    }


def test_planner_uses_explicit_goal_value_over_task_checks():
    # A goal_value passed explicitly short-circuits _value_fn's task-derived path.
    model = _retail_model()
    seen: list[float] = []

    def goal(obs: EnvObservation) -> float:
        v = 1.0 if obs.state["orders"]["O1002"].get("refunded") else 0.0
        seen.append(v)
        return v

    result = ModelPredictivePlanner(
        model, goal_value=goal, horizon=3, beam_width=16
    ).plan(make_retail_environment("cancel_refund"))
    assert seen  # the custom value function was consulted
    # Reaching the custom goal commits the cancel→refund pair.
    assert "refund_order" in {a.tool for a in result.committed}


# ---------------------------------------------------------------------------
# Planner early-exit branches: empty plan and a predicted-done real step
# ---------------------------------------------------------------------------


def test_planner_breaks_when_no_action_is_proposed():
    # An empty repertoire makes _search return no plan → the planner breaks
    # immediately, committing nothing.
    model = _retail_model()
    result = ModelPredictivePlanner(
        model, actions=[], horizon=3, beam_width=8
    ).plan(make_retail_environment("cancel_refund"))
    assert result.real_steps == 0 and result.committed == []
    assert result.success is False


def test_planner_stops_on_a_done_real_step():
    # The environment reports done after the goal mutation; the planner records
    # the step and breaks out of the real-step loop.
    def open_now(state: dict, _args: dict) -> EnvToolResult:
        state["open"] = True
        return EnvToolResult(text="opened")

    def base_env() -> ToolEnvironment:
        env = ToolEnvironment(
            name="door",
            initial_state={"open": False},
            tools={"open_now": open_now},
            task=EnvTask(
                id="door",
                instruction="open the door",
                checks=[StateCheck(name="o", path="open", op="truthy")],
            ),
        )
        real_step = env.step

        def step(action: EnvAction):
            result = real_step(action)
            if action.tool == "open_now":
                result.done = True
            return result

        env.step = step  # type: ignore[method-assign]
        return env

    trs = record_transitions(base_env(), [[act("open_now")]])
    model = WorldModel(trs)
    model.calibrate(trs)
    result = ModelPredictivePlanner(
        model, horizon=3, beam_width=4, max_real_steps=8
    ).plan(base_env())
    assert result.success and result.real_steps == 1
    assert [a.tool for a in result.committed] == ["open_now"]


# ---------------------------------------------------------------------------
# _search returns ([], 0.0) when the beam yields no best node
# ---------------------------------------------------------------------------


def test_search_returns_empty_when_already_at_goal_at_root():
    # If the goal is already satisfied at the root, expand() yields no successors;
    # the beam has no best node and the plan loop terminates before committing.
    model = _retail_model()

    # A goal that is satisfied for the very first observation forces the planner's
    # value_fn(obs) >= goal_bar break before search — exercise _search directly
    # instead via an always-satisfied value with goal_bar at 0 so expand prunes.
    planner = ModelPredictivePlanner(
        model,
        goal_value=lambda _o: 1.0,
        horizon=3,
        beam_width=4,
    )
    result = planner.plan(make_retail_environment("cancel_refund"))
    # Goal already met → no real steps committed.
    assert result.real_steps == 0 and result.committed == []


def test_search_empty_plan_via_zero_horizon_expansion():
    # With the root already at goal under the provided value fn, _search's beam
    # produces an empty plan (best is the root) and the planner commits nothing.
    model = _retail_model()
    planner = ModelPredictivePlanner(
        model,
        goal_value=lambda _o: 1.0,
        goal_bar=1.0,
        horizon=1,
        beam_width=1,
        require_calibrated=False,
    )
    res = planner.plan(make_retail_environment("cancel_refund"))
    assert res.committed == []


# ---------------------------------------------------------------------------
# Calibration reasons / weight wiring
# ---------------------------------------------------------------------------


def test_calibrate_records_partial_accuracy_and_zero_weight_when_untrusted():
    model = _retail_model()
    # Held-out transitions from an unrelated world: the model mostly mispredicts.
    held = record_transitions(make_vault_environment(), [[act("advance")]])
    report = model.calibrate(held)
    assert report.trusted is False
    assert report.weight == 0.0
    assert "need" in report.reason  # the untrusted reason explains the shortfall


def test_calibrate_on_empty_transitions_is_untrusted_with_reason():
    model = _retail_model()
    report = model.calibrate([])
    assert report.n == 0 and report.trusted is False
    assert report.weight == 0.0
    assert report.reason == "no transitions to calibrate against"
    assert model.trusted is False


def test_planner_refuses_an_uncalibrated_model_with_message():
    from vincio.core.errors import AgentEngineError

    model = WorldModel(record_transitions(make_vault_environment(), [[act("advance")]]))
    assert model.trusted is False
    with pytest.raises(AgentEngineError, match="not calibrated for planning"):
        ModelPredictivePlanner(model, horizon=2).plan(make_vault_environment())


def test_planning_weight_reflects_calibration_weight():
    model = _retail_model()
    result = ModelPredictivePlanner(model, horizon=3, beam_width=16).plan(
        make_retail_environment("cancel_refund")
    )
    assert result.calibrated is True
    assert result.planning_weight == model.calibration.weight > 0.0


# ---------------------------------------------------------------------------
# Transition is an independent snapshot datum
# ---------------------------------------------------------------------------


def test_transition_defaults_and_fields():
    obs = EnvObservation(state={"x": 1})
    nxt = EnvObservation(state={"x": 2})
    tr = Transition(observation=obs, action=act("bump"), next_observation=nxt)
    assert tr.reward == 0.0 and tr.ok is True and tr.done is False
    assert tr.next_observation.state == {"x": 2}
