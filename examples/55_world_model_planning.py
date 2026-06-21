"""World-model / simulation-based planning.

The environment harness already lets an agent *evaluate* a trajectory against the
live world. This example shows the next rung: an agent that **learns a model of
its tools and plans against it** — searching imagined rollouts before acting, so a
wrong move costs a simulated step, not a live one.

Four steps, all offline and deterministic (no model required):

  1. WorldModel: fit a dynamics model from recorded reset/step transitions. It
     learns each tool's *parameterized* effect under a learned *precondition*, so
     it predicts a refund fails on a processing order and succeeds on a cancelled
     one — and generalizes a cancel it only ever saw on one order to another.
  2. CalibrationReport: the model earns planning weight only after its predictions
     track the real environment within a tolerance, the way a judge ensemble earns
     gating weight. An uncalibrated model is refused by the planner.
  3. ModelPredictivePlanner: a receding-horizon planner that searches imagined
     rollouts with the test-time-search beam, commits the best first action to the
     real world, observes, and re-plans.
  4. Planning beats reacting: on a world with a locally-attractive shortcut that
     dead-ends, the imagined-rollout planner opens the vault while a reactive
     (one-step) planner is trapped.

Everything here is opt-in and additive; nothing below is required to run Vincio.
"""

from __future__ import annotations

from vincio.agents import (
    ModelPredictivePlanner,
    WorldModel,
    record_transitions,
)
from vincio.evals.environment import (
    EnvAction,
    make_retail_environment,
    make_vault_environment,
)


def act(tool: str, **kwargs: object) -> EnvAction:
    return EnvAction(kind="tool", tool=tool, arguments=kwargs)


def learn_a_world_model() -> WorldModel:
    print("1. WorldModel — learn each tool's effect and precondition from experience")
    # Explore the retail world: cancel/refund/update, including a refund that fails
    # (on a processing order) and one that succeeds (after a cancel) so the model
    # can learn the precondition, not just the effect.
    explore = [
        [act("refund_order", order_id="O1002")],
        [act("cancel_order", order_id="O1002"), act("refund_order", order_id="O1002")],
        [act("cancel_order", order_id="O1002")],
        [act("refund_order", order_id="O1001")],
        [act("update_address", order_id="O1002", address="9 New Rd")],
        [act("get_order", order_id="O1002")],
    ]
    transitions = record_transitions(make_retail_environment("cancel_refund"), explore)
    model = WorldModel(transitions)
    print(f"   fit from {len(transitions)} recorded transitions")

    base = make_retail_environment("cancel_refund").observe()
    fail = model.predict(base, act("refund_order", order_id="O1002"))
    after_cancel = model.predict(base, act("cancel_order", order_id="O1002")).observation
    ok = model.predict(after_cancel, act("refund_order", order_id="O1002"))
    print(f"   refund on a processing order → ok={fail.ok} ({fail.reason})")
    print(f"   refund after a cancel        → ok={ok.ok} ({ok.reason})")
    # cancel was only ever seen on O1002; the model generalizes it to O1001.
    gen = model.predict(base, act("cancel_order", order_id="O1001"))
    print(
        "   generalizes cancel to the unseen order O1001 → "
        f"status={gen.observation.state['orders']['O1001']['status']}"
    )
    return model


def calibrate_for_planning(model: WorldModel) -> None:
    print("\n2. CalibrationReport — the model earns planning weight")
    transitions = record_transitions(
        make_retail_environment("cancel_refund"),
        [
            [act("cancel_order", order_id="O1002"), act("refund_order", order_id="O1002")],
            [act("refund_order", order_id="O1002")],
            [act("update_address", order_id="O1002", address="9 New Rd")],
        ],
    )
    report = model.calibrate(transitions)
    print(f"   next-state accuracy {report.state_accuracy:.2f}, reward MAE {report.reward_mae:.2f}")
    print(f"   trusted={report.trusted} (weight {report.weight:.2f}) — {report.reason}")


def plan_the_retail_task(model: WorldModel) -> None:
    print("\n3. ModelPredictivePlanner — plan the cancel→refund task in imagination")
    planner = ModelPredictivePlanner(model, horizon=3, beam_width=16)
    result = planner.plan(make_retail_environment("cancel_refund"))
    plan = " → ".join(a.tool for a in result.committed)
    print(f"   committed (real steps={result.real_steps}): {plan}")
    print(f"   success={result.success} — {result.reason}")


def planning_beats_reacting() -> None:
    print("\n4. Planning beats reacting — a shortcut that looks fast but dead-ends")
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
    transitions = record_transitions(make_vault_environment(), explore)
    model = WorldModel(transitions).fit(transitions)
    model.calibrate(transitions)

    budget = 6
    reactive = ModelPredictivePlanner(
        model, horizon=1, beam_width=64, max_real_steps=budget
    ).plan(make_vault_environment())
    planned = ModelPredictivePlanner(
        model, horizon=5, beam_width=64, max_real_steps=budget
    ).plan(make_vault_environment())

    print(
        f"   reactive (1-step): success={reactive.success} after {reactive.real_steps} steps "
        f"→ took the shortcut and got trapped"
    )
    print(
        f"   imagined-rollout : success={planned.success} after {planned.real_steps} steps "
        f"→ {' → '.join(a.tool for a in planned.committed)}"
    )


def main() -> None:
    model = learn_a_world_model()
    calibrate_for_planning(model)
    plan_the_retail_task(model)
    planning_beats_reacting()
    print("\nThe agent planned against a model of its tools, not the live world.")


if __name__ == "__main__":
    main()
