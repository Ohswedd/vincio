"""Computer-use action plane — perceive, ground, gate, act, verify, undo.

A growing class of real work requires *acting on interfaces built for humans*: a
browser, a desktop GUI, an OS shell, a remote machine. The flat navigate / click /
type / screenshot vocabulary (`app.enable_computer_use`) exposes that as ordinary
permissioned tools — a thin GUI adapter. This example shows the rung above it: a
first-class **action plane** that drives a screen *safely*.

`app.computer_use(...)` returns a `ComputerEnvironment` that, for every action:

  1. **Perceives** the screen as typed, addressable `UIElement`s (a screenshot plus
     an accessibility tree), not raw pixels.
  2. **Grounds** an intent to a concrete target by a **stable selector** (role +
     accessible name), so an action is replayable and survives a layout shift.
  3. **Pre-gates** the action against an `ActionPolicy` — a destructive or
     out-of-scope action is gated like a write tool, behind an approval callback.
  4. **Acts**, then **post-verifies** the effect against the action's expectation.
  5. **Undoes** the action on divergence — the computer-use analogue of a saga's
     compensation — and records every step on the same hash-chained audit log.

A `ComputerTask` carries a goal and a declarative verifier, so a run projects onto
the same `Trajectory` the existing trajectory metrics and test-time search already
score — success is a verified end state, not turn-by-turn plausibility. Everything
here is deterministic and offline against an in-process, WebArena/OSWorld-shaped app
(`make_web_checkout`); a real browser / OS / remote-desktop driver sits behind
`vincio[computer-use]`.

This is a library capability inside your process — never a hosted agent or RPA
service.
"""

from __future__ import annotations

import asyncio

from vincio import ActionPolicy, ContextApp, UIAction, make_web_checkout
from vincio.providers import MockProvider

_ADDRESS = "role=textbox[name='Address']"
_CHECKOUT = "role=button[name='Checkout']"
_PLACE = "role=button[name='Place order']"
_DELETE = "role=button[name='Delete account']"


def main() -> None:
    app = ContextApp(name="operator", provider=MockProvider(default_text="ok"))
    spec, task = make_web_checkout()

    # The pre-gate: the agent may act only within the store, and a destructive action
    # needs explicit approval. Here we approve *only* the in-task purchase (placing the
    # order) and nothing else — so the out-of-task "Delete account" can never run.
    def approve(action: UIAction, decision) -> bool:
        return "Place order" in action.selector

    env = app.computer_use(
        screen=spec,
        policy=ActionPolicy(allow_urls=["https://shop.test"]),
        approve=approve,
    )

    # A policy maps the perceived screen to the next grounded action. It addresses the
    # cart, checks out, then places the order — grounding each intent to a stable
    # selector rather than a coordinate.
    def policy(state):
        s = state.state
        if s["screen"] == "cart" and not s["fields"].get(_ADDRESS):
            return UIAction(kind="type", selector=_ADDRESS, text="1 Main St")
        if s["screen"] == "cart":
            return UIAction(kind="click", selector=_CHECKOUT, expect_change=True)
        if s["screen"] == "review" and not s["flags"].get("order_placed"):
            return UIAction(kind="click", selector=_PLACE)
        return None

    # 1. Drive the app to a verified end state within the action budget.
    run = env.run(policy, task)
    print(
        f"1. Task '{task.id}': success={run.success} (verified end-state), "
        f"safe={run.safe}, steps={run.steps_taken}/{task.max_steps}; "
        f"trajectory.source={run.trajectory.source!r}"
    )

    # 2. The run is safe by construction: no destructive action ran without approval.
    placed = next(o for o in run.outcomes if o.action.selector == _PLACE)
    print(
        f"2. Placed the order under approval: performed={placed.performed}, "
        f"approved={placed.decision.approved}; unapproved-destructive actions="
        f"{run.unapproved_destructive}"
    )

    async def safety_and_undo() -> None:
        # 3. Safety gate: an unapproved destructive action is *blocked*, not logged.
        guard = app.computer_use(
            screen=make_web_checkout()[0],
            policy=ActionPolicy(allow_urls=["https://shop.test"]),
        )
        blocked = await guard.act(UIAction(kind="click", selector=_DELETE))
        print(
            f"3. Unapproved 'Delete account' gated: gated={blocked.gated}, "
            f"performed={blocked.performed} — {blocked.reason}"
        )

        # 4. Out-of-scope navigation is refused the same way a write tool would be.
        offscope = await guard.act(UIAction(kind="navigate", url="https://evil.test/x"))
        print(f"4. Out-of-scope navigation gated: gated={offscope.gated}")

        # 5. Post-verify + auto-undo: an action whose effect diverges from its
        #    expectation is rolled back to the prior state.
        from vincio.evals.environment import StateCheck

        undo_env = app.computer_use(screen=make_web_checkout()[0])
        before = (await undo_env.observe()).digest
        diverged = await undo_env.act(
            UIAction(
                kind="type", selector=_ADDRESS, text="oops",
                expect=[StateCheck(name="impossible", path="flags.never", op="truthy")],
            )
        )
        print(
            f"5. Divergent action undone: diverged={diverged.diverged}, "
            f"undone={diverged.undone}, state restored={diverged.after_digest == before}"
        )

    asyncio.run(safety_and_undo())

    # 6. Every action rode the same hash-chained audit log the rest of the platform uses.
    print(
        f"6. Audit: {len(app.audit.query(action='computer_use_session'))} session(s), "
        f"{len(app.audit.query(action='computer_action'))} action(s) on the chain; "
        f"chain intact={app.audit.verify_chain()}."
    )

    assert run.success and run.safe and app.audit.verify_chain()


if __name__ == "__main__":
    main()
