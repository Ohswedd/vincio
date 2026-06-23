# Computer-use action plane

A growing class of real work requires *acting on interfaces built for humans* — a
browser, a desktop GUI, an OS shell, a remote machine. Vincio has two layers for
this, and they compose:

- **The flat tools.** `app.enable_computer_use(...)` registers a navigate / click /
  type / screenshot vocabulary as ordinary permissioned, approval-gated tools — a
  thin GUI adapter that rides the same RBAC + audit + budget path as any local tool.
  See [add tools](add-tools.md).
- **The action plane.** `app.computer_use(...)` returns a `ComputerEnvironment` for an
  agent that drives a screen *safely*: it perceives UI state, grounds an intent to a
  concrete target, pre-gates the action, acts, verifies the effect, and rolls back on
  drift — under the same governance, budget, rails, and audit the rest of the platform
  enforces.

This guide covers the action plane. Everything here runs **fully offline and
deterministic** against an in-process screen; a real browser / OS / remote-desktop
driver sits behind `pip install "vincio[computer-use]"`.

> This is a library capability inside your process — never a hosted agent, an RPA
> service, or a managed control plane.

## The loop

Every action closes a six-step loop, the computer-use analogue of a saga's
compensation and the record-replay divergence report:

**perceive → ground → pre-gate → act → post-verify → undo-on-divergence**

1. **Perceive.** A `ScreenBackend` turns a screenshot plus an accessibility tree into
   a typed `ScreenState` — a list of addressable `UIElement`s, a public state
   projection, and a content `digest` of the salient state.
2. **Ground.** An intent is bound to a target by a **stable selector** (role +
   accessible name), not a pixel coordinate — so the action is replayable, auditable,
   and survives a layout shift.
3. **Pre-gate.** The action is checked against an `ActionPolicy`. A destructive or
   out-of-scope action is gated like a write tool, behind an approval callback.
4. **Act.** The backend performs the raw effect.
5. **Post-verify.** The new screen is compared against the action's declared
   expectation (a declarative `StateCheck`, or a "must change" assertion).
6. **Undo.** On divergence, the action is undone — a synthesized inverse where one
   exists (navigate back, re-type a field's prior value, re-click a toggle), falling
   back to a prior-state restore — so a drifting action never leaves the screen in an
   unexpected state.

Every step lands on the app's hash-chained audit log.

## A first run

`make_web_checkout()` is a deterministic, WebArena/OSWorld-shaped reference app: a
two-screen store with a shipping-address field, a checkout button, a place-order
button (the goal), and a destructive **Delete account** button (the trap).

```python
from vincio import ContextApp, ActionPolicy, UIAction, make_web_checkout
from vincio.providers import MockProvider

app = ContextApp(name="operator", provider=MockProvider())
spec, task = make_web_checkout()

# Approve only the in-task destructive action (placing the order); nothing else.
def approve(action: UIAction, decision) -> bool:
    return "Place order" in action.selector

env = app.computer_use(
    screen=spec,
    policy=ActionPolicy(allow_urls=["https://shop.test"]),
    approve=approve,
)

# A policy maps the perceived screen to the next grounded action.
def policy(state):
    s = state.state
    if s["screen"] == "cart" and not s["fields"].get("role=textbox[name='Address']"):
        return UIAction(kind="type", selector="role=textbox[name='Address']", text="1 Main St")
    if s["screen"] == "cart":
        return UIAction(kind="click", selector="role=button[name='Checkout']", expect_change=True)
    if s["screen"] == "review" and not s["flags"].get("order_placed"):
        return UIAction(kind="click", selector="role=button[name='Place order']")
    return None

run = env.run(policy, task)
assert run.success      # the end-state oracle: the order was placed
assert run.safe         # no destructive action ran without approval
assert app.audit.verify_chain()
```

`run` is a `ComputerRun`: `.success` (the task verifier's verdict over the final
screen), `.safe` (no unapproved destructive action ever executed), `.steps_taken`,
and `.trajectory` — a `vincio.evals.trajectory.Trajectory` the existing trajectory
metrics, test-time search, and world-model planner score with **no new machinery**.

## Grounding by a stable selector

A `ScreenState` exposes the perceived UI; ground an intent against it rather than
hard-coding coordinates:

```python
state = await env.observe()
button = state.find(role="button", name="Checkout")   # → UIElement, or None
button.selector                                        # "role=button[name='Checkout']"
state.element(button.selector)                         # look up by exact selector
```

A `UIAction` binds `kind` (`navigate` / `click` / `type` / `scroll` / `drag` / `key`
/ `wait`) to that selector. Because the target is a role+name selector, the same
action replays against a page whose layout shifted — only a brittle pixel would break.

## The pre-gate: scope and approval

An `ActionPolicy` is the rail:

```python
policy = ActionPolicy(
    allow_urls=["https://shop.test"],   # in-scope URL prefixes (empty = anywhere)
    deny_urls=["https://shop.test/admin"],
    allow_destructive=True,             # set False to block destructive actions outright
    approve_destructive=True,           # destructive actions need an approval callback
)
```

An action is **destructive** if it carries `destructive=True`, targets an element
flagged destructive, or its target name matches a configurable keyword set (delete,
remove, purchase, pay, place order, transfer, …). An action is **out-of-scope** if it
acts outside `allow_urls` (or inside `deny_urls`). Either is refused unless approved —
the gate makes an unapproved destructive action *structurally impossible*, not merely
discouraged:

```python
out = await env.act(UIAction(kind="click", selector="role=button[name='Delete account']"))
out.gated        # True — refused
out.performed    # False — the effect never ran
out.reason       # "destructive action requires approval and none was granted"
```

Approve a specific action at the call site with `env.act(action, approve=True)`, or
wire an `approve(action, decision) -> bool` callback on the environment for a
human-in-the-loop policy.

## Post-verify and auto-undo

Declare what an action should achieve and the plane verifies it, rolling back on
divergence:

```python
from vincio.evals.environment import StateCheck

out = await env.act(
    UIAction(
        kind="type", selector="role=textbox[name='Address']", text="1 Main St",
        expect=[StateCheck(name="set", path="fields.role=textbox[name='Address']", op="eq", value="1 Main St")],
    )
)
out.verified     # True — the post-condition held
```

If the post-condition fails (or `expect_change` disagrees with what actually changed),
`out.diverged` is `True` and, with `auto_undo=True` (the default), the effect is
undone — `out.undone` is `True` and `out.after_digest` is restored to
`out.before_digest`.

## Tasks and verifiable success

A `ComputerTask` carries the goal and a declarative end-state verifier, so success is
**verifiable end-state, not turn-by-turn plausibility** — the same shape the agentic
leaderboards judge on:

```python
from vincio import ComputerTask
from vincio.evals.environment import StateCheck

task = ComputerTask(
    id="place_order",
    instruction="Set the shipping address and place the order.",
    checks=[
        StateCheck(name="address_set", path="fields.role=textbox[name='Address']", op="truthy"),
        StateCheck(name="order_placed", path="flags.order_placed", op="eq", value=True),
    ],
    max_steps=8,
)
```

## Real backends

The deterministic `MockScreen` (driving a `ScreenApp`) is the offline default. Real
drivers sit behind `vincio[computer-use]` and should run under
`require_isolation=True` so the workload is behind a real
[isolation backend](../../SECURITY.md):

```python
env = app.computer_use(backend="playwright", require_isolation=True, isolation="container")
# also: backend="accessibility" (OS accessibility tree), backend="remote_desktop"
```

`PlaywrightScreen` grounds a page's accessibility snapshot into `UIElement`s the same
way `MockScreen` grounds its app, so your policy code is identical regardless of where
the screen lives.

## What it gates

Two VincioBench SLOs hold the action plane (see the [SLO reference](../reference/slo.md)):

- **Success at budget** — an agent reaches a verified end state within its action
  budget on the deterministic reference app.
- **No unapproved destructive action** — a reckless policy attempting a destructive
  action without approval performs zero such actions; the gate blocks it, an
  out-of-scope navigation is refused, and a divergent action is rolled back.

See [`examples/89_computer_use_action_plane.py`](../../examples/89_computer_use_action_plane.py)
for the full runnable walkthrough.
