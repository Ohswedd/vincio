# Build a chat product: the Assistant

`ContextApp.run` executes one stateless context-engineering pipeline. A chat
product needs more around it: turns threaded into a session, state carried
forward, write tools gated behind an approval, and a memory of what the user
said. `app.assistant(...)` is that loop, written once, a thin layer in
[`vincio.assistant`](../reference/api.md).

Every turn is still a full `ContextApp` run, so retrieval, grounding, validation,
rails, budgets, tracing, and the audit chain apply unchanged. The Assistant adds
four things and nothing else:

1. **Session threading**: every turn runs under one stable `session_id` (plus
   optional `user_id` / `tenant_id`), so traces, cost, and memory recall are
   scoped to the conversation.
2. **Multi-turn state via memory write-back**: each turn is written back to
   session-scoped memory, so the next turn's pipeline recalls it as scored,
   budgeted context. Conversational state flows through the context compiler, not
   a side channel.
3. **Tool approvals**: an approval surface for write tools. Approval-required
   tools are *denied by default* (a chat reply can never silently run a write
   tool) and surfaced for the caller to approve.
4. **A recorded transcript**: the running thread, available to the caller and as
   a [`Simulator`](agentic-eval.md) target for multi-turn evaluation.

## Quickstart

```python
from vincio import ContextApp

app = ContextApp(name="support_chat")
app.add_source("kb", path="./help-center")

chat = app.assistant(user_id="cust-42")
print(chat.send("How do I reset my password?").text)
print(chat.send("And change my email too?").text)   # remembers the thread
```

`send` returns an `AssistantTurn`: the reply `text`, the validated `output`,
`citations`, the tool `approvals` raised this turn, the `memory_writes` made,
`trace_id`, and `cost_usd`. Use `chat.history()` for the recorded transcript and
`chat.reset()` to start a fresh conversation (a new session).

## Tool approvals

A write tool surfaces as a *pending approval* rather than running, a reply never
silently triggers a side effect:

```python
app.add_tool(refund_create, permissions=["billing:write"],
             approval_required=True, side_effects="write")
chat = app.assistant(user_id="cust-42")

turn = chat.send("Please refund invoice INV-123.")
turn.needs_approval            # True
chat.pending_approvals         # [ApprovalRecord(tool="refund_create", status="pending", ...)]

chat.approve("refund_create")  # the human approves
chat.send("Yes, go ahead.")    # now the tool runs, through the permissioned, audited runtime
```

You can pre-allow trusted tools with `auto_approve=["..."]`, or pass an
`on_approval(request) -> bool` callback (sync or async) for an interactive
decision. The decision is enforced in code through the same
[permissioned tool runtime](add-tools.md) as everything else, so the approval
surface inherits RBAC, the sandbox, and the audit chain.

## Memory write-back

With `memory_writeback=True` (the default), each turn is written to
session-scoped memory as a `summary` of the exchange, so the next turn's run
recalls it automatically. The memory engine is created on first use if you have
not already called `add_memory`. Pass `memory_writeback=False` to keep the
transcript in process only.

Because state is memory, the usual memory guarantees apply: scoping, decay,
privacy class, and the [governed hygiene](../concepts/memory.md) operations. A
durable user fact you `app.remember(..., user_id=...)` outside the session is
recalled across conversations.

## Multi-turn evaluation

The Assistant satisfies the `Simulator`'s agent contract, give it the running
thread, get back the next reply, so a persona-driven simulator can drive a whole
conversation offline and convert it to a scorable `EvalCase`:

```python
from vincio.evals.simulator import Persona, Simulator

chat = app.assistant(user_id="sim-1")
convo = Simulator(seed=7).simulate(
    lambda messages: chat.send(messages[-1]["content"]).text,
    Persona(name="customer", goal="resolve a duplicate charge", facts={"plan": "Pro"}),
)
case = convo.to_eval_case(id="sim_billing")   # score with the conversational metrics
```

See [`examples/01_quickstart.py`](../../examples/01_quickstart.py) for a runnable
end-to-end version, and [agentic evaluation](agentic-eval.md) for the multi-turn
metrics.

## When to use the Assistant (and when not)

- **Use it** when you are building a *chat product*: multi-turn threads, a memory
  of the conversation, write tools behind a human approval, a transcript to
  replay or evaluate.
- **Skip it** when a call is a one-shot transformation — `app.run(...)` is the
  stateless pipeline, and the Assistant is only the loop around it. Wrapping a
  single stateless task in a session buys you nothing but a `session_id`.
- **It is additive, not a fork.** Every turn is still a full `ContextApp` run, so
  retrieval, rails, budgets, validation, tracing, and the audit chain are the
  ones you already configured — there is no second, weaker path.

## Gotchas

- **Approval-required tools are denied by default, per turn.** A write tool
  surfaces as a pending approval; you `chat.approve(name)` and then send again to
  actually run it. A single message can never both request and execute a write.
- **`memory_writeback=True` (the default) turns each turn into recalled context.**
  That is what makes the thread coherent, but it means the summary of a turn
  competes for budget on the next run. Pass `memory_writeback=False` to keep the
  transcript in-process only.
- **Session memory is scoped and decays; durable facts are not.** A fact you want
  to survive across conversations must be written with
  `app.remember(..., user_id=...)` *outside* the session — the session's
  write-back is conversation-scoped and subject to the usual decay/hygiene.
- **`reset()` starts a new session** (a fresh `session_id`), so traces, cost, and
  recall no longer join the prior thread.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: Memory](../concepts/memory.md)
- [Guide: close the loop](close-the-loop.md)
- [Example: 03_memory.py](../../examples/03_memory.py)
- [Concept: Context packets & long-horizon governance](../concepts/context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
