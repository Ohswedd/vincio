# How-to: orchestrate multi-agent systems

This guide builds a support-triage system three ways, a crew, a durable
graph, and a composed pipeline, all bounded, traced, and resumable.

## How it works

Every shape below runs on the same executor spine: a **bounded DAG** whose
fan-out goes through `gather_bounded`, whose every node opens a trace span, and
whose crew rounds, agent steps, and tool calls are hard-capped, so the system
**terminates by construction**. A crew's manager plans and delegates over a
shared blackboard; a graph checkpoints its state after each node into
`app.store`; a `compose` pipeline threads all of it into one observable stream so
`vincio trace view` shows the whole system. Point the app at SQLite/Postgres
storage and a graph becomes resumable across a process restart — the
durable-timer and `interrupt` machinery holds a pause without holding a worker.

## 1. A crew with roles and delegation

```python
from vincio import ContextApp

app = ContextApp("support", provider="openai", model="gpt-5.2-mini")
app.add_source("kb", path="./docs")

crew = app.crew(
    name="triage",
    members=[
        {"name": "billing", "description": "invoices, refunds, payments",
         "keywords": ["invoice", "refund"], "tools": [lookup_invoice]},
        {"name": "legal", "description": "contracts and clauses",
         "keywords": ["contract", "clause"]},
        {"name": "writer", "goal": "draft the customer reply",
         "budget_fraction": 0.2},
    ],
    process="hierarchical",     # a manager delegates and reviews
    max_rounds=3,
)
result = crew.run("Customer disputes invoice INV-77 under their contract")
print(result.output, result.metrics())
for d in result.delegations:
    print(f"{d.from_agent} -> {d.to_agent}: {d.task} ({d.reason})")
```

Members coordinate through the shared blackboard (`result.blackboard`), each
runs under its budget share, and the manager is capped at `max_rounds`, the
crew terminates by construction. Offline (or if the manager's plan fails to
validate) delegation falls back to deterministic keyword routing, so the same
code runs in CI.

Assign explicit per-member work with `tasks=`:

```python
crew = app.crew(name="report", members=[...], process="parallel")
result = crew.run("Quarterly review", tasks={
    "billing": "Summarize refund volume",
    "legal": "List contract renewals at risk",
})
```

## 2. A durable graph with a human gate

```python
from vincio.agents import interrupt

graph = app.graph("escalation")        # checkpoints persist in app.store
graph.add_node("classify", classify)
graph.add_node("approve", lambda s: {"ok": interrupt(s, {"q": s["summary"]})})
graph.add_node("refund", issue_refund)
graph.add_node("reply", draft_reply)
graph.add_conditional_edge("classify", lambda s: s["severity"],
                           {"high": "approve", "low": "reply"})
graph.add_edge("approve", "refund")
graph.add_edge("refund", "reply")

flow = graph.compile()
paused = flow.invoke({"ticket": ticket_text})
if paused.status == "interrupted":
    # ... show paused.interrupt_payload to a human, possibly days later ...
    flow.update_state(paused.thread_id, {"refund_cap": 50})   # edit
    done = flow.resume(paused.thread_id, value=True)          # and resume
```

With the app configured for SQLite/Postgres storage the thread survives a
process restart: recompile the same graph, point it at the same store, and
`resume(thread_id)`. Audit a run with `flow.history(thread_id)` and replay an
alternative from any step with `flow.fork(checkpoint_id)`.

## 3. Compose the pieces

```python
from vincio import compose
from vincio.agents import branch

pipeline = (
    compose(normalize)
    | branch(lambda t: t["lane"], {"crew": crew, "graph": flow})
    | format_reply
)
reply = pipeline.call(raw_ticket)
```

Every node, including the crew and the graph, streams `node_start` /
`node_end` events from `pipeline.astream(...)` and lands in the same trace,
so one `vincio trace view` shows the whole system.

## 4. Run on another runtime (no lock-in)

```python
from vincio.agents import LangGraphBackend, OpenAIAgentsBackend

lg = LangGraphBackend()                    # pip install langgraph
compiled = lg.compile(graph)               # a real langgraph CompiledGraph

oa = OpenAIAgentsBackend()                 # pip install openai-agents
triage_agent = oa.export_crew(crew)        # manager + handoffs
answer = await oa.run(triage_agent, "Customer disputes INV-77")
```

## 5. Plan deeper, recover, and schedule at scale

Decompose a goal hierarchically, repair the plan when a step fails, spend the
cheapest capable model per step, and schedule independent sub-graphs concurrently.

```python
from vincio import SubgraphScheduler, SubgraphTask, Budget
from vincio.agents import HTNDomain, wait_for_event, TimerService

# Hierarchical (HTN) plan + in-place repair + cost-aware selection in one agent.
domain = (
    HTNDomain()
    .method("root", ["assess", "respond"])
    .method("assess", ["classify", "enrich"], ordering="parallel")
    .operator("classify", step_type="think", instruction="classify severity")
    .operator("enrich", step_type="tool", tool_name="enrich_primary", fallbacks=["enrich_backup"])
    .operator("respond", step_type="finalize", instruction="recommend a response")
)
agent = app.agent(
    tools=["enrich_primary", "enrich_backup"],
    planner="hierarchical", domain=domain,
    cost_aware_models=["gpt-5.2-mini", "gpt-5.2"],   # cheapest capable per step
)
state = agent.run("Triage the alert")    # a failed enrich re-binds to its backup, recorded in
state.repairs                            #   state.repairs, the run finishes instead of restarting

# Run independent sub-graphs across a worker pool under one fair-share budget.
result = SubgraphScheduler(workers=4, budget=Budget(max_cost_usd=1.0), deadline_s=30).run(
    [SubgraphTask(build_branch(r), {"region": r}) for r in regions]
)
result.completed, result.partial, result.shares_usd

# Pause a graph durably, for a delay or an approval, without holding a worker.
def gate(s):
    return {"approval": wait_for_event(s, "approved")}
TimerService(flow).tick()                            # wake due sleep timers (restart-safe)
TimerService(flow).deliver(thread_id, "approved")    # wake an event wait
```

See [`examples/04_agents_and_tools.py`](../../examples/04_agents_and_tools.py)
and the [agents concept guide](../concepts/agents.md#orchestrator--planner-depth).

## Gotchas

- **Offline, a hierarchical crew falls back to keyword routing.** With no
  provider (or when the manager's plan fails to validate), delegation is
  deterministic keyword routing — so the same code runs in CI, but a crew tested
  only offline has never exercised its manager. Give each member `keywords=` so
  the fallback still routes sensibly.
- **Durable resume needs durable storage.** A graph is restart-safe only when
  `app.store` is SQLite/Postgres; the in-memory default loses the thread on exit.
  Recompile the *same* graph against the *same* store to `resume(thread_id)`.
- **`interrupt` pauses the thread, it does not block a worker.** A human gate can
  sit for days; wake it with `flow.resume(...)`, or `TimerService(flow).deliver`
  for an event wait — the scheduler is free in the meantime.
- **Bound every scheduler.** `SubgraphScheduler` takes a `Budget` and a
  `deadline_s`; an unbounded fan-out over regions is exactly the anti-pattern the
  bounded executor exists to prevent.

## Choosing a shape

| Shape | Use when |
|---|---|
| `app.agent` | one uncertain task, bounded exploration |
| `app.agent(planner="hierarchical", domain=...)` | a goal that decomposes into bounded sub-tasks |
| `app.crew` | several specialties, shared findings, delegation |
| `app.graph` | long-running, interruptible, auditable processes |
| `SubgraphScheduler` | many independent sub-graphs under one budget + deadline |
| `app.workflow` | fixed business steps with retries/compensation |
| `compose` | gluing any of the above into one observable pipeline |

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: Agents & orchestration](../concepts/agents.md)
- [Guide: add tools](add-tools.md)
- [Example: 04_agents_and_tools.py](../../examples/04_agents_and_tools.py)
- [Example: 05_orchestration.py](../../examples/05_orchestration.py)
- [Concept: Prompt compiler](../concepts/prompt-compiler.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
