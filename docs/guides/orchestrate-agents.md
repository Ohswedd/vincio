# How-to: orchestrate multi-agent systems

This guide builds a support-triage system three ways — a crew, a durable
graph, and a composed pipeline — all bounded, traced, and resumable.

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
runs under its budget share, and the manager is capped at `max_rounds` — the
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
from vincio.agents import branch, parallel

pipeline = (
    compose(normalize)
    | branch(lambda t: t["lane"], {"crew": crew, "graph": flow})
    | format_reply
)
reply = pipeline.call(raw_ticket)
```

Every node — including the crew and the graph — streams `node_start` /
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

## Choosing a shape

| Shape | Use when |
|---|---|
| `app.agent` | one uncertain task, bounded exploration |
| `app.crew` | several specialties, shared findings, delegation |
| `app.graph` | long-running, interruptible, auditable processes |
| `app.workflow` | fixed business steps with retries/compensation |
| `compose` | gluing any of the above into one observable pipeline |
