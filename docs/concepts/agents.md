# Agents and workflows

Vincio avoids uncontrolled agent loops. The default is a **bounded DAG**
with explicit budgets and validation.

## Agent engine

```python
agent = app.agent(
    tools=[web_search, database_lookup],
    planner="dag",          # dag | dynamic | react | direct
    max_steps=8,
)
state = agent.run("Find the latest pricing discrepancy and draft a report")
state.termination_reason   # objective_complete | validation_passed | max_steps | budget_exhausted | ...
state.metrics()            # success, steps, tool calls/errors, cost, tokens
```

- **Planners** — static task-shaped DAGs, dynamic LLM-generated DAGs
  (schema-validated, safe fallback), ReAct loops, direct answers.
- **Steps** — retrieve / think / tool / validate / ask_human / finalize.
- **Termination** — objective complete, validation passed, max steps,
  budget exhausted, safety violation, approval required, unrecoverable error.
- **Critic/validator** — drafts are critiqued against objective and evidence
  before finalization; structured output is validated by the output engine.
- **Handoffs** — `HandoffRouter` routes objectives between named agents with
  bounded depth and merged provenance.

## Multi-agent crews

A crew binds named roles to bounded executors and runs them as a team over a
shared **blackboard** — versioned, author-attributed working memory that every
member reads and writes:

```python
crew = app.crew(
    name="refund_team",
    members=[
        {"name": "researcher", "goal": "gather the numbers", "keywords": ["find"]},
        {"name": "writer", "goal": "draft the recommendation"},
    ],
    process="sequential",   # sequential | parallel | hierarchical
)
result = crew.run("Explain the Q3 refund trend")
result.output            # final answer
result.reports           # per-member answers, termination reasons, metrics
result.delegations       # who delegated what to whom (hierarchical)
result.blackboard        # JSON snapshot of the shared board
```

- **Processes** — sequential (each member sees prior posts), parallel
  (concurrent fan-out, dict of answers), hierarchical (a manager decomposes
  the objective, delegates, reviews, and either finishes or delegates
  follow-ups — LLM-planned with a deterministic keyword-routing fallback).
- **Termination guarantees** — every member runs under a scaled share of the
  crew budget (`budget_fraction` or an equal split), the crew checks its own
  budget before each delegation, and hierarchical review is capped at
  `max_rounds`. A crew cannot loop.
- **Tracing & evals** — the crew emits a `crew` span, every member a
  `crew_agent` span, and `result.metrics()` aggregates per-member
  `AgentState.metrics()` for eval gates.

## Durable stateful graphs

For long-running, interruptible processes, `app.graph()` builds a
**checkpointed state graph**: nodes are functions over a shared state dict,
edges (static or conditional) pick what runs next, and a checkpoint persists
in the app's metadata store after every step — so threads survive interrupts
and, with SQLite/Postgres, process restarts.

```python
graph = app.graph("contract_review")
graph.add_node("analyze", lambda s: {"risk": assess(s["clauses"])})
graph.add_node("approve", lambda s: {"ok": interrupt(s, {"q": "proceed?"})})
graph.add_node("report", lambda s: {"report": render(s)})
graph.add_edge("analyze", "approve")
graph.add_edge("approve", "report")

review = graph.compile(interrupt_before=[])      # or interrupt_before=["report"]
paused = review.invoke({"clauses": [...]})       # status == "interrupted"
review.update_state(paused.thread_id, {"risk": "medium"})   # edit…
done = review.resume(paused.thread_id, value=True)          # …and resume
```

- **Human-in-the-loop** — pause statically (`interrupt_before` /
  `interrupt_after`) or dynamically from inside a node (`interrupt(state,
  payload)`); resume with a value and the paused node re-runs and receives it.
- **Time-travel** — `history(thread_id)` lists every checkpoint;
  `fork(checkpoint_id)` branches a new thread that re-executes
  deterministically from that step.
- **Bounded** — `max_steps` caps cyclic graphs; per-key `reducers` merge
  parallel branch updates deterministically; an optional Pydantic
  `state_schema` validates state after every merge.
- Every node emits a `graph_node` span. Workflow approval gates pause the
  same way: a gate with no `approval_fn` returns a `paused` result, and
  `workflow.resume(result, approvals={"ship": True})` continues without
  re-running done steps.

## Declarative composition

Chains read like data: wrap any step — function, agent, crew, workflow, or
compiled graph — and pipe with `|`. Results are normalized between steps and
every node streams events and emits a `compose_node` span.

```python
pipeline = compose(fetch) | summarize | app.agent(planner="direct")
answer = pipeline.call("Q3 refunds")
async for event in pipeline.astream("Q3 refunds"):
    print(event.type, event.node)        # node_start / node_end / done

parallel(fast=cheap_agent, thorough=rag_agent)   # dict of branch results
branch(router, {"billing": billing_flow, "legal": legal_flow})
```

## Runtime backends

Vincio orchestrates without lock-in: adapters export the same definitions to
external runtimes. `LangGraphBackend` translates a `StateGraph` (nodes
transfer as-is; edges, conditional edges, entry, and `END` are mapped) and
`OpenAIAgentsBackend` exports agents and crews (a crew becomes a manager
agent with handoffs to every member). Both import their runtime lazily —
nothing in core Vincio depends on either package.

## Workflow engine

For deterministic business processes:

```python
workflow = app.workflow("contract_review")
workflow.step("ingest", ingest_documents)
workflow.step("retrieve", retrieve_clauses, depends_on=["ingest"])
workflow.step("analyze", analyze_risk, depends_on=["retrieve"], retries=2, timeout_s=30)
workflow.step("approve", request_signoff, depends_on=["analyze"], approval=True)
result = workflow.run({"files": ["msa.pdf"]})
```

Features: DAG levels run in parallel, retries with backoff, timeouts,
conditional branching (`when=`), compensation on failure (reverse order),
human approval gates, typed parameter binding (a step argument named after
a prior step receives its output), and trace spans for every step.

| Mode | Use when |
|---|---|
| Chain | fixed known sequence |
| Workflow DAG | business process with branches |
| Agent | uncertain task requiring exploration |
| Hybrid | workflow steps that call `app.agent(...)` |

## The agentic frontier (1.10, `@experimental`)

The agent loop gains four capabilities, all on the same bounded, permissioned,
audited runtime:

- **Level-parallel DAG + `plan_and_execute`** — the executor runs each
  topological level's independent steps concurrently (bounded by
  `max_parallel_steps`), and `planner="plan_and_execute"` drives a real
  plan → execute → observe → replan loop (`Planner.replan`) bounded by
  `max_replans` and the budget.
- **In-loop context compaction** — `agents/compaction.py` `ContextCompactor`
  folds older tool/observation turns into a rolling extractive summary once the
  working context exceeds a token budget, replacing fixed slicing (tool-call
  pairs stay intact).
- **Deep-research agent** — `app.research(question, budget=ResearchBudget(...))`
  loops search → read → reflect → verify → synthesize over the query-understanding
  planners and the grounded-fact extractor, dedups sources, and emits a cited
  report through the 1.9 `CitedReportBuilder` — every claim cited and grounded by
  construction, scored for citation coverage / grounding / source diversity.
- **Agent memory OS** — `app.enable_memory_os(...)` exposes self-editing memory
  (`memory_append` / `memory_replace` / `memory_search` / `memory_archive`) as
  permissioned, audited tools over the guarded write pipeline, with a
  context-pressure pager between in-context core memory and the archival store.

**Computer-use** (`app.enable_computer_use("mock"|"playwright"|"provider")`) adds
a navigate / click / type / screenshot action vocabulary as approval-gated tools,
and **provider-native hosted tools** (`app.use_hosted_tools([...])`) surface
OpenAI Responses built-ins as namespaced, permissioned tools. Both run behind a
pluggable `IsolationBackend` in `tools/sandbox.py` (subprocess is the zero-dep
default but not a security boundary; container / microVM / gVisor / WASM are real
boundaries, enforced by `require_real_isolation` for code-executing and
computer-use workloads). See
[`examples/34_continual_loop_and_agentic_frontier.py`](../../examples/34_continual_loop_and_agentic_frontier.py).
