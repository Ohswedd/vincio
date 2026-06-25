# Agents and workflows

Vincio avoids uncontrolled agent loops. The default is a **bounded DAG**
with explicit budgets and validation.

## Agent engine

```python
agent = app.agent(
    tools=[web_search, database_lookup],
    planner="dag",          # dag | dynamic | react | direct | plan_and_execute | hierarchical
    max_steps=8,
)
state = agent.run("Find the latest pricing discrepancy and draft a report")
state.termination_reason   # objective_complete | validation_passed | max_steps | budget_exhausted | ...
state.metrics()            # success, steps, tool calls/errors, cost, repairs, tokens
```

- **Planners**: static task-shaped DAGs, dynamic LLM-generated DAGs
  (schema-validated, safe fallback), ReAct loops, direct answers, a
  `plan_and_execute` replanning loop, and a hierarchical (HTN) planner
  (see [Orchestrator & planner depth](#orchestrator--planner-depth)).
- **Steps**: retrieve / think / tool / validate / ask_human / finalize.
- **Termination**: objective complete, validation passed, max steps,
  budget exhausted, safety violation, approval required, unrecoverable error.
- **Critic/validator**: drafts are critiqued against objective and evidence
  before finalization; structured output is validated by the output engine.
- **Handoffs**: `HandoffRouter` routes objectives between named agents with
  bounded depth and merged provenance.

## Multi-agent crews

A crew binds named roles to bounded executors and runs them as a team over a
shared **blackboard**, versioned, author-attributed working memory that every
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

- **Processes**: sequential (each member sees prior posts), parallel
  (concurrent fan-out, dict of answers), hierarchical (a manager decomposes
  the objective, delegates, reviews, and either finishes or delegates
  follow-ups, LLM-planned with a deterministic keyword-routing fallback).
- **Termination guarantees**: every member runs under a scaled share of the
  crew budget (`budget_fraction` or an equal split), the crew checks its own
  budget before each delegation, and hierarchical review is capped at
  `max_rounds`. A crew cannot loop.
- **Tracing & evals**: the crew emits a `crew` span, every member a
  `crew_agent` span, and `result.metrics()` aggregates per-member
  `AgentState.metrics()` for eval gates.

## Durable stateful graphs

For long-running, interruptible processes, `app.graph()` builds a
**checkpointed state graph**: nodes are functions over a shared state dict,
edges (static or conditional) pick what runs next, and a checkpoint persists
in the app's metadata store after every step, so threads survive interrupts
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

- **Human-in-the-loop**: pause statically (`interrupt_before` /
  `interrupt_after`) or dynamically from inside a node (`interrupt(state,
  payload)`); resume with a value and the paused node re-runs and receives it.
- **Time-travel**: `history(thread_id)` lists every checkpoint;
  `fork(checkpoint_id)` branches a new thread that re-executes
  deterministically from that step.
- **Bounded**: `max_steps` caps cyclic graphs; per-key `reducers` merge
  parallel branch updates deterministically; an optional Pydantic
  `state_schema` validates state after every merge.
- Every node emits a `graph_node` span. Workflow approval gates pause the
  same way: a gate with no `approval_fn` returns a `paused` result, and
  `workflow.resume(result, approvals={"ship": True})` continues without
  re-running done steps.

## Orchestrator & planner depth

Multi-step execution plans deeper, recovers from failure, and schedules fairly
at scale, all additive on the same bounded, permissioned, audited runtime.

### Hierarchical (HTN) planning

An HTN domain decomposes a goal into a sub-goal tree and binds each leaf to a
bounded step (a tool, a retrieval, a reasoning turn, or the finalize). It is just
another planner mode, so it composes with plan repair and cost-aware selection:

```python
from vincio.agents import HTNDomain

domain = (
    HTNDomain()
    .method("root", ["assess", "respond"])
    .method("assess", ["classify", "enrich"], ordering="parallel")  # leaves run concurrently
    .operator("classify", step_type="think", instruction="classify the severity")
    .operator("enrich", step_type="tool", tool_name="enrich_primary", fallbacks=["enrich_backup"])
    .operator("respond", step_type="finalize", instruction="recommend a response")
)
agent = app.agent(tools=[...], planner="hierarchical", domain=domain)
```

With a domain the decomposition is deterministic (offline-safe, the first method
whose `when=` precondition matches is chosen); without one, the model proposes a
two-level goal → sub-goal → step decomposition, falling back to a static plan
offline. The resolved tree is on `planner.last_plan_tree` for inspection.

### In-place plan repair

When a step fails the executor edits the *remaining* plan instead of restarting:

- **re-bind** a failed tool to an alternative (an explicit `fallback_tools` set,
  then a name-overlap sibling in the toolset);
- **substitute** a failed tool with no alternative to a reasoning step;
- **reorder** a validation contradiction into a corrective re-analysis before the
  finalize; and
- **drop** the optional tail under a budget shock, finalizing on what is done.

Repair is on by default (`app.agent(...)`; disable with
`AgentExecutor(repair=False)`). Each repair is recorded as an `AgentState.repairs`
entry, a `plan.repaired` event, and a `plan_repair` trajectory step, auditable,
not a silent retry. Repairs are bounded (`PlanRepairer(max_repairs=3)`).

### Cost-aware action selection

`cost_aware_models=[cheap, …, strong]` makes the executor read `ModelRegistry`
pricing and capabilities and the live budget to spend the **cheapest capable
model per step**, escalating one tier only when the prior step's confidence was
low:

```python
agent = app.agent(cost_aware_models=["gpt-5.2-mini", "gpt-5.2"])
```

A model that cannot serve the call (missing tools, structured output, reasoning,
or context) is never chosen; cost never overrides capability. Each pick is a
`SelectionDecision` recorded in working memory.

### Parallel sub-graph scheduling

`SubgraphScheduler` runs independent durable sub-graphs concurrently across a
worker pool under one fair-share budget, with an SLA deadline that returns a
partial result rather than blowing the deadline:

```python
from vincio import SubgraphScheduler, SubgraphTask, Budget

tasks = [SubgraphTask(build_branch(r), {"region": r}, weight=1.0) for r in regions]
result = SubgraphScheduler(workers=4, budget=Budget(max_cost_usd=1.0), deadline_s=30).run(tasks)
result.completed        # finished sub-graphs (each a GraphResult)
result.partial          # deadline / unfinished, durable latest checkpoint, resumable
result.shares_usd       # the fair-share split (sums to the budget)
result.peak_concurrency # genuine parallelism achieved
```

Each sub-graph is a lease-guarded, CAS-committed durable thread on the
[distributed backend](#runtime-backends), so a sub-graph that misses the deadline
is not lost; its latest checkpoint is the partial result.

### Durable timers & scheduled steps

A graph can pause for a wall-clock delay, a webhook, or an approval **without
holding a worker**; the wake condition rides the checkpoint, so it survives a
restart:

```python
from vincio.agents import sleep_for, wait_for_event, TimerService

def cool_off(state):
    sleep_for(state, 86_400)          # pause ~1 day, durably
    return {"stage": "cooled_off"}

def gate(state):
    return {"approval": wait_for_event(state, "approved")}

# elsewhere, possibly after a restart, a fresh process resumes due work:
TimerService(compiled_graph).tick()                       # wake due sleep timers
TimerService(compiled_graph).deliver(thread_id, "approved", payload={...})  # wake an event wait
```

`pending_timers` / `due_timers` / `resume_due_timers` / `deliver_event` are the
module-level forms. See
[`examples/04_agents_and_tools.py`](../../examples/04_agents_and_tools.py).

## Declarative composition

Chains read like data: wrap any step (function, agent, crew, workflow, or
compiled graph) and pipe with `|`. Results are normalized between steps and
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
agent with handoffs to every member). Both import their runtime lazily;
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

## The agentic frontier

The agent loop gains four capabilities, all on the same bounded, permissioned,
audited runtime:

- **Level-parallel DAG + `plan_and_execute`**: the executor runs each
  topological level's independent steps concurrently (bounded by
  `max_parallel_steps`), and `planner="plan_and_execute"` drives a real
  plan → execute → observe → replan loop (`Planner.replan`) bounded by
  `max_replans` and the budget.
- **In-loop context compaction**: `agents/compaction.py` `LoopCompactor`
  folds older tool/observation turns into a rolling extractive summary once the
  working context exceeds a token budget, replacing fixed slicing (tool-call
  pairs stay intact).
- **Deep-research agent**: `app.research(question, budget=ResearchBudget(...))`
  loops search → read → reflect → verify → synthesize over the query-understanding
  planners and the grounded-fact extractor, dedups sources, and emits a cited
  report through the `CitedReportBuilder`, every claim cited and grounded by
  construction, scored for citation coverage / grounding / source diversity.
- **Agent memory OS**: `app.enable_memory_os(...)` exposes self-editing memory
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
[`examples/08_optimization_self_improvement.py`](../../examples/08_optimization_self_improvement.py).

The rung above the flat tools is the **action plane**: `app.computer_use(...)`
returns a `ComputerEnvironment` for an agent that drives a screen *safely*. Over a
pluggable `ScreenBackend` (deterministic `MockScreen` offline; browser / OS
accessibility / remote-desktop adapters behind `vincio[computer-use]`) it perceives
the UI as typed, addressable `UIElement`s and grounds an intent to a `UIAction` bound
by a **stable selector** (role + accessible name, not a pixel). Every action is
**pre-gated** against an `ActionPolicy` (a destructive or out-of-scope action is gated
like a write tool, behind an approval callback), performed, **post-verified** against
its expected effect, and **undone on divergence**, the computer-use analogue of a
saga's compensation, into a typed `ActionOutcome`, every step on the same
hash-chained audit log. A `ComputerTask` carries a goal and a declarative verifier, so
a run projects onto the same `Trajectory` the trajectory metrics and test-time search
already score. See the [computer-use guide](../guides/computer-use.md) and
[`examples/04_agents_and_tools.py`](../../examples/04_agents_and_tools.py).

## World-model / simulation-based planning

The bounded planners and the stateful-environment harness let an agent *act on* and
*evaluate* the live world. `vincio.agents.world_model` adds the rung above: an agent
that **learns a model of its tools and plans against it**, searching imagined
rollouts before acting so a wrong move costs a simulated step, not a live one.

- **`WorldModel`** is a deterministic dynamics model fit offline from recorded
  reset/step `Transition`s (`record_transitions(env, sequences)`). For each tool it
  learns the *parameterized* state effect (constant / argument / numeric-step value)
  under a *learned precondition* (the discriminative state field), so
  `predict(obs, action)` returns a `PredictedStep` that fails a refund on a
  processing order but succeeds on a cancelled one, and generalizes a cancel seen
  on one order to another. `imagine(obs, actions)` rolls a plan forward without
  touching a tool.
- **`WorldModel.calibrate(holdout)`** returns a `CalibrationReport`; the model earns
  planning weight only once its predictions track the real environment within
  tolerance, the way a judge ensemble earns gating weight.
- **`ModelPredictivePlanner`** searches imagined rollouts with the test-time-search
  beam, commits the best first action to the real environment, observes, and
  re-plans, returning an `MPCResult`. By default it refuses an uncalibrated model.
  At a fixed action budget it matches or beats reactive (one-step) planning: on the
  `make_vault_environment` trap world it opens the vault a reactive planner is
  trapped short of. See
  [`examples/11_advanced_context.py`](../../examples/11_advanced_context.py).
