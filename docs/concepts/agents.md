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
