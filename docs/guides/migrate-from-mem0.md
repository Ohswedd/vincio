# Coming from Mem0 to Vincio

Mem0 is a memory layer: you `add` facts scoped to a user, agent, or session
and `search` them back later. Vincio's `remember` / `recall` surface maps
those calls almost one-for-one, and because memory feeds the same context
packet, eval, and trace loop as the rest of the runtime you can move it over
incrementally without a hosted dependency.

## Concept mapping

| Mem0 | Vincio | Notes |
|---|---|---|
| `m.add(text, user_id=...)` | `app.remember("...", user_id="u1")` | write passes the guarded write policy before it lands |
| `m.search(query, user_id=...)` | `app.recall("query", user_id="u1")` | utility-scored against the task, not a raw vector dump |
| `m.get_all(user_id=...)` | `MemoryEngine` listing | enumerate via the engine instead of a flat dump |
| `m.update(id, data=...)` | `engine` edit / `app.run` consolidation | edits are audited |
| `m.delete(id)` | forget / decay | forgetting is first-class, not just deletion |
| `user_id` / `agent_id` / `run_id` | `MemoryScope.USER` / `AGENT` / `SESSION` | scope is inferred from the most specific owner id |
| extraction on add | guarded write policy | privacy, stability, contradiction, confidence checks |
| hosted platform | local engine | runs fully offline against the runtime's own stores |

## Bring your assets across

Mem0 has no document/tool interop adapters — the migration is the native
memory API. Point `remember` / `recall` at the same owner ids you used in
Mem0:

```python
from vincio import ContextApp

app = ContextApp(name="assistant", provider="openai", model="gpt-5.2")
app.add_memory()

# m.add("Alex prefers dark mode", user_id="alex")
app.remember("Alex prefers dark mode", user_id="alex")

# m.search("ui preferences", user_id="alex")
hits = app.recall("ui preferences", user_id="alex")
```

For lower-level control — listing, scopes, export — drive the engine
directly:

```python
from vincio import MemoryEngine, MemoryScope

engine = app.memory
user = engine.for_user("alex")          # also for_agent / for_session
user.remember("Alex works on the billing team")
user.recall("which team", top_k=3)

# session-scoped, like Mem0's run_id
session = engine.for_session("conv-42")
session.remember("Asked about refund policy")
```

`MemoryScope.SESSION` / `USER` / `AGENT` replace Mem0's `run_id` /
`user_id` / `agent_id` — `remember` infers the scope from the most
specific owner id you pass.

## In Vincio

Before — a standalone Mem0 store wired by hand into each prompt:

```python
from mem0 import Memory

m = Memory()
m.add("Customer is on the enterprise plan", user_id="acme")
ctx = m.search("plan tier", user_id="acme")
prompt = build_prompt(question, ctx)     # you assemble + budget context yourself
answer = llm(prompt)
```

After — memory is part of the run, scored and budgeted into the packet
alongside retrieved evidence:

```python
from vincio import ContextApp

app = ContextApp(name="support", provider="openai", model="gpt-5.2")
app.add_source("kb", path="./docs", retrieval="hybrid")
app.add_memory()

app.remember("Customer is on the enterprise plan", user_id="acme")

result = app.run("what plan is the customer on?", user_id="acme")
result.output        # memory + evidence, compiled into one packet
result.citations     # provenance for both
result.trace_id      # the whole run is traced
```

Turn on write-back so grounded facts from a run are remembered for next
time, instead of you calling `add` after every turn — set it in `vincio.yaml`:

```yaml
memory:
  write_back: [facts]      # verifiable, evidence-supported facts from runs persist
  fact_min_support: 0.5    # how much evidence a claim needs to become a candidate memory
```

Facts still pass the guarded write policy and land as *candidate* memories
(with evidence provenance) until confirmed — they never bypass the same
admission checks as a manual `remember`.

## What Vincio adds

- **Guarded write policy** on every memory: credential/PII screening,
  stability scoring (volatile statements rejected), contradiction detection,
  and confidence assignment — Mem0 stores whatever you hand it.
- **Decay and TTL** so stale memories fade instead of accumulating;
  forgetting and consolidation are first-class operations, not just delete.
- **Hash-chained audit log** — edits, forgets, and exports are tamper-evident
  entries, not silent mutations.
- **GDPR-style owner export** via `engine.export_owner_data(...)` for
  auditable access and erasure across a user's memories.
- **Episodic→semantic consolidation** that promotes session memories to
  durable user/agent scope while keeping provenance.
- **One loop**: memory is scored into the same context packet, measured by
  the same evals and gates, and recorded in the same traces as retrieval and
  tools — no separate hosted service to reason about.

## Next steps

- [Memory concepts](../concepts/memory.md) — layers, scoring, decay, consolidation.
- [Build a RAG app](build-rag-app.md) — combine memory with retrieved sources.
- [Optimize context](optimize-context.md) — how memory is scored and budgeted into a packet.
- [Run evals](run-evals.md) and [evals concepts](../concepts/evals.md) — measure recall and personalization.
- [Observability concepts](../concepts/observability.md) — traces, cost, and feedback for memory-backed runs.
- [Vincio vs Mem0](../comparisons/mem0.md) — the full comparison.
