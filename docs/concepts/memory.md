# Memory

Vincio memory is layered, scoped, scored, and decaying, never a raw dump
of conversation history into prompts. Memories carry **confidence,
provenance, decay, and conflict resolution**, and every recall
utility-scores them against the task before they enter a packet.

## Layers

| Layer | Scope | Lifetime |
|---|---|---|
| L0 working memory | one run | agent state dict |
| L1 session (episodic) | one conversation | `MemoryScope.SESSION` |
| L2/L3 episodic + semantic | user / agent | `MemoryScope.USER` / `AGENT` |
| L4 tenant/org | organization | `MemoryScope.TENANT` / `ORGANIZATION` |
| L5 knowledge graph | long-term | `MemoryGraph` over all items |

## Personalization API

`remember` / `recall` infer scope from the most specific owner id and
classify the memory type; scoped handles bind one owner:

```python
app.remember("User prefers concise technical answers", user_id="u1")
app.recall("how should answers be written", user_id="u1")

memory = app.memory
user = memory.for_user("u1")        # also: for_agent, for_session, for_tenant
user.remember("User works in the compliance department")
user.recall("which department", top_k=3)
user.export()                       # GDPR-style, audited
```

## Write policy

Every write passes: extraction → type classification (fact/preference/goal/
decision) → privacy check (credentials always blocked, sensitive PII blocked
by default) → stability check (volatile statements rejected) → contradiction
check → confidence assignment → provenance (`source_trace_id`).

```python
app.add_memory(scope="user", strategy="semantic_graph")
memory = app.memory
memory.write_fact("User prefers concise technical answers", scope="user", owner_id="u1", type="preference")
memory.search("current project goals", user_id="u1")
memory.confirm(memory_id)
memory.edit(memory_id, content="User prefers short bullet answers")  # audited
memory.forget(memory_id, reason="user_request")                      # audited
```

## Hybrid retrieval scoring

Recall fuses **lexical + vector relevance with graph adjacency** in one
query: relevance is `(1−w)·lexical + w·cosine` over any `Embedder`
(offline hash embedder by default, content-addressed vector cache), plus a
boost for memories the graph links to the task's entities. Candidates
written back from evidence/tools carry a status penalty until confirmed.

```text
MemoryValue = relevance · recency · confidence · scope_match · stability · status
              ─────────────────────────────────────────────────────────────────
              token_cost + privacy_risk + staleness_penalty
```

## Consolidation tiers

Episodic session memories summarize into a few durable semantic memories,
promote to user/agent scope, and deduplicate, with full provenance:
promoted items record `consolidated_from`, and the episodes are archived
(never silently dropped) with a `consolidated_into` backref.

```python
report = await memory.consolidate("session-1", user_id="u1")
report.promoted, report.deduplicated, report.items[0].metadata["consolidated_from"]
```

## Decay, TTL, and importance-weighted retention

`confidence_t = confidence₀ · e^(−λ·age_days) · usage_boost · confirmation_boost`

Per-scope TTLs apply on write (`memory.ttl_days`, sessions default to 30
days); expired items never surface. `memory.decay_pass()` (run it
periodically or via `vincio memory decay`) transitions
`candidate → validated → active → decayed → archived → deleted`, and
retention is **importance-weighted**: heavily used, confirmed, stable
preferences/decisions tolerate lower decayed confidence before archival.

## Forgetting & GDPR-style hygiene

User-driven `edit` / `forget` / `export_owner_data` / `erase_owner_data`
flow through the hash-chained audit log as `memory_edit`, `memory_delete`,
`memory_export`, and `memory_erase` entries.

## Conflicts

A new memory that contradicts an old one supersedes it when clearly more
confident (`new > old + margin`); otherwise the conflict is stored and a
confirmation is required. Restatements confirm the existing memory instead
of duplicating it.

## Write-back from runs

Step 16 of the runtime writes back what the run learned, governed by
`memory.write_back`: durable statements from the `input` (default), cited
`evidence`, and successful `tools` results, the latter two as *candidate*
memories with provenance that must earn their way into future packets.

## Eval harness

`evaluate_memory(engine, cases)` measures recall precision/recall@k,
contradiction rate, staleness, and personalization lift against labeled
cases; VincioBench runs it as the `memory` family and
`benchmarks/budgets.json` gates the results in CI.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: Build a chat product: the Assistant](../guides/assistant.md)
- [Guide: close the loop](../guides/close-the-loop.md)
- [Example: 03_memory.py](../../examples/03_memory.py)
- [Concept: Context packets & long-horizon governance](context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
