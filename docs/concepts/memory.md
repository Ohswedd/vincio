# Memory

Vincio memory is layered, scoped, scored, and decaying — never a raw dump
of conversation history into prompts.

## Layers

| Layer | Scope | Lifetime |
|---|---|---|
| L0 working memory | one run | agent state dict |
| L1 session | one conversation | `MemoryScope.SESSION` |
| L2/L3 episodic + semantic | user | `MemoryScope.USER` |
| L4 tenant/org | organization | `MemoryScope.TENANT` / `ORGANIZATION` |
| L5 knowledge graph | long-term | `MemoryGraph` over all items |

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
memory.delete(memory_id)
```

## Retrieval scoring

```text
MemoryValue = relevance · recency · confidence · scope_match · stability
              ─────────────────────────────────────────────────────────
              token_cost + privacy_risk + staleness_penalty
```

## Decay and lifecycle

`confidence_t = confidence₀ · e^(−λ·age_days) · usage_boost · confirmation_boost`

`candidate → validated → active → decayed → archived → deleted` — run
`memory.decay_pass()` periodically (or via a cron) to apply transitions.

## Conflicts

A new memory that contradicts an old one supersedes it when clearly more
confident (`new > old + margin`); otherwise the conflict is stored and a
confirmation is required. Restatements confirm the existing memory instead
of duplicating it.
