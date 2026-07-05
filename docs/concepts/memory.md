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
user = memory.for_user("u1")        # also: for_agent, for_session, for_team
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

## How recall works — filter, then score

`asearch` (behind `search` / `recall`) is two passes over the store, never a
raw vector top-k:

1. **Filter pass** removes anything ineligible *before* scoring: wrong scope
   (`_scope_match`), privacy above the caller's `max_privacy`, TTL-expired, a
   bi-temporal interval that does not contain the recall moment
   (`valid_at`), a per-memory ACL the `reader` is not on (`readable_by`), a
   `purpose` whose consent was withdrawn (a configured `consent_ledger`), and
   decayed confidence below `min_confidence`.
2. **Score pass** ranks the survivors. Lexical and vector relevance are fused in
   one batched embed call (`(1−vector_weight)·lexical + vector_weight·cosine`),
   lifted `+0.25` per task-entity mention and `+0.2` when the graph links the
   memory to a task entity, then divided through the `MemoryValue` ratio above.
   Every term is returned on `MemorySearchResult.components` for audit, and the
   selected `top_k` increment `usage_count` — which feeds the `usage_boost` in
   the next decay pass, so *use* is what keeps a memory alive.

## Bi-temporal validity, ACLs & consent

A memory carries two independent clocks: *transaction time*
(`created_at` / `updated_at` — when the system learned the fact) and *valid
time* (`valid_from` / `valid_to` — when the fact is true in the world). Keeping
them apart makes recall answerable **as of** a past moment without rewriting
history: a correction closes the old item's `valid_to` and opens a new one, and
`recall(..., as_of=T)` returns what was believed valid at `T`, superseded items
included.

```python
memory = app.memory
memory.write_fact("Plan tier: Pro", scope="user", owner_id="u1",
                  valid_from=jan, valid_to=jun)     # true Jan–Jun
memory.write_fact("Plan tier: Enterprise", scope="user", owner_id="u1",
                  valid_from=jun)                   # true from Jun
memory.recall("plan tier", user_id="u1", as_of=march)   # → Pro, not Enterprise
```

Two more per-memory governance rails ride the same recall:

- **Reader ACLs** — an `acl=[...]` on the write plus `reader=` on recall gate
  team-shared memory to permitted members (`readable_by`); an empty ACL stays
  scope-governed, so existing memories are unaffected.
- **Consent / purpose** — each memory records the `purpose` it was collected for
  and its `consent_id`; a `MemoryEngine` configured with a `consent_ledger`
  drops any item whose purpose has lost consent, so a withdrawal is enforced at
  read time, not only at erasure.

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

## Gotchas

- **Volatile statements are refused at write.** The stability check keeps
  memory to durable facts, preferences, goals, and decisions — a passing remark
  never earns a slot, and credentials (always) and sensitive PII (by default)
  are blocked outright.
- **Write-backs start penalized.** Memories written back from evidence or tool
  results are *candidates* carrying a status penalty until `confirm`ed, so they
  cannot outrank curated facts on their first appearance.
- **`as_of` widens the status filter.** Current recall sees only the live set;
  an as-of query deliberately includes superseded and archived items that were
  valid then, so counts differ from a plain recall.
- **Decay is lazy.** Expired items never surface, but the
  `candidate → … → deleted` transitions only run on `decay_pass` (schedule it,
  or `vincio memory decay`) — nothing garbage-collects on its own.
- **A populated ACL flips the default.** With `acl` set, a `reader=None` recall
  is refused; only listed reader ids get the item. Leave the ACL empty to keep
  the scope's ownership rule.

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
