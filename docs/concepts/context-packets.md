# Context packets and the context compiler

The central unit in Vincio is not a prompt template; it is the **Context
Packet**: the complete, budgeted, provenance-aware bundle of instructions,
user input, evidence, memory, tool specs, schema, and policies passed to a
model.

## The pipeline

```text
collect_candidates → normalize → classify → score → remove_duplicates
→ resolve_conflicts → compress_or_distill → allocate_budget
→ order_context → render_context_packet → validate_packet
```

## Scoring

Every candidate context item is scored (`vincio.context.scoring`):

```text
Score(c_i | τ) = w_r·relevance + w_n·novelty + w_a·authority + w_f·freshness
              + w_p·provenance + w_q·answerability + w_m·memory_value
              − w_t·token_cost − w_d·duplication − w_k·leakage_risk
```

Each component is normalized to `[0, 1]` with a `ScoringWeights` vector. By
default relevance is a fast lexical estimator — stemmed content-term overlap
with IDF-free dampening (`lexical_similarity`) — and upgrades to embedding
cosine when an embedder is available; duplication uses word-shingle Jaccard
(`shingle_similarity`) for near-duplicate detection. Items below the threshold
are excluded, and every exclusion is reported:

```python
result = app.run("Which clauses are risky?")
print(result.excluded_context)
# [{"id": "D4:C9", "reason": "low_relevance", "score": 0.12},
#  {"id": "M7", "reason": "privacy_scope_mismatch"}]
```

## Budget allocation

The input-token budget is split across blocks (instructions, examples, user
task, evidence, memory, tool results, schema) with task-adaptive fractions.
The `BudgetAllocator` holds a `DEFAULT_ALLOCATION` table and reshapes it per
`TaskType`: document QA spends ~70% on evidence and ~2% on examples;
classification spends 0% on evidence and ~35% on examples; a tool action
reserves ~30% for tool results. Allocation is two-pass — blocks whose size is
already known (instructions, schema, user task) are charged **at cost** as
`fixed_costs`, then the *entire* remainder is distributed over the flexible
blocks (evidence, memory, tool results) proportionally to their fractions, so
tokens a fixed block does not need flow to evidence rather than going unused.
Two additive knobs sharpen this: `reserve_tokens` holds headroom back from the
flexible blocks for the model's response and tool-loop turns (so the allocator
accounts for the *full* window, not input only), and a learned per-task table
tuned from eval outcomes by `vincio.optimize`'s budget learner can override the
fixed fractions for a task. Oversized evidence is compressed extractively
(`extractive_compress` keeps query-relevant sentences) instead of being dropped.

## Conflict resolution

When two items contradict:

- higher authority wins if the authority gap is large,
- newer wins if authority is similar but freshness differs,
- otherwise both are kept and the conflict is reported to the model.

## Evidence ledger

Instead of raw chunks, the compiler can distill evidence into a ledger of
claims with provenance:

```yaml
- id: E1
  source: D1:C7
  claim: "The contract renews automatically unless terminated 60 days before renewal."
  confidence: 0.94
```

Enable with `ContextCompilerOptions(use_evidence_ledger=True)`.

## Context IR

The `ContextIR` is the provider-neutral intermediate representation; the
prompt compiler renders it into provider-specific messages (system blocks
with cache hints for Anthropic, developer messages for OpenAI, etc.).

## Caching, recompiles, and zero-copy packets

Compilation is a pure function of its inputs, so it caches
content-addressed: with `cache.context_compile_cache` (on by default),
identical compile inputs return the compiled result without re-running
scoring, dedup, conflict resolution, or selection. Edits recompile
incrementally:

```python
edited = await compiler.recompile(
    compiled, add_evidence=[new_item], remove_evidence_ids=["D4:C9"]
)
```

Large packets can also skip duplicating evidence text
(`performance.slim_packets`): evidence entries carry a `text_hash`
reference, the text lives once on the IR, and
`packet.evidence_text(id)` / `packet.materialize()` resolve it lazily.
`packet.iter_json()` streams the serialized packet chunk by chunk. See the
[performance guide](../guides/performance.md).

## Best practice

- **Read `excluded_context` before you widen retrieval.** A weak answer is
  usually a scoring or budget decision, not a missing document — the exclusion
  reasons tell you whether the evidence was dropped for `low_relevance`,
  `budget_exceeded`, or a `privacy_scope_mismatch`, each of which has a
  different fix.
- **Let the task type drive the budget.** The allocator already spends ~70% on
  evidence for document QA and ~0% for classification; overriding fractions by
  hand usually fights a table that eval-tuned learned budgets improve for free.
- **Recompile, don't rebuild.** For an edit-and-rerun loop, `compiler.recompile`
  with `add_evidence` / `remove_evidence_ids` reuses the content-addressed
  compile cache instead of re-scoring the whole candidate set.

## Gotchas

- **Scoring is deterministic but relevance is only lexical by default.** Without
  an embedder, "renews annually" won't match "yearly renewal" on terms alone —
  wire an embedder (or [anchors](context-anchors.md)) when paraphrase matters.
- **Conflicts that are close are kept, not resolved.** When authority and
  freshness are similar, both items travel into the packet and the discrepancy
  is surfaced to the model — the compiler refuses to silently pick a winner.
- **`slim_packets` trades a copy for a lookup.** With `performance.slim_packets`
  on, evidence text lives once on the IR behind a `text_hash`; downstream code
  must resolve it via `packet.evidence_text(id)` / `packet.materialize()` rather
  than expecting inline text on every entry.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: optimize prompts, context, and routing](../guides/optimize-context.md)
- [Guide: Performance & streaming](../guides/performance.md)
- [Example: 11_advanced_context.py](../../examples/11_advanced_context.py)
- [Concept: Packet compile receipt](compile-receipt.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
