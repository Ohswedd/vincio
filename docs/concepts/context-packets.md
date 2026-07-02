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

Items below the threshold are excluded, and every exclusion is reported:

```python
result = app.run("Which clauses are risky?")
print(result.excluded_context)
# [{"id": "D4:C9", "reason": "low_relevance", "score": 0.12},
#  {"id": "M7", "reason": "privacy_scope_mismatch"}]
```

## Budget allocation

The input-token budget is split across blocks (instructions, examples, user
task, evidence, memory, tool results, schema) with task-adaptive fractions:
document QA gets ~70% evidence; classification gets 0% evidence and more
examples. Known-size blocks are charged at cost; the remainder is
distributed to flexible blocks. Oversized evidence is compressed
extractively (query-relevant sentences) instead of being dropped.

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
