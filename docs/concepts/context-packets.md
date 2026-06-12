# Context packets and the context compiler

The central unit in Vincio is not a prompt template — it is the **Context
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

Items below the threshold are excluded — and every exclusion is reported:

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
