# The packet compile receipt

The [context packet](context-packets.md) is the governed boundary — candidate
evidence scored, deduped, budgeted, and packed into the exact unit sent to the
model. The [trace](observability.md) records the *stages* that produced it, and
the packet itself carries provenance. What neither gives you is a single, compact
artifact you can attach to a pull request or an incident and *diff* across compile
changes.

The **compile receipt** is that artifact. It is a small manifest that proves *why
this exact packet was compiled* — which items were included and why, which were
excluded and why, which conflicts were resolved and by which rule, what the budget
and privacy posture were — and it does so **fingerprint-heavy and text-light**: it
carries ids, content hashes, per-item scores, and summaries, and never the raw
prompt or evidence text. That makes it safe to share when a production run is
surprising: a bad answer, stale memory, a privacy-scope mismatch, budget trimming,
or a replay divergence.

## What it records

A `CompileReceipt` is derived purely from a compiled `ContextPacket` (plus the
run's render context). Its shape:

- **Identity & provenance** — `packet_id`, `run_id`, `trace_id` (the pointer back
  to the run trace), `compiler_version`, `policy_profile`, and an
  `input_fingerprint` (the packet's content hash over the compile inputs).
- **Budget** — `max_input_tokens`, `used_tokens`, and the per-block allocation.
- **Included** — one entry per kept item: its id, kind, citation locator,
  `source_hash`, and the scoring signals that drove selection (`score`,
  `relevance`, `authority`, `freshness`) — never its text.
- **Excluded** — one entry per dropped item: its id, kind, `source_hash`, the
  exclusion `reason` (`low_relevance`, `duplicate`, `conflict_lower_authority`,
  `conflict_stale`, `budget_exceeded`, `memory_budget_exceeded`,
  `privacy_scope_mismatch`, …), and `superseded_by` when another item won.
- **Conflicts** — the resolved conflicts as `winner` / `loser` / `rule`, plus any
  unresolved conflict where both items were kept and the discrepancy was reported
  to the model.
- **Privacy** — the privacy scope, whether PII redaction is on, the count of PII
  spans redacted and of items dropped for a scope mismatch, and the constant
  guarantee `omitted_raw_text: true`.
- **Render** — the provider and model the packet rendered to, the `context_ir_hash`
  (the provider-neutral IR) and the `rendered_packet_hash` (the exact bytes sent
  to the model).

## Deterministic, verifiable, and diffable

The receipt exposes a stable `receipt_hash` over the compile *decision* — the
inputs, the inclusions and exclusions with their scores, the budget, the privacy
posture, the conflicts, and the render identity — excluding the per-run ids.
Two properties follow:

- **Determinism.** Recompiling identical inputs yields the same `receipt_hash`, so
  a replay of the same inputs is provably unchanged.
- **Explicit divergence.** A changed source yields a different `receipt_hash`, and
  `receipt.diverges_from(baseline)` returns a structured delta — which items were
  added or removed, which scores moved, the budget delta, whether the render
  identity changed — so a surprising run can be diffed against a known-good one.

`receipt.verify()` re-derives the receipt from its own serialized bytes and checks
its invariants (it round-trips to the same hash, the budget is not overspent, the
included and excluded id sets are disjoint, and raw text is omitted), so a receipt
attached to a review or an incident is self-checking.

## Getting a receipt

Every run links a receipt from its trace. After a run it is on the result:

```python
result = app.run("What is the refund window?")
receipt = result.metadata["compile_receipt"]          # a JSON-safe dict
```

The same receipt is set on the run's `prompt_render` span (so it travels with the
trace) and can be printed from the CLI:

```bash
vincio trace receipt <trace_id>     # print the receipt for a stored run's trace
```

From a compiled context or a packet directly:

```python
from vincio.context import CompileReceipt

compiled = await app.context_compiler.compile(...)
receipt = compiled.receipt()                          # a CompileReceipt
print(receipt.receipt_hash)
export = receipt.to_export()                           # JSON-safe, no raw text
```

Because it is derived from the packet and text-light by construction, a compile
receipt is the natural thing to attach to a PR that changes retrieval, scoring, or
budgeting — the reviewer sees exactly what moved into and out of the context, and
why, without ever seeing the underlying documents.

## How it works

A receipt is a pure projection, not a second run. `CompileReceipt.from_packet`
walks an already-compiled `ContextPacket` (plus the render context) and copies out
the *decisions*: the kept items with their scoring signals, the dropped items with
their exclusion reasons, the resolved and unresolved conflicts, the budget lines,
and the privacy posture. Nothing is recomputed and no model is called — so a
receipt cannot diverge from the packet it describes. `receipt_hash` then hashes
that decision (deliberately excluding the per-run `run_id` / `trace_id`), which is
what makes two runs of the same inputs compare equal while a changed source
compares different. `diverges_from(baseline)` returns `None` when the decisions
match and a structured delta otherwise.

## Best practice

- **Baseline in the PR, diff in the incident.** Store a known-good receipt as the
  reviewed baseline; when a production run surprises you, run `diverges_from`
  against it to get the item- and score-level delta instead of eyeballing traces.
- **Ship the receipt, not the packet.** The receipt is safe to paste into a
  ticket or attach to a review precisely because `omitted_raw_text` is a
  structural guarantee — reach for the full packet only when you have the
  authority to see the underlying documents.

## Gotchas

- **`receipt_hash` intentionally ignores the run ids.** Two identical-input runs
  share a `receipt_hash` even though their `run_id` / `trace_id` differ — that's
  the determinism property, not a collision. Compare `input_fingerprint` /
  `rendered_packet_hash` when you need per-run identity.
- **A stable `receipt_hash` proves the *decision* is unchanged, not that the
  bytes are.** A render-only change (provider swap, format) can move
  `rendered_packet_hash` while the compile decision — and therefore the receipt
  hash — holds; the delta from `diverges_from` flags the render-identity change
  explicitly.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: Attach a compile receipt to a PR or incident](../guides/compile-receipt.md)
- [Example: 17_compile_receipt.py](../../examples/17_compile_receipt.py)
- [Concept: Context packets & long-horizon governance](context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
