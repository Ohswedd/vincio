# Context anchors — the always-on task frame

Not every document is "look it up when asked" evidence. Some are the *frame* of
the whole task. A vibe-coder building a CLI editor starts with a bulk of MD
files — a PRD, a brand guide, an architecture note, coding standards — that are
**100% needed for the global context of the task but not in full on every single
call**. Two common approaches both fail:

- **Stuff every MD file into every call** (the `CLAUDE.md` / paste-the-repo
  pattern): correct, but token-hungry and re-paid on every step — a 50-step agent
  run blows the window paying for the same corpus 50 times.
- **Pure per-query RAG**: cheap, but a step like "add a settings panel" does not
  lexically match "brand voice: warm and concise", so the constraint is not
  retrieved and the model silently violates the frame.

A **context anchor** solves this with two tiers.

## Tier 1 — the always-on frame (cheap, cached, guaranteed)

When you mark a source `anchor=True`, Vincio distills its documents **once** into
a compact `AnchorBrief`: a token-bounded (default 400) digest that **prefers the
normative lines** — *must / never / always / required* — because those are the
constraints a task has to respect, over narrative prose. The brief is:

- **deterministic** and **content-hash-cached** — built once, re-derivable
  offline, free to re-request across a 50-call chain;
- injected as **pinned** evidence into *every* run, at a flat few-hundred-token
  cost regardless of how large the anchor corpus is;
- **rendered first** in the packet, so the model reads the frame before the
  detail and the prompt prefix stays cacheable.

On the reference corpus a ~4,300-token PRD/brand/architecture bundle becomes a
~150-token frame — a **~28× reduction** — with every constraint retained.

## Tier 2 — on-demand detail (per query)

Anchor documents are still fully chunked and indexed like any other source, so a
call that needs a specific section retrieves it normally — and, because the
extractor exempts an anchor's own detail from being deduplicated against its
digest, the retrieved detail arrives *alongside* the frame, never suppressed by
it.

## The guarantee — pinned, reserved, never dropped

The frame is a **pinned** evidence item (`EvidenceItem.pinned=True`), and the
context compiler treats pinned evidence as guaranteed at **every** drop point:

- exempt from the relevance gate and the min-score flush;
- exempt from deduplication (on both sides) and conflict resolution;
- eviction-exempt under the resident-footprint ceiling;
- **budget-reserved off the *total* window** with a deterministic
  *compress → truncate* ladder — so injecting the frame never pushes the packet
  over `max_input_tokens` (a config that compiled before never crashes once a
  source becomes an anchor) and never starves the retrieved detail. The frame
  takes at most `pinned_budget_fraction` (default half) of the window.

Every ladder step and the reserved cost are recorded on the excluded-context
report and the [compile receipt](compile-receipt.md), so the frame's presence
and cost are auditable.

## Why this beats the competitors' shape

| | tokens/call | frame retained on a lexical miss |
|---|--:|:--|
| Stuff every MD file (Claude Code) | the whole corpus, every call | yes, but expensive and lost-in-the-middle |
| Pure per-query RAG (LangChain / ChatGPT) | top-k only | **no** — dropped when the query doesn't match |
| **Context anchors (Vincio)** | **flat ~few hundred** | **yes** — pinned and guaranteed |

Measured live (same model, three arms, on a rule the coding tasks never
mention), anchors match stuffing on adherence at ~3× fewer tokens per call while
pure RAG drops the rule — see `benchmarks/rag_anchor_uplift_live.py`.

## Using it

```python
app.add_source("spec", documents=[prd, brand, standards], anchor=True, brief_tokens=300)
print(app.task_brief())          # inspect the compact frame
report = app.run("add a settings panel")   # the frame is present, even here
```

Anchors compose with everything else — the docs stay in the normal retrieval
path (Tier 2), the frame rides the same governed, budgeted, cited context
pipeline as any evidence, and the whole thing is offline and deterministic.

See also: [Retrieval (RAG)](retrieval.md), [Context packets](context-packets.md),
[the compile receipt](compile-receipt.md).

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: build a RAG app](../guides/build-rag-app.md)
- [Example: 20_context_anchors.py](../../examples/20_context_anchors.py)
- [Concept: Context packets & long-horizon governance](context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
