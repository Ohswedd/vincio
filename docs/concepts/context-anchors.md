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

Each fit step is recorded on the item's own metadata (`pinned_fit`), and the
reserved cost surfaces as a dedicated `anchor` line in the
[compile receipt](compile-receipt.md)'s budget summary — a fitted frame is *on*
the packet, so it is never in the excluded set — so the frame's presence and
cost stay auditable.

## How it works

**Building the brief** (`build_anchor_brief`) is a deterministic, constraint-first
distillation, not a summarizer:

```text
split every anchor doc into whole sentences
  → classify: normative (must/never/always/required/should/avoid/…) vs prose
  → global priority queue: ALL normative first (doc order), then ALL prose
  → greedy fill under body_budget, skipping exact + prefix/superset dupes
  → regroup picks by document → render "### <title>" blocks under the frame header
  → trim whole trailing sentences until it fits brief_tokens
  → id = sha256(rendered text)
```

The priority queue is **global across documents**, so a tiny rules file's
constraints outrank a verbose README's prose — a small normative doc is never
starved by a large narrative one. `body_budget` reserves the frame header and
~6 tokens per document up front; an oversize sentence is *skipped* rather than
admitted (so it can't block every later constraint and then be trimmed away),
and if every sentence alone exceeds the budget the best one is carried as a
`_verified_truncate` whole-word cut — never nothing, never mid-word. The id
hashes the **rendered output**, so two environments whose tokenizers disagree
get different ids and `verify()` never false-matches a brief it did not produce;
`_BRIEF_ALGO_VERSION` bumps invalidate stale cached briefs on upgrade.

**Caching** lives in `AnchorSet`: the brief rebuilds only when the corpus hash
(`_corpus_hash` over algorithm version + `brief_tokens` + each document's
title/header/text, length-framed) changes, so re-adding identical documents is
free. `erase_source` calls `AnchorSet.remove`, so an erased anchor stops
injecting its frame — the anchor plane is inside the erasure sweep, not a leak
past it.

**Reserving the window** (`ContextCompiler._reserve_pinned`) fits the frame
into `cap = int(max_input_tokens × pinned_budget_fraction)` (default half),
taken off the *top* before the flexible evidence block is scored:

```text
total pinned ≤ cap        → ride uncompressed
total pinned > cap        → per-item share = floor + proportional extra
                            (Σ shares ≤ cap by construction, order-independent)
item over its share       → compress (if ≥24 tok) → hard-cut, re-verified
                            against the live token counter  (pinned_fit)
more pinned items than cap → ContextCompileError (observable, never silent)
```

The fitted frame is then **prepended before the retrieved detail** and kept out
of the competitive MMR pool entirely, which is *why* the relevance gate, the
min-score flush, dedup, conflict resolution, and eviction can never touch it.

## Best practice

- Anchor the documents that must hold *for the whole task* — a PRD, a spec, a
  brand/voice guide, coding standards — not per-question reference material.
- Size `brief_tokens` to the *normative* content, not the corpus: the frame is
  constraints, and detail is still retrievable on demand (Tier 2). A few hundred
  tokens is usually right; the default is 400.
- Write the source with real normative verbs (**must / never / always /
  required**). Those lines are what the brief prefers to keep — a rule buried in
  narrative prose may be trimmed before a crisp `MUST` elsewhere.
- Inspect what actually shipped with `app.task_brief()` before a long run.

## Gotchas

- **`pinned_budget_fraction` is a ceiling, not a target.** More pinned items
  than the cap in tokens is unsatisfiable and raises `ContextCompileError` —
  raise the budget or unpin, rather than expecting a silent drop.
- **The frame is distilled, not verbatim.** A constraint that never uses a
  normative verb can lose to one that does under a tight `brief_tokens`; widen
  the budget or reword the source.
- **Detail is not suppressed by the frame.** An anchor's own chunks are exempt
  from being deduplicated against its digest, so a retrieved section arrives
  *alongside* the frame — but it still competes for the flexible budget like any
  evidence.
- **Changing the corpus changes the id.** The `anchor:<hash>` id (and therefore
  the compile cache and receipt hash) is stable across a call chain and moves
  only when the anchor documents change — which is the signal that surfaces a
  frame update in a diff.

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
