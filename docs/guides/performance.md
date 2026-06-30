# Performance & streaming

Vincio's spine is built for speed: concurrent hot paths, content-addressed
compilation caches, zero-copy context packets, end-to-end streaming, and
throughput primitives, all measured by VincioBench and gated in CI. This
guide shows how to use each.

## Streaming end to end

`ContextApp.astream` runs the full 17-step pipeline (policy, retrieval,
context compile, validation, nothing is skipped) and streams as it goes:

```python
import asyncio
from vincio import ContextApp

app = ContextApp(name="docs_qa")
app.add_source("docs", path="./docs", retrieval="hybrid")

async def main():
    async for event in app.astream("How do I configure SSO?"):
        if event.type == "stage":          # pipeline progress
            print(f"[{event.stage}]", event.data)
        elif event.type == "text_delta":   # real provider token deltas
            print(event.text, end="", flush=True)
        elif event.type == "partial_output":
            partial = event.partial_output  # best-effort parse so far
        elif event.type == "tool_result":
            print(f"\n→ {event.tool_name}: {event.tool_result.status}")
        elif event.type == "done":
            result = event.result           # full RunResult, validated

asyncio.run(main())
```

Event types: `stage`, `text_delta`, `partial_output` (incremental
partial-JSON parse when the app has a structured output contract),
`tool_call` / `tool_result`, `usage`, `error`, and a terminal `done`
carrying the final `RunResult`.

Notes:

- The model span records `ttft_ms` (time to first token), so streaming
  latency is a number in the trace, not a feeling.
- The server's `POST /v1/apps/{id}/stream` endpoint emits these same events
  over SSE, real deltas as the provider produces them.
- `app.stream(...)` is a synchronous convenience that collects the async
  event stream.
- `parse_partial_json` never invents content: a partial parse is the
  balanced prefix of what the model has produced so far.

## Concurrency & cancellation

The hot paths are concurrent by default:

- Memory recall, file ingestion, and retrieval run in parallel per run.
- Retrieval fans out every (query × index) search pair concurrently.
- Tool calls within one model round execute concurrently.
- Eval cases already ran concurrently.

Every fan-out is bounded and cancellation-correct: cancelling `arun`/`astream`
cancels all in-flight subtasks, and `Budget.max_latency_ms` is enforced as a
hard deadline on the whole run (the run fails with a budget error instead of
hanging).

```yaml
# vincio.yaml
performance:
  max_concurrency: 8        # retrieval/memory/ingest fan-out bound
  tool_parallelism: 4       # concurrent tool calls per model round
```

For your own fan-outs, use the same primitives:

```python
from vincio.core.concurrency import gather_bounded, map_bounded

results = await gather_bounded((fetch(u) for u in urls), limit=8)
```

`gather_bounded` preserves order, bounds in-flight tasks, and cancels the
group on first failure.

## Incremental & cached compilation

Three content-addressed caches make unchanged inputs free. Keys cover every
input that affects the output, so they are on by default:

| Cache | Stage | Key |
|---|---|---|
| `ChunkCache` | document chunking | content hash + strategy + sizes |
| `PromptCompileCache` | prompt compile | spec hash + task + context blocks + options |
| `ContextCompileCache` | context compile | every compile input (evidence/memory texts, budget, policies, options) |

```yaml
cache:
  prompt_compile_cache: true   # default
  chunk_cache: true            # default
  context_compile_cache: true  # default
```

Embedding caching is content-addressed too (`CachedEmbedder` keys by text
hash) and accepts a persistent backend so vectors survive restarts:

```python
from vincio.caching import SQLiteCache
from vincio.retrieval.embeddings import CachedEmbedder, ProviderEmbedder

embedder = CachedEmbedder(ProviderEmbedder(provider), backend=SQLiteCache(".vincio/emb.db"))
```

All caches invalidate through the existing tag-based invalidation manager
(document updated → chunk/retrieval/context entries; prompt changed →
prompt-compile entries; and so on).

### Partial recompile on packet edits

Edit a compiled packet without paying for a full recompile, selection,
budgeting, and ordering re-run over the retained inputs, and unchanged texts
hit the memoized scorers:

```python
compiled = await app.context_compiler.compile(...)
edited = await app.context_compiler.recompile(
    compiled,
    add_evidence=[new_item],
    remove_evidence_ids=["doc_old:C3"],
)
```

### Compiled-prompt render program

A prompt's stable prefix, role, objective, rules, safety policies,
definitions, the output contract, and examples, depends only on the spec, not
on the per-call task or evidence. The prompt compiler compiles that prefix once
into a render program and reuses it across calls that share the spec, rendering
only the volatile suffix:

```yaml
# vincio.yaml, on by default
# (CompilerOptions.use_render_program)
```

The output is byte-identical to compiling from scratch; the program is a
hot-path accelerator, not a behaviour change. On a representative spec the warm
prefix reuse is several times faster than re-rendering it every call;
`compiler.program_hits` counts the reuses. This is distinct from the
`PromptCompileCache`, which only hits when *every* input (task and context
blocks included) is unchanged.

### Warm candidate arena

Collecting and normalizing candidates, building a candidate for every
evidence, memory, and tool item, collapsing whitespace, and screening privacy
scope, is query-independent. In the common session pattern (the same retrieved
corpus, a new turn each time) the compiler reuses that prepared set instead of
rebuilding it, so a warm recompile is dominated by the query-dependent scoring
and selection:

```yaml
performance:
  reuse_candidate_set: true   # default
```

The reuse is correctness-preserving: the cached state is query-independent, and
each compile works from a fresh copy, so the shared compiler stays safe under
concurrent runs. `compiler.arena_hits` counts the reuses.

### Single-pass feature arena

The dedup, conflict, and selection passes each need a candidate's stemmed terms,
word shingles, and similarity-blocking tokens. Deriving those once per compile —
instead of pass after pass through the bounded global cache, which thrashes once
a candidate pool is larger than it can hold — keeps the constant factor of the
O(n²) passes paid exactly once, however large the pool grows:

```yaml
performance:
  single_pass_selection: true   # default
```

A per-compile *feature arena* memoizes each candidate's features in an unbounded
dict that is discarded when the compile finishes, so a 10k-candidate pool that
overruns the global cache pays each derivation once. The semantic path caches
each embedding's norm once and reuses it across every pairwise cosine, the
normalization pass counts tokens in one batch, and BM25 bounds its top-k with a
partial selection instead of a full sort. Every one of these is
**selection-preserving**: the features are byte-identical to the per-pass
derivation, so the same context is selected with the flag on or off — held by a
PerfBench `selection_byte_identical` gate. The arena is a fresh per-compile
object, so the shared compiler stays safe under concurrent runs.

The win is held honest: rather than a loose latency ceiling, a PerfBench
`compile_speedup` **ratio floor** gates that the single-pass path stays
measurably faster than the per-pass derivation on a large pool, so an erased win
fails the build. Measure the before/after yourself:

```bash
python benchmarks/profile_stages.py --compare   # single-pass off vs on, with speedup
```

## Vectorized scoring

Candidate scoring runs in a single pass. The per-component scores for the whole
candidate set are reduced against the weight vector together, and, when NumPy
is installed, semantic relevance and the weighted reduction each collapse to a
single matrix product over the candidate set the feature arena prepared. The
pure-Python fallback is the zero-dependency default and produces bit-for-bit the
same selection, so NumPy is an optional accelerator and never a requirement:

```bash
pip install numpy   # optional: accelerates large semantic candidate sets
```

Nothing to configure, the batched scorer is always on, and a build without
NumPy compiles identical context to one with it.

## Streaming-first compilation

`ContextCompiler.compile_streaming` streams the compiled context as it becomes
available. The stable prefix, objective, instructions, constraints, and the
task, is always included and independent of which evidence is selected, so it
is emitted *before* any candidate is scored; the selected evidence follows, then
a terminal event carries the full `CompiledContext`:

```python
from vincio.context import CompileStreamEvent

async for event in app.context_compiler.compile_streaming(
    objective=app.objective,
    user_input=UserInput(text="How do I configure SSO?"),
    evidence=evidence,
):
    if event.type == "prefix":
        begin_prompt(event.text)        # available before scoring runs
    elif event.type == "evidence":
        attach_evidence(event.evidence)
    elif event.type == "done":
        compiled = event.result         # authoritative, equals compile()
```

Back-pressure is the async generator itself: scoring and rendering advance only
as the consumer pulls. The terminal `done` event is authoritative and identical
to `compile()` for the same inputs.

`compile_streaming` and `recompile` are **advanced, deep-import APIs on
`ContextCompiler`** (`app.context_compiler`), not `app.*` verbs: a full run
already streams end-to-end through `app.stream`, so these are for callers driving
the context compiler directly — assembling a packet for a custom transport, or
editing a compiled packet between turns. `CompileStreamEvent` is exported from
`vincio.context` for typing the stream. See
[`examples/11_advanced_context.py`](../../examples/11_advanced_context.py) (the
compile-hot-path section) for both in one offline run.

## Speculative retrieval prefetch

Task classification is cheap and finishes before retrieval runs. With prefetch
enabled, the query embedding is computed speculatively from the classification
while memory recall and ingestion proceed, so retrieval's query embed lands as a
cache hit instead of a fresh call:

```yaml
performance:
  speculative_prefetch: true   # opt-in
```

The prefetch shares the app's embedder, so a warm that lands is a cache hit and
a warm that does not costs only an embed retrieval would have done anyway. It is
cancelled cleanly once preparation finishes, and a failed warm never affects the
run.

## Per-app memory-footprint budget

Declare a resident-memory ceiling for the compiled context packet. When the
selected context would exceed it, the compiler slims the packet (text moves to a
hash reference) and then evicts the lowest-utility evidence until the estimate
fits, recording each eviction in the excluded report:

```yaml
performance:
  memory_budget_mb: 8   # resident-memory ceiling for the compiled packet
```

The estimated footprint is surfaced on every run as `result.memory_bytes` and
rolled up as `peak_resident_bytes` in the cost summary, and a footprint
regression gate in VincioBench holds the reference packet under its ceiling.

## Long-horizon runs: the context governor

The per-packet footprint budget bounds *one* compile. Million-token, multi-day,
multi-session runs need more: across many turns, naïve accumulation rots quality
("context rot") and grows the footprint without bound. The `ContextGovernor`
holds a **context budget** across the whole run the way the cost report holds a
dollar budget:

```python
from vincio import ContextApp, ContextBudget

app = ContextApp(name="assistant")
app.use_context_governor(ContextBudget(max_tokens=8000, max_resident_bytes=2_000_000))

for turn in conversation:
    result = app.run(turn)
    app.govern_packet(result)          # admit this turn's evidence into the governor

report = app.context_budget_report()   # live tokens / residency / KV-cache footprint
```

Three mechanisms keep the live context bounded as the horizon grows:

- **Intra-run relevance decay** (`RelevanceDecay`) applies the memory
  subsystem's exponential decay *within a single run*, so a span admitted many
  steps ago loses weight before it crowds out fresh signal. Demotions are
  surfaced in the governor's excluded-context report.
- **Provenance-preserving compaction** (`ContextCompactor`) folds the coldest
  cold spans into a hierarchical extractive summary, writes the summary into the
  memory OS (audited), and pages the originals' full text to a content-addressed
  store, recoverable on demand via the same cross-process path slim packets
  use, so `governor.recall(query)` pages a compacted fact back when it is needed.
- **The budget itself** (`ContextBudget`) caps live tokens, resident bytes, and
  the estimated decode KV-cache footprint; the governor compacts (or evicts, when
  no compactor is configured) until the live footprint fits.

A horizon-scaling SLO holds the guarantee: at 10× horizon the governed footprint
stays within a bounded multiple of the 1× footprint (flat, not the ~linear growth
of naïve accumulation) and a compacted fact is still recalled, see the
`families.long_horizon` VincioBench family and example
[`54_long_horizon_context.py`](https://github.com/Ohswedd/vincio/blob/main/examples/11_advanced_context.py).

## Zero-copy context packets

Large packets no longer have to duplicate every evidence text:

```yaml
performance:
  slim_packets: true   # packets reference evidence text by content hash
```

- A slim packet's `evidence_items` carry `text_hash` references; the text
  lives exactly once, on the Context IR. `packet.evidence_text(id)`
  materializes lazily; `packet.materialize()` converts slim → full in place.
- `packet.iter_json()` streams the serialized packet chunk by chunk
  (`"".join(packet.iter_json())` equals a full dump), so persisting or
  shipping a large packet never builds the whole blob in memory.
- `packet.approx_size_bytes()` reports serialized size without building it.

## Throughput primitives

- **Pooled transport**: every HTTP provider uses a connection-pooled
  `httpx.AsyncClient` (`performance.max_connections`,
  `performance.max_keepalive_connections`), and the app reuses provider
  instances across runs so pools stay warm.
- **Request coalescing**: identical in-flight `generate` calls share one
  provider call (`performance.coalesce_requests: true` by default). Only
  byte-identical requests coalesce; each caller gets an independent copy.
- **Batch embedding**: `ProviderEmbedder` splits large inputs into bounded
  batches embedded concurrently; `BatchingEmbedder` micro-batches concurrent
  small calls into one provider round-trip:

```python
from vincio.retrieval.embeddings import BatchingEmbedder

embedder = BatchingEmbedder(inner_embedder, max_batch=64, window_ms=5)
```

## Measuring: benchmark gates & profiling

Every claim above is a number in VincioBench, and CI fails on regression:

```bash
python benchmarks/vinciobench.py perf   # latency/throughput/cache-speedup family
python benchmarks/check_budgets.py      # enforce budgets.json (exit 1 on breach)
python benchmarks/profile_stages.py     # per-stage breakdown from trace spans
python benchmarks/profile_stages.py --compare               # single-pass before/after
python benchmarks/profile_stages.py --cprofile vincio.prof  # flamegraph input
```

The per-stage profiler reads the same trace spans the runtime always emits,
so production traces give you the identical breakdown.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: Context packets & long-horizon governance](../concepts/context-packets.md)
- [Concept: Streaming & out-of-core](../concepts/streaming-and-out-of-core.md)
- [Concept: Observability](../concepts/observability.md)
- [Guide: Cookbook: task-shaped recipes](cookbook.md)
- [Guide: optimize prompts, context, and routing](optimize-context.md)
- [Guide: Analyze data](analyze-data.md)
- [Guide: Cost, reliability & scale](cost-and-reliability.md)
- [Example: 01_quickstart.py](../../examples/01_quickstart.py)
- [Example: 11_advanced_context.py](../../examples/11_advanced_context.py)
- [Example: 13_data_and_analytics.py](../../examples/13_data_and_analytics.py)
- [Example: 07_evaluation_observability.py](../../examples/07_evaluation_observability.py)
- [Concept: Prompt compiler](../concepts/prompt-compiler.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
