# Performance & streaming

Vincio 0.2 made the spine fast: concurrent hot paths, content-addressed
compilation caches, zero-copy context packets, end-to-end streaming, and
throughput primitives — all measured by VincioBench and gated in CI. This
guide shows how to use each.

## Streaming end to end

`ContextApp.astream` runs the full 17-step pipeline (policy, retrieval,
context compile, validation — nothing is skipped) and streams as it goes:

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
  over SSE — real deltas as the provider produces them.
- `app.stream(...)` is a synchronous convenience that collects the async
  event stream.
- `parse_partial_json` never invents content: a partial parse is the
  balanced prefix of what the model has produced so far.

## Concurrency & cancellation

The hot paths are concurrent by default:

- Memory recall, file ingestion, and retrieval run in parallel per run.
- Retrieval fans out every (query × index) search pair concurrently.
- Tool calls within one model round execute concurrently.
- Eval cases already ran concurrently (0.1).

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

Edit a compiled packet without paying for a full recompile — selection,
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

- **Pooled transport** — every HTTP provider uses a connection-pooled
  `httpx.AsyncClient` (`performance.max_connections`,
  `performance.max_keepalive_connections`), and the app reuses provider
  instances across runs so pools stay warm.
- **Request coalescing** — identical in-flight `generate` calls share one
  provider call (`performance.coalesce_requests: true` by default). Only
  byte-identical requests coalesce; each caller gets an independent copy.
- **Batch embedding** — `ProviderEmbedder` splits large inputs into bounded
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
python benchmarks/profile_stages.py --cprofile vincio.prof  # flamegraph input
```

The per-stage profiler reads the same trace spans the runtime always emits,
so production traces give you the identical breakdown.
