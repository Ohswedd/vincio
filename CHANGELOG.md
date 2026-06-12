# Changelog

All notable changes to Vincio are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-12

Performance & core hardening — the 0.2 roadmap milestone. The spine is now
fast, streaming, measured, and regression-gated.

### Added

- **End-to-end streaming** — `ContextApp.astream` (and sync `stream`) runs
  the full 17-step pipeline with real provider token streaming:
  `RunStreamEvent`s for pipeline stages, text deltas, incremental
  partial-JSON output (structure-only, never invents content), tool
  activity, and a terminal `done` with the validated `RunResult`. The model
  span records `ttft_ms`; the server `/stream` endpoint now emits real
  deltas over SSE instead of chunking the finished answer. `MockProvider`
  streams in genuine chunks so the path is exercised offline.
- **Async-first hot paths** — memory recall, file ingestion, and retrieval
  run concurrently per run; retrieval fans out every (query × index) pair;
  tool calls within a model round execute concurrently (bounded by
  `performance.tool_parallelism`). New `vincio.core.concurrency` module
  (`gather_bounded`, `map_bounded`): order-preserving, semaphore-bounded,
  first-failure-cancels-the-group fan-out.
- **Cancellation & deadlines** — cancelling `arun`/`astream` cancels every
  in-flight subtask; `Budget.max_latency_ms` is enforced as a hard deadline
  (the run fails with a budget error instead of hanging); cancelled runs
  persist with status `cancelled`.
- **Incremental & cached compilation** — content-addressed caches, on by
  default, keyed over every input that affects the output:
  `PromptCompileCache`, `ChunkCache` (keyed by document *content*, with
  provenance restored per requesting document), and `ContextCompileCache`.
  `ContextCompiler.recompile(previous, add_evidence=, remove_evidence_ids=, ...)`
  re-runs selection over retained inputs for cheap packet edits; the
  lexical scorers (`_terms`/`_shingles`) are memoized, removing the
  re-tokenization cost from the O(n²) dedupe/conflict passes. All caches
  invalidate through the existing tag-based invalidation manager.
- **Zero-copy Context Packet** — `slim_packets` mode references evidence
  text by content hash (text lives once, on the IR) with lazy
  materialization (`packet.evidence_text(id)`, `packet.materialize()`);
  `packet.iter_json()` streams serialization chunk by chunk;
  `packet.approx_size_bytes()` reports size without building the blob.
- **Throughput primitives** — connection-pooled provider transport
  (`httpx.Limits`, configurable pool sizes) with provider instances reused
  across runs; `CoalescingProvider` dedupes identical in-flight `generate`
  calls (on by default via `performance.coalesce_requests`);
  `ProviderEmbedder` splits large inputs into bounded concurrent batches;
  `BatchingEmbedder` micro-batches concurrent embed calls into one
  provider round-trip; `CachedEmbedder` is now thread-safe,
  content-addressed (SHA-256 keys), and accepts a persistent backend.
- **Benchmark gates in CI** — new VincioBench `perf` family (compile/
  retrieval/run latency percentiles, cache speedups, concurrent
  throughput, streaming TTFT); `benchmarks/budgets.json` +
  `benchmarks/check_budgets.py` fail the build on regression; new CI
  `bench` job uploads the report. `benchmarks/profile_stages.py` gives a
  per-stage breakdown from trace spans plus cProfile output for
  flamegraphs.
- **Config** — new `performance` section (`max_concurrency`,
  `tool_parallelism`, `embed_batch_size`, `embed_window_ms`,
  `coalesce_requests`, `max_connections`, `max_keepalive_connections`,
  `slim_packets`, `partial_parse_min_chars`) and new `cache` flags
  (`prompt_compile_cache`, `chunk_cache`, `context_compile_cache`).
- Docs: new [performance & streaming guide](docs/guides/performance.md);
  API/config references updated. New example
  `11_streaming_performance.py`. 34 new tests (229 total, offline).

### Fixed

- `PromptCompiler.compile` no longer temporarily mutates shared options to
  toggle schema rendering — it was a data race under concurrent compiles.

## [0.1.0] - 2026-06-12

Initial public release.

### Added

- **Prompt engine** — typed `PromptSpec`, AST, cache-aware compiler, linter, variant generation.
- **Context compiler** — candidate scoring, token budgeting, compression/distillation, evidence
  ledger, and excluded-candidate reports.
- **Engines** — input (normalization, classification, routing), documents (loaders, parsers, OCR,
  multimodal), retrieval (hybrid BM25 + vector RRF, rerankers, graph, reasoning), memory (layered
  store, decay, conflict resolution, graph), tools (permissioned runtime, sandbox), agents
  (bounded DAG, ReAct, handoffs), workflows (deterministic DAG), and output (schemas, robust
  parsers, validation, principled repair).
- **Evaluation** — datasets, metrics, judges, runner, regression gates, and reports.
- **Optimization** — gated prompt / context / routing / cache search.
- **Observability** — traces, spans, JSONL/OTel exporters, cost tracking.
- **Security** — PII and secret handling, prompt-injection defense, RBAC/ABAC access control,
  deterministic policy engine, and audit logging.
- **Caching** — response / retrieval / packet / semantic caches with invalidation.
- **Storage adapters** — SQLite, Postgres (pgvector), Qdrant, Neo4j, Redis, DuckDB.
- **Providers** — OpenAI, Anthropic, Google, Mistral, local, and a deterministic offline mock.
- **Surfaces** — FastAPI server (API key + JWT auth) and an argparse CLI.
- 195 offline tests, 10 runnable examples, documentation, and the VincioBench benchmark suite.

[0.2.0]: https://github.com/Ohswedd/vincio/releases/tag/v0.2.0
[0.1.0]: https://github.com/Ohswedd/vincio/releases/tag/v0.1.0
