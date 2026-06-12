# Changelog

All notable changes to Vincio are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-06-12

Retrieval & RAG superiority — the 0.3 roadmap milestone. Every advanced
retrieval technique behind one `Index` interface, fused in one weighted RRF,
budgeted and cited inside the compiled packet, and measured by CI-gated
benchmarks.

### Added

- **Learned sparse retrieval** — `SparseIndex`, an inverted impact index
  scored by SPLADE-style dot products, behind the same `Index` protocol as
  BM25/dense so it fuses in the existing weighted-RRF merge. Encoders:
  `LocalImpactEncoder` (offline, deterministic: sublinear tf + morphological
  stem expansion) and `CallableSparseEncoder` (adapter for served SPLADE /
  uniCOIL / ELSER models).
- **Late-interaction retrieval** — `LateInteractionIndex` with ColBERT-style
  per-token MaxSim scoring over any `Embedder` (offline hash embedder by
  default, ColBERT checkpoints behind the same protocol). `compressed=True`
  enables PLAID-style two-stage search: deterministic k-means centroid
  codes, candidate generation over inverted centroid lists, exact rerank of
  survivors. Token-vocabulary vector caching keeps indexing cheap.
- **Advanced indexing** — new chunking strategies: `sentence_window` (score
  the sentence, cite the ±2-sentence window — the engine swaps the window in
  at evidence time), `hierarchical`/`parent_document` (small children linked
  to large parents), and `contextual` (situating prefix per chunk).
  `AutoMergingIndex` wraps any index and merges sibling child hits back into
  their parent; `contextualize_chunks()` writes LLM chunk prefixes
  (contextual retrieval) with a heuristic offline fallback.
- **Query understanding** — `QueryUnderstanding` strategies: HyDE
  (hypothetical answer passage as a search probe), multi-query expansion,
  decomposition for multi-hop, and step-back prompting. LLM-backed with
  deterministic offline fallbacks; expansions are recorded on the
  `QueryPlan`, fused with per-strategy RRF weights, and surfaced in
  retrieval metadata/traces. Configure per engine
  (`RetrievalEngine(query_strategies=[...])`), per call
  (`retrieve(strategies=[...])`), or app-wide
  (`retrieval.query_strategies`).
- **GraphRAG** — `detect_communities` (deterministic label propagation over
  the entity graph), `Community` hierarchy (communities of communities),
  extractive community summaries with an LLM hook, and `GraphRAG` retrieval
  with global vs local routing: entity questions walk graph paths,
  corpus-level questions retrieve community summaries that carry provenance
  to their member chunks.
- **Incremental & live indexes** — `LiveIndex` wraps any index with upsert
  semantics, per-entry TTLs, lazy `purge_expired()`, and `indexed_at`
  freshness stamps; the retrieval engine surfaces `indexed_at` and
  `age_days` in evidence metadata. `VectorIndex.migrate(new_embedder)`
  re-embeds in place — an embedding-model migration without re-chunking or
  rebuilding.
- **Connector hub** — new `vincio.connectors` package: `web`, `github`,
  `sql` (SQLite built in, any DB-API connection), `s3` (`vincio[s3]`),
  `gcs` (`vincio[gcs]`), `notion`, `confluence`, and `slack` connectors,
  all returning provenance-tracked `Document`s; a `connect()` factory and
  `register_connector()` plugin point; `app.add_source(connector=...)`
  loads, chunks, and indexes in one call. REST connectors accept injected
  httpx clients (offline-testable); cloud connectors accept injected
  boto3/GCS clients.
- **App retrieval modes** — `add_source(retrieval=...)` now also accepts
  `sparse`, `late_interaction`, and `hybrid_full` (BM25 + dense + sparse +
  late interaction in one fusion).
- **VincioBench** — the `rag` family now compares every retrieval mode
  (bm25, dense, sparse, late_interaction, late_interaction_plaid, hybrid,
  hybrid_full, hybrid_full + query understanding) on recall@3/MRR and
  exercises GraphRAG community building; new `budgets.json` gates hold each
  mode at recall@3 ≥ 0.8 and verify GraphRAG produces communities and
  global evidence.
- **Docs & examples** — rewritten retrieval concepts page, a new
  connectors guide (`docs/guides/connectors.md`), a new RAGatouille/ColBERT
  comparison (`docs/comparisons/ragatouille.md`), an updated LlamaIndex
  comparison, and `examples/12_advanced_rag.py` (sparse + late-interaction
  fusion, query understanding, auto-merging, GraphRAG routing, live-index
  TTL, SQL connector → full app — offline).

### Changed

- `QueryPlan` gains `expansions`; `RetrievalResult.metadata` reports the
  strategies used; evidence from sentence-window chunks carries the window
  text plus a `matched_sentence` marker.
- 277 tests passing offline (~2s); ruff clean.

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
