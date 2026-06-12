# Changelog

All notable changes to Vincio are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/Ohswedd/vincio/releases/tag/v0.1.0
