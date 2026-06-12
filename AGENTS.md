# AGENTS.md — working on the Vincio codebase

## What this is

Vincio (`vincio/`) is a context engineering platform: it compiles prompts,
memory, retrieval, tools, schemas, and policies into budgeted, validated,
traced context packets. Build status and the roadmap live in `ROADMAP.md`.

## Layout

```
vincio/core         types, errors, events, config, tokens, concurrency, ContextApp, 17-step runtime (sync + streaming)
vincio/prompts      PromptSpec, AST, compiler (cache-aware), lint, variants
vincio/context      ContextIR/Packet, scoring, budgeting, compression, compiler
vincio/input        normalization, language/task classification, routing
vincio/documents    loaders (md/html/csv/pdf/docx/xlsx/eml/code), parsers, OCR, multimodal
vincio/retrieval    chunkers, embeddings, BM25/vector/sparse/late-interaction indexes, hybrid RRF, query understanding, rerankers, graph+GraphRAG, live indexes, reasoning
vincio/connectors   data connectors (web/github/sql/s3/gcs/notion/confluence/slack) feeding the document engine
vincio/memory       engine (L0–L5), write policy, decay, conflicts, graph, summarizers
vincio/tools        registry, permissioned runtime, sandbox
vincio/agents       bounded DAG executor, planners, ReAct, handoffs
vincio/workflows    deterministic DAG workflows (retries/compensation/approvals)
vincio/output       schemas, robust parsers, validation pipeline, principled repair
vincio/evals        datasets, metrics, judges, runner, gates, reports
vincio/optimize     fitness, evolution loop, prompt/context/routing/cache optimization
vincio/observability traces/spans, JSONL/OTel exporters, cost tracking
vincio/security     PII/secrets, injection defense, RBAC/ABAC, policy engine, audit
vincio/caching      LRU/SQLite backends, response/retrieval/packet/semantic + compile/chunk caches, invalidation
vincio/storage      metadata stores (memory/sqlite/postgres), qdrant/neo4j/redis/duckdb adapters
vincio/providers    openai/anthropic/google/mistral/local over pooled httpx + coalescing + deterministic mock
vincio/server       FastAPI app (API key + JWT auth, real-token SSE streaming)
vincio/cli          argparse CLI
```

## Commands

```bash
.venv/bin/pip install -e ".[dev]"     # setup
.venv/bin/python -m pytest tests/ -q  # full suite (offline, ~2s, must stay green)
.venv/bin/ruff check vincio/ tests/   # lint
```

## Rules

- **Offline-first tests**: everything must pass with no network/API keys —
  use `MockProvider` (it generates schema-valid structured output).
- **Optional dependencies import lazily** inside functions/constructors with
  a helpful `pip install "vincio[extra]"` error. Core deps are only
  pydantic, httpx, typing-extensions, pyyaml.
- **Every public data contract is a Pydantic v2 model**; engines are
  async-first with sync wrappers via `vincio.providers.base.run_sync`.
- **Security is deterministic** — never gate a security decision on model
  output. Policy/permission checks happen in code before execution.
- **Repair never touches facts** — output repair fixes structure only.
- **Every run must produce a trace**; spans nest via contextvars
  (`app.tracer.span(name, type=...)`).
- **Bound every fan-out** — concurrent work goes through
  `vincio.core.concurrency.gather_bounded` (order-preserving, cancellation-
  correct), never a bare `asyncio.gather` over unbounded inputs.
- **Performance is gated** — `python benchmarks/vinciobench.py` +
  `python benchmarks/check_budgets.py` must pass; budgets live in
  `benchmarks/budgets.json` and run in CI.
- Update `ROADMAP.md` when adding subsystems or changing release status.
