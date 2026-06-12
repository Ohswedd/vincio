<p align="center">
  <img src="assets/logo.svg" alt="Vincio" width="96">
</p>

# Vincio Roadmap

This is the public roadmap for the Vincio library — package `vincio`, CLI `vincio`, configuration
`vincio.yaml`, benchmark suite **VincioBench**. It records what ships today, what is planned next,
and what is intentionally out of scope.

**Legend:** ✅ shipped · 🚧 planned (next) · 🔭 exploring (later)

## What "done" means here

Vincio 0.1.0 is feature-complete for its scope: a single, coherent context-engineering library with
every subsystem implemented, tested offline, documented, and demonstrated by a runnable example.
Future work deepens and broadens the library — it does not change that scope.

---

## ✅ Shipped — 0.1.0

### Foundation

- Repository scaffolding (`pyproject`, Apache-2.0 license, packaged layout)
- Core data contracts — Objective, UserInput, Budget / BudgetUsage, EvidenceItem, MemoryItem,
  ToolSpec / ToolResult, PolicySet, Document / Chunk, Message / ModelRequest / ModelResponse /
  ModelCapabilities, RunConfig / RunResult — all Pydantic v2
- Unified error hierarchy rooted at `VincioError`
- Event bus with wildcard subscriptions
- Config loading: `vincio.yaml` discovery + `VINCIO_*` environment overrides + deep merge
- Token counting: calibrated offline heuristic with optional `tiktoken`

### Subsystems

- **Prompt engine** — `PromptSpec` with typed `${variables}`, a prompt AST, compiler passes
  (normalize, dedupe, conflict check, cache-aware stable-prefix layout, example selection, schema
  render, budget validation, hashing), Markdown / XML / JSON / minimal renderers, lint rules
  PROMPT001–009, spec and render hashes, diffing, and variant generation
- **Context compiler** — the full pipeline (collect → normalize → classify → score → dedupe →
  conflict → compress → budget → order → render → validate), utility scoring across all signal
  terms, near-duplicate detection, authority / freshness conflict resolution, extractive
  compression, evidence ledger, adaptive per-task budget allocation, boundary-sandwich ordering,
  and an excluded-context report
- **Input engine** — Unicode / whitespace normalization, offline language detection, task
  classification, file / media classification, PII / secret pre-scan, injection detection, trust
  tagging, scope resolution, and ambiguity detection
- **Document engine** — loaders for text / Markdown / HTML / CSV-TSV / JSON / YAML / code / email /
  PDF / DOCX / XLSX, section and table extraction, CSV schema inference and quality checks,
  spreadsheet formulas and sheets, code symbol extraction with repository import graphs, OCR, and an
  image-to-evidence multimodal pipeline
- **Retrieval engine** — chunkers (fixed / recursive / semantic / heading / table / code /
  adaptive) with provenance, offline and provider embeddings with caching, pure-Python BM25 and
  vector indexes with metadata filtering, weighted RRF hybrid merge, heuristic and LLM query
  planners, rerankers (heuristic / recency / authority / LLM / cross-encoder hook), entity-graph
  retrieval with path queries, multi-hop, and reasoning retrieval with fact-coverage reports
- **Memory engine** — L0–L5 layers, a guarded write pipeline (extract → classify → privacy →
  stability → contradiction → confidence → provenance), a decay formula, contradiction supersede /
  conflict flagging, restatement-as-confirmation, retrieval scoring, scope / tenant isolation,
  lifecycle transitions, a memory graph, and session summarizers
- **Tool engine** — a registry with decorator-based schema derivation from type hints, a permission
  model (RBAC scopes, ABAC rules, tenant boundary, sensitivity scan), a full lifecycle
  (validate → permission → approve → execute → validate → sanitize → trace), reliability scoring,
  scoped read-tool caching, write guardrails with idempotency keys and approval callbacks, and a
  subprocess sandbox
- **Agent engine** — AgentState / AgentStep, an acyclic step DAG with parallel levels, planners
  (direct / static / dynamic-LLM / ReAct / plan-and-execute), a bounded executor with full
  termination conditions, critic / validator steps, human gates, metrics, and a handoff router
- **Workflow engine** — DAG execution with parallel levels, retries with backoff, timeouts,
  conditional branching, compensation in reverse order, approval gates, typed parameter binding, and
  trace spans
- **Output engine** — output schemas (Pydantic / JSON-schema), output contracts with validator
  specs and a repair policy, robust parsers (fenced / embedded / lenient JSON, partial-JSON
  streaming, citations, front-matter), a validation pipeline, and principled structure-only repair
  that never invents facts
- **Evaluation engine** — JSONL datasets with rubrics / tags / difficulty / filter / sample / split,
  17+ registered metrics across task / grounding / operational / retrieval categories, judges
  (deterministic / model with repeated-sample calibration / embedding / hybrid), a concurrent
  runner, regression gates with aggregates, and reports with summaries, distributions, failures, and
  baseline diffs
- **Optimization engine** — a fitness function, an evolution loop
  (baseline → candidates → subset → top-N → full eval → gated promotion), safety rules (schema /
  safety-regression block, cost budget, minimum dataset coverage), prompt optimizer, context
  optimizer, routing policy with offline threshold optimization and ε-greedy / UCB1 bandits, and
  cache-layout tuning with advisory findings
- **Observability engine** — a trace / span hierarchy with contextvar nesting, JSONL / in-memory /
  console / multi exporters, an OpenTelemetry exporter, cost tracking with price tables, and trace
  show / replay / diff tooling
- **Caching** — LRU+TTL in-memory and SQLite backends with tag invalidation; response, retrieval,
  context-packet, and eval caches; a semantic cache with strict policy-scope + schema + freshness
  matching; event-bus-wired invalidation triggers; and a Redis backend
- **Security engine** — PII detectors (email / phone / names / addresses / government IDs / cards
  with Luhn / IBAN / health / API keys / secrets / IPs) with redaction, a secret scanner (patterns +
  entropy + key-name heuristics) and `SecretString`, prompt-injection defense (trust tags, heuristic
  signal detection, untrusted wrappers, classifier hook), RBAC / ABAC / tenant isolation / document
  permissions, a deterministic policy engine, a hash-chained audit log, and retention policies
- **Storage** — metadata stores (in-memory / SQLite / Postgres), a file blob store, DuckDB
  analytics, Qdrant and pgvector vector indexes, a Neo4j graph store, a Redis cache, and a URL-based
  factory
- **ContextApp runtime** — the full input-to-output flow with a public API
  (`configure` / `add_source` / `add_memory` / `add_tool` / `add_evaluator` / `add_validator` /
  `add_optimizer` / `set_policy` / `run` / `arun` / `agent` / `workflow` / `evaluate` / `task`),
  bounded tool loops, per-run file ingestion, run and packet persistence, and audit integration
- **Server mode** — a FastAPI `create_app` with run / stream / evals / runs / traces / indexes /
  memory endpoints, API-key and JWT (HS256) auth with tenant-scoped tokens, and SSE streaming
- **CLI** — `init`, `run`, `eval run` / `report` (gates + baseline compare, CI exit codes),
  `prompt lint` / `compile`, `trace show` / `replay` / `diff`, `optimize run`, `index build`,
  `memory inspect`
- **Plugin architecture** — registries for providers, metrics, chunkers, rerankers, judges,
  validators, tools, extractors, distillers, and classifiers; every extension point accepts a custom
  implementation

### Quality & release

- Unit tests across every subsystem, plus integration tests for ingest → retrieve → answer,
  tool → context → answer, memory → answer, agent pipelines, eval runner → report → baseline diff,
  trace replay, server endpoints, and end-to-end CLI
- Golden datasets in `tests/golden/` (document QA, support triage, extraction)
- A deterministic mock provider that generates schema-valid structured output
- **195 tests passing offline in ~1.5s; ruff clean**
- Documentation: getting started, six concept guides, five how-to guides, API / CLI / config
  reference, four comparison write-ups, `llms.txt`, and `AGENTS.md`
- Ten runnable, offline-capable examples
- **VincioBench**: eight benchmark families with naive baselines; improvement hypotheses are
  measured, not assumed

### Release checklist

- [x] `pip install vincio` works (editable install and wheel build verified)
- [x] Apps run against OpenAI and Anthropic (adapters implemented and payload-tested; offline via mock)
- [x] Prompt compiler supports Markdown / XML (plus JSON and minimal)
- [x] Context compiler scores and budgets context
- [x] RAG pipeline works on local documents
- [x] Pydantic structured-output validation works
- [x] Every run produces a trace
- [x] Eval runner supports JSONL datasets
- [x] CLI supports init / run / eval / trace (plus prompt / optimize / index / memory)
- [x] Documentation includes 10 full examples
- [x] CI tests pass (195 / 195 offline)
- [x] License chosen (Apache 2.0)
- [x] Public roadmap published (this file)

---

## 🚧 Next

Near-term work that deepens the existing library:

- Broader provider and embedding-model coverage behind the existing interfaces
- Additional document loaders and richer multimodal extraction
- More built-in evaluation metrics and reranker strategies
- Expanded VincioBench corpora and reporting
- Performance and memory-footprint tuning across the compiler hot paths

## 🔭 Later

Directions under exploration, all as optional, installable parts of the library:

- **Domain packs** — opt-in prompt / schema / eval bundles for common verticals (support,
  engineering, finance, legal), shipped as extras you choose to install
- **Synthetic dataset generation** — tooling to bootstrap golden eval sets from your own corpora
- **Context-aware offline optimization** — richer search strategies for the evolution loop
- Additional storage and vector-index adapters as the ecosystem grows

## Out of scope

Vincio is a library, and stays one. The building blocks for running it in production — a
hash-chained audit log, retention policies, tenant isolation, RBAC / ABAC, and a server — ship in
the package so you can deploy them on your own infrastructure. **Hosted services, managed control
planes, dashboards-as-a-service, and compliance programs are not part of this project.** Vincio
gives you the engine; how and where you run it is yours.
