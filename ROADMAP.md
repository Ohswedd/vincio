<p align="center">
  <img src="assets/logo.svg" alt="Vincio" width="96">
</p>

# Vincio Roadmap

This is the public roadmap for the Vincio library — package `vincio`, CLI `vincio`, configuration
`vincio.yaml`, benchmark suite **VincioBench**. It records what ships today, what is planned next,
and what is intentionally out of scope.

**Legend:** ✅ shipped · 🚧 planned (next) · 🔭 exploring (later)

## What "done" means here

Vincio 0.1.0 was feature-complete for its scope: a single, coherent context-engineering library with
every subsystem implemented, tested offline, documented, and demonstrated by a runnable example.
Future work deepens and broadens the library — it does not change that scope. 0.2.0 made the spine
fast: streaming, concurrent, cached, and regression-gated. 0.3.0 made retrieval best-in-field:
learned sparse and late interaction fused with BM25/dense/graph, query understanding, hierarchical
and contextual indexing, GraphRAG, live indexes, and a connector hub. 0.4.0 made memory personal
and governed: scoped remember/recall, hybrid vector+graph recall, episodic→semantic consolidation
with provenance, audited forgetting, and a CI-gated memory eval harness. 0.5.0 made evaluation and
observability platform-grade in-process: quality/safety/conversational metrics, G-Eval judging with
calibration, a pytest plugin, red-teaming, synthetic data, experiments with significance, a prompt
registry, sessions and feedback on traces, OTel GenAI export, and a local trace viewer. 0.6.0 made
orchestration expressive and safe: multi-agent crews with roles, delegation, and a shared
blackboard; durable stateful graphs with checkpoint/resume/time-travel; first-class
human-in-the-loop; declarative composition with streaming node events; and runtime backends for
LangGraph and the OpenAI Agents SDK. 0.7 made reliability a guarantee: provider-native constrained
decoding with strict schema sanitization, streaming validation with early-abort, DSPy-style typed
signatures that feed the optimizer, programmable rails in the deterministic policy engine, bounded
self-correcting loops that never invent facts, and multi-schema routing — plus provider-transport
reliability fixes (event-loop-safe clients, rate-limit cooldowns honored from error bodies). 0.8
closed the loop: trace→dataset→eval→optimize→promote as one audited, reproducible cycle, grounded
auto-memory from runs, eval-driven retrieval feedback, cost/quality Pareto optimization, learned
context budgeting, and guided offline search strategies.

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

## ✅ Shipped — 0.2.0

Performance & core hardening — the full milestone as specified below, delivered:

- **Async-first hot paths** — concurrent memory/ingest/retrieval, (query × index) retrieval fan-out,
  concurrent tool rounds, bounded worker pools (`vincio.core.concurrency`), cancellation
  propagation, and `max_latency_ms` enforced as a hard deadline.
- **Incremental & cached compilation** — content-addressed prompt-compile / chunk / context-compile
  caches (on by default), content-addressed embedding cache with optional persistent backend, and
  `ContextCompiler.recompile()` for partial recompiles on packet edits.
- **Zero-copy Context Packet** — slim packets (evidence by content hash, lazy materialization) and
  chunked streaming serialization (`packet.iter_json()`).
- **Streaming end to end** — `ContextApp.astream` / server SSE with real token deltas, incremental
  partial-JSON output, and TTFT recorded on the model span.
- **Throughput primitives** — pooled provider transport with instance reuse, in-flight request
  coalescing, batched and micro-batched embedding.
- **Benchmark gates in CI** — the VincioBench `perf` family + `budgets.json` gates fail the build on
  regression; per-stage profiling via trace spans and cProfile flamegraph output.
- **229 tests passing offline in ~2s; ruff clean**; eleven runnable examples; performance guide.

See the [CHANGELOG](CHANGELOG.md) for the complete 0.2.0 notes.

---

## ✅ Shipped — 0.3.0

Retrieval & RAG superiority — the full milestone as specified below, delivered:

- **Late-interaction retrieval** — `LateInteractionIndex` (ColBERT-style per-token MaxSim) behind
  the existing `Index` interface, with PLAID-style centroid compression (candidate generation over
  inverted centroid lists + exact rerank) for scale.
- **Learned sparse retrieval** — `SparseIndex` over SPLADE-style impact vectors (offline
  `LocalImpactEncoder`, served models via `CallableSparseEncoder`), fused with dense and BM25 in the
  existing weighted-RRF merge; new app modes `sparse`, `late_interaction`, `hybrid_full`.
- **Advanced indexing** — `sentence_window`, `hierarchical`/`parent_document`, and `contextual`
  chunking strategies; `AutoMergingIndex` merges sibling hits into parents; `contextualize_chunks`
  writes LLM chunk prefixes (heuristic offline fallback).
- **Query understanding** — HyDE, multi-query expansion, decomposition, and step-back as planner
  strategies (LLM-backed with deterministic offline fallbacks), recorded on the plan/trace and fused
  with per-strategy weights.
- **GraphRAG** — deterministic label-propagation communities over the entity graph, hierarchical
  community summaries (extractive offline, LLM hook), and global vs local query routing.
- **Incremental & live indexes** — `LiveIndex` (upsert, TTL expiry, purge), `VectorIndex.migrate`
  re-embedding without rebuilds, and `indexed_at`/`age_days` freshness in evidence metadata.
- **Connector hub** — `vincio.connectors` with web, GitHub, SQL, S3, GCS, Notion, Confluence, and
  Slack connectors (plus `register_connector` for custom ones), wired into
  `app.add_source(connector=...)`; S3/GCS as optional extras.
- **277 tests passing offline in ~2s; ruff clean**; twelve runnable examples; VincioBench `rag`
  family compares every retrieval mode with CI-gated recall/MRR budgets.

See the [CHANGELOG](CHANGELOG.md) for the complete 0.3.0 notes.

---

## Where this goes next

0.1.0 made every subsystem real. The road to 1.0 makes each one **best-in-class on its own** *and*
**stronger because the others exist** — the thing no single-purpose library can do. The wager of the
whole project holds: the context compiler is the spine, and retrieval, memory, tools, agents, evals,
optimization, and observability are organs on it that share one data model (the Context Packet), one
event/trace stream, and one closed feedback loop.

Three principles govern everything below:

- **Beat the specialist at its own game.** For each competitor we name the capability that makes it
  worth using, then ship a Vincio feature that matches it and adds what the specialist structurally
  cannot — provenance, budgeting, eval-gating, or a shared trace.
- **Interconnect, don't bolt on.** Every new feature must read from and write to the same packet,
  ledger, and trace as the rest. A retriever improvement should be visible to evals; an eval result
  should be able to drive the optimizer; a confirmed fact should flow into memory.
- **Performance is a feature.** Determinism, caching, parallelism, and tight token budgets are how a
  unified system stays *faster* than a stack of glued-together specialist libraries, not slower.

Everything here is a library capability or an installable extra. Nothing below is a hosted service —
see [Out of scope](#out-of-scope).

### Competitive coverage map

| Competitor | What it's good at | Vincio answer (and the edge we add) | Milestone |
|---|---|---|---|
| **LangChain / LangGraph** | Orchestration breadth, integrations, stateful graphs | Declarative composition + durable stateful graphs with checkpoint/resume, *plus* one trace and eval loop across the whole graph | 0.6, 0.9 |
| **LlamaIndex** | Data connectors, advanced indexing, query engines | Hierarchical / auto-merging / GraphRAG retrieval + a connector hub, *plus* every retriever scored and budgeted by the context compiler | 0.3, 0.9 |
| **RAGatouille / ColBERT** | Late-interaction multi-vector retrieval | Native ColBERT-style and learned-sparse (SPLADE) retrieval behind the Index interface, *plus* fusion with BM25/dense/graph in one RRF | 0.3 |
| **Mem0** | User / long-term personalization memory | Personalization APIs over the existing layered memory, *plus* confidence, provenance, decay, conflict resolution, and utility scoring before inclusion | 0.4 |
| **LangSmith / Langfuse** | Tracing, prompt management, datasets, experiment tracking | Sessions, user feedback, scored spans, a prompt registry, and a local/exportable trace viewer, *plus* a provider-neutral model with no hosted dependency | 0.5 |
| **Ragas** | RAG metrics + synthetic test data | Faithfulness / context-precision-recall / answer-relevance metrics + synthetic dataset generation, *plus* results that gate releases and feed the optimizer | 0.5, 0.8 |
| **DeepEval** | Unit-test-style LLM assertions, red-teaming | A `pytest` plugin, assertion API, and adversarial/red-team suite, *plus* the same metrics reused at runtime as guardrails | 0.5 |
| **CrewAI** | Multi-agent teams and roles | Role/crew/delegation model with a shared blackboard, *plus* bounded budgets, termination guarantees, and per-agent traces & evals | 0.6 |
| **DSPy** | Typed signatures, automatic program optimization | Typed signatures and a pluggable optimizer interface, *plus* optimization extended across the whole context lifecycle, not just the prompt | 0.7, 0.8 |
| **Pydantic AI / Guardrails / NeMo** | Typed agents, output validation, programmable rails | Constrained/streaming structured output and rails-as-policies, *plus* repair that never invents facts and validation wired to the audit log | 0.7 |

---

## Roadmap

Milestones are ordered by dependency: we make the engine fast first, then deepen the data layers
(retrieval, memory), then the quality layers (eval, observability), then the orchestration and output
layers, then close the loop that ties them together, then broaden reach and stabilize.

### ✅ 0.2 — Performance & core hardening (shipped)

*The foundation everything else compounds on. A unified system only beats a pile of specialist
libraries if the spine is fast.*

- ✅ **Async-first hot paths** — memory recall, file ingestion, and retrieval run concurrently per
  run; retrieval fans out every (query × index) pair; tool calls within a model round execute
  concurrently. All fan-out goes through bounded, order-preserving worker pools
  (`vincio.core.concurrency`), and cancelling a run cancels every in-flight subtask;
  `Budget.max_latency_ms` is a hard deadline.
- ✅ **Incremental & cached compilation** — content-addressed caches (on by default) for
  prompt-compile, chunking, and context-compile; embedding caching is content-addressed with an
  optional persistent backend. `ContextCompiler.recompile()` re-runs selection over retained inputs
  for cheap packet edits, with memoized lexical scorers.
- ✅ **Zero-copy Context Packet** — `slim_packets` mode references evidence text by content hash with
  lazy materialization; `packet.iter_json()` streams serialization chunk by chunk so large packets
  never build the whole blob in memory.
- ✅ **Streaming end to end** — `ContextApp.astream` streams provider tokens through the full
  pipeline with incremental partial-JSON parsing, TTFT recorded on the model span, and the same
  events emitted over the server SSE path.
- ✅ **Throughput primitives** — batched + micro-batched embedding (`ProviderEmbedder`,
  `BatchingEmbedder`), in-flight request coalescing (`CoalescingProvider`), and a connection-pooled
  provider transport with instances reused across runs.
- ✅ **Benchmark gates in CI** — the VincioBench `perf` family measures compile/retrieval/run latency
  percentiles, cache speedups, throughput, and streaming TTFT; `benchmarks/budgets.json` budgets
  fail the build on regression; `benchmarks/profile_stages.py` gives per-stage breakdowns and
  cProfile flamegraph input.
- *Interconnection (held):* every optimization is measured by the same trace/cost model — cache hits,
  TTFT, and per-stage timings are span attributes, and "faster" is a number in the VincioBench
  report, gated in CI.

### ✅ 0.3 — Retrieval & RAG superiority (vs LlamaIndex, RAGatouille) (shipped)

*Make retrieval the best in the field while keeping it one scored, budgeted subsystem of the
compiler — not the center of gravity.*

- ✅ **Late-interaction retrieval** — ColBERT-style multi-vector indexing and MaxSim scoring behind
  the existing `Index` interface (`LateInteractionIndex`), with PLAID-style centroid compression
  (inverted centroid lists for candidate generation, exact rerank) for scale.
- ✅ **Learned sparse retrieval** — SPLADE-style impact-weighted sparse vectors (`SparseIndex`;
  offline `LocalImpactEncoder`, served models via `CallableSparseEncoder`) fused with dense and BM25
  in the existing weighted-RRF merge; `retrieval="hybrid_full"` fuses all four.
- ✅ **Advanced indexing** — hierarchical / auto-merging retrieval (`AutoMergingIndex`),
  sentence-window and parent-document retrieval, and "contextual retrieval" (LLM-written chunk
  prefixes via `contextualize_chunks`, heuristic prefixes offline) as chunking strategies.
- ✅ **Query understanding** — HyDE, multi-query expansion, query decomposition for multi-hop, and
  step-back prompting, all as planner strategies with deterministic offline fallbacks, recorded on
  the query plan and in traces.
- ✅ **GraphRAG** — deterministic label-propagation community detection and hierarchical community
  summaries over the entity graph; global vs local query routing (`GraphRAG.route`).
- ✅ **Incremental & live indexes** — `LiveIndex` upserts, deletes, TTL with purge, and
  `VectorIndex.migrate` re-embedding without full rebuilds; freshness (`indexed_at`, `age_days`)
  surfaced in evidence metadata.
- ✅ **Connector hub** — pluggable data connectors (web, S3/GCS, Notion, Confluence, Slack, GitHub,
  SQL) feeding the document engine via `app.add_source(connector=...)`; cloud-store extras
  (`vincio[s3]`, `vincio[gcs]`), custom connectors via `register_connector`.
- *Interconnection (held):* every chunk keeps full provenance into the evidence ledger; freshness and
  retrieval scores ride evidence metadata into context scoring; every mode is measured in the
  VincioBench `rag` family with CI-gated recall/MRR budgets (eval-scoring per retriever lands in 0.5,
  optimizer feedback in 0.8).
- *Edge over specialists (delivered):* RAGatouille gives you ColBERT; Vincio gives you ColBERT fused
  with sparse, dense, and graph, then **budgeted and cited** inside a compiled packet.

### ✅ 0.4 — Memory & personalization (vs Mem0) (shipped)

*Personalization without the failure mode of stale, ungrounded memories.*

- ✅ **Personalization APIs** — first-class user / agent / session memory scopes (new
  `MemoryScope.AGENT`) with `remember` / `recall` ergonomics over the existing L0–L5 layers, on
  both the engine and `ContextApp`; `ScopedMemory` handles (`memory.for_user("u1")`, `for_agent`,
  `for_session`, `for_tenant`) bind one owner, and scope/type are inferred when not stated.
- ✅ **Consolidation tiers** — automatic episodic→semantic summarization, dedup, and promotion with
  full provenance retained: `MemoryConsolidator` / `memory.consolidate(session_id)` promote session
  episodes to user/agent-scope semantic memories carrying `consolidated_from`, archive the episodes
  with `consolidated_into`, merge near-duplicates (`merged_from`), and
  `promote_aged_episodes()` runs the background tier transition.
- ✅ **Hybrid memory store** — vector + graph recall in one query: `asearch()` fuses lexical and
  vector relevance over any `Embedder` (offline hash embedder by default, content-addressed vector
  cache) with graph adjacency boosts for memories linked to the task's entities, with the memory
  graph as the relationship backbone.
- ✅ **Forgetting & hygiene** — tunable decay, per-scope TTLs applied on write (expired items never
  surface), importance-weighted retention (heavily used, confirmed, stable preferences survive
  longer), and explicit user-driven edit/delete/export/erase (GDPR-style) flowing through the
  hash-chained audit log as `memory_edit` / `memory_delete` / `memory_export` / `memory_erase`.
- ✅ **Memory eval harness** — `evaluate_memory` measures recall precision, recall@k, contradiction
  rate, staleness, and personalization lift; the VincioBench `memory` family runs it offline and
  eleven `budgets.json` gates hold the results in CI.
- *Interconnection (held):* cited evidence and successful tool results write back as candidate
  memories with provenance (`memory.write_back`), carrying a status penalty until confirmed; every
  memory is utility-scored against the task (objective + extracted entities) before it ever enters
  a packet.
- *Edge over specialists (held):* Mem0 stores memories; Vincio stores memories **with confidence,
  provenance, decay, and conflict resolution**, scored for relevance before inclusion — see
  [docs/comparisons/mem0.md](docs/comparisons/mem0.md).
- **301 tests passing offline in ~2s; ruff clean**; thirteen runnable examples; the VincioBench
  `memory` family holds recall precision, contradiction rate, staleness, and personalization lift
  under CI-gated budgets.

See the [CHANGELOG](CHANGELOG.md) for the complete 0.4.0 notes.

### ✅ 0.5 — Evaluation, testing & observability (vs Ragas, DeepEval, LangSmith, Langfuse) (shipped)

*Make evaluation and observability so good you stop reaching for an external platform — and keep them
provider-neutral and dependency-free.*

- ✅ **Metric library expansion** — `faithfulness`, `answer_relevance`, context precision/recall,
  `hallucination` (strict number checking catches numeric contradictions), `toxicity`, `bias`,
  `summarization_quality`, and conversational/session metrics (`knowledge_retention`,
  `conversation_relevance`) — all deterministic and offline; rubric-based **G-Eval** judging
  (`GEvalJudge`) with auto-derived evaluation steps, repeated-sample scoring, and
  `calibrate()` against human labels.
- ✅ **Testing ergonomics** — the `vincio.testing` package and a `pytest` plugin (auto-registered):
  `assert_eval` / `assert_grounded` / `assert_metric` / `assert_safe` with direction-aware
  thresholds, and snapshot tests for packets and traces (volatile fields normalized away;
  `pytest --vincio-update-snapshots` to refresh).
- ✅ **Red-teaming & robustness** — `RedTeamSuite` with 13 built-in probes (jailbreaks, injection,
  PII/secret-leak, bias, toxicity) judged deterministically via canary tokens and the security
  engine's detectors; reports attack success rate *and* input-side detector coverage; the injection
  detector gained persona/fake-authority signals (7/7 probe coverage, no new false positives).
- ✅ **Synthetic data generation** — `SyntheticGenerator` bootstraps golden sets from your corpora
  with difficulty mix (stated-fact / cloze / multi-hop), round-robin source coverage, and full
  provenance; deterministic offline templates with an LLM hook.
- ✅ **Experiment tracking** — `ExperimentTracker` on the existing metadata store: variant
  comparison (direction-aware best-per-metric), ablations vs a baseline, and
  `ab_test()` with paired/Welch t-tests and pure-Python p-values.
- ✅ **Prompt registry** — `PromptRegistry`: content-hash-keyed versions, moving tags, field-level
  and rendered diffs, rollback-as-new-head, and eval runs linked to the exact version measured;
  `vincio prompt push / versions / diff / rollback`.
- ✅ **Richer trace model** — sessions and threaded runs (`session_id` / `thread_id`), user feedback
  capture (`trace.add_feedback`, `vincio trace feedback`), scores attached to spans and traces by
  the runtime evaluators, and **OpenTelemetry GenAI semantic conventions** (`chat {model}`,
  `gen_ai.*` attributes, `gen_ai.conversation.id`).
- ✅ **Local trace viewer** — `vincio trace view` (TUI tree with scores and feedback),
  `vincio trace export [--session]` (one self-contained static HTML file — no server, no account),
  and `vincio trace diff --html` (visual side-by-side diff).
- *Interconnection (held):* metrics defined here are the *same objects* used as runtime evaluators
  today and as guardrails (0.7) / fitness terms (0.8) next; traces become datasets with one command
  (`dataset_from_traces`, `vincio eval dataset --min-feedback`); red-team findings hardened the
  security engine's detectors.
- *Edge over specialists (delivered):* LangSmith/Langfuse are platforms you send data to; Vincio's
  evals and traces live **in your process, in the same model as the runtime**, and gate releases
  offline — see [docs/comparisons/langsmith-langfuse.md](docs/comparisons/langsmith-langfuse.md),
  [ragas.md](docs/comparisons/ragas.md), and [deepeval.md](docs/comparisons/deepeval.md).
- **367 tests passing offline in ~2s; ruff clean**; fourteen runnable examples; the VincioBench
  `evals` family holds metric agreement, red-team judging, synthetic determinism/coverage,
  significance, sessions, viewer self-containment, and G-Eval calibration under 13 CI-gated budgets.

See the [CHANGELOG](CHANGELOG.md) for the complete 0.5.0 notes.

### ✅ 0.6 — Agents & orchestration (vs LangChain/LangGraph, CrewAI, OpenAI Agents SDK) (shipped)

*Match the orchestration frameworks on expressiveness, beat them on safety and observability.*

- ✅ **Multi-agent teams** — `Crew` / `app.crew()`: named roles (`AgentRole` with description,
  goal, keywords, `budget_fraction`) bound to bounded executors over a shared, versioned,
  author-attributed `Blackboard` (JSON snapshot/restore, event-bus posts); sequential, parallel,
  and hierarchical processes — the manager delegates with a schema-validated LLM plan and a
  deterministic keyword-routing offline fallback, every delegation is recorded, and termination is
  guaranteed (scaled per-member budgets, a crew-level budget check before each delegation, and
  `max_rounds` on review).
- ✅ **Durable stateful graphs** — `StateGraph` / `app.graph()`: dict-state nodes, conditional
  edges, per-key reducers, optional Pydantic state schema; a `Checkpointer` persists every
  super-step on the existing metadata stores (memory/SQLite/Postgres), giving `resume(thread_id)`,
  `history()`, and `fork(checkpoint_id)` — deterministic re-execution from any step — with
  `max_steps` bounding cyclic graphs.
- ✅ **Human-in-the-loop** — static (`interrupt_before` / `interrupt_after`) and dynamic
  (`interrupt(state, payload)`) graph interrupts; resume with a value re-runs the paused node with
  the answer; `update_state()` edits state as a new checkpoint (edit-and-resume). Workflow approval
  gates with no `approval_fn` now pause (`status="paused"`, `pending_approvals`) and
  `workflow.resume(result, approvals={...})` continues without re-running done steps.
- ✅ **Declarative composition** — `compose()` and the `|` operator pipe functions, agents, crews,
  workflows, and compiled graphs with results normalized between steps; `parallel()` and
  `branch()` combinators; `astream()` yields `NodeEvent`s and every node emits a span.
- ✅ **Runtime backends** — `LangGraphBackend` (StateGraph → LangGraph builder; nodes transfer
  as-is, edges/conditional edges/entry/END translated) and `OpenAIAgentsBackend` (agents and crews
  → SDK `Agent`s; a crew becomes a manager with handoffs) with lazy imports and injectable modules
  for offline tests — Vincio orchestrates without lock-in.
- *Interconnection (held):* crews, graph nodes, and composed steps emit `crew` / `crew_agent` /
  `graph_node` / `compose_node` spans on the shared tracer; `CrewResult.metrics()` aggregates the
  same per-agent metrics the eval runner gates; `app.graph()` checkpoints persist in the same
  metadata store as runs and packets; crew members built by `app.crew()` read context through the
  compiler, so budgeting and guardrails apply automatically.
- *Edge over specialists (delivered):* CrewAI gives you a crew; Vincio gives you a crew that is
  **bounded, traced, eval-gated, and budget-aware** by construction — see
  [docs/comparisons/crewai.md](docs/comparisons/crewai.md) and
  [openai-agents-sdk.md](docs/comparisons/openai-agents-sdk.md).
- **426 tests passing offline in ~2s; ruff clean**; sixteen runnable examples; the VincioBench
  `agent` family holds crew termination, delegation recording, interrupt→resume and fork-replay
  determinism, and composition streaming coverage under six new CI-gated budgets.

See the [CHANGELOG](CHANGELOG.md) for the complete 0.6.0 notes.

### ✅ 0.7 — Structured output, guardrails & reliability (vs Pydantic AI, Guardrails, NeMo, DSPy) (shipped)

*Reliability as a guarantee, not a hope.*

- ✅ **Constrained generation** — provider-native grammar/JSON-schema-constrained decoding where
  available (OpenAI strict json_schema, Anthropic forced tool use, Gemini responseSchema), with the
  robust-parser fallback everywhere else. Schemas are strict-sanitized for constrained decoders
  (`to_strict_json_schema`: objects closed, all properties required, optionals nullable) while
  validation runs against the original schema; the negotiated decoding mode
  (`native` / `prompt`) is recorded on every trace. Grammar-style constraints
  (`choice_schema`, `regex_schema`) ride the same path, with `pattern` now enforced by the
  deterministic schema validator.
- ✅ **Streaming validation** — `StreamingValidator` parses balanced partial JSON as it streams and
  prefix-checks it against the schema: missing required fields are tolerated until the stream ends,
  definite mismatches (wrong type, unknown field on a closed object) surface mid-stream.
  `app.astream()` emits `valid_prefix` / `validation_errors` on every `partial_output` event so
  consumers can abort doomed generations early; `finalize()` applies allowed structural repair.
- ✅ **Typed signatures** — DSPy-style input→output signatures over the prompt AST: class-based
  (`Signature` with `InputField` / `OutputField`) and string form
  (`signature("question, context -> answer, confidence: float")`). `Predict` /
  `app.predictor(sig)` executes them with native constrained decoding and full output validation;
  `Signature.to_prompt_spec()` makes every signature a drop-in prompt-optimization target.
- ✅ **Rails as policies** — programmable input/output rails (topic, format, safety, custom
  predicates) expressed in the deterministic policy engine (`app.add_rail(...)`,
  `RailEngine`) and enforced before/after every generation; safety rails reuse the security
  engine's PII / secret / injection detectors, and `action="redact"` masks instead of blocking.
- ✅ **Self-correcting loops** — `SelfCorrector` / `app.enable_self_correction()`: bounded
  validate→critique→repair cycles with a deterministic critique built from the validation report, a
  hard `max_cost_usd` ceiling, and a structure-only repair contract — facts are never invented, and
  semantic/citation/policy validators re-run every cycle.
- ✅ **Multi-schema routing** — `SchemaRouter` / `app.add_output_schema(...)`: choose the output
  contract per run by task type, keywords, or predicate; content-side `classify` / `validate_any`
  validate heterogeneous outputs against the registered alternatives.
- ✅ **Provider reliability fixes (shipped with 0.7)** — HTTP provider clients are recreated when
  bound to a closed/stale event loop (sync usage across `asyncio.run` calls no longer raises
  "Event loop is closed"); 429 cooldowns are honored from provider error bodies (Gemini
  `RetryInfo.retryDelay` / "retry in Ns" messages) when no `Retry-After` header is set, with the
  retry backoff cap raised to 60s so free-tier RPM limits self-heal; Gemini GA model pricing
  (2.5 pro/flash/flash-lite, 2.0 flash/flash-lite) and the `gemini-embedding-001` default
  embedding model reflect the live API.
- *Interconnection (held):* every validation failure, repair, and correction cycle is a trace event
  on the `output_validation` span *and* an `output_validation` entry in the hash-chained audit log;
  rail violations are `PolicyViolation`s (`rail:<name>`) on the same trace/audit path as every other
  policy decision; rails reuse the security detectors; signatures feed the optimizer via
  `to_prompt_spec()`.
- *Edge over specialists (delivered):* Pydantic AI retries, Guardrails re-asks, NeMo scripts a
  dialog runtime — Vincio repairs **deterministically first, model-second, facts never**, with every
  decision audited — see [docs/comparisons/pydantic-ai.md](docs/comparisons/pydantic-ai.md),
  [guardrails.md](docs/comparisons/guardrails.md),
  [nemo-guardrails.md](docs/comparisons/nemo-guardrails.md), and the updated
  [dspy.md](docs/comparisons/dspy.md).
- **467 tests passing offline in ~2s; ruff clean**; seventeen runnable examples; the VincioBench
  `reliability` family holds strict-schema closure, mid-stream invalid detection (with abort
  savings), correction recovery rate, rail catch rate (zero false positives), signature validity,
  and routing accuracy under 13 CI-gated budgets.

See the [CHANGELOG](CHANGELOG.md) for the complete 0.7.0 notes.

### ✅ 0.8 — The closed-loop ecosystem (the differentiator) (shipped)

*This is the milestone no single-purpose library can ship, because it requires owning the whole
lifecycle.*

- ✅ **Trace → dataset → eval → optimize → promote** — one continuous loop, all in the library, all
  reproducible: `ImprovementLoop` / `app.improvement_loop()` / `vincio loop run` captures the
  traces production runs already write, curates them with `dataset_from_traces`
  (feedback-filtered, fingerprinted for reproducibility), evaluates the current prompt as the
  baseline, runs the gated prompt optimizer (candidate evaluations are memory-write-free so they
  never pollute recall state), and promotes the winner — pushed to the `PromptRegistry`, tagged,
  eval-linked, applied to the live app, written to the hash-chained audit log
  (`loop_promotion`), announced on the event bus (`loop.promoted`), and logged (baseline and
  winner) to the `ExperimentTracker`; `--dry-run` reports the decision without acting.
- ✅ **Auto-memory from runs** — `memory.write_back: [facts]`: verifiable output claims that the
  cited evidence supports (`extract_grounded_facts`, deterministic, support-thresholded) become
  *candidate* memories through the existing guarded write policy, carrying measured support and
  evidence provenance (`origin: run_fact`) and a status penalty in recall until confirmed.
- ✅ **Retrieval feedback** — `RetrievalFeedback` tunes per-index RRF fusion weights and the
  heuristic reranker's blend from eval relevance labels (`records_from_report` /
  `records_from_dataset`), deterministically and gated: weights only change when recall@k + MRR
  measurably improve; `recommend_chunking` picks the chunking config whose eval report scored
  best.
- ✅ **Cost/quality Pareto optimization** — `pareto_loop` / `ParetoFrontier`: candidates are kept
  as a non-dominated accuracy/groundedness/latency/cost frontier with knee-point selection,
  per-objective constraints (`{"cost": 0.01}`), and `prefer=` overrides; promotion still passes
  the same safety rules as the scalar loop.
- ✅ **Learned context budgeting** — `BudgetLearner` searches bounded perturbations of the
  per-task allocation tables and adopts a learned table only through gated promotion;
  `LearnedAllocations` persists as JSON and installs via `app.use_learned_budgets()` /
  `BudgetAllocator(learned=...)`, with fixed tables as the fallback.
- ✅ **Context-aware offline optimization** — guided search strategies for the evolution loop
  (`hill_climb` single-knob mutation of the incumbent, `anneal` with Metropolis acceptance and a
  cooling schedule), deterministic under seeds, hard-bounded by the evaluation budget, pluggable
  into `ContextOptimizer(strategy=...)` and exposed as `guided_search`; pre-scored candidates
  flow into the evolution loop without re-screening.
- *Interconnection (held):* the loop reuses the tracer's exporter, the eval runner, the registry,
  and the tracker — no new stores; promotions are audit-log entries and event-bus events; grounded
  facts ride the same guarded memory pipeline and provenance metadata as every other write;
  retrieval tuning mutates the live engine only through measured, gated improvement.
- *Edge over the field (delivered):* each competitor optimizes one organ; Vincio optimizes the
  **organism**, with every signal flowing through one packet, ledger, and trace — see the updated
  [docs/comparisons/dspy.md](docs/comparisons/dspy.md) and
  [ragas.md](docs/comparisons/ragas.md), and the new guide
  [docs/guides/close-the-loop.md](docs/guides/close-the-loop.md).
- **495 tests passing offline in ~2s; ruff clean**; eighteen runnable examples; the VincioBench
  `loop` family holds promotion (fires, deterministic, gate-blocked, registry-tagged,
  eval-linked), auto-memory grounding, retrieval-feedback gating, Pareto frontier correctness,
  learned-budget promotion, and guided-search bounds under 14 CI-gated budgets (81 total).

See the [CHANGELOG](CHANGELOG.md) for the complete 0.8.0 notes.

### 🔭 0.9 — Integrations, connectors & developer experience (vs LangChain ecosystem breadth)

*Win on coverage and ergonomics so real projects adopt Vincio without rewriting their stack.*

- **Provider & embedding breadth** — more LLM, embedding, reranker, and vector-store adapters behind
  the existing interfaces; an OpenAI-compatible passthrough for any endpoint.
- **Framework interop** — import/export LangChain and LlamaIndex tools, retrievers, and loaders so
  existing assets work inside Vincio (and vice versa).
- **Scaffolding & templates** — `vincio init` templates for RAG, agent, and eval projects; typed
  `vincio.yaml` schema with validation and editor completion.
- **Notebook & TUI ergonomics** — rich reprs for packets/traces/evals; an interactive TUI for runs,
  traces, and memory inspection.
- **Domain packs** — opt-in prompt/schema/eval bundles for support, engineering, finance, and legal,
  shipped as extras you choose to install.
- **Migration guides** — "coming from LangChain / LlamaIndex / Ragas / Mem0" guides mapping concepts
  one-to-one to Vincio.

### 🔭 1.0 — Stabilization & guarantees

*Earn production trust.*

- **API stability** — semantic-versioning guarantees on the public surface; deprecation policy.
- **Performance SLOs** — published latency/throughput/token-efficiency targets enforced by
  VincioBench gates.
- **Security hardening** — a full security review of the tool sandbox, injection defense, and access
  control; supply-chain attestations on releases.
- **VincioBench at large** — expanded corpora, baselines against each competitor, and a transparent,
  reproducible methodology (run it yourself; no hosted leaderboard).
- **Docs completeness** — a guide and tested example for every subsystem and every public API.

## Out of scope

Vincio is a library, and stays one. The building blocks for running it in production — a
hash-chained audit log, retention policies, tenant isolation, RBAC / ABAC, and a server — ship in
the package so you can deploy them on your own infrastructure. **Hosted services, managed control
planes, dashboards-as-a-service, and compliance programs are not part of this project.** Vincio
gives you the engine; how and where you run it is yours.
