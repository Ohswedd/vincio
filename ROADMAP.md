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
context budgeting, and guided offline search strategies. 0.9 won on breadth and ergonomics: an
OpenAI-compatible passthrough with hosted-gateway presets, hosted rerankers/embedders and Chroma/
Pinecone/LanceDB vector stores behind the existing interfaces, LangChain/LlamaIndex interop for
tools/retrievers/loaders/embeddings, `vincio init` templates with a typed `vincio.yaml` schema,
notebook reprs and an interactive TUI, opt-in domain packs, and migration guides — adopt Vincio
without rewriting your stack. **1.0 turns the library into a product you can trust in production:**
SemVer on a frozen public API with a mechanical deprecation policy, published performance/quality
SLOs gated by VincioBench, a documented threat model with offline audit-chain verification and
resource-limited tool sandboxing, supply-chain attestations (SBOM + SLSA provenance) on releases,
and a docs-completeness gate that runs every example and proves every subsystem is documented. **1.1
makes Vincio speak the ecosystem's interoperability protocols** — an MCP client *and* server, A2A
agent-to-agent delegation, and Anthropic Agent Skills, plus a unified reasoning control across
providers. **1.2 makes Vincio *score* what it runs** — trajectory, tool-use, multi-turn, and online
metrics that double as runtime guardrails and optimizer fitness, plus drift detection and Cohen's-κ
judge calibration. **1.3 makes Vincio *survive and account for* production traffic** — batch execution
at half cost, circuit breakers and health-aware failover, key pooling, runtime model cascades, cost
attribution by tenant/feature, enforced budget SLOs, and provider-aware prompt caching. **1.4 makes
Vincio *optimize itself and get cheaper*** — a reflective (GEPA-style) optimizer and MIPRO joint
proposal evolving a Pareto frontier from eval failures, a grounded-and-gated distillation flywheel that
turns production traces into a cheaper student in the routing cascade, faithfulness-gated learned
prompt compression, and reflective calibration of the optimizer's own judge. All additive behind
`@experimental` entry points on the frozen 1.0 API, in your process, never a hosted dependency.

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
| **LangChain / LangGraph** | Orchestration breadth, integrations, stateful graphs | Declarative composition + durable stateful graphs with checkpoint/resume and two-way tool/retriever/loader/embedding interop, *plus* one trace and eval loop across the whole graph | 0.6, 0.9 ✅ |
| **LlamaIndex** | Data connectors, advanced indexing, query engines | Hierarchical / auto-merging / GraphRAG retrieval, a connector hub, reader/retriever/embedding interop, and Chroma/Pinecone/LanceDB/Qdrant/pgvector behind one Index, *plus* every retriever scored and budgeted by the context compiler | 0.3, 0.9 ✅ |
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

### ✅ 0.9 — Integrations, connectors & developer experience (vs LangChain ecosystem breadth) (shipped)

*Win on coverage and ergonomics so real projects adopt Vincio without rewriting their stack — every
new adapter sits behind an interface that already existed, so breadth costs no new concepts.*

- ✅ **Provider & embedding breadth** — an OpenAI-compatible passthrough (`OpenAICompatibleProvider`
  / `openai_compatible(...)`) reaches *any* Chat-Completions endpoint, with named presets for the
  popular hosted gateways (`groq`, `together`, `fireworks`, `openrouter`, `deepseek`, `perplexity`,
  `xai`, `nvidia`) whose keys resolve from the conventional `<NAME>_API_KEY` env var. Hosted
  rerankers (`CohereReranker`, `JinaReranker`, `VoyageReranker`) and embedders (`JinaEmbedder`,
  `VoyageEmbedder`, `CohereEmbedder`) ride the core `httpx` dependency — no SDK — behind
  `build_reranker` / `build_embedder`; new vector-store adapters (Chroma, Pinecone, LanceDB) join
  Qdrant and pgvector behind the retrieval `Index` protocol via one `build_vector_index` factory.
- ✅ **Framework interop** (`vincio.interop`) — bring LangChain and LlamaIndex **tools, retrievers,
  loaders/readers, and embeddings** into Vincio, and hand Vincio's back. The `from_*` direction is
  duck-typed (it imports nothing heavy), so existing assets drop in without a new dependency;
  `add_langchain_tool` / `add_llamaindex_tool` register *and* enable a tool in one call; the `to_*`
  direction builds real framework objects (needs `vincio[langchain]` / `vincio[llamaindex]`).
- ✅ **Scaffolding & templates** — `vincio init --template {minimal,rag,agent,eval}` generates a
  tailored `ContextApp`, config, and golden set; every generated `vincio.yaml` carries a
  `# yaml-language-server: $schema=…` hint and ships a JSON Schema (`vincio config schema`, from the
  typed `VincioConfig`) for editor completion; `vincio config validate` / `vincio config show` check
  and print the effective merged config.
- ✅ **Notebook & TUI ergonomics** — `enable_rich_reprs()` gives `RunResult`, `Trace`, `EvalReport`,
  `MemoryItem`, and `SearchHit` HTML/Markdown reprs for Jupyter (pure render functions you can also
  call directly); `vincio tui` is a dependency-free, keyboard-driven inspector for runs, traces, and
  memory, with pure screen renderers and injectable IO so it is fully unit-tested.
- ✅ **Domain packs** (`vincio.packs`) — opt-in, dependency-free bundles for **support, engineering,
  finance, and legal**: a role/objective/rules prompt config, a structured output schema,
  recommended policies + evaluators, and a small golden eval set. `app.use_pack("support")` applies
  one through the public app API (so you can layer your own settings on top); `vincio packs
  list/show` and `register_pack(...)` round it out.
- ✅ **Migration guides** — "coming from LangChain / LlamaIndex / Ragas / Mem0" guides that map
  concepts one-to-one to Vincio, plus an integrations guide covering the new providers, vector
  stores, and interop adapters.
- *Already-shipped fixes (noted here for the record):* the provider-transport reliability work —
  event-loop-safe HTTP clients and 429 cooldowns honored from provider error bodies (Gemini
  `RetryInfo.retryDelay` / "retry in Ns") with the backoff cap raised to 60s — shipped with 0.7/0.8
  and is documented in the 0.8.0 [CHANGELOG](CHANGELOG.md) and the 0.7/0.8 notes above.
- *Interconnection (held):* every new provider, embedder, reranker, and vector store implements an
  interface the engine already speaks, so breadth changes nothing downstream — context compilation,
  budgeting, scoring, evals, traces, and security apply unchanged. Imported LangChain/LlamaIndex
  documents chunk, index, budget, and cite exactly like a local file; imported tools run through the
  same permissioned, sandboxed, audited runtime as native tools.
- *Edge over the field (delivered):* you adopt Vincio's compiler, evals, and closed loop **without
  rewriting your stack** — keep your LangChain tools and LlamaIndex readers, point at any
  OpenAI-compatible model, and pick the vector store you already run.
- **561 tests passing offline in ~2.5s; ruff clean; VincioBench 81/81 budgets**; twenty runnable
  examples. New 0.9 tests cover provider presets + key resolution, hosted reranker/embedder wire
  formats (httpx `MockTransport`), the vector-store factory and its helpful missing-dependency
  errors, both interop bridges (duck-typed fakes), pack loading/application/idempotent re-apply/run,
  the notebook reprs (including defensive formatting), the TUI loop (with memory-store caching), and
  every new CLI command.

See the [CHANGELOG](CHANGELOG.md) for the complete 0.9.0 notes, and the new
[migration guides](docs/guides/migrate-from-langchain.md) and
[integrations guide](docs/guides/integrations.md).

### ✅ 1.0 — Stabilization & guarantees (shipped)

*Earn production trust — make every guarantee mechanical, not aspirational.*

- ✅ **API stability** — Vincio now follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
  on a **frozen public surface** (`vincio.__all__`, returned by `vincio.stability.public_api()`, plus
  the documented subsystem entry points). A new `vincio.stability` module makes the deprecation
  policy mechanical: `@deprecated(since=, removed_in=, alternative=)` and `@experimental(since=)`
  emit `VincioDeprecationWarning` / `VincioExperimentalWarning` (escalatable to errors in CI),
  `deprecated_alias` keeps renamed symbols working for one major cycle, and `stability_of(obj)`
  introspects any symbol's contract. The contract: nothing public is removed in a minor/patch, only
  marked deprecated; removal waits for the next major. See the
  [stability policy](docs/reference/stability.md).
- ✅ **Performance SLOs** — a published [SLO table](docs/reference/slo.md)
  (`benchmarks/slos.json`) states latency/throughput/token-efficiency/quality/security targets, each
  naming the VincioBench budget that enforces it. The budgets are held **at least as strict** as the
  public promise, so a green build provably honors every SLO; `tests/test_slos.py` enforces that
  invariant. Reports now carry an `environment` block (version, Python, platform) for reproducibility.
- ✅ **Security hardening** — a documented [threat model](docs/security/threat-model.md) (STRIDE over
  the real controls). Concrete hardening: the hash-chained audit log gains **offline file
  verification** (`AuditLog.verify_file()` / `verify_audit_file()` / `vincio audit verify`) that
  detects post-restart tampering and pinpoints the broken line; the tool sandbox adds POSIX
  `setrlimit` **CPU / memory / file-descriptor limits** (best-effort, alongside the existing
  timeout, output caps, and scrubbed env). Releases ship a **CycloneDX SBOM** and **SLSA
  build-provenance attestations** (`.github/workflows/release.yml`).
- ✅ **VincioBench methodology** — a transparent, reproducible [methodology](benchmarks/METHODOLOGY.md):
  what each family measures, its named naive baseline, corpus provenance, the two-tier
  budgets-vs-SLOs design, and how to run it yourself. No hosted leaderboard — every number is
  reproducible offline from this repo.
- ✅ **Docs completeness** — a guide/reference and a **tested** example for every subsystem.
  `tests/test_examples.py` runs all 22 examples end-to-end offline (new `21_security_governance.py`
  covers the security subsystem); `tests/test_docs_completeness.py` asserts every public subsystem is
  documented and every example is indexed. The API reference adds the previously-undocumented
  `vincio.input`, `vincio.documents`, `vincio.cli`, and `vincio.stability` surfaces.
- *Already-shipped fix (noted here for the record):* `ContextApp.add_evaluator` registered a callable
  without a `__name__` (e.g. a `functools.partial`) under a key one greater than the one it recorded
  in `app.evaluators`, so later metric lookup missed it; the name is now resolved once. Shipped with
  0.9.0 and documented in the [CHANGELOG](CHANGELOG.md).
- **646 offline tests; ruff clean; VincioBench 81/81 budgets**; twenty-two runnable examples (the
  example smoke suite runs all of them end-to-end). New 1.0 tests cover the stability decorators and
  warnings, the frozen public surface, offline audit-chain tamper detection, sandbox resource limits
  and env scrubbing, the SLO↔budget invariant, every example running offline, and docs completeness.

See the [CHANGELOG](CHANGELOG.md) for the complete 1.0.0 notes, the
[stability policy](docs/reference/stability.md), the published [SLOs](docs/reference/slo.md), and the
[threat model](docs/security/threat-model.md).

---

## The road beyond 1.0

1.0 made Vincio trustworthy: a frozen public surface under SemVer, published SLOs gated by
VincioBench, a documented threat model, and a tested example for every subsystem. While that
stabilized, the field moved. Six shifts during 2025–2026 redefined what an AI/LLM library is expected
to do, and an audit of Vincio against them — and against LangChain/LangGraph, LlamaIndex, DSPy, Mem0,
Ragas, DeepEval, Pydantic AI, CrewAI, Haystack, and the serving layer (vLLM/SGLang/Outlines) — found
six concrete gaps:

1. **Interoperability protocols became table stakes.** Consuming **MCP** servers as tools is now
   universal (LangChain, DSPy, CrewAI, LlamaIndex, Pydantic AI, Haystack all ship it); *serving* an
   MCP endpoint, **A2A** agent-to-agent delegation (Google → Linux Foundation, 150+ orgs), and
   Anthropic's **Agent Skills** (`SKILL.md`, donated to the Agentic AI Foundation) are the new bar.
   Vincio had a first-class tool engine but spoke none of these protocols — **1.1 (shipped) closes
   this gap** with an MCP client *and* server, A2A, and Agent Skills.
2. **Evaluation moved from output to trajectory.** Tool-call accuracy/F1, goal accuracy, plan
   adherence, step efficiency, multi-turn simulation, *online* eval on sampled production traffic,
   and drift detection are now expected (Ragas, DeepEval, LangSmith, Phoenix). Vincio's 17+ metrics
   were output-and-grounding-shaped; it could trace a crew but not *score the trace* — **1.2 (shipped)
   closes this gap** with trajectory/tool-use/goal/plan metrics, a multi-turn simulator, online eval,
   drift detection, and Cohen's-κ annotation, every metric reusable as a guardrail and optimizer term.
3. **Cost and reliability at scale outgrew retry-and-cache.** Provider **Batch APIs** (a flat 50% cut)
   were absent; `FailoverChain` and `RetryingProvider` existed but there was no **circuit breaker**, no
   key/region load balancing, no **health-aware** routing; per-tenant/per-feature **cost attribution**
   and enforced **budget/cost SLOs** were not first-class. **1.3 (shipped) closes this gap** with batch
   execution, circuit breakers + health-aware failover, key pooling, runtime model cascades, cost
   attribution, and enforced budget SLOs — all in-process.
4. **Optimization got reflective, and the data flywheel got real.** **GEPA** (reflective genetic-
   Pareto, beating RL with ~35× fewer rollouts) and MIPROv2/SIMBA reset the optimizer bar beyond
   Vincio's evolution/anneal/hill-climb/bandit search; **distillation / fine-tuning data export**
   (teacher-trace → training set → cheaper student) is genuine whitespace across the whole field; and
   **learned prompt compression** (LLMLingua-class) goes beyond Vincio's extractive compression.
   **1.4 (shipped) closes this gap** with a reflective (GEPA-style) optimizer and MIPRO joint proposal
   over the whole context lifecycle, a grounded-and-gated distillation flywheel from production traces
   into cheaper inference, a faithfulness-gated learned compressor, and reflective calibration of the
   optimizer's own judge.
5. **Multimodal and embedding breadth advanced.** **Matryoshka** dimension truncation, **contextual**
   chunk embeddings (Voyage `context-3`), unified text+image embeddings (Cohere v4), and vector stores
   Vincio doesn't yet adapt (Weaviate, Milvus, Elasticsearch/OpenSearch, Vespa) are now standard.
   **1.5 (shipped) closes this gap** with MRL truncation and contextual/multimodal embedders behind the
   existing `build_embedder`, the five new stores behind `build_vector_index`, layout-aware extraction,
   and an opt-in voice/realtime module — every one behind an interface that already existed.
6. **Enterprise governance hardened into law.** The EU AI Act's GenAI transparency duties land
   **2 Aug 2026** (machine-readable synthetic-content marking); **model/system cards**, **OWASP LLM
   Top 10 (2025)** / **OWASP Agents** / **NIST AI RMF** / **MITRE ATLAS** mapping, **AI-BOM**, data
   lineage with right-to-erasure-by-source, data-residency-aware routing, and **multilingual** support
   (non-English PII, per-language eval slicing, the tokenizer "token tax") are what regulated buyers
   now require. Vincio has the audit/security spine but not the compliance evidence on top of it.

The three principles from the road to 1.0 still govern every item below — **beat the specialist at its
own game and add what it structurally cannot** (provenance, budgeting, eval-gating, one trace);
**interconnect, don't bolt on** (every new capability reads from and writes to the same Context Packet,
evidence ledger, audit log, and trace stream); and **performance is a feature**. Two more now join them:

- **Everything is additive.** 1.0 froze the public API under SemVer. Every 1.x feature below is new
  surface behind a new entry point or an opt-in extra — no public symbol is removed or repurposed, so
  upgrading across the entire 1.x line never breaks working code. Breaking changes are reserved for a
  future 2.0 (see [Exploring](#-exploring--later-and-20)).
- **Standards, in your process — never a hosted dependency.** MCP, A2A, Skills, OWASP/NIST/MITRE
  mappings, model cards, and OTel/OpenInference spans are all *implemented in the library and run on
  your infrastructure*. Vincio adopts the protocols and the compliance vocabulary; it does not become a
  service to do so. [Out of scope](#out-of-scope) is unchanged.

**Legend:** ✅ shipped · 🚧 planned (next) · 🔭 exploring (later). Milestones are ordered by dependency and urgency:
the protocol layer first (nothing else interoperates without it), then evaluation depth (so every
later change is measurable), then cost/reliability at scale, then optimization and the flywheel, then
multimodal/embedding breadth, then the governance layer that ties the audit spine to regulation.

### Post-1.0 competitive coverage map

| Competitor / standard | What it's good at (2025–26) | Vincio answer (and the edge we add) | Milestone |
|---|---|---|---|
| **MCP (Anthropic/OpenAI/Google)** | Universal tool/resource/prompt protocol | MCP **client** (servers as sources) + **server** (expose Vincio), *plus* every MCP tool runs through the same permissioned, sandboxed, audited, budgeted runtime as native tools | 1.1 ✅ |
| **A2A (Linux Foundation)** | Cross-vendor agent-to-agent delegation | A2A client/server + Agent Cards over the existing crew/graph model, *plus* bounded budgets, termination guarantees, and one trace across the delegation | 1.1 ✅ |
| **Anthropic Agent Skills** | Portable `SKILL.md` procedural knowledge | A Skills loader with progressive disclosure into the compiler, *plus* skills that are budgeted, cited, and eval-gated like any other context | 1.1 ✅ |
| **LangSmith / Ragas / DeepEval (agentic)** | Trajectory, tool-use, multi-turn, online eval | Trajectory/tool-use/goal/plan metrics over the spans Vincio already emits, online eval + drift, *plus* the same metrics reused as runtime guardrails and optimizer fitness | 1.2 ✅ |
| **OpenAI/Anthropic Batch APIs** | 50% async cost cut for offline work | A `BatchRunner` behind the provider interface for evals/extraction/synthetic data, *plus* the same call sites, cost-tracked and traced | 1.3 ✅ |
| **LiteLLM / gateways** | Failover, circuit breaking, key/region LB, cost attribution | Circuit breakers + health-aware routing on the existing `FailoverChain`, per-tenant/feature cost attribution + enforced budget SLOs, *plus* it lives in-process with your policies, not as a proxy hop | 1.3 ✅ |
| **DSPy 3 (GEPA / MIPROv2 / SIMBA)** | Reflective program optimization | A reflective optimizer over the whole context lifecycle (not just the prompt), *plus* gated promotion, Pareto cost/quality, and the closed loop already shipped | 1.4 ✅ |
| **DSPy BootstrapFinetune / distillation** | Teacher-trace → cheaper student | A distillation/fine-tune data flywheel from production traces, *plus* grounding, provenance, and eval-gating on every exported example | 1.4 ✅ |
| **LLMLingua** | Learned prompt compression | A learned compressor as a compiler pass alongside extractive compression, *plus* per-task budget integration and faithfulness gating | 1.4 ✅ |
| **Voyage / Cohere v4 / LlamaParse** | Matryoshka, contextual & multimodal embeddings, rich extraction | MRL truncation, contextual & multimodal embedders, and more vector stores behind the existing `Embedder`/`Index`, *plus* one scored, budgeted, cited packet | 1.5 ✅ |
| **DeepTeam / NeMo / governance** | OWASP/NIST/MITRE mapping, safety classifiers | Red-team + audit mapped to OWASP LLM 2025 / OWASP Agents / NIST AI RMF / MITRE ATLAS, model/system cards, AI-BOM, lineage, residency, multilingual — all from the existing audit/security spine | 1.6 🚧 |

---

### ✅ 1.1 — Protocols & interoperability (MCP, A2A, Skills) (shipped)

*Speak the protocols the ecosystem standardized on in 2025–26 — without becoming a service to do it.
A tool from an MCP server, an agent reached over A2A, and a `SKILL.md` all flow through the same
packet, ledger, permission model, and trace as everything Vincio already runs. The whole milestone is
**additive behind `@experimental` entry points** on the frozen 1.0 API, and uses only the core
`httpx` dependency — no SDKs.*

- ✅ **MCP client** — `vincio.mcp.MCPClient` and `app.add_mcp_server(...)` connect to MCP servers over
  **stdio**, **Streamable HTTP**, and an **in-process** transport (the offline-test path), negotiate
  capabilities, and surface `tools` / `resources` / `prompts`. MCP tools register through the
  *existing* tool registry (namespaced `<server>.<tool>`), so they inherit RBAC/ABAC scopes, the
  permission lifecycle, the subprocess/sandbox limits, idempotency keys, reliability scoring, and the
  audit log unchanged. MCP **resources** become first-class evidence with provenance
  (`origin: mcp:<server>`) that the compiler chunks, scores, budgets, and cites like any local
  document; MCP **prompts** import as `PromptSpec`s. Server-initiated **sampling** routes to the app's
  provider; **elicitation** routes to the human-gate callback. OAuth 2.1 seams (`pkce_pair`,
  bearer headers, PRM discovery URL) and the long-running **Tasks** primitive (poll/await) are
  supported. `vincio mcp add` / `mcp tools` inspect a server from the CLI.
- ✅ **MCP server** — `app.serve_mcp()` / `vincio mcp serve` expose a configured `ContextApp` as an MCP
  server over stdio (and any transport): registered tools become MCP tools (JSON Schema derived from
  the same type hints), evidence/sources become MCP resources, and the prompt spec becomes an MCP
  prompt — with the deterministic policy engine and audit log enforced on every inbound call
  (`mcp_serve`), and OAuth 2.1 resource-server token validation. One `ContextApp` is now both a
  consumer and a provider of context.
- ✅ **A2A (agent-to-agent)** — `vincio.a2a` serves an **Agent Card** at `/.well-known/agent.json`
  describing a crew's or graph's capabilities, and a JSON-RPC server/client implements the A2A
  **Task** lifecycle (`submitted → working → input-required → completed/failed`) with token auth and
  per-task audit. `app.serve_a2a(crew | graph | None)` exposes a crew, a durable graph (whose
  human-in-the-loop interrupts surface as `input-required` and resume by `taskId`), or the app itself
  with one call; `RemoteA2AAgent` makes a remote A2A agent reachable as a delegate inside a local
  crew. Delegation stays **bounded** (scaled budgets, termination guarantees) and **traced** end to
  end — the edge no raw A2A SDK gives you.
- ✅ **Agent Skills** — `vincio.skills` parses Anthropic-style `SKILL.md` (YAML frontmatter + Markdown,
  optional bundled scripts), and `app.add_skill(path)` injects skill instructions through the compiler
  with **progressive disclosure** (a one-line index always in budget, the body loaded only on
  relevance) so skills cost context only when used. Bundled scripts run as sandboxed, permissioned
  tools. Skills are scored, budgeted, and cited like any other context — not a privileged side
  channel.
- ✅ **Reasoning & Responses surface** — a unified reasoning control (`reasoning_effort` / thinking
  `budget_tokens`) across providers that expose it (OpenAI reasoning models, Anthropic extended
  thinking, Gemini thinking budget); the negotiated reasoning mode is recorded on the `prompt_render`
  span and `reasoning_tokens` on the `model_call` span, and an optional OpenAI **Responses API**
  adapter (`OpenAIResponsesProvider`: `previous_response_id` server-state, reasoning preserved across
  tool calls) rides the same `ModelProvider` interface, with Chat Completions kept as the portable
  default.
- *Interconnection (held):* MCP tools, A2A delegates, and Skills emit the same `tool` / `crew` /
  `crew_agent` / `model_call` spans on the shared tracer and write the same `tool_call` / `mcp_serve`
  / `a2a_serve` entries to the hash-chained audit log; MCP resources and skill bodies ride the
  evidence ledger with full provenance; protocol errors are ordinary `VincioError`s.
- *Edge over specialists (delivered):* others bolt MCP on as an adapter; in Vincio an MCP tool is
  **permissioned, sandboxed, budgeted, cited, and audited by construction**, and an A2A delegation is
  **bounded and traced** — guarantees the raw protocols and thin adapters do not provide. See the new
  guides [docs/guides/mcp.md](docs/guides/mcp.md), [a2a.md](docs/guides/a2a.md),
  [agent-skills.md](docs/guides/agent-skills.md), and [reasoning.md](docs/guides/reasoning.md).
- *Already-shipped fix (noted here for the record):* the Google/Gemini adapter recorded thinking
  tokens (`thoughtsTokenCount`) as `reasoning_tokens` but excluded them from the billable output
  (`candidatesTokenCount`), so thinking was costed at **$0** even though Gemini bills it at the output
  rate (`totalTokenCount` includes it). The adapter now folds thinking tokens into the billable
  output while keeping `reasoning_tokens` as the telemetry subset; OpenAI/Anthropic were already
  correct (reasoning is part of completion/output tokens). Shipped with 1.1 and documented in the
  [CHANGELOG](CHANGELOG.md).
- **698 tests passing offline; ruff clean; VincioBench 88/88 budgets**; twenty-five runnable examples.
  MCP client/server round-trip, A2A task lifecycle (incl. graph HITL resume), skill progressive
  disclosure, and the reasoning surface are covered offline with the in-process transport and
  `httpx.MockTransport`; four new examples (`22_mcp_tools_and_resources.py`, `23_a2a_delegation.py`,
  `24_agent_skills.py`, `25_reasoning_control.py`); the VincioBench `protocols` family gates MCP
  schema-fidelity, resource-provenance, A2A termination, and skill progressive-disclosure budget (with
  three new SLOs).

See the [CHANGELOG](CHANGELOG.md) for the complete 1.1.0 notes.

### ✅ 1.2 — Agentic evaluation & continuous quality (vs LangSmith, Ragas, DeepEval) (shipped)

*Vincio can run and trace a crew, a graph, and a tool loop — 1.2 makes it **score** them, over the
trajectory, over a multi-turn conversation, and over live traffic, reusing the same metric objects as
runtime guardrails and optimizer fitness, all in-process and dependency-free. Additive behind
`@experimental` entry points on the frozen 1.0 API.*

- ✅ **Trajectory & tool-use metrics** — seven new evaluators score *how* a run reached its answer:
  `tool_call_accuracy` / `tool_call_f1` (right tool, right args, in the right order),
  `goal_accuracy` (successful termination + answer match), `plan_adherence` (LCS vs the expected
  plan), `plan_quality` (failed/redundant steps, reference-free), `step_efficiency` (steps vs an
  optimal path), and `topic_adherence`. They read a provider-neutral `Trajectory` carried on the
  `RunOutput` — built with `RunOutput.from_agent_state(state)` / `from_crew_result(result)` /
  `from_trace(trace)` — so a crew, a `StateGraph` run, or a captured trace is scored **without
  re-instrumentation**. `EvalReport.metric_families()` shows final-output-only and trajectory
  evaluation side by side (a run can answer right while taking the wrong path — output-only eval can't
  see that, and the VincioBench family proves the gap).
- ✅ **Multi-turn & simulation** — a deterministic-offline **user simulator** (`Simulator`, LLM-backed
  with a seeded template fallback) drives multi-turn sessions from a `Persona` + goal; same seed →
  identical conversation. New conversational metrics `conversation_outcome` and `intent_resolution`
  join `knowledge_retention` / `conversation_relevance` to score the whole thread, and
  `dataset_from_traces(..., group_by_session=True)` stitches a session's traces into a multi-turn
  golden case.
- ✅ **Online / continuous eval** — `app.add_online_evaluator(metric, sample_rate=...)` scores a
  sampled fraction of live runs after the response is finalized (scheduled off the hot path; sampling
  bounds the overhead), writing each score as a time series on the existing metadata store
  (`OnlineEvaluator.series()`) — no traffic mirrored to any external service.
- ✅ **Drift detection** — `DriftMonitor` tracks rolling metric deltas (score drift) and
  **embedding-distribution drift** of inputs against the golden-set distribution, raising a
  `drift.detected` event on the bus and persisting baselines (`drift_baselines`) to the store when a
  baseline shifts; `vincio eval drift baseline.json current.json` reports it and exits non-zero.
- ✅ **Human-in-the-loop annotation** — a local `AnnotationQueue` records human labels next to
  LLM-judge scores and tracks **Cohen's κ**; `GEvalJudge.calibrate()` now also returns κ, and
  `judge.gating_weight(threshold)` / `queue.judge_trusted()` mean a judge only earns CI-gating weight
  once agreement clears the bar. `vincio eval annotate labels.jsonl` reports it.
- ✅ **Production A/B** — `app.experiment(name, variants=..., dataset=..., metrics=...)` evaluates
  prompt/model/config variants and compares eval metrics **and** cost per variant
  (`exp.compare()` / `exp.cost()` / `exp.significance(metric)`) with the paired/Welch significance
  tests `ExperimentTracker` already ships.
- *Interconnection (held):* every metric here is the same object usable as a runtime guardrail (0.7) —
  `app.add_metric_rail(metric, threshold=...)` / `metric_guardrail(...)` wrap a metric as a rail
  predicate — and as an optimizer/Pareto fitness term (0.8) via the new `AGENTIC_OBJECTIVES` preset
  (trajectory metrics are ordinary metrics, so they flow into `report.metric_values` and the frontier
  unchanged); online scores and drift baselines live in the same store as runs and packets.
- *Edge over specialists (delivered):* LangSmith/Ragas/DeepEval send your traces to a platform to
  score them; Vincio scores the **trajectory in your process, in the same model as the runtime**,
  gates releases offline, and turns the very same metric into a guardrail and an optimization target.
- *Already-shipped fix (noted here for the record):* the Google/Gemini cost table referenced a dead
  embedding model (`text-embedding-004`) while the provider defaulted to `gemini-embedding-001`, which
  was **absent from the table** — so a price lookup fell through to the zero default and embedding cost
  was tracked as **$0**. `gemini-embedding-001` is now priced ($0.15 / 1M input tokens), with a
  regression test. Documented in the [CHANGELOG](CHANGELOG.md).
- **740 tests passing offline; ruff clean; VincioBench 94/94 budgets**; twenty-six runnable examples.
  Trajectory metrics are validated against labeled agent traces in `tests/golden/agentic_eval.jsonl`;
  simulator determinism, online sampling, drift sensitivity/specificity, κ tracking, A/B significance,
  and the metric-as-guardrail path are covered offline; example `26_agentic_eval.py`; the VincioBench
  `agentic_evals` family gates trajectory-metric agreement, the output-only/trajectory gap, simulator
  determinism, drift sensitivity/specificity, and κ tracking (with six new SLOs).

See the [CHANGELOG](CHANGELOG.md) for the complete 1.2.0 notes.

### ✅ 1.3 — Cost, reliability & scale (FinOps + resilience) (shipped)

*What real teams hit when an LLM app meets production traffic: provider outages, rate limits, runaway
spend, and the need to attribute every dollar. Vincio already had failover, retries-with-cooldown, a
routing policy, prompt caching, and cost tracking — 1.3 turns those into a complete, enforced cost-and-
reliability layer that lives in your application, not in a proxy. Additive behind `@experimental`
entry points on the frozen 1.0 API, using only the core `httpx` dependency — no SDKs.*

- ✅ **Batch execution** — `vincio.providers.BatchRunner` / `app.batch([...])` / `app.abatch` and
  `vincio batch` submit request sets to the OpenAI **Batch API** and Anthropic **Message Batches API**
  (flat ~50% cost), poll job status, and reconcile results **by custom id** with partial-failure
  surfacing — missing ids become failed results, never silently dropped. `InProcessBatchBackend` is the
  offline/default path; `OpenAIBatchBackend` / `AnthropicBatchBackend` drive the real endpoints over the
  provider's own `httpx` client, reusing its payload-building and response-parsing so a batched call is
  byte-for-byte the sync one. Same `RunResult` contract, cost-tracked at the discounted rate and traced.
- ✅ **Circuit breakers & health-aware routing** — a `CircuitBreaker` wrapper tracks per-provider
  failure rate **and** latency over a rolling window, opens on threshold with half-open probing, and
  fast-fails (non-retryable `CircuitOpenError`) so `HealthAwareFailover` steers to healthy entries in
  microseconds; `KeyPool` round-robins health-aware across multiple API keys and regions with dual
  RPM+TPM token-bucket queueing and full-jitter backoff that honors `retry_after`. The documented
  pattern, made explicit: retries for transient (`RetryingProvider`), fallback for persistent
  (`HealthAwareFailover`), circuit-break for systemic (`CircuitBreaker`) — composed inner-to-outer.
- ✅ **Runtime model cascades** — the offline-optimized `RoutingPolicy` gains a runtime counterpart,
  `ModelCascade`: start on the cheapest rung and escalate to a stronger model only when a response's
  confidence falls below the rung threshold (default signal: a clean, schema-valid stop is confident; a
  truncated/filtered/unparseable answer is not), with per-route cost tracked. `app.use_cascade(...)`
  wires it as a first-class app feature; a custom confidence callable drives escalation from your own
  metric, and the routing optimizer keeps tuning the thresholds offline.
- ✅ **Cost attribution & budget SLOs** — every run carries request-time metadata
  (`user` / `tenant` / `feature` / `run`); cost is recorded as an attributed `CostEvent` at each model
  call in a run (tool loop, self-correction, batch, and the `agent`/`crew` handles included) against
  the versioned price table and rolled up by any dimension (`app.cost_report(by=...)` /
  `vincio cost report --by tenant|feature`). Per-tenant/feature/user **budgets** (`app.set_cost_budget`)
  enforce a policy on breach — **hard cap** (deny), **degrade-to-cheaper-model**, or
  **queue-to-batch** — as a `PolicyViolation` on the same audit path as every other decision; an
  `anomaly_factor` raises a `cost.anomaly` event on a spend spike. Attribution is captured at request
  creation, not retrofitted from logs, so long agentic traces are counted honestly.
- ✅ **Provider-aware prompt-cache strategy** — `PromptCacheStrategy` / `app.enable_prompt_caching`
  attaches an Anthropic `cache_control` breakpoint with a **TTL choice (5-minute / 1-hour)** to the
  compiler's stable prefix when it is long enough to be worth caching (Anthropic caches tools → system,
  so one system breakpoint covers both); auto-cache providers (OpenAI/Gemini) rely on the stable→volatile
  ordering the compiler already produces. **Cache-hit rate** is recorded on every model span from the
  `cached_input_tokens` providers report. The pass is purely additive — it only adds a TTL to
  breakpoints the compiler already chose.
- ✅ **Incremental indexing at scale** — `LiveIndex` gained **content-hash change detection** so only
  changed chunks re-embed (`UpsertStats` reports the re-embedding avoided), `upsert_stream` for
  streaming ingestion, and `ShardedIndex` — a corpus split across N backends, queried in parallel and
  merged, behind the existing `Index` protocol (a document's chunks co-locate by default), so it drops
  into the retrieval engine, behind a `LiveIndex`, or anywhere a single index would go.
- *Interconnection (held):* batch, circuit breakers, key pools, and cascades all implement the one
  `ModelProvider` interface, so the compiler, evals, guardrails, and security apply unchanged; cost
  attribution reuses the trace/cost model and the `tenant_id`/`user_id` already on traces; budget
  breaches are `PolicyViolation`s on the hash-chained audit path; the cache strategy builds on the
  compiler's cache-aware stable-prefix layout; `ShardedIndex`/`LiveIndex` keep full chunk provenance.
- *Edge over gateways (delivered):* LiteLLM/Bifrost give you failover and cost tracking as a **proxy
  hop** you operate separately; Vincio gives you the same — circuit breaking, cascades, attribution,
  enforced budgets, batch — **in-process, governed by your policy engine, and on one trace** with the
  rest of the run. See [docs/comparisons/litellm.md](docs/comparisons/litellm.md) and the new guide
  [docs/guides/cost-and-reliability.md](docs/guides/cost-and-reliability.md).
- **797 tests passing offline; ruff clean; VincioBench 103/103 budgets**; twenty-seven runnable
  examples. Batch reconciliation (in-process and both wire backends via `httpx.MockTransport`),
  circuit-breaker state machine + half-open recovery, health-aware failover, key-pool round-robin and
  429 backoff, cascade escalation, cost attribution/rollup, budget cap/degrade/queue-to-batch + anomaly
  events, the Anthropic cache-control TTL wire format, cache-hit telemetry, and incremental/sharded
  indexing are all covered offline; example `27_cost_and_reliability.py`; the VincioBench `scale` family
  gates batch-result correctness, failover/circuit recovery, cache-hit rate, attribution accuracy, and
  cascade savings (with four new SLOs).

See the [CHANGELOG](CHANGELOG.md) for the complete 1.3.0 notes.

### ✅ 1.4 — Reflective optimization & the data flywheel (vs DSPy 3) (shipped)

*0.8 shipped the closed loop: trace → dataset → eval → optimize → promote. 1.4 sharpens the optimizer
to the 2025–26 state of the art and adds the one lever the whole field is missing — turning production
traces into cheaper models — while keeping every promotion gated, grounded, and audited. Additive
behind `@experimental` entry points on the frozen 1.0 API, dependency-free.*

- ✅ **Reflective optimizer (GEPA-style)** — a `ReflectiveOptimizer` that, instead of blind mutation,
  reads the eval report's failures, **reflects** on why a prompt lost (a deterministic
  `HeuristicReflector`, or an `LLMReflector` with a deterministic fallback), and proposes targeted
  edits, evolving a **Pareto frontier** (it reuses `ParetoFrontier`). A child is screened on a minibatch
  and only earns a full rollout when it beats its parent, so the GEPA sample-efficiency win holds under a
  **hard evaluation budget**, deterministic under seed. MIPROv2-style joint instruction+example proposal
  is the second strategy (`strategy="mipro"`). The result is a drop-in `OptimizationResult`, so
  `ImprovementLoop(optimizer="reflective")` / `app.reflective_optimize(...)` / `vincio optimize
  reflective` promote through the identical gated path.
- ✅ **Distillation / fine-tune flywheel** — `app.export_training_set(...)` / `vincio distill` curates
  production traces (feedback-filtered, grounding-checked against the cited evidence, deduped, with full
  provenance) into provider-ready fine-tuning **JSONL** (OpenAI and Anthropic shapes), and a
  `BootstrapFinetune` teacher→student loop measures whether a cheaper student (optionally fine-tuned via
  an injected trainer) holds quality on the eval suite before it is promoted into a runtime
  `ModelCascade`. Every exported example is grounded and gated — the flywheel never trains on
  hallucinations. Export from `RunResult`s (`app.export_training_set(runs=[...])` /
  `export_training_set_from_runs`) is faithful by construction — they carry the full output and cited
  evidence and the runtime stamps the input — so no opt-in is needed; the trace path adds
  `enable_training_capture()` (covering streaming runs too) for teams curating from captured traces.
- ✅ **Learned prompt compression** — an `LLMLinguaCompressor` compiler pass (token-importance
  compression with a deterministic offline scorer and an optional learned hook) that sits alongside the
  extractive compressor as a drop-in `ContextCompiler.compressor`, protects the answer-bearing tokens
  (numbers, entities, citations, query terms), and is **faithfulness-gated**: `CompressionTuner` /
  `app.gate_compression(...)` adopt it only when it preserves the cited-fact set and holds quality under
  eval. `app.use_learned_compression()` installs it directly for opt-in users.
- ✅ **Optimizer-judge calibration** — `JudgeCalibrator` / `app.calibrate_judge(...)` reflectively tunes
  a `GEvalJudge`'s evaluation steps against κ-validated human labels (1.2), adopting a new procedure only
  when its Cohen's κ strictly beats the incumbent — and leaving the judge's gating weight reflecting the
  higher agreement. The judge that gates the optimizer is itself optimized.
- *Interconnection (held):* the reflective optimizer reuses the fitness function, the eval runner, the
  registry, the tracker, the Pareto frontier, and gated promotion — no new stores; distillation reuses
  the grounded-fact extractor from 0.8 and promotes into the 1.3 routing cascade; the compressor is just
  another compiler pass measured by the same VincioBench budgets; judge calibration reuses the 1.2
  Cohen's-κ machinery.
- *Edge over DSPy (delivered):* DSPy optimizes a program's prompts; Vincio applies reflective,
  Pareto-aware optimization across the **whole context lifecycle** (prompt, examples, retrieval weights,
  budget, compression) *and* exports the result as cheaper inference — with every step grounded, gated,
  and on one trace. See the updated [docs/comparisons/dspy.md](docs/comparisons/dspy.md) and the
  [close-the-loop guide](docs/guides/close-the-loop.md).
- **866 tests passing offline in ~4s; ruff clean; VincioBench 112/112 budgets**; twenty-eight runnable
  examples. The reflective optimizer (promotion, determinism, budget bound, safety-gated rejection,
  MIPRO), grounded export from runs and traces + dedup + feedback filter + streaming capture, the
  teacher→student gate, the LLMLingua pass + faithfulness gate, and judge-step calibration are all
  covered offline; example
  `28_reflective_optimization.py`; the VincioBench `loop` family gates reflective-search-vs-baseline
  lift, distillation grounded-only export + quality-hold, and compression fidelity + faithfulness gating
  (nine new budgets, three new SLOs).

See the [CHANGELOG](CHANGELOG.md) for the complete 1.4.0 notes.

### ✅ 1.5 — Multimodal, embeddings & retrieval breadth (vs LlamaIndex, Voyage/Cohere) (shipped)

*Keep retrieval best-in-field as the embedding and ingestion frontier moves — every new embedder, store,
and parser sits behind an interface that already exists, so breadth costs no new concepts. Additive
behind the frozen 1.0 API; the hosted embedders use only the core `httpx` dependency, and every store,
parser, and the realtime module is an opt-in extra.*

- ✅ **Matryoshka embeddings** — output-dimension truncation (MRL) on the existing `Embedder` interface:
  `build_embedder(kind, dimensions=N)` (or `MatryoshkaEmbedder` / `app` config `embedding_dimensions`)
  truncates and L2-renormalizes to `N` leading dimensions; hosted embedders request the shorter vector
  natively, everything else is wrapped, so the output is exactly `N` long. Storage/latency vs. recall is
  tracked per dimension in the `rag` benchmark family (recall@3 holds to one-eighth of the base
  dimension on the reference corpus).
- ✅ **Contextual & multimodal embedders** — `VoyageContextualEmbedder` (`voyage-context-3`, where the
  chunk vector carries document context, complementing `contextualize_chunks`) and unified text+image
  embedders `VoyageMultimodalEmbedder` (`voyage-multimodal-3`) and `CohereMultimodalEmbedder`
  (`embed-v4.0`) via `build_embedder` and `MultimodalInput` / `embed_multimodal`. Query-vs-document
  `input_type` hints are plumbed through `VectorIndex` (document on add, query on search) for every
  input-type-aware embedder, with `embed_texts` keeping custom embedders working unchanged.
- ✅ **More vector stores** — Weaviate, Milvus, Elasticsearch/OpenSearch, and Vespa adapters behind the
  one `Index` protocol and `build_vector_index` factory, joining Qdrant, pgvector, Chroma, Pinecone, and
  LanceDB — each lazy-imports its SDK with a helpful `StorageError` and accepts an injected client for
  offline round-trip tests.
- ✅ **Richer extraction** — a layout-aware document-extraction path (`load_document(path, layout=True)` /
  `extract_pdf_layout`) that recovers column-aware reading order, tables with bounding boxes, and figure
  regions for complex PDFs via `vincio[pdf-layout]` (pdfplumber); the dependency-free pypdf text path
  stays the default. The reading-order/assembly logic is pure and offline-tested.
- ✅ **Voice / realtime (optional module)** — `vincio.realtime`: a provider-neutral `RealtimeSession`
  over OpenAI Realtime / Gemini Live (WebSocket) or a deterministic in-process backend, with VAD,
  interruption (barge-in), and **in-session tool calls routed through the same permissioned, sandboxed,
  audited tool runtime** (`app.realtime_session(...)`). A separate `vincio[realtime]` extra, explicitly
  scoped as a stateful bidirectional module (`@experimental`), *not* core context engineering.
- *Interconnection (held):* every new embedder, store, and parser feeds the same compiler — chunked,
  scored, budgeted, cited, and benchmarked exactly like a local file; nothing downstream changes.
  Realtime tool calls ride the existing tool registry, so they are permissioned and audited like any
  other tool.
- *Edge over specialists (delivered):* Voyage/Cohere give you MRL, contextual, and multimodal embeddings,
  and LlamaIndex gives you the store integrations; Vincio gives you all of them **behind one
  `build_embedder` / `build_vector_index` and inside one scored, budgeted, cited packet** — see the
  updated [docs/comparisons/llamaindex.md](docs/comparisons/llamaindex.md) and
  [ragatouille.md](docs/comparisons/ragatouille.md).
- **919 tests passing offline; ruff clean; VincioBench 116/116 budgets**; twenty-nine runnable examples.
  MRL truncation + native dimensions, input-type plumbing, contextual and multimodal embedder wire
  formats (httpx `MockTransport`), the four vector stores (injected-fake round trips + helpful
  missing-dependency errors), layout reading-order/table/figure assembly, and the realtime session
  (lifecycle, VAD, interruption, tool dispatch, wire-event translation) are all covered offline; example
  `29_multimodal_retrieval.py`; the VincioBench `rag` family gates MRL recall-vs-dimension and unified
  multimodal recall/MRR (four new budgets, three new SLOs).

See the [CHANGELOG](CHANGELOG.md) for the complete 1.5.0 notes.

### 🚧 1.6 — Enterprise governance & compliance

*Turn the audit and security spine Vincio already has into the evidence regulated buyers now require —
all generated in the library, on your infrastructure. No hosted compliance program (that stays
[out of scope](#out-of-scope)); just the artifacts and controls, emitted as files you own.*

- 🚧 **Model & system cards** — `vincio governance card` generates machine-readable **model cards**
  (model id/version, capabilities, limitations, pricing) and **system cards** (model + retrieval +
  memory + safety filters + human-oversight points) from the live app configuration and eval evidence;
  pluggable schema (Open Model Card / AI Cards) since no format has won.
- 🚧 **Compliance-framework mapping** — the `RedTeamSuite` and audit log map findings to **OWASP LLM Top
  10 (2025)**, **OWASP Agents**, **NIST AI RMF (GenAI profile)**, and **MITRE ATLAS**; `vincio
  governance report` emits the coverage matrix as compliance evidence, and evaluation results (1.2)
  attach as measured evidence for the relevant controls.
- 🚧 **EU AI Act artifacts** — a synthetic-content **output marking** hook (C2PA-style provenance
  manifest / watermark metadata) for the 2 Aug 2026 transparency duty, an AI-interaction-disclosure
  helper, and a training-/grounding-data summary export — all configurable and dated-deadline-agnostic.
- 🚧 **AI-BOM & supply chain** — extend the shipped CycloneDX SBOM + SLSA provenance with an **AI-BOM**
  (base-model identity and version, embedding/rerank models, fine-tune datasets, prompt/registry
  versions) and SHA-256 model-hash verification, for blast-radius assessment when a model or dataset is
  found compromised.
- 🚧 **Data lineage & erasure-by-source** — first-class lineage from source document → chunk → evidence
  → output (extending the evidence ledger's provenance), so `app.erase_source(...)` can satisfy a GDPR
  right-to-erasure across indexes, caches, and memory, logged on the audit chain.
- 🚧 **Data-residency-aware routing** — pin a run (or tenant) to provider regions and refuse egress when
  policy requires in-jurisdiction processing, enforced by the deterministic policy engine.
- 🚧 **Multilingual** — non-English PII/secret detectors (locale packs) extending today's English-centric
  detectors, per-language **eval slicing** (so the ~24% high-vs-low-resource accuracy gap can't hide in
  an aggregate), and tokenizer-aware **fertility/cost telemetry** that surfaces the non-English "token
  tax" per tenant/language and makes it routable.
- 🚧 **RAG-poisoning & injection hardening** — authority/provenance-based poisoning detection on
  retrieved evidence (a handful of crafted docs can flip ~90% of answers) and a hook for a
  PromptArmor-class injection classifier, extending the existing trust-tag/heuristic defense, with
  FP/FN telemetry.
- *Interconnection:* every artifact is generated from data Vincio already holds — the audit chain, the
  evidence ledger, eval reports, the price table, the registry — so governance is a *view* over the
  running system, not a parallel bookkeeping burden; residency and poisoning controls are
  `PolicyViolation`s on the same audit path.
- *Edge over the field:* governance bolted onto an app is documentation; Vincio's is **mechanical and
  measured** — cards and BOMs generated from the live config, framework mappings backed by red-team and
  eval evidence, erasure enforced through the same lineage that cites your answers.
- *Target:* card/BOM generation, framework-mapping coverage, erasure completeness, residency
  enforcement, and a non-English PII suite covered offline; example `30_governance_compliance.py`; a
  VincioBench `governance` family gating card/BOM completeness, mapping coverage, erasure correctness,
  and multilingual PII recall.

### 🔭 Exploring — later, and 2.0

Candidates that are real but not yet scheduled — pulled forward when demand and the standards settle:

- 🔭 **Distributed execution** — sharded retrieval and a distributed work queue for graph/crew super-steps
  across processes, keeping the single-process path as the default. (Durable checkpoint/resume already
  ships; this is horizontal scale.)
- 🔭 **AGNTCY / ACP** — the REST-native agent-interop alternative to A2A, if it gains adoption.
- 🔭 **MCP Apps & 2026 spec** — server-rendered UI and the stateless-core changes from the in-flight
  2026 MCP spec, once it ships stable (current target is the 2025-11-25 spec).
- 🔭 **On-device / edge embedding & inference** — first-class quantized local models beyond the existing
  OpenAI-compatible passthrough.
- 🔭 **2.0 — the one breaking window.** Reserved for changes the frozen 1.x surface cannot make
  additively: collapsing any deprecated aliases accumulated across 1.x, adopting finalized OTel GenAI
  *agentic* semantic conventions if they break the current attribute names, and any Pydantic/Python
  floor bumps. 2.0 ships only when there is a real breaking need — never for its own sake — and with the
  same mechanical deprecation runway 1.0 established.

---

## Out of scope

Vincio is a library, and stays one. The building blocks for running it in production — a
hash-chained audit log, retention policies, tenant isolation, RBAC / ABAC, and a server — ship in
the package so you can deploy them on your own infrastructure. **Hosted services, managed control
planes, dashboards-as-a-service, and compliance programs are not part of this project.** Vincio
gives you the engine; how and where you run it is yours.
