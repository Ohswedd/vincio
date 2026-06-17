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
   now require. Vincio had the audit/security spine but not the compliance evidence on top of it —
   **1.6 (shipped) closes this gap** with model/system cards, OWASP/NIST/MITRE framework mapping backed
   by red-team and eval evidence, an AI-BOM with model-hash verification, EU AI Act synthetic-content
   marking, data lineage with right-to-erasure-by-source, data-residency-aware egress refusal, and
   non-English PII locale packs with per-language eval slicing and token-tax telemetry.

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
| **DeepTeam / NeMo / governance** | OWASP/NIST/MITRE mapping, safety classifiers | Red-team + audit mapped to OWASP LLM 2025 / OWASP Agents / NIST AI RMF / MITRE ATLAS, model/system cards, AI-BOM, lineage, residency, multilingual — all from the existing audit/security spine | 1.6 ✅ |

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

### ✅ 1.6 — Enterprise governance & compliance (shipped)

*Turn the audit and security spine Vincio already has into the evidence regulated buyers now require —
all generated in the library, on your infrastructure. No hosted compliance program (that stays
[out of scope](#out-of-scope)); just the artifacts and controls, emitted as files you own. Additive
behind `@experimental` entry points on the frozen 1.0 API, dependency-free.*

- ✅ **Model & system cards** — `vincio.governance.generate_model_card` / `generate_system_card`,
  `app.model_card()` / `app.system_card()`, and `vincio governance card` generate machine-readable
  **model cards** (id/version, capabilities, limitations, live pricing) and **system cards** (model +
  retrieval + memory + safety filters + human-oversight + governance controls) from the running
  configuration and optional eval evidence. The schema is pluggable (`CardFormat`: Vincio native,
  Open Model Card, EU "AI Cards") since no format has won; cards render from one captured fact set.
- ✅ **Compliance-framework mapping** — `ComplianceMapper` / `app.compliance_report()` / `vincio
  governance report` map a data-driven control catalog for **OWASP LLM Top 10 (2025)**, **OWASP
  Agentic AI**, **NIST AI RMF (GenAI profile)**, and **MITRE ATLAS** onto Vincio's capabilities,
  backed by *measured* evidence — `RedTeamSuite` probe outcomes, the security configuration, and
  `EvalReport` metrics. The `ComplianceReport` is a coverage matrix (`covered`/`partial`/`not_covered`
  with the evidence string for each, `to_markdown()` for auditors); uncovered controls are reported
  honestly, never hidden in an aggregate.
- ✅ **EU AI Act artifacts** — `mark_synthetic_content` emits a **C2PA-style provenance manifest**
  (IPTC `trainedAlgorithmicMedia`, bound to the output by SHA-256), `ai_disclosure` returns a
  localized **AI-interaction disclosure**, and `data_summary` exports a **grounding-data summary**.
  `governance.content_marking` attaches the manifest + disclosure to every run's `result.metadata`.
  Deadline-agnostic and configurable; signing is left to your pipeline.
- ✅ **AI-BOM & supply chain** — `generate_aibom` / `app.aibom()` / `vincio governance aibom` extend
  the shipped CycloneDX SBOM + SLSA provenance with an **AI-BOM** (base model + version,
  embedding/rerank models, fine-tune datasets, prompt/registry versions) as CycloneDX-1.6
  `machine-learning-model` / `data` components, each with an optional **SHA-256 hash**;
  `AIComponent.verify` / `AIBOM.verify_all` confirm artifacts for blast-radius assessment.
- ✅ **Data lineage & erasure-by-source** — a `LineageIndex` records source → document → chunk →
  evidence → output as the app ingests and runs (`app.trace_lineage(...)`), so
  `app.erase_source(...)` satisfies a GDPR right-to-erasure across **every index, memory, and cache**,
  logged on the hash-chained audit chain (`erase_source`) and idempotent by construction.
- ✅ **Data-residency-aware routing** — `ResidencyPolicy` / `app.set_residency(...)` /
  `governance.allowed_regions` pin allowed provider regions and **refuse egress** to others as a
  blocking `PolicyViolation` recorded as a `residency_check` deny — enforced deterministically at the
  provider-resolution choke point before any request leaves the process.
- ✅ **Multilingual** — non-English PII **locale packs** (`vincio.security.locales`: France, Germany,
  Spain, India, Singapore, Brazil, UK national-ID and phone formats) via `PIIDetector(locales=[...])`
  and `governance.locales`, layered on the English path without changing it; per-language **eval
  slicing** (`EvalReport.slice_by_tag` / `tag_gap`) surfaces the high-vs-low-resource gap; and a
  tokenizer **fertility tracker** (`app.fertility`) makes the non-English "token tax" visible and
  routable per language and tenant.
- ✅ **RAG-poisoning & injection hardening** — `PoisoningDetector` flags likely-poisoned retrieved
  evidence from **authority/provenance** signals (embedded instructions, low-authority/high-promotion
  sources, consensus outliers) before it reaches the model, with an optional async PromptArmor-class
  classifier hook and **FP/FN telemetry** (`PoisoningReport.telemetry`), extending the existing
  trust-tag/heuristic defense.
- *Interconnection (held):* every artifact is generated from data Vincio already holds — the audit
  chain, the evidence ledger, eval reports, the price table, the registry — so governance is a *view*
  over the running system, not a parallel bookkeeping burden; residency and erasure are
  `PolicyViolation`s and audit entries on the same hash-chained path as every other decision.
- *Edge over the field (delivered):* governance bolted onto an app is documentation; Vincio's is
  **mechanical and measured** — cards and BOMs generated from the live config, framework mappings
  backed by red-team and eval evidence, erasure enforced through the same lineage that cites your
  answers. See the new guide [docs/guides/governance.md](docs/guides/governance.md).
- **986 tests passing offline; ruff clean; mypy clean; VincioBench 131/131 budgets**; thirty runnable
  examples. Cards/AI-BOM completeness, framework-mapping coverage and red-team/eval evidence, erasure
  correctness across indexes + audit, residency egress refusal, multilingual PII recall + English-path
  intactness, RAG-poisoning FP/FN telemetry, fertility token-tax, and eval slicing are all covered
  offline; example `30_governance_compliance.py`; the VincioBench `governance` family gates card/BOM
  completeness, mapping coverage, erasure correctness, multilingual PII recall, residency
  endpoint-inference, and signature verification (15 budgets, three SLOs).
- *1.6.1 (no gaps):* the type-checker is now a CI gate (`mypy vincio` clean across the package);
  residency infers the region from a **region-pinned endpoint** (AWS/GCP/Vertex/sovereign URLs) with
  jurisdiction-aware matching, not just a hand-maintained map; and synthetic-content manifests are
  **signable** (`HmacSigner` / custom `ContentSigner`) and verifiable (`verify_manifest`).

See the [CHANGELOG](CHANGELOG.md) for the complete 1.6.0 and 1.6.1 notes.

---

## The road to 3.0

1.6 closed the last of the six 2025–26 shifts: Vincio now speaks the protocols, scores its
trajectories, survives production traffic, optimizes itself, embeds the modern way, and emits the
governance evidence regulated buyers require. Finishing the field's last lap surfaced a sharper
question than *what's missing* — *what is true*. An honest audit of the running code found a recurring
shape: a carefully-engineered spine wrapped around heuristic muscle and a few load-bearing shortcuts.
The single-shot run path advertises a `Budget` but enforces only latency and tool-count; the compiler
shipped embeddings in 1.5 yet still scores, dedupes, and resolves conflicts with bag-of-words;
`trace_replay_plan` extracts a replay but nothing replays it; A2A advertises streaming it never
dispatches; the model "registry" is substring-sniffing across three provider files with a price table
that silently bills $0 for an unknown model; the `output` module, despite its name, only *validates*
model text and cannot produce a document; and the whole continual-learning story observes drift but
never acts on it. None of these are bugs of neglect — they are the seams where an ambitious library
outgrew its first implementation.

The road from 1.7 to 3.0 makes the spine's promises literally true, then extends it into the three
places the frontier moved since 1.6: documents and images flowing *out* as cited, governed artifacts
(not just *in*); models rotating and regressing under a statistical gate on every swap; and the loop
closing itself — continual, online, and safe. **1.7** makes the spine honest and fast and lays the
model-registry foundation; **1.8** turns that registry into a rotation-and-regression discipline that
gates every model swap; **1.9** makes documents and images flow out as cited, provenance-stamped,
eval-gated artifacts; **1.10** closes the loop into continual, online, safely-rolled-back
self-improvement and opens the agentic frontier (deep research, computer-use, memory-as-tools) behind
hardened isolation. **2.0** (shipped) was the one breaking window — the structural refactor the frozen
surface could not make additively (facades, async-first stores, structured filter pushdown, enterprise
auth, egress DLP, a signed audit chain), plus the flagship multimodal-native Context Packet that
genuinely needed the schema change.
**2.1** (shipped) was additive again — distributed durable execution (lease/CAS + worker pool +
`Send` map-reduce), executed swap-gated fine-tuning, a served (still self-hosted) observability and
alerting plane, Redis shared state with `vincio serve`, and quantized two-stage retrieval plus
batteries-included local neural models. **2.2** is the environment-based agentic eval harness and the
governed agent fabric. **3.0** is the next breaking culmination: a single declarative self-improvement
contract, provable erasure with consent modeling, and an async-first canonical core.

The three founding principles are unchanged — **beat the specialist and add what it structurally
cannot** (provenance, budgeting, eval-gating, one trace); **interconnect, don't bolt on** (every
capability reads and writes the same Context Packet, evidence ledger, audit log, and trace stream);
**performance is a feature** — and so are the two that joined at 1.0: **everything in the 1.x line is
additive** behind a new entry point or opt-in extra on the frozen 1.0 API, and **standards** (MCP,
A2A, OWASP/NIST, OTel, C2PA) are implemented in-library and run in your process, never as a hosted
dependency. **2.0 and 3.0 are the only breaking windows**, each shipped with the mechanical
deprecation runway 1.0 established and never for its own sake.

**Legend:** ✅ shipped · 🚧 planned (next) · 🔭 exploring (later). Milestones are ordered by
dependency and urgency: honesty-and-foundation first (nothing rotates, regresses, or self-improves
correctly on an unenforced budget and a lexical scorer), then rotation and regression, then documents
and images out, then continual learning, then the breaking refactor, then scale and benchmarks, then
the breaking culmination.

### The honesty audit — what the running code revealed

An audit of the running code against the field (LangChain/LangGraph, LlamaIndex, DSPy 3, Pydantic AI,
CrewAI, Haystack, Letta/MemGPT, LiteLLM/OpenRouter, Unstructured/Docling, Voyage/Cohere, the serving
layer, and the MCP/A2A/AGNTCY and EU AI Act standards) found six places where an ambitious spine has
outgrown its first implementation — each now scheduled:

1. **Model rotation became a discipline, not an incident.** Providers publish lifecycle metadata
   (GA/deprecation/retirement dates) and the field bakes in model catalogs (LiteLLM model map,
   OpenRouter `/models`, the Vercel AI SDK) with capability- and cost-aware routing, shadow/canary
   rollout, and eval-gated promotion. *The gap:* Vincio has no model registry — capabilities are
   substring-sniffed across three provider files, pricing is a hand-maintained dict that silently
   bills $0 for an unknown model, and failover/cascade swap models blindly with no capability guard or
   regression check. → *1.7 (registry foundation), 1.8 (rotation + swap-regression discipline)*
2. **Regression testing moved from output diffs to behavioral replay with statistical rigor.** Teams
   replay captured production traffic against a candidate, run model-swap A/B with significance and
   effect sizes, gate on confidence intervals, and pair canary rollout with automatic rollback. *The
   gap:* Vincio has the ingredients (`trace_replay_plan`, `evaluate_gates`, `DriftMonitor`,
   paired/Welch `ab_test`) but unassembled — the replay plan is only extracted and never executed,
   gates compare point estimates while the t-test is never called at the gate, and there is no
   model-swap regression command, flake control, or auto-rollback. → *1.7 (significance-gated
   promotion + replay executor), 1.8 (swap gate)*
3. **The compiler's intelligence is expected to be semantic, not lexical.** Embedding relevance, MMR
   diversity, semantic dedup, and value-level contradiction are table stakes, and reranker scores
   drive ranking, not a yes/no gate. *The gap:* despite embeddings shipping in 1.5, the compiler
   instantiates its scorer with no `similarity_fn`, so relevance, novelty, dedup, and conflict are all
   bag-of-words; the reranker's relevance is read only as a min-relevance gate; and conflict is a
   negation-word XOR that misses every value disagreement. → *1.7 (embedding-wired scoring, value
   contradiction, full-window budgeting)*
4. **The run contract is expected to be enforced, not advisory.** Hard cost/token/step budgets, a
   unified streaming/non-streaming pipeline, cooperative cancellation, provider-native token counts,
   and plugin entry points are the production baseline. *The gap:* on `app.run()` only latency and
   tool-count are enforced — `max_cost_usd` / `max_input_tokens` / `max_output_tokens` / `max_steps`
   are silently ignored and `BudgetExceededError` is unreachable; the streaming path lacks the latency
   deadline and a shared cancellation epilogue; persistence blocks the event loop mid-pipeline; and
   there is no entry-point discovery for providers/embedders/stores. → *1.7*
5. **Documents and images now flow OUT, not just in, with the same guarantees as text.** Enterprise
   context engineering means cited, governed deliverables (DOCX/PDF/PPTX/HTML, filled forms, redlines,
   generated images), with image generation/editing and TTS as first-class output modalities carrying
   embedded provenance. *The gap:* the `output` module, despite its name, only parses and validates
   model text — no document generation, cited-report assembly, form filling, redlining, image-gen/edit,
   or TTS; OCR and the image analyzer exist but are never wired into the loaders; audio is typed but
   ingested by nothing; and the C2PA marker can only hash `str`, never media bytes. → *1.9*
6. **Self-improvement became continual and online, with safe rollout.** The bar is a system that
   detects drift on live traffic, proposes what to optimize, runs safe online updates behind
   canary/shadow with automatic rollback, and meta-optimizes its own search — plus a frontier of deep
   research, computer-use, and self-editing memory. *The gap:* every optimizer is offline /
   manually-triggered — `OnlineEvaluator` only samples, `DriftMonitor` only emits an event, the routing
   bandits are unwired, promotion swaps the live prompt with no canary/rollback, and the GEPA
   reflector's LLM path is an unwired shim; there is no deep-research primitive, computer-use surface,
   or agent-controlled memory, and the subprocess + `setrlimit` sandbox bounds any code-executing
   use-case. → *1.10 (online controller, real reflector, deep research, memory-as-tools, computer-use),
   3.0 (unified self-improvement contract)*

### Post-1.6 competitive coverage map

| Competitor / standard | What it's good at (2026) | Vincio answer (and the edge we add) | Milestone |
|---|---|---|---|
| **LiteLLM / model cost & context map** | Data-driven model catalog (capabilities, context window, pricing) keyed by model id | A data-driven ModelRegistry that consumes the previously-dead ModelProfile, binds lifecycle (GA/deprecation/retirement) dates, and is the single source capability guards, cost SLOs, and rotation all consult — in-process, overridable, shippable as data | 1.7 |
| **LiteLLM Router / OpenRouter auto-routing** | Cost/latency/least-busy routing and capability-aware model selection across many providers | A registry-backed router with capability preflight that routes by cost/latency inside your policy/budget/audit boundary and refuses capability-mismatched substitutions on failover — not a proxy hop | 1.8 |
| **Braintrust / LangSmith model experiments** | Model-vs-model comparison views over eval datasets | A SwapGate that replays golden traces + diffs quality/cost/latency/behavior with significance and PASS/FAILs the migration — the swap is gated, not just compared, on one trace | 1.8 |
| **LangSmith trace-based regression / promptfoo replay** | Replay captured production traffic against a candidate build | A ReplayRunner that actually executes trace_replay_plan (today only extracted), diffs outputs/trajectory/cost, and gates with confidence intervals + flake quarantine — reproducible behavioral regression in-process | 1.7, 1.8 |
| **LaunchDarkly / Statsig progressive delivery** | Canary/shadow rollout with guarded automatic rollback | A ShadowProvider and CanaryRouter that qualify a candidate model on live traffic and auto-roll-back on statistical regression — in-process, with the canary-driven prompt/policy promotion landing at the 2.0 serving surface | 1.8, 2.0 |
| **Unstructured.io + python-docx/reportlab/python-pptx writers** | Rendering documents to DOCX/PDF/PPTX/HTML | A DocumentBuilder that emits cited, structurally-validated, provenance-stamped, budget-metered, eval-gated documents from a validated result — the deliverable closes the same loop the audit chain opened | 1.9 |
| **LlamaIndex CitationQueryEngine / Anthropic citations** | Cited answers with per-span source attribution | A CitedReportBuilder that resolves [E1] markers to footnotes + bibliography AND verifies per-claim entailment, rendered into a shippable document — every claim cited and supported | 1.9 |
| **OpenAI Images/gpt-image-1, Gemini Imagen, OpenAI/Gemini TTS** | Image generation/editing and speech synthesis | An image-gen/edit and TTS provider abstraction where every generated asset is budgeted, eval-gated, and C2PA-provenance-stamped on the same audit chain as text — the governance machinery finally reaches generated media | 1.9 |
| **DSPy 3 SIMBA / online RL optimization loops** | Self-improving modules that update from production signals | An online improvement controller that turns live drift into a gated, reversible re-optimization or rollback, plus a real provider-backed GEPA reflector and guarded online bandits with auto-rollback — the loop closes itself, in-process | 1.10 |
| **OpenAI/Gemini/Perplexity Deep Research, GPT-Researcher** | Iterative search→read→synthesize research with cited reports | A ResearchAgent where every claim is grounded, cited, budget-bounded, and eval-gated by construction, composed from the existing query-understanding planners, grounded-fact extractor, and cited-report renderer | 1.10 |
| **Letta / MemGPT agent memory OS** | Self-editing core/archival memory with context-window paging | Memory operations as permissioned tools over the existing audited write pipeline plus a context-pressure pager — a self-editing memory that is still guarded, provenance-tracked, and audited | 1.10 |
| **Anthropic computer-use / OpenAI Operator** | GUI/browser agent loops (screenshot→action→observe) | A computer-use action surface running through the permissioned/audited/budgeted tool runtime behind a hardened pluggable isolation backend (container/microVM/gVisor/WASM) — the safety envelope thin adapters lack | 1.10 |
| **LiteLLM enterprise endpoints (Bedrock/Vertex/Azure)** | 100+ providers including the enterprise deployment surfaces | Bedrock/Vertex/Azure behind a pluggable HTTPProvider auth strategy, routed through the same registry, swap gate, residency, and audit chain as every other provider — enterprise endpoints inside the governance boundary | 2.0 |
| **Qdrant/Weaviate/pgvector native filtering & OTel GenAI** | Server-side metadata filter pushdown, agentic semantic conventions | A structured FilterSpec compiled to every backend's native filter (closing an under-fill bug and a cross-tenant exfiltration risk) plus the finalized OTel agentic conventions on a unified telemetry contract | 2.0 |
| **Temporal/Ray + DSPy BootstrapFinetune + Langfuse/Phoenix** | Distributed durable execution, executed fine-tuning, served observability | A lock-free distributed RuntimeBackend, executed-and-swap-gated fine-tune jobs, and a served observability+alert plane — all self-hosted inside one audit chain, never a control plane | 2.1 ✅ |
| **τ-bench/SWE-bench/WebArena/GAIA + AGNTCY + AG-UI** | Stateful-environment agent leaderboards, agent fabric, generative UI | An Environment eval harness feeding the optimizer + benchmark adapters in VincioBench, a governed AGNTCY/A2A agent directory under one allow-list, and AG-UI streaming that inherits the run's provenance | 2.2 |

### ✅ 1.7 — Make the spine honest & fast — enforced budgets, semantic compiler, the model registry foundation (vs LiteLLM, LlamaIndex, the gateways) (shipped)

*Before Vincio rotates models, regresses swaps, or improves itself, the spine's promises must be literally true and the model-knowledge layer must be data, not substring guesswork. 1.7 turns the advertised `Budget` into a hard cap, wires the 1.5 embeddings into the compiler so selection is semantic instead of bag-of-words, unifies the divergent streaming path, takes persistence off the event loop, fixes local-image input, and lands a data-driven `ModelRegistry` that finally consumes the underused `ModelProfile` type (today read only by the prompt compiler) — every change additive behind a new entry point or opt-in flag on the frozen 1.0 API, every promotion now gated on statistical significance instead of a point estimate, all `@experimental`.*

- **Enforced full Budget on the single-shot run path** — thread a `BudgetUsage` through `_execute_inner` / `_model_tool_loop` in `core/runtime.py` and call `exceeds()` after each model call and tool round so `max_cost_usd`, `max_input_tokens`, `max_output_tokens`, and `max_steps` become hard caps for `app.run()` / `arun()` — today only latency and tool-count are enforced and `BudgetExceededError` is dead code. A pre-flight input-token estimate is checked against `max_input_tokens` before the first call, and the `BudgetAllocator` (`context/budgeting.py`) now reserves response and tool-loop tokens so the input-only allocator finally accounts for the full window. An opt-out flag preserves legacy soft-cap behavior for one minor. The cap fires at the same choke point as residency and policy, recorded on the same audit chain and trace.
- **Embedding-wired semantic scoring, MMR selection & value-level contradiction** — thread the app embedder into `ContextScorer.similarity_fn` (cosine over cached embeddings) and into `near_duplicate_score` / `novelty`, blend the reranker's `upstream_relevance` into `ContextScores.relevance` instead of using it only as a `min_relevance` gate, and make `_select` a real embedding-cosine MMR with a relevance/diversity lambda. The negation-XOR conflict trigger is replaced by a salient-unit value-disagreement check (`salient_units` over numbers/dates/entities in `context/llmlingua.py`) that emits structured conflict deltas into the packet. Defaults stay lexical when no embedder is configured, so it is fully additive — and when one is, selection *and* ordering are driven by the cross-encoder signal inside one scored, budgeted, cited packet.
- **Unified run pipeline + cooperative cancellation + async stores** — collapse `execute` and `execute_stream` onto one inner generator so the two paths share the latency-deadline wrapper and cancellation semantics (today `execute_stream` already persists and emits `run.completed`, but lacks the `asyncio.timeout` deadline `execute` enforces and reimplements the inner loop, so the divergence drifts as features land). `arun` / `astream` return a `RunHandle` exposing `cancel()` that propagates cooperative cancellation into the bounded-concurrency groups in `core/concurrency.py` with a clean `CANCELLED` epilogue that still persists and audits. An async store contract (`asave` / `aquery` in `storage/base.py`) with `to_thread` wrappers for sync impls batches the packet and run writes so persistence stops blocking the event loop mid-pipeline. A cancelled run is still fully recorded on one trace and audit chain.
- **ModelRegistry: data-driven capabilities, pricing & lifecycle (consumes ModelProfile)** — introduce `vincio/providers/registry.py`: a versioned, hot-reloadable, config-overridable data catalog keyed by exact model id, binding `ModelCapabilities` + pricing (batch/cache tiers and effective dates) + context window + modalities + GA/deprecation/retirement dates, instantiating the underused `core/types.py` `ModelProfile` (today consumed only by the prompt compiler, never by a provider or registry) as the registry record. `capabilities()` and the `observability/costs.py` `PriceTable` both derive *from* it, with substring sniffing demoted to a last-resort fallback; unknown-model lookups warn and emit `model.unknown` instead of silently costing $0. `importlib.metadata` entry-point groups (`vincio.providers` / `embedders` / `stores`) let third parties ship adapters as separate pip packages that auto-register, and provider-native exact token counters (Anthropic `count_tokens`, Gemini `countTokens`) sit behind the `TokenCounter` Protocol (`core/tokens.py`), selected by resolved model. The catalog becomes the single source of truth that capability guards, cost SLOs, and (next milestone) rotation all consult.
- **Significance-gated promotion + trace-replay executor** — make every auto-promotion (loop, reflective, budget, compression, distill, retrieval) require a statistically significant improvement using the existing paired / Welch `ab_test` from `evals/experiments.py` — reported as a p-value, confidence interval, and effect size in the result and audit record — replacing the `1e-6` mean-delta gate, with `min_dataset_coverage` raised and under-powered runs warned. Build the `ReplayRunner` that the existing `trace_replay_plan` only describes: re-run captured trace inputs through a target app (optionally pinning recorded tool outputs for determinism) and diff outputs, trajectory, and cost by reusing `trace_diff` and `EvalReport.diff`, surfaced via `vincio trace replay --against`. Promotions are now defensible at a confidence level on the same audit chain, and behavioral regression becomes a reproducible primitive instead of a stub.
- **OpenAI local-image fix + truthful protocol capabilities** — stop emitting unreachable `file://` image URLs in `providers/openai.py` by base64-encoding local paths into data URLs like the Anthropic/Google providers already do, via one shared `ImageRef`-to-data-url helper (with size/dimension caps) reused by all three chat providers and the multimodal embedders. Paired with the honest-protocols fixes the audit flagged: default A2A `capabilities.streaming=False` until `message/stream` is actually dispatched, replace the no-sleep MCP `_await_task` busy-loop (`mcp/client.py`) with exponential backoff plus a wall-clock deadline, and make the A2A client poll `working` / `submitted` tasks to terminal instead of mis-reporting them as failed. Vision input works on the default provider and the advertised protocol surface matches the implemented one — correctness fixed without changing any public verdict shape.
- **Sub-quadratic compilation, inverted-index BM25 & an optional numpy/ANN path** — pay down the algorithmic hot paths the performance audit flagged, all additive. `_select` re-scores the whole candidate pool on every pick (O(n²)) and dedup/conflict shingle pairwise — replace them with incremental top-k selection and a MinHash/LSH blocking pass so near-duplicate and conflict detection go near-linear; make pure-Python BM25 (`retrieval/indexes.py`) actually use its own `_df` posting lists instead of rescanning every document per query term; add an optional `numpy`/HNSW path for the local vector index so cosine isn't a Python loop at corpus scale; and memoize `count_tokens` (`core/tokens.py`) so the same text isn't re-tokenized across compiler passes. Pure-Python stays the zero-dependency default behind availability checks, and a new `pytest-benchmark` **perf-regression gate** in the VincioBench `perf` family fails CI on any compile/retrieval-latency regression.
- **Hardened detectors, normalized injection defense & evidence-gated compliance** — close the security audit's top weaknesses without a breaking change. The injection detector (`security/injection.py`) gains a normalization + decode pre-pass (NFKC fold, zero-width strip, leetspeak/homoglyph fold, recursive base64/hex/rot13 decode) before its regex and heuristic signals run, so obfuscated attacks stop slipping past pattern matches; the PII / injection / secret detectors accept a pluggable ML backend (`DetectorBackend` Protocol) alongside the deterministic default; the tenant filter (`security/access.py`) stops treating `tenant_id=None` as global — a cross-tenant fail-open that is both a correctness and an exfiltration risk — by requiring an explicit scope, an additive fix ahead of the 2.0 native filter pushdown; and `ComplianceMapper` (`governance/frameworks.py`) reads a control as `covered` only when backed by *measured* red-team / eval evidence, not a config flag alone, so the auditor matrix reflects defense that was actually exercised.
- *Interconnection:* everything new reads and writes the same organs — enforced budgets ride the same choke point and audit chain as residency and policy; semantic scoring feeds the same Context Packet the compiler already produces; the `ModelRegistry`'s `ModelProfile` becomes the addressing unit that cost, capabilities, and (1.8) rotation all share; significance gates and the replay executor reuse the existing `ab_test`, `evaluate_gates`, `trace_diff`, and audit record rather than inventing parallel bookkeeping.
- *Edge over specialists:* LiteLLM gives you a model map, LlamaIndex/LangChain an MMR retriever, the gateways a soft cost meter, and the OpenAI/Anthropic SDKs per-request cancellation; Vincio makes the budget a hard cap on the same trace, the reranker a selection *and* ordering signal in the same cited packet, the model catalog the single registry that cost, capability guards, and rotation all consult, and cancellation identical across streaming and non-streaming with the cancelled run still fully recorded — structural guarantees in-process, not a service, every promotion statistically gated on the same audit chain.
- *Definition of done (delivered):* budget enforcement, semantic-scoring + MMR + value-contradiction, unified-pipeline parity + cooperative cancellation, async-store persistence, `ModelRegistry` capability/price/lifecycle correctness vs the substring fallback, significance-gated promotion, and the trace-replay executor are all covered offline and gated by the VincioBench `cost`, `rag`, `reliability`, `perf`, and `loop` families — budget-cap enforcement and the unknown-model $0-warning in `cost`; embedding-MMR + value-contradiction in `rag`; streaming/non-streaming parity and cancellation recording in `reliability`; inverted-index BM25 + token memoization in `perf`; registry lookup, significance-gated promotion, and replay fidelity in `loop` — with `examples/31_honest_fast_spine.py` and the SLOs extended accordingly.
- **1034 tests passing offline in ~4.5s; ruff + mypy clean**; thirty-one runnable examples; the VincioBench `cost` / `rag` / `reliability` / `perf` / `loop` families hold the 1.7 guarantees under CI-gated budgets.

See the [CHANGELOG](CHANGELOG.md) for the complete 1.7.0 notes.

### ✅ 1.8 — Provider/model rotation & swap regression: the migration safety net (vs LiteLLM Router, Braintrust/LangSmith model experiments) (shipped)

*A model swap is the single most common and the single riskiest change in production, and today Vincio's failover and cascade make it blindly — substituting whatever is healthy, never asking whether the replacement can serve the request or whether it quietly regresses quality, cost, or behavior. 1.8 turns the 1.7 model registry into a rotation-and-regression discipline: capability guards refuse to substitute a model that can't actually serve the request, a `SwapGate` replays golden traces and runs an eval + cost + latency + behavioral diff with statistical backing on every candidate, a shadow provider and a capped canary qualify a model on live traffic without touching the user, and a lifecycle watcher proposes migrations off deprecated models — every piece reusing the registry, replay executor, `ab_test`, `DriftMonitor`, and cost model from 1.7. Additive behind `@experimental` entry points on the frozen 1.0 API; nothing changes for callers who don't opt in.*

- **Capability-aware routing preflight + cost/latency router** — before any substitution (a `ModelCascade` step, a `FailoverChain`/`HealthAwareFailover` entry, or a router pick), intersect the request's needs (vision parts, tool calling, structured-output mode, reasoning, required context length) with the registry's `ModelCapabilities` and **skip or escalate** incompatible candidates instead of erroring or silently dropping content. A registry-backed `Router` provider (`optimize/routing.py`) then picks the cheapest / fastest / least-busy *capable* model per request, load-balances across equivalents, and can **downgrade tier to honor a per-request budget** — reusing the registry's pricing and capabilities so routing is one read over the same `ModelRegistry`, recorded as a routing decision on the trace.
- **`SwapGate`: replay + eval + diff + gate on every model change** — `app.gate_swap(...)` / `vincio providers regress` assemble a `SwapGate` (new `evals` swap module) that, on any model/provider change, runs the 1.7 `ReplayRunner` over golden traces, then `evaluate_gates` + `DriftMonitor` + `ab_test`, emitting per-case win/lose/tie, quality/cost/latency deltas, **behavioral shape diffs** (tool-call rate, refusal rate, output-length distribution), and a PASS/FAIL migration verdict with p-value and effect size. It wires into the cascade/router/failover defaults so a model is **only promoted into the live path if it clears the gate** — the swap isn't merely compared, it is gated, all on one trace.
- **Model-swap regression command + flakiness control** — `vincio eval regress --baseline-model X --candidate-model Y dataset.jsonl` holds prompt, data, and config fixed, swaps only the model, and reports per-metric significance + per-case deltas + the cost/latency trade and the **worst-regressed slices**, reusing `Experiment.arun_variant` + `ab_test`. `EvalRunner` gains `repeats=N` with per-case mean/stdev and configurable aggregation, plus a **flake-quarantine** tag so non-mock provider variance doesn't make the gate noisy — turning "is the cheaper model safe?" into a statistically grounded answer rather than one noisy run.
- **Shadow provider + progressive canary with auto-rollback** — a `ShadowProvider` wrapping `(primary, candidate)` returns the primary's response to the user while **asynchronously dual-dispatching** to the candidate and recording both to traces for offline diff; a `CanaryRouter` ramps a configurable percentage of live traffic to a candidate, feeds online metrics (`evals/online.py`) into the `SwapGate` continuously, and **auto-rolls-back to the last known-good registry version** on regression (rollback-as-new-head in `prompts/registry.py`). Both implement `ModelProvider`, so they nest cleanly inside `CircuitBreaker`/`KeyPool`. (Canary-driven prompt/policy promotion, which needs a new serving surface, is reserved for 2.0; 1.8 ships the observe-and-revert provider-layer form.)
- **Lifecycle watcher + auto-migration proposal** — a watcher reads the registry's deprecation/retirement dates and price/quality data to emit **early sunset warnings** as a pinned model nears retirement and to propose a migration (retired→successor, or to a cheaper Pareto-dominating model), runs it through the `SwapGate` and optionally a canary, then offers to rewrite `RoutingPolicy` / `ModelCascade` / `config.model` through the existing flywheel/promotion machinery (`optimize/loop.py`). `FailoverChain` learns to classify **terminal lifecycle/config errors** distinctly from availability errors, so a retired-model 404 surfaces "rotate now" instead of being buried in "all providers failed."
- **Live model discovery + Google/Vertex batch parity** — optional runtime discovery from provider model-list endpoints (OpenAI `/v1/models`, Anthropic `/v1/models`, Gemini `ListModels`, Ollama `/api/tags`, vLLM `/v1/models`, OpenRouter `/models`) reconciled into the `ModelRegistry` with deprecation flags via `providers/base.py` `list_models`, **offline-safe** with the shipped data file as fallback so local/gateway deployments stay current. A Gemini/Vertex batch backend joins `providers/batch.py`, extending the half-cost offline path to Google and **completing batch parity** with OpenAI/Anthropic for the eval/regression workloads the `SwapGate`'s replay leans on.
- *Interconnection:* rotation and regression are pure composition of 1.7 organs — capability guards and the router read the `ModelRegistry`; the `SwapGate`, the model-swap regression flow, and the canary reuse the `ReplayRunner`, `ab_test`, `DriftMonitor`, `evaluate_gates`, and the cost model; the lifecycle watcher feeds the same `ImprovementLoop` promotion path; and every swap decision lands on one trace and the same hash-chained audit chain as every other decision.
- *Edge over specialists:* LiteLLM Router and OpenRouter route as a proxy hop and the experiment trackers (Braintrust/LangSmith) only compare; Vincio is the only layer that routes by **capability *and* cost inside your audit boundary**, refuses capability-mismatched substitutions that today produce silent wrong answers on failover, **gates every swap** on replayed golden traces with statistical backing, qualifies it on live shadow/canary traffic with **automatic rollback**, and proposes the migration off a sunsetting model — one closed loop, in your process, not a config edit after the 404s start.
- *Definition of done (delivered):* capability preflight, cost/latency routing, the `SwapGate`'s pass/fail diff, model-swap regression with flake quarantine, shadow dual-dispatch, canary auto-rollback, and the lifecycle migration proposal are all covered offline against the deterministic mock and recorded provider cassettes; gated by the VincioBench **reliability** (capability-guard correctness, lifecycle-error classification), **cost** (routing cost/latency trade, Google/Vertex batch parity), **evals** (swap-gate significance, replay-diff fidelity), and **scale** (canary rollback under load) families, with the SLOs extended and a new runnable example (`32_swap_regression.py`) that swaps a model end to end through the gate and a canary.
- **1104 tests passing offline in ~5s; ruff + mypy clean**; thirty-two runnable examples; the VincioBench `reliability` / `cost` / `evals` / `scale` families hold the 1.8 guarantees under CI-gated budgets (159 budgets, 48 SLOs). Residency is enforced at the run boundary over every reachable model (router/cascade/canary candidates and budget-degrade targets), and the Google batch wire format is covered by recorded-cassette tests — the milestone carries no deferred follow-ups.

See the [CHANGELOG](CHANGELOG.md) for the complete 1.8.0 notes.

### ✅ 1.9 — Documents & images flow OUT — cited, governed, eval-gated artifacts (vs Unstructured/python-docx writers, gpt-image-1, OpenAI/Gemini TTS, C2PA) (shipped)

*Vincio can read a DOCX, a PDF, and a scanned KYC packet, and validate a JSON answer — but it stops one step short of the deliverable. The `output` module, despite its name, only validates model **text**; there is no way to produce a cited board memo, a filled form, a redline, or a generated image, and the C2PA/eval/budget machinery Vincio applies to text never touches generated media. 1.9 closes the documents-in/documents-out and images-in/images-out loops: a document **generation** engine that turns a validated, cited result into DOCX/PDF/PPTX/HTML/Markdown artifacts; an image-generation/editing and TTS provider abstraction so media is a first-class output modality; OCR auto-wired into the loaders and audio finally ingestible; and every produced asset stamped with provenance, metered against budget, and gated by eval. Additive behind new `vincio.generation` / provider subpackages and opt-in extras on the frozen 1.0 API (`@experimental`); the deliverable closes the same loop the audit chain opened.*

- ✅ **Document generation engine — cited DOCX/PDF/PPTX/HTML/Markdown** — a new `vincio.generation` subpackage whose `DocumentBuilder` turns a validated `OutputContract` result (or a `RunResult` / structured mapping / Markdown) into rendered artifacts: Markdown/HTML dependency-free, DOCX via python-docx, PDF via reportlab, PPTX via python-pptx behind extras (`vincio[gen-docx|gen-pdf|gen-pptx]`). Because it consumes a *validated* result, the document is grounded by construction; it supports **structural document contracts** (`DocumentContract`: required sections, `TableSpec` column specs, length bounds, citation-per-section) with formatting-only repair (`repair_formatting`) mirroring the JSON-repair path, and records every render in the hash-chained audit log as a `document_generate` event carrying the source evidence ids. Adds template/form filling (`fill_text_template` / `fill_docx_form` / `fill_pdf_form` with typed citation-aware `Slot`s, DOCX merge fields, PDF AcroForm) and a `generate_redline` generator pairing the existing `DOCUMENT_COMPARISON` intent with tracked-change DOCX (and `**ins**`/`~~del~~` text) output.
- ✅ **Cited-report assembly with resolved citations & bibliography** — a `CitedReportBuilder` takes a validated answer plus its `EvidenceItem`s and renders inline `[E1]`-style markers resolved to numbered footnotes/endnotes, a generated bibliography, and per-claim provenance (trust level, `source_uri`, page/section), output to any `DocumentBuilder` format. Adds **field/claim-level citation contracts** (`CitationContract`: coverage floor, no-unresolved-markers, entailment), sentence-level citation-coverage metrics (`evals.metrics.citation_coverage`), and an optional NLI/entailment check (`claim_entailment`, pluggable backend) that the cited evidence actually *supports* the claim — replacing the flat "one valid citation anywhere" membership check. Citation extraction reuses `output/parsers.extract_citations`, so the resolution apparatus reads the same markers the validator already trusts.
- ✅ **Image generation/editing provider abstraction with provenance** — adds `generate_image` / `edit_image` / `variation` as a new capability surface (`ImageProvider`) over OpenAI `gpt-image-1`, Gemini/Imagen, and an HTTP/Replicate adapter, with a neutral `ImageGenRequest` (prompt, size, n, quality, mask, reference images, seed) and `ImageGenResponse` (image bytes, `revised_prompt`, usage, cost), plus a `MockImageProvider` that emits real PNGs for offline tests. Every generated asset auto-attaches a media-aware C2PA manifest bound to its bytes, is metered against the budget (`meter_media_cost`), and is audited. `governance.transparency.mark_synthetic_content` is now **media-aware** (accepts bytes, `compositeWithTrainedAlgorithmicMedia` for edits, plus `embed_provenance` for PNG metadata and `write_sidecar_manifest` for any format with an invisible-watermark hook) so the marker binds images/audio, not just `str`.
- ✅ **TTS / speech-synthesis output modality** — a `synthesize_speech` surface (`SpeechProvider`) with a neutral voice/format/speed model over OpenAI TTS, Gemini TTS, and ElevenLabs/Cartesia plus a deterministic mock that emits real WAVs; outputs are marked with audio provenance and metered against budgets, unified with the realtime audio path so synthetic speech is governed exactly like every other output. Wires `ContentPart.audio` through the OpenAI (`input_audio`) and Gemini (`inlineData`) chat providers via a shared `encode_audio_bytes`, so the already-typed `AudioRef` is usable as **input** outside the realtime WebSocket path — activating an input type the type system exposed but never accepted on the chat path.
- ✅ **OCR auto-fallback, audio transcript ingestion & richer inputs** — wires OCR into the loader: `load_pdf` detects low text-yield pages and routes them through a supplied `OCREngine` (rasterize via pypdfium2 → Tesseract/`VisionModelOCR`), recording `extractor='ocr'` per page so provenance stays honest. Adds `load_media(path, transcriber=...)` producing a timestamped, optionally speaker-diarized transcript `Document` via a `Transcriber` protocol mirroring `OCREngine` (Whisper/provider-audio backends + a deterministic mock), turning the dead "audio" file classification into a real ingestion path. Crops `LayoutFigure` regions into citable evidence with bounding boxes (`figure_evidence`); structures JSON/JSONL/YAML into sections/tables (`structure_data`); and adds a real-parser HTML path with table extraction (`parse_html`).
- ✅ **New input formats + forms/KYC structured extraction** — dependency-free loaders for PPTX, EPUB, RTF, ODT, plus Parquet/Arrow (as `TableData`, `vincio[parquet]`), mbox, and `.msg` (`vincio[msg]`), each yielding sections/tables consistent with `TableData`, plus a unified **parser registry** (`documents/registry`, `register_loader`) that replaces the if/elif suffix chain so formats register additively. Adds a forms-extraction path (Textract / Azure Document Intelligence / Google Document AI adapters behind a `DocumentAI` protocol, plus an offline `HeuristicFormExtractor`) returning structured `FormField`s with bbox + confidence as citable evidence (`form_fields_to_evidence`) for the dominant invoice/receipt/ID use-case — unifying the classifier's promises with the loader's reality.
- ✅ **EU AI Act conformity pack — Annex IV docs as generated, cited artifacts** — built as the document-generation engine's first governance application. A `RiskTierClassifier` places a configured `ContextApp` into the Act's risk tiers (prohibited / high-risk / limited / minimal) from its declared purpose, data sources, and human-oversight controls (advisory; the operator decides); an `AnnexIVBuilder` renders the **Annex IV technical documentation** as a cited document through the 1.9 `DocumentBuilder`, drawing every field from the live config, the model/system cards, the compliance matrix, and the eval/red-team evidence Vincio already holds (grounded by construction); an Article 27 **FRIA** (`FRIAGenerator`) and an **ISO/IEC 42001** control catalog join the `ComplianceMapper` (`governance/frameworks.py`) family. Deadline-agnostic and pluggable like the existing card formats; the pack is a *view* over the running system, regenerated on every config change, recorded as a `conformity_doc` audit event (`app.risk_tier` / `app.annex_iv` / `app.fria`).
- *Interconnection:* generation rides the existing organs in reverse — the `DocumentBuilder` consumes a validated `OutputContract` result and the same `EvidenceItem`s the compiler cited; produced documents and generated media land on the same hash-chained audit log (`document_generate` / `image_generate` events) carrying source evidence ids, are metered by the same budget, and gated by the same eval families, so a generated deliverable is as auditable as an ingested source. OCR'd pages, transcripts, and figure crops enter the same retrieval/citation pipeline as a local file — no new bookkeeping, no parallel ledger.
- *Edge over specialists:* the writers (Unstructured + python-docx/reportlab/WeasyPrint/python-pptx) give you a DOCX, the image APIs (gpt-image-1/Imagen) give you a PNG, LlamaIndex's `CitationQueryEngine` gives you `[E1]` markers, standalone TTS gives you audio; Vincio gives you a **cited, structurally-validated, provenance-stamped, budget-metered, eval-gated deliverable** — documents and images flow OUT under the exact same guarantees Vincio applies to text IN, all on one trace and one audit chain, in-process and not a service. It is the only path where a generated image is C2PA-stamped and a cited report is per-claim *entailed*, not just `[E1]`-marked.
- *Definition of done (delivered):* document generation in every format, structural document-contract validation, cited-report citation resolution + per-claim entailment, image-gen/TTS with media C2PA provenance, media-aware synthetic-content marking, audio chat input, OCR auto-fallback, audio transcript ingestion, figure-to-evidence, the new format loaders + parser registry, forms/KYC extraction, and the EU AI Act conformity pack (risk tier + Annex IV + FRIA + ISO/IEC 42001) are all covered offline against mocks; gated by a new VincioBench **generation** family (document-contract validity, cited-report coverage + entailment, media-provenance binding + disclosure, redline correctness, new-format ingestion recall, generated-media prompt safety) alongside the **governance** family (ISO/IEC 42001 mapping), with `examples/33_documents_and_media_out.py` and three new SLOs. Depends on 1.7.
- **1196 tests passing offline; ruff + mypy clean**; thirty-three runnable examples; the VincioBench `generation` / `governance` families hold the 1.9 guarantees under CI-gated budgets (173 budgets, 51 SLOs). Everything is additive behind the new `vincio.generation` subpackage, the `vincio[gen-docx|gen-pdf|gen-pptx|ocr|parquet|msg]` extras, and `@experimental` markers on the frozen 1.0 API — the forms cloud Document-AI adapters (Textract/Azure/Google) are real dependency-injected implementations and embedded PNG C2PA credentials are self-verifying, so the milestone carries no deferred follow-ups.

See the [CHANGELOG](CHANGELOG.md) for the complete 1.9.0 notes.

### ✅ 1.10 — The loop closes itself: continual, online, safe self-improvement & the agentic frontier (vs DSPy 3 SIMBA, OpenAI/Gemini Deep Research, Letta/MemGPT, Anthropic computer-use) (shipped)

*Vincio can already measure drift and run an offline optimizer, but today the loop only closes when a human presses go: the `OnlineEvaluator` samples and persists, `DriftMonitor` emits an event into the void, the routing bandits are unwired primitives, and the GEPA reflector is a fixed heuristic table whose LLM path is a shim. 1.10 makes self-improvement continual, online, and safe — a controller that turns live drift into a gated re-optimization or a rollback, a real provider-backed reflector that reads the actual failing cases, guarded online bandits with auto-rollback, and persisted online state — and opens the agentic frontier the field now expects (deep research, computer-use, self-editing memory) on top of the same cited, grounded spine, all behind hardened isolation. Additive behind `@experimental` entry points on the frozen 1.0 API; the canary-driven prompt/policy promotion that needs a new serving surface stays reserved for 2.0.*

- **Online improvement controller** — `app.continuous_improvement(...)` subscribes to `drift.detected` + `eval.online` on the existing event bus, debounces with per-trigger cooldowns and a global eval budget, and turns a sustained signal into one of three *gated* actions: a fresh `ImprovementLoop` run, a targeted re-eval, or a rollback to the last known-good `prompts/registry.py` version — closing the observe-only online stack into a real loop without touching the optimizer internals. The `OnlineEvaluator` sampling counter and bandit/online-learner state persist to the shared store so continual learning is restart-safe and aggregatable across workers, and `evals/drift.py` gains KS/PSI/MMD distributional drift plus a CUSUM changepoint detector feeding the controller. Every trigger, debounce, decision, and rollback lands on the audit chain and one trace.
- **Real provider-backed reflective optimizer (GEPA proper)** — a first-class `LLMReflector` in `optimize/reflective.py` wired to the app's own provider reads actual failing cases (input + output + expected + the evidence that grounded them), clusters them into failure modes, and proposes targeted edits validated against the existing edit schema — with `HeuristicReflector` kept as the air-gapped, deterministic fallback so offline runs stay reproducible. This realizes the informed-proposal advantage the module's docstring has always promised but the canned-rule floor map never delivered, and feeds the same Pareto frontier and gated promotion already shipped in `optimize/loop.py`.
- **Autonomous experiment proposer + guarded online bandits** — a meta-controller in `optimize/loop.py` ranks where the system is weakest from online eval + drift and proposes/schedules the highest-ROI experiment (prompt vs. retrieval vs. budget vs. routing vs. distillation), spending a global eval budget across candidates with every decision recorded. The `EpsilonGreedy` / `UCB1` bandits in `optimize/routing.py` (joined by a contextual `LinUCB`) are wired into the live route behind a **safety floor** — never explore on safety- or high-risk-tagged traffic — with persisted arm stats, per-arm regret, and auto-freeze/rollback on regression, turning dead primitives into a real, safe online learner. A held-out, *growing* golden regression suite in `evals/datasets.py` gates every promotion with provenance replay, so sequential auto-promotions can never silently undo a prior fix.
- **Deep-research agent — budgeted, citation-gated, eval-scored** — a first-class `ResearchAgent` loops search→read→reflect→verify→synthesize over the existing query-understanding planners (`retrieval/query_understanding.py`: HyDE / multi-query / decompose / step-back) and the grounded-fact extractor (`memory/facts.py`), under explicit breadth/depth budgets, with source dedup and verification reusing `evals/judges`, emitting a cited report through the 1.9 `CitedReportBuilder` and the compiler's budgeting — scored by a new research-quality eval family. Mostly composition of organs Vincio already ships, exposed behind one new `@experimental` entry point.
- **Agent memory OS + in-loop context compaction** — memory operations become first-class permissioned tools (`memory_append` / `memory_replace` / `memory_search` / `memory_archive`) over the existing audited write pipeline in `memory/engine.py`, with a context-pressure controller that pages between in-context core memory and the archival store — a MemGPT/Letta-class self-editing memory, but provenance-tracked and audited. The shipped `memory/summarizers.py` are wired into the ReAct/DAG loop so old tool/observation turns are compacted (rolling summary + observation pruning) when a token budget is hit, replacing the fixed `[-8]` / `[:24]` slicing in `agents/executor.py`; the agent DAG runs level-parallel over `StepDAG.topological_levels`, and `agents/planner.py` gains a real `plan_and_execute` replanning loop.
- **Computer-use / agentic browsing behind hardened isolation** — a browser/computer action vocabulary (navigate / click / type / screenshot) joins the tool family and the agent action types in `agents/executor.py`, with Playwright and provider-native computer-use (Anthropic / OpenAI) backends and a deterministic mock for offline tests, all flowing through the permissioned, audited, budgeted tool runtime. It is gated on a new pluggable `IsolationBackend` in `tools/sandbox.py` (container/Docker, microVM/Firecracker, gVisor, WASM) behind the existing sandbox interface — subprocess + `setrlimit` stays the zero-dep default, but code-executing and computer-use workloads *require* real isolation. Provider-native hosted tools (`web_search` / `file_search` / `code_interpreter` / `computer_use`) are surfaced as namespaced Vincio tools through `providers/openai_responses.py`, riding the same RBAC, audit, and budget path as any local tool.
- *Interconnection:* the continual loop is composition of existing organs — the controller subscribes to the same event bus and reuses the `ImprovementLoop`, `ab_test` gates, `DriftMonitor`, and prompt-registry rollback; the real reflector feeds the same Pareto frontier; deep research reuses query understanding, grounded-fact extraction, judges, and the 1.9 cited-report renderer; memory-as-tools rides the existing audited write pipeline; computer-use rides the existing permissioned, sandboxed, audited tool runtime — so every new action and every self-update lands on one trace and the one audit chain, against the same cost model.
- *Edge over specialists:* the field gives you a self-improving module (DSPy 3 SIMBA), a deep-research product (OpenAI/Gemini/Perplexity Deep Research, GPT-Researcher), a memory OS (Letta/MemGPT), or a computer-use tool (Anthropic computer-use, OpenAI Operator) as four separate things; Vincio gives you continual self-improvement that is **gated and reversible**, deep research where **every claim is cited and budget-bounded**, self-editing memory that is **provenance-tracked and audited**, and computer-use that is **isolated and audited** — all on the same packet, ledger, audit log, and trace, in-process and not a service, with the loop finally closing itself under a held-out non-regression guard the field's offline-only bandits and thin GUI adapters structurally lack.
- *Definition of done (delivered):* gated by the VincioBench `loop`, `agentic_evals`, `agent`, `security`, and `memory` families — drift-triggered gated re-optimization/rollback correctness and held-out non-regression in `loop`; reflector failure-mode diagnosis and deep-research citation/grounding/budget in `agentic_evals`; level-parallel DAG and `plan_and_execute` replanning in `agent`; `IsolationBackend` enforcement and hosted-tool permissioning in `security`; memory-as-tools provenance and in-loop compaction in `memory` — all covered offline against mocks, with `examples/34_continual_loop_and_agentic_frontier.py` and nine new SLOs, and persisted online state proven restart-safe and worker-aggregatable.
- **1304 tests passing offline in ~5s; ruff + mypy clean**; thirty-four runnable examples; the VincioBench `loop` / `agentic_evals` / `agent` / `security` / `memory` families hold the 1.10 guarantees under CI-gated budgets (205 budgets, 60 SLOs). Everything is additive behind the `vincio.optimize.controller` / `vincio.agents.research` / `vincio.tools.computer_use` / `vincio.providers.hosted_tools` entry points, the `IsolationBackend` family in `tools/sandbox.py`, the `vincio[computer-use]` extra, and `@experimental` markers on the frozen 1.0 API — the canary-driven prompt/policy promotion that needs a new serving surface stays reserved for 2.0.

See the [CHANGELOG](CHANGELOG.md) for the complete 1.10.0 notes.

### ✅ 2.0 — The one breaking window: the structural refactor, async-first stores, the multimodal-native Context Packet & enterprise endpoints (vs LiteLLM/Bedrock/Vertex, the OTel GenAI agentic conventions) (shipped)

*Five milestones of additive growth exposed the structural debt the frozen 1.0 surface could not pay down: a ~2235-line `ContextApp` god-object that couples every feature, sync-only storage and event contracts called from async code, a text-only compiler spine that can't make images first-class candidates, a Python-callable `SearchFilter` that can't push down to a vector DB, and an `HTTPProvider` that assumes static api-key auth and so locks out Bedrock/Vertex/Azure — the very enterprise endpoints the 1.6 governance buyer runs on. 2.0 is the single deliberate breaking window, and nothing breaks outside it: it collapses the deprecated aliases accumulated across 1.x, decomposes the god-object into facades, makes the async store/event/metric contracts canonical, lands a multimodal-native Context Packet as the flagship capability that genuinely needs schema changes, pushes structured filters down to every backend, adds enterprise endpoints behind a pluggable auth strategy, and adopts the finalized OTel GenAI agentic conventions and Pydantic/Python floor bumps — every change one Vincio could not make additively, each retired through the same mechanical deprecation runway 1.0 established.*

- ✅ **Decompose the `ContextApp` god-object into facades** — split the ~2235-line `ContextApp` (`core/app.py`) into a thin facade over lazily-constructed capability modules — `RunFacade`, `RetrievalFacade`, `GovernanceFacade`, `OptimizationFacade`, `ServingFacade`, `TrainingFacade` — so the runtime hot path (`core/runtime.py`) depends on narrow interfaces, cold start and memory footprint scale with what the app actually constructs, and the import graph shrinks. Deprecated attribute-access aliases (`app.<old_method>`) bridge the move under the mechanical deprecation policy with a removal version named per call; the breaking reshuffle is the point, and it makes every feature decoupled and testable in isolation.
- ✅ **Async-first storage, a typed/versioned event catalog & unified telemetry** — make the async store/index protocol (`asave`/`aquery`/`asearch`) the canonical contract with thin sync shims, add a psycopg3-async connection pool for Postgres (`storage/postgres.py`), and remove blocking DB I/O from the run path. Promote events from stringly-typed names + free-form dicts to a typed, versioned event catalog (`core/events.py`) with documented Pydantic payload models, so observers and external sinks bind to a stable schema. Re-architect telemetry (`observability/otel.py`, `observability/finops.py`) so token usage and cost are first-class signals emitted once to spans, OTel metrics, and the cost ledger together — adopting the finalized OTel GenAI **agentic** semantic conventions (`gen_ai.agent.*`, `invoke_agent`/`execute_tool` spans, token/duration histograms) even where they rename current attributes.
- ✅ **Multimodal-native Context Packet (the flagship breaking capability)** — generalize `ContextCandidate.content` and `EvidenceItem` (`core/types.py`, `context/ir.py`) from `str` to typed parts (text/image/table) with modality-aware `token_cost`, scoring (`context/scoring.py`), budgeting, and packet rendering (`context/packet.py`, `context/compiler.py`), so images, tables, and screenshots become first-class candidates the compiler selects, dedupes, orders, and cites alongside text — not bolted-on observations appended after the fact. Back slim packets with a persisted content-addressed evidence store so `materialize()` works after deserialization (zero-copy cross-process packet shipping), and populate the evidence-ledger `supports` links via entailment for a real claim/contradiction graph. This is the one capability that genuinely requires the schema break the window is reserved for.
- ✅ **Structured `FilterSpec` with native pushdown to every backend** — introduce a declarative, serializable `FilterSpec` (`eq`/`in`/`range`/`and`/`or` over chunk fields + metadata) alongside the existing Python-callable `SearchFilter`, and compile it to each backend's native filter — Qdrant `Filter`, pgvector `WHERE` on `jsonb` with a GIN index + HNSW DDL, Pinecone metadata filter, Weaviate `where`, Milvus `expr`, ES `bool`/`term` (`storage/vectorstores.py`, `storage/qdrant.py`, `storage/postgres.py`, `retrieval/engine.py`). This fixes both the client-side **over-fetch under-fill** correctness bug (selective filters silently starve `top_k`) and the cross-tenant **fetch-to-filter exfiltration** risk by pushing tenant/ACL scope (`security/access.py`) into the engine. Breaking because the `Index.search` `where` parameter type and the `Index` protocol change.
- ✅ **Enterprise endpoints behind a pluggable auth strategy** — add a per-request auth/signing hook to `HTTPProvider` (`providers/base.py`, today static api-key + `Bearer`/`x-api-key` only) and implement **AWS Bedrock** (SigV4-signed `converse`), **Google Vertex** (service-account auth, regional endpoints, batch), and **Azure OpenAI** (deployment-name routing, `api-version`) as new providers registered through `providers/registry.py` — the deployment surfaces the 1.6 governance buyer actually runs on. Refactoring `_headers`/`_post_json` into an `AuthStrategy` is a breaking interface change for `HTTPProvider` subclasses, hence the window; the payoff is enterprise endpoints inside the same registry, capability guards, swap gate, residency, and audit chain as every other provider.
- ✅ **Mandatory egress DLP, a signed audit chain & breaking eval semantics** — a deterministic last-mile **DLP scan** of the fully-assembled provider request (system + messages + tool schemas) for PII/secrets/residency, enforced at the provider boundary independent of call-site wiring (always-on, hence breaking for some flows) in `security/policy.py`; **HMAC/asymmetric per-entry audit-chain signatures** plus periodic Merkle-root export (`security/audit.py`) for tamper-*evidence* against a privileged attacker, which changes the audit-entry schema and `verify` semantics; and the **eval metric refactor** (`evals/metrics.py`, `evals/reports.py`) that stops returning a neutral `1.0` for unscoreable cases (returns `skipped`/`None`, excluded from gate aggregation) and renames `semantic_similarity` to its true lexical identity, reserving the name for a real embedding-backed metric.
- *Interconnection:* the refactor preserves the principle of **one packet / ledger / audit / trace** while making the contracts that carry them async-first and multimodal-native — the facades read and write the same Context Packet; the multimodal packet flows through the same compiler, budget, and citation path; `FilterSpec` rides the same retrieval engine and tenant scope; enterprise endpoints register in the same `ModelRegistry` and pass through the same swap gate, residency, and audit; the signed audit chain and unified telemetry are the same organs, made canonical rather than added beside.
- *Edge over specialists:* 2.0 is not a rewrite — it is the one window where the frozen surface's structural debt is paid down with the mechanical runway 1.0 established. It beats **LiteLLM's** provider breadth by routing Bedrock/Vertex/Azure through the same governance boundary instead of a separate proxy; beats the **gateways' proxy telemetry** by making cost and tokens first-class signals on one async-safe contract stitched into distributed traces via the standard **OTel agentic conventions**; beats **GPT-4o/Claude vision** and visual-RAG stacks by selecting, budgeting, deduping, ordering, and citing image/table evidence in the *same scored packet* as text; and beats **post-filter adapters and unsigned audit logs** by pushing tenant/ACL filters into the engine and making the chain tamper-evident — every specialist-beating capability the prior surface structurally could not land additively.
- *Definition of done (delivered):* facade decomposition into six lazily-constructed capability views, the canonical async store contract (`AsyncMetadataStore` + `aget`/`adelete`/`acount` and a psycopg3 async pool), the typed/versioned event catalog + OTel agentic conventions (`invoke_agent`, `gen_ai.agent.*`, cost histograms), multimodal-native packet selection/budgeting/citation with cross-process `materialize()` from a content-addressed store, `FilterSpec` native pushdown (Qdrant + pgvector server-side) with shared-or-mine tenant-scope correctness, the three enterprise endpoints behind `AuthStrategy`, always-on egress DLP, and the signed Merkle-checkpointed audit chain are all covered offline and gated by the new VincioBench **breaking_2_0** family, with `examples/35_breaking_window_2_0.py` and new SLOs.
- **1389 tests passing offline in ~5s; ruff + mypy clean**; thirty-five runnable examples; VincioBench holds the 2.0 guarantees under CI-gated budgets (18 families, 218 budgets, 65 SLOs). The flat `app.<method>` API remains fully supported alongside the facades; the public-API contract moves to `2.0`. Native filter pushdown ships server-side for **every** named backend — Qdrant and pgvector (structured/`jsonb`), plus Pinecone, Weaviate, Milvus, and Elasticsearch/OpenSearch, which (since 2.0.1) persist flat filterable fields alongside the chunk blob and pass the compiled `FilterSpec` into the backend's native query (each verified offline against its fake). **The 2.0 milestone carries no deferred items.**

See the [CHANGELOG](CHANGELOG.md) for the complete 2.0.0 notes.

### ✅ 2.1 — Scale out & train for real — distributed execution, executed fine-tuning, served observability (vs Temporal/Ray, DSPy BootstrapFinetune, Langfuse/Phoenix) (shipped)

*With the 2.0 contracts async-first and the god-object split behind us, horizontal scale and a real — still self-hosted — operational plane become additive again. 2.1 turns the no-lock-in backend story from export-only into distributed run, turns the distillation flywheel from data-prep-plus-a-gate into an executed cheaper model, and turns the static HTML trace viewer into a served observability and alerting plane (yours, never SaaS), plus the Redis-backed shared state that multi-worker serving needs. All additive behind `@experimental` on the 2.0 surface; the single-process asyncio path stays the default and nothing here is required to run Vincio.*

- ✅ **Distributed durable-execution backend** — `WorkerPoolBackend` (the in-process reference distributed executor) plus `RayBackend` / `TemporalBackend` export adapters (`agents/backends.py`) run the same `StateGraph` / `Workflow` / `Crew` across a worker pool with cross-restart durability. Concurrent resumers are made safe by optimistic-concurrency on the checkpointer (checkpoint-version CAS) plus a TTL `running` lease on each graph thread (`agents/distributed.py`: `GraphCoordinator`, in-memory + `RedisGraphCoordinator`, `DistributedCheckpointer`), so two workers can't double-execute a step — the loser raises `CheckpointConflictError`. `agents/graph.py` gains true BSP parallel super-steps (`compile(parallel=True)`) and `Send` map-reduce fan-out; `workflows/engine.py` gains a `map_step` for data-dependent level-parallel spawning. Single-process asyncio remains the default and the lease/CAS metadata rides the same checkpoint records, so a run moves between backends without losing its evidence ledger or trace.
- ✅ **Executed distillation & provider fine-tune jobs** — `providers/finetune.py` ships `OpenAIFineTuneBackend` / `GoogleFineTuneBackend` / `AnthropicFineTuneBackend` that submit and poll real fine-tune jobs; `optimize.provider_trainer` graduates the `StudentTrainer` from a no-op into an executed trainer that trains, registers the resulting model in `providers/registry.py`, and lets `BootstrapFinetune` gate-promote it through the existing significance swap gate — so the flywheel produces an actual cheaper model, not just grounded JSONL. The export gains semantic dedup (`semantic_dedupe`) and a truncation guard (`max_example_chars`) so the training set stays diverse and faithful. Offline, the job lifecycle runs against cassette-backed (`httpx.MockTransport`) submit/poll/status and the promotion decision is fully deterministic.
- ✅ **Served (self-hosted) observability & alerting plane** — `observability/store.py` (`IndexedTraceStore`) is an indexed SQLite trace/cost store with time-bucketed pre-aggregates, retention (`purge`), and rollups that replace the O(n) JSONL scans; `observability/viewer.py` (`ViewerApp` + `serve_viewer`, stdlib-only) serves a live trace tail, attribute/tenant/model search, latency and cost p50/p95/p99, and cost-by-dimension dashboards. An `AlertSink` protocol (webhook / Slack / PagerDuty / Prometheus, `observability/exporters.py`) plus a rule engine (threshold, EWMA/Welford anomaly, SRE burn-rate; `AlertManager`/`AlertRule` in `observability/finops.py`) runs over the same cost ledger and event bus, with tail-based error-prioritized sampling (`TailSamplingExporter`). The zero-dependency static viewer stays; this plane is opt-in, runs on your infrastructure, and emits on the same audit chain — never a hosted service.
- ✅ **Redis-backed shared server state + content-capture controls** — `RedisRateLimiter` / `RedisIdempotencyStore` (`storage/redis.py`) over the shared-state protocols (`storage/shared_state.py`: in-memory defaults + `TenantQuotaManager`) keep multi-worker uvicorn deployments coherent; a first-class `vincio serve` launcher (`cli/main.py`) closes the audit finding that server mode shipped but was unlaunchable, adding health/readiness/metrics endpoints, a Prometheus exposition, and graceful shutdown. Prompt/completion content capture is gated behind a `ContentCapturePolicy` (off by default) with truncation and PII-redaction applied at the export boundary (`observability/otel.py`, `tools/runtime.py`) before any content reaches OTel events, JSONL, or the viewer — so the served plane never widens your data-exposure surface.
- ✅ **Quantization + two-stage retrieval & batteries-included local neural models** — binary/scalar quantization and Matryoshka two-stage retrieval (coarse search on truncated/quantized vectors, exact rerank on full precision) ship as `retrieval/quantization.py` (`TwoStageIndex`, `quantize_scalar`/`quantize_binary`, reusing `mrl_truncate`), with native quantization config wired into the Qdrant adapter (`storage/qdrant.py`). Optional-dependency real local models arrive batteries-included: a fastembed/ONNX dense embedder (`FastEmbedEmbedder`), a local cross-encoder reranker (`LocalCrossEncoderReranker`), a real SPLADE encoder (`SpladeEncoder`), and a ColBERT token-embedder (`ColBERTTokenEmbedder`) in `retrieval/`, plus a native llama.cpp/GGUF in-process provider with on-device embedding (`GGUFProvider`, `providers/local.py`) — each with a deterministic offline fallback, so air-gapped and edge deployments get semantic quality and true offline inference behind the same `Embedder` / `Reranker` / `Index` / provider interfaces.
- *Interconnection (held):* scale rides the 2.0 async contracts and facades rather than forking them — the distributed backend runs the same `StateGraph` / `Workflow` on the same checkpointer; the trained student promotes through the same significance swap gate as a model rotation; the served plane reads the same indexed trace/cost store and emits on the same event bus; and quantized two-stage retrieval reuses the same `Index` protocol and embedder/reranker interfaces, so distribution and serving are views over the same organs, not a parallel system.
- *Edge over specialists (delivered):* the field sells a durable-execution cloud (Temporal/Ray, LangGraph Platform), a fine-tune toolkit (DSPy BootstrapFinetune), and an observability SaaS (Langfuse/LangSmith/Phoenix) as three separate hosted products. Vincio keeps the single-process path the default and the distributed path lock-free, gates the trained student on the same replay-plus-significance swap gate so the flywheel ships a cheaper model only if it provably doesn't regress, and serves a dashboard inside the same audit chain and cost ledger — distributed run, executed gated fine-tuning, and served observability all on your infrastructure, in one trace, in-process not a control plane.
- **1485 tests passing offline in ~6s; ruff + mypy clean**; thirty-six runnable examples; VincioBench holds the 2.1 guarantees under CI-gated budgets (229 budgets, 71 SLOs): distributed durability + multi-worker shared-state coherence in `scale`, the executed-distillation swap-gate in `loop`, quantized two-stage recall in `rag`, and burn-rate/EWMA alerting in `cost`. Everything is additive behind the `vincio.agents` distributed entry points, `vincio.providers.finetune`, `vincio.observability` (store/viewer/alerts), `vincio.storage.shared_state`, and `vincio.retrieval.quantization` plus the local-model classes, all `@experimental` on the frozen 2.0 surface. **The 2.1 milestone carries no deferred items.**

See the [CHANGELOG](CHANGELOG.md) for the complete 2.1.0 notes.

### 🚧 2.2 — Prove it on the world's benchmarks: environment eval, agentic leaderboards & the agent fabric (vs τ-bench/SWE-bench/WebArena, AGNTCY, generative UI)

*The field now judges agents by task success in stateful environments and on public leaderboards, lets them discover one another through agent directories, and streams structured UI to frontends. With distributed execution and computer-use already shipped at 2.0/2.1, this milestone makes Vincio **measurable** on the benchmarks buyers actually compare on, **composable** into a governed agent fabric, and **embeddable** in interactive products — turning leaderboard numbers into a closed-loop training signal rather than a post-hoc score. Entirely additive behind `@experimental` entry points on the 2.0/2.1 surface, all self-hosted; the benchmark adapters and registry clients use only the core `httpx` dependency, and the reference environments run deterministically in-process.*

- 🚧 **Stateful-environment eval harness + agentic benchmark adapters** — an `Environment` protocol (`reset` / `step` / `observe` / `verify`) in `evals/simulator.py` with deterministic in-process reference environments and a task-success oracle, so the multi-turn simulator drives an agent through a *mutable* world and the existing `Trajectory` metrics (`evals/trajectory.py`) score **verifiable end-state**, not just turn-by-turn plausibility. This turns agentic eval from post-hoc trajectory scoring into a closed-loop training signal. Ship adapters in `benchmarks/vinciobench.py` to run Vincio agents against **SWE-bench Verified, τ-bench/τ²-bench, GAIA, WebArena, and BFCL** behind the VincioBench runner, so agents earn market-recognized scores that the flywheel can optimize against — each adapter pinning its task set by hash for reproducibility and degrading to a recorded-fixture replay offline.
- 🚧 **Retrieval evaluation harness (recall@k / nDCG / MRR / context-precision)** — a golden-set harness scoped to retrieval that quantitatively benchmarks embedder / reranker / chunker / index configs, with versioned **data/index regression artifacts** keyed on `(embedder, chunker, corpus hash)` in `storage` so a re-embed or a chunking tweak that regresses recall is caught against a stable golden query set. Results render through `evals/reports.py` and gate on recall/nDCG deltas using the *same* significance machinery as a model swap, making the offline-vs-real retrieval tradeoffs measurable and optimizable instead of a vibe check.
- 🚧 **Agent registry / discovery (AGNTCY / ACP)** — an agent directory and capability-discovery layer over the existing A2A **Agent Card** (`a2a/protocol.py`, `a2a/client.py`) with an **allow-list governance gate** in `security/access.py`, so orgs control which agents and servers are reachable. Adds an **AGNTCY / ACP** (REST-native Agent Connect Protocol) adapter so Vincio spans both interop camps, plus an **MCP Registry** discovery client so MCP servers resolve from the official registry under that same allow-list. Point-to-point delegation becomes a *governed, discoverable fabric* — every resolution recorded as an access decision on the audit chain.
- 🚧 **Generative UI / AG-UI streaming protocol** — a streaming UI-event protocol (**AG-UI / MCP-UI** compatible) in `server/app.py` so runs drive interactive frontends with structured UI deltas over the existing SSE / `astream` path, plus **token- and tool-event streaming** from `agents/executor.py` and `agents/crew.py` (matching the streaming surface `graph` and `compose` already expose) and UI resources from `mcp/server.py`. The interactive frontend inherits the run's provenance, budget metering, and audit — one streamed run, not a bolt-on UI layer.
- *Interconnection:* benchmarks and the fabric ride the organs that already exist — the environment harness reuses the `Trajectory` model and feeds the **same optimizer / Pareto loop** that tunes prompts, routing, and budgets; the benchmark adapters run inside the VincioBench runner; the retrieval harness reuses `evaluate_gates` and the significance tests; the agent directory extends the A2A Agent Card and the allow-list gate; and generative-UI events stream over the existing SSE / `astream` path and land on **one trace** with the same cost model.
- *Edge over specialists:* the leaderboards (τ-bench/τ²-bench, WebArena, SWE-bench, GAIA, BFCL), the agent directories (AGNTCY/ACP), the retrieval benches (RAGAS/BEIR), and the generative-UI toolkits (CopilotKit AG-UI, MCP Apps) are four separate ecosystems. Vincio runs the benchmarks **inside its own bench** and feeds verifiable task success back into the Pareto optimizer; makes the agent fabric **governed and discoverable under one allow-list** across both A2A and AGNTCY; gates retrieval changes on recall deltas as a **first-class CI gate**; and streams UI that **inherits the run's provenance and audit** — measurability, composability, and embeddability on one spine, in your process, not as four services.
- *Definition of done:* the intent is that environment eval with a success oracle, the five benchmark adapters, the retrieval eval harness plus index-version regression, AGNTCY/ACP + registry discovery under the allow-list, and generative-UI streaming are all exercised offline against deterministic environments and recorded mocks — gated by the VincioBench **agentic_evals** (environment task-success and benchmark-adapter determinism), **rag** (retrieval-eval recall/nDCG and index-version regression), **protocols** (AGNTCY/registry discovery and allow-list enforcement), and **agent** (UI and token/tool-event streaming) families, with a new runnable example and new SLOs. *Depends on 2.0 and 2.1.*

### 🚧 3.0 — The next breaking culmination: one self-improvement contract, provable erasure & the async-first canonical core (vs the field's hosted self-improvement & governance platforms)

*By 2.x, continual self-improvement, rotation, regression, documents/images-out, distribution, and
benchmarks all ship — but as composed capabilities, each carrying its own surface. 3.0 is the next
breaking window: it unifies them under one declarative, governed self-improvement contract; makes the
data model honest about consent and **provable** erasure (not merely traceable); and makes the async
API canonical with sync as the thin wrapper — the three changes that genuinely require reshaping
interfaces the 2.x surface still carries as additive bolt-ons. It ships only when there is a real
breaking need, never for its own sake, and lands with the same mechanical deprecation runway 1.0
established — every collapsed alias deprecated, warned, and dated before removal.*

- **Unified declarative self-improvement contract** — one `SelfImprovementPolicy` composes scheduling,
  autonomous experiment proposal, online updates, canary/rollback, label acquisition (active learning),
  and meta-optimization — learned fitness weights plus auto strategy/budget selection via
  successive-halving — under a single audited, governed contract. It reshapes the 1.10/2.x
  `ImprovementLoop` and optimizer interfaces (`optimize/loop.py`, `optimize/search.py`,
  `optimize/reflective.py`) from composed tools into one streaming controller (`policy.stream()` emits
  proposal → canary → promote/rollback events), which is why this is breaking. It also lands the
  canary-driven prompt/policy **promotion** reserved out of 1.10 as a new serving/deploy surface on
  `core/app.py` (`app.deploy(...)` gated by a canary verdict), so the system tunes itself, decides what
  to tune, and rolls itself back — all under one policy you own.
- **Provable erasure + consent/purpose modeling** — a persistent lineage index (`governance/lineage.py`)
  gains **erasure proof artifacts**: signed manifests recording exactly what was removed across every
  index, cache, memory, and **generated artifact**, on the same hash-chained audit log the citations
  already use. GDPR purpose / lawful-basis tags ride on the data feeding access decisions
  (`security/access.py`), a `ConsentLedger` binds consent to that data, and `memory/engine.py` becomes
  bi-temporal (`valid_from` / `valid_to` plus as-of recall) with per-memory ACLs / team-shared memory.
  Reshaping the `MemoryItem` and lineage data model is why this is breaking; multimodal memory items
  (image / doc-chunk content) ride the 2.0 multimodal packet, so an erased document is erased as evidence,
  as memory, and as generated output in one operation.
- **Async-first canonical core & finalized telemetry contract** — the async API becomes canonical: sync
  `run()` becomes a thin wrapper that requires no running loop, and the async store / index / event
  protocols (`storage/base.py`, `core/events.py`) become *the* contract, removing the sync-store and
  sync-event ambiguities that constrained scale. The unified spans + metrics + cost telemetry model
  (`core/runtime.py`, observability) is locked as the single source of truth, retiring the transitional
  shims 2.0 carried. Pydantic / Python floor bumps land here if needed — the structural simplifications
  the 1.x sync/async duality could not resolve, made at the only window where the duality can be collapsed.
- *Interconnection:* 3.0 collapses the 2.x composed surfaces into canonical contracts while preserving
  the one packet / ledger / audit / trace: the self-improvement policy drives the same gated promotion and
  rollback the loop always used; provable erasure rides the same lineage and audit chain the citations
  already walk; and the async-first core makes every store, event, and telemetry contract the one Vincio
  actually runs on. The culmination is fewer, truer abstractions — not more features.
- *Edge over specialists:* hosted self-improvement / optimizer platforms make continual tuning a service
  you send traces to; Vincio makes it a **single declarative, governed, in-process contract** that decides
  what to tune and rolls itself back under one audited policy. OneTrust/Transcend orchestrate erasure
  across systems and report it; Vincio emits a **signed erasure proof** across every index, cache, memory,
  and generated document on the same audit chain — erasure that is provable, with consent bound to the
  data, not merely traceable. And against the dual sync/async SDKs (Vercel AI SDK, the async-canonical
  frontier), Vincio makes async the one true contract with sync as a zero-cost wrapper — the scale ceiling
  the `run_sync` bridge and sync stores imposed, finally removed.
- *Definition of done:* the unified self-improvement contract end-to-end; signed erasure proofs across
  every store with consent / purpose enforcement and bi-temporal recall; and the async-canonical core with
  the finalized telemetry contract — all covered offline. Gated by the VincioBench **loop**, **governance**,
  **scale**, and **memory** families (declarative self-improvement + meta-optimization in `loop`;
  erasure-proof correctness + consent enforcement in `governance`; async-canonical throughput in `scale`;
  bi-temporal recall + per-memory ACL in `memory`), with a new runnable example, the SLOs, and the
  mechanical deprecation runway for every collapsed surface documented before release.

### 🔭 Exploring — beyond 3.0

Candidates that are real but not yet scheduled — pulled forward when demand and the standards settle:

- 🔭 **Federated / cross-org self-improvement** — sharing gated optimizations and learned routing
  across trust boundaries without sharing raw traffic, once privacy-preserving aggregation standards
  settle.
- 🔭 **World-model / simulation-based planning** — agents that learn a tool/environment model and plan
  against it, beyond the reset/step/verify environment-eval harness of 2.2.
- 🔭 **Native video understanding & generation** — a video `ContentPart` with frame sampling, temporal
  segmentation, and generative output, extending multimodal beyond the image/audio in-and-out of 1.9
  and the multimodal packet of 2.0.
- 🔭 **On-device fine-tuning / continual local adaptation** — LoRA-class local adaptation of the
  in-process GGUF provider from the same flywheel, beyond the executed hosted fine-tune jobs of 2.1.
- 🔭 **MCP Apps & the post-2026 MCP spec** — server-rendered UI and stateless-core changes, adopted
  once the spec ships stable (tracked alongside the AG-UI streaming of 2.2).
- 🔭 **Formal verification of governance invariants** — machine-checkable proofs that residency,
  erasure, and budget invariants hold across the whole pipeline, beyond the signed audit chain and
  provable erasure of 2.0/3.0.
- 🔭 **A further breaking window beyond 3.0** — reserved, as always, only for changes the frozen 3.x
  surface cannot make additively, shipped with the same mechanical deprecation runway and never for
  its own sake.

---

## Out of scope

Vincio is a library, and stays one. The building blocks for running it in production — a
hash-chained audit log, retention policies, tenant isolation, RBAC / ABAC, and a server — ship in
the package so you can deploy them on your own infrastructure. **Hosted services, managed control
planes, dashboards-as-a-service, and compliance programs are not part of this project.** This stays
true through 3.0: everything the road to 3.0 adds that *looks* operational — the served observability
and alerting plane and the `vincio serve` launcher (2.1), distributed execution (2.1), the agent
directory (2.2), and the canary/rollback serving surface (2.0/3.0) — ships in the package as something
you run on your own infrastructure. The served plane is self-hosted over your own indexed store, the
distributed backend is a lock-free adapter to your Temporal/Ray, the agent fabric is a governed
directory you operate, and every standard (MCP, A2A, AGNTCY, OWASP/NIST, OTel GenAI, C2PA) is
implemented in-library. Vincio gives you the engine; how and where you run it is yours.
