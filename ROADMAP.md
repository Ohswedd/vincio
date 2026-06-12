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
and contextual indexing, GraphRAG, live indexes, and a connector hub.

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

### 🚧 0.4 — Memory & personalization (vs Mem0)

*Personalization without the failure mode of stale, ungrounded memories.*

- **Personalization APIs** — first-class user / agent / session memory scopes with simple
  `remember` / `recall` ergonomics over the existing L0–L5 layers.
- **Consolidation tiers** — automatic episodic→semantic summarization, dedup, and promotion with
  full provenance retained.
- **Hybrid memory store** — vector + graph recall in one query, with the memory graph as the
  relationship backbone.
- **Forgetting & hygiene** — tunable decay, TTL, importance-weighted retention, and explicit
  user-driven edit/delete/export (GDPR-style) flowing through the audit log.
- **Memory eval harness** — metrics for recall precision, contradiction rate, staleness, and
  personalization lift, runnable in VincioBench.
- *Interconnection:* confirmed evidence and tool results can be written back as candidate memories;
  every memory is utility-scored against the task before it ever enters a packet.
- *Edge over specialists:* Mem0 stores memories; Vincio stores memories **with confidence,
  provenance, decay, and conflict resolution**, scored for relevance before inclusion.

### 🔭 0.5 — Evaluation, testing & observability (vs Ragas, DeepEval, LangSmith, Langfuse)

*Make evaluation and observability so good you stop reaching for an external platform — and keep them
provider-neutral and dependency-free.*

- **Metric library expansion** — faithfulness, answer relevance, context precision/recall,
  hallucination, toxicity, bias, summarization quality, and conversational/session metrics;
  rubric-based **G-Eval**-style LLM judges with calibration.
- **Testing ergonomics** — a `pytest` plugin with `assert_eval` / `assert_grounded` assertions,
  snapshot tests for packets and traces, and CI-friendly thresholds.
- **Red-teaming & robustness** — an adversarial suite (jailbreaks, injection, PII-leak probes, bias
  prompts) that reuses the security engine's detectors.
- **Synthetic data generation** — bootstrap golden eval sets from your own corpora with
  difficulty/coverage controls and provenance.
- **Experiment tracking** — local run store, experiment comparison, ablations, and prompt/retriever
  A/Bs with statistical significance.
- **Prompt registry** — versioned prompt store with diffs, tags, rollbacks, and links to eval runs.
- **Richer trace model** — sessions, threaded runs, user feedback capture, scores attached to spans,
  and **OpenTelemetry GenAI semantic conventions**.
- **Local trace viewer** — a TUI and a self-contained static-HTML export of a trace/session (no
  server, no account); diff two traces visually.
- *Interconnection:* metrics defined here are the *same objects* used as runtime guardrails (0.7) and
  as the optimizer's fitness terms (0.8); traces become datasets with one command.
- *Edge over specialists:* LangSmith/Langfuse are platforms you send data to; Vincio's evals and
  traces live **in your process, in the same model as the runtime**, and can gate a release offline.

### 🔭 0.6 — Agents & orchestration (vs LangChain/LangGraph, CrewAI, OpenAI Agents SDK)

*Match the orchestration frameworks on expressiveness, beat them on safety and observability.*

- **Multi-agent teams** — roles, crews, delegation, and a shared blackboard/working memory, with
  per-agent budgets and termination guarantees.
- **Durable stateful graphs** — checkpointing, resume, time-travel/replay, and persistent run state
  on the existing storage layer; deterministic re-execution from any step.
- **Human-in-the-loop** — first-class interrupts, approval gates, and edit-and-resume on the agent
  and workflow graphs.
- **Declarative composition** — a small, typed composition API (compose/pipe) so chains and graphs
  read like data, with streaming events for every node.
- **Runtime backends** — adapters that can target LangGraph or the OpenAI Agents SDK underneath the
  provider-neutral compiler layer, so Vincio orchestrates without lock-in.
- *Interconnection:* every agent step emits the same spans and can be eval-scored and optimized;
  agents read context through the compiler, so budgeting and guardrails apply automatically.
- *Edge over specialists:* CrewAI gives you a crew; Vincio gives you a crew that is **bounded,
  traced, eval-gated, and budget-aware** by construction.

### 🔭 0.7 — Structured output, guardrails & reliability (vs Pydantic AI, Guardrails, NeMo, DSPy)

*Reliability as a guarantee, not a hope.*

- **Constrained generation** — provider-native grammar/JSON-schema-constrained decoding where
  available, with the robust-parser fallback everywhere else.
- **Streaming validation** — validate and repair partial structured output as it streams.
- **Typed signatures** — DSPy-style input→output signatures over the prompt AST, usable as
  optimization targets.
- **Rails as policies** — programmable input/output rails (topic, format, safety) expressed in the
  deterministic policy engine and enforced before/after generation.
- **Self-correcting loops** — bounded validate→critique→repair cycles with cost ceilings; structure
  is fixed, facts are never invented.
- **Multi-schema routing** — choose/validate against alternative schemas by task or content.
- *Interconnection:* every validation failure and repair is a trace event and an audit entry; rails
  reuse the security detectors; signatures feed the optimizer.

### 🔭 0.8 — The closed-loop ecosystem (the differentiator)

*This is the milestone no single-purpose library can ship, because it requires owning the whole
lifecycle.*

- **Trace → dataset → eval → optimize → promote** — one continuous loop: capture production traces,
  curate them into datasets, evaluate, run the gated optimizer, and promote the winner — all in the
  library, all reproducible.
- **Auto-memory from runs** — high-confidence, well-grounded facts surfaced during runs become
  candidate memories under the existing write policy.
- **Retrieval feedback** — eval-scored relevance feeds reranker weights and chunking choices
  automatically.
- **Cost/quality Pareto optimization** — the optimizer searches the prompt/context/routing/cache
  space against a multi-objective (accuracy, groundedness, latency, cost) frontier, not a single
  score.
- **Learned context budgeting** — per-task budget allocation tuned from eval outcomes instead of
  fixed tables.
- **Context-aware offline optimization** — richer offline/RL-style search strategies for the
  evolution loop, bounded and gated.
- *Edge over the field:* each competitor optimizes one organ; Vincio optimizes the **organism**, with
  every signal flowing through one packet, ledger, and trace.

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
