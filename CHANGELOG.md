# Changelog

All notable changes to Vincio are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-13

Stabilization & guarantees — the 1.0 roadmap milestone. This release does not
add subsystems; it turns the library into a product you can trust in
production. Every guarantee is mechanical: SemVer on a frozen public surface
with an enforceable deprecation policy, published SLOs that CI budgets hold at
least as strict, a documented threat model backed by offline audit-chain
verification and a resource-limited tool sandbox, supply-chain attestations on
releases, and a docs-completeness gate that runs every example.

### Added

- **API stability module** (`vincio.stability`) — `deprecated(since=,
  removed_in=, alternative=)` and `experimental(since=, note=)` decorators
  (working on functions and classes) that emit `VincioDeprecationWarning` /
  `VincioExperimentalWarning`; `deprecated_alias(...)` for renamed symbols;
  `stability_of(obj)` to introspect any symbol's contract; `public_api()` and
  `API_VERSION`. All are re-exported from the top-level `vincio` package, which
  is now the SemVer-covered public surface. See `docs/reference/stability.md`.
- **Published SLOs** (`benchmarks/slos.json`, `docs/reference/slo.md`) —
  latency/throughput/token-efficiency/quality/security targets, each naming the
  VincioBench budget that enforces it. The budget is held at least as strict as
  the public promise, so a passing CI run provably honors the SLO;
  `tests/test_slos.py` verifies the invariant.
- **Offline audit-chain verification** — `verify_audit_file(path)` and
  `AuditLog.verify_file()` re-read the persisted JSONL and validate the SHA-256
  hash chain, detecting tampering after a process restart and pinpointing the
  first broken line (`ChainVerification`). New CLI: `vincio audit verify [path]`.
- **Threat model** (`docs/security/threat-model.md`) — STRIDE over the real
  controls (access, audit, injection, PII/secrets, sandbox), with the explicit
  out-of-scope statement and the supply-chain story.
- **Supply-chain attestations** — the release workflow now generates a
  **CycloneDX SBOM** and emits **SLSA build-provenance attestations**
  (`actions/attest-build-provenance`) for the published wheel and sdist.
- **VincioBench methodology** (`benchmarks/METHODOLOGY.md`) — what each family
  measures, its naive baseline, corpus provenance, the budgets-vs-SLOs design,
  and how to reproduce every number offline. Reports now include an
  `environment` block (Vincio/Python versions, platform, schema version).
- **Security & governance example** (`examples/21_security_governance.py`) —
  PII/secret redaction, injection defense, RBAC/ABAC + tenant isolation,
  programmable rails, and a tamper-evident audit log, all offline.
- **Docs-completeness gate** (`tests/test_docs_completeness.py`,
  `tests/test_examples.py`) — runs all 22 examples end-to-end offline and
  asserts every public subsystem is documented and every example is indexed.
  The API reference now documents `vincio.input`, `vincio.documents`,
  `vincio.cli`, and `vincio.stability`.

### Changed

- **Tool sandbox hardening** — `run_subprocess_sandboxed` and `SandboxedPython`
  accept `max_cpu_seconds` / `max_memory_bytes` / `max_open_files` and apply
  them via POSIX `setrlimit` in the child (best-effort; the wall-clock timeout
  and output caps always apply). `SandboxedPython` defaults to conservative
  10s CPU / 512 MB / 64-fd limits.
- `__version__` is now `1.0.0`; the package classifier moves to
  `Development Status :: 5 - Production/Stable`. Top-level exports add
  `API_VERSION`, `StabilityLevel`, `VincioDeprecationWarning`,
  `VincioExperimentalWarning`, `deprecated`, `experimental`, and `stability_of`.
- `SECURITY.md` now lists 1.0.x as supported and documents SBOM/provenance.

### Fixed

- Carried forward from 0.9.0 and noted here for the 1.0 record: the
  `ContextApp.add_evaluator` key mismatch for nameless callables (e.g.
  `functools.partial`) — the name is resolved once so later metric lookup
  succeeds.

## [0.9.0] - 2026-06-13

Integrations, connectors & developer experience — the 0.9 roadmap milestone.
Win on coverage and ergonomics so real projects adopt Vincio without rewriting
their stack: an OpenAI-compatible passthrough for any endpoint, hosted
rerankers/embedders and three more vector stores behind the existing
interfaces, two-way LangChain/LlamaIndex interop, scaffolding templates with a
typed config schema, notebook reprs and an interactive TUI, opt-in domain
packs, and migration guides. Every new adapter implements an interface the
engine already speaks, so breadth adds no new concepts — context compilation,
budgeting, evals, traces, and security apply unchanged.

### Added

- **OpenAI-compatible passthrough** (`vincio.providers.openai_compat`) —
  `OpenAICompatibleProvider` reaches any Chat-Completions endpoint;
  `openai_compatible("groq")` / `openai_compatible(base_url=..., api_key=...)`
  construct one, with named presets for `groq`, `together`, `fireworks`,
  `openrouter`, `deepseek`, `perplexity`, `xai`, and `nvidia`. Presets are
  registered in the provider registry (so `build_provider("groq")` and
  `provider.default: groq` work) and their keys resolve from the conventional
  `<NAME>_API_KEY` env var — no extra wiring.
- **Hosted rerankers** (`vincio.retrieval.rerankers`) — `CohereReranker`,
  `JinaReranker`, and `VoyageReranker` call the real rerank endpoints over the
  core `httpx` dependency (no SDK), behind `build_reranker("cohere"|"jina"|
  "voyage", api_key=..., model=...)` and the `retrieval.reranker` config. An
  injectable `httpx.AsyncClient` keeps them offline-testable.
- **Hosted embedders** (`vincio.retrieval.embeddings`) — `JinaEmbedder`,
  `VoyageEmbedder`, and `CohereEmbedder` (Cohere's v2 `embeddings.float` shape
  handled), plus a `build_embedder("local"|"jina"|"voyage"|"cohere"|<provider>)`
  factory that also wraps any embedding-capable provider as a `ProviderEmbedder`.
- **More vector stores** — `ChromaVectorIndex`, `PineconeVectorIndex`, and
  `LanceDBVectorIndex` join Qdrant and pgvector behind the retrieval `Index`
  protocol, unified by `vincio.storage.build_vector_index(kind, embedder,
  **opts)` (`memory`, `qdrant`, `pgvector`, `chroma`, `pinecone`, `lancedb`).
  Missing optional dependencies raise a clear, actionable `StorageError`. New
  extras: `vincio[chroma]`, `vincio[pinecone]`, `vincio[lancedb]`.
- **Framework interop** (`vincio.interop`) — bring LangChain and LlamaIndex
  **tools, retrievers, loaders/readers, and embeddings** into Vincio, and hand
  Vincio's back. The `from_*` adapters are duck-typed (they import nothing
  heavy), so existing assets drop in without a new dependency;
  `add_langchain_tool` / `add_llamaindex_tool` register *and* enable a tool in
  one call; imported documents chunk, index, budget, and cite like any local
  file, and imported tools run through the same permissioned, sandboxed, audited
  runtime. The `to_*` adapters build real framework objects (extras
  `vincio[langchain]` / `vincio[llamaindex]`).
- **Scaffolding & templates** — `vincio init --template {minimal,rag,agent,eval}`
  generates a tailored `ContextApp`, `vincio.yaml`, golden set, and (for `rag`)
  sample docs. Every generated config carries a `# yaml-language-server:
  $schema=…` hint and ships a JSON Schema for editor completion; `--provider`
  sets the default provider.
- **Typed config tooling** — `config_json_schema()` derives a JSON Schema from
  the typed `VincioConfig`; `vincio config schema` emits it, `vincio config
  validate` checks a config file with clear errors, and `vincio config show`
  prints the effective merged configuration.
- **Notebook & TUI ergonomics** — `vincio.notebook.enable_rich_reprs()` attaches
  HTML/Markdown reprs to `RunResult`, `Trace`, `EvalReport`, `MemoryItem`, and
  `SearchHit` for Jupyter (pure `*_html`/`*_markdown` render functions you can
  also call directly; `enable_rich_reprs` is exported from the top level).
  `vincio.tui.TUI` / `vincio tui` is a dependency-free, keyboard-driven inspector
  for runs, traces, and memory, with pure screen renderers and injectable IO so
  it is fully unit-tested.
- **Domain packs** (`vincio.packs`) — opt-in, dependency-free bundles for
  **support, engineering, finance, and legal**: a role/objective/rules prompt
  config, a structured output schema, recommended policies + evaluators, and a
  golden eval set. `app.use_pack("support")` applies one through the public app
  API (layer your own settings on top); `load_pack` / `available_packs` /
  `register_pack` and `vincio packs list` / `vincio packs show` round it out.
- **Migration guides** — "coming from LangChain / LlamaIndex / Ragas / Mem0"
  guides that map concepts one-to-one, plus an
  [integrations guide](docs/guides/integrations.md) covering the new providers,
  vector stores, embedders, rerankers, and interop adapters. Two new runnable
  examples: `19_framework_interop.py` and `20_domain_pack.py`.

### Fixed

- `ContextApp.add_evaluator` registered a callable without `__name__` (e.g. a
  `functools.partial`) under a key one greater than the one it recorded in
  `app.evaluators`, so later lookup missed the metric; the name is now resolved
  once.
- Removed a duplicate `dist/` entry in `.gitignore`. (The provider-transport
  reliability fixes — event-loop-safe HTTP clients and 429 cooldowns honored
  from provider error bodies — shipped with 0.7/0.8 and are documented under
  [0.8.0].)

### Changed

- `__version__` is now `0.9.0`; top-level exports add `Pack`, `load_pack`,
  `available_packs`, and `enable_rich_reprs`. New offline tests cover provider
  presets and key resolution, hosted reranker/embedder wire formats, the
  vector-store factory, both interop bridges, pack loading/application/run, the
  notebook reprs, the TUI loop, and every new CLI command; the suite stays
  fully offline and ruff-clean.

## [0.8.0] - 2026-06-13

The closed-loop ecosystem — the 0.8 roadmap milestone, and the differentiator:
the milestone no single-purpose library can ship, because it requires owning
the whole lifecycle. One continuous, reproducible improvement cycle —
trace → dataset → eval → optimize → promote — plus the feedback paths that
let every organ tune the others: runs write grounded facts back to memory,
eval-scored relevance tunes retrieval, the optimizer keeps a cost/quality
Pareto frontier instead of one score, budget allocation is learned from eval
outcomes, and guided offline search strategies drive the evolution loop.

### Added

- **The improvement loop** (`vincio.optimize.loop`) — `ImprovementLoop` /
  `app.improvement_loop()` / `vincio loop run` wires the pieces that already
  exist into one call: capture the traces production runs already write
  (any exporter), curate them with `dataset_from_traces` (only successful
  runs whose mean user feedback clears `min_feedback_score`; the dataset's
  case-id fingerprint is recorded for reproducibility), evaluate the current
  prompt as the baseline, run the gated prompt optimizer, and promote the
  winner: pushed to the `PromptRegistry`, tagged (`production` by default),
  linked to the eval report that justified it, applied to the live app,
  written to the hash-chained audit log (`loop_promotion`), and announced on
  the event bus (`loop.promoted`). Baseline and winner reports land in the
  `ExperimentTracker` (same metadata store as runs), so `compare()` and
  `ab_test()` work across cycles; `dry_run=True` reports the decision
  without acting. Candidate evaluations are memory-write-free: an eval run
  never pollutes user memory or hands later candidates different recall
  state than earlier ones saw.
- **Auto-memory from runs** (`vincio.memory.facts`) — with
  `memory.write_back: [facts]`, verifiable claims from a run's output that
  the cited evidence supports become *candidate* memories:
  `extract_grounded_facts()` is deterministic (claim-shaped sentences,
  support-thresholded lexical grounding against the cited evidence,
  citation markers stripped), `MemoryEngine.write_back(facts=...)` writes
  them with measured support and evidence provenance
  (`origin: run_fact`, confidence scaling with support), and admission
  still runs the guarded write policy — privacy, stability, contradiction,
  confidence — with the candidate status penalty in recall until confirmed.
  New config: `memory.fact_min_support`, `memory.max_facts_per_run`.
- **Retrieval feedback** (`vincio.optimize.retrieval_feedback`) —
  `RetrievalFeedback` tunes a live `RetrievalEngine` from relevance labels
  that already live on eval cases (`rubric.relevant_ids`, via
  `records_from_dataset` / `records_from_report`): a deterministic
  coordinate search over per-index RRF fusion weights and a grid over the
  heuristic reranker's blend, both **gated** — weights change only when
  recall@k + MRR over the records measurably improve, and the engine is
  restored untouched otherwise. `recommend_chunking(reports_by_config)`
  picks the chunking config whose eval report scored best, staying on the
  baseline unless beaten by `min_improvement`.
- **Cost/quality Pareto optimization** (`vincio.optimize.pareto`) —
  `pareto_loop` keeps the full multi-objective frontier instead of one
  scalar: `ObjectiveSpec` axes (defaults: accuracy, groundedness, cost,
  latency), `ParetoFrontier` with non-dominated filtering, `knee()`
  (best summed normalized goodness), and `select(constraints=, prefer=)`
  for per-objective bounds like `{"cost": 0.01}`. Screening still uses
  scalar fitness (cheap); the final pick comes from the frontier of
  full-dataset reports and passes the same promotion safety rules as the
  scalar loop.
- **Learned context budgeting** (`vincio.optimize.budget_learning`) —
  `BudgetLearner` searches bounded perturbations of the per-task allocation
  tables (move a slice of budget between blocks, renormalize) and adopts a
  learned table only through gated promotion; `LearnedAllocations`
  persists as JSON and installs via `app.use_learned_budgets()` or
  `BudgetAllocator(learned=...)` — tasks without a learned table keep the
  fixed defaults.
- **Guided offline search strategies** (`vincio.optimize.strategies`) —
  `hill_climb` (single-knob mutations of the incumbent) and `anneal`
  (Metropolis acceptance with a cooling schedule) condition each proposal
  batch on subset scores already observed; both are deterministic under a
  seed, hard-bounded by the evaluation budget, and pluggable into
  `ContextOptimizer(strategy=...)` or usable directly via
  `guided_search()`. Pre-scored candidates flow into `evolution_loop`
  without re-screening, and `OptimizationResult` now carries the evaluated
  baseline candidate.
- **CLI** — `vincio loop run --app app.py [--dataset ds.jsonl |
  --min-feedback X] [--gate "metric=>= 0.9"] [--tag production]
  [--experiment NAME] [--dry-run]`.
- **Docs & examples** — a "close the loop" guide
  (`docs/guides/close-the-loop.md`), updated API/CLI/config references,
  0.8 sections in the DSPy and Ragas comparisons, and runnable example
  `18_closed_loop.py` (the full cycle offline: auto-memory, promotion,
  the frontier behind the decision, retrieval feedback, learned budgets).
- **VincioBench `loop` family** — promotion fires and is deterministic,
  gates block regressions, the registry version is tagged and eval-linked,
  grounded facts are written (and ungrounded ones never are), retrieval
  tuning improves and is gated, the frontier excludes dominated points with
  a balanced knee, learned budgets promote, and guided search respects its
  budget — under 14 new CI-gated budgets (81 total).

### Changed

- `OptimizationResult` gains a `baseline: Candidate` field (the evaluated
  baseline with its full report), so loop callers can log and compare it.
- `evolution_loop` skips subset screening for candidates that arrive with
  `subset_fitness` already set (guided-search support); fresh candidates
  behave exactly as before.
- `BudgetAllocator` accepts `learned=` per-task allocation tables that
  override the fixed `TASK_ALLOCATIONS` entry for their task type.
- `MemoryEngine.write_back` accepts `facts=` (a list of `GroundedFact`)
  alongside `evidence=` and `tool_results=`.
- **495 tests passing offline in ~2s; ruff clean**; eighteen runnable
  examples; 81 CI-gated VincioBench budgets.

## [0.7.0] - 2026-06-13

Structured output, guardrails & reliability — the 0.7 roadmap milestone.
Reliability as a guarantee, not a hope: provider-native constrained decoding
with strict schema sanitization, streaming validation with early abort,
DSPy-style typed signatures that feed the optimizer, programmable rails in
the deterministic policy engine, bounded self-correcting loops that never
invent facts, and multi-schema routing — every failure, repair, and rail
decision landing on the trace and in the hash-chained audit log.

### Added

- **Constrained generation** (`vincio.output.constrained`) —
  `to_strict_json_schema()` transforms any JSON schema for strict
  provider-native constrained decoding (every object closed via
  `additionalProperties: false`, every property required, optional fields
  made nullable, `default`/`format` stripped) while validation keeps running
  against the original schema; `negotiate_decoding()` picks
  `native`/`prompt`/`none` from the provider capability matrix per run, and
  the chosen mode is recorded on the `prompt_render` and `output_validation`
  spans. Grammar-style constraints `choice_schema(options)` and
  `regex_schema(pattern)` express fixed choices and regex-shaped strings as
  schemas that ride the same native path; the deterministic JSON-schema
  validator now also enforces `pattern`.
- **Streaming validation** (`vincio.output.streaming`) —
  `StreamingValidator` accumulates text deltas, parses the balanced partial
  JSON, and prefix-checks it against the schema (`validate_partial`):
  missing required fields are tolerated while streaming, definite
  mismatches — wrong type, unknown field on a closed object — are reported
  mid-stream. `app.astream()` wires it in automatically: `partial_output`
  events now carry `valid_prefix` and `validation_errors`, so consumers can
  abort a generation that can no longer be valid; `finalize()` applies the
  allowed structural repairs at stream end.
- **Typed signatures** (`vincio.prompts.signatures`) — DSPy-style
  input→output signatures over the prompt AST: subclass `Signature` with
  `InputField` / `OutputField` markers (docstring becomes the instruction)
  or use the string form
  `signature("question, context -> answer, confidence: float")`.
  `Signature.to_prompt_spec()` compiles to a `PromptSpec` (drop-in target
  for `PromptOptimizer` variants/rewrites); `Predict` /
  `app.predictor(sig)` executes with provider-native constrained decoding
  and the full validation pipeline, returning typed results
  (`result.label`, `result.confidence`); inputs are type-checked before the
  call.
- **Rails as policies** (`vincio.security.rails`) — programmable rails as
  plain data (`Rail`: kind `topic` / `format` / `safety` / `custom`,
  direction, action `block` / `warn` / `redact`, parameters) evaluated by
  `RailEngine` inside the deterministic policy engine: topic rails match
  blocked/allowed topics by word-boundary patterns, format rails check
  length and require/forbid regexes, safety rails reuse the security
  engine's PII detector, secret scanner, and injection detector
  (`action="redact"` masks PII instead of blocking), and custom rails call
  predicates registered via `app.register_rail_predicate()`. Input rails
  run before the model is called (a blocking violation denies the run);
  output rails run inside the validation pipeline's policy step. Every
  violation is a `PolicyViolation` named `rail:<name>` on the trace and in
  the audit log. New app APIs: `app.add_rail(...)`.
- **Self-correcting loops** (`vincio.output.correction`) — `SelfCorrector`
  runs bounded validate → critique → repair cycles: the critique is built
  deterministically from the `ValidationReport` (`build_critique`), the
  repair request is structure-only (re-serialize, rename, retype — never
  add, remove, or change factual content), semantic/citation/policy
  validators re-run every cycle, and the loop stops at the first valid
  output, `max_cycles`, or the hard `max_cost_usd` ceiling.
  `app.enable_self_correction(max_cycles=, max_cost_usd=)` wires it into
  the run flow; cycles, cost, and outcome are a `self_correction` trace
  event and audit-log details.
- **Multi-schema routing** (`vincio.output.routing`) — `SchemaRouter` holds
  named `SchemaRoute`s (schema + task types / keywords / predicate /
  priority): `route()` picks the output contract for a run before
  generation (keywords match at word starts, so "crash" matches
  "crashed"), `classify()` finds which registered schema some structured
  data matches, and `validate_any()` validates against the alternatives.
  `app.add_output_schema(schema, keywords=..., task_types=..., when=...)`
  routes per run; the chosen schema is recorded on the `prompt_render`
  span.
- **Interconnection** — every validation failure and repair is now a trace
  event (`repair` / `validation_failed` / `self_correction` /
  `stream_invalid_prefix` events on the `output_validation` and
  `model_call` spans) *and* an `output_validation` entry in the
  hash-chained audit log (`decision=repair|deny`, with errors, repairs, and
  correction cycles); rails reuse the security detectors; signatures feed
  the optimizer.
- **VincioBench** — new `reliability` family measures strict-schema closure
  (100% objects closed and fully required), mid-stream invalid detection
  with abort savings (~98% of an invalid output's tokens saved offline),
  self-correction recovery rate with cycle bounds, rail catch rate with
  zero false positives on clean text, signature prediction validity and
  optimizer variant generation, and schema-routing/classification accuracy
  — held by 13 new `budgets.json` gates in CI.
- **Docs & examples** — a new how-to guide
  (`docs/guides/reliability-guardrails.md`), an expanded structured-output
  guide (constrained decoding, streaming validation, routing, signatures,
  self-correction), comparison write-ups for Pydantic AI, Guardrails AI,
  and NeMo Guardrails, a typed-signatures section in the DSPy comparison,
  and runnable example `17_reliable_structured_output.py`; the examples
  index now also lists the 0.6 crew and durable-graph examples.

### Fixed

- **HTTP provider clients no longer die with the event loop** — a provider
  reused across `asyncio.run()` calls (the natural sync usage of
  `generate_sync` / `stream_sync` / `app.run`) recreates its pooled
  `httpx.AsyncClient` when the cached client is bound to a closed or
  different loop, instead of raising "Event loop is closed".
- **Rate-limit cooldowns are honored from error bodies** — when a 429
  carries no `Retry-After` header, the retry delay is extracted from the
  provider error body (Google's `RetryInfo.retryDelay` detail or
  "retry in Ns" message), and `RetryingProvider`'s backoff cap was raised
  from 20s to 60s, so free-tier per-minute limits self-heal inside the
  retry loop.
- **Gemini defaults match the live API** — the default Google embedding
  model is now `gemini-embedding-001` (the live batch-embedding model), and
  the price table covers the current GA models (`gemini-2.5-pro`,
  `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`,
  `gemini-2.0-flash-lite`, `text-embedding-004`) so cost tracking reports
  paid-tier rates instead of $0 for unknown models.

### Changed

- `RunStreamEvent` gains `valid_prefix` and `validation_errors` on
  `partial_output` events (streaming validation).
- `PolicyEngine` accepts a `rails=` engine and `check_output()` can return
  `transformed_text` (redact-action rails); the validation pipeline ships
  the redacted text for plain-text outputs.
- The runtime negotiates the structured-output decoding mode per run and
  sends the strict-sanitized schema to capable providers (previously the
  raw schema was sent, which strict decoders such as OpenAI
  `strict: true` reject for open objects).
- **467 tests passing offline in ~2s; ruff clean**; seventeen runnable
  examples; 67 CI-gated VincioBench budgets.

## [0.6.0] - 2026-06-12

Agents & orchestration — the 0.6 roadmap milestone. Match the orchestration
frameworks on expressiveness, beat them on safety and observability:
multi-agent crews over a shared blackboard, durable stateful graphs with
checkpoint/resume/time-travel, first-class human-in-the-loop on graphs and
workflows, a declarative composition API with streaming node events, and
runtime backends that export to LangGraph and the OpenAI Agents SDK.

### Added

- **Multi-agent crews** — `Crew` / `app.crew(members=[...], process=...)`
  binds named `AgentRole`s (description, goal, keywords, `budget_fraction`)
  to bounded `AgentExecutor`s and runs them as a team: `sequential` (each
  member sees everything posted so far), `parallel` (bounded concurrent
  fan-out, dict of answers), and `hierarchical` (a manager decomposes the
  objective, delegates with a schema-validated plan, reviews the board, and
  either finishes or delegates follow-ups — with a deterministic
  keyword-routing fallback offline). Termination is guaranteed by
  construction: members run under a scaled share of the crew budget, the
  crew checks its budget before every delegation, and review rounds are
  capped at `max_rounds`. `CrewResult` carries per-member reports,
  `DelegationRecord`s, the blackboard snapshot, aggregated usage, and
  eval-ready `metrics()`.
- **Shared blackboard** — `Blackboard`: versioned, author-attributed shared
  working memory with per-key history, optional `blackboard.posted` events
  on the app event bus, prompt rendering (`as_context()`), and JSON
  `snapshot()` / `restore()` so crew coordination persists and replays.
- **Durable stateful graphs** — `StateGraph` / `app.graph()`: dict-state
  nodes (sync or async), static and conditional edges, optional per-key
  `reducers` for deterministic parallel-branch merges, and an optional
  Pydantic `state_schema` validated after every merge. `compile()` produces
  a `CompiledGraph` whose `Checkpointer` persists a checkpoint after every
  super-step on any `MetadataStore` (in-memory/SQLite/Postgres — `app.graph()`
  binds the app's store, so threads survive restarts): `resume(thread_id)`
  continues an interrupted thread, `history()` lists every checkpoint,
  `fork(checkpoint_id)` time-travels by branching a new thread that
  re-executes deterministically from that step, and `max_steps` bounds
  cyclic graphs. `astream()` yields node/checkpoint/interrupt/done events.
- **Human-in-the-loop** — pause graphs statically (`interrupt_before` /
  `interrupt_after` node lists) or dynamically from inside a node
  (`interrupt(state, payload)`); resume with a value and the paused node
  re-runs and receives it; `update_state(thread_id, values)` edits state as
  a new checkpoint before resuming. Workflow approval gates pause too:
  a gate with no `approval_fn` returns status `"paused"` with
  `pending_approvals`, and `workflow.resume(result, approvals={...})`
  continues without re-running done steps (edit the saved context to steer
  the continuation).
- **Declarative composition** — `compose(...)` / the `|` operator build
  typed pipelines from any mix of functions, agents, crews, workflows, and
  compiled graphs, normalizing results between steps (`AgentState` → final
  answer, `WorkflowResult`/`CrewResult` → output, `GraphResult` → state);
  `parallel(...)` fans out to named branches, `branch(router, routes)`
  routes by a function. `astream()` yields `NodeEvent`s
  (node_start/node_end/error/done) and every node emits a `compose_node`
  span.
- **Runtime backends** — `LangGraphBackend` exports a Vincio `StateGraph`
  to a LangGraph builder (nodes transfer as-is; edges, conditional edges,
  entry point, and `END` are translated) and `OpenAIAgentsBackend` exports
  agents and crews to OpenAI Agents SDK `Agent` objects (a crew becomes a
  manager agent with handoffs to every member; tools wrap via
  `function_tool`). Both import their runtime lazily and accept an injected
  module, so Vincio orchestrates without lock-in and the adapters test
  offline.
- **Observability** — new span types `crew`, `crew_agent`, `graph_node`,
  and `compose_node`; every crew member, graph node, and composed step is
  traced and scoreable like any other Vincio run.
- **VincioBench** — the `agent` family now also measures crew over-budget
  termination, full-crew success, delegation recording, interrupt→resume
  and fork-replay determinism (state must equal the uninterrupted run), and
  composition streaming coverage; six new `budgets.json` gates hold them in
  CI.
- **Docs & examples** — a new how-to guide
  (`docs/guides/orchestrate-agents.md`), expanded
  `docs/concepts/agents.md`, comparison write-ups for CrewAI and the OpenAI
  Agents SDK, a durable-graphs section in the LangChain/LangGraph
  comparison, and runnable examples `15_multi_agent_crew.py` and
  `16_durable_graph.py`.

### Changed

- **Workflow approval gates without an `approval_fn` now pause instead of
  failing** — `WorkflowResult.status` gains `"paused"` and
  `pending_approvals`; `arun(context=..., approvals=...)` /
  `aresume(previous, approvals=...)` continue a prior run, never re-running
  steps already done. Gates answered by a configured `approval_fn` behave
  exactly as before.
- `ContextApp.agent()` executor construction was factored into a shared
  builder reused by `app.crew()` (per-member tools/planner/model
  overrides); public behavior is unchanged.
- New error type `GraphError` (subclass of `AgentEngineError`) for graph
  definition and execution failures.
- **Pre-merge review hardening** — crew members built by `app.crew()`
  receive only their own (or the crew-level) tools, never the app-wide
  enabled set; per-member budget shares are clamped to what remains of the
  crew budget, and an explicit `budget_fraction=0.0` is honored; an
  approvals map can never bypass a configured `approval_fn`, and unknown
  approval names raise; a step failure beside a paused gate in the same
  level is terminal (compensation runs, the run is not reported paused);
  resumed workflow segments rebuild every non-`done` step result so
  compensated/failed steps never leak stale outputs; graph threads that
  ended at `max_steps` resume from their checkpoint (recompile with a
  higher bound), re-invoking a finished thread raises `GraphError` (fork it
  instead), and a dynamic interrupt mid-frontier re-queues the successors
  of siblings that already ran; `Crew` rejects unknown `process` values and
  `app.crew()` rejects unknown member fields; the LangGraph export gives
  routers exclusive edge precedence like the native engine; tracer
  trace/span cleanup tolerates abandoned streaming generators
  (`break` out of `astream`) without contextvar corruption.
- **426 tests passing offline in ~2s; ruff clean**; sixteen runnable
  examples; the VincioBench `agent` family holds the new orchestration
  guarantees under six additional CI-gated budgets.

## [0.5.0] - 2026-06-12

Evaluation, testing & observability — the 0.5 roadmap milestone. Make
evaluation and observability so good you stop reaching for an external
platform: metric parity with the eval specialists, unit-test ergonomics,
red-teaming, synthetic data, experiments with significance, a prompt
registry, sessions and feedback on traces, and a local viewer — all
provider-neutral, offline, and in-process.

### Added

- **Metric library expansion** — `faithfulness` (Ragas-style claim
  attribution), `answer_relevance` (penalizes evasive answers),
  `hallucination` (unsupported verifiable claims with **strict number
  checking** — "90 days" against evidence saying "30 days" fails; citation
  markers are stripped first), `toxicity` and `bias` (deterministic
  pattern-based rates), `summarization_quality`
  (min(coverage, faithfulness) against the source), and conversational
  metrics `knowledge_retention` (flags re-asking for facts the user already
  gave) and `conversation_relevance` (both read `context["messages"]`).
  All deterministic, offline, and usable as eval metrics, runtime
  evaluators, and test assertions.
- **G-Eval judge** — `GEvalJudge(provider, model=..., criteria=...)`
  auto-derives evaluation steps from plain-language criteria (cached for the
  judge's lifetime), scores on a 1–5 form-filling scale normalized to 0–1,
  approximates probability-weighted scoring with `samples > 1`, and
  `calibrate(pairs)` fits a linear correction against human labels
  (returns scale/offset/Pearson r) applied to future scores.
- **Testing ergonomics** — new `vincio.testing` package: `assert_eval`,
  `assert_grounded`, `assert_metric`, `assert_safe` raise AssertionErrors
  with the metric breakdown and offending output; quality metrics assert
  `>=`, rate metrics (`hallucination`, `toxicity`, ...) assert `<=`. A
  pytest plugin (registered via the `pytest11` entry point) adds the
  `vincio_snapshot` fixture and `--vincio-update-snapshots`; snapshots
  capture packet/trace *structure* with volatile fields (ids, timestamps,
  durations, hashes) normalized away, stored as JSON next to the tests.
- **Red-teaming & robustness** — `RedTeamSuite` sends 13 built-in probes
  (jailbreaks, prompt injections, PII/secret-leak probes, bias and toxicity
  provocations) at a `ContextApp` or any callable and judges responses
  deterministically: attack probes carry a canary token, leak probes run the
  secret scanner and PII detector, bias/toxicity probes reuse the new
  metrics. Reports separate `attack_success_rate` (output level) from
  `detector_coverage` (input-side injection detection); custom probes via
  `RedTeamProbe`. The injection detector gained `persona_without_rules` and
  `fake_authority` signals plus hardened override/exfiltration patterns —
  built-in probe coverage is 7/7 with no new false positives.
- **Synthetic data generation** — `SyntheticGenerator` bootstraps golden
  datasets from documents/chunks/text with difficulty mix (`easy` stated
  facts, `medium` cloze values, `hard` multi-hop across sources), coverage
  controls (round-robin over sources, near-duplicate dedupe), and full
  provenance (`metadata.source_ids`, source sentences in `rubric.facts` so
  grounding metrics work immediately). Deterministic offline templates by
  default; LLM-written questions when a provider is given, falling back to
  templates on failure.
- **Experiment tracking** — `ExperimentTracker` logs eval reports under
  experiment/variant (SQLite via the existing metadata store), `compare()`
  picks the best variant per metric (direction-aware: cost/latency/
  hallucination-style metrics minimize), `ablation()` reports deltas vs a
  baseline with p-values, and `ab_test(report_a, report_b, metric)` runs a
  paired t-test when reports share case ids, Welch's t-test otherwise —
  pure-Python t-distribution (regularized incomplete beta), no SciPy.
- **Prompt registry** — `PromptRegistry`: file-backed versioned prompt store
  keyed by `spec_hash` (re-pushing unchanged content is idempotent), tags
  that move between versions ("production", "candidate"), field-level and
  rendered diffs, `rollback()` that re-publishes an old version as a new
  head (history kept), and `link_eval()` attaching eval-run summaries to the
  exact version they measured. CLI: `vincio prompt push / versions / diff /
  rollback`.
- **Richer trace model** — traces carry `session_id` / `thread_id`
  (`app.run(..., session_id=...)` threads them through), `scores` (runtime
  evaluators attach metric scores to the eval span and the trace), and
  first-class `Feedback` (`trace.add_feedback`, `record_feedback(...,
  exporter=...)` persists updates; `vincio trace feedback`). Sessions are a
  derived view: `sessions_from_traces()` groups traces (deduping re-exported
  records) into `Session` objects with run/duration/error/score/feedback
  aggregates; `vincio trace sessions` lists them.
- **Traces become datasets** — `dataset_from_traces(traces,
  min_feedback_score=...)` curates captured runs into an eval dataset with
  full provenance (trace/run/session ids, scores); CLI:
  `vincio eval dataset golden.jsonl --min-feedback 0.5`.
- **OpenTelemetry GenAI semantic conventions** — the OTel exporter emits
  `chat {model}` / `execute_tool {tool}` span names with
  `gen_ai.operation.name`, `gen_ai.request.model`,
  `gen_ai.usage.input_tokens` / `output_tokens`,
  `gen_ai.response.finish_reasons`, `gen_ai.tool.name`, and
  `gen_ai.conversation.id` (sessions), alongside the full `vincio.*`
  attributes and span scores.
- **Local trace viewer** — `render_trace_text` / `render_session_text` (TUI
  tree with status glyphs, durations, scores, feedback; `vincio trace
  view`), `trace_to_html` / `session_to_html` (one self-contained static
  HTML file, inline CSS, no server or account; `vincio trace export
  [--session]`), and `trace_diff_html` (side-by-side visual diff;
  `vincio trace diff --html`).
- **Surface** — `vincio.evals` exports `GEvalJudge`, `SyntheticGenerator`,
  `RedTeamSuite` / `RedTeamProbe` / `BUILTIN_PROBES`, `ExperimentTracker` /
  `ab_test`, `dataset_from_traces`; `vincio.observability` exports
  `Session`, `Feedback`, `sessions_from_traces`, `record_feedback`, and the
  viewer functions; `vincio.prompts` exports `PromptRegistry` /
  `PromptVersion`; new `vincio.testing` package.
- **VincioBench `evals` family** — measures metric agreement on labeled
  examples, red-team judging on guarded vs naive targets, synthetic-data
  determinism and coverage, the significance machinery (detects a real
  shift, ignores a null one), session grouping, HTML self-containment,
  trace→dataset conversion, and G-Eval calibration — 13 new `budgets.json`
  gates hold the results in CI.
- Documentation: new observability concept guide and pytest testing guide,
  expanded evals concept guide, comparison write-ups for DeepEval and
  LangSmith/Langfuse, updated Ragas comparison; example
  `14_evaluation_observability.py`.

### Changed

- **OTel span names for model/tool spans changed** to the GenAI semantic
  conventions: `model_call:<name>` → `chat {model}`, `tool_call:<name>` →
  `execute_tool {tool}`. Dashboards or alerts keyed on the old span-name
  prefixes need updating; all `vincio.*` attributes (including
  `vincio.span_id`) are unchanged, and non-model/tool spans keep the
  `{type}:{name}` format.
- Model spans now record `input_tokens` (alongside `output_tokens`), and
  completed runs store their output (truncated) and eval scores on the
  trace, so traces are curatable into datasets.
- `JSONLExporter.load_all()` now returns the latest record per trace id
  (re-exports act as updates, e.g. after `record_feedback`).
- `EvalReport.diff()` is direction-aware: a rising `hallucination` /
  `toxicity` / `bias` / `unsupported_claim_rate` now counts as a regressed
  case (previously only falling scores did). Metric direction has a single
  source of truth: `vincio.evals.metrics.LOWER_IS_BETTER`.
- `EvidenceItem`-based grounding metrics accept reference context from
  `case.context["reference"]` / `["source"]` when a run carries no
  evidence.
- **367 tests passing offline in ~2s; ruff clean**; fourteen runnable
  examples; 48 VincioBench budget gates.

## [0.4.0] - 2026-06-12

Memory & personalization — the 0.4 roadmap milestone. Personalization
without the failure mode of stale, ungrounded memories: every memory
carries confidence, provenance, decay, and conflict resolution, and is
utility-scored against the task before it ever enters a packet.

### Added

- **Personalization APIs** — `remember()` / `recall()` ergonomics over the
  L0–L5 layers, on both `MemoryEngine` and `ContextApp` (`app.remember(...,
  user_id="u1")` auto-creates the engine). Scope and memory type are
  inferred (session > agent > user > tenant; preference/goal/decision/fact
  classification). New `MemoryScope.AGENT` gives every agent durable memory
  of its own, and `ScopedMemory` handles (`memory.for_user("u1")`,
  `for_agent`, `for_session`, `for_tenant`) bind one owner for
  `remember` / `recall` / `forget` / `items` / `export`.
- **Hybrid memory recall** — `MemoryEngine.asearch()` fuses lexical and
  vector relevance (`(1−w)·lexical + w·cosine` over any `Embedder`, offline
  hash embedder by default, content-addressed vector cache) with graph
  adjacency (memories linked to the task's entities get a boost) in one
  scored, scope- and privacy-filtered query; `search()` stays as the sync
  wrapper. The runtime's memory step extracts task entities and recalls
  hybrid by default (`memory.hybrid_recall`, `memory.vector_weight`).
- **Consolidation tiers** — `MemoryConsolidator` (and
  `await memory.consolidate(session_id, user_id=...)`): episodic session
  memories summarize into semantic memories promoted to user/agent scope,
  deduplicate (the survivor absorbs confirmations and records
  `merged_from`), and retain full provenance — promoted items carry
  `consolidated_from`, episodes are archived with `consolidated_into`,
  never silently dropped. `promote_aged_episodes()` runs the background
  tier transition.
- **Forgetting & hygiene** — per-scope TTL defaults applied on write
  (`memory.ttl_days`, sessions default to 30 days) with expired items
  excluded from recall; importance-weighted retention in `decay_pass()`
  (heavily used, confirmed, stable preferences/decisions survive longer —
  `importance_score`, `memory.retention_weight`); and user-driven
  `edit` / `forget` / `export_owner_data` / `erase_owner_data`
  (GDPR-style access, rectification, portability, erasure) flowing through
  the hash-chained audit log as `memory_edit` / `memory_delete` /
  `memory_export` / `memory_erase` entries.
- **Memory eval harness** — `vincio.memory.evaluate_memory` measures recall
  precision, recall@k, contradiction rate, staleness, and personalization
  lift (owner-scoped vs anonymous recall) against labeled
  `MemoryEvalCase`s; the VincioBench `memory` family runs it plus
  consolidation/TTL checks, gated in CI by eleven new `budgets.json`
  entries.
- **Run write-back** — step 16 is now governed by `memory.write_back`
  (`input` | `evidence` | `tools`): cited evidence and successful tool
  results write back as *candidate* memories with provenance
  (`origin` / `source_id` / `tool_name`), carrying a status penalty in
  recall until confirmed (restatement or `confirm()` promotes them to
  active).
- **Surface** — CLI `vincio memory remember | recall | forget | export |
  consolidate | decay`; server endpoints `POST /v1/memory/consolidate`,
  `GET /v1/memory/export`, `GET /v1/memory/stats`,
  `DELETE /v1/memory/{id}`; `extract_entities` is now public in
  `vincio.retrieval.chunking`; new docs (rewritten memory concepts page, a
  Mem0 comparison) and `examples/13_memory_personalization.py` (offline).

### Changed

- `MemoryEngine` accepts `embedder`, `vector_weight`, `retention_weight`,
  `ttl_days`, and `audit`; `app.add_memory()` wires the app's embedder and
  audit log automatically. Search components now report `lexical`,
  `vector`, `graph`, and `status` alongside the existing factors.
- `MemoryEngine.search()` includes `candidate`-status memories with a 0.7
  status weight, and restatements re-activate confirmed candidates.
- 301 tests passing offline (~2s); ruff clean.

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

[1.0.0]: https://github.com/Ohswedd/vincio/releases/tag/v1.0.0
[0.2.0]: https://github.com/Ohswedd/vincio/releases/tag/v0.2.0
[0.1.0]: https://github.com/Ohswedd/vincio/releases/tag/v0.1.0
