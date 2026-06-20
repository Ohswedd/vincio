# Changelog

All notable changes to Vincio are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2026-06-20

Runtime performance & efficiency: make the compile spine fast enough that
context engineering is never the bottleneck. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the dependency-free offline
path is the default, and every default run path is unchanged. NumPy is an
optional accelerator, never a requirement.

### Added

- **Vectorized candidate scoring.** `ContextScorer.score_batch` scores a whole
  candidate set in one pass — the per-component scores are reduced against the
  weight vector together (a single matrix product under NumPy via the new
  `vincio.context.vectorized`, an identical pure-Python reduction otherwise), and
  each `ContextScores` is built without per-item validation. Bit-for-bit
  identical selection to the per-candidate loop. VincioBench gates
  `families.perf.vectorized_scoring.equivalent`.
- **Compiled-prompt render program.** `PromptCompiler` compiles a spec's stable
  prefix (role/objective/rules/safety/definitions/output-contract/examples) once
  into a reusable render program (`vincio.prompts.program`,
  `CompilerOptions.use_render_program`, default on) and reuses it across calls
  that share the spec, rendering only the volatile suffix. Byte-identical output;
  `program_hits` counts reuses. VincioBench gates
  `families.perf.render_program.byte_identical`.
- **Warm candidate arena.** When the candidate set (inputs + privacy scope) is
  unchanged, the context compiler reuses the collected, normalized, and
  privacy-screened candidates (`vincio.context.arena`,
  `performance.reuse_candidate_set` / `ContextCompilerOptions.reuse_candidate_set`,
  default on) instead of rebuilding them. Correctness-preserving and safe under
  concurrent use; `arena_hits` counts reuses. VincioBench gates
  `families.perf.warm_arena.equivalent`.
- **Streaming-first compilation.** `ContextCompiler.compile_streaming` yields a
  new `CompileStreamEvent` stream — the stable prefix (objective / instructions /
  constraints / task) before any candidate is scored, then the selected evidence,
  then a terminal `done` carrying the full `CompiledContext` (identical to
  `compile`). Back-pressure is the async generator itself. VincioBench gates
  `families.perf.streaming_compile.prefix_before_scoring`.
- **Speculative retrieval prefetch.** Opt-in `performance.speculative_prefetch`
  warms the query embedding (`vincio.retrieval.SpeculativePrefetcher` /
  `PrefetchHandle`) from the task classification while preparation runs, so
  retrieval's query embed lands as a cache hit; cancelled cleanly and best-effort.
  VincioBench gates `families.perf.prefetch.warms_cache`.
- **Per-app memory-footprint budget.** `performance.memory_budget_mb` declares a
  resident-memory ceiling for the compiled packet; the compiler slims the packet
  and evicts the lowest-utility evidence to fit (`vincio.context.footprint`),
  recording each eviction. The footprint is surfaced as `RunResult.memory_bytes`
  and rolled up as `peak_resident_bytes` in the cost summary. VincioBench gates
  `families.perf.footprint.budget_enforced` and a resident-footprint regression
  gate `families.perf.footprint.packet_bytes`.

### Performance

- New SLOs: p99 cold-compile latency, a sub-millisecond warm-compile hot path
  (`families.perf.context_compile.cached_p50_ms`), and a resident-footprint
  ceiling, plus the equivalence/byte-identity/streaming/prefetch invariants —
  each backed by an at-least-as-strict VincioBench budget.

**1663 tests passing offline; ruff + mypy clean; VincioBench 287 budgets / 96 SLOs.**

## [3.0.1] - 2026-06-18

Closes the two honest scoping notes the 3.0.0 milestone shipped with. Both
additive and backward-compatible — `API_VERSION` stays `3.0`, and the default
`app.deploy(dataset=...)` and `run`/`arun`/`abatch` paths are unchanged.

### Added

- **Live-traffic canary bound to the deploy surface.** `app.deploy` (and
  `deploy_candidate`) gain a live mode — `app.deploy(candidate, live_inputs=...,
  score_fn=...)` — that ramps `CanarySpec.percent` of the supplied live runs onto
  the candidate prompt/policy, scores each arm online with `score_fn(RunResult)`,
  and once `min_samples` candidate observations land applies the same
  no-regression verdict: promote, or **freeze + auto-roll-back** on a regression.
  The new `LiveCanary` (`vincio.optimize`) is the reusable prompt-layer analog of
  the 1.8 `CanaryRouter` (per-run observation via `aobserve`, `verdict()`,
  `afinalize()`); each observation still returns a real answer to the caller.
  `CanarySpec` gains `percent`. VincioBench gates
  `families.loop.self_improvement.live_canary_promotes` / `live_canary_rolls_back`.

### Changed

- **The async-canonical run path is now literally true on every path.** The
  batch path's `VincioRuntime._persist_run` is now a coroutine that persists
  through the canonical async store contract (`await asave`), matching the
  interactive/streaming epilogue — so no run path blocks the event loop with a
  synchronous store write. VincioBench gates
  `families.scale.async_canonical.run_path_persists_async`.

**1623 tests passing offline; ruff + mypy clean; VincioBench 277 budgets / 87 SLOs.**

## [3.0.0] - 2026-06-18

The breaking culmination — fewer, truer abstractions. 3.0 is the second
deliberate breaking window (after 2.0): it unifies the 2.x self-improvement
organs under one declarative contract, makes erasure **provable** with consent
modeling, and makes the async store/event contracts canonical. `API_VERSION`
moves to `3.0` and `EVENT_SCHEMA_VERSION` to `3.0`. Nothing breaks *outside* the
window — the flat `app.<method>` API, the 2.x organs, and every existing run path
stay fully supported; the new surface is `@experimental(since="3.0")`.

### Added

- **Unified declarative self-improvement contract** (`vincio.optimize.self_improvement`).
  One `SelfImprovementPolicy` composes scheduling, autonomous proposal, online
  updates, canary/rollback, active-learning label acquisition, and
  meta-optimization. `app.self_improvement(policy, dataset=...)` returns a
  `SelfImprovementController` whose `astream()` / `step()` / `run()` drive the
  existing `ImprovementLoop`, `ExperimentProposer`, `ContinuousImprovementController`,
  and canary as **one streaming engine**, emitting `observe → proposal → meta →
  label → reeval → canary → promote/rollback` events on the shared audit chain and
  event bus. Meta-optimization ships as `successive_halving` (over the
  strategy/budget grid) + `learn_fitness_weights`; active learning as
  `select_for_labeling`. Every promotion still passes the same significance +
  safety + golden non-regression gates.
- **Canary-gated deployment** — `app.deploy(candidate, dataset=...)` /
  `deploy_candidate` promote a prompt/policy live (registry push + tag + apply +
  audit) only on a no-regression `CanaryVerdict`, and refuse + roll back to the
  last known-good version otherwise. This is the canary-driven promotion surface
  reserved out of 1.10.
- **Provable erasure** — `app.erase_source(...)` now returns a signed,
  content-bound `ErasureProof` on `ErasureResult.proof`: a manifest of exactly
  which chunk / document / memory / **generated-artifact** ids were removed, bound
  by SHA-256 over the sorted removed-id set, signed with the app's
  `content_signer`, and anchored to the audit chain's Merkle root
  (`build_erasure_proof` / `verify_erasure_proof`). `LineageRecord` gains
  `artifacts` + `LineageIndex.record_artifact`, so an erased source is erased as
  evidence, memory, *and* generated output in one operation.
- **Consent & purpose modeling** (`vincio.governance.consent`) — a `ConsentLedger`
  binds a data subject to a GDPR `Purpose` and `LawfulBasis`
  (`grant` / `revoke` / `check`), persisted to the store and audited.
  `app.use_consent_ledger()` wires it into `AccessController.check_purpose` and
  memory recall, which drops any item whose purpose lost consent. `AccessDecision`
  carries `purpose` / `lawful_basis`.
- **Bi-temporal, ACL-gated memory** — `MemoryItem` gains `valid_from` / `valid_to`
  (+ `valid_at()`), a per-memory `acl` (+ `readable_by()`), and `purpose` /
  `consent_id`. `MemoryScope.TEAM` and `MemoryEngine.for_team(...)` add team-shared
  memory; `MemoryEngine.correct(...)` closes a fact's valid interval and opens a
  corrected one; `recall` / `asearch` accept `as_of=` (as-of recall, including
  superseded facts), `reader=` (ACL), and `team_id=`. SQLite persists the new
  columns and migrates a pre-3.0 store in place.
- **Async-canonical core & finalized telemetry** — `InMemoryMetadataStore` is now
  async-native (`asave` / `aget` / `aquery` / `adelete` / `acount`), so the
  module-level helpers take the native fast path with no worker-thread hop. The
  typed event catalog gains `SelfImprovementPhaseEvent`, `DeployCompleted`, and
  `SourceErased`; `EVENT_SCHEMA_VERSION` is `3.0`.
- `examples/38_self_improvement_and_provable_erasure.py`, the VincioBench `loop` /
  `governance` / `scale` / `memory` family checks for the above, eight new SLOs
  (**274 budgets, 85 SLOs**), and a runnable example smoke-tested offline.

### Changed

- `API_VERSION` → `3.0`; `vincio.__version__` → `3.0.0`.
- The public surface adds `SelfImprovementPolicy`, `SelfImprovementController`,
  `CanarySpec`, `DeployResult`, `ErasureProof`, `verify_erasure_proof`,
  `ConsentLedger`, `Purpose`, and `LawfulBasis` (plus the `vincio.optimize` /
  `vincio.governance` subpackage exports).

### Deprecated

- `app.continuous_improvement(...)` and `app.experiment_proposer(...)` are
  deprecated (`since=3.0`, `removed_in=4.0`) in favour of `app.self_improvement`.
  Both stay fully functional through the 3.x line; the underlying
  `ContinuousImprovementController` / `ExperimentProposer` classes remain public.

**1613 tests passing offline in ~7s; ruff + mypy clean.** The 3.0 milestone
carries no deferred items.

See [ROADMAP.md](ROADMAP.md) for the milestone framing.

## [2.2.1] - 2026-06-18

Closes the two honest scoping notes the 2.2.0 milestone shipped with. Both
additive and backward-compatible — `API_VERSION` stays `2.0`, and the default
`run` / `arun` / `replay` paths are unchanged.

### Changed

- **Token streaming is now genuine provider-driven streaming.**
  `AgentExecutor.astream` (and, through it, `Crew.astream`) route the
  answer-producing model calls through `provider.stream()`, emitting the
  provider's **real token deltas** as they arrive and reconstructing the final
  `ModelResponse` from the stream's `done` event — replacing the 2.2.0 post-hoc
  word-grouping of the finished text. Structured-output (schema) calls stay on
  `generate` (their JSON is not emitted as user-facing text). For the
  deterministic `MockProvider` this surfaces as real 16-char chunk deltas
  offline; for hosted providers it is true token-by-token streaming. VincioBench
  gates `families.agent.streaming.genuine_token_streaming` / `provider_deltas`.

### Added

- **A live-run path for the benchmark adapters — the identical scorer on fresh
  agent output, not just recorded replay.** `adapter.run(solver)` solves each
  task live and scores it with the same `score()` as `replay()`. `make_agent_solver`
  turns a `ContextApp` / `AgentExecutor` (or any callable) into a solver
  (`mode="text"` for an answer; `mode="calls"` captures the agent's function calls
  from its event stream for BFCL), and `make_env_solver(policy)` runs a policy
  through the τ-bench world. Official task sets load with `tasks_from_jsonl` and
  the per-benchmark `gaia_tasks_from_export` / `swebench_tasks_from_export` /
  `bfcl_tasks_from_export` (which parse the released field names, including
  SWE-bench's JSON-encoded `FAIL_TO_PASS` / `PASS_TO_PASS`). VincioBench gates
  `families.agentic_evals.environment_eval.adapters.live_run_scored`, exercised
  end to end offline against a real `AgentExecutor` (the agent genuinely calls
  tools; the BFCL AST scorer grades the calls it made). **258 budgets, 77 SLOs.**

## [2.2.0] - 2026-06-18

Prove it on the world's benchmarks: environment eval, agentic leaderboards, the
governed agent fabric, and generative UI. Entirely additive behind
`@experimental(since="2.2")` on the frozen 2.0 surface — `API_VERSION` stays
`2.0`, the single-process asyncio path stays the default, and nothing here is
required to run Vincio. All offline and deterministic; the benchmark adapters and
registry clients use only the core `httpx` dependency.

### Added

- **Stateful-environment eval harness + task-success oracle.** A new
  `vincio.evals.environment` ships an `Environment` protocol
  (`reset` / `step` / `observe` / `verify`), a deterministic in-process
  `ToolEnvironment` (whose world is a dict mutated by tools), a declarative
  end-state oracle (`StateCheck` / `TaskVerification`), and an
  `EnvironmentSimulator` that drives an agent *policy* through a *mutable* world
  and projects the interaction onto the existing `Trajectory` — scoring
  **verifiable end-state**, not turn-by-turn plausibility. `make_retail_environment`
  is a τ-bench-style reference world; `scripted_policy` / `task_success` round it
  out. Re-exported from `vincio` (`Environment`, `ToolEnvironment`,
  `EnvironmentSimulator`, `make_retail_environment`).
- **Agentic benchmark adapters (SWE-bench Verified / τ-bench / τ²-bench / GAIA /
  WebArena / BFCL).** `vincio.evals.benchmarks` ships one `BenchmarkAdapter`
  contract and the five adapters, each scoring the benchmark's own **verifiable
  end state** (SWE-bench's fail-to-pass/pass-to-pass transition, τ-bench's database
  end state via the environment oracle, GAIA's normalized exact match, WebArena's
  functional check, BFCL's AST match). Each pins its task set by a content hash
  (`task_set_hash()`, verified against the fixture on load) and degrades to
  recorded-fixture replay offline (`adapter.replay()`; fixtures in
  `benchmarks/fixtures/`); `BenchmarkReport.to_eval_report()` projects onto an
  `EvalReport` the Pareto optimizer consumes. `load_benchmark` / `available_benchmarks`.
- **Retrieval evaluation harness + index-version regression.**
  `vincio.evals.retrieval_eval` (`RetrievalEvaluator` / `RetrievalGoldenSet` /
  `RetrievalConfig`) benchmarks an embedder / reranker / chunker / index config on
  recall@k / nDCG@k / MRR / context-precision (reusing the retrieval metrics), and
  `retrieval_regression(...)` records a versioned artifact and gates a recall/nDCG
  regression on **the same significance test as a model swap** (`ab_test`).
  Artifacts persist through `vincio.storage.index_regression`
  (`IndexRegressionStore` / `IndexRegressionArtifact` / `config_key`), keyed on
  `(embedder, chunker, corpus hash)` over the `MetadataStore`.
- **The governed agent fabric (AGNTCY / ACP + MCP Registry).** `vincio.registry`
  ships an `AgentDirectory` (`AgentRecord` / `AgentResolution`) over the existing
  A2A Agent Card — `find` by capability/tag/query, `resolve` governed by an
  allow-list and recorded as an `agent_resolve` access decision on the audit chain.
  An **AGNTCY/ACP** (REST-native Agent Connect Protocol) adapter (`ACPClient` /
  `ACPAgentManifest` + `acp_to_agent_card` / `agent_card_to_acp`) and an **MCP
  Registry** discovery client (`MCPRegistryClient` / `MCPServerRecord`) discover
  agents/servers into the same directory under the same allow-list. A new
  `AllowListGate` (`vincio.security.access`) is a fail-closed reachability gate over
  `AccessController`; `app.agent_directory(allow=..., deny=...)` builds a directory
  wired to the app's audit chain. Re-exported from `vincio` (`AgentDirectory`,
  `AllowListGate`).
- **Generative UI / AG-UI streaming.** `vincio.server.agui` ships an AG-UI /
  MCP-UI compatible event protocol (`AGUIEvent` / `AGUIEventType`) and translators
  (`run_stream_to_agui`, `agent_stream_to_agui`, `agui_sse`), plus the SSE endpoint
  `POST /v1/apps/{app_id}/agui`. `AgentExecutor.astream(...)` and `Crew.astream(...)`
  now yield flat `AgentEvent` / `CrewEvent` streams (run/step lifecycle, real text
  deltas, `tool_call` / `tool_result`, a terminal `done` carrying the state/result)
  matching the `graph` / `compose` streaming surface — crew streams forward each
  member's tool/text events. `mcp.MCPUIResource` (`from_html` / `from_agui`) serves
  MCP-UI resources via `build_app_server(..., ui_resources=[...])` /
  `app.serve_mcp(ui_resources=[...])`. The interactive UI inherits the run's
  provenance, budget metering, and audit — one streamed run.
- **VincioBench guarantees.** New CI-gated checks fold into the existing families:
  environment task-success oracle + benchmark-adapter determinism in
  `agentic_evals.environment_eval`, retrieval-eval recall/nDCG + index-version
  regression in `rag.retrieval_eval`, the governed fabric (AGNTCY/ACP + MCP-registry
  discovery under the allow-list, audited resolution) in `protocols.fabric`, and
  token/tool-event + AG-UI streaming in `agent.streaming` — 255 budgets, 77 SLOs.
  New runnable example `37_benchmarks_and_fabric.py`.

### Notes

- Backward-compatible and additive: every new symbol is `@experimental(since="2.2")`
  and reachable through a new entry point; no existing API changes behavior. The
  benchmark adapters and reference environments are offline and deterministic and
  never reach the network; the agent fabric is governed by construction (fail-closed
  allow-list, every resolution audited); AG-UI streaming opens no new data-exposure
  boundary (it is a translation of the run's existing `astream`).

## [2.1.1] - 2026-06-18

Closes the three known limitations the 2.1.0 adversarial review surfaced. All
additive and backward-compatible — `API_VERSION` stays `2.0`, the single-process
asyncio path stays the default, and no existing graph, reducer, or backend
changes behavior.

### Added

- **Channel-default reducers — a map-reduce no longer needs a seed node.**
  `StateGraph(..., defaults={...})` (and non-required `state_schema` field
  defaults, inferred automatically) declare a reduced key's empty value, so the
  reducer folds the **first** write into that default instead of passing the raw
  value through. A `Send` map-reduce can now use a non-defensive reducer
  (`operator.add`) with no upstream node seeding the collected key. The legacy
  first-write passthrough is unchanged whenever no default is known, so existing
  bare-callable reducers keep their exact semantics. Defaults ride through
  `app.graph(defaults=...)` and survive `RayBackend` export. This replaces the
  2.1.0 workaround of seeding the collected key, at its root.
- **`vincio.testing.assert_backend_conformance`** (+ `conformance_cases`) — the
  offline contract every runtime backend must satisfy: it runs a battery
  (sequential, conditional routing, `Send` map-reduce with a channel default)
  through a backend and asserts it reproduces the native durable engine. The
  `RayBackend` / `TemporalBackend` export adapters — which can only be exercised
  against injected fakes offline — are now held to this contract, not merely
  "runs one graph," and a real cluster wiring can validate itself the same way.
  VincioBench's scale family gains `backend_conformant` and
  `map_reduce_no_seed_ok` budgets.

### Changed

- **The real local-neural-model paths are now exercised offline.**
  `SpladeEncoder`, `LocalCrossEncoderReranker`, and `FastEmbedEmbedder` accept an
  injected model object (`model=` / `tokenizer=` / `torch_module=`), mirroring
  `GGUFProvider(llama=...)`, so the real forward / `predict` / `embed` paths run
  against faithful fakes with the heavy deps absent. `SpladeEncoder.pool_logits`
  extracts the SPLADE log-saturated max-pool + top-k into pure, directly tested
  Python (the model forward stays in `torch`). The `# pragma: no cover` markers
  on those real-model paths are removed — they are covered now.

### Notes

- 1497 tests passing offline; ruff + mypy clean. VincioBench: 18 families, 231
  CI budgets, 71 SLOs. The 2.1 milestone now carries no deferred items.

## [2.1.0] - 2026-06-17

Scale out & train for real — distributed execution, executed fine-tuning, and a
served (still self-hosted) observability plane. Entirely additive behind
`@experimental` on the frozen 2.0 surface; `API_VERSION` stays `2.0`, the
single-process asyncio path stays the default, and nothing here is required to
run Vincio.

### Added

- **Distributed durable-execution backend.** `vincio.agents.distributed` adds a
  `GraphCoordinator` protocol (in-memory `InMemoryGraphCoordinator` +
  `RedisGraphCoordinator`) and a `DistributedCheckpointer` that lease-guards
  each graph thread (a TTL `running` lease) and CAS-commits every super-step
  (checkpoint-version optimistic concurrency), so two workers can never
  double-execute a step — the loser raises `CheckpointConflictError`. New
  runtime backends in `agents/backends.py`: `WorkerPoolBackend` (the in-process
  reference distributed executor, with `run_batch` fan-out) plus `RayBackend`
  and `TemporalBackend` export adapters (lazy/injectable, offline-testable). The
  durable graph gains true BSP parallel super-steps (`StateGraph.compile(parallel=True)`)
  and a `Send` primitive for map-reduce fan-out; `Workflow.map_step` adds
  data-dependent level-parallel spawning. Lease/CAS metadata rides the same
  checkpoint records, so a thread moves between the single-process and
  distributed backends without losing its ledger or trace.
- **Executed distillation & provider fine-tune jobs.** `vincio.providers.finetune`
  ships `OpenAIFineTuneBackend`, `GoogleFineTuneBackend`, and
  `AnthropicFineTuneBackend` (submit/poll/cancel) plus `run_finetune` and a
  `make_finetune_backend` factory. `optimize.provider_trainer` turns the
  `StudentTrainer` from an injected no-op into an executed trainer that submits a
  fine-tune job, registers the resulting model in the registry, and returns the
  trained model id; `BootstrapFinetune` gains an optional `swap_gate` so the
  student is promoted only past the significance gate. The export gains
  `semantic_dedupe` and a `max_example_chars` truncation guard. Offline, the job
  lifecycle runs against `httpx.MockTransport` cassettes and the promotion
  decision is fully deterministic.
- **Served observability & alerting plane.** `observability.IndexedTraceStore` is
  an indexed SQLite trace/cost store with time-bucketed cost rollups, retention
  (`purge`), and percentile/cost-by-dimension queries that replace O(n) JSONL
  scans. `observability.ViewerApp` + `serve_viewer` serve a dashboard, live trace
  tail, search, and JSON APIs over it using only the standard library. A new
  `AlertSink` protocol with `WebhookAlertSink` / `SlackAlertSink` /
  `PagerDutyAlertSink` / `PrometheusExporter`, plus an `AlertManager` rule engine
  (`AlertRule`: threshold / EWMA-Welford anomaly / SRE burn-rate) that runs over
  the cost ledger and event bus, and a `TailSamplingExporter` (error-prioritized,
  deterministic). The zero-dependency static viewer stays; this plane is opt-in
  and emits on the same audit chain.
- **Redis-backed shared server state + `vincio serve`.** `storage.shared_state`
  adds `RateLimiter` / `IdempotencyStore` protocols (in-memory defaults +
  `TenantQuotaManager`); `storage/redis.py` adds `RedisRateLimiter` and
  `RedisIdempotencyStore` so multi-worker deployments stay coherent. A first-class
  `vincio serve` launcher (uvicorn) plus `/v1/health/ready` and `/v1/metrics`
  (Prometheus), a lifespan with graceful shutdown, and an optional per-caller
  rate-limit middleware.
- **Content-capture controls.** `observability.ContentCapturePolicy` gates
  prompt/completion content at the export boundary — **off by default** — and
  redacts (PII) + truncates when opted in, before content reaches OTel events,
  JSONL, or the viewer. Wired into the OTel exporter and the tool runtime.
- **Quantization + two-stage retrieval.** `retrieval.quantization` adds
  `quantize_scalar` / `quantize_binary` and a `TwoStageIndex` (coarse search on
  quantized/Matryoshka-truncated vectors, exact rerank on full precision),
  reusing `mrl_truncate`. The Qdrant adapter accepts a native `quantization=`
  config.
- **Batteries-included local neural models** (optional deps, with deterministic
  offline fallbacks): `FastEmbedEmbedder` (ONNX/fastembed dense), `SpladeEncoder`
  (real SPLADE sparse), `ColBERTTokenEmbedder` (late-interaction tokens),
  `LocalCrossEncoderReranker`, and a native llama.cpp `GGUFProvider` with
  on-device embedding. New extras: `vincio[fastembed]`, `vincio[splade]`,
  `vincio[cross-encoder]`, `vincio[gguf]`, `vincio[local-neural]`.

### Quality & release

- **1485 tests passing offline in ~6s; ruff + mypy clean**; thirty-six runnable
  examples; VincioBench gates the 2.1 guarantees under CI budgets (229 budgets,
  71 SLOs): distributed durability + multi-worker shared-state coherence in
  `scale`, the executed-distillation swap-gate in `loop`, quantized two-stage
  recall in `rag`, and burn-rate/EWMA alerting in `cost`.

## [2.0.1] - 2026-06-17

Closes the one deferred 2.0 follow-up and a secret-scanning hygiene issue. No
public-API changes (`API_VERSION` stays `2.0`).

### Changed

- **Native filter pushdown now reaches every named backend.** Pinecone,
  Weaviate, Milvus, and Elasticsearch/OpenSearch persist flat filterable fields
  alongside the chunk blob (`flat_filter_fields`) and pass the compiled
  `FilterSpec` into the backend's native query (Pinecone metadata filter,
  Weaviate `where`, Milvus `expr`, ES/OpenSearch kNN `filter`), so tenant /
  document / kind / metadata scope is applied server-side — not only client-side.
  Each is verified offline against its fake (which now applies the pushed-down
  filter). The shared-or-mine tenant scope matches both null (in-memory) and the
  empty-string-stored untagged case so it is correct in-memory and natively.
  `PineconeVectorIndex` now lazy-imports its SDK only when building a real client
  (consistent with the other adapters), so an injected client works without the
  package.
- **Secret-scanning hygiene** — the synthetic OpenAI-key fixture used to exercise
  the egress DLP scanner (tests, example, benchmark) is now assembled at runtime,
  so no contiguous secret-shaped literal lives in source. It still trips the
  `sk-...` detector at scan time, which is the point of the test.

### Notes

- 1389 tests passing offline; ruff + mypy clean. VincioBench: 18 families, 218
  CI budgets, 65 SLOs. The 2.0 milestone now carries no deferred items.

## [2.0.0] - 2026-06-17

The one breaking window. Five milestones of additive growth exposed structural
debt the frozen 1.0 surface could not pay down. 2.0 is the single deliberate
breaking release — nothing breaks outside it — and it lands the flagship
multimodal-native Context Packet that genuinely needs the schema change. The
public-API contract (`API_VERSION`) moves to `2.0`.

### Added

- **Capability facades** — `ContextApp`'s surface is decomposed into six narrow,
  lazily-constructed, independently-testable views (`vincio.core.facades`):
  `app.runs` / `.knowledge` / `.governance` / `.optimization` / `.serving` /
  `.training`. Each exposes one cohesive method group and delegates to the app's
  implementation; reaching across a boundary raises `AttributeError`. Built on
  first access, so cold start and footprint scale with what an app uses.
- **Multimodal-native Context Packet** — `EvidenceItem` and `ContextCandidate`
  generalize from text-only to typed `modality` (`text` / `image` / `table`)
  with `image` (`ImageRef`) / `table` carriers and modality-aware token cost, so
  the compiler selects, budgets, orders, and cites image and table evidence in
  the same scored packet as text. Slim packets are backed by a content-addressed
  evidence store (`vincio.context.evidence_store`: `InMemoryEvidenceStore` /
  `BlobEvidenceStore`) so `ContextPacket.materialize(store=...)` recovers text
  after cross-process deserialization. The evidence ledger gains entailment
  `supports` / `contradicts` links (`link_entailments`).
- **Structured `FilterSpec`** (`vincio.retrieval.filters`) — a declarative,
  serializable filter (`eq` / `ne` / `in_` / `range_` / `exists` / `contains`
  over `and_` / `or_` / `not_`) compiled to each backend's native filter (Qdrant
  `Filter`, pgvector GIN-indexed `jsonb` `WHERE`, Pinecone, Weaviate, Milvus,
  Elasticsearch). Qdrant and pgvector push down server-side and fetch exactly
  `top_k`, fixing the over-fetch under-fill bug; `app.tenant_filter` returns a
  pushdown `FilterSpec` (shared-or-mine), closing the cross-tenant
  fetch-to-filter exfiltration risk.
- **Enterprise endpoints behind a pluggable `AuthStrategy`** — AWS Bedrock
  (pure-stdlib SigV4 Converse), Google Vertex (regional service-account OAuth),
  and Azure OpenAI (deployment routing + `api-version`), registered as
  `bedrock` / `vertex` / `azure` through the same `ProviderRegistry`, capability
  guards, swap gate, residency, and audit chain as every other provider.
- **Async-first storage + typed event catalog + unified telemetry** — an
  `AsyncMetadataStore` protocol with `aget` / `adelete` / `acount` alongside
  `asave` / `aquery` (native async or threaded shim) and a psycopg3
  `AsyncConnectionPool` Postgres fast path; a typed, versioned event catalog
  (`vincio.core.events`: `EVENT_CATALOG`, Pydantic payload models, `publish()`,
  `EVENT_SCHEMA_VERSION`); and one trace fanned out to spans **and** OTel metric
  histograms under the GenAI **agentic** conventions (`invoke_agent`,
  `gen_ai.agent.*`, `gen_ai.usage.cost`).
- **Mandatory egress DLP + signed audit chain** — `PolicyEngine.scan_egress`
  scans the fully-assembled provider request (system + messages + tool schemas)
  at both provider-dispatch boundaries regardless of call-site wiring
  (`security.egress_dlp`: `off` / `warn` / `block`); the hash-chained audit log
  gains per-entry HMAC/Ed25519 signatures and Merkle-root checkpoints
  (`security.audit_signing_key`), making it tamper-evident against a privileged
  attacker who can recompute the public hashes.

### Changed (breaking)

- **Eval metric semantics** — unscoreable cases (no ground truth, no claims, no
  trajectory) return `MetricResult(skipped=True)` and are excluded from gate
  aggregation instead of a neutral `1.0` that inflated means and silently passed
  gates. The lexical metric formerly named `semantic_similarity` is renamed to
  its true identity `lexical_overlap`; `semantic_similarity` is now a real
  embedding-backed metric (configurable via `set_semantic_embedder`).
- **`Index.search` `where` type** widens to `Where = FilterSpec | SearchFilter`;
  the `MetadataStore` async methods are the canonical contract.
- **`HTTPProvider` auth** is refactored behind `AuthStrategy` (`_prepare`); the
  audit-entry schema gains `signature` / `key_id` and `verify` validates them.
- `API_VERSION` → `2.0`; `EvidenceItem` / `ContextCandidate` carry `modality`.

### Notes

- 1386 tests passing offline in ~5s; ruff + mypy clean. VincioBench: 18 families,
  217 CI budgets, 65 SLOs. Thirty-five runnable examples. Every change is retired
  or introduced through the mechanical deprecation runway 1.0 established; the
  flat `app.<method>` API remains fully supported alongside the facades.

## [1.10.0] - 2026-06-17

The loop closes itself. Vincio could already *measure* drift and run an offline
optimizer, but the online loop only closed when a human pressed go. 1.10 makes
self-improvement continual, online, and safe — and opens the agentic frontier
(deep research, self-editing memory, computer-use) on the same cited, grounded,
audited spine. Everything is additive behind `@experimental` entry points on the
frozen 1.0 API; the canary-driven prompt/policy promotion that needs a new
serving surface stays reserved for 2.0.

### Added

- **Online improvement controller** — `app.continuous_improvement(...)`
  (`vincio.optimize.controller.ContinuousImprovementController`) subscribes to
  `drift.detected` + `eval.online`, streams online scores into a CUSUM
  changepoint detector, and turns a *sustained* signal into one of three gated
  actions: a targeted re-eval, a fresh `ImprovementLoop` run, or a rollback to
  the last known-good `prompts/registry.py` version. Per-trigger cooldown
  debouncing and a global eval budget bound it; every trigger, debounce,
  decision, and rollback lands on the hash-chained audit log and an event. State
  (budget spent, sustain counts, cooldowns) persists to the shared store, so the
  controller is restart-safe.
- **Distributional drift + CUSUM** — `evals/drift.py` gains two-sample
  Kolmogorov–Smirnov (`ks_statistic` / `ks_drift`), Population Stability Index
  (`psi`), RBF Maximum Mean Discrepancy (`rbf_mmd2`), and a streaming
  `CUSUMDetector`; `DriftMonitor.observe_score` feeds online scores into a
  per-metric CUSUM that fires `drift.detected` on a sustained shift (the event
  the controller acts on), with restart-safe persisted accumulators.
- **Restart-safe, worker-aggregatable online state** — `OnlineEvaluator`
  persists its 1-in-N sampling counter to the store (`online_state`) keyed by
  `worker_id`; `observed_total()` aggregates across workers.
- **Real provider-backed reflective optimizer (GEPA proper)** — `LLMReflector`
  (`optimize/reflective.py`) wired to the app's own provider reads the *actual*
  failing cases (input + output + expected + grounding), clusters them into
  failure modes (`cluster_failures`), and proposes targeted edits validated
  against the existing edit schema. `HeuristicReflector` stays the air-gapped
  deterministic fallback; `app.reflective_optimize(..., reflector="llm")` and
  `ImprovementLoop(reflector="llm")` opt in. Feeds the same Pareto frontier and
  gated promotion.
- **Autonomous experiment proposer** — `ExperimentProposer` /
  `app.experiment_proposer(...)` ranks where the system is weakest from online
  eval + drift and proposes/schedules the highest-ROI experiment (prompt /
  retrieval / budget / routing / distillation) under a global eval budget, every
  decision recorded.
- **Guarded online bandits** — a contextual `LinUCB` joins `EpsilonGreedyBandit`
  / `UCB1Bandit`, wired into the live route by `GuardedBanditRouter` (a
  `ModelProvider`) behind a **safety floor** (never explores on safety-/high-risk
  traffic), with persisted arm stats, cumulative regret, and auto-freeze /
  rollback-to-safe-arm on regression. `app.use_bandit_router(...)`.
- **Held-out, growing golden regression suite** — `GoldenRegressionSuite`
  (`evals/datasets.py`) records the cases each promotion fixes with provenance
  and gates every later promotion by replay, so sequential auto-promotions can
  never silently undo a prior fix; wired into `ImprovementLoop(golden_suite=...)`.
- **Deep-research agent** — `ResearchAgent` / `app.research(...)` loops
  search → read → reflect → verify → synthesize over the query-understanding
  planners and the grounded-fact extractor under explicit breadth/depth/source/
  token budgets, dedups sources, verifies with judges, and emits a cited report
  through the 1.9 `CitedReportBuilder` — every claim cited and grounded by
  construction, scored for citation coverage / grounding / source diversity.
- **Agent memory OS** — `MemoryOS` / `app.enable_memory_os(...)` exposes
  self-editing memory as permissioned, audited tools (`memory_append` /
  `memory_replace` / `memory_search` / `memory_archive`) over the existing
  guarded write pipeline, with a context-pressure pager between in-context core
  memory and the archival store.
- **In-loop context compaction** — `agents/compaction.py` `ContextCompactor`
  folds old tool/observation turns into a rolling extractive summary at a token
  budget, replacing the fixed `[-8]`/`[:24]` slicing in `agents/executor.py`
  (DAG and ReAct paths), keeping tool-call pairs intact.
- **Level-parallel agent DAG + `plan_and_execute`** — the executor runs each
  topological level's independent steps concurrently (bounded), and
  `Planner.replan` drives a real plan → execute → observe → replan loop for the
  `plan_and_execute` mode.
- **Computer-use / agentic browsing** — `tools/computer_use.py` adds a
  navigate / click / type / screenshot action vocabulary with a deterministic
  `MockComputerUse`, a `PlaywrightComputerUse` backend, and a provider-native
  adapter, exposed via `app.enable_computer_use(...)` as permissioned, audited,
  approval-gated tools.
- **Pluggable isolation backends** — `tools/sandbox.py` gains an
  `IsolationBackend` interface with `Subprocess` (zero-dep default, not a
  security boundary), `Container`, `gVisor`, `microVM`, and `WASM` backends;
  `require_real_isolation` enforces that code-executing and computer-use
  workloads run behind a real boundary.
- **Provider-native hosted tools** — `providers/hosted_tools.py` surfaces OpenAI
  Responses built-ins (`web_search` / `file_search` / `code_interpreter` /
  `computer_use`) as namespaced, permissioned Vincio tools
  (`app.use_hosted_tools(...)`); the Responses adapter emits each as its
  built-in descriptor. `computer_use` is approval-gated.
- New error `SandboxError`; new optional extra `vincio[computer-use]`
  (Playwright); `examples/34_continual_loop_and_agentic_frontier.py`.

### Notes

- 1304 tests passing offline in ~5s; ruff + mypy clean. VincioBench: 17 families,
  205 CI budgets, 60 SLOs. Thirty-four runnable examples.

## [1.9.1] - 2026-06-17

Closes the two thin spots in the 1.9 generation surface so the milestone carries
no deferred follow-ups. Additive and `@experimental`; no public symbol removed.

### Changed

- **Forms Document-AI cloud adapters are now real, dependency-injected
  implementations** instead of `NotImplementedError` stubs. `TextractDocumentAI`,
  `AzureDocumentAI`, and `GoogleDocumentAI` take the SDK client you build and run
  the real `analyze_document` / `begin_analyze_document` / `process_document`
  calls in a worker thread; the response→`FormField` parsing (key/value text,
  confidence, page, and a bounding box) is a pure `parse(...)` function tested
  offline against synthetic responses (no SDK is a hard dependency).
- **Embedded PNG C2PA credentials are now self-verifying.** `embed_provenance`
  binds the embedded credential to the pre-insert bytes; the new
  `extract_embedded_manifest` / `verify_embedded_manifest` reconstruct the
  original bytes by removing the `c2pa.manifest` chunk and confirm the digest, so
  an extracted credential is independently verifiable against the file it travels
  in (a tampered asset fails). The sidecar / returned manifest still bind the
  final bytes.

### Added

- Offline tests for each cloud Document-AI parser, the self-verifying embedded
  credential (incl. tamper rejection), and the optional-dependency error
  messages (PPTX render / Parquet / `.msg` raise a clear install hint when the
  extra is absent). A `families.generation.media.embedded_self_verifies`
  VincioBench budget.

### Notes

- 1196 tests passing offline; ruff + mypy clean. VincioBench: 17 families,
  173 CI budgets, 51 SLOs.

## [1.9.0] - 2026-06-17

Documents & images flow OUT — cited, governed, eval-gated artifacts. Vincio could
read a DOCX, a PDF, and a scanned packet and validate a JSON answer, but stopped
one step short of the deliverable. 1.9 closes the documents-/images-out loop: a
document-generation engine, cited-report assembly, image-generation/editing and
TTS as first-class output modalities, OCR/transcript/figure inputs, new-format
loaders, and an EU AI Act conformity pack — every produced asset cited,
provenance-stamped, budget-metered, and audited on the same chain as text.
Entirely additive behind a new `vincio.generation` subpackage, new `vincio[...]`
extras, and `@experimental` markers on the frozen 1.0 API; no public symbol
removed or repurposed.

### Added

- **`vincio.generation` document engine.** `DocumentBuilder` turns a *validated*
  result (an `OutputContract` output, a `RunResult`, a structured mapping, or
  Markdown) into rendered artifacts — Markdown/HTML dependency-free, DOCX
  (`vincio[gen-docx]`), PDF (`vincio[gen-pdf]`), PPTX (`vincio[gen-pptx]`) — via a
  format-neutral `DocumentModel` IR. Because the input already passed validation,
  the document is grounded by construction. Structural `DocumentContract`
  (required sections, `TableSpec` column specs, length bounds, citation-per-
  section) validates the result with **formatting-only repair** (`repair_formatting`)
  mirroring the JSON-repair path; every render records a `document_generate` audit
  event with the source evidence ids. Adds template/form filling
  (`fill_text_template` / `fill_docx_form` / `fill_pdf_form`, typed citation-aware
  `Slot`s) and `generate_redline` (tracked-change DOCX, `**ins**`/`~~del~~` text).
- **`CitedReportBuilder`.** Resolves inline `[E1]`-style markers to numbered
  footnotes/endnotes and a generated bibliography with per-claim provenance,
  computes sentence-level **citation coverage**, and optionally verifies
  **per-claim entailment** (pluggable backend; strict lexical+numeric default).
  A `CitationContract` enforces a coverage floor, rejects unresolved markers, and
  gates on entailment — replacing the flat "one valid citation anywhere" check.
  New `citation_coverage` and `claim_entailment` eval metrics.
- **Image generation/editing provider abstraction.** `ImageProvider` with
  `generate_image` / `edit_image` / `variation`, a neutral `ImageGenRequest` /
  `ImageGenResponse`, backends for OpenAI `gpt-image-1`, Gemini/Imagen, and a
  generic HTTP/Replicate adapter, plus a `MockImageProvider` that emits real PNGs
  offline. Every asset auto-attaches a media-aware C2PA manifest bound to its
  bytes, is metered against the budget, and is audited (`image_generate`).
- **TTS / speech-synthesis output modality.** `SpeechProvider` with
  `synthesize_speech`, a neutral `SpeechRequest` (voice/format/speed), backends
  for OpenAI TTS, Gemini TTS, and ElevenLabs/Cartesia, plus a `MockSpeechProvider`
  that emits real WAVs. Audio provenance + budget metering + audit
  (`speech_synthesize`), unified with the realtime audio path.
- **Audio as chat input.** `ContentPart.audio` is now rendered by the OpenAI
  (`input_audio`) and Gemini (`inlineData`) chat providers via a shared
  `core.media.encode_audio_bytes`, activating the already-typed `AudioRef` outside
  the realtime WebSocket path.
- **Media-aware synthetic-content marking.** `mark_synthetic_content` accepts
  `str` *or* `bytes` (binds by SHA-256), marks edits with
  `compositeWithTrainedAlgorithmicMedia`, and records the asset's media type.
  New `embed_provenance` (PNG metadata, dependency-free, with an invisible-
  watermark hook) and `write_sidecar_manifest` (a `*.c2pa.json` for any format).
- **Richer document inputs.** OCR auto-fallback in `load_pdf` (low-text pages
  rasterized + OCR'd, `extractor='ocr'` per page, `vincio[ocr]`); `load_media` for
  audio transcript ingestion via a `Transcriber` protocol
  (`MockTranscriber` / `WhisperTranscriber` / `ProviderAudioTranscriber`);
  `figure_evidence` turning PDF figure crops into citable evidence with bounding
  boxes; a real-parser HTML path (`parse_html`, table extraction) and structured
  JSON/JSONL/YAML (`structure_data`).
- **New format loaders + parser registry.** Dependency-free PPTX/EPUB/RTF/ODT,
  plus Parquet (`vincio[parquet]`), mbox, and `.msg` (`vincio[msg]`), behind a
  unified `ParserRegistry` (`register_loader`) that replaces the if/elif suffix
  chain. Forms/KYC extraction via a `DocumentAI` protocol (Textract / Azure /
  Google adapters) and an offline `HeuristicFormExtractor`, returning `FormField`s
  with confidence (+ bbox) convertible to evidence (`form_fields_to_evidence`).
- **EU AI Act conformity pack.** `RiskTierClassifier` (advisory risk-tier
  placement), `AnnexIVBuilder` (cited Annex IV technical documentation), and
  `FRIAGenerator` (Article 27 fundamental-rights impact assessment) — all
  generated from the live config, cards, compliance matrix, and eval/red-team
  evidence through the document engine, recorded as `conformity_doc` audit
  events (`app.risk_tier` / `app.annex_iv` / `app.fria`). An ISO/IEC 42001
  control catalog joins the `ComplianceMapper` family.
- **App methods** (all `@experimental`, since 1.9): `build_document`,
  `cited_report` / `acited_report`, `generate_image` / `agenerate_image`,
  `synthesize_speech` / `asynthesize_speech`, `load_media`, `risk_tier`,
  `annex_iv`, `fria`.
- **VincioBench `generation` family** + three SLOs and CI budgets covering
  document-contract validity, cited-report coverage + entailment, media-provenance
  binding/disclosure, redline correctness, new-format ingestion recall, and
  generated-media prompt safety. New `examples/33_documents_and_media_out.py`.

### Changed

- `ComplianceFramework` gains `ISO_42001` and `EU_AI_ACT`; `CONTROL_CATALOG` adds
  ISO/IEC 42001 controls (so the compliance matrix now spans five mapped
  frameworks). `ModelCapabilities.output_modalities` is the idiomatic generation-
  capability flag.

### Notes

- 1189 tests passing offline in ~5s; ruff + mypy clean. VincioBench: 17 families,
  172 CI budgets, 51 SLOs. No deferred follow-ups.

## [1.8.1] - 2026-06-17

Closes the two deliberately-scoped follow-ups documented at 1.8.0, so the
milestone carries no deferred items. Additive under the frozen 1.0 API; no public
symbol removed or repurposed.

### Changed

- **Residency is now a run-boundary choke point over *every* reachable model.**
  Previously `app.use_router` / `shadow` / `canary` / `use_cascade` validated their
  candidate models against the residency policy at wiring time, but a run that
  picked a different candidate per request was only checked for the primary model
  at the choke point. `check_residency` now enumerates the full reachable set —
  the configured/per-run model, any budget-degrade target, every cascade rung, and
  the candidates of a `Router` / `ShadowProvider` / `CanaryRouter` wrapper — and
  refuses egress for any disallowed-region model, on the same hash-chained audit
  path. Wiring-time enforcement stays as a fail-fast. A no-op when no residency
  policy is configured (the default), so there is zero overhead otherwise.

### Added

- **Recorded-cassette tests for the `GoogleBatchBackend` wire format.** The full
  Gemini Batch Mode lifecycle (submit → poll → results → cancel) is now exercised
  offline against an `httpx` mock transport returning recorded Gemini-shaped
  responses — asserting the request URL/path, the inlined-request envelope keyed
  by `custom_id`, `BATCH_STATE_*` status mapping, response parsing through the
  provider's own parser, reconciliation, and half-cost billing — so the wire
  handling is verified without a live endpoint. The backend docstring now scopes
  it precisely to the Google Developer API (Vertex AI's service-account + GCS
  batch surface lands with the 2.0 enterprise endpoints).

### Notes

- 1104 tests passing offline; ruff + mypy clean. VincioBench unchanged
  (159 budgets, 48 SLOs), all green.

See the [roadmap](ROADMAP.md) (1.8 milestone).

## [1.8.0] - 2026-06-17

Turns the 1.7 model registry into a **rotation-and-regression discipline** — the
migration safety net for the single most common and riskiest production change, a
model swap. Capability guards refuse to substitute a model that cannot serve the
request; a `SwapGate` replays golden traces and runs an eval + cost + latency +
behavioral diff with statistical backing on every candidate; a shadow provider
and a capped canary qualify a model on live traffic with automatic rollback; and
a lifecycle watcher proposes migrations off deprecated models. Every piece is
pure composition of 1.7 organs (the registry, `ReplayRunner`, `ab_test`,
`DriftMonitor`, `evaluate_gates`, the cost model). All additive behind
`@experimental` entry points on the frozen 1.0 API; nothing changes for callers
who do not opt in.

### Added

- **Capability-aware routing preflight + cost/latency `Router`.** A new
  `vincio.providers.capabilities` module (`requirements_for`, `capability_check`)
  intersects a request's needs (vision, tool calling, structured output,
  reasoning, context length) with a candidate's `ModelCapabilities`. A registry-
  backed `Router` (`vincio.optimize.routing.Router`, also re-exported from
  `vincio`) picks the cheapest / fastest / least-busy *capable* model per request,
  load-balances across equivalents, and **downgrades** to honor a per-request
  budget, emitting a `model.routed` decision. Wire it with `app.use_router(...)`.
- **Capability + lifecycle guard on failover & cascades.** `FailoverChain` and
  `HealthAwareFailover` now (by default, opt out with `guard_capabilities=False`)
  skip a capability-mismatched substitution instead of returning a silently wrong
  answer, classify a **terminal lifecycle/config error** (retired/removed/unknown
  model) distinctly from a transient outage (`is_lifecycle_error`), and surface
  `ModelRetiredError` ("rotate now") when every candidate is retired. The runtime
  cascade starts on, and escalates only into, a capable rung. New errors
  `CapabilityMismatchError` / `ModelRetiredError`. Unknown models are never blocked.
- **`SwapGate` + model-swap regression.** A new `vincio.evals.swap` module:
  `SwapGate` (`app.gate_swap(...)` / `vincio providers regress`) replays golden
  traces and runs `evaluate_gates` + `DriftMonitor` + `ab_test` with behavioral
  shape diffs (tool-call rate, refusal rate, output-length distribution) into a
  PASS/FAIL verdict with p-value and effect size; `model_swap_regression`
  (`app.swap_regression(...)` / `vincio eval regress --baseline-model X
  --candidate-model Y`) holds prompt/data/config fixed, swaps only the model, and
  reports per-metric significance, per-case deltas, the cost/latency trade, and
  the worst-regressed slices.
- **Flake control on `EvalRunner`.** `repeats=N` runs each case N times with
  per-case mean/stdev and configurable `repeat_aggregate`; `flake_quarantine`
  tags noisy cases and excludes them from gate aggregation so non-mock provider
  variance never flips a gate on a single run.
- **Shadow provider + progressive canary with auto-rollback.** `ShadowProvider`
  returns the primary's response while asynchronously dual-dispatching the
  candidate and recording both for offline diff; `CanaryRouter` ramps a configurable
  percentage of traffic to a candidate, scores both arms online, and
  auto-rolls-back to the last known-good model (and prompt-registry head) on
  regression. Both implement `ModelProvider`, so they nest inside `CircuitBreaker`
  / `KeyPool`. Wire with `app.shadow(...)` / `app.canary(...)`.
- **Lifecycle watcher + migration proposals.** `LifecycleWatcher`
  (`app.watch_lifecycle(...)` / `vincio providers lifecycle`) emits early sunset
  warnings and proposes a migration — to a model's declared successor or a cheaper
  Pareto-dominating, at-least-as-capable model — that can rewrite a
  `ModelCascade` / `RoutingPolicy` / `config.model` in place.
- **Live model discovery + Google/Vertex batch parity.** `ModelProvider.list_models`
  (implemented for OpenAI/Anthropic/Google) + `ModelRegistry.reconcile` and
  `discover_models` (`vincio providers discover`) reconcile a provider's live model
  list into the registry offline-safe. A `GoogleBatchBackend` joins
  `providers.batch`, and Google models gain batch-tier pricing, completing
  half-cost batch parity with OpenAI/Anthropic.
- **CLI.** `vincio eval regress` and a new `vincio providers` group
  (`list` / `lifecycle` / `discover` / `regress`).

### Notes

- 1090 tests passing offline in ~4.5s; ruff + mypy clean. Thirty-two runnable
  examples (`examples/32_swap_regression.py` swaps a model end to end through the
  gate and a canary). VincioBench extended in the `reliability`, `cost`, `evals`,
  and `scale` families (159 budgets, 48 SLOs), all green.
- Backward compatible. The one intentional, non-breaking behavior change: failover
  chains guard capabilities by default — they skip a *known-incapable* model and
  try the next capable one rather than attempting a substitution that would drop
  content. Unknown models are never blocked, and `guard_capabilities=False`
  restores the pre-1.8 attempt-everything behavior.

See the [roadmap](ROADMAP.md) (1.8 milestone).

## [1.7.1] - 2026-06-17

Closes the one documented 1.7 known limitation: the intermittent
`test_improvement_loop_reflective_promotes` flake. Additive under the frozen 1.0
API; no public symbol removed or repurposed.

### Fixed

- **Reflective optimizer honors `FitnessWeights` when building its Pareto
  frontier.** `ReflectiveOptimizer` accepted `weights` but always selected over
  the full `DEFAULT_OBJECTIVES`, so an axis the caller weighted to `0.0` still
  reached multi-objective selection. For wall-clock `latency`, that let timing
  jitter flip the knee point between otherwise-tied candidates — the root of the
  intermittent `test_improvement_loop_reflective_promotes` failure (it surfaced
  hash-seed/ordering-sensitively at the frontier-selection step). A new
  `objectives_from_weights()` helper derives the frontier axes from the weights
  (dropping zero-weighted axes, and tracking the configured `accuracy_metric`),
  and the reflective optimizer defaults to it when no explicit `objectives` are
  given. Selection is now deterministic when latency is weighted out, so screening
  fitness and frontier selection agree on which axes matter. An explicit
  `objectives=` argument still overrides; default weights keep all four axes.

### Notes

- 1039 tests passing offline; ruff + mypy clean. The single known limitation
  documented in the 1.7.0 release (the reflective-optimizer flake) is now closed
  at its root cause rather than worked around.

See the [roadmap](ROADMAP.md) (1.7 milestone).

## [1.7.0] - 2026-06-17

Makes the spine honest and fast, and lays the model-registry foundation. The
advertised `Budget` becomes a hard cap, the 1.5 embeddings are wired into the
compiler so selection is semantic instead of bag-of-words, the streaming and
non-streaming run paths are unified, persistence moves off the event loop,
local-image input is fixed, and a data-driven `ModelRegistry` finally consumes
the underused `ModelProfile`. Every change is additive behind a new entry point
or opt-in flag on the frozen 1.0 API, all `@experimental`; promotions are now
gated on statistical significance instead of a point estimate.

### Added

- **Enforced full Budget on the single-shot run path.** `max_cost_usd` /
  `max_input_tokens` / `max_output_tokens` / `max_steps` are now hard caps on
  `app.run()` / `arun()`: a `BudgetUsage` is threaded through the model+tool loop
  and `exceeds()` is checked after each model call and tool round, raising the
  (previously dead) `BudgetExceededError` at the same choke point as residency
  and the cost SLO — recorded on the audit chain (`budget` decision) and the
  `budget.exceeded` event. A pre-flight input-token estimate is checked before
  the first call, and `BudgetAllocator` can reserve response + tool-loop tokens
  so it accounts for the full window. `RunConfig(enforce_budget_caps=False)`
  preserves the pre-1.7 soft-cap behavior for one minor.
- **Data-driven `ModelRegistry`** (`vincio.providers.registry`, exported as
  `ModelRegistry` / `default_model_registry`). A versioned, hot-reloadable,
  config-overridable catalog keyed by exact model id, instantiating
  `core.types.ModelProfile` (now carrying batch/cache pricing tiers, modalities,
  and GA/deprecation/retirement lifecycle dates). `ModelProvider.capabilities()`
  and `observability.costs.PriceTable` derive from it, with substring sniffing
  demoted to a last-resort fallback; an unknown-model lookup warns
  (`ModelUnknownWarning`) and emits `model.unknown` instead of silently billing
  $0. `importlib.metadata` entry-point groups (`vincio.providers` /
  `vincio.embedders` / `vincio.stores`) let third parties ship auto-registering
  adapters, and provider-native exact token counters register behind the
  `TokenCounter` Protocol (`register_token_counter`). Overlay a catalog with
  `VINCIO_MODEL_REGISTRY=<path.json|yaml>`.
- **Opt-in semantic context scoring** (`app.use_semantic_context_scoring()` /
  `retrieval.semantic_context_scoring`). When a real embedder is configured,
  context relevance, novelty, dedup, and conflict use cosine over the cached
  embeddings, the reranker's `upstream_relevance` is blended into relevance (no
  longer just a gate), and `_select` runs embedding-cosine maximal-marginal
  relevance with an `mmr_lambda` trade-off. The default stays lexical.
- **Value-level contradiction.** The compiler's negation-XOR conflict trigger is
  replaced by a salient-unit value-disagreement check: same-topic evidence that
  cites different numbers/dates (or flips polarity) is emitted as a structured
  conflict delta in the packet.
- **`RunHandle` + cooperative cancellation.** `app.submit(...)` returns a
  `RunHandle` whose `cancel()` propagates a cancellation into the run's
  bounded-concurrency groups; the cancelled run is still fully recorded on its
  trace and audit chain (a `CANCELLED` epilogue both run paths share). The
  streaming path gained the same `asyncio.timeout` latency deadline as the
  non-streaming path.
- **Async store contract.** `storage.base.asave` / `aquery` run a store's
  `save`/`query` off the event loop (`to_thread` for sync stores, native
  `asave`/`aquery` when present); the runtime now persists packets and runs
  without blocking the pipeline.
- **Significance-gated promotion.** `evals.experiments.ab_test` now returns a
  confidence interval and Cohen's-d effect size alongside the p-value, and the
  shared `evolution_loop` calls the t-test at the gate: a statistically
  significant regression on the primary metric blocks promotion, and an
  under-powered or non-significant gain is warned. The `loop_promotion` audit
  record carries the verdict.
- **Trace-replay executor.** `evals.replay.ReplayRunner(app).replay(traces,
  pin_tools=...)` re-runs captured trace inputs through a target app and diffs
  outputs, trajectory (`trace_diff`), and cost (`EvalReport.diff`), optionally
  pinning recorded tool outputs for determinism. Surfaced as `vincio trace
  replay --against <app>`.
- **Pluggable detector backends.** `security.DetectorBackend` / `DetectorSpan`
  let an ML model merge with the deterministic PII / injection / secret
  detectors; passing none keeps detection byte-for-byte unchanged.
- A new runnable example, `examples/31_honest_fast_spine.py`, and VincioBench
  metrics + CI budgets/SLOs across the **cost**, **rag**, **reliability**,
  **perf**, and **loop** families (budget-cap enforcement, unknown-model
  warning, embedding-MMR + value-contradiction, stream/non-stream parity,
  cancellation recording, inverted-index BM25, token memoization, registry
  lookup, significance-gated promotion, and replay fidelity).

### Changed

- **OpenAI local-image input is fixed.** A local image path is base64-encoded
  into a `data:` URL instead of an unreachable `file://` URL, via one shared
  `vincio.core.media` helper (with a byte-size cap) reused by the OpenAI,
  Anthropic, and Google chat providers and the multimodal embedders. Google
  also accepts Google-hosted image URIs (GCS `gs://` or the Files API host) via
  Gemini `fileData` (arbitrary public URLs, which Gemini cannot fetch, are not
  sent — supply a local path to inline them).
- **Truthful protocol capabilities.** A2A agent cards default
  `capabilities.streaming=False` until `message/stream` is actually dispatched;
  the MCP client's task-poll busy-loop is replaced with exponential backoff and a
  wall-clock deadline; and the A2A client polls `submitted`/`working` tasks to a
  terminal state instead of mis-reporting them as failed.
- **Hardened injection defense.** The injection detector runs a normalization +
  decode pre-pass (NFKC fold, zero-width strip, homoglyph/leetspeak fold,
  recursive base64/hex/rot13 decode, depth- and size-bounded) before its regex
  and heuristic signals, catching obfuscated attacks with no new false positives.
- **Tenant isolation can fail closed.** `AccessController(require_explicit_tenant
  =True)` stops treating an untagged (`tenant_id=None`) resource as globally
  readable — closing a cross-tenant fail-open. Defaults to the legacy behavior
  for one minor.
- **Evidence-gated compliance.** `ComplianceMapper` reads a control as `covered`
  only when backed by measured red-team / eval evidence; a configured-but-
  unmeasured control is now `partial` (structural, by-construction guarantees
  stay `covered`).
- **Sub-quadratic hot paths.** BM25 search scans inverted posting lists instead
  of every document per query term, `_select` selects incrementally (O(n·k))
  with inverted-index blocking for dedup/conflict, `count_tokens` is memoized,
  and the local vector index gains an optional numpy path — all behind
  availability checks, pure-Python staying the zero-dependency default.
- **1034 tests passing offline in ~4.5s; ruff + mypy clean**; thirty-one runnable
  examples; the VincioBench `cost` / `rag` / `reliability` / `perf` / `loop`
  families hold the 1.7 guarantees under CI-gated budgets.

## [1.6.1] - 2026-06-16

Completes the 1.6 governance follow-ups (no gaps): a real type-check gate, a
stronger residency control, and signable content credentials. Additive and
backward-compatible.

### Added

- **mypy is now a CI gate.** A new `Types (mypy)` job runs `mypy vincio` on every
  PR; the whole package type-checks clean (0 errors across 230 modules). Fixing
  the type errors this surfaced also hardened several latent issues — a
  mislabeled `HealthAwareFailover._ordered` return type, a `StateGraph` frontier
  dedup that used `set.add` as a value, an unguarded `anomaly_factor` multiply,
  and tightened `evidence_ids` / event-handler / finish-reason typing.
- **Residency endpoint-region inference.** `ResidencyPolicy` now infers the
  provider region from a region-bearing endpoint URL (AWS `us-east-1`-style,
  GCP/Vertex `europe-west4`-style, and sovereign-gateway jurisdiction
  subdomains) via the new `infer_region_from_url`, and run egress checks read the
  configured `provider.base_urls`. Matching is jurisdiction-aware:
  `allowed_regions=["eu"]` admits `eu-west-1` and `europe-west4`. Combined with a
  region-pinned endpoint, the egress-refusal control now reflects the real
  endpoint rather than only a hand-maintained map.
- **Signable synthetic-content manifests.** `mark_synthetic_content(...,
  signer=...)` attaches a cryptographic signature over a deterministic binding
  payload, and `verify_manifest(manifest, content, signer=...)` checks both the
  SHA-256 content binding and the signature (failing closed when a signature is
  present but no verifier is supplied). A dependency-free `HmacSigner`
  (HMAC-SHA256 over a `SecretString`) ships built in; supply your own
  `ContentSigner` for asymmetric, third-party-verifiable provenance.
  `app.content_signer` signs every auto-marked run.

### Changed

- `vincio.governance` gains `infer_region_from_url`, `ContentSigner`,
  `HmacSigner`, and `verify_manifest` (additive). `EvalRunner` and
  `gather_bounded` accept `Sequence` inputs; `PIIDetector(locales=...)` accepts
  any `Sequence`. `GovernanceConfig.card_format` is now a validated `Literal`.

### Quality

- **986 tests passing offline; ruff clean; mypy clean; VincioBench 131/131
  budgets** (two new governance budgets gate residency inference and signature
  verification); thirty runnable examples.

## [1.6.0] - 2026-06-16

Enterprise governance & compliance. Turns the audit and security spine into the
evidence regulated buyers require — model/system cards, OWASP/NIST/MITRE control
coverage, an AI-BOM, EU AI Act transparency artifacts, data lineage with
right-to-erasure, data-residency routing, multilingual PII, and RAG-poisoning
detection — all generated in the library from data Vincio already holds.
Additive behind `@experimental` 1.6 entry points on the frozen 1.0 API,
dependency-free; no public symbol removed or repurposed.

### Added

- **Model & system cards.** `vincio.governance.generate_model_card` /
  `generate_system_card`, `app.model_card()` / `app.system_card()`, and `vincio
  governance card` generate machine-readable cards from the live configuration
  and optional `EvalReport` evidence. A model card carries id/version,
  capabilities, limitations, and live pricing; a system card adds retrieval,
  memory, safety filters, human-oversight points, and governance controls. The
  schema is pluggable (`CardFormat`: Vincio native, Open Model Card, EU "AI
  Cards") and rendered from one captured fact set.
- **Compliance-framework mapping.** `ComplianceMapper` / `map_compliance` /
  `app.compliance_report()` / `vincio governance report` map a data-driven
  control catalog (`CONTROL_CATALOG`) for **OWASP LLM Top 10 (2025)**, **OWASP
  Agentic AI**, **NIST AI RMF (GenAI profile)**, and **MITRE ATLAS** onto
  Vincio's capabilities, backed by measured evidence — `RedTeamSuite` probe
  outcomes, the security configuration, and `EvalReport` metrics. The
  `ComplianceReport` is a `covered`/`partial`/`not_covered` matrix with the
  evidence string for each control, `coverage_rate`, `by_framework()`, `gaps()`,
  and `to_markdown()`.
- **AI-BOM.** `generate_aibom` / `app.aibom()` / `vincio governance aibom`
  produce a CycloneDX-1.6 AI bill of materials (base model + version,
  embedding/rerank models, fine-tune datasets, prompt/registry versions) as
  `machine-learning-model` / `data` components with optional **SHA-256 hashes**;
  `sha256_file` / `sha256_text`, `AIComponent.verify`, and `AIBOM.verify_all`
  support blast-radius assessment. Complements the shipped dependency SBOM + SLSA
  provenance.
- **EU AI Act transparency.** `mark_synthetic_content` emits a C2PA-style
  `ProvenanceManifest` (IPTC `trainedAlgorithmicMedia`, bound to the output by
  SHA-256), `ai_disclosure` returns a localized AI-interaction disclosure, and
  `data_summary` exports a grounding-data summary. `governance.content_marking`
  (or `app.content_marking`) attaches the manifest + disclosure to every run's
  `result.metadata`.
- **Data lineage & erasure-by-source.** A `LineageIndex` records source →
  document → chunk → evidence → output as the app ingests and runs
  (`app.trace_lineage(...)`); `app.erase_source(...)` / `vincio governance erase`
  satisfies a GDPR right-to-erasure across **every index, memory, and cache**,
  logged on the hash-chained audit chain (`erase_source`) and idempotent. Returns
  an `ErasureResult`.
- **Data-residency-aware routing.** `ResidencyPolicy` / `app.set_residency(...)`
  / `governance.allowed_regions` pin allowed provider regions and **refuse
  egress** to others as a blocking `PolicyViolation` recorded as a
  `residency_check` deny (raising `ResidencyViolationError`), enforced at the
  provider-resolution choke point before any request leaves the process.
- **Multilingual PII.** Non-English locale packs (`vincio.security.locales`:
  France, Germany, Spain, India, Singapore, Brazil, UK national-ID and phone
  formats) via `PIIDetector(locales=[...])`, `available_locales`,
  `get_locale_pack`, and `governance.locales` — layered on the English path
  without changing it (`PIIMatch.type` widened to `str` with a `locale` tag;
  built-in `PIIType` unchanged and still accepted).
- **Per-language eval slicing.** `EvalReport.slice`, `slice_by_tag`, and
  `tag_gap` surface the high-vs-low-resource accuracy gap so it can't hide in an
  aggregate.
- **Tokenizer fertility telemetry.** `FertilityTracker` / `app.fertility` track
  tokens-per-word/char per language and tenant, exposing the non-English "token
  tax" (`token_tax(language)`) so it is visible and routable; recorded
  automatically on each run from `UserInput.locale`.
- **RAG-poisoning detection.** `PoisoningDetector` / `PoisonVerdict` /
  `PoisoningReport` flag likely-poisoned retrieved evidence from
  authority/provenance signals (embedded instructions, low-authority/high-
  promotion sources, consensus outliers), with an optional async classifier hook
  and FP/FN telemetry (`PoisoningReport.telemetry`).
- **Config.** A new `governance` section (`GovernanceConfig`): `allowed_regions`,
  `provider_regions`, `deny_on_unknown_region`, `content_marking`, `locales`,
  `card_format`.
- **CLI.** `vincio governance card | report | aibom | lineage | erase`.
- **Errors.** `GovernanceError`, `ResidencyViolationError`, `ErasureError`.
- **Example & docs.** `examples/30_governance_compliance.py`; a new
  [governance guide](docs/guides/governance.md); API/CLI/config reference and
  SECURITY/ROADMAP updates.
- **VincioBench.** A new `governance` family gating card/AI-BOM completeness,
  framework-mapping coverage, erasure correctness, multilingual PII recall, and
  RAG-poisoning telemetry — 13 new `budgets.json` budgets (129 total) and three
  new SLOs.

### Changed

- `vincio.__all__` gains `ModelCard`, `SystemCard`, `ComplianceReport`,
  `ComplianceFramework`, `AIBOM`, `ResidencyPolicy`, `LineageRecord`,
  `ErasureResult`, `ProvenanceManifest`, `FertilityTracker`, and
  `PoisoningDetector` (additive; the frozen surface only grows).
- `PIIMatch.type` is now `str` (was a closed `Literal`) with a new optional
  `locale` field, so locale packs can contribute new category labels. Backward
  compatible — the built-in categories are unchanged and still accepted.

### Quality

- **980 tests passing offline; ruff clean; VincioBench 129/129 budgets**; thirty
  runnable examples.

## [1.5.0] - 2026-06-16

Multimodal, embeddings & retrieval breadth (vs LlamaIndex, Voyage/Cohere).
Keeps retrieval best-in-field as the embedding and ingestion frontier moves —
every new embedder, store, and parser sits behind an interface that already
existed. Additive under the frozen 1.0 API; no public symbol removed or
repurposed.

### Added

- **Matryoshka (MRL) embeddings.** `build_embedder(kind, dimensions=N)` (and the
  experimental `MatryoshkaEmbedder`, the `retrieval.embedding_dimensions` config
  field, and `mrl_truncate`) truncate each output vector to its `N` leading
  dimensions and L2-renormalize. Hosted embedders (Jina/Voyage/Cohere) request
  the shorter vector natively; everything else is wrapped, so the result is
  exactly `N` long. Storage/latency vs. recall is gated per dimension in the
  VincioBench `rag` family.
- **Query-vs-document input-type hints.** All built-in embedders accept an
  optional `input_type` (`"document"` / `"query"`); `VectorIndex` passes the
  right one on add vs. search. The `embed_texts(embedder, texts, input_type=...)`
  helper dispatches the hint only to embedders that support it, so custom
  embedders implementing only `embed(texts)` keep working unchanged.
- **Contextual & multimodal embedders.** `VoyageContextualEmbedder`
  (`voyage-context-3`, chunk vectors carry document context — complements
  `contextualize_chunks`) and unified text+image embedders
  `VoyageMultimodalEmbedder` (`voyage-multimodal-3`) and
  `CohereMultimodalEmbedder` (`embed-v4.0`) via `build_embedder`,
  `MultimodalInput`, and `embed_multimodal`. All ride core `httpx` — no SDK.
- **Five new vector stores.** Weaviate, Milvus, Elasticsearch/OpenSearch, and
  Vespa behind the one `build_vector_index` factory and the `Index` protocol,
  joining Qdrant, pgvector, Chroma, Pinecone, and LanceDB. Each lazy-imports its
  SDK with a helpful `StorageError` and accepts an injected client for offline
  round-trip tests. New extras: `vincio[weaviate|milvus|elasticsearch|opensearch|vespa]`.
- **Layout-aware PDF extraction.** `load_document(path, layout=True)` /
  `load_pdf(path, layout=True)` / `extract_pdf_layout` recover column-aware
  reading order, tables with bounding boxes, and figure regions for complex PDFs
  via `vincio[pdf-layout]` (pdfplumber); the dependency-free pypdf text path
  stays the default. Pure, offline-tested helpers `group_words_into_lines` /
  `order_blocks` / `assemble_layout`.
- **Voice / realtime (optional module).** `vincio.realtime`: a provider-neutral
  `RealtimeSession` over OpenAI Realtime / Gemini Live (WebSocket) or a
  deterministic in-process backend, with VAD, interruption (barge-in), and
  **in-session tool calls routed through the permissioned, sandboxed, audited
  tool runtime** (`app.realtime_session(...)`). A separate `vincio[realtime]`
  extra, `@experimental`, explicitly scoped as a stateful bidirectional module —
  not core context engineering.
- **New top-level symbols:** `MatryoshkaEmbedder`, `RealtimeSession`
  (both `@experimental`, since 1.5). Example `29_multimodal_retrieval.py`.

### Notes

- 919 tests passing offline; ruff clean; VincioBench 116/116 budgets;
  twenty-nine examples. The `rag` family gained MRL recall-vs-dimension and
  unified multimodal recall/MRR (four new budgets, three new SLOs).

See the [roadmap](ROADMAP.md) (1.5 milestone).

## [1.4.1] - 2026-06-16

Completes the 1.4 distillation-flywheel capture so faithful, grounded training
data needs no opt-in and covers every run path. Additive under the frozen 1.0
API; no public symbol removed or repurposed.

### Added

- **Flag-free faithful export from `RunResult`s.** A `RunResult` already carries
  the full untruncated output (`raw_text`) and the full cited evidence
  (`evidence` / `citations`), and the runtime now stamps the original input on
  `result.metadata["input"]` — so `app.export_training_set(runs=[...])` /
  `export_training_set_from_runs(...)` build grounding-checked, deduped,
  provenance-stamped fine-tuning JSONL **without `enable_training_capture()`**.
  The trace-based path stays for the "I only have traces" case.

### Fixed

- **Training capture now covers streaming runs.** `app.astream` records the full
  output and cited evidence on its trace when `training_capture` is on (and a
  truncated `output` span attribute for parity with non-streaming), so
  streaming-sourced traces curate into faithful training data too — previously
  only the `run` / `arun` / `batch` / eval path was instrumented.

### Notes

- 866 tests passing offline; ruff clean; VincioBench 112/112 budgets;
  twenty-eight examples. The two follow-ups documented in the 1.4.0 release are
  now closed: faithful capture no longer requires an opt-in flag (via the
  `RunResult` path), and streaming runs are covered.

See the [roadmap](ROADMAP.md) (1.4 milestone).

## [1.4.0] - 2026-06-15

Reflective optimization & the data flywheel (vs DSPy 3). 0.8 shipped the closed
loop; 1.4 sharpens the optimizer to the 2025–26 state of the art and adds the
lever the field is missing — turning production traces into cheaper inference —
while keeping every promotion gated, grounded, and audited. Like the rest of the
1.x line, the milestone is **additive under the frozen 1.0 API**: new surfaces sit
behind `@experimental` entry points, no public symbol is removed or repurposed,
and it uses only the core `httpx` dependency — no SDKs.

### Added

- **Reflective optimizer (GEPA-style)** (`vincio.optimize.ReflectiveOptimizer`,
  `ReflectiveResult`, `Reflector`, `HeuristicReflector`, `LLMReflector`,
  `MIPROProposer`, `ProposedEdit`, `Reflection`, `apply_edits`). Instead of blind
  mutation, the optimizer reads the eval report's failures, reflects on why a
  prompt lost, and proposes targeted edits, evolving a `ParetoFrontier`. A child
  is screened on a minibatch and earns a full rollout only when it beats its
  parent, so the GEPA sample-efficiency win holds under a **hard evaluation
  budget**, deterministic under seed. `strategy="mipro"` switches to MIPROv2-style
  joint instruction+example proposal. The result is a drop-in `OptimizationResult`:
  `ImprovementLoop(optimizer="reflective")`, `app.reflective_optimize(...)`, and
  `vincio optimize reflective` (and `vincio loop run --reflective`) promote through
  the identical gated path (registry push, eval-link, audit, event).
- **Distillation / fine-tune flywheel** (`vincio.optimize.export_training_set`,
  `TrainingSet`, `TrainingExample`, `BootstrapFinetune`, `DistillationResult`).
  `app.export_training_set(...)` / `vincio distill` curate production traces
  (feedback-filtered, grounding-checked against cited evidence, deduped, with full
  provenance) into provider-ready fine-tuning **JSONL** (OpenAI and Anthropic
  shapes); a teacher→student loop measures whether a cheaper student holds quality
  on the eval suite before promoting it into a runtime `ModelCascade`. Every
  exported example is grounded and gated. Opt-in `app.enable_training_capture()`
  (config `observability.training_capture`) records the full output and cited
  evidence on each trace so the export is faithful, not truncated to the span.
- **Learned prompt compression** (`vincio.context.LLMLinguaCompressor`,
  `TokenImportanceScorer`, `compression_faithfulness`, `faithfulness_preserved`,
  `salient_units`). A token-importance compressor that drops low-information tokens
  while protecting numbers, entities, citations, and query terms — a drop-in
  `ContextCompiler.compressor` alongside extractive compression.
  `vincio.optimize.CompressionTuner` / `app.gate_compression(...)` adopt it only
  when it preserves the cited-fact set and holds quality under eval;
  `app.use_learned_compression()` installs it directly.
- **Optimizer-judge calibration** (`vincio.optimize.JudgeCalibrator`,
  `JudgeStepReflector`, `JudgeStepProposal`, `JudgeCalibrationResult`).
  `app.calibrate_judge(...)` reflectively tunes a `GEvalJudge`'s evaluation steps
  against κ-validated human labels, adopting a new procedure only when its Cohen's
  κ strictly beats the incumbent, and leaving the judge's gating weight reflecting
  the higher agreement.
- New top-level exports: `ReflectiveOptimizer`, `TrainingSet`, `BootstrapFinetune`,
  `LLMLinguaCompressor`, `JudgeCalibrator`. New example
  `28_reflective_optimization.py`. The VincioBench `loop` family gains
  reflective-search-vs-baseline lift, distillation grounded-only export +
  quality-hold, and compression fidelity + faithfulness-gating gates (nine new
  budgets, three new SLOs).

### Notes

- **854 tests passing offline; ruff clean; VincioBench 112/112 budgets**;
  twenty-eight runnable examples. All 1.4 surfaces are `@experimental(since="1.4")`
  on the frozen 1.0 API — no existing behaviour changes, and the default compressor
  remains extractive until a learned one is installed.

See the [roadmap](ROADMAP.md) (1.4 milestone).

## [1.3.1] - 2026-06-15

Completes the 1.3 cost-and-reliability layer so it has no attribution or
behavioral gaps. All additive/fixes under the frozen 1.0 API.

### Added

- **Cost attribution now spans agents and crews.** `app.agent(...).run(...)` and
  `Crew.run` / `arun` accept `tenant_id` / `user_id` / `feature`, and every agent
  step and crew (manager + member) model call is recorded on the app's
  `CostLedger` — `app.cost_report` and budgets now cover agentic workloads, not
  just the `run` / `arun` / `astream` / `batch` pipeline.

### Fixed

- **Runtime cascades now escalate on streaming runs.** `app.astream` with a
  cascade buffers each rung and streams the accepted (escalated) answer, instead
  of silently using only the first rung — streaming and non-streaming runs now
  behave identically.
- **Response-cache hits are free.** A `response_cache` hit served the answer
  without an API call, so it is billed `$0` (and recorded as a `$0` cost event)
  rather than at the full uncached price; `cost_report` reflects real spend.

### Internal

- Hardened from an adversarial review: `RateLimiter.acquire` is lock-guarded;
  `KeyPool.stream` no longer falls back to a known-open breaker; the circuit
  breaker releases a half-open probe slot on a cancelled probe; self-correction
  cost is recorded on the ledger; `LiveIndex` keeps unchanged chunks' freshness
  consistent; Anthropic multi-part messages honor `cache_hint`. Strengthened
  offline batch-wire tests (OpenAI error files / failed status; Anthropic errored
  results / cancel). 797 tests; ruff clean; VincioBench 103/103.

See the [roadmap](ROADMAP.md) (1.3 milestone).

## [1.3.0] - 2026-06-15

Cost, reliability & scale (FinOps + resilience). What real teams hit when an LLM
app meets production traffic — provider outages, rate limits, runaway spend, and
the need to attribute every dollar — handled **in your application, not a proxy
hop**. Like 1.1/1.2, the milestone is **additive under the frozen 1.0 API**: new
surfaces sit behind `@experimental` entry points, no public symbol is removed or
repurposed, and it uses only the core `httpx` dependency — no SDKs.

### Added

- **Batch execution** (`vincio.providers.BatchRunner`, `BatchRequest`,
  `BatchResult`, `BatchJob`, `BatchRunResult`, `BatchBackend`,
  `InProcessBatchBackend`, `OpenAIBatchBackend`, `AnthropicBatchBackend`) —
  `app.batch([...])` / `app.abatch` / `vincio batch` submit a request set to the
  OpenAI **Batch API** or Anthropic **Message Batches API** (flat ~50% cost), poll
  to completion, and reconcile responses **by custom id** with partial-failure
  surfacing (missing ids become failed results, never dropped). The in-process
  backend is the offline/default path; the wire backends drive the real endpoints
  over the provider's own `httpx` client, reusing its payload-building and parsing.
  Same `RunResult` contract, cost-tracked at the discounted rate and traced.
- **Circuit breaking & health-aware failover** (`vincio.providers.CircuitBreaker`,
  `CircuitState`, `HealthAwareFailover`, `CircuitOpenError`) — a breaker tracks
  per-provider failure rate **and** latency over a rolling window, opens on
  threshold with half-open probing, and fast-fails (non-retryable) so the failover
  chain steers to healthy entries in microseconds. The documented pattern, made
  explicit: retries for transient (`RetryingProvider`), fallback for persistent
  (`HealthAwareFailover`), circuit-break for systemic (`CircuitBreaker`).
- **Key pooling & rate limiting** (`vincio.providers.KeyPool`, `RateLimiter`) —
  round-robins health-aware across multiple API keys/regions, enforces per-key
  dual **RPM + TPM** token buckets so a limit self-heals instead of erroring, and
  applies full-jitter backoff that honors `retry_after` on 429.
- **Runtime model cascades** (`vincio.optimize.ModelCascade`, `CascadeRung`,
  `response_confidence`) — `app.use_cascade([...])` starts on the cheapest rung and
  escalates only when a response's confidence is below the rung threshold (default:
  a clean, schema-valid stop is confident); a custom confidence callable drives it
  from your own metric. The offline `RoutingOptimizer` keeps tuning thresholds.
- **Cost attribution & budget SLOs** (`vincio.observability.finops`: `CostLedger`,
  `CostEvent`, `CostReport`, `CostBudget`, `BudgetManager`, `BudgetDecision`) —
  every model call in a `ContextApp` run (including its tool loop, self-correction,
  and batch) records an attributed `CostEvent` (`tenant` / `user` / `feature` /
  `run`), rolled up by any dimension (`app.cost_report(by=...)` /
  `vincio cost report --by tenant|feature`). `app.set_cost_budget(...)` enforces a
  per-scope budget on breach — **hard cap** (deny), **degrade-to-cheaper-model**,
  or **queue-to-batch** — as a `PolicyViolation` on the hash-chained audit path; an
  `anomaly_factor` raises a `cost.anomaly` event on a spend spike. Attribution is
  captured at request creation, so long agentic traces are counted honestly.
- **Provider-aware prompt caching** (`vincio.providers.PromptCacheStrategy`,
  `cache_hit_rate`) — `app.enable_prompt_caching(ttl="5m"|"1h")` attaches an
  Anthropic `cache_control` breakpoint with the chosen TTL to the compiler's stable
  prefix (when long enough to be worth caching); auto-cache providers (OpenAI/Gemini)
  rely on the stable→volatile ordering the compiler already produces. **Cache-hit
  rate** is recorded on every model span. On by default (`cache.provider_cache`).
- **Incremental & sharded indexing** (`vincio.retrieval.ShardedIndex`,
  `UpsertStats`) — `LiveIndex.upsert` gained **content-hash change detection** so
  only changed chunks re-embed, `LiveIndex.upsert_stream` for streaming ingestion,
  and `ShardedIndex` splits a corpus across N backends queried in parallel and
  merged, behind the existing `Index` protocol (a document's chunks co-locate).
- **VincioBench `scale` family** — gates batch-result correctness, circuit/failover
  recovery, prompt-cache hit rate, cost-attribution accuracy, and cascade savings;
  four new SLOs hold them (**103 budgets total, all green**).
- Example `27_cost_and_reliability.py` (27 examples, all run offline). New guide:
  [Cost, reliability & scale](docs/guides/cost-and-reliability.md); new comparison:
  [vs LiteLLM / gateways](docs/comparisons/litellm.md).

### Changed

- `__version__` is now `1.3.0`. `UserInput` gains an optional `feature` field and
  `Message` gains an optional `cache_ttl` field; `arun` / `astream` accept a
  `feature=` attribution argument; `CacheConfig` gains `provider_cache` /
  `provider_cache_ttl` / `provider_cache_min_prefix_tokens`. `HTTPProvider` gains
  `_get_json` / `_get_text` helpers and the Anthropic adapter sends the
  extended-cache-ttl beta header. All additive and backward-compatible.

See the [roadmap](ROADMAP.md) (1.3 milestone) for the full picture.

## [1.2.0] - 2026-06-14

Agentic evaluation & continuous quality. Vincio could run and trace a crew, a
graph, and a tool loop — 1.2 makes it **score** them: over the trajectory, over a
multi-turn conversation, and over live traffic. Every new metric is the same
object reused as an offline gate, a runtime guardrail, and an optimizer fitness
term. Like 1.1, the milestone is **additive under the frozen 1.0 API** — new
surfaces sit behind `@experimental` entry points, no public symbol is removed or
repurposed — and runs in your process with no hosted dependency.

### Added

- **Trajectory & tool-use metrics** (`vincio.evals.metrics`) — `tool_call_accuracy`
  / `tool_call_f1` (right tool, right args, in the right order),  `goal_accuracy`
  (successful termination + answer match), `plan_adherence` (LCS vs the expected
  plan), `plan_quality` (failed/redundant steps, reference-free),
  `step_efficiency` (steps vs an optimal path), and `topic_adherence`. They read a
  provider-neutral `Trajectory` (`vincio.evals.trajectory`) carried on the
  `RunOutput`, built with `RunOutput.from_agent_state(state)` /
  `from_crew_result(result)` / `from_trace(trace)` — a crew, a `StateGraph` run,
  or a captured trace is scored without re-instrumentation. Expected/optimal
  references live in `rubric['expected_tools' | 'plan' | 'optimal_steps' |
  'topic']`. `EvalReport.metric_families()` splits the report into final-output-only
  vs trajectory evaluation.
- **Conversational metrics** — `conversation_outcome` (did the thread achieve the
  user's goal) and `intent_resolution` (fraction of user turns addressed), joining
  `knowledge_retention` / `conversation_relevance`.
- **Multi-turn simulator** (`vincio.evals.Simulator`, `Persona`,
  `SimulatedConversation`, experimental) — drives multi-turn sessions from a
  persona + goal; LLM-backed with a seeded template fallback, so it is
  deterministic offline (same seed → identical conversation).
  `SimulatedConversation.to_eval_case()` feeds the conversational metrics;
  `dataset_from_traces(..., group_by_session=True)` stitches a session's traces
  into a multi-turn golden case.
- **Online / continuous eval** — `app.add_online_evaluator(metric,
  sample_rate=...)` (experimental) scores a sampled fraction of live runs after
  the response is finalized (scheduled off the hot path; `app.aflush_online()`
  drains in tests), writing each score as a time series on the metadata store
  (`OnlineEvaluator.series()`). No traffic mirrored to any external service.
- **Drift detection** (`vincio.evals.DriftMonitor`, `DriftReport`) — rolling
  score drift and embedding-distribution drift of inputs against the golden-set
  distribution; raises a `drift.detected` event on the bus and persists baselines
  (`drift_baselines`). `vincio eval drift baseline.json current.json` reports it.
- **Human-in-the-loop annotation** (`vincio.evals.AnnotationQueue`,
  `cohens_kappa`) — records human labels next to LLM-judge scores and tracks
  **Cohen's κ**; `GEvalJudge.calibrate()` now also returns `cohens_kappa`, and
  `judge.gating_weight(threshold)` / `queue.judge_trusted()` gate a judge on
  agreement. `vincio eval annotate labels.jsonl` reports it.
- **Production A/B** — `app.experiment(name, variants=..., dataset=...,
  metrics=...)` (experimental) returns an `Experiment` comparing variants on eval
  metrics **and** cost (`.compare()` / `.cost()` / `.significance(metric)`) with
  the paired/Welch tests `ExperimentTracker` already ships.
- **Metric-as-guardrail** — `app.add_metric_rail(metric, threshold=...)` /
  `vincio.evals.metric_guardrail(metric, threshold=...)` wrap any metric as a
  deterministic runtime rail predicate (direction from `LOWER_IS_BETTER`).
- **Optimizer interconnection** — `vincio.optimize.AGENTIC_OBJECTIVES`, a Pareto
  objective preset over `goal_accuracy` / `tool_call_accuracy` / `step_efficiency`
  / `cost`; trajectory metrics are ordinary metrics, so they flow into
  `report.metric_values` and the frontier unchanged.
- **VincioBench `agentic_evals` family** — gates trajectory-metric agreement
  against labeled traces, the output-only/trajectory gap, simulator determinism,
  drift sensitivity/specificity, and κ tracking; six new SLOs hold them (94
  budgets total, all green).
- Example `26_agentic_eval.py` and a labeled golden set
  `tests/golden/agentic_eval.jsonl` (26 examples, all run offline). New guide:
  [Agentic evaluation & continuous quality](docs/guides/agentic-eval.md).

### Fixed

- **Gemini embedding cost tracked as $0** — the cost table referenced the dead
  `text-embedding-004` while the Google provider defaults to `gemini-embedding-001`,
  which was absent from the table, so a price lookup fell through to the zero
  default and embedding cost was billed at $0. `gemini-embedding-001` is now priced
  ($0.15 / 1M input tokens), with a regression test.

### Changed

- `__version__` is now `1.2.0`. `RunOutput` gains an optional `trajectory` field
  and `from_agent_state` / `from_crew_result` / `from_trace` constructors;
  `dataset_from_traces` gains `group_by_session`; `GEvalJudge.calibrate` also
  returns `cohens_kappa`. All additive and backward-compatible.

See the [roadmap](ROADMAP.md) (1.2 milestone) for the full picture.

## [1.1.0] - 2026-06-13

Protocols & interoperability — the first post-1.0 milestone. Vincio now speaks
the interoperability protocols the ecosystem standardized on in 2025–26 —
**MCP** (client *and* server), **A2A** agent-to-agent, and Anthropic **Agent
Skills** — plus a unified reasoning control across providers. Everything is
**additive under the frozen 1.0 API**: every new surface sits behind a new
entry point and is marked `@experimental`; no public symbol is removed or
repurposed, so upgrading across the 1.x line never breaks working code. The new
protocols use only the core `httpx` dependency — no SDKs — and run in your
process; Vincio adopts the standards, it does not become a service.

### Added

- **MCP client + server** (`vincio.mcp`, experimental) — `MCPClient` /
  `app.add_mcp_server(name, command=/url=/server=)` connect to MCP servers over
  **stdio**, **Streamable HTTP**, and an **in-process** transport (offline
  tests), negotiate capabilities, and surface `tools` / `resources` / `prompts`.
  MCP tools register through the *existing* permissioned, sandboxed, audited,
  budgeted tool runtime (namespaced `<server>.<tool>`); MCP resources become
  evidence with `origin: mcp:<server>` provenance; MCP prompts import as
  `PromptSpec`. Server-initiated **sampling** routes to the app's provider,
  **elicitation** to a human-gate callback; OAuth 2.1 seams (`pkce_pair`,
  `static_token_validator`) and a long-running **Tasks** poll path are included.
  `app.serve_mcp()` / `vincio mcp serve` expose a `ContextApp` as an MCP server
  (tools/resources/prompts), with the policy engine and audit log enforced on
  every inbound call and OAuth 2.1 resource-server token validation.
  `vincio mcp tools` / `mcp add` inspect and wire servers from the CLI.
- **A2A (agent-to-agent)** (`vincio.a2a`, experimental) — `app.serve_a2a(crew |
  graph | None)` serves an **Agent Card** (`/.well-known/agent.json`) and a
  JSON-RPC **task lifecycle** (`submitted → working → input-required →
  completed/failed`); graph human-in-the-loop interrupts surface as
  `input-required` and resume by `taskId`. `A2AClient` / `connect_a2a` reach
  remote agents, and `RemoteA2AAgent` plugs a remote agent into a local crew as
  a **bounded, traced** delegate. Token validation + per-task audit (`a2a_serve`).
- **Agent Skills** (`vincio.skills`, experimental) — `app.add_skill(path)` loads
  Anthropic-style `SKILL.md` (YAML frontmatter + Markdown + optional bundled
  scripts) and injects it through the compiler with **progressive disclosure**:
  a one-line index is always available; a skill's full body enters the budget
  only when the task is relevant (scored and cited like any evidence). Bundled
  scripts run as sandboxed, permissioned tools (`register_scripts=True`).
- **Unified reasoning control** — `RunConfig(reasoning_effort="minimal"|"low"|
  "medium"|"high")` / `thinking_budget_tokens` map to OpenAI reasoning effort,
  Anthropic extended thinking (sampling left at default), and Gemini thinking
  budgets; providers without reasoning ignore them. The negotiated reasoning
  mode is recorded on the `prompt_render` span and `reasoning_tokens` on the
  `model_call` span. `ModelCapabilities.reasoning` declares support.
- **OpenAI Responses API adapter** (`OpenAIResponsesProvider`,
  `build_provider("openai_responses")`) — stateful `previous_response_id`,
  built-in tools, reasoning preserved across tool calls, behind the same
  `ModelProvider` interface; Chat Completions stays the portable default.
- **VincioBench `protocols` family** — gates MCP tool schema-fidelity + resource
  provenance, A2A delegation termination, and Agent-Skill progressive-disclosure
  budget savings; three new SLOs hold them (88 budgets total, all green).
- Four new examples: `22_mcp_tools_and_resources.py`, `23_a2a_delegation.py`,
  `24_agent_skills.py`, `25_reasoning_control.py` (25 examples, all run offline).
- New guides: [MCP](docs/guides/mcp.md), [A2A](docs/guides/a2a.md),
  [Agent Skills](docs/guides/agent-skills.md), and
  [reasoning control](docs/guides/reasoning.md).

### Fixed

- **Reasoning-token cost accounting** — Gemini reported thinking tokens
  (`thoughtsTokenCount`) as `reasoning_tokens` but excluded them from the
  billable output (`candidatesTokenCount`), so thinking was costed at $0. The
  Google adapter now folds thinking tokens into the billable output (they are
  billed at the output rate, matching `totalTokenCount`), while
  `reasoning_tokens` keeps the thinking subset for telemetry. OpenAI/Anthropic
  were already correct (reasoning is part of completion/output tokens).

### Changed

- `__version__` is now `1.1.0`. `ModelRequest` gains `reasoning_effort`,
  `thinking_budget_tokens`, and `previous_response_id`; `RunConfig` gains
  `reasoning_effort` / `thinking_budget_tokens`; `ModelCapabilities` gains
  `reasoning`; `MockProvider(reasoning=True)` emulates thinking tokens offline.
  All additive and backward-compatible.

See the [roadmap](ROADMAP.md) (1.1 milestone) for the full picture.

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
