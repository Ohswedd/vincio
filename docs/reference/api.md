# Reference: API

## `vincio.ContextApp`

```python
ContextApp(name, *, objective=None, output_schema=None, config=None,
           provider=None, model=None, budget=None, policies=None, prompt_spec=None)
```

| Method | Purpose |
|---|---|
| `configure(objective=, role=, rules=, examples=, citation_policy=, ...)` | declarative prompt setup |
| `add_source(name, path=, documents=, connector=, chunking=, retrieval=)` | load + chunk + index a knowledge source (local files, in-memory documents, or any connector) |
| `add_memory(scope=, strategy=, store=, embedder=)` | enable the memory engine (hybrid recall by default) |
| `remember(content, user_id=, agent_id=, session_id=, ...)` | ergonomic policy-checked memory write |
| `recall(query, user_id=, agent_id=, session_id=, top_k=)` | scored hybrid memory recall |
| `add_tool(fn_or_name, permissions=, approval_required=, side_effects=)` | enable a tool |
| `add_evaluator(name_or_fn)` | score every run with a metric |
| `add_validator(name, fn, blocking=)` | semantic output validator |
| `add_optimizer(name)` | register an optimization dimension |
| `set_policy(name, value)` | set a run policy (e.g. `answer_only_from_sources`) |
| `add_rail(name=, kind=, direction=, action=, ...)` | programmable input/output rail (topic / format / safety / custom) |
| `register_rail_predicate(name, fn)` | register a custom rail predicate `(text, params) -> falsy \| message` |
| `add_output_schema(schema, keywords=, task_types=, when=, priority=)` | register an alternative output schema, routed per run |
| `enable_self_correction(max_cycles=, max_cost_usd=)` | bounded validate→critique→repair on failed outputs |
| `predictor(signature, model=, temperature=)` | `Predict` bound to the app's provider for a typed `Signature` |
| `run(input, files=, tenant_id=, user_id=, session_id=, config=)` / `arun` | execute the 17-step pipeline → `RunResult`; `arun` / `astream` also accept `feature=` *(1.3)* as a cost-attribution dimension |
| `astream(input, ...)` / `stream` | the same pipeline with end-to-end streaming → `RunStreamEvent` iterator |
| `submit(input, ...)` *(experimental, 1.7)* | start a run in the background → `RunHandle` (`cancel()` propagates cooperative cancellation; `await handle` / `result()` → `RunResult`; the cancelled run is still fully recorded) |
| `use_semantic_context_scoring(enabled=True, *, mmr_lambda=None)` *(experimental, 1.7)* | score/select context by embedding cosine + MMR with reranker-blended relevance (needs a real `retrieval.embedder`) → `ContextApp` |
| `agent(tools=, planner=, max_steps=, evaluator=)` | bounded agent handle |
| `crew(name, members=, process=, tools=, max_rounds=)` | multi-agent `Crew` over a shared blackboard (sequential / parallel / hierarchical) |
| `graph(name, state_schema=, reducers=)` | durable `StateGraph` checkpointed in the app's metadata store |
| `workflow(name)` | deterministic `Workflow` builder (approval gates pause; `resume(result, approvals=)` continues) |
| `evaluate(dataset, metrics=, concurrency=, gates=, judges=)` | `EvalReport` |
| `add_online_evaluator(metric, sample_rate=, name=)` *(experimental, 1.2)* | score a sampled fraction of live runs off the hot path; writes a score time series to the store |
| `add_metric_rail(metric, threshold=, direction=, action=, name=, **params)` *(experimental, 1.2)* | use an eval metric as a runtime guardrail |
| `experiment(name, variants=, dataset=, metrics=)` *(experimental, 1.2)* | production A/B over prompt/model/config variants → `Experiment` (`compare`/`cost`/`significance`) |
| `aflush_online()` *(1.2)* | await in-flight online evaluations (tests/shutdown) |
| `improvement_loop(metrics=, weights=, gates=, experiment=, optimizer="evolution"\|"reflective", strategy=, ...)` | `ImprovementLoop`: trace → dataset → eval → optimize → promote on this app |
| `reflective_optimize(dataset, *, strategy="reflective"\|"mipro", metrics=, budget=12, minibatch_size=8, seed=7, weights=, gates=, objectives=, apply=False)` *(experimental, 1.4)* | GEPA-style reflective optimization → `ReflectiveResult` (drop-in `OptimizationResult` with `.frontier` / `.reflections`) |
| `enable_training_capture(enabled=True)` *(experimental, 1.4)* | record full output + cited evidence on each trace (incl. streaming) for faithful trace-based distillation → `ContextApp` |
| `export_training_set(*, name=, runs=, traces=, limit=500, min_feedback_score=, require_grounding=True, min_support=0.5, max_examples=, path=, format="openai"\|"anthropic")` *(experimental, 1.4)* | curate grounded fine-tuning JSONL from `RunResult`s (`runs=`, faithful, flag-free) or captured traces → `TrainingSet` |
| `distill(training_set, dataset, *, teacher, student, trainer=, quality_metric=, min_quality_ratio=0.97, gates=, apply=True)` *(experimental, 1.4)* | gated teacher→student distillation; promotes a `ModelCascade` on quality-hold → `DistillationResult` |
| `use_learned_compression(compressor=None)` *(experimental, 1.4)* | install a learned compressor (`LLMLinguaCompressor`) on the compiler, ungated → `ContextApp` |
| `gate_compression(dataset, *, compressor=, metrics=, min_faithfulness=0.9, min_quality_ratio=0.98)` *(experimental, 1.4)* | adopt a learned compressor only if it preserves cited facts + quality under eval → `CompressionTuningResult` |
| `calibrate_judge(judge, samples, *, budget=4)` *(experimental, 1.4)* | reflectively tune a `GEvalJudge`'s steps for Cohen's-κ agreement → `JudgeCalibrationResult` |
| `use_learned_budgets(source)` | install eval-tuned per-task budget allocations (`LearnedAllocations`, path, or mapping) |
| `use_pack(pack, set_schema=, merge_rules=)` | apply a domain pack (`"support"`/`"engineering"`/`"finance"`/`"legal"` or a `Pack`): prompt config + schema + policies + evaluators + rails |
| `add_skill(path_or_skill, register_scripts=)` *(experimental, 1.1)* | load an Agent Skill (`SKILL.md`) with progressive disclosure; optionally expose bundled scripts as sandboxed tools |
| `add_mcp_server(name, command=/url=/server=/transport=, tools=, resources=, prompts=, permissions=, sampling=, elicitation=, auth=, headers=, http_client=)` *(experimental, 1.1)* | connect an MCP server; its tools register through the permissioned runtime, resources become evidence |
| `serve_mcp(name=, expose_resources=, expose_prompts=, token_validator=)` *(experimental, 1.1)* | expose this app as an MCP server (`MCPServer`) |
| `serve_a2a(target=, name=, url=, description=, token_validator=)` *(experimental, 1.1)* | expose a crew / compiled graph / the app over A2A (`A2AServer`, Agent Card + task lifecycle) |
| `batch(inputs, *, backend=, config=, discount=0.5, timeout_s=)` / `abatch` *(experimental, 1.3)* | run many inputs through a batch backend (~50% cost) → `list[RunResult]` |
| `use_cascade(models=, *, rungs=, min_confidence=0.5, max_escalations=, confidence=)` *(experimental, 1.3)* | route runs up a cheap→capable model cascade, escalating on low confidence → `ContextApp` |
| `set_cost_budget(*, limit_usd, scope="tenant"\|"feature"\|"user"\|"global", id=, period="run"\|"hour"\|"day"\|"month"\|"total", on_breach="cap"\|"degrade"\|"queue_to_batch", degrade_model=, anomaly_factor=)` *(experimental, 1.3)* | enforce a spend budget with on-breach action → `ContextApp` |
| `cost_report(*, by="tenant"\|"feature"\|"user"\|"model"\|"provider"\|"run", since=)` *(experimental, 1.3)* | aggregate ledger spend along a dimension → `CostReport` |
| `enable_prompt_caching(*, ttl="5m"\|"1h", min_prefix_tokens=1024)` *(1.3, experimental)* | mark long stable prefixes for provider prompt caching → `ContextApp` |
| `task` (decorator) | configure from a task class |
| `stats()` | sources, tools, memory, cost, run counts |

## `RunResult`

`output` (typed when a schema is set), `raw_text`, `status`, `error`,
`trace_id`, `context_packet_id`, `evidence`, `citations`, `tool_results`,
`usage`, `cost_usd`, `latency_ms`, `validation`, `eval_scores`,
`excluded_context`.

## `RunStreamEvent`

Yielded by `astream` / `stream`. `type` is one of `stage`, `text_delta`,
`partial_output` (incremental partial-JSON parse for structured output,
with `output_complete`, plus streaming validation: `valid_prefix` and
`validation_errors`), `tool_call`, `tool_result`, `usage`, `error`, or
the terminal `done` (carrying `result: RunResult`).

## Key subsystem entry points

| Module | Entry points |
|---|---|
| `vincio.prompts` | `PromptSpec`, `PromptCompiler`, `lint_spec`, `generate_variants`, `diff_specs`, `PromptRegistry`, `Signature`, `InputField`, `OutputField`, `signature`, `Predict` |
| `vincio.input` | `normalize_text`, `classify_task`, `classify_file`, `detect_language`, `detect_ambiguity`, `InputRouter` — task/language classification and normalization feeding the runtime router |
| `vincio.documents` | `load_document`, `load_directory`, `load_pdf`, `load_docx`, `load_xlsx`, `extract_markdown_tables`, `extract_code_symbols`, `TesseractOCR` — multimodal loaders (md/html/csv/pdf/docx/xlsx/eml/code), OCR, table/code-aware parsing; *(1.5)* layout-aware PDF extraction `load_document(path, layout=True)` / `extract_pdf_layout` (reading order, tables, figures via `vincio[pdf-layout]`) with pure helpers `group_words_into_lines` / `order_blocks` / `assemble_layout` |
| `vincio.context` | `ContextCompiler`, `ContextPacket`, `ContextIR`, `ContextScorer`, `BudgetAllocator`, `extractive_compress`, and *(experimental, 1.4)* `LLMLinguaCompressor` / `TokenImportanceScorer` / `compression_faithfulness` / `faithfulness_preserved` / `salient_units`. *(experimental, 1.7)* opt-in semantic scoring via `ContextCompilerOptions(semantic_scoring=True)` / `app.use_semantic_context_scoring(mmr_lambda=...)` — embedding-cosine relevance + MMR `_select`, reranker `upstream_relevance` blended into relevance, and salient-unit value-level contradiction (structured packet conflicts); `BudgetAllocator.allocate(..., reserve_tokens=)` reserves response/tool-loop headroom |
| `vincio.retrieval` | `RetrievalEngine`, `BM25Index`, `VectorIndex`, `SparseIndex`, `LateInteractionIndex`, `AutoMergingIndex`, `LiveIndex`, `QueryUnderstanding`, `EntityGraph`, `GraphRAG`, `ReasoningRetriever`, `chunk_document`, `contextualize_chunks`, `build_embedder`, `JinaEmbedder`, `VoyageEmbedder`, `CohereEmbedder`, `build_reranker`, `CohereReranker`, `JinaReranker`, `VoyageReranker`, *(experimental, 1.3)* `ShardedIndex` (parallel fan-out over the `Index` protocol, document chunks co-located) / `UpsertStats` (`LiveIndex.upsert(...) -> UpsertStats` is content-hash incremental, with `LiveIndex.upsert_stream(...)`), and *(1.5)* Matryoshka `build_embedder(kind, dimensions=N)` / `MatryoshkaEmbedder` (experimental) / `mrl_truncate`, query-vs-document `input_type` hints via `embed_texts`, contextual `VoyageContextualEmbedder` (`voyage-context`), and multimodal `VoyageMultimodalEmbedder` / `CohereMultimodalEmbedder` (`voyage-multimodal` / `cohere-multimodal`) with `MultimodalInput` |
| `vincio.connectors` | `connect`, `register_connector`, `Connector`, `CONNECTORS` — built-in kinds via `connect("web"\|"github"\|"sql"\|"s3"\|"gcs"\|"notion"\|"confluence"\|"slack", ...)` |
| `vincio.memory` | `MemoryEngine`, `ScopedMemory`, `MemoryConsolidator`, `MemoryGraph`, `SessionSummarizer`, `SQLiteMemoryStore`, `evaluate_memory`, `GroundedFact`, `extract_grounded_facts` |
| `vincio.tools` | `ToolRegistry`, `ToolRuntime`, `ToolPermissionChecker`, `SandboxedPython` |
| `vincio.agents` | `AgentExecutor`, `Planner`, `StepDAG`, `HandoffRouter`, `Crew`, `AgentRole`, `Blackboard`, `StateGraph`, `Checkpointer`, `interrupt`, `compose`, `parallel`, `branch`, `LangGraphBackend`, `OpenAIAgentsBackend` |
| `vincio.workflows` | `Workflow` (pause/resume approval gates, edit-and-resume) |
| `vincio.output` | `OutputSchema`, `OutputContract`, `OutputValidator`, `Repairer`, `to_strict_json_schema`, `choice_schema`, `regex_schema`, `StreamingValidator`, `SelfCorrector`, `SchemaRouter` |
| `vincio.evals` | `Dataset`, `dataset_from_traces`, `EvalRunner`, `RunOutput`, `ModelJudge`, `GEvalJudge`, `evaluate_gates`, `METRICS`, `SyntheticGenerator`, `RedTeamSuite`, `ExperimentTracker`, `ab_test`, and *(1.2)* `Trajectory` / `trajectory_from_agent_state` / `Simulator` / `Persona` / `SimulatedConversation` / `OnlineEvaluator` / `DriftMonitor` / `DriftReport` / `AnnotationQueue` / `cohens_kappa` / `Experiment` / `metric_guardrail`, and *(experimental, 1.7)* `ReplayRunner` / `ReplayResult` / `ReplayCase` (re-run captured traces and diff output/trajectory/cost; `vincio trace replay --against <app>`); `ab_test` now also returns a confidence interval (`ci_low`/`ci_high`) and Cohen's-d `effect_size` |
| `vincio.optimize` | `PromptOptimizer`, `ContextOptimizer`, `RoutingPolicy`, `evolution_loop`, `fitness`, `ImprovementLoop`, `pareto_loop`, `ParetoFrontier`, `ObjectiveSpec`, `DEFAULT_OBJECTIVES`, `AGENTIC_OBJECTIVES`, `RetrievalFeedback`, `recommend_chunking`, `BudgetLearner`, `LearnedAllocations`, `guided_search`, *(experimental, 1.3)* `ModelCascade` (`from_models`, `next_rung`, `first`, `escalation_cap`) / `CascadeRung` / `response_confidence`, and *(experimental, 1.4)* `ReflectiveOptimizer` / `ReflectiveResult` / `HeuristicReflector` / `LLMReflector` / `MIPROProposer` / `ProposedEdit` / `Reflection` / `apply_edits` / `export_training_set` / `export_training_set_from_runs` / `TrainingSet` / `TrainingExample` / `BootstrapFinetune` / `DistillationResult` / `CompressionTuner` / `CompressionTuningResult` / `JudgeCalibrator` / `JudgeStepReflector` / `JudgeCalibrationResult` |
| `vincio.observability` | `Tracer`, `JSONLExporter`, `OTelExporter`, `CostTracker`, `trace_diff`, `Session`, `sessions_from_traces`, `record_feedback`, `trace_to_html`, `render_trace_text`, and *(experimental, 1.3)* FinOps: `CostLedger`, `CostEvent`, `CostReport`, `CostRow`, `CostBudget`, `BudgetManager`, `BudgetDecision` |
| `vincio.testing` | `assert_eval`, `assert_grounded`, `assert_metric`, `assert_safe`, `Snapshot` (+ pytest plugin: `vincio_snapshot` fixture, `--vincio-update-snapshots`) |
| `vincio.security` | `PIIDetector`, `SecretScanner`, `InjectionDetector`, `AccessController`, `PolicyEngine`, `Rail`, `RailEngine`, `AuditLog`, and *(1.6)* multilingual PII `LocalePack` / `available_locales` / `get_locale_pack` (`PIIDetector(locales=[...])`) and RAG-poisoning `PoisoningDetector` / `PoisonVerdict` / `PoisoningReport` (authority/provenance signals + injection-classifier hook, FP/FN telemetry), and *(1.7)* a pluggable `DetectorBackend` / `DetectorSpan` Protocol on the PII/injection/secret detectors, an injection normalization + recursive base64/hex/rot13 decode pre-pass, and `AccessController(require_explicit_tenant=True)` (fail-closed on untagged tenants) |
| `vincio.governance` *(experimental, 1.6)* | `generate_model_card` / `ModelCard`, `generate_system_card` / `SystemCard`, `CardFormat`; `ComplianceMapper` / `map_compliance` / `ComplianceReport` / `ComplianceFramework` / `Control` / `CONTROL_CATALOG` (OWASP LLM 2025 / OWASP Agentic / NIST AI RMF / MITRE ATLAS); `generate_aibom` / `AIBOM` / `AIComponent` / `sha256_file` / `sha256_text`; `mark_synthetic_content` / `verify_manifest` / `ProvenanceManifest` / `ContentSigner` / `HmacSigner` / `ai_disclosure` / `data_summary`; `LineageIndex` / `LineageRecord` / `ErasureResult`; `ResidencyPolicy` / `residency_violation` / `infer_region_from_url`; `FertilityTracker` / `LanguageFertility`. App methods: `app.model_card()`, `app.system_card()`, `app.compliance_report()`, `app.aibom()`, `app.trace_lineage()`, `app.erase_source()`, `app.set_residency()`, `app.mark_output()` |
| `vincio.caching` | `InMemoryCache`, `SQLiteCache`, `ResponseCache`, `SemanticCache`, `PromptCompileCache`, `ContextCompileCache`, `ChunkCache`, `InvalidationManager` |
| `vincio.providers` | `build_provider`, `openai_compatible`, `OpenAICompatibleProvider`, `OpenAIResponsesProvider`, `PRESETS`, `MockProvider`, `OpenAIProvider`, `AnthropicProvider`, `GoogleProvider`, `MistralProvider`, `LocalProvider`, `FailoverChain`, `CoalescingProvider`. Unified reasoning control (1.1): `RunConfig(reasoning_effort="low"\|"medium"\|"high")` / `thinking_budget_tokens` map to OpenAI reasoning effort, Anthropic extended thinking, and Gemini thinking budgets; thinking tokens are recorded on the model span and billed. *(experimental, 1.3)* batch: `BatchRunner`, `BatchRequest`, `BatchResult`, `BatchJob`, `BatchRunResult`, `BatchStatus`, `BatchBackend` (Protocol), `InProcessBatchBackend`, `OpenAIBatchBackend`, `AnthropicBatchBackend` (~50% cost via `discount=0.5`); resilience: `CircuitBreaker` (is a `ModelProvider`), `CircuitState`, `HealthAwareFailover`, `KeyPool`, `RateLimiter`; prompt caching: `PromptCacheStrategy`, `cache_hit_rate`. Compose inner→outer `CircuitBreaker(RetryingProvider(provider))`. *(experimental, 1.7)* `ModelRegistry` / `default_model_registry` / `ModelUnknownWarning` / `discover_entry_points` — a data-driven catalog (capabilities + standard/batch pricing + GA/deprecation/retirement lifecycle, keyed by exact model id) that `capabilities()` and `PriceTable` derive from; unknown models warn + emit `model.unknown`; overlay via `VINCIO_MODEL_REGISTRY=<path.json\|yaml>`; third-party adapters auto-register via the `vincio.providers` / `vincio.embedders` / `vincio.stores` entry-point groups. |
| `vincio.interop` | `add_langchain_tool`, `from_langchain_tool`, `from_langchain_loader`, `from_langchain_retriever`, `from_langchain_embeddings`, `to_langchain_*`; `add_llamaindex_tool`, `from_llamaindex_reader`, `from_llamaindex_retriever`, `from_llamaindex_embedding`, `to_llamaindex_*` |
| `vincio.mcp` *(experimental, 1.1)* | `MCPClient`, `MCPServer`, `build_app_server`, `serve_stdio`, `connect_stdio`, `connect_http`, `connect_in_process`, `InProcessTransport`, `StdioTransport`, `StreamableHTTPTransport`, `static_token_validator`, `pkce_pair` — MCP client *and* server over stdio / Streamable HTTP / in-process |
| `vincio.a2a` *(experimental, 1.1)* | `A2AServer`, `A2AClient`, `RemoteA2AAgent`, `AgentCard`, `AgentSkill`, `A2ATask`, `crew_a2a_server`, `graph_a2a_server`, `app_a2a_server`, `connect_a2a`, `connect_a2a_in_process`, `static_token_validator` — Agent Card + JSON-RPC task lifecycle |
| `vincio.skills` *(experimental, 1.1)* | `Skill`, `SkillScript`, `SkillLibrary`, `load_skill`, `load_skills`, `parse_skill_md`, `register_skill_scripts` — `SKILL.md` loading with progressive disclosure |
| `vincio.packs` | `Pack`, `load_pack`, `available_packs`, `register_pack` (`app.use_pack(...)`) |
| `vincio.notebook` | `enable_rich_reprs`, `disable_rich_reprs`, `display`, `run_result_html`/`_markdown`, `trace_html`, `eval_report_html`, `memory_item_html`, `search_hit_html` |
| `vincio.tui` | `TUI`, `render_home`, `render_trace`, `render_memory` (`vincio tui`) |
| `vincio.core.concurrency` | `gather_bounded`, `map_bounded` (bounded, cancellation-correct fan-out) |
| `vincio.storage` | `create_metadata_store`, `SQLiteMetadataStore`, `build_vector_index` (memory/qdrant/pgvector/chroma/pinecone/lancedb and *(1.5)* weaviate/milvus/elasticsearch/opensearch/vespa), Neo4j/Redis/DuckDB adapters, and *(1.7)* an async store contract `asave` / `aquery` (native or `to_thread`) plus `vincio.stores` entry-point discovery |
| `vincio.realtime` *(experimental, 1.5, optional module)* | `RealtimeSession`, `connect_realtime`, `RealtimeConfig`, `VADConfig`, `RealtimeEvent`, `RealtimeToolCall`, `InProcessRealtimeBackend`, `OpenAIRealtimeBackend`, `GeminiLiveBackend` — bidirectional voice/realtime over OpenAI Realtime / Gemini Live / in-process, with VAD, interruption, and in-session tool calls through the permissioned runtime (`app.realtime_session(...)`; `vincio[realtime]`) |
| `vincio.core.config` | `VincioConfig`, `load_config`, `config_json_schema` |
| `vincio.server` | `create_app` (FastAPI) |
| `vincio.cli` | `main`, `build_parser` — the `vincio` command (see [CLI reference](cli.md)) |
| `vincio.stability` | `deprecated`, `experimental`, `deprecated_alias`, `stability_of`, `public_api`, `StabilityLevel`, `VincioDeprecationWarning`, `VincioExperimentalWarning`, `API_VERSION` |

All public data contracts are Pydantic models; all engines are async-first
with sync wrappers (`run` / `arun`).

## Stability & versioning

From 1.0, Vincio follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
on its public API — every symbol re-exported from the top-level `vincio` package
(`vincio.__all__`, also returned by `vincio.stability.public_api()`) plus the
documented entry points above. See the [stability policy](stability.md) for the
deprecation contract; `vincio.deprecated` / `vincio.experimental` mark symbols
and `vincio.stability_of(obj)` introspects any symbol's guarantee.
