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
| `run(input, files=, tenant_id=, user_id=, session_id=, config=)` / `arun` | execute the 17-step pipeline → `RunResult` |
| `astream(input, ...)` / `stream` | the same pipeline with end-to-end streaming → `RunStreamEvent` iterator |
| `agent(tools=, planner=, max_steps=, evaluator=)` | bounded agent handle |
| `crew(name, members=, process=, tools=, max_rounds=)` | multi-agent `Crew` over a shared blackboard (sequential / parallel / hierarchical) |
| `graph(name, state_schema=, reducers=)` | durable `StateGraph` checkpointed in the app's metadata store |
| `workflow(name)` | deterministic `Workflow` builder (approval gates pause; `resume(result, approvals=)` continues) |
| `evaluate(dataset, metrics=, concurrency=, gates=, judges=)` | `EvalReport` |
| `add_online_evaluator(metric, sample_rate=, name=)` *(experimental, 1.2)* | score a sampled fraction of live runs off the hot path; writes a score time series to the store |
| `add_metric_rail(metric, threshold=, direction=, action=, name=, **params)` *(experimental, 1.2)* | use an eval metric as a runtime guardrail |
| `experiment(name, variants=, dataset=, metrics=)` *(experimental, 1.2)* | production A/B over prompt/model/config variants → `Experiment` (`compare`/`cost`/`significance`) |
| `aflush_online()` *(1.2)* | await in-flight online evaluations (tests/shutdown) |
| `improvement_loop(metrics=, weights=, gates=, experiment=, ...)` | `ImprovementLoop`: trace → dataset → eval → optimize → promote on this app |
| `use_learned_budgets(source)` | install eval-tuned per-task budget allocations (`LearnedAllocations`, path, or mapping) |
| `use_pack(pack, set_schema=, merge_rules=)` | apply a domain pack (`"support"`/`"engineering"`/`"finance"`/`"legal"` or a `Pack`): prompt config + schema + policies + evaluators + rails |
| `add_skill(path_or_skill, register_scripts=)` *(experimental, 1.1)* | load an Agent Skill (`SKILL.md`) with progressive disclosure; optionally expose bundled scripts as sandboxed tools |
| `add_mcp_server(name, command=/url=/server=/transport=, tools=, resources=, prompts=, permissions=, sampling=, elicitation=, auth=, headers=, http_client=)` *(experimental, 1.1)* | connect an MCP server; its tools register through the permissioned runtime, resources become evidence |
| `serve_mcp(name=, expose_resources=, expose_prompts=, token_validator=)` *(experimental, 1.1)* | expose this app as an MCP server (`MCPServer`) |
| `serve_a2a(target=, name=, url=, description=, token_validator=)` *(experimental, 1.1)* | expose a crew / compiled graph / the app over A2A (`A2AServer`, Agent Card + task lifecycle) |
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
| `vincio.documents` | `load_document`, `load_directory`, `load_pdf`, `load_docx`, `load_xlsx`, `extract_markdown_tables`, `extract_code_symbols`, `TesseractOCR` — multimodal loaders (md/html/csv/pdf/docx/xlsx/eml/code), OCR, table/code-aware parsing |
| `vincio.context` | `ContextCompiler`, `ContextPacket`, `ContextIR`, `ContextScorer`, `BudgetAllocator` |
| `vincio.retrieval` | `RetrievalEngine`, `BM25Index`, `VectorIndex`, `SparseIndex`, `LateInteractionIndex`, `AutoMergingIndex`, `LiveIndex`, `QueryUnderstanding`, `EntityGraph`, `GraphRAG`, `ReasoningRetriever`, `chunk_document`, `contextualize_chunks`, `build_embedder`, `JinaEmbedder`, `VoyageEmbedder`, `CohereEmbedder`, `build_reranker`, `CohereReranker`, `JinaReranker`, `VoyageReranker` |
| `vincio.connectors` | `connect`, `register_connector`, `Connector`, `CONNECTORS` — built-in kinds via `connect("web"\|"github"\|"sql"\|"s3"\|"gcs"\|"notion"\|"confluence"\|"slack", ...)` |
| `vincio.memory` | `MemoryEngine`, `ScopedMemory`, `MemoryConsolidator`, `MemoryGraph`, `SessionSummarizer`, `SQLiteMemoryStore`, `evaluate_memory`, `GroundedFact`, `extract_grounded_facts` |
| `vincio.tools` | `ToolRegistry`, `ToolRuntime`, `ToolPermissionChecker`, `SandboxedPython` |
| `vincio.agents` | `AgentExecutor`, `Planner`, `StepDAG`, `HandoffRouter`, `Crew`, `AgentRole`, `Blackboard`, `StateGraph`, `Checkpointer`, `interrupt`, `compose`, `parallel`, `branch`, `LangGraphBackend`, `OpenAIAgentsBackend` |
| `vincio.workflows` | `Workflow` (pause/resume approval gates, edit-and-resume) |
| `vincio.output` | `OutputSchema`, `OutputContract`, `OutputValidator`, `Repairer`, `to_strict_json_schema`, `choice_schema`, `regex_schema`, `StreamingValidator`, `SelfCorrector`, `SchemaRouter` |
| `vincio.evals` | `Dataset`, `dataset_from_traces`, `EvalRunner`, `RunOutput`, `ModelJudge`, `GEvalJudge`, `evaluate_gates`, `METRICS`, `SyntheticGenerator`, `RedTeamSuite`, `ExperimentTracker`, `ab_test`, and *(1.2)* `Trajectory` / `trajectory_from_agent_state` / `Simulator` / `Persona` / `SimulatedConversation` / `OnlineEvaluator` / `DriftMonitor` / `DriftReport` / `AnnotationQueue` / `cohens_kappa` / `Experiment` / `metric_guardrail` |
| `vincio.optimize` | `PromptOptimizer`, `ContextOptimizer`, `RoutingPolicy`, `evolution_loop`, `fitness`, `ImprovementLoop`, `pareto_loop`, `ParetoFrontier`, `ObjectiveSpec`, `DEFAULT_OBJECTIVES`, `AGENTIC_OBJECTIVES`, `RetrievalFeedback`, `recommend_chunking`, `BudgetLearner`, `LearnedAllocations`, `guided_search` |
| `vincio.observability` | `Tracer`, `JSONLExporter`, `OTelExporter`, `CostTracker`, `trace_diff`, `Session`, `sessions_from_traces`, `record_feedback`, `trace_to_html`, `render_trace_text` |
| `vincio.testing` | `assert_eval`, `assert_grounded`, `assert_metric`, `assert_safe`, `Snapshot` (+ pytest plugin: `vincio_snapshot` fixture, `--vincio-update-snapshots`) |
| `vincio.security` | `PIIDetector`, `SecretScanner`, `InjectionDetector`, `AccessController`, `PolicyEngine`, `Rail`, `RailEngine`, `AuditLog` |
| `vincio.caching` | `InMemoryCache`, `SQLiteCache`, `ResponseCache`, `SemanticCache`, `PromptCompileCache`, `ContextCompileCache`, `ChunkCache`, `InvalidationManager` |
| `vincio.providers` | `build_provider`, `openai_compatible`, `OpenAICompatibleProvider`, `OpenAIResponsesProvider`, `PRESETS`, `MockProvider`, `OpenAIProvider`, `AnthropicProvider`, `GoogleProvider`, `MistralProvider`, `LocalProvider`, `FailoverChain`, `CoalescingProvider`. Unified reasoning control (1.1): `RunConfig(reasoning_effort="low"\|"medium"\|"high")` / `thinking_budget_tokens` map to OpenAI reasoning effort, Anthropic extended thinking, and Gemini thinking budgets; thinking tokens are recorded on the model span and billed. |
| `vincio.interop` | `add_langchain_tool`, `from_langchain_tool`, `from_langchain_loader`, `from_langchain_retriever`, `from_langchain_embeddings`, `to_langchain_*`; `add_llamaindex_tool`, `from_llamaindex_reader`, `from_llamaindex_retriever`, `from_llamaindex_embedding`, `to_llamaindex_*` |
| `vincio.mcp` *(experimental, 1.1)* | `MCPClient`, `MCPServer`, `build_app_server`, `serve_stdio`, `connect_stdio`, `connect_http`, `connect_in_process`, `InProcessTransport`, `StdioTransport`, `StreamableHTTPTransport`, `static_token_validator`, `pkce_pair` — MCP client *and* server over stdio / Streamable HTTP / in-process |
| `vincio.a2a` *(experimental, 1.1)* | `A2AServer`, `A2AClient`, `RemoteA2AAgent`, `AgentCard`, `AgentSkill`, `A2ATask`, `crew_a2a_server`, `graph_a2a_server`, `app_a2a_server`, `connect_a2a`, `connect_a2a_in_process`, `static_token_validator` — Agent Card + JSON-RPC task lifecycle |
| `vincio.skills` *(experimental, 1.1)* | `Skill`, `SkillScript`, `SkillLibrary`, `load_skill`, `load_skills`, `parse_skill_md`, `register_skill_scripts` — `SKILL.md` loading with progressive disclosure |
| `vincio.packs` | `Pack`, `load_pack`, `available_packs`, `register_pack` (`app.use_pack(...)`) |
| `vincio.notebook` | `enable_rich_reprs`, `disable_rich_reprs`, `display`, `run_result_html`/`_markdown`, `trace_html`, `eval_report_html`, `memory_item_html`, `search_hit_html` |
| `vincio.tui` | `TUI`, `render_home`, `render_trace`, `render_memory` (`vincio tui`) |
| `vincio.core.concurrency` | `gather_bounded`, `map_bounded` (bounded, cancellation-correct fan-out) |
| `vincio.storage` | `create_metadata_store`, `SQLiteMetadataStore`, `build_vector_index` (memory/qdrant/pgvector/chroma/pinecone/lancedb), Neo4j/Redis/DuckDB adapters |
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
