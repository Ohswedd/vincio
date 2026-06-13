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
| `improvement_loop(metrics=, weights=, gates=, experiment=, ...)` | `ImprovementLoop`: trace → dataset → eval → optimize → promote on this app |
| `use_learned_budgets(source)` | install eval-tuned per-task budget allocations (`LearnedAllocations`, path, or mapping) |
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
| `vincio.context` | `ContextCompiler`, `ContextPacket`, `ContextIR`, `ContextScorer`, `BudgetAllocator` |
| `vincio.retrieval` | `RetrievalEngine`, `BM25Index`, `VectorIndex`, `SparseIndex`, `LateInteractionIndex`, `AutoMergingIndex`, `LiveIndex`, `QueryUnderstanding`, `EntityGraph`, `GraphRAG`, `ReasoningRetriever`, `chunk_document`, `contextualize_chunks` |
| `vincio.connectors` | `connect`, `register_connector`, `WebConnector`, `GitHubConnector`, `SQLConnector`, `S3Connector`, `GCSConnector`, `NotionConnector`, `ConfluenceConnector`, `SlackConnector` |
| `vincio.memory` | `MemoryEngine`, `ScopedMemory`, `MemoryConsolidator`, `MemoryGraph`, `SessionSummarizer`, `SQLiteMemoryStore`, `evaluate_memory`, `GroundedFact`, `extract_grounded_facts` |
| `vincio.tools` | `ToolRegistry`, `ToolRuntime`, `ToolPermissionChecker`, `SandboxedPython` |
| `vincio.agents` | `AgentExecutor`, `Planner`, `StepDAG`, `HandoffRouter`, `Crew`, `AgentRole`, `Blackboard`, `StateGraph`, `Checkpointer`, `interrupt`, `compose`, `parallel`, `branch`, `LangGraphBackend`, `OpenAIAgentsBackend` |
| `vincio.workflows` | `Workflow` (pause/resume approval gates, edit-and-resume) |
| `vincio.output` | `OutputSchema`, `OutputContract`, `OutputValidator`, `Repairer`, `to_strict_json_schema`, `choice_schema`, `regex_schema`, `StreamingValidator`, `SelfCorrector`, `SchemaRouter` |
| `vincio.evals` | `Dataset`, `dataset_from_traces`, `EvalRunner`, `ModelJudge`, `GEvalJudge`, `evaluate_gates`, `METRICS`, `SyntheticGenerator`, `RedTeamSuite`, `ExperimentTracker`, `ab_test` |
| `vincio.optimize` | `PromptOptimizer`, `ContextOptimizer`, `RoutingPolicy`, `evolution_loop`, `fitness`, `ImprovementLoop`, `pareto_loop`, `ParetoFrontier`, `ObjectiveSpec`, `RetrievalFeedback`, `recommend_chunking`, `BudgetLearner`, `LearnedAllocations`, `guided_search` |
| `vincio.observability` | `Tracer`, `JSONLExporter`, `OTelExporter`, `CostTracker`, `trace_diff`, `Session`, `sessions_from_traces`, `record_feedback`, `trace_to_html`, `render_trace_text` |
| `vincio.testing` | `assert_eval`, `assert_grounded`, `assert_metric`, `assert_safe`, `Snapshot` (+ pytest plugin: `vincio_snapshot` fixture, `--vincio-update-snapshots`) |
| `vincio.security` | `PIIDetector`, `SecretScanner`, `InjectionDetector`, `AccessController`, `PolicyEngine`, `Rail`, `RailEngine`, `AuditLog` |
| `vincio.caching` | `InMemoryCache`, `SQLiteCache`, `ResponseCache`, `SemanticCache`, `PromptCompileCache`, `ContextCompileCache`, `ChunkCache`, `InvalidationManager` |
| `vincio.providers` | `build_provider`, `MockProvider`, `OpenAIProvider`, `AnthropicProvider`, `GoogleProvider`, `MistralProvider`, `LocalProvider`, `FailoverChain`, `CoalescingProvider` |
| `vincio.core.concurrency` | `gather_bounded`, `map_bounded` (bounded, cancellation-correct fan-out) |
| `vincio.storage` | `create_metadata_store`, `SQLiteMetadataStore`, Qdrant/pgvector/Neo4j/Redis adapters |
| `vincio.server` | `create_app` (FastAPI) |

All public data contracts are Pydantic models; all engines are async-first
with sync wrappers (`run` / `arun`).
