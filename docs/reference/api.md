# Reference: API

## `vincio.ContextApp`

```python
ContextApp(name, *, objective=None, output_schema=None, config=None,
           provider=None, model=None, budget=None, policies=None, prompt_spec=None)
```

| Method | Purpose |
|---|---|
| `configure(objective=, role=, rules=, examples=, citation_policy=, ...)` | declarative prompt setup |
| `add_source(name, path=, documents=, chunking=, retrieval=)` | load + chunk + index a knowledge source |
| `add_memory(scope=, strategy=, store=)` | enable the memory engine |
| `add_tool(fn_or_name, permissions=, approval_required=, side_effects=)` | enable a tool |
| `add_evaluator(name_or_fn)` | score every run with a metric |
| `add_validator(name, fn, blocking=)` | semantic output validator |
| `add_optimizer(name)` | register an optimization dimension |
| `set_policy(name, value)` | set a run policy (e.g. `answer_only_from_sources`) |
| `run(input, files=, tenant_id=, user_id=, session_id=, config=)` / `arun` | execute the 17-step pipeline → `RunResult` |
| `agent(tools=, planner=, max_steps=, evaluator=)` | bounded agent handle |
| `workflow(name)` | deterministic `Workflow` builder |
| `evaluate(dataset, metrics=, concurrency=, gates=, judges=)` | `EvalReport` |
| `task` (decorator) | configure from a task class |
| `stats()` | sources, tools, memory, cost, run counts |

## `RunResult`

`output` (typed when a schema is set), `raw_text`, `status`, `error`,
`trace_id`, `context_packet_id`, `evidence`, `citations`, `tool_results`,
`usage`, `cost_usd`, `latency_ms`, `validation`, `eval_scores`,
`excluded_context`.

## Key subsystem entry points

| Module | Entry points |
|---|---|
| `vincio.prompts` | `PromptSpec`, `PromptCompiler`, `lint_spec`, `generate_variants`, `diff_specs` |
| `vincio.context` | `ContextCompiler`, `ContextPacket`, `ContextIR`, `ContextScorer`, `BudgetAllocator` |
| `vincio.retrieval` | `RetrievalEngine`, `BM25Index`, `VectorIndex`, `EntityGraph`, `ReasoningRetriever`, `chunk_document` |
| `vincio.memory` | `MemoryEngine`, `MemoryGraph`, `SessionSummarizer`, `SQLiteMemoryStore` |
| `vincio.tools` | `ToolRegistry`, `ToolRuntime`, `ToolPermissionChecker`, `SandboxedPython` |
| `vincio.agents` | `AgentExecutor`, `Planner`, `StepDAG`, `HandoffRouter` |
| `vincio.workflows` | `Workflow` |
| `vincio.output` | `OutputSchema`, `OutputContract`, `OutputValidator`, `Repairer` |
| `vincio.evals` | `Dataset`, `EvalRunner`, `ModelJudge`, `evaluate_gates`, `METRICS` |
| `vincio.optimize` | `PromptOptimizer`, `ContextOptimizer`, `RoutingPolicy`, `evolution_loop`, `fitness` |
| `vincio.observability` | `Tracer`, `JSONLExporter`, `OTelExporter`, `CostTracker`, `trace_diff` |
| `vincio.security` | `PIIDetector`, `SecretScanner`, `InjectionDetector`, `AccessController`, `PolicyEngine`, `AuditLog` |
| `vincio.caching` | `InMemoryCache`, `SQLiteCache`, `ResponseCache`, `SemanticCache`, `InvalidationManager` |
| `vincio.providers` | `build_provider`, `MockProvider`, `OpenAIProvider`, `AnthropicProvider`, `GoogleProvider`, `MistralProvider`, `LocalProvider`, `FailoverChain` |
| `vincio.storage` | `create_metadata_store`, `SQLiteMetadataStore`, Qdrant/pgvector/Neo4j/Redis adapters |
| `vincio.server` | `create_app` (FastAPI) |

All public data contracts are Pydantic models; all engines are async-first
with sync wrappers (`run` / `arun`).
