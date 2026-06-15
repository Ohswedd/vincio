# AGENTS.md — working on the Vincio codebase

## What this is

Vincio (`vincio/`) is a context engineering platform: it compiles prompts,
memory, retrieval, tools, schemas, and policies into budgeted, validated,
traced context packets. Build status and the roadmap live in `ROADMAP.md`.

## Layout

```
vincio/core         types, errors, events, config, tokens, concurrency, ContextApp, 17-step runtime (sync + streaming)
vincio/prompts      PromptSpec, AST, compiler (cache-aware), lint, variants, versioned registry, typed signatures (DSPy-style Signature/Predict)
vincio/context      ContextIR/Packet, scoring, budgeting, compression, compiler; (1.4) llmlingua.py — LLMLinguaCompressor (learned token-importance compression, drop-in ContextCompiler.compressor) + faithfulness helpers
vincio/input        normalization, language/task classification, routing
vincio/documents    loaders (md/html/csv/pdf/docx/xlsx/eml/code), parsers, OCR, multimodal
vincio/retrieval    chunkers, embeddings (local + hosted jina/voyage/cohere + build_embedder), BM25/vector/sparse/late-interaction indexes, hybrid RRF, query understanding, rerankers (heuristic/recency/authority/llm + hosted cohere/jina/voyage), graph+GraphRAG, live indexes (1.3: content-hash upsert re-embeds only changed chunks), reasoning; (1.3) sharded.py ShardedIndex (parallel fan-out shards over the Index protocol)
vincio/connectors   data connectors (web/github/sql/s3/gcs/notion/confluence/slack) feeding the document engine
vincio/interop      LangChain + LlamaIndex bridges (tools/retrievers/loaders/embeddings, both directions; from_* duck-typed, to_* needs the extra)
vincio/mcp          (1.1, experimental) MCP client + server over stdio/Streamable HTTP/in-process; tools→permissioned runtime, resources→evidence, prompts→PromptSpec, sampling→provider, elicitation→human gate; app.add_mcp_server / app.serve_mcp
vincio/a2a          (1.1, experimental) Agent-to-Agent: Agent Card + JSON-RPC task lifecycle, crew/graph exposure, RemoteA2AAgent as a bounded crew delegate; app.serve_a2a
vincio/skills       (1.1, experimental) Agent Skills: SKILL.md loader with progressive disclosure into the compiler, bundled scripts as sandboxed tools; app.add_skill
vincio/packs        opt-in domain packs (support/engineering/finance/legal): prompt+schema+policies+evaluators+golden evals; app.use_pack
vincio/memory       engine (L0–L5), write policy, decay, conflicts, graph, summarizers, grounded-fact auto-memory
vincio/tools        registry, permissioned runtime, sandbox
vincio/agents       bounded DAG executor, planners, ReAct, handoffs, crews + blackboard, durable state graphs (checkpoint/resume/fork), compose/pipe, LangGraph & OpenAI Agents SDK backends
vincio/workflows    deterministic DAG workflows (retries/compensation/approval gates with pause+resume)
vincio/output       schemas, robust parsers, validation pipeline, principled repair, constrained decoding (strict schema transform), streaming validation, self-correction loops, multi-schema routing
vincio/evals        datasets (+synthetic, +from-traces, +multi-turn), metrics (task/grounding/quality/conversational/+1.2 trajectory & tool-use), judges (+G-Eval +Cohen's-κ calibration), runner, gates, reports, experiments (A/B significance), red-teaming; (1.2) Trajectory + RunOutput.from_*, Simulator, OnlineEvaluator (app.add_online_evaluator), DriftMonitor, AnnotationQueue, Experiment (app.experiment), metric_guardrail (app.add_metric_rail)
vincio/optimize     fitness, evolution loop, prompt/context/routing/cache optimization, improvement loop (trace→dataset→eval→optimize→promote), Pareto frontier, retrieval feedback, learned budgets, guided search strategies; (1.3) ModelCascade in routing.py (confidence-gated model escalation); (1.4) reflective.py (GEPA/MIPRO ReflectiveOptimizer), distill.py (export_training_set + BootstrapFinetune flywheel), compression_tuning.py (faithfulness-gated compressor adoption), judge_calibration.py (κ-tuned judge steps)
vincio/observability traces/spans (sessions, feedback, scores), JSONL/OTel (GenAI semconv) exporters, viewer (TUI/HTML/diff), cost tracking; (1.3) finops.py — CostLedger, CostBudget, BudgetManager (cost attribution & budget SLOs)
vincio/testing      assert_eval/assert_grounded/assert_metric/assert_safe, packet/trace snapshots, pytest plugin (pytest11 entry point)
vincio/security     PII/secrets, injection defense, RBAC/ABAC, policy engine, programmable rails, audit
vincio/caching      LRU/SQLite backends, response/retrieval/packet/semantic + compile/chunk caches, invalidation
vincio/storage      metadata stores (memory/sqlite/postgres), qdrant/pgvector/chroma/pinecone/lancedb vector adapters (build_vector_index), neo4j/redis/duckdb adapters
vincio/providers    openai/anthropic/google/mistral/local + OpenAI-compatible passthrough & presets (groq/together/fireworks/openrouter/deepseek/perplexity/xai/nvidia) + OpenAI Responses adapter; unified reasoning control (reasoning_effort/thinking budget, billed); over pooled httpx + coalescing + deterministic mock; (1.3) batch.py (BatchRunner + in-process/OpenAI/Anthropic backends, ~50% cost), circuit.py (CircuitBreaker, HealthAwareFailover), keypool.py (KeyPool, RateLimiter), cache_strategy.py (PromptCacheStrategy)
vincio/notebook     rich Jupyter reprs (enable_rich_reprs) for RunResult/Trace/EvalReport/MemoryItem/SearchHit
vincio/tui          interactive terminal inspector (TUI) for runs/traces/memory; pure renderers + injectable IO
vincio/server       FastAPI app (API key + JWT auth, real-token SSE streaming)
vincio/cli          argparse CLI (init --template, config schema/validate/show, packs, tui, run, eval, prompt, trace, optimize run/reflective, loop, distill, index, memory, audit verify, mcp tools/add/serve)
vincio/stability    API-stability contract (1.0): @deprecated/@experimental, deprecated_alias, stability_of, public_api, API_VERSION, VincioDeprecationWarning/VincioExperimentalWarning
```

## Commands

```bash
.venv/bin/pip install -e ".[dev]"     # setup
.venv/bin/python -m pytest tests/ -q  # full suite (offline, must stay green; example smoke tests add a few seconds)
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_examples.py  # fast core suite (~2s)
.venv/bin/ruff check vincio/ tests/   # lint
```

## Rules

- **Offline-first tests**: everything must pass with no network/API keys —
  use `MockProvider` (it generates schema-valid structured output).
- **Optional dependencies import lazily** inside functions/constructors with
  a helpful `pip install "vincio[extra]"` error. Core deps are only
  pydantic, httpx, typing-extensions, pyyaml.
- **Every public data contract is a Pydantic v2 model**; engines are
  async-first with sync wrappers via `vincio.providers.base.run_sync`.
- **Security is deterministic** — never gate a security decision on model
  output. Policy/permission checks happen in code before execution.
- **Repair never touches facts** — output repair fixes structure only.
- **Every run must produce a trace**; spans nest via contextvars
  (`app.tracer.span(name, type=...)`).
- **Bound every fan-out** — concurrent work goes through
  `vincio.core.concurrency.gather_bounded` (order-preserving, cancellation-
  correct), never a bare `asyncio.gather` over unbounded inputs.
- **Performance is gated** — `python benchmarks/vinciobench.py` +
  `python benchmarks/check_budgets.py` must pass; budgets live in
  `benchmarks/budgets.json` and run in CI. Published SLOs (`benchmarks/slos.json`,
  `docs/reference/slo.md`) are each held by a budget at least as strict;
  `tests/test_slos.py` enforces that invariant.
- **Public API is frozen under SemVer (1.0)** — the public surface is
  `vincio.__all__` plus the documented subsystem entry points. Don't remove or
  break a public symbol in a minor/patch; mark it with `@deprecated(since=,
  removed_in=, alternative=)` and remove only at the next major. New, unproven
  API goes behind `@experimental`. See `docs/reference/stability.md`.
- **Docs stay complete** — `tests/test_docs_completeness.py` requires every
  public subsystem to be documented and every example indexed;
  `tests/test_examples.py` runs all examples offline. Add a doc + example when
  you add a subsystem.
- Update `ROADMAP.md` when adding subsystems or changing release status.
