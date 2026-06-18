# AGENTS.md — working on the Vincio codebase

This is the contributor's map of the Vincio source tree: what each package
does, how to build and test, and the invariants every change must hold. For the
product overview see [`README.md`](README.md); for release status and the
forward plan see [`ROADMAP.md`](ROADMAP.md).

## What Vincio is

Vincio (`vincio/`) is a context-engineering platform. It compiles prompts,
memory, retrieval, tools, schemas, and policies into budgeted, validated, traced
**context packets**, then validates and evaluates every output. The single entry
point is `from vincio import ContextApp`; `app.run()` executes one coherent
pipeline from raw input to traced result.

## Mental model

- **Pydantic v2 everywhere.** Every public data contract (`RunResult`, `Budget`,
  `EvidenceItem`, `MemoryItem`, `EvalReport`, …) is a Pydantic v2 model.
- **Async-first, sync wrappers.** Engines expose `arun`/async methods; the sync
  `run()` is a thin wrapper over `vincio.providers.base.run_sync` and works with
  or without a running loop. Stream with `async for event in app.astream(...)`.
- **One run pipeline.** `app.run()` is: normalize → classify → policy → memory
  recall → retrieve → compile context (score / dedupe / conflict / compress /
  budget) → compile prompt (cache-aware) → model (+ bounded tool loop) → validate
  (schema / citations / policy, principled repair) → evaluate → trace → memory
  write.
- **Deterministic where it matters.** Security, permissions, validation, and
  budgets are enforced in code, never gated on model output.
- **Offline by default.** With no provider or key, `MockProvider` emits
  schema-valid output so the whole pipeline runs in CI without a network.
- **Capability in your process.** Observability, evaluation, distribution, and
  the agent fabric run on your own infrastructure — never a hosted control plane.

## Package layout

```
vincio/core           types, errors, config, tokens, concurrency, media; ContextApp + the run pipeline (sync + streaming) with enforced Budget hard caps and cooperative cancellation (app.submit → RunHandle); six lazy capability facades (runs / knowledge / governance / optimization / serving / training); the typed, versioned event catalog (EventBus)
vincio/prompts        PromptSpec, prompt AST, cache-aware compiler, lint rules, variants, the versioned prompt registry, and typed DSPy-style signatures (Signature / Predict)
vincio/context        ContextIR / ContextPacket, scoring, budgeting, compression, and the context compiler; optional semantic scoring (embedding-cosine + MMR), learned-importance compression, entailment-linked evidence, and a content-addressed evidence store for cross-process packets; multimodal-native (text / image / table evidence)
vincio/input          input normalization, language / task classification, routing
vincio/documents      loaders (md / html / csv / pdf / docx / xlsx / eml / code / pptx / epub / audio), parsers, layout-aware PDF extraction, OCR with auto-fallback, form extraction, and a parser registry; turns documents into evidence
vincio/retrieval      chunkers, embeddings (local + hosted), BM25 / vector / sparse / late-interaction indexes, hybrid RRF, query understanding, rerankers, graph + GraphRAG, and live (content-hash) indexes; serializable FilterSpec pushed down server-side per backend, Matryoshka / contextual / multimodal embedders, sharded fan-out, and two-stage quantized search; local neural models run offline against injected weights
vincio/connectors     data connectors (web / github / sql / s3 / gcs / notion / confluence / slack) feeding the document engine
vincio/interop        LangChain + LlamaIndex bridges (tools / retrievers / loaders / embeddings, both directions)
vincio/mcp            MCP client + server over stdio / Streamable HTTP / in-process; tools → permissioned runtime, resources → evidence, prompts → PromptSpec, sampling → provider, elicitation → human gate (app.add_mcp_server / app.serve_mcp)
vincio/a2a            Agent-to-Agent: Agent Card + JSON-RPC task lifecycle, crew / graph exposure, RemoteA2AAgent as a bounded crew delegate (app.serve_a2a)
vincio/skills         Agent Skills: SKILL.md loader with progressive disclosure into the compiler, bundled scripts as sandboxed tools (app.add_skill)
vincio/packs          opt-in domain packs (support / engineering / finance / legal): prompt + schema + policies + evaluators + golden evals (app.use_pack)
vincio/memory         memory engine (L0–L5), write policy, decay, conflict resolution, graph, summarizers, grounded-fact auto-memory; bi-temporal items with ACL / purpose / consent fields, team scope, history-preserving correction, and as-of / reader / consent-filtered recall
vincio/tools          tool registry, permissioned runtime, sandbox
vincio/agents         bounded DAG executor, planners, ReAct, handoffs, crews + blackboard, durable state graphs (checkpoint / resume / fork), compose / pipe; distributed execution with TTL-lease + checkpoint-version CAS for exactly-once super-steps, BSP parallel super-steps + Send map-reduce, and LangGraph / OpenAI Agents SDK / Ray / Temporal export adapters
vincio/workflows      deterministic DAG workflows (retries / compensation / approval gates with pause + resume)
vincio/output         schemas, robust parsers, validation pipeline, principled repair, constrained decoding, streaming validation, self-correction loops, multi-schema routing
vincio/generation     documents & media flowing OUT — DocumentModel IR, DocumentContract + validation, markdown / html / docx / pdf / pptx render, cited reports ([E1] → footnotes with per-claim entailment), template / form fill, image and speech providers with cost metering and provenance (app.build_document / cited_report / generate_image / synthesize_speech)
vincio/evals          datasets (+ synthetic, + from-traces, + multi-turn), metrics (task / grounding / quality / conversational / trajectory & tool-use), judges (+ G-Eval + κ calibration), runner, gates, reports, A/B experiments with significance, red-teaming; trace replay, online evaluation, drift monitoring, annotation queues, and the SwapGate model-swap regression contract
vincio/optimize       fitness, evolution loop, prompt / context / routing / cache optimization, the improvement loop (trace → dataset → eval → optimize → promote), Pareto frontier, learned budgets; reflective (GEPA / MIPRO) optimizers, the distillation flywheel, model cascade + capability-aware Router, and the unified self-improvement controller with canary-gated deploy (app.self_improvement, app.deploy — offline dataset or live-traffic canary with auto-rollback)
vincio/observability  traces / spans (sessions, feedback, scores), JSONL / OTel (GenAI semconv) exporters, viewer (TUI / HTML / diff), cost tracking; FinOps cost ledger + budget SLOs, an indexed trace / cost store with rollups, a served dashboard (serve_viewer), an alert rule engine (threshold / EWMA / burn-rate over webhook / Slack / PagerDuty / Prometheus), and an off-by-default content-capture gate at the export boundary
vincio/testing        assert_eval / assert_grounded / assert_metric / assert_safe, packet / trace snapshots, the pytest plugin, and assert_backend_conformance — the offline contract a runtime backend must satisfy vs the native durable engine
vincio/security       PII / secrets, injection defense (normalization + recursive decode pre-pass, pluggable detector backends), RBAC / ABAC, the policy engine, programmable rails, RAG-poisoning detection, fail-closed tenant isolation, always-on egress DLP, and a hash-chained audit log with per-entry signatures + Merkle checkpoints
vincio/governance     compliance evidence over the live system — model / system cards, framework mapping (OWASP LLM / Agentic / NIST AI RMF / MITRE ATLAS / ISO 42001), AI-BOM, C2PA provenance marking, source → output lineage with signed erasure proofs, residency-aware egress, EU AI Act risk tiering / Annex IV / FRIA, and the consent ledger
vincio/caching        LRU / SQLite backends; response / retrieval / packet / semantic / compile / chunk caches, with invalidation
vincio/storage        metadata stores (memory / sqlite / postgres) and vector adapters (qdrant / pgvector / chroma / pinecone / lancedb / weaviate / milvus / elasticsearch / opensearch / vespa) plus neo4j / redis / duckdb; the async store contract is canonical (asave / aquery), with shared-state rate-limit / idempotency primitives (in-memory + Redis)
vincio/providers      openai / anthropic / google / mistral / local + OpenAI-compatible passthrough & presets; unified reasoning control, pooled httpx with coalescing, the deterministic mock, batch backends (~50% cost), circuit breaker + health-aware failover, key pool, and prompt-cache strategy; a data-driven ModelRegistry (capabilities + pricing + lifecycle) drives capability guards, shadow / canary dispatch, lifecycle migration, fine-tune backends, and enterprise auth (Bedrock / Vertex / Azure OpenAI)
vincio/notebook       rich Jupyter reprs (enable_rich_reprs) for RunResult / Trace / EvalReport / MemoryItem / SearchHit
vincio/tui            interactive terminal inspector for runs / traces / memory; pure renderers + injectable IO
vincio/server         FastAPI app (API key + JWT auth, real-token SSE streaming), health / readiness / Prometheus metrics, graceful shutdown, optional Redis-coherent rate-limit middleware
vincio/realtime       (optional) voice / realtime — RealtimeSession, connect_realtime (in-process / OpenAI / Gemini), VAD, interruption, in-session tool calls through the permissioned runtime (app.realtime_session; extra vincio[realtime])
vincio/cli            argparse CLI: init, config, packs, tui, run, eval, prompt, trace, optimize, loop, distill, index, memory, audit, governance, mcp, and serve (uvicorn HTTP launcher)
vincio/stability      the API-stability contract — @deprecated / @experimental, deprecated_alias, stability_of, public_api, API_VERSION, and the Vincio deprecation / experimental warnings
```

## Commands

```bash
.venv/bin/pip install -e ".[dev]"     # setup
.venv/bin/python -m pytest tests/ -q  # full suite (offline, must stay green)
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_examples.py  # fast core suite (~2s)
.venv/bin/ruff check vincio/ tests/   # lint
.venv/bin/python -m mypy vincio        # type check (CI gate; must stay clean)
```

CI gates (`.github/workflows/ci.yml`): ruff, **mypy (`mypy vincio`)**, pytest on
py3.11 / 3.12 / 3.13, VincioBench budgets, and a package build.

## Rules

- **Offline-first tests** — everything must pass with no network or API keys; use
  `MockProvider` (it generates schema-valid structured output).
- **Optional dependencies import lazily** inside functions / constructors with a
  helpful `pip install "vincio[extra]"` error. Core deps are only `pydantic`,
  `httpx`, `typing-extensions`, and `pyyaml`.
- **Every public data contract is a Pydantic v2 model**; engines are async-first
  with sync wrappers via `vincio.providers.base.run_sync`.
- **Security is deterministic** — never gate a security decision on model output.
  Policy and permission checks happen in code before execution.
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
- **Public API is frozen under SemVer** — the public surface is `vincio.__all__`
  plus the documented subsystem entry points. Don't remove or break a public
  symbol in a minor / patch; mark it with `@deprecated(since=, removed_in=,
  alternative=)` and remove only at the next major. New, unproven API goes behind
  `@experimental`. See `docs/reference/stability.md`.
- **Docs stay complete** — `tests/test_docs_completeness.py` requires every public
  subsystem to be documented and every example indexed; `tests/test_examples.py`
  runs all examples offline. Add a doc + example when you add a subsystem.
- **Update `ROADMAP.md`** when adding subsystems or changing release status.
