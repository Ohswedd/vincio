# AGENTS.md — working on the Vincio codebase

This is the contributor's and agent's map of the Vincio source tree: what each
package does, how to build and test it, the invariants every change must hold,
and the checklist for adding a subsystem. For the product overview see
[`README.md`](README.md); for release status and the forward plan see
[`ROADMAP.md`](ROADMAP.md); for the full release history see
[`CHANGELOG.md`](CHANGELOG.md).

> If you are an AI coding agent: read [`llms.txt`](llms.txt) first — it is the
> generated, always-current digest of the entire public surface, the capability
> map, and the gotchas, derived from `vincio.__all__`.

## What Vincio is

Vincio (`vincio/`) is a context-engineering platform. It compiles prompts,
memory, retrieval, tools, schemas, and policies into budgeted, validated, traced
**context packets**, then validates and evaluates every output. The single entry
point is `from vincio import ContextApp`; `app.run()` executes one coherent
pipeline from raw input to traced result. It is a **library you run yourself** —
there is no hosted service, control plane, or account.

## Mental model

- **Pydantic v2 everywhere.** Every public data contract (`RunResult`, `Budget`,
  `EvidenceItem`, `MemoryItem`, `EvalReport`, …) is a Pydantic v2 model.
- **Async-first, sync wrappers.** Engines expose `arun` / async methods; the sync
  `run()` is a thin wrapper over `vincio.providers.base.run_sync` and works with
  or without a running loop. Stream with `async for event in app.astream(...)`.
- **One run pipeline.** `app.run()` is: normalize → classify → policy → memory
  recall → retrieve → compile context (score / dedupe / conflict / compress /
  budget) → compile prompt (cache-aware) → model (+ bounded tool loop) → validate
  (schema / citations / policy, principled repair) → evaluate → trace → memory
  write.
- **Deterministic where it matters.** Security, permissions, validation, and
  budgets are enforced in code, never gated on model output. Same input → same
  packet.
- **Offline development uses the bundled mock — explicitly.** The default
  provider is OpenAI; to run with no key, pass `provider=MockProvider()` (it
  emits schema-valid output, so the whole pipeline runs in CI with no network).
  The examples and tests do this via `examples/_shared.example_provider`.
- **Capability in your process.** Observability, evaluation, distribution, and
  the agent fabric run on your own infrastructure, never a hosted control plane.
- **The public surface is `vincio.__all__`**, frozen under SemVer
  (`API_VERSION = "5.0"`). New, unproven API ships `@experimental`; nothing is
  removed before its `removed_in` runway.

## Package layout

```
vincio/core           types, errors, config, tokens, concurrency, media; ContextApp + the run pipeline (sync + streaming) with enforced Budget hard caps and cooperative cancellation (app.submit → RunHandle); six lazy capability facades (runs / knowledge / governance / optimization / serving / training); the typed, versioned event catalog (EventBus)
vincio/prompts        PromptSpec, prompt AST, cache-aware compiler, lint rules, variants, the versioned prompt registry, and typed DSPy-style signatures (Signature / Predict)
vincio/context        ContextIR / ContextPacket, scoring, budgeting, compression, and the context compiler; optional semantic scoring (embedding-cosine + MMR), learned-importance compression, entailment-linked evidence, and a content-addressed evidence store for cross-process packets; multimodal-native (text / image / table evidence)
vincio/input          input normalization, language / task classification, routing
vincio/documents      loaders (md / html / csv / pdf / docx / xlsx / eml / code / pptx / epub / audio), parsers, layout-aware PDF extraction, OCR with auto-fallback, form extraction, and a parser registry; turns documents into evidence
vincio/retrieval      chunkers, embeddings (local + hosted), BM25 / vector / sparse / late-interaction indexes, hybrid RRF, query understanding, rerankers, graph + GraphRAG, and live (content-hash) indexes; serializable FilterSpec pushed down server-side per backend, Matryoshka / contextual / multimodal embedders, sharded fan-out, and two-stage quantized search; local neural models run offline against injected weights
vincio/connectors     data connectors (web / github / sql / s3 / gcs / notion / confluence / slack / jira / linear / gdrive / sharepoint / salesforce / zendesk / bigquery / snowflake) feeding the document engine with provenance; REST connectors on core httpx, warehouse connectors on injected clients; the SQL-family connectors take an opt-in reservoir `sample=` that stands a representative sample in for the first-N cutoff
vincio/data           the data & analytics plane: a typed, columnar Dataset + lossless, header-once DataEncoder + TableEvidence (app.table_evidence); deterministic bounded-memory profile_dataset / sample_dataset / fit_to_window and DataQualityRails (app.profile_dataset / sample_dataset / fit_dataset / screen_data); governed text-to-query with cell-level provenance (app.query_data); the multi-step analysis agent (app.analyze_data); content- and data-bound charts (app.generate_chart); streaming / out-of-core (RowStream / stream_aggregate / encode_stream; app.stream_dataset / aggregate_stream / map_stream); the semantic layer (app.semantic_layer / query_metric / metric_lineage); real-time windowed analytics over an unbounded event stream (StreamWindow tumbling/sliding/session in streaming_analytics.py — windowed profile/query/query_metric/screen/aggregate emitting per-window results that cite the exact events via EventCitation and verify() against a bounded CapturedWindow; app.stream_analytics → governed StreamingAnalytics driver, live or replayed); the capstone DataEngagement (app.data_engagement) threading the whole plane into a signed, hash-chained, data-bound DataNarrative; and cross-org federated analytics (app.federated_data_engagement → FederatedDataEngagement running one governed FederatedQuery across orgs over the cross-org fabric — negotiated as a Contract, choreographed as a Saga, only aggregated cell-cited MetricResults cross, never raw rows — into a signed, data-bound FederatedNarrative, with residency/consent/DP governance at the boundary); and the notebook-native surface binding the plane into the vincio/notebook reprs + a governed notebook_session (see vincio/notebook); held by the data_plane and data_analysis_conformance VincioBench families
vincio/tasks          the ergonomic front door (5.3): one-line, task-shaped constructors rag / extractor / tool_agent / evaluation / chat over ContextApp plus the fluent, immutable Flow; each @experimental, re-exported at top level, and lowers byte-identical to ContextApp.run (proven by vincio.testing.run_signature). `.app` is the escape hatch to every deep method
vincio/interop        LangChain / LlamaIndex / Haystack / DSPy bridges (tools / retrievers / loaders / embedders / components / compiled modules, both directions)
vincio/plugins        the versioned entry-point plugin contract, discover / load third-party providers, embedders, stores, connectors, chunkers, rerankers, judges, metrics, and packs on install (installed_plugins / load_plugins; vincio plugins list)
vincio/mcp            MCP client + server over stdio / Streamable HTTP / in-process; tools → permissioned runtime, resources → evidence, prompts → PromptSpec, sampling → provider, elicitation → human gate (app.add_mcp_server / app.serve_mcp); marketplace bridge (app.add_mcp_from_registry) discovers → governs → lands tools in one call
vincio/a2a            Agent-to-Agent: Agent Card + JSON-RPC task lifecycle, crew / graph exposure, RemoteA2AAgent as a bounded crew delegate (app.serve_a2a)
vincio/negotiation    bounded, terminating offer/counter-offer bargain minting a typed, signed, offline-verifiable Contract (price / SLA / scope / quality, enforced like a budget) over the A2A fabric, reputation-weighted (app.negotiate / serve_negotiation / enforce_contract)
vincio/choreography   durable, compensating cross-org saga over A2A + the negotiated Contract (Saga / Choreography / Participant); per-org self-governance, a hash-chained restart-surviving SagaJournal, deterministic reverse-order compensation (app.choreograph / resume_choreography / serve_choreography)
vincio/settlement     the cross-org settlement & credit fabric: metered settlement over a Contract (Meter / SettlementRecord / SettlementBook); multilateral netting (net_settlements / app.clear_settlements); dispute arbitration (app.arbitrate); portable reputation (attestations, freshness/revocation, gossip, transitive trust, admission); collateral (escrow, pooling, rehypothecation guards); proof-of-reserves / solvency / liability completeness / non-equivocation / history; insolvency resolution (seniority waterfall + close-out set-off); and the CrossOrgEngagement capstone (app.cross_org_engagement); held by cross_org_conformance
vincio/registry       governed agent fabric (AgentDirectory over A2A / ACP / MCP-registry, allow-list-gated + audited) and the signed community pack & skill registry (CommunityRegistry)
vincio/skills         Agent Skills: SKILL.md loader with progressive disclosure into the compiler, bundled scripts as sandboxed tools (app.add_skill)
vincio/packs          opt-in packs (app.use_pack): domain packs (support / engineering / finance / legal) and full-stack vertical packs (healthcare / ediscovery / kyc / customer_support / code_review) that also preconfigure retrieval / scoped memory / rails / metrics / residency
vincio/assistant      Assistant, a conversational, session-aware layer over ContextApp (app.assistant): session threading, multi-turn state via session-scoped memory write-back, and a tool-approval surface; drives as a Simulator target
vincio/memory         memory engine (L0–L5), write policy, decay, conflict resolution, graph, summarizers, grounded-fact auto-memory; bi-temporal items with ACL / purpose / consent fields, team scope, history-preserving correction, as-of / reader / consent-filtered recall
vincio/tools          tool registry, permissioned runtime (RBAC / ABAC), resource-limited sandbox; computer-use action plane (app.computer_use) over a pluggable ScreenBackend
vincio/agents         bounded DAG executor, planners (direct / static / dynamic / ReAct / plan-and-execute / hierarchical HTN), in-place plan repair, cost-aware action selection, crews + blackboard, durable state graphs (checkpoint / resume / fork) with durable timers, distributed execution (TTL-lease + CAS super-steps, BSP, Send map-reduce, work-stealing scheduler), and LangGraph / OpenAI Agents SDK / Ray / Temporal export adapters
vincio/workflows      deterministic DAG workflows (retries / compensation / approval gates with pause + resume)
vincio/output         schemas, robust parsers, validation pipeline, principled repair, constrained decoding, streaming validation, self-correction loops, multi-schema routing
vincio/generation     documents & media flowing OUT: DocumentModel IR, DocumentContract + validation, markdown / html / docx / pdf / pptx render, cited reports ([E1] → footnotes with per-claim entailment), template / form fill, image and speech providers with cost metering and provenance (app.build_document / cited_report / generate_image / synthesize_speech)
vincio/evals          datasets (+ synthetic / from-traces / multi-turn), metrics (task / grounding / quality / conversational / trajectory & tool-use), judges (+ G-Eval + κ calibration) and κ-gated ensembles, runner, gates, reports, A/B experiments with significance, red-teaming; trace replay, online evaluation, drift monitoring, the SwapGate model-swap contract, Shapley regression attribution, adaptive sampling, and nine agentic benchmark adapters behind one contract
vincio/optimize       fitness, evolution loop, prompt / context / routing / cache optimization, the improvement loop (trace → dataset → eval → optimize → promote), Pareto frontier, learned budgets; reflective (GEPA / MIPRO) optimizers, the distillation flywheel, model cascade + capability-aware Router, the unified self-improvement controller with canary-gated deploy; RLVR (rewards.py, trajectory_opt.py, app.learn); on-device LoRA local adaptation; federated self-improvement; cross-fleet reputation
vincio/observability  traces / spans (sessions, feedback, scores), JSONL / OTel (GenAI semconv) exporters, viewer (TUI / HTML / diff), cost tracking; FinOps cost ledger + budget SLOs, an indexed trace / cost store with rollups, a served dashboard (serve_viewer), an alert rule engine, energy / carbon accounting, and an off-by-default content-capture gate
vincio/testing        assert_eval / assert_grounded / assert_metric / assert_safe, packet / trace snapshots, the pytest plugin, the byte-identical-lowering harness (run_signature / selection_signature), and assert_backend_conformance
vincio/security       PII / secrets, injection defense (normalization + recursive decode pre-pass, pluggable detector backends), RBAC / ABAC, the policy engine, programmable rails, RAG-poisoning detection, fail-closed tenant isolation, always-on egress DLP, a hash-chained audit log with per-entry signatures + Merkle checkpoints, provable prompt-injection containment (taint labels + capability tokens + DualPlaneExecutor), and agent identity / delegation (DID + attenuating chains)
vincio/governance     compliance evidence over the live system, model / system cards, framework mapping (OWASP LLM / Agentic / NIST AI RMF / MITRE ATLAS / ISO 42001), AI-BOM, C2PA provenance marking, source → output lineage with signed erasure proofs, residency-aware egress, EU AI Act risk tiering / Annex IV / FRIA, the consent ledger, the governance-invariant verifier (app.verify_governance), and continuous assurance cases (app.assurance_case / certify)
vincio/verify         verified reasoning & neuro-symbolic certificates (app.verify_reasoning), incl. statistical kernels (Trend/Correlation/Interval/Forecast) that certify a data answer's analytical claims from its cited cells and refute correlation-stated-as-causation (statistical_claims=); runtime behaviour shielding (app.shield / behavior_monitor / use_shield), and verified tool use / program synthesis (app.synthesize_program); deterministic kernels dependency-free, optional SMT / CAS behind vincio[verify]
vincio/cultivate      autonomous skill acquisition & open-ended curriculum (app.cultivate): propose → attempt → verify → distill → promote under the no-regression gate; content-addressed LearnedSkillLibrary
vincio/assurance      the assurance-case engine (Claim / Evidence binders over the platform's own verdicts, freshness horizons, incidents, certification)
vincio/edge           the dependency-free compile → score → rail → pack core packaged for constrained / WASM targets (app.edge_runtime); verify_edge_parity / edge_manifest prove parity-not-a-fork
vincio/caching        LRU / SQLite backends; response / retrieval / packet / semantic / compile / chunk caches, with invalidation; learned semantic cache + near-miss KV reuse
vincio/storage        metadata stores (memory / sqlite / postgres) and vector adapters (qdrant / pgvector / chroma / pinecone / lancedb / weaviate / milvus / elasticsearch / opensearch / vespa) plus neo4j / redis / duckdb; canonical async store contract (asave / aquery); shared-state rate-limit / idempotency primitives (in-memory + Redis)
vincio/providers      openai / anthropic / google / mistral / local + OpenAI-compatible passthrough & presets; unified reasoning control, pooled httpx with coalescing, the deterministic mock, batch backends (~50% cost), circuit breaker + health-aware failover, key pool, prompt-cache strategy; a data-driven ModelRegistry (capabilities + pricing + lifecycle, shipped as model_catalog.json) driving capability guards, shadow / canary dispatch, lifecycle migration, fine-tune backends, and enterprise auth (Bedrock / Vertex / Azure OpenAI); coverage gate (registry.coverage_report / vincio registry coverage)
vincio/notebook       rich Jupyter reprs (enable_rich_reprs) for RunResult / Trace / EvalReport / MemoryItem / SearchHit — and the data artifacts QueryResult / AnalysisResult / Chart / DataNarrative (clickable cell citations, lineage verdict, audit id); notebook_session(app, ...) → NotebookSession, a thin governed front over app.data_engagement threading register → query → analyze → chart → cite into the same signed DataNarrative a script seals, session.verify() re-derives every inline finding from the bytes; held by the data_plane.notebook VincioBench sub-family (repr_faithful / session_verifies)
vincio/tui            interactive terminal inspector for runs / traces / memory; pure renderers + injectable IO
vincio/server         FastAPI app (API key + JWT auth, real-token SSE streaming), health / readiness / Prometheus metrics, graceful shutdown, optional Redis-coherent rate-limit middleware, AG-UI generative-UI events (from vincio.server import create_app; vincio serve)
vincio/realtime       (optional) voice / realtime, RealtimeSession, connect_realtime (in-process / OpenAI / Gemini), VAD, interruption, in-session tools (app.realtime_session; vincio[realtime]); and the end-to-end VoiceAgent (app.voice_agent)
vincio/cli            argparse CLI: init, config, packs, plugins, tui, run, batch, cost, eval, prompt, trace, optimize, loop, distill, index, memory, audit, governance, mcp, providers, registry, docs, and serve (uvicorn HTTP launcher)
vincio/stability      the API-stability contract, @deprecated / @experimental, deprecated_alias, stability_of, public_api, API_VERSION, and the Vincio deprecation / experimental warnings
vincio/_apiref.py     docstring-driven public API reference generator (renders docs/reference/api-generated.md from vincio.__all__); the docstring-coverage and frozen-public-surface gates
vincio/_docmap.py     the connected-docs doc graph: binds every public app.* verb to its concept / guide / example / reference (by capability facade) and renders docs/reference/capability-map.md, docs/learning-path.md, the api.md app-method index, the per-page Related blocks, and llms.txt; the docs-graph checks behind `vincio docs check`
```

## Examples & applications

The examples are a three-tier on-ramp, all runnable fully offline on the mock:

```
examples/00–22            complete, heavily-commented feature tours, one per subsystem
                          (00 one-liners, 01 quickstart, … 22 connected docs); run under
                          tests/test_examples.py, which globs examples/[0-9]*.py.
examples/_shared.py       example_provider() (mock offline / real via VINCIO_PROVIDER), responders, sample docs.
examples/notebooks/*.ipynb  Google Colab-ready notebooks (pip install + offline run); code cells are
                          gated to parse and run offline by tests/test_example_notebooks.py.
examples/applications/    real-world small backends. Each splits a dependency-free, offline-testable
                          core.py from a FastAPI main.py (CI has no fastapi, so core.py must never
                          need it); rag_service, support_triage_api, extraction_service, and a
                          single-file CLI cli_research_agent. Gated by tests/test_example_apps.py.
```

When you add a numbered feature tour, bump the count assertion in
`tests/test_examples.py` and add the row to `examples/README.md` and the README.

## The docs & capability system

The docs are a connected graph, not loose pages. Three generators keep them
honest and current; never hand-edit a generated file — edit its generator and
regenerate.

- **`vincio/_apiref.py`** renders `docs/reference/api-generated.md` and freezes
  the public surface (`docs/reference/public-surface.txt`).
- **`vincio/core/error_catalog.py`** renders `docs/reference/errors.md`.
- **`vincio/_docmap.py`** renders `docs/reference/capability-map.md`,
  `docs/learning-path.md`, the `api.md` app-method index, the per-page **Related**
  blocks, and `llms.txt`, and runs the docs-graph check.

```bash
vincio docs map      # regenerate every generated doc artifact (use --check to gate freshness)
vincio docs check    # docs-graph gate: links resolve, every app.* verb mapped & in api.md, no orphans, llms.txt current
```

The completeness gate (`tests/test_docs_graph.py`, bridged from
`tests/test_docs_completeness.py`) fails the build on any drift. Add a concept,
a guide, an example, and a `Topic` entry in `_docmap.py::TOPICS` when you add a
subsystem.

## Commands

```bash
.venv/bin/pip install -e ".[dev]"     # setup
.venv/bin/python -m pytest tests/ -q  # full suite (offline, must stay green)
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_examples.py  # fast core suite
.venv/bin/ruff check vincio/ tests/   # lint
.venv/bin/python -m mypy vincio        # type check (CI gate; must stay clean)
.venv/bin/python benchmarks/vinciobench.py && .venv/bin/python benchmarks/check_budgets.py  # perf/quality budgets
.venv/bin/python -m vincio.cli.main docs check   # docs-graph gate
```

CI gates (`.github/workflows/ci.yml`): ruff, **mypy (`mypy vincio`)**, pytest on
py3.11 / 3.12 / 3.13, a line+branch coverage floor, VincioBench budgets, and a
package build. CI installs only `.[dev]` — so optional-dependency code paths
(fastapi, vector stores, …) must import lazily and tests for them must skip when
the dependency is absent.

## Rules (invariants every change holds)

- **Offline-first tests**: everything passes with no network or API keys; use
  `MockProvider` (it generates schema-valid structured output). The default
  provider is OpenAI, so tests/examples pass the mock explicitly.
- **Optional dependencies import lazily** inside functions / constructors with a
  helpful `pip install "vincio[extra]"` error. Core deps are only `pydantic`,
  `httpx`, `typing-extensions`, and `pyyaml`.
- **Every public data contract is a Pydantic v2 model**; engines are async-first
  with sync wrappers via `vincio.providers.base.run_sync`.
- **Security is deterministic**: never gate a security decision on model output.
  Policy and permission checks happen in code before execution.
- **Repair never touches facts**: output repair fixes structure only.
- **Every run must produce a trace**; spans nest via contextvars
  (`app.tracer.span(name, type=...)`).
- **Bound every fan-out**: concurrent work goes through
  `vincio.core.concurrency.gather_bounded` (order-preserving, cancellation-
  correct), never a bare `asyncio.gather` over unbounded inputs.
- **Performance is gated**: `benchmarks/vinciobench.py` + `check_budgets.py` must
  pass; budgets live in `benchmarks/budgets.json`. Published SLOs
  (`benchmarks/slos.json`, `docs/reference/slo.md`) are each held by a budget at
  least as strict; `tests/test_slos.py` enforces that invariant.
- **Public API is frozen under SemVer**: the public surface is `vincio.__all__`
  plus the documented subsystem entry points. Don't remove or break a public
  symbol in a minor / patch; mark it `@deprecated(since=, removed_in=,
  alternative=)` and remove only at the next major. New, unproven API goes behind
  `@experimental`. See `docs/reference/stability.md`.
- **Docs stay complete and current**: `tests/test_docs_graph.py` /
  `test_docs_completeness.py` require the docs graph to be intact and every
  generated artifact current; `tests/test_examples.py` runs all feature tours
  offline. Run `vincio docs map` after any docstring or doc-graph change.
- **Update `ROADMAP.md` and `CHANGELOG.md`** when adding a subsystem or changing
  release status, and bump the version in `pyproject.toml` + `vincio/__init__.py`.

## Adding a subsystem (the checklist)

1. Implement it async-first behind a public entry point (a `ContextApp` method
   and/or a top-level symbol added to `vincio.__all__`), optional deps lazy.
2. Add a docstring to every public symbol (the coverage gate requires it) and a
   stable `.code` to any new error (the error catalog requires it).
3. Add a concept page, a guide, and a numbered runnable example; wire them into
   `vincio/_docmap.py::TOPICS` so every new `app.*` verb is mapped.
4. Add a VincioBench family + SLOs (`slos.json`) + budgets (`budgets.json`).
5. Run the full gate set above (`vincio docs map`, pytest, ruff, mypy, bench).
6. Update `ROADMAP.md`, `CHANGELOG.md`, the README, and the version.
