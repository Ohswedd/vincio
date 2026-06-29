# AGENTS.md: working on the Vincio codebase

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
  the agent fabric run on your own infrastructure, never a hosted control plane.

## Package layout

```
vincio/core           types, errors, config, tokens, concurrency, media; ContextApp + the run pipeline (sync + streaming) with enforced Budget hard caps and cooperative cancellation (app.submit → RunHandle); six lazy capability facades (runs / knowledge / governance / optimization / serving / training); the typed, versioned event catalog (EventBus)
vincio/prompts        PromptSpec, prompt AST, cache-aware compiler, lint rules, variants, the versioned prompt registry, and typed DSPy-style signatures (Signature / Predict)
vincio/context        ContextIR / ContextPacket, scoring, budgeting, compression, and the context compiler; optional semantic scoring (embedding-cosine + MMR), learned-importance compression, entailment-linked evidence, and a content-addressed evidence store for cross-process packets; multimodal-native (text / image / table evidence)
vincio/input          input normalization, language / task classification, routing
vincio/documents      loaders (md / html / csv / pdf / docx / xlsx / eml / code / pptx / epub / audio), parsers, layout-aware PDF extraction, OCR with auto-fallback, form extraction, and a parser registry; turns documents into evidence
vincio/retrieval      chunkers, embeddings (local + hosted), BM25 / vector / sparse / late-interaction indexes, hybrid RRF, query understanding, rerankers, graph + GraphRAG, and live (content-hash) indexes; serializable FilterSpec pushed down server-side per backend, Matryoshka / contextual / multimodal embedders, sharded fan-out, and two-stage quantized search; local neural models run offline against injected weights
vincio/connectors     data connectors (web / github / sql / s3 / gcs / notion / confluence / slack / jira / linear / gdrive / sharepoint / salesforce / zendesk / bigquery / snowflake) feeding the document engine with provenance; REST connectors on core httpx, warehouse connectors on injected clients; the SQL-family connectors take an opt-in reservoir `sample=` that stands a representative sample in for the first-N cutoff
vincio/data           the data & analytics plane: a typed, columnar Dataset + lossless, header-once DataEncoder + TableEvidence (app.table_evidence) so structured data is first-class, schema-bearing, token-cheap evidence (encoding kernel in vincio/core/tabular); deterministic, bounded-memory profile_dataset / profile_stream (DatasetProfile, itself fixed-size evidence); reservoir / stratified / systematic sample_dataset (representative, not first-N); fit_to_window / fit_stream (WindowFit) fit a table far larger than the window under a fixed token budget, size invariant to row count; DataQualityRails (app.screen_data, audited) screen for schema / constraint / anomaly defects + PII / secrets / injection in string cells on the same deterministic rail path (app.profile_dataset / sample_dataset / fit_dataset / screen_data); streaming / out-of-core (vincio/data/streaming): a lazy, re-iterable RowStream over a source larger than memory (records / generator factory / CSV / JSON-Lines read line by line) processed in bounded passes, bounded-memory stream_aggregate group-by, header-once encode_stream (gzip), and stream_map at scale on the BatchRunner (app.stream_dataset / aggregate_stream / map_stream); the context compiler's streaming candidate pre-filter (ContextCompilerOptions.max_candidates) bounds a 10k+ evidence pool before full scoring; semantic layer (vincio/data/semantic): a SemanticLayer (app.semantic_layer) of Measure / Dimension / DerivedColumn defined once over one registered table so a question maps to a governed metric compiled to one canonical read-only SELECT and run through the existing query plane (app.query_metric / query_metric), computed one way everywhere, cell-cited, with MetricResult.verify proving the SQL is the layer's canonical compilation (an ad-hoc query rejected); column-level MetricLineage (app.metric_lineage) resolves a metric's base columns + source through the derived-column graph and ratios, and register_dataset(..., source=) + erase_source carry a metric's provenance and a subject's right-to-erasure into the dataset plane (ErasureProof records the removed dataset); SemanticLayerError (code SEMANTIC_LAYER_ERROR); the data & analytics capstone (vincio/data/engagement): a purely-compositional DataEngagement (app.data_engagement) threads the whole plane (register / profile / sample / fit / screen / query / analyze / chart / query_metric / cite, each delegating to the same app.* primitive) into a content-bound, signed, hash-chained DataNarrative (DataStage links) that verifies offline from the bytes alone and is data-bound (eng.verify(catalog=) re-executes every captured query/analysis/chart/metric against the content-hashed source so a tampered source is caught even when the chain is intact), the analytics analogue of CrossOrgEngagement, sealed onto the audit log (action data_engagement); held by the data_analysis_conformance VincioBench family (end-to-end, data-bound, tamper-evident SLOs)
vincio/interop        LangChain / LlamaIndex / Haystack / DSPy bridges (tools / retrievers / loaders / embedders / components / compiled modules, both directions)
vincio/plugins        the versioned entry-point plugin contract, discover / load third-party providers, embedders, stores, connectors, chunkers, rerankers, judges, metrics, and packs on install (installed_plugins / load_plugins; vincio plugins list)
vincio/mcp            MCP client + server over stdio / Streamable HTTP / in-process; tools → permissioned runtime, resources → evidence, prompts → PromptSpec, sampling → provider, elicitation → human gate (app.add_mcp_server / app.serve_mcp); marketplace bridge (app.add_mcp_from_registry) discovers → governs → lands tools in one call
vincio/a2a            Agent-to-Agent: Agent Card + JSON-RPC task lifecycle, crew / graph exposure, RemoteA2AAgent as a bounded crew delegate (app.serve_a2a)
vincio/negotiation    bounded, terminating offer/counter-offer bargain (Negotiation / NegotiationPosition / LocalParty) minting a typed, signed, offline-verifiable Contract (price / SLA / scope / quality, enforced like a budget) over the A2A fabric, reputation-weighted (app.negotiate / serve_negotiation / A2ANegotiator)
vincio/choreography   durable, compensating cross-org saga over A2A + the negotiated Contract (Saga / Choreography / Participant); coordinator-driven dispatch with per-org self-governance (each side audits its own steps), a hash-chained SagaJournal checkpointed to the store for restart-survival + offline verification, deterministic reverse-order compensation on a failed or contract-breaching step (app.choreograph / resume_choreography / serve_choreography / RemoteParticipant)
vincio/settlement     metered, auditable settlement over the negotiated Contract (Meter / MeterReading / UsageEvent, total-preserving usage accrual; SettlementRecord, signed, offline-verifiable reconciliation of delivery vs price/SLA/quality; reconcile, two orgs' records tie out; SettlementBook, durable, hash-chained ledger that verifies offline + reports per counterparty); a settled overrun/shortfall closes the reputation loop (app.meter / settle / settle_saga / use_settlement_book / settlement_report). Multilateral netting: net_settlements / net_books fold a fleet's bilateral books into a content-bound NettingSet, each org's NetPositions cleared to the minimal set of NetObligation transfers (<= N-1), dedup-not-double-counted, verifies offline (positions balance + transfers reproduce them), a tampered source refused and a disagreement pinpointed as a NettingDispute (app.clear_settlements / book.net). Dispute resolution: arbitrate / Resolution / ClaimVerdict adjudicate a pinpointed disagreement over the parties' submitted signed records, a reconciliation hash both co-signed is upheld, a contradicting unilateral claim rejected and pinpointed, a tampered claim marked inadmissible (not raised), a genuine standoff left unresolved; the Resolution verifies offline (hash recomputes + decision re-derives from the recorded claims) and closes the reputation loop on the dissenter (app.arbitrate / book.arbitrate). Reputation portability: ReputationAttestation / attest_reputation / book.attest issue a signed, offline-verifiable attestation over a counterparty's earned standing (from the org's own SettlementBook records + arbitration Resolutions), verify recomputes the hash + re-derives the reputation from the evidence counts (a tampered score caught even after re-seal, a forged issuer refused); combine_attestations / app.import_reputation pool several issuers' attestations into a bounded, evidence-weighted PortableReputation prior (a self-attestation refused, a tampered one pinpointed, an issuer cannot stack its pull) exposing weight(member_id) for the existing negotiation path under the same [floor, 1] rule (app.attest_reputation / app.import_reputation). Freshness & revocation: an attestation carries an issuer horizon_days validity window; AttestationRevocation / revoke_attestation / book.revoke / app.revoke_attestation withdraw or supersede one by its hash, and combine_attestations(..., revocations=, as_of=) excludes a revoked or stale one (pinpointed .revoked/.stale) and decays an older one by an importer half_life_days, a forged or cross-issuer revocation cannot cancel a claim. Reputation gossip (exchange.py): attestation_a2a_server / app.serve_attestations expose an org's book as a queryable, pull-only A2A peer returning a ReputationBundle of its own signed artifacts; AttestationExchange / gather_reputation / app.gather_reputation (+ agather_reputation) pull a bounded (max_peers), AgentDirectory-governed set of peers, verify each fetched artifact from the bytes, dedup by content hash, and fold them into the same combine_attestations (a PeerVisit per peer, a GatheredReputation result exposing weight/standing), a denied peer skipped, a forged artifact refused, a gossiped revocation excluding the withdrawn claim, every peer (reputation_peer) and artifact (reputation_fetch) audited; in-process via connect_a2a_in_process, identical over the live fabric. Transitive trust & Sybil resistance (build_trust_model / TrustConfig / TrustModel / IssuerTrust): opt-in issuer-trust weighting (combine_attestations(..., trust=/trust_config=) / app.import_reputation / app.gather_reputation) scales each issuer's contributed evidence mass by the importer's own trust in it, a bounded, transitive web-of-trust rooted in the local ReputationLedger (hop 0 first-hand, hops 1..max_depth a trusted issuer vouches for the issuers it attests with per-hop decay, an unknown one floored never zeroed); lent only outward from a trusted root so a mutually-vouching Sybil ring stays at the floor (pull follows earned trust, not issuer count); bounded [trust_floor,1], pinpointed (AttestationVerdict.trust / SubjectStanding.issuer_trust), reversible, strictly opt-in (no trust source = unchanged equal-pull pooling)
vincio/registry       governed agent fabric (AgentDirectory over A2A / ACP / MCP-registry, allow-list-gated + audited) and the signed community pack & skill registry (CommunityRegistry, content-bound, signed, allow-list-gated, audited bundle resolution)
vincio/skills         Agent Skills: SKILL.md loader with progressive disclosure into the compiler, bundled scripts as sandboxed tools (app.add_skill)
vincio/packs          opt-in packs (app.use_pack): domain packs (support / engineering / finance / legal, prompt + schema + policies + evaluators + golden evals) and full-stack vertical packs (healthcare / ediscovery / kyc / customer_support / code_review, also preconfigure retrieval / scoped memory / rails / metrics / residency via the Pack contract's retrieval / memory / residency / purpose fields)
vincio/assistant      Assistant, a conversational, session-aware layer over ContextApp (app.assistant): session threading, multi-turn state via session-scoped memory write-back, and a tool-approval surface (write tools denied until approved / auto_approve / on_approval); drives as a Simulator target
vincio/memory         memory engine (L0–L5), write policy, decay, conflict resolution, graph, summarizers, grounded-fact auto-memory; bi-temporal items with ACL / purpose / consent fields, team scope, history-preserving correction, and as-of / reader / consent-filtered recall
vincio/tools          tool registry, permissioned runtime, sandbox
vincio/agents         bounded DAG executor, planners (direct / static / dynamic / ReAct / plan-and-execute / hierarchical HTN), in-place plan repair (re-bind / substitute / reorder / drop), cost-aware action selection over the ModelRegistry, ReAct, handoffs, crews + blackboard, durable state graphs (checkpoint / resume / fork) with durable timers (sleep_until / wait_for_event), compose / pipe; distributed execution with TTL-lease + checkpoint-version CAS for exactly-once super-steps, BSP parallel super-steps + Send map-reduce, a work-stealing sub-graph scheduler (fair-share budget + SLA deadlines), and LangGraph / OpenAI Agents SDK / Ray / Temporal export adapters
vincio/workflows      deterministic DAG workflows (retries / compensation / approval gates with pause + resume)
vincio/output         schemas, robust parsers, validation pipeline, principled repair, constrained decoding, streaming validation, self-correction loops, multi-schema routing
vincio/generation     documents & media flowing OUT, DocumentModel IR, DocumentContract + validation, markdown / html / docx / pdf / pptx render, cited reports ([E1] → footnotes with per-claim entailment), template / form fill, image and speech providers with cost metering and provenance (app.build_document / cited_report / generate_image / synthesize_speech)
vincio/evals          datasets (+ synthetic, + from-traces, + multi-turn), metrics (task / grounding / quality / conversational / trajectory & tool-use), judges (+ G-Eval + κ calibration) and κ-gated judge ensembles with disagreement detection (ensemble.py), runner, gates, reports, A/B experiments with significance, red-teaming; trace replay, online evaluation, drift monitoring, annotation queues, the SwapGate model-swap regression contract, Shapley causal regression attribution (attribution.py), adaptive eval sampling (adaptive.py), and nine agentic benchmark adapters behind one contract (benchmarks.py)
vincio/optimize       fitness, evolution loop, prompt / context / routing / cache optimization, the improvement loop (trace → dataset → eval → optimize → promote), Pareto frontier, learned budgets; reflective (GEPA / MIPRO) optimizers, the distillation flywheel, model cascade + capability-aware Router, and the unified self-improvement controller with canary-gated deploy (app.self_improvement, app.deploy, offline dataset or live-traffic canary with auto-rollback); on-policy reinforcement from verifiable rewards (rewards.py: RewardModel / OracleReward / BenchmarkReward / JudgeEnsembleReward; trajectory_opt.py: TrajectoryAdvantage step-credit, GRPO TrajectoryOptimizer with KL clamp + no-regression gate, app.learn)
vincio/observability  traces / spans (sessions, feedback, scores), JSONL / OTel (GenAI semconv) exporters, viewer (TUI / HTML / diff), cost tracking; FinOps cost ledger + budget SLOs, an indexed trace / cost store with rollups, a served dashboard (serve_viewer), an alert rule engine (threshold / EWMA / burn-rate over webhook / Slack / PagerDuty / Prometheus), and an off-by-default content-capture gate at the export boundary
vincio/testing        assert_eval / assert_grounded / assert_metric / assert_safe, packet / trace snapshots, the pytest plugin, and assert_backend_conformance, the offline contract a runtime backend must satisfy vs the native durable engine
vincio/security       PII / secrets, injection defense (normalization + recursive decode pre-pass, pluggable detector backends), RBAC / ABAC, the policy engine, programmable rails, RAG-poisoning detection, fail-closed tenant isolation, always-on egress DLP, and a hash-chained audit log with per-entry signatures + Merkle checkpoints
vincio/governance     compliance evidence over the live system, model / system cards, framework mapping (OWASP LLM / Agentic / NIST AI RMF / MITRE ATLAS / ISO 42001), AI-BOM, C2PA provenance marking, source → output lineage with signed erasure proofs, residency-aware egress, EU AI Act risk tiering / Annex IV / FRIA, and the consent ledger
vincio/caching        LRU / SQLite backends; response / retrieval / packet / semantic / compile / chunk caches, with invalidation
vincio/storage        metadata stores (memory / sqlite / postgres) and vector adapters (qdrant / pgvector / chroma / pinecone / lancedb / weaviate / milvus / elasticsearch / opensearch / vespa) plus neo4j / redis / duckdb; the async store contract is canonical (asave / aquery), with shared-state rate-limit / idempotency primitives (in-memory + Redis)
vincio/providers      openai / anthropic / google / mistral / local + OpenAI-compatible passthrough & presets; unified reasoning control, pooled httpx with coalescing, the deterministic mock, batch backends (~50% cost), circuit breaker + health-aware failover, key pool, and prompt-cache strategy; a data-driven ModelRegistry (capabilities + pricing + lifecycle) drives capability guards, shadow / canary dispatch, lifecycle migration, fine-tune backends, and enterprise auth (Bedrock / Vertex / Azure OpenAI)
vincio/notebook       rich Jupyter reprs (enable_rich_reprs) for RunResult / Trace / EvalReport / MemoryItem / SearchHit
vincio/tui            interactive terminal inspector for runs / traces / memory; pure renderers + injectable IO
vincio/server         FastAPI app (API key + JWT auth, real-token SSE streaming), health / readiness / Prometheus metrics, graceful shutdown, optional Redis-coherent rate-limit middleware
vincio/realtime       (optional) voice / realtime, RealtimeSession, connect_realtime (in-process / OpenAI / Gemini), VAD, interruption, in-session tool calls through the permissioned runtime (app.realtime_session; extra vincio[realtime]); and the end-to-end VoiceAgent (app.voice_agent) wiring the session to the deep-research agent (in-session research tool), the memory OS, and the app's input/output rails
vincio/cli            argparse CLI: init, config, packs, plugins, tui, run, eval, prompt, trace, optimize, loop, distill, index, memory, audit, governance, mcp, and serve (uvicorn HTTP launcher)
vincio/stability      the API-stability contract, @deprecated / @experimental, deprecated_alias, stability_of, public_api, API_VERSION, and the Vincio deprecation / experimental warnings
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

- **Offline-first tests**: everything must pass with no network or API keys; use
  `MockProvider` (it generates schema-valid structured output).
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
- **Performance is gated**: `python benchmarks/vinciobench.py` +
  `python benchmarks/check_budgets.py` must pass; budgets live in
  `benchmarks/budgets.json` and run in CI. Published SLOs (`benchmarks/slos.json`,
  `docs/reference/slo.md`) are each held by a budget at least as strict;
  `tests/test_slos.py` enforces that invariant.
- **Public API is frozen under SemVer**: the public surface is `vincio.__all__`
  plus the documented subsystem entry points. Don't remove or break a public
  symbol in a minor / patch; mark it with `@deprecated(since=, removed_in=,
  alternative=)` and remove only at the next major. New, unproven API goes behind
  `@experimental`. See `docs/reference/stability.md`.
- **Docs stay complete**: `tests/test_docs_completeness.py` requires every public
  subsystem to be documented and every example indexed; `tests/test_examples.py`
  runs all examples offline. Add a doc + example when you add a subsystem.
- **Update `ROADMAP.md`** when adding subsystems or changing release status.
