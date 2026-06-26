# Vincio documentation

Vincio is a Python platform for context-engineered AI applications. It compiles
prompts, memory, retrieval, tools, schemas, and policies into optimized,
validated, observable **context packets**, then validates and evaluates every
output. The single entry point is `from vincio import ContextApp`.

This page is the map. It lists every guide, concept, and reference page in a
reading order, from your first app through the full platform. New to Vincio?
Read [Getting started](getting-started.md), then the [core concepts](#core-concepts)
in order. Building something specific? Jump straight to the matching
[task guide](#build-an-application). For the product pitch and benchmarks, see
the [root README](../README.md); for the source-tree map, see
[`AGENTS.md`](../AGENTS.md).

## Contents

- [Start here](#start-here)
- [Core concepts](#core-concepts)
- [Build an application](#build-an-application)
- [Evaluate and improve](#evaluate-and-improve)
- [Orchestrate and interoperate](#orchestrate-and-interoperate)
- [The cross-organization agent economy](#the-cross-organization-agent-economy)
- [Govern, secure, and assure](#govern-secure-and-assure)
- [Advanced runtimes](#advanced-runtimes)
- [Reference](#reference)
- [Migrating from another library](#migrating-from-another-library)
- [How Vincio compares](#how-vincio-compares)

## Start here

- **[Getting started](getting-started.md)**: install, scaffold a project, write
  your first app, run it offline with the deterministic mock provider, and run a
  first eval.
- **[Cookbook](guides/cookbook.md)**: short, runnable, tested recipes for common
  tasks, each one a small end-to-end example you can copy.

## Core concepts

The model of how Vincio works. Read these in order to understand what happens
between your input and the model's validated output.

- **[Context packets and the context compiler](concepts/context-packets.md)**:
  the central unit. How candidate evidence is scored, deduplicated, budgeted,
  and packed into a provider-neutral packet.
- **[Tabular evidence and the compact data encoder](concepts/tabular-evidence.md)**:
  a typed, columnar `Dataset`, a lossless `DataEncoder` that renders it
  header-once, and `TableEvidence` that scores and budgets a table token-cheap.
- **[Dataset profiling, sampling, and quality rails](concepts/dataset-profiling.md)**:
  `profile_dataset` for a fixed-size column profile, reservoir/stratified
  sampling, `fit_to_window` to fit a table far larger than the window under a
  fixed token budget, and `DataQualityRails` for deterministic screening.
- **[Prompt compiler](concepts/prompt-compiler.md)**: how a typed `PromptSpec`
  becomes a cache-aware, lint-checked prompt with a stable prefix and a volatile
  suffix.
- **[Retrieval](concepts/retrieval.md)**: BM25, dense, sparse, late-interaction,
  and graph indexes behind one interface, hybrid fusion, query understanding,
  reranking, and GraphRAG.
- **[Memory](concepts/memory.md)**: the layered, scoped, scored, decaying memory
  engine, with provenance, consent, and history-preserving correction.
- **[Agents and workflows](concepts/agents.md)**: bounded DAG agents, crews,
  durable state graphs, planners, in-place plan repair, and deterministic
  workflows.
- **[Evaluation](concepts/evals.md)**: datasets, metrics, judges, the runner and
  gates, online and drift evaluation, and the rule that every metric is also a
  guardrail and an optimizer term.
- **[Observability](concepts/observability.md)**: one trace, one cost, and one
  hash-chained audit entry per run, with OpenTelemetry export and an offline
  viewer.

## Build an application

Task-oriented guides, roughly in the order you would reach for them while
building.

- **[Build a RAG app](guides/build-rag-app.md)**: a grounded document-QA app in
  under 30 lines.
- **[Add tools](guides/add-tools.md)**: register functions as permissioned,
  idempotent tools with approval gates.
- **[Structured output](guides/structured-output.md)**: define a Pydantic
  contract, validate, repair structure (never facts), and route multiple
  schemas per run.
- **[Reliability and guardrails](guides/reliability-guardrails.md)**:
  deterministic rails before and after the model, injection defense, PII
  redaction, and metric-backed guardrails.
- **[Optimize prompts, context, and routing](guides/optimize-context.md)**: turn
  eval results into better configurations through gated optimization.
- **[Cost, reliability, and scale](guides/cost-and-reliability.md)**: the
  in-process FinOps and resilience layer, half-cost batch, caching, circuit
  breakers, and budget SLOs.
- **[Performance and streaming](guides/performance.md)**: concurrent hot paths,
  content-addressed caches, slim packets, partial-JSON streaming, and the server
  SSE endpoint.
- **[Connect external data sources](guides/connectors.md)**: feed the document
  engine from web, GitHub, SQL, object stores, and SaaS systems with provenance.
- **[Generate documents and media](guides/generate-documents.md)**: documents
  and media flowing out, cited reports, DOCX/PDF/PPTX render, images, and speech.
- **[Build a chat product: the Assistant](guides/assistant.md)**: a
  conversational, session-aware layer with multi-turn state and tool approval.
- **[Voice and realtime](guides/realtime.md)**: the optional realtime module,
  sessions, voice activity detection, interruption, and in-session tools.
- **[Plugins](guides/plugins.md)**: extend Vincio from a separate package through
  the versioned entry-point contract.
- **[Vertical packs](guides/vertical-packs.md)**: apply a regulated domain
  (healthcare, e-discovery, KYC, support, code review) in one line.
- **[Integrations](guides/integrations.md)**: providers, vector stores, and
  frameworks behind interfaces that already exist.

## Evaluate and improve

- **[Run evals](guides/run-evals.md)**: build a golden dataset, run metrics, and
  gate CI on the results.
- **[Test LLM apps with pytest](guides/test-llm-apps.md)**: the pytest plugin and
  the `assert_eval` / `assert_grounded` / `assert_metric` / `assert_safe`
  assertions.
- **[Agentic evaluation and continuous quality](guides/agentic-eval.md)**: score
  trajectories and tool use, run online evaluation, and watch for drift.
- **[Close the loop](guides/close-the-loop.md)**: one continuous improvement loop
  from production traces to a gated, promoted configuration.

## Orchestrate and interoperate

Multi-agent execution and the protocols that connect agents across processes and
vendors.

- **[Orchestrate multi-agent systems](guides/orchestrate-agents.md)**: the same
  support-triage system built three ways, as a crew, a durable graph, and a
  workflow.
- **[The governed agent fabric](guides/agent-fabric.md)**: registry, discovery,
  and allow-list-gated delegation across a fleet of agents.
- **[Model Context Protocol (MCP)](guides/mcp.md)**: consume and serve tools,
  resources, prompts, sampling, and elicitation over MCP.
- **[Agent-to-Agent (A2A)](guides/a2a.md)**: the cross-vendor agent
  interoperability protocol, Agent Cards, and the JSON-RPC task lifecycle.
- **[Agent Skills](guides/agent-skills.md)**: `SKILL.md` packages with
  progressive disclosure and bundled scripts as sandboxed tools.
- **[Agent identity, delegation, and accountability](guides/agent-identity.md)**:
  DID-based identity, attenuating delegation chains, and signed, verifiable
  artifacts.
- **[Reasoning control](guides/reasoning.md)**: one portable knob for thinking
  and reasoning effort across providers, with honest cost accounting.

## The cross-organization agent economy

A layered stack for agents that transact across organizational boundaries. Read
in order; each layer builds on the one before.

- **[Negotiation and contracting](guides/negotiation.md)**: a bounded
  offer/counter-offer that mints a typed, signed, offline-verifiable `Contract`.
- **[Cross-org workflow choreography](guides/choreography.md)**: durable,
  compensating sagas across organizations over A2A and the negotiated contract.
- **[Settlement and metering](guides/settlement.md)**: metered, auditable
  settlement of delivered work, multilateral netting, dispute arbitration, and
  portable reputation.

## Govern, secure, and assure

Compliance evidence, formal guarantees, and safety, all produced from the live
system rather than asserted.

- **[Enterprise governance and compliance](guides/governance.md)**: model and
  system cards, framework mapping, AI-BOM, lineage, residency, and EU AI Act
  artifacts.
- **[Formal verification of governance invariants](guides/governance-verification.md)**:
  bounded model checking of the invariants the runtime enforces.
- **[Differential-privacy memory and training](guides/differential-privacy.md)**:
  per-subject privacy accounting and bounded per-member influence in federated
  rounds.
- **[Verified reasoning and certificates](guides/verified-reasoning.md)**:
  proof-carrying answers checked by deterministic kernels, with refuse-or-repair
  self-correction.
- **[Continuous assurance and certification](guides/assurance.md)**: an assurance
  argument tree bound to the platform's own verdicts, with freshness horizons and
  a certification report.
- **[Threat model](security/threat-model.md)**: the trust boundaries, assets, and
  mitigations for a library you run on your own infrastructure.

## Advanced runtimes

Capabilities that extend Vincio beyond the default server path.

- **[Edge / WASM in-process runtime](guides/edge.md)**: the dependency-free
  context-engineering core packaged for constrained and browser targets.
- **[Computer-use action plane](guides/computer-use.md)**: a grounded
  perceive, gate, act, verify, undo loop over a pluggable screen backend.
- **[Native video understanding and generation](guides/video.md)**: video as a
  first-class evidence modality, with temporal grounding and C2PA-marked output.
- **[Autonomous skill acquisition](guides/skill-acquisition.md)**: an open-ended
  curriculum that proposes, attempts, verifies, distills, and promotes new
  skills under the no-regression gate.

## Reference

- **[API](reference/api.md)**: the curated, grouped narrative of `ContextApp` and
  every subsystem entry point.
- **[Public API index](reference/api-generated.md)**: the exhaustive,
  docstring-driven index of every name in `vincio.__all__` (generated).
- **[CLI](reference/cli.md)**: every `vincio` command, from `init` to `serve`.
- **[Configuration](reference/config.md)**: the layered configuration model and
  every `vincio.yaml` / `VINCIO_*` setting.
- **[Error catalog](reference/errors.md)**: every error code, its meaning, and
  its remediation (generated).
- **[Performance and quality SLOs](reference/slo.md)**: the published Service
  Level Objectives, each held by a CI budget at least as strict.
- **[API stability and deprecation policy](reference/stability.md)**: the
  SemVer contract, the deprecation lifecycle, and the `@experimental` marker.
- **[Typing](reference/typing.md)**: the inline, PEP 561 type contract Vincio
  ships for downstream type-checkers.
- **[Frozen public surface](reference/public-surface.txt)**: the exact set of
  names SemVer is applied against.

## Migrating from another library

Concept-by-concept maps from a library you already use.

- **[From LangChain / LangGraph](guides/migrate-from-langchain.md)**
- **[From LlamaIndex](guides/migrate-from-llamaindex.md)**
- **[From Ragas](guides/migrate-from-ragas.md)**
- **[From Mem0](guides/migrate-from-mem0.md)**

## How Vincio compares

Honest, side-by-side comparisons that name what each tool does well and where
Vincio differs.

- **Frameworks and agents**: [LangChain / LangGraph](comparisons/langchain.md),
  [LlamaIndex](comparisons/llamaindex.md), [DSPy](comparisons/dspy.md),
  [CrewAI](comparisons/crewai.md),
  [OpenAI Agents SDK](comparisons/openai-agents-sdk.md),
  [Pydantic AI](comparisons/pydantic-ai.md)
- **Retrieval**: [RAGatouille / ColBERT](comparisons/ragatouille.md)
- **Memory**: [Mem0](comparisons/mem0.md)
- **Evaluation**: [Ragas](comparisons/ragas.md),
  [DeepEval](comparisons/deepeval.md)
- **Guardrails**: [Guardrails AI](comparisons/guardrails.md),
  [NeMo Guardrails](comparisons/nemo-guardrails.md)
- **Observability**: [LangSmith / Langfuse](comparisons/langsmith-langfuse.md)
- **Gateways**: [LiteLLM, Bifrost, Portkey](comparisons/litellm.md)
