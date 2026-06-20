<p align="center">
  <img src="assets/logo.svg" alt="Vincio" width="96">
</p>

# Vincio Roadmap

This is the public roadmap for the Vincio library — package `vincio`, CLI `vincio`, configuration
`vincio.yaml`, benchmark suite **VincioBench**. It records what ships today, what is planned next,
and what is intentionally out of scope. The complete release-by-release history lives in the
[CHANGELOG](CHANGELOG.md).

**Legend:** ✅ shipped · 🚧 planned (next) · 🔭 exploring (later)

## What "done" means here

Vincio is a single, coherent context-engineering library: every subsystem is implemented, tested
offline, documented, and demonstrated by a runnable example. The platform is production-stable —
the public surface (`vincio.__all__`) is frozen under [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with a mechanical [deprecation policy](docs/reference/stability.md), performance and quality targets
are [published as SLOs](docs/reference/slo.md) and held by at-least-as-strict VincioBench budgets, and
every release ships a CycloneDX SBOM with SLSA build-provenance attestations.

Forward work **deepens and broadens** the platform without changing that contract. Each new capability
sits behind a new entry point or an opt-in extra; the dependency-free, offline-first path is always
the default. Vincio adopts the ecosystem's standards — MCP, A2A, Agent Skills, AGNTCY/ACP, the OTel
GenAI conventions, C2PA, and the OWASP LLM / OWASP Agentic / NIST AI RMF / MITRE ATLAS frameworks —
**in your process**. It never becomes a hosted service to do so.

---

## ✅ What ships today

The platform is complete and stable across these subsystems. Each is covered by a VincioBench family
and a runnable example.

| Subsystem | Capability |
|---|---|
| **Prompt compiler** | Typed prompt ASTs, lint rules, cache-aware stable-prefix layout, versioning, hashing, diffing, variant generation. |
| **Context compiler** | Multi-signal candidate scoring, dedup, conflict resolution, compression, token-budget packing, and an excluded-context report; image / table / text evidence in one scored, multimodal-native packet with cross-process `materialize()` from a content-addressed store. |
| **Retrieval (RAG)** | BM25 + dense + learned-sparse + late-interaction fused in one RRF; query understanding (HyDE / multi-query / decompose / step-back); sentence-window / parent-document / auto-merging / contextual chunking; GraphRAG; live indexes; entity-graph, multi-hop, and reasoning retrieval; Matryoshka, contextual, and multimodal embedders; a structured `FilterSpec` pushed down to each backend with tenant scope enforced in the engine. |
| **Memory** | Layered, guarded, decaying, conflict-resolving, privacy-scoped memory; hybrid vector+graph recall; consolidation with provenance; audited GDPR hygiene; a CI-gated memory eval harness; bi-temporal records with as-of recall, `correct()`, per-memory ACLs, and a `TEAM` scope. |
| **Tools** | Permissioned registry (RBAC/ABAC), schema derivation, a resource-limited sandbox, reliability scoring, approval-gated writes; computer-use and provider-native hosted tools behind a pluggable `IsolationBackend`. |
| **Agents & orchestration** | Bounded DAG agents with planners (direct / static / dynamic / ReAct / plan-and-execute / hierarchical HTN), critics, validators, and human gates; in-place plan repair (re-bind / substitute / reorder / drop on a tool failure, contradiction, or budget shock) and cost-aware action selection over `ModelRegistry` pricing and the live budget; a deep-research agent and a self-editing memory OS; multi-agent crews with a shared blackboard; durable graphs with checkpoint / resume / time-travel and durable timers (`sleep_until` / `wait_for_event`); distributed durable execution across a worker pool with lease + CAS, BSP super-steps, `Send` map-reduce, and a work-stealing sub-graph scheduler under a fair-share budget with SLA deadlines. |
| **Workflows** | Deterministic DAGs with retries, branching, parallelism, compensation, and resumable approval gates. |
| **Structured output** | Pydantic contracts, provider-native constrained decoding, streaming validation with early abort, typed signatures, bounded self-correction, multi-schema routing, and structure-only repair. |
| **Evaluation** | Golden datasets, 30+ metrics, judges with calibration, synthetic data, red-teaming, experiments with significance, regression gates, a pytest plugin, a stateful-environment harness with a task-success oracle, five agentic benchmark adapters, and retrieval-eval with index-version regression. |
| **Optimization & self-improvement** | The closed loop (trace → dataset → eval → optimize → promote) with safety-gated promotion; reflective (GEPA-style) optimization and MIPRO; a distillation flywheel with executed fine-tune jobs; learned prompt compression; one declarative `SelfImprovementPolicy` driving a streaming controller; and canary-gated `app.deploy`. |
| **Observability** | Full trace span trees, sessions, feedback, eval scores on spans, JSONL + OpenTelemetry export; a local viewer; a served, self-hosted observability + alerting plane; a versioned prompt registry; per-run cost. |
| **Security & governance** | Deterministic PII / secret / injection / RAG-poisoning detection, programmable rails, RBAC/ABAC, tenant isolation, a signed Merkle-checkpointed audit chain; model & system cards, a compliance coverage matrix, an AI-BOM, an EU AI Act conformity pack, provable erasure, a consent ledger, data lineage, and residency-aware egress refusal. |
| **Generation** | Cited DOCX/PDF/PPTX/HTML/Markdown, a cited-report builder with per-claim entailment, redlines, image generation and TTS with C2PA provenance, and richer inputs (OCR, transcripts, new-format loaders, forms/KYC). |
| **Providers & storage** | OpenAI, Anthropic, Google, Mistral, any OpenAI-compatible endpoint, enterprise endpoints behind an `AuthStrategy`, a deterministic mock, and local neural models; a data-driven `ModelRegistry`; pluggable metadata / blob / analytics / vector / graph backends with Redis shared state. |
| **Protocols & interoperability** | MCP client + server, A2A, Agent Skills, a governed agent fabric over an `AllowListGate`, AG-UI generative-UI streaming, and LangChain / LlamaIndex interop. |
| **Cost, reliability & rotation** | Batch execution, circuit breaking, health-aware failover, key pooling, model cascades, cost attribution with budget SLOs, prompt caching, incremental + sharded indexing, a capability-aware router, a swap gate, and a lifecycle watcher. |
| **Runtime performance** | A single-pass vectorized scorer (NumPy-optional, pure-Python fallback); a compiled-prompt render program and a warm candidate arena that reuse the stable prefix and the prepared candidate set so a warm compile is dominated by scoring, not allocation; streaming-first compilation that emits the prefix before scoring; speculative retrieval prefetch that warms the query embedding from the task classification; and a per-app resident-memory budget held by slim packets and evidence eviction, surfaced in the cost report and gated by an SLO. |

VincioBench holds these guarantees under CI-gated budgets and SLOs; the full test suite runs offline.

---

## 🚧 Where this goes next

Forward phases are scoped by theme and gated the same way everything else is — covered offline, held
by VincioBench budgets and SLOs, and demonstrated by a runnable example. Each is additive on the
frozen public surface; breaking changes are reserved for an announced major window and never shipped
for their own sake.

### 🚧 Ecosystem & integration breadth

*Meet teams where their data and tools already live.*

- **Connector breadth** — first-party connectors for Jira, Linear, Google Drive, SharePoint,
  Salesforce, Zendesk, BigQuery, and Snowflake, all feeding the document engine with full provenance
  behind the existing `register_connector` contract.
- **Plugin discovery** — an entry-point plugin system so third-party providers, metrics, chunkers,
  rerankers, judges, connectors, and packs register themselves on install, discoverable via
  `vincio plugins list` and gated by a stable plugin-API contract.
- **A community pack & skill registry** — a governed, signed index of opt-in domain packs and
  `SKILL.md` bundles, resolvable through the existing `AgentDirectory` allow-list gate.
- **Deeper framework interop** — Haystack and DSPy module interop alongside LangChain / LlamaIndex,
  and an MCP-server marketplace bridge so a discovered server's tools land in the permissioned runtime
  with one call.
- *Definition of done:* each connector and interop bridge round-trips offline against a recorded
  fixture in the `integrations` family, the plugin contract is versioned and documented, and the
  registry resolution is an audited access decision.

### 🚧 Use-case coverage & verticals

*Go from primitives to a working app in one file, in more domains.*

- **Vertical packs** — healthcare/PHI, legal e-discovery, financial KYC/AML, customer support, and
  code review packs that preconfigure retrieval, memory, rails, metrics, residency, and a golden set
  for the domain, on top of the existing pack contract.
- **A higher-level Assistant abstraction** — a conversational, session-aware layer over `ContextApp`
  that manages multi-turn state, tool approvals, and memory write-back, so a chat product is a few
  lines rather than a hand-wired loop.
- **End-to-end voice agents** — the realtime session wired to the deep-research agent, memory OS, and
  rails so a spoken assistant inherits the full grounding, budget, and audit guarantees.
- **A recipe cookbook** — a curated set of task-shaped recipes (multimodal RAG over slides/PDFs,
  contract redlining, incident triage, data-room Q&A) as runnable, tested examples.
- *Definition of done:* each vertical pack ships with a golden eval set and a runnable example, the
  Assistant layer is covered by a multi-turn simulator suite, and the cookbook recipes are part of the
  offline example gate.

### 🚧 Professionalism & API ergonomics

*Make the platform's surface as trustworthy as its internals.*

- **Auto-generated API reference** — a docstring-driven reference site built in CI from the public
  surface, with a docstring-coverage gate so no public symbol ships undocumented.
- **Strict typing end to end** — graduate the package to `mypy --strict` with shipped type stubs, so
  downstream type-checkers see the full contract.
- **Config schema migrations** — versioned, automatic `vincio.yaml` migrations with a
  `vincio config migrate` command, so a config never silently drifts from the schema.
- **Deprecation tooling** — `vincio doctor` reports any deprecated API a project still uses, its
  replacement, and its removal version, driven by the same `stability_of` metadata the library uses.
- **Internationalized, actionable errors** — every `VincioError` carries a stable code, a remediation
  hint, and a docs link, with a catalog gated for completeness.
- *Definition of done:* the docs-completeness gate extends to docstring coverage and error-catalog
  completeness, the strict-typing gate is green in CI, and the migration/doctor commands are covered
  by the CLI test suite.

### 🚧 Evaluation & quality frontier

*Measure more of what buyers compare on, and explain regressions instead of just flagging them.*

- **More benchmark adapters** — AgentBench, ToolBench, LiveCodeBench, and MMLU-Pro behind the same
  `BenchmarkAdapter` contract, pinned by a task-set hash and scored by each benchmark's own scorer.
- **Judge ensembles with disagreement detection** — a panel of judges whose disagreement is surfaced
  as an uncertainty signal, with the ensemble itself calibrated against κ-validated labels.
- **Causal regression attribution** — when a gate fails, attribute the regression to the changed
  component (prompt / retrieval / model / budget) by counterfactual replay, not just report the score
  drop.
- **Adaptive eval sampling** — spend the eval budget where the variance is, so a CI gate converges on
  the same verdict for less cost.
- *Definition of done:* the new adapters and attribution cases join the `agentic_evals` and
  `benchmarks` families offline, judge-ensemble calibration is gated, and adaptive sampling is proven
  to preserve the gate verdict under a cost budget.

---

## 🔭 Exploring — later

Candidates that are real but not yet scheduled — pulled forward when demand and the standards settle:

- 🔭 **Federated / cross-org self-improvement** — sharing gated optimizations and learned routing
  across trust boundaries without sharing raw traffic, once privacy-preserving aggregation standards
  settle.
- 🔭 **World-model / simulation-based planning** — agents that learn a tool/environment model and plan
  against it, beyond the reset/step/verify environment-eval harness.
- 🔭 **Native video understanding & generation** — a video `ContentPart` with frame sampling, temporal
  segmentation, and generative output, extending multimodal beyond image and audio.
- 🔭 **On-device fine-tuning / continual local adaptation** — LoRA-class local adaptation of the
  in-process GGUF provider from the same flywheel, beyond executed hosted fine-tune jobs.
- 🔭 **MCP Apps & the evolving MCP spec** — server-rendered UI and stateless-core changes, adopted once
  the spec ships stable, tracked alongside AG-UI streaming.
- 🔭 **Formal verification of governance invariants** — machine-checkable proofs that residency,
  erasure, and budget invariants hold across the whole pipeline, beyond the signed audit chain and
  provable erasure.
- 🔭 **A future breaking window** — reserved, as always, only for changes the frozen surface cannot
  make additively, shipped with the same mechanical deprecation runway and never for its own sake.

---

## Out of scope

Vincio is a library, and stays one. The building blocks for running it in production — a
hash-chained audit log, retention policies, tenant isolation, RBAC / ABAC, and a server — ship in the
package so you can deploy them on your own infrastructure. **Hosted services, managed control planes,
dashboards-as-a-service, and compliance programs are not part of this project.**

Everything that *looks* operational is something you run yourself: the served observability and
alerting plane is self-hosted over your own indexed store, the `vincio serve` launcher is a process
you manage, the distributed backend is a lock-free adapter to your Temporal/Ray, the agent fabric is a
governed directory you operate, and every standard (MCP, A2A, AGNTCY, OWASP/NIST, OTel GenAI, C2PA) is
implemented in-library. Vincio gives you the engine; how and where you run it is yours.
