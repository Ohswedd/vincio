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
| **Evaluation** | Golden datasets, 30+ metrics, judges with calibration, judge ensembles whose disagreement is an uncertainty signal, synthetic data, red-teaming, experiments with significance, regression gates that attribute a failure to its cause (prompt / retrieval / model / budget) by Shapley counterfactual replay, adaptive sampling that converges a gate verdict for less budget, a pytest plugin, a stateful-environment harness with a task-success oracle, nine agentic benchmark adapters (SWE-bench, τ-bench, GAIA, WebArena, BFCL, AgentBench, ToolBench, LiveCodeBench, MMLU-Pro), and retrieval-eval with index-version regression. |
| **Optimization & self-improvement** | The closed loop (trace → dataset → eval → optimize → promote) with safety-gated promotion; reflective (GEPA-style) optimization and MIPRO; a distillation flywheel with executed fine-tune jobs; learned prompt compression; one declarative `SelfImprovementPolicy` driving a streaming controller; canary-gated `app.deploy`; and on-policy reinforcement from verifiable rewards (RLVR) — a `RewardModel` over the task-success oracle, benchmark scorers, and disagreement-down-weighted judge ensembles, step-level Shapley credit, and a GRPO `TrajectoryOptimizer` (`app.learn`) with a KL-to-reference clamp and a monotonic no-regression gate that emits a fine-tune job through the flywheel. |
| **Observability** | Full trace span trees, sessions, feedback, eval scores on spans, JSONL + OpenTelemetry export; a local viewer; a served, self-hosted observability + alerting plane; a versioned prompt registry; per-run cost. |
| **Security & governance** | Deterministic PII / secret / injection / RAG-poisoning detection, programmable rails, RBAC/ABAC, tenant isolation, a signed Merkle-checkpointed audit chain; provable prompt-injection containment that separates the control plane from the data plane — typed `TrustLabel` / `TaintedValue` information-flow labels, unforgeable `CapabilityToken`s minted from the user's request, a `DualPlaneExecutor` whose privileged planner sees only typed extractions of untrusted bytes, and a machine-checked containment invariant (`untrusted ⇒ no unapproved capability`); model & system cards, a compliance coverage matrix, an AI-BOM, an EU AI Act conformity pack, provable erasure, a consent ledger, data lineage, and residency-aware egress refusal. |
| **Generation** | Cited DOCX/PDF/PPTX/HTML/Markdown, a cited-report builder with per-claim entailment, redlines, image generation and TTS with C2PA provenance, and richer inputs (OCR, transcripts, new-format loaders, forms/KYC). |
| **Providers & storage** | OpenAI, Anthropic, Google, Mistral, any OpenAI-compatible endpoint, enterprise endpoints behind an `AuthStrategy`, a deterministic mock, and local neural models; a data-driven `ModelRegistry`; pluggable metadata / blob / analytics / vector / graph backends with Redis shared state. |
| **Protocols & interoperability** | MCP client + server, A2A, Agent Skills, a governed agent fabric over an `AllowListGate`, AG-UI generative-UI streaming, and LangChain / LlamaIndex / Haystack / DSPy interop. |
| **Ecosystem & integration breadth** | First-party connectors for Jira, Linear, Google Drive, SharePoint, Salesforce, Zendesk, BigQuery, and Snowflake feeding the document engine with full provenance behind `register_connector`; an entry-point plugin system (`vincio plugins list`) registering third-party providers, metrics, chunkers, rerankers, judges, connectors, and packs on install under a versioned plugin-API contract; a signed, allow-list-gated, audited `CommunityRegistry` of opt-in packs and `SKILL.md` bundles; and an MCP-server marketplace bridge (`app.add_mcp_from_registry`) that discovers, governs, and lands a server's tools in the permissioned runtime in one call. |
| **Use-case coverage & verticals** | Full-stack vertical packs (healthcare/PHI, legal e-discovery, financial KYC/AML, customer support, code review) that preconfigure retrieval, scoped memory, deterministic rails, domain metrics, a data-residency posture, and a golden eval set on top of the pack contract; a higher-level `Assistant` over `ContextApp` that threads turns into a session, carries multi-turn state via memory write-back, and gates write tools behind an approval; an end-to-end `VoiceAgent` wiring the realtime session to the deep-research agent, the memory OS, and the rails; and a cookbook of task-shaped recipes (contract redlining, incident triage, data-room Q&A, multimodal RAG over slides/PDFs) as offline-gated runnable examples. |
| **Cost, reliability & rotation** | Batch execution, circuit breaking, health-aware failover, key pooling, model cascades, cost attribution with budget SLOs, prompt caching, incremental + sharded indexing, a capability-aware router, a swap gate, and a lifecycle watcher. |
| **Runtime performance** | A single-pass vectorized scorer (NumPy-optional, pure-Python fallback); a compiled-prompt render program and a warm candidate arena that reuse the stable prefix and the prepared candidate set so a warm compile is dominated by scoring, not allocation; streaming-first compilation that emits the prefix before scoring; speculative retrieval prefetch that warms the query embedding from the task classification; and a per-app resident-memory budget held by slim packets and evidence eviction, surfaced in the cost report and gated by an SLO. |
| **Professionalism & API ergonomics** | A docstring-driven, completeness-gated public API reference (`vincio._apiref`); `py.typed` shipped with a graduated, CI-enforced `mypy --strict` ladder; versioned, automatic `vincio.yaml` migrations (`vincio config migrate`, in-memory upgrade on load); a deprecation-aware `vincio doctor` driven by the same `stability_of` metadata; and an internationalizable, completeness-gated error catalog — every `VincioError` carries a stable `.code`, a `.remediation` hint, and a `.docs_url`. |

VincioBench holds these guarantees under CI-gated budgets and SLOs; the full test suite runs offline.

---

## 🚧 Where this goes next

Forward phases are scoped by theme and gated the same way everything else is — covered offline, held
by VincioBench budgets and SLOs, and demonstrated by a runnable example. Each is additive on the
frozen public surface (`API_VERSION` stays `3.0`), sits behind a new entry point or an opt-in extra,
keeps the dependency-free offline path as the default, and ships with a deterministic-mock substitute
for every model or external call so the whole theme is testable offline. Breaking changes are reserved
for an announced major window and never shipped for their own sake.

The most recent scheduled theme — **provable prompt-injection containment & capability-secure
agents** (`TrustLabel` / `TaintedValue` information-flow labels, an unforgeable `CapabilityToken`
minted from the user's request by a `CapabilityBroker`, a `DualPlaneExecutor` whose privileged
planner sees only typed extractions of untrusted bytes, and a machine-checked containment invariant
`untrusted ⇒ no unapproved capability` held at escalation rate **0** on an adversarial corpus) — has
shipped and folded into the **Security & governance** row above. The next two themes are scheduled
below in priority order. Each closes a specific gap in the platform's *own* frontier — a rung that
exists in the literature and in buyer demand but not yet in the package — rather than a gap measured
against any one competitor. Indicative minor-version targets are given; cadence holds one coherent
theme per minor.

### 1 · Test-time compute & reasoning orchestration *(target 3.9)*

Reasoning-model thinking budgets and parallel test-time search are the cheapest quality lever left,
and the platform already owns the pieces to orchestrate them: cost-aware action selection over
`ModelRegistry` pricing, critics and validators that can act as verifiers, and reasoning-effort
control. This theme makes test-time compute a *first-class, budgeted, cache-aware* dimension of the
compile rather than a per-call knob.

- **`ReasoningController`** — sets thinking effort per step from the task classification and the live
  budget (the same signals that drive speculative retrieval prefetch and the capability-aware
  router), with a hard token ceiling held by an SLO so a hard task cannot silently exhaust the run.
- **`TestTimeSearch`** — verifier-guided best-of-N, self-consistency, and beam / MCTS over tool-use
  trajectories, scored by the *existing* critics, validators, and judge ensembles, with early-exit
  the moment the verifier's confidence interval clears the bar (the adaptive-sampling stop rule,
  reused). Bounded by the same fair-share budget and SLA deadlines the orchestrator already enforces.
- **Reasoning-trace-aware caching** — the compiled-prompt render program and warm candidate arena
  extend to cache *reasoning prefixes*, so a re-ask that shares a thinking prefix reuses it under the
  resident-memory budget.

*Ships as:* `vincio.agents` / `vincio.optimize` gain `ReasoningController`, `TestTimeSearch`, and a
verifier protocol; a `test_time_compute` VincioBench family with a quality-per-dollar SLO (Pareto
improvement over single-shot at a fixed budget); runnable example `53_test_time_compute.py`.

### 2 · Long-horizon context engineering *(target 3.10)*

Vincio's namesake is context engineering, and the regime where it matters most is the one the
platform has not yet made first-class: **million-token, multi-day, multi-session agent runs** where
naïve accumulation degrades quality ("context rot") and blows the budget. The context compiler,
memory OS, and content-addressed store already hold the primitives; this theme composes them into an
explicit long-horizon governor.

- **`ContextCompactor`** — hierarchical, provenance-preserving compaction that summarizes cold
  spans of a long run into the memory OS and *pages them back* on demand via the existing
  cross-process `materialize()`, so the live packet stays inside the resident-memory budget without
  losing recall.
- **Intra-run relevance decay** — the memory subsystem's decay model, applied *within a single long
  run* so stale candidates lose weight before they crowd out fresh signal, surfaced in the
  excluded-context report.
- **`ContextGovernor`** — a per-run controller that holds a *context budget* (tokens, residency,
  KV-cache footprint) the way the cost report holds a dollar budget, gated by an SLO that the same
  task at 10× horizon stays within a bounded quality and cost envelope.

*Ships as:* `vincio.context` gains `ContextCompactor`, `ContextGovernor`, and intra-run decay; a
`long_horizon` VincioBench family with a horizon-scaling SLO (quality and cost bounded as run length
grows 10×); runnable example `54_long_horizon_context.py`.

---

## 🔭 Exploring — later

Candidates that are real but not yet scheduled — pulled forward when demand and the standards settle.
Grouped by where they would land.

**Learning & adaptation**

- 🔭 **Federated / cross-org self-improvement** — sharing gated optimizations and learned routing
  across trust boundaries without sharing raw traffic, once privacy-preserving aggregation standards
  settle. Builds on the shipped on-policy learning loop and the existing canary-promoted release.
- 🔭 **On-device fine-tuning / continual local adaptation** — LoRA-class local adaptation of the
  in-process GGUF provider from the same flywheel, beyond executed hosted fine-tune jobs.
- 🔭 **Differential-privacy memory & training** — a DP accountant over memory consolidation and the
  learning loop so a per-user privacy budget is provable, beyond the consent ledger and provable
  erasure.
- 🔭 **World-model / simulation-based planning** — agents that learn a tool/environment model and plan
  against it, beyond the reset/step/verify environment-eval harness; a natural consumer of the
  test-time search verifiers (theme 1).

**Modality & interaction**

- 🔭 **Native video understanding & generation** — a video `ContentPart` with frame sampling, temporal
  segmentation, and generative output, extending multimodal beyond image and audio.
- 🔭 **MCP Apps & the evolving MCP spec** — server-rendered UI, elicitation, and stateless-core
  changes, adopted once the spec ships stable, tracked alongside AG-UI generative-UI streaming.

**Assurance & governance**

- 🔭 **Formal verification of governance invariants** — machine-checkable proofs that residency,
  erasure, budget, and the shipped injection-containment invariant hold across the whole
  pipeline, beyond the signed audit chain, provable erasure, and the per-run containment check.
- 🔭 **Causal record-replay debugger** — deterministic, byte-faithful replay of a full agent run from
  its trace for time-travel debugging, generalizing the eval-replay and durable-graph time-travel
  already shipped into a first-class developer tool.
- 🔭 **Energy & carbon accounting** — per-run energy and estimated carbon reported alongside cost and
  held by an optional SLO, anticipating sustainability-disclosure demand, on the existing cost-report
  surface.

**Efficiency & reach**

- 🔭 **Learned semantic cache & KV reuse** — a semantic-similarity cache and cross-request KV-prefix
  reuse trained on the platform's own traces, extending exact-match prompt caching toward
  near-miss reuse under the resident-memory budget.
- 🔭 **Edge / WASM in-process runtime** — the dependency-free core compiled for constrained and
  browser/WASM targets, extending "runs in your process" to "runs at the edge."
- 🔭 **Agent negotiation & reputation** — bounded negotiation, contracting, and a reputation signal
  over the existing A2A agent fabric and reliability scoring, for multi-org crews.

**Breaking window**

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
