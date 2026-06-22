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
| **Security & governance** | Deterministic PII / secret / injection / RAG-poisoning detection, programmable rails, RBAC/ABAC, tenant isolation, a signed Merkle-checkpointed audit chain; provable prompt-injection containment that separates the control plane from the data plane — typed `TrustLabel` / `TaintedValue` information-flow labels, unforgeable `CapabilityToken`s minted from the user's request, a `DualPlaneExecutor` whose privileged planner sees only typed extractions of untrusted bytes, and a machine-checked containment invariant (`untrusted ⇒ no unapproved capability`); a formal **governance-invariant verifier** (`app.verify_governance`) that *proves* containment, residency, the budget cap, and the erasure-proof binding hold across their whole bounded, typed state space ahead of any run — a deterministic, offline, on-chain artifact that yields a minimal counterexample on violation; model & system cards, a compliance coverage matrix, an AI-BOM, an EU AI Act conformity pack, provable erasure, a consent ledger, data lineage, and residency-aware egress refusal. |
| **Generation** | Cited DOCX/PDF/PPTX/HTML/Markdown, a cited-report builder with per-claim entailment, redlines, image generation and TTS with C2PA provenance, and richer inputs (OCR, transcripts, new-format loaders, forms/KYC). |
| **Video understanding & generation** | Video is a first-class modality on the existing multimodal packet, not a new plane. A `VideoRef` / video `ContentPart`, deterministic frame sampling (`sample_frame_times`) and temporal segmentation (`segment_timeline`), and a `VideoAnalyzer` (deterministic `MockVideoAnalyzer` offline; `ProviderVideoAnalyzer` + a `PyAVFrameExtractor` behind the `vincio[video]` extra) turn a clip into typed `modality="video"` evidence (`video_evidence_items`) the context compiler scores, budgets, orders, and cites beside text and images. Temporal grounding carries a segment's `time_range` from the loader through retrieval to the cited-report footnote, so a clip-grounded answer points at the moment it came from (`<source>:t<start>-<end>`), auditable at sub-clip resolution. Generated or edited video (`VideoProvider` — `MockVideoProvider` / OpenAI Sora / Google Veo / a generic HTTP adapter) carries a C2PA manifest bound to its bytes, metered against the run budget and audited, exactly the way generated images and audio do (`app.load_video` / `app.generate_video` / `app.edit_video`). |
| **Providers & storage** | OpenAI, Anthropic, Google, Mistral, any OpenAI-compatible endpoint, enterprise endpoints behind an `AuthStrategy`, a deterministic mock, and local neural models; a data-driven `ModelRegistry`; pluggable metadata / blob / analytics / vector / graph backends with Redis shared state. |
| **Protocols & interoperability** | MCP client + server, A2A, Agent Skills, a governed agent fabric over an `AllowListGate`, AG-UI generative-UI streaming, and LangChain / LlamaIndex / Haystack / DSPy interop. |
| **Ecosystem & integration breadth** | First-party connectors for Jira, Linear, Google Drive, SharePoint, Salesforce, Zendesk, BigQuery, and Snowflake feeding the document engine with full provenance behind `register_connector`; an entry-point plugin system (`vincio plugins list`) registering third-party providers, metrics, chunkers, rerankers, judges, connectors, and packs on install under a versioned plugin-API contract; a signed, allow-list-gated, audited `CommunityRegistry` of opt-in packs and `SKILL.md` bundles; and an MCP-server marketplace bridge (`app.add_mcp_from_registry`) that discovers, governs, and lands a server's tools in the permissioned runtime in one call. |
| **Use-case coverage & verticals** | Full-stack vertical packs (healthcare/PHI, legal e-discovery, financial KYC/AML, customer support, code review) that preconfigure retrieval, scoped memory, deterministic rails, domain metrics, a data-residency posture, and a golden eval set on top of the pack contract; a higher-level `Assistant` over `ContextApp` that threads turns into a session, carries multi-turn state via memory write-back, and gates write tools behind an approval; an end-to-end `VoiceAgent` wiring the realtime session to the deep-research agent, the memory OS, and the rails; and a cookbook of task-shaped recipes (contract redlining, incident triage, data-room Q&A, multimodal RAG over slides/PDFs) as offline-gated runnable examples. |
| **Cost, reliability & rotation** | Batch execution, circuit breaking, health-aware failover, key pooling, model cascades, cost attribution with budget SLOs, prompt caching, incremental + sharded indexing, a capability-aware router, a swap gate, and a lifecycle watcher. |
| **Runtime performance** | A single-pass vectorized scorer (NumPy-optional, pure-Python fallback); a compiled-prompt render program and a warm candidate arena that reuse the stable prefix and the prepared candidate set so a warm compile is dominated by scoring, not allocation; streaming-first compilation that emits the prefix before scoring; speculative retrieval prefetch that warms the query embedding from the task classification; and a per-app resident-memory budget held by slim packets and evidence eviction, surfaced in the cost report and gated by an SLO. |
| **Test-time compute & reasoning** | A `ReasoningController` (`app.use_reasoning_controller`) that sets thinking effort and a thinking-token budget per step from the task classification and the live budget under a hard reasoning-token ceiling held by an SLO; reasoning-trace-aware caching (`ReasoningTraceCache`) that reuses a warm thinking prefix under the resident-memory budget; and a verifier-guided `TestTimeSearch` (`app.test_time_search`) — best-of-N, self-consistency, and beam search over tool-use trajectories scored by the *existing* critics and judge ensembles through one `Verifier` protocol, early-exiting the moment the verifier clears the bar, bounded by the same budgets the orchestrator enforces. |
| **Long-horizon context engineering** | A per-run `ContextGovernor` (`app.use_context_governor`) holding a `ContextBudget` (live tokens, residency, KV-cache footprint) the way the cost report holds a dollar budget; intra-run `RelevanceDecay` that demotes stale spans before they crowd out fresh signal, surfaced in the excluded-context report; and a provenance-preserving `ContextCompactor` that folds cold spans into hierarchical summaries in the memory OS and pages their full text back on demand from the content-addressed store — so a million-token, multi-day, multi-session run stays inside a bounded quality and cost envelope as the horizon grows 10×, held by a horizon-scaling SLO. |
| **World-model / simulation-based planning** | A deterministic, offline `WorldModel` fit from recorded reset/step transitions that learns each tool's parameterized effect under a learned precondition (predicting the next observation and a verifier-scored reward, generalizing over arguments) and earns planning weight only once a `CalibrationReport` shows its predictions track the real environment; and a `ModelPredictivePlanner` that searches imagined rollouts with the test-time-search beam, commits the best first action, and re-plans on the real observation — bounded by the same budgets the orchestrator enforces and held by a planning-accuracy SLO (an imagined-rollout planner matches or beats reactive planning at a fixed action budget on the environment harness). |
| **Causal record-replay debugger** | A `Recorder` (`vincio.observability`) that captures every non-deterministic edge of a run — model responses, tool outputs, retrieval hits, the negotiated capabilities, and the clock/seed — keyed to its trace spans into a portable, content-addressed, verifiable `Recording`; a deterministic `Replayer` that serves each edge back so a recorded run replays byte-for-byte (the recording, not the live provider, drives the run) with a step/inspect surface over the span tree and a `Divergence` report the moment live code no longer matches; and branch-and-edit that forks a recording, changes an edge or the input, and re-executes only the affected suffix while the unchanged prefix is still served from the recording — held by a replay-fidelity SLO (a recorded run replays byte-identically and a divergence is detected). |
| **Learned semantic cache & near-miss KV reuse** | A `LearnedSemanticCache` (`app.use_semantic_cache`) that answers a *semantically-equivalent* (not byte-identical) request from cache, serving a near-miss only above a `ThresholdCalibrator` acceptance bar *learned from the platform's own traces* so an accepted hit clears a precision target (never serving below the floor); a `KVPrefixPool` (`app.use_kv_prefix_reuse`) that reuses a shared stable-prefix KV footprint across a family of requests that share a head, reporting the serving-engine KV the shared head avoids recomputing; and a `SemanticCacheGate` that catches a drifted cache with the same eval-replay no-regression check that gates a model swap — every near-miss auditable and reversible, all held under the resident-memory budget and a hit-quality SLO (an accepted near-miss is at-least-as-good as a live answer at a fixed budget). |
| **On-device fine-tuning & continual local adaptation** | A `LocalLoRATrainer` that fits a parameter-efficient, low-rank `LocalAdapter` on-device from the flywheel's grounded dataset (deterministic and dependency-free, with a `NativeLoRABackend` hook for a real GGUF/LoRA); an `AdaptedProvider` that applies it to any provider so in-distribution traffic is answered the way it was taught while off-distribution traffic falls through to the base model unchanged (bounded); a `ContinualAdaptation` loop (`app.adapt_locally` / `app.local_adaptation`) that promotes a new adapter version only when the locally-adapted model is at-least-as-good as its base on a held-out set — the same no-regression gate a hosted fine-tune job clears — and an `AdapterRegistry` that versions every adapter and rolls it back on regression. Apply or unload one live with `app.use_local_adapter`; the run never leaves the process, held by a no-regression SLO (a locally-adapted model is at-least-as-good as its base on the eval set). |
| **Federated / cross-org self-improvement** | Sharing what was learned across organizations without sharing the raw traffic. Each member builds a numeric, raw-text-free `Contribution` — the clipped, optionally DP-noised, and secure-aggregation-masked subspace scatter of its local adapter geometry (`app.contribute_federated`) — behind the consent ledger's TRAINING purpose and the residency posture, never a prompt or a response. A `SecureAggregator` merges the fleet's contributions into a shared `FederatedSubspace` by deterministic federated PCA — the pairwise masks cancel exactly, so no single member's update is ever observed — refusing a round below the `PrivacyConfig` k-anonymity contributor floor. The adopting member re-fits its own adapter against the shared geometry (`app.adopt_federated` / `app.federated_improvement`), keeping its own grounded answers local, and adopts it only when at-least-as-good as its base on a held-out set — the same no-regression and canary gates a local promotion clears, versioned in the `AdapterRegistry` and rolled back on regression. Only numeric, masked, bounded-sensitivity aggregates cross a trust boundary; held by a privacy SLO and a no-regression SLO. |
| **Differential-privacy memory & training** | A provable, composing, per-subject privacy budget over memory consolidation and the whole learning loop. A Rényi/moments `PrivacyAccountant` (`app.use_privacy_accountant`) tracks the cumulative `(ε, δ)` a subject's data has spent across every consolidation and learning round — composing across rounds far more tightly than naively summing each step's `ε` — and a `PrivacyBudget` gates a learning step the way the cost report gates a dollar: a consolidation or a federated contribution that would exceed a subject's remaining budget is refused (a hard cap) or down-weighted (clipped harder, so its sensitivity and privacy cost fit). It wires into memory consolidation and federated contributions automatically; `app.privacy_report()` rolls up each subject's spent / remaining `ε` next to the cost report; and every spend and refusal lands on the hash-chained audit log, so the guarantee is a mechanical, auditable number. `gaussian_rdp` / `rdp_to_epsilon` expose the accountant's math; held by a budget-composition SLO and a refusal SLO. |
| **Cross-fleet reputation & weighting** | A per-member reputation that discounts an unreliable or adversarial member's pull on the federated consensus — earned only from how each contribution fared against the no-regression gate, never from raw traffic. A `ReputationLedger` (`app.use_reputation_ledger`) keeps each member's reliability as a Beta-Bernoulli posterior over gate outcomes (a robust generalization of the existing reliability scoring): a newcomer earns the benefit of the doubt from a prior, a repeatedly-regressing member decays toward a floor, and a reformed member recovers. The `SecureAggregator` weights a member's contribution by its reputation before distilling the consensus subspace — folded in before the secure-aggregation masks so they still cancel exactly — so a regressor is discounted **without being singled out**. The discount is bounded and reversible: a weight never leaves `[floor, 1]` (lowering pull, never zeroing or bypassing the quality bar), and adoption still clears the same no-regression and canary gates a local promotion does. Every update lands on the signed audit chain and replays from it (`ReputationLedger.from_audit`), and `app.reputation_report()` rolls up each member's standing next to the cost and privacy reports; held by a discount-the-regressor SLO and a no-regression SLO. |
| **Energy & carbon accounting** | A per-run energy (watt-hours) and estimated carbon (grams CO₂e) figure on the existing cost-report surface — the sustainability analogue of the dollar budget, never a new plane. `app.use_energy_accounting(region=)` turns it on; every run then accrues a deterministic estimate from its own token accounting against a per-model intensity (by tier, from the `ModelRegistry`) and a per-region grid factor, surfaced on `result.energy_wh` / `result.co2e_grams`, `app.cost_tracker.summary()`, and `app.energy_report()` (rolled up by model/tenant/feature from the same attributed events the cost report uses). Budgeted like a dollar: `app.set_energy_budget(limit_wh= / limit_co2e_grams=)` refuses a run that would exceed its sustainability envelope, the way a hard cost cap refuses spend. Auditable and offline: the estimate is computed in-process from a built-in intensity table (no external service), and both the per-run number and every refusal land on the hash-chained audit log. Held by a per-run-estimate SLO, a budget-refusal SLO, and an auditable-offline SLO. |
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

The most recent scheduled theme — **native video understanding & generation** (a `VideoRef` / video
`ContentPart`, deterministic frame sampling and temporal segmentation, a `VideoAnalyzer` that lowers a clip
into typed `modality="video"` evidence the compiler scores and cites beside text and images, temporal
grounding that carries a segment's `time_range` through to the cited-report footnote, and C2PA-bound video
generation/editing — `app.load_video` / `app.generate_video`) — has shipped and folded into the **Video
understanding & generation** row above. The next theme is scheduled below. It closes a specific gap in the
platform's *own* frontier — a rung that exists in the literature and in buyer demand but not yet in the
package — rather than a gap measured against any one competitor. An indicative minor-version target is given;
cadence holds one coherent theme per minor.

### 1 · Edge / WASM in-process runtime *(target 3.21)*

Vincio's promise is "runs in your process." The dependency-free core — the prompt and context compilers, the
vectorized scorer with its pure-Python fallback, the deterministic rails, and the offline-first evidence path —
already has no native dependencies on the default path, which makes the next reach a natural one: compile that
core for constrained and browser/WASM targets, so the same context engineering runs at the edge and in the
browser, not only on a server.

- **The core compiled for WASM** — the compile / score / rail / pack path running in a browser or an
  edge worker with no Python runtime, behind a thin in-process boundary, keeping the offline-first default.
- **A bounded edge profile** — a resident-memory and latency profile for constrained targets, held by an
  edge-scaling SLO the way the resident-memory budget holds a server run today.
- **Parity, not a fork** — the edge build is the same library under a build target, exercised by the same
  offline test suite, so a capability never silently diverges between server and edge.

*Ships as:* a WASM/edge build target for the dependency-free core (with the server path unchanged as the
default); a bounded edge resident-memory and latency profile; an `edge` VincioBench family with an
edge-scaling SLO; a runnable example.

---

## 🔭 Exploring — later

Candidates that are real but not yet scheduled — pulled forward when demand and the standards settle.
Grouped by where they would land.

**Modality & interaction**

- 🔭 **MCP Apps & the evolving MCP spec** — server-rendered UI, elicitation, and stateless-core
  changes, adopted once the spec ships stable, tracked alongside AG-UI generative-UI streaming.

**Efficiency & reach**

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
