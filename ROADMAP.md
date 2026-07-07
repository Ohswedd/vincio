<p align="center">
  <img src="assets/logo.svg" alt="Vincio" width="96">
</p>

# Vincio Roadmap

This is the public roadmap for the Vincio library — package `vincio`, CLI `vincio`,
configuration `vincio.yaml`, benchmark suite **VincioBench**. It records what ships
today, the support posture for forward work, and what is intentionally out of scope.
The complete release-by-release history lives in the [CHANGELOG](CHANGELOG.md).

## What "done" means here

Vincio is a single, coherent context-engineering library. Every subsystem is
implemented, tested offline, documented, and demonstrated by a runnable example.
The platform is production-stable: the public surface (`vincio.__all__`) is frozen
under [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a mechanical
[deprecation policy](docs/reference/stability.md), performance and quality targets are
[published as SLOs](docs/reference/slo.md) and held by at-least-as-strict VincioBench
budgets, and every release ships a CycloneDX SBOM with SLSA build-provenance
attestations.

Forward work **deepens and broadens** the platform without changing that contract.
Each new capability sits behind a new entry point or an opt-in extra; the
dependency-free, offline-first path is always the default. Vincio adopts the
ecosystem's standards — MCP, A2A, Agent Skills, the OpenTelemetry GenAI conventions,
C2PA, and the OWASP LLM / NIST AI RMF / MITRE ATLAS frameworks — **in your process**.
It never becomes a hosted service to do so.

## Status

Vincio is **feature-complete and in long-term support**. There is no standing backlog
of new domains; the work that remains is bug-fix, security, and standards-tracking,
plus additive refinements that preserve the frozen surface. New capability is proposed
and gated from scratch when it meets a real need — covered offline, held by VincioBench
budgets and SLOs, and demonstrated by a runnable example — never carried as an
open-ended backlog. Long-term support means **no breaking changes, not no improvement**.
The unified **open evaluation plane** shipped in `v7.0` (additive: ten new top-level
entry points and the `app.benchmark_suite` verb, behind opt-in extras, with no existing
symbol changed) and was **completed in `v7.1`** as fit-and-finish: a machine-readable
benchmark provenance manifest and map spanning all four benchmark planes, the Recorded/Live
tiers made runnable (self-contained live prompts + the `benchmarks/eval_live.py` SOTA runner),
the `vincio eval suite list` catalog command, and the README/docs/asset reconciliation — all
additive. In `v7.2` the benchmark story was **unified into one three-track platform**
(`vincio bench model|uplift|feature`): the public-benchmark plane joined a new **uplift** track (the
same model through Vincio vs direct) and a new **feature** track (a Vincio feature vs a competitor
library, measured live), sharing the provenance tiers, one reporting/CLI surface, and CI gating —
additive, no existing symbol changed. `v7.3` added the **packet compile receipt** — a compact,
text-light `CompileReceipt` (in `vincio.context`, with a `vincio trace receipt` command) that proves
*why* a context packet was compiled (inclusions, exclusions, per-item scores, budget, privacy, and
conflict winners), linked from the run trace and safe to attach to a PR or incident — additive, no
existing symbol changed. `v7.4` added the **DS4 local-inference provider** — a running `ds4-server`
(antirez's self-contained engine for DeepSeek V4) as a first-class Vincio provider, flowing through the
same registry, capability guards, cost table, reasoning controller (thinking modes), residency
(fail-closed on-prem), and audit chain as every other provider, with the DS4 models priced at an honest
self-hosted `$0` — additive, no new hard dependency, no existing symbol changed. `v7.5` was the
**consistency & structure line**: one name per concept across the public surface (`build_*`
factories, `verifier=`, `as_of=`, `digest()`, `content_hash`) behind Vincio's first **active
deprecation runway** (old names warn until their scheduled removal in `8.0`, `vincio migrate 8.0`
ships the codemod today), one implementation per canonical-JSON recipe (zero bytes changed, golden-
pinned), flat hot paths (footprint eviction and multilateral clearing shed their quadratic folds,
proven byte-identical by reference oracles), and one module per responsibility (`ContextApp` verb
mixins, staged `compile()`, phased `combine_attestations` — pure code motion under the byte-identical
SLOs, with the standing-guard lints extended so their coverage did not shrink). `v7.6` added
**universal web browsing & search** — the `vincio.web` plane gives **every** model Vincio serves
(hosted, gateway, or a local model with no function calling at all) the same two Vincio-executed,
governed tools, `web_search` and `web_read`, over DuckDuckGo's keyless endpoints or any pluggable
engine: token-budgeted page reading (only the passages relevant to the model's own query), the
when-to-search judgement shipped as a progressively-disclosed built-in skill, pre-egress `WebPolicy`
rails (SSRF fail-closed, robots, domains, budgets), offline-verifiable content-hashed `WebEvidence`,
the `ToolProtocolProvider` that grants native-grade tool use to models without function calling, and
the `websearch` connector that makes the deep-research agent web-backed — additive, no new hard
dependency, no existing symbol changed. It is the first phase of **native skill integration**:
the when/what/how contract reaches the model through the context plane, identically for every
provider. `v7.7` added **context anchors & dynamic retrieval** — mark a source `anchor=True` and a
PRD / spec / brand corpus that binds a whole multi-call task is distilled once into a compact,
constraint-first, content-hash-cached brief injected as **pinned** evidence into every call at a flat
few-hundred-token cost (~28× smaller than the corpus), guaranteed into the packet at every drop point
(gate, dedup, conflict, footprint, budget) via a budget reservation with a never-raising overflow
ladder, while on-demand detail still flows through normal retrieval — beating "paste every MD file
every call" on tokens and "pure per-query RAG" on frame retention (live: anchors match stuffing on
adherence at ~3× fewer tokens/call while pure RAG drops the rule to 50%); plus opt-in, byte-identical-
default dynamic-retrieval knobs (`embedder="auto"`, grow-only adaptive `top_k`) — additive, no new hard
dependency, no existing symbol changed. `v7.8` added **LAGER — reasoning-driven
retrieval**: the `vincio.lager` plane transforms a corpus into byte-exact, offline-verifiable Evidence
Objects connected in a typed knowledge graph (with a precision-gated contradiction detector), and
replaces fixed top-k retrieval with a lazy, needs-driven loop that expands the graph only while
marginal information gain justifies it — finding multi-hop bridges that share zero words with the
query where same-budget top-k structurally cannot, at ~23× fewer evidence tokens on the gated fixture
(live: 100% vs classic RAG's 75% at ~8× fewer input tokens/call), with honest abstention naming the
uncovered needs, cross-process determinism, and a per-round gain trace making every retrieval decision
explainable — additive, opt-in via `app.use_lager()`, no new hard dependency, no existing symbol
changed. `v7.9` **wired the optional `embedder=` dense signal into LAGER's coverage decision**, tightening
the two deliberate residuals of the embedder-off lexical path that `v7.8` documented — a dense rescue
recalls a topic paraphrase of the cause (via an entity-neutralized topic probe that keeps an
entity-sharing decoy rejected), and an opt-in bridge floor rejects an off-topic same-document decoy —
with the pure-stdlib default left byte-identical (differentially verified) and new optional
`LazyOptions` fields plus an `EvidenceIndex.semantic_similarity` method the only additions. There is
now also a `v7.10` **universal in-house reasoning engine**: an opt-in adaptive layer above every
provider that keeps simple work one-pass, selects decomposition/calculation/logic/tool/evidence-first
strategies for hard work, proactively joins the governed browser to freshness-sensitive reasoning,
and runs bounded candidate → deterministic verifier → correction loops on models with or without
native thinking. Native effort is used when present; models without it receive provider-neutral
test-time passes. Receipts expose decisions, grounding and cost but never model chain-of-thought, and
the mechanism is held by a new offline VincioBench family plus a paid, non-CI OpenRouter uplift
harness. The surface is experimental and additive; `API_VERSION` remains `5.0`. There is
currently **no proposed capability** on the roadmap;
the platform is feature-complete, and new capability is proposed from scratch when it meets a real need
(see [Forward work](#forward-work)).

---

## What ships today

The platform is complete and stable across the subsystems below. Each is covered by a
VincioBench family, published SLOs, and a runnable example, and the full test suite runs
offline.

### Context and prompts

| Subsystem | Capability |
|---|---|
| **Prompt compiler** | Typed prompt ASTs, lint rules, cache-aware stable-prefix layout, versioning, hashing, diffing, and variant generation. |
| **Context compiler** | Multi-signal candidate scoring, deduplication, conflict resolution, compression, and token-budget packing, with an excluded-context report explaining every omission. Text, image, and table evidence in one scored, multimodal-native packet, with cross-process `materialize()` from a content-addressed store. |
| **Long-horizon context** | A per-run context budget (live tokens, residency, KV-cache footprint), intra-run relevance decay, and a provenance-preserving compactor that folds cold spans into hierarchical summaries and pages full text back on demand — so a multi-day, multi-session run stays inside a bounded quality and cost envelope. |

### Retrieval and memory

| Subsystem | Capability |
|---|---|
| **Retrieval (RAG)** | BM25, dense, learned-sparse, and late-interaction retrieval fused in one RRF; query understanding (HyDE, multi-query, decomposition, step-back); sentence-window, parent-document, auto-merging, and contextual chunking; GraphRAG; entity-graph, multi-hop, and reasoning retrieval; Matryoshka, contextual, and multimodal embedders; and a structured `FilterSpec` pushed down to each backend with tenant scope enforced in the engine. |
| **Memory** | Layered, guarded, decaying, conflict-resolving, privacy-scoped memory; hybrid vector + graph recall; consolidation with provenance; audited GDPR-style edit, forget, and export; and bi-temporal records with as-of recall, history-preserving correction, per-memory ACLs, and team scope. |

### Agents and orchestration

| Subsystem | Capability |
|---|---|
| **Tools** | A permissioned registry (RBAC and ABAC), schema derivation from type hints, a resource-limited sandbox, reliability scoring, and idempotent approval-gated writes. |
| **Web browsing & search** | Governed `web_search` / `web_read` tools every model can call — DuckDuckGo or any pluggable engine, token-budgeted page reading (only the passages relevant to the model's own query), a built-in progressively-disclosed when-to-search skill, pre-egress policy (SSRF fail-closed, robots, domains, budgets), offline-verifiable content-hashed evidence, and a text protocol that grants native-grade tool use to models without function calling. |
| **Universal reasoning** | Adaptive direct/standard/deep routing over task taxonomy, modalities, constraints, math, logic, causal/decision/temporal/spatial structure, exact tool matches and freshness; a bounded model-native semantic route extends this to every language understood by the selected model without a locale allow-list; evidence-first browser integration with multilingual no-web handling; Unicode evidence support, bounded candidates, deterministic verifier certificates and answer-only correction on native and non-reasoning models, with no stored chain-of-thought. |
| **Computer-use action plane** | A grounded perceive, gate, act, verify, and undo loop over a pluggable screen backend (a deterministic mock offline; Playwright/CDP, an OS accessibility tree, or remote desktop behind an extra). Actions bind to stable role-and-name selectors, are pre-gated like a write tool, post-verified against an expected end state, and undone on divergence. |
| **Agents** | Bounded DAG execution with planners (ReAct, plan-and-execute, hierarchical HTN), in-place plan repair, cost-aware action selection over live pricing and budget, a deep-research agent, and a self-editing memory OS. |
| **Orchestration** | Multi-agent crews with a shared blackboard; durable graphs with checkpoint, resume, time-travel, and durable timers; deterministic workflows with retries, compensation, and resumable approval gates; and a distributed durable-execution backend over a worker pool. |

### Output, evaluation, and observability

| Subsystem | Capability |
|---|---|
| **Structured output** | Pydantic contracts, provider-native constrained decoding, streaming validation with early abort, typed signatures, multi-schema routing, and bounded self-correction that repairs structure only — never facts. |
| **Evaluation** | Golden datasets, 30+ metrics, calibrated judges and disagreement-aware judge ensembles, synthetic data, red-teaming, experiments with significance, regression gates that attribute a failure to its cause, adaptive sampling, a pytest plugin, a stateful-environment harness with a task-success oracle, and adapters for nine agentic benchmarks (SWE-bench, τ-bench, GAIA, WebArena, BFCL, AgentBench, ToolBench, LiveCodeBench, MMLU-Pro). |
| **Open evaluation plane** | One pluggable harness for the standard public model benchmarks (MMLU, GPQA, GSM8K, HumanEval, IFEval, TruthfulQA, RULER, …) grouped by ten niches behind one `BenchmarkAdapter` contract, scored by reused metrics, and reported the same way for every model and version. Every number carries an enforced **provenance tier** (Static / Recorded / Live) the engine refuses to let a lower tier inflate; long-context benchmarks run twice (with and without the `ContextGovernor`) so the uplift is measured, and prompt injection reports *contained vs compromised*. Deterministic, concurrent, resumable runs over a model or app; Markdown / HTML / JSON / CSV / PDF reports and a ranked leaderboard; charts; a SQLite (or Postgres) run store with model-version diffs; and a `vincio.benchmarks` plugin API. In-process and offline-reproducible; never a hosted leaderboard. |
| **Observability** | Full trace span trees, sessions, feedback, eval scores on spans, JSONL and OpenTelemetry export, a local viewer, a self-hosted observability and alerting plane, a versioned prompt registry, and per-run cost tracking — no account or hosted backend required. |

### The closed loop

| Subsystem | Capability |
|---|---|
| **Optimization and self-improvement** | One reproducible cycle (trace → dataset → eval → optimize → promote) with safety-gated promotion: reflective (GEPA-style) optimization and MIPRO, a distillation flywheel, learned prompt compression, canary-gated deploy with rollback, and on-policy reinforcement from verifiable rewards (RLVR). No promotion ships without clearing the gates. |
| **On-device and federated adaptation** | Parameter-efficient on-device LoRA adaptation promoted only through a no-regression gate; federated self-improvement that shares masked, bounded-sensitivity numeric aggregates — never raw traffic — across organizations; a per-subject differential-privacy accountant over memory and training; and a per-member reputation that discounts an unreliable contributor without singling it out. |

### Security and governance

| Subsystem | Capability |
|---|---|
| **Security** | Deterministic PII, secret, injection, and RAG-poisoning detection; programmable rails; RBAC and ABAC; tenant isolation; and a signed, Merkle-checkpointed audit chain with offline tamper verification. Provable prompt-injection containment separates the control plane from the data plane through typed trust labels, unforgeable capability tokens, a dual-plane executor, and a machine-checked containment invariant. |
| **Governance** | A formal governance-invariant verifier that *proves* containment, residency, the budget cap, and the erasure-proof binding hold across their whole bounded state space ahead of any run; model and system cards; a compliance coverage matrix; an AI-BOM; an EU AI Act conformity pack; provable erasure; a consent ledger; data lineage; and residency-aware egress refusal. |
| **Identity and accountability** | DID-based agent identity on self-certifying Ed25519 keys, a signed key-rotation chain, and attenuating delegation chains, so every audited action, contract, and settlement carries cryptographic provenance of who authorized it, down what chain, within what bounds. |
| **Verified reasoning** | Checkable certificates for the classes of question that admit them: deterministic kernels (arithmetic, units, temporal, schema, constraints, citation entailment, and statistical trend / correlation / interval / forecast) recompute a claim and refuse to emit a refuted answer. A runtime shield blocks a policy-violating action before it executes, and tool contracts and program synthesis carry proofs into the tool plane. |
| **Continuous assurance** | An assurance-case argument tree bound by hash to the evidence the platform already emits (eval gates, governance proofs, certificates, identity, the audit chain, SBOM/SLSA), re-checked on every change, with a portable certification report. |

### Generation and multimodal

| Subsystem | Capability |
|---|---|
| **Generation** | Cited DOCX, PDF, PPTX, HTML, and Markdown; a cited-report builder with per-claim entailment; redlines; image generation and text-to-speech with C2PA provenance; and richer inputs (OCR, transcripts, additional loaders, forms/KYC). |
| **Video** | Video as a first-class modality on the multimodal packet: a video content part, deterministic frame sampling and temporal segmentation, and an analyzer that turns a clip into typed evidence the context compiler scores, budgets, and cites beside text and images. Temporal grounding points a clip-grounded answer at the moment it came from, and generated or edited video carries a C2PA manifest bound to its bytes. |

### Data and analytics

| Subsystem | Capability |
|---|---|
| **Tabular evidence** | A typed, columnar `Dataset` and a deterministic, lossless `DataEncoder` that renders it header-once — far cheaper than JSON or a Markdown table, and columnar-accurate in token cost. `TableEvidence` scores and cites a table like any other evidence. |
| **Profiling, sampling, and quality rails** | Bounded-memory dataset profiling, reservoir and stratified sampling, a fit-to-window representation whose size is invariant to the row count, and deterministic data-quality rails reusing the same PII, secret, and injection detectors as the text path. |
| **Governed text-to-query** | A natural-language question (or explicit SQL or a dataframe pipeline) grounded into a schema-checked, read-only-verified, cost-bounded query executed where the data lives, returning a result that cites the exact source cells and verifies offline against the content-hashed source. |
| **Analysis agent and charts** | A bounded plan → query → inspect → refine → synthesize loop whose every finding is grounded and cited by construction, and content-bound, data-bound charts that carry a C2PA data-driven credential and re-derive from their source on verification. |
| **Streaming and out-of-core** | A lazy, re-iterable `RowStream` over a source larger than memory, a bounded-memory group-by, a header-once streaming encoder, and a streaming candidate pre-filter that bounds a large evidence pool before scoring. |
| **Semantic layer and real-time analytics** | Measures, dimensions, and derived columns defined once so a question maps to a governed metric computed one way everywhere, cell-cited and verifiable; the same plane re-expressed over an unbounded event stream a window at a time; and a cross-organization federated path in which only aggregated, cited results cross the trust boundary, never the raw rows. |
| **Data engagement** | A facade that threads the whole plane (register → profile → … → cite) into a hash-chained, signed `DataNarrative` that verifies offline and is data-bound — every finding re-executes against the content-hashed source — plus an interactive notebook-native front over the same governed primitives. |

### Providers, cost, and performance

| Subsystem | Capability |
|---|---|
| **Providers and storage** | OpenAI, Anthropic, Google, Mistral, any OpenAI-compatible endpoint, enterprise endpoints behind an auth strategy, a deterministic mock, local neural models, and a self-hosted **DS4 DeepSeek V4** box (antirez's `ds4-server`) as a first-class provider — thinking modes driven by the reasoning controller, disk-KV cache accounting, fail-closed on-prem residency, and an honest self-hosted `$0`. A data-driven `ModelRegistry` whose shipped catalog prices the current lineup of every provider is the single source of truth for the cost table, the capability guard, the router, the cascades, and energy accounting; a coverage gate proves no current model resolves to nothing and silently bills $0 (a self-hosted model legitimately at $0 carries an explicit `self_hosted` flag). Pluggable metadata, blob, analytics, vector, and graph backends with Redis shared state. |
| **Cost and reliability** | Half-cost batch execution, circuit breaking, health-aware failover, key pooling, model cascades, cost attribution with budget SLOs, prompt caching, incremental and sharded indexing, a capability-aware router, a swap gate, and a lifecycle watcher. |
| **Runtime performance** | A single-pass vectorized scorer (NumPy-optional, pure-Python fallback) and a per-compile feature arena that derives each candidate's terms, shingles, and blocking tokens once and threads them through every pass; compiled render programs and warm candidate arenas; streaming-first compilation; speculative retrieval prefetch; and a per-app resident-memory budget. The single-pass path is byte-identical to the per-pass derivation, and its speedup is held by a ratio floor so an erased win fails the build. |
| **Test-time compute** | A reasoning controller that sets thinking effort and a token budget per step under a hard ceiling, reasoning-trace-aware caching, a learned semantic cache that serves a near-miss only above a learned precision bar, and a verifier-guided test-time search (best-of-N, self-consistency, beam) over tool-use trajectories. |

### Protocols and interoperability

| Subsystem | Capability |
|---|---|
| **Protocols** | MCP client and server, A2A agent-to-agent, and Agent Skills, all in-process; MCP Apps surface (UI resources over the AG-UI channel and governed elicitation); and protocol-version negotiation with a stateless-core transport. |
| **Ecosystem** | Import and export of LangChain, LlamaIndex, Haystack, and DSPy assets; first-party data connectors (Jira, Linear, Google Drive, SharePoint, Salesforce, Zendesk, BigQuery, Snowflake, and more); an entry-point plugin system under a versioned contract; a signed, allow-list-gated community pack and skill registry; and any OpenAI-compatible model or vector store you already run. |
| **Verticals** | Full-stack vertical packs (healthcare, legal e-discovery, financial KYC/AML, customer support, code review) that preconfigure retrieval, scoped memory, rails, metrics, residency, and a golden eval set; a conversational `Assistant` over `ContextApp`; an end-to-end voice agent; and a cookbook of task-shaped recipes. |

### Cross-organization agent economy

| Subsystem | Capability |
|---|---|
| **Negotiation and contracting** | A bounded, terminating offer/counter-offer bargain that mints a typed, signed, offline-verifiable contract over price, SLA, scope, and quality — enforced like any other budget — running offline or over the A2A fabric, weighted by the counterparty's reputation. |
| **Choreography and settlement** | Durable, compensating cross-organization sagas over the negotiated contract with per-org self-governance and a restart-surviving journal; metered, signed, offline-verifiable settlement records; multilateral netting; and deterministic dispute arbitration. |
| **Reputation and credit** | Portable, signed reputation attestations with freshness, revocation, pull-based gossip, and Sybil-resistant transitive trust; reputation-gated admission and progressive exposure; collateral escrow, pooling, and rehypothecation guards; and proof-of-reserves, proof-of-solvency, liability completeness and consistency, and seniority-waterfall insolvency resolution. |
| **Engagement lifecycle** | A facade that threads the whole fabric (discover → negotiate → contract → deliver → settle → net → arbitrate → attest → admit → collateralize → resolve) into a hash-chained, signed engagement narrative that verifies offline. |

### Edge, reach, and sustainability

| Subsystem | Capability |
|---|---|
| **Edge / WASM runtime** | The dependency-free core (the prompt and context compilers, the vectorized scorer's pure-Python fallback, the deterministic rails, the offline evidence path) packaged for constrained and browser/WASM targets, with a manifest that statically certifies the core imports nothing native and a parity check proving an edge compile is byte-identical to a server compile. |
| **Autonomous skill acquisition** | An open-ended propose → attempt → verify → distill → promote loop whose every proposed objective is screened by the rails and the governance verifier before it runs, distilling oracle-verified trajectories into a content-addressed skill library and promoting a skill only through the same no-regression gate a deploy clears. |
| **Energy and carbon accounting** | A per-run energy and estimated-carbon figure on the existing cost-report surface, budgeted like a dollar cap, computed in-process from a built-in intensity table with every figure and refusal audited. |

### Ergonomics and documentation

| Subsystem | Capability |
|---|---|
| **Ergonomic front door** | A small `vincio.tasks` namespace of task-shaped one-line constructors (`rag`, `extractor`, `tool_agent`, `evaluation`, `chat`) and a fluent, immutable `Flow`, each lowering byte-identically to the same governed `ContextApp.run`. `.app` is the escape hatch to every deep method. |
| **Connected documentation** | A single source of truth binds every public `app.*` verb to the concept, guide, example, and reference that document it; from it Vincio renders a capability map, per-page cross-links, a learning path, and `llms.txt`, all held current by a docs-graph completeness gate. |

---

## Forward work

Vincio is feature-complete and in long-term support, so new capability is the exception,
not a backlog. **There is currently no proposed capability.** New capability is proposed
against a real need and held to the same bar as everything that ships today: **covered
offline**, gated by **VincioBench budgets and published SLOs**, demonstrated by a **runnable
example**, and **additive** — delivered behind a new entry point or opt-in extra so the frozen
`vincio.__all__` contract and the dependency-free, offline-first default are preserved, and
designed as a *composition of subsystems Vincio already ships*, not a parallel stack beside
them.

The two capabilities previously proposed here have shipped: the open evaluation plane in `v7.0`,
and the **DS4 local-inference provider** in `v7.4` (a running `ds4-server` as a first-class
provider — thinking modes on the reasoning controller, disk-KV accounting, fail-closed on-prem
residency, an honest self-hosted `$0`, gated by the `ds4_provider` VincioBench family with six
published SLOs, and a runnable `examples/18_ds4_local_inference.py`). Both are now in
[What ships today](#what-ships-today).

---

## Out of scope

Vincio is a library, and stays one. The building blocks for running it in production —
a hash-chained audit log, retention policies, tenant isolation, RBAC/ABAC, and a
server — ship in the package so you can deploy them on your own infrastructure.
**Hosted services, managed control planes, dashboards-as-a-service, and compliance
programs are not part of this project.**

Everything that *looks* operational is something you run yourself: the observability
and alerting plane is self-hosted over your own store, the `vincio serve` launcher is a
process you manage, the distributed backend is a lock-free adapter to your Temporal or
Ray, the agent fabric is a governed directory you operate, and every standard (MCP, A2A,
OWASP/NIST, OpenTelemetry GenAI, C2PA) is implemented in-library. Vincio gives you the
engine; how and where you run it is yours.
