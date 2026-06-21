<p align="center">
  <img src="assets/banner.svg" alt="Vincio — the context engineering platform for AI applications" width="660">
</p>

<p align="center">
  <em>The scarce resource is not the model. It is the context you feed it.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/vincio/"><img src="https://img.shields.io/badge/vincio-3.7.0-B98B2E" alt="Vincio 3.7.0"></a>
  <a href="https://github.com/Ohswedd/vincio/actions/workflows/ci.yml"><img src="https://github.com/Ohswedd/vincio/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/pypi/pyversions/vincio?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-4C6EF5" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/tests-1929%20passing-2ea44f" alt="1929 tests passing">
  <img src="https://img.shields.io/badge/lint-ruff-D7FF64" alt="Ruff">
  <img src="https://img.shields.io/badge/typed-pydantic%20v2-E92063" alt="Pydantic v2">
  <img src="https://img.shields.io/badge/offline-first-555" alt="Offline-first">
</p>

---

**Vincio** is a Python platform for building **context-engineered** AI applications. It compiles
prompts, memory, retrieval, tools, schemas, and policies into optimized, testable, observable,
provider-neutral **context packets** — then validates and evaluates every output.

Most frameworks help you *call* a model. Vincio governs the **boundary** between your application
state and the model: what evidence is selected, how it is scored and budgeted, how it is rendered
for cache reuse, and how the result is validated, measured, and traced. Named for **Leonardo da
Vinci** — engineering and craft in equal measure.

```text
Raw Input → Normalization → Objective Detection → Memory Selection
→ Retrieval Planning → Evidence Retrieval → Ranking + Distillation
→ Tool Planning → Context Compilation → Model Execution
→ Parsing + Validation → Evaluation + Guardrails → Trace + Learning Loop
```

## Contents

[Why Vincio](#why-vincio) · [Install](#install) · [Quickstart](#quickstart) ·
[Features](#features) · [Benchmarks](#benchmarks) · [Comparison](#how-vincio-compares) ·
[Use cases](#use-cases) · [CLI](#command-line) · [Architecture](#architecture) ·
[Roadmap](#roadmap) · [Documentation](#documentation)

## Why Vincio

Teams ship a prompt, watch it work, then spend months fighting everything around it: context that
overflows the window, retrieved chunks that contradict each other, outputs that fail to parse,
silent quality regressions, untraceable costs, and prompt-injection risk. These are not model
problems — they are **context** problems.

Vincio treats context as a compiled artifact with a clear contract:

- **Deterministic where it matters.** Security, permissions, and validation are enforced in code —
  never gated on model output. The same input compiles to the same packet.
- **Measured, not asserted.** Every run is traced and costed; every change can be gated by an eval
  suite before it ships.
- **Provider-neutral.** OpenAI, Anthropic, Google, Mistral, any OpenAI-compatible endpoint, or a
  deterministic offline mock — behind one interface.
- **One coherent model** from input to output, instead of a bag of loosely-coupled utilities.

## Install

```bash
pip install vincio                  # core — runs fully offline with the mock provider
pip install "vincio[openai]"        # + OpenAI provider
pip install "vincio[anthropic]"     # + Anthropic provider
pip install "vincio[chroma]"        # + a vector store (also: pinecone, lancedb, postgres,
                                    #   weaviate, milvus, elasticsearch, opensearch, vespa)
pip install "vincio[realtime]"      # + voice/realtime sessions (OpenAI Realtime, Gemini Live)
pip install "vincio[langchain]"     # + framework interop export (also: llamaindex, haystack, dspy)
pip install "vincio[snowflake]"     # + a warehouse connector (also: bigquery)
pip install "vincio[all]"           # every optional integration
```

Python 3.11+. Core dependencies are just `pydantic`, `httpx`, `pyyaml`, and `typing-extensions`;
every heavy integration (vector stores, OCR, server, OpenTelemetry, …) is an opt-in extra.

## Quickstart

```python
from vincio import ContextApp

app = ContextApp(name="docs_qa")
app.add_source("docs", path="./docs", retrieval="hybrid")
app.set_policy("answer_only_from_sources", True)

result = app.run("How do I configure SSO?")
print(result.output)      # the grounded answer
print(result.citations)   # evidence the answer actually cited
print(result.trace_id)    # every run produces a full trace
print(result.cost_usd)    # …and a cost
```

No API key? It runs offline out of the box on a deterministic mock provider that emits
schema-valid output — so your whole pipeline (retrieval, validation, evals, traces) runs for real
in CI.

### Typed output

```python
from pydantic import BaseModel
from vincio import ContextApp

class TicketClassification(BaseModel):
    label: str
    confidence: float
    reason: str

app = ContextApp(name="triage", output_schema=TicketClassification)
result = app.run("The dashboard crashes after login")

result.output.label        # → a validated TicketClassification instance
```

### Agents with tools and memory

```python
app = ContextApp(name="support_refunds", output_schema=RefundDecision)
app.add_memory(scope="user", strategy="semantic")
app.add_tool("billing_lookup", permissions=["billing:read"])
app.add_tool("refund_create", permissions=["billing:write"], approval_required=True)

agent = app.agent(max_steps=6)
result = agent.run("Customer asks for a refund on invoice INV-123.")
```

### Multi-agent crews and durable graphs

```python
from vincio.agents import interrupt

crew = app.crew(members=[
    {"name": "researcher", "goal": "gather the numbers", "keywords": ["find"]},
    {"name": "writer", "goal": "draft the recommendation"},
])
result = crew.run("Explain the Q3 refund trend")   # bounded, traced, blackboard-shared

graph = app.graph("review")                        # checkpointed in your own store
graph.add_node("analyze", analyze)
graph.add_node("approve", lambda s: {"ok": interrupt(s, "proceed?")})
graph.add_edge("analyze", "approve")
flow = graph.compile()
paused = flow.invoke({"doc": "msa.pdf"})           # pauses at the human gate
done = flow.resume(paused.thread_id, value=True)   # later — even after a restart
```

### Reliability as a guarantee

```python
from vincio import Signature, InputField, OutputField

class Triage(Signature):
    """Classify a support ticket."""
    ticket: str = InputField(desc="the raw ticket text")
    label: str = OutputField(desc="bug | billing | feature | other")
    confidence: float = OutputField()

result = app.predictor(Triage)(ticket="The export button 500s")  # typed + validated

app.add_rail(name="no_leaks", kind="safety", detectors=["pii", "secrets"], action="redact")
app.add_rail(name="on_topic", kind="topic", direction="input", blocked_topics=["legal advice"])
app.enable_self_correction(max_cycles=2, max_cost_usd=0.05)      # facts never invented
app.add_output_schema(BugReport, keywords=["bug", "crash"])       # multi-schema routing

async for event in app.astream("Extract the invoice"):
    if event.type == "partial_output" and event.valid_prefix is False:
        break   # streaming validation: stop paying for a doomed answer
```

### Evaluation as a gate

```python
from vincio.evals import Dataset, EvalRunner

dataset = Dataset.load("golden/support_triage.jsonl")
report = EvalRunner(app).run(dataset)
report.print_summary()     # groundedness, citation accuracy, schema validity, cost — with CI exit codes
```

### Self-improvement as a contract

```python
from vincio.optimize import SelfImprovementPolicy

# One declarative policy composes scheduling, autonomous proposal, online updates,
# meta-optimization, active-learning labels, and canary-gated promotion/rollback.
controller = app.self_improvement(SelfImprovementPolicy(), dataset=golden)
async for ev in controller.astream():
    print(ev.phase, ev.reason)   # observe → proposal → meta → canary → promote/rollback

# Promote a candidate live only if it clears a no-regression canary verdict.
app.deploy(candidate, dataset=golden)

# On-policy reinforcement from verifiable rewards: improve a policy from the
# task-success oracle and benchmark scorers, behind a KL clamp + no-regression gate.
from vincio.optimize import OracleReward, RewardModel
result = app.learn(tasks, reward=RewardModel([OracleReward()]))
result.promoted, result.reward_delta, result.kl_to_reference
```

### Interoperate: MCP, A2A, Skills

```python
# Consume an MCP server — its tools run through the permissioned, sandboxed,
# audited runtime; its resources become cited evidence.
app.add_mcp_server("weather", command=["python", "weather_server.py"])

# Load portable SKILL.md procedural knowledge (progressive disclosure).
app.add_skill("skills/pdf-invoice")

# Expose your app over the protocols — one ContextApp, both consumer and provider.
mcp_server = app.serve_mcp()                       # serve tools/resources/prompts
a2a_server = app.serve_a2a(crew, name="research")  # Agent Card + task lifecycle

# One portable reasoning knob across providers (thinking tokens are billed).
from vincio.core.types import RunConfig
app.run("Plan the migration", config=RunConfig(reasoning_effort="high"))
```

## Features

Vincio is organized into composable subsystems. Use the high-level `ContextApp` runtime, or reach
for any engine directly. Everything below is implemented, tested offline, and documented.

| Subsystem | What it does |
|---|---|
| **Prompt compiler** | Typed prompt ASTs with `${variables}`, lint rules, cache-aware stable-prefix layout, versioning, hashing, diffing, variant generation. |
| **Context compiler** | Scores every candidate (relevance, novelty, authority, freshness, provenance, token cost, leakage risk), deduplicates, resolves conflicts, compresses, and packs to a token budget — with an *excluded-context report* explaining every omission. Image, table, and text evidence are first-class candidates in **one scored packet**, with modality-aware token cost and slim packets that `materialize()` cross-process from a content-addressed evidence store. |
| **Retrieval (RAG)** | BM25 + dense + learned-sparse (SPLADE-style) + late-interaction (ColBERT-style MaxSim with PLAID-style compression) fused in one weighted RRF; query understanding (HyDE, multi-query, decomposition, step-back); sentence-window, parent-document/auto-merging, and contextual chunking; GraphRAG with community summaries and global/local routing; live indexes (upsert/TTL/migrations); entity-graph, multi-hop, and reasoning retrieval; Matryoshka (MRL) dimension truncation, contextual (Voyage context-3) and unified text+image multimodal (Cohere v4 / Voyage) embedders behind one `build_embedder`; a structured **`FilterSpec`** (`eq`/`in`/`range`/`and`/`or`) that pushes down to each backend's native filter with tenant scope enforced in the engine. |
| **Memory** | Layered (session → episodic → semantic → tenant → graph) with a guarded write pipeline, confidence decay, contradiction resolution, and privacy scoping; `remember`/`recall` personalization over user/agent/session/team scopes, hybrid vector+graph recall, episodic→semantic consolidation with provenance, TTL + importance-weighted retention, audited GDPR-style edit/forget/export/erase, and a CI-gated memory eval harness. Memory is **bi-temporal** (`valid_from`/`valid_to` + as-of recall, `correct()` that preserves history) with **per-memory ACLs** and a `TEAM` scope. |
| **Tools** | Permissioned registry (RBAC scopes + ABAC rules), schema derivation from type hints, a resource-limited sandbox (timeout, output caps, scrubbed env, POSIX CPU/memory/fd `setrlimit`), reliability scoring, idempotent write-action guardrails with approval callbacks; **computer-use** and provider-native **hosted tools** behind a pluggable `IsolationBackend` (container / microVM / gVisor / WASM). |
| **Agents** | Bounded DAG execution with planners (direct / static / dynamic / ReAct / plan-and-execute / **hierarchical HTN**), critics, validators, human gates, and hard budget enforcement; **in-place plan repair** (re-bind / substitute / reorder / drop on a tool failure, contradiction, or budget shock — recorded as a trajectory event, not a restart); **cost-aware action selection** that reads `ModelRegistry` pricing and the live budget to spend the cheapest capable model per step, escalating only on low confidence; a budgeted, citation-gated **deep-research agent** (`app.research`) that loops search → read → reflect → verify → synthesize into a cited report; a self-editing **memory OS** (`app.enable_memory_os`) exposing memory ops as audited tools with a context-pressure pager. |
| **Orchestration** | Multi-agent crews — roles, delegation, and a shared versioned blackboard — with per-agent budget shares and guaranteed termination; durable stateful graphs with checkpoints on your storage, resume, edit-and-resume, and time-travel forks; first-class human-in-the-loop interrupts; a declarative `compose`/pipe API with streaming node events. A **distributed durable-execution backend** runs the same graph/workflow across a worker pool with a TTL lease + checkpoint-version CAS so two workers never double-execute a step, with BSP parallel super-steps, a `Send` map-reduce primitive, and LangGraph / OpenAI Agents SDK / Ray / Temporal export adapters. A **work-stealing sub-graph scheduler** runs independent sub-graphs concurrently across the pool under one fair-share budget with an SLA deadline that returns a partial result, and **durable timers** (`sleep_until` / `wait_for_event`) pause a graph for a delay, a webhook, or an approval — surviving a restart without holding a worker. |
| **Workflows** | Deterministic DAGs with retries, branching, parallelism, compensation, and approval gates that pause the run and resume without re-executing finished steps. |
| **Structured output** | Pydantic output contracts, provider-native constrained decoding with strict schema sanitization (robust-parser fallback everywhere else), streaming validation with mid-stream early abort, DSPy-style typed signatures (`Signature` / `Predict`) that feed the optimizer, bounded self-correcting loops with cost ceilings, multi-schema routing by task or content, and **principled repair that fixes structure only — never invents facts**. |
| **Evaluation** | Golden JSONL datasets, 30+ task / grounding / quality / safety / conversational / trajectory & tool-use / retrieval / operational metrics, deterministic / model / G-Eval judges with calibration, synthetic dataset generation with provenance, red-teaming judged by the security detectors, experiment tracking with statistical significance, regression gates, and baseline-diff reports — plus a `pytest` plugin (`assert_eval` / `assert_grounded`, packet/trace snapshots). A **stateful-environment harness** (`Environment` reset/step/observe/verify with a task-success oracle) scores the verifiable end state of a mutable world, and nine **agentic benchmark adapters** (SWE-bench Verified, τ-bench/τ²-bench, GAIA, WebArena, BFCL, AgentBench, ToolBench, LiveCodeBench, MMLU-Pro) run inside VincioBench behind one contract — replayed offline or solved live, scored either way by the benchmark's own scorer. A **retrieval-eval harness** (recall@k / nDCG / MRR / context-precision) records versioned index-regression artifacts keyed on `(embedder, chunker, corpus hash)` and gates a regression on the same significance test as a model swap. When a gate does regress, a **`CausalAttributor`** attributes the drop to the component that caused it (prompt / retrieval / model / budget) by Shapley counterfactual replay — so a failure names its cause instead of just its score — and an **`AdaptiveSampler`** spends the eval budget where the variance is, converging a noisy gate on the same verdict as the exhaustive run for far fewer samples. |
| **Agentic eval & continuous quality** | Score *how* a run reached its answer: trajectory & tool-use metrics over a `Trajectory` projected from any crew / graph / trace (no re-instrumentation); a deterministic multi-turn `Simulator`; **online evaluation** on a sampled slice of live traffic (restart-safe, worker-aggregatable, off the hot path); **drift detection** — score, embedding-distribution, KS / PSI / MMD distributional, and a streaming CUSUM changepoint — raising a `drift.detected` event; human annotation with Cohen's-κ judge calibration; **judge ensembles** (`JudgeEnsemble`) that turn a panel's disagreement into an uncertainty signal — flagging split cases for review and earning CI-gating weight only once the panel's κ against human labels clears the bar; production A/B with cost + significance per variant. Every metric doubles as a runtime guardrail (`add_metric_rail`) and optimizer fitness term. |
| **Optimization & the closed loop** | One continuous, reproducible cycle — trace → dataset → eval → optimize → promote (`ImprovementLoop` / `vincio loop run`): production traces become datasets, the gated optimizer searches, and the winner lands in the prompt registry tagged, eval-linked, applied live, and audited — with safety-gated promotion that blocks any candidate regressing schema validity or safety. Plus grounded auto-memory from runs, eval-driven retrieval feedback, cost/quality Pareto frontiers with knee-point selection, and learned per-task budget allocation. |
| **Self-improvement** | One declarative **`SelfImprovementPolicy`** composes scheduling, autonomous proposal, online updates, meta-optimization (learned fitness weights + successive-halving), active-learning label acquisition, and canary/rollback; `app.self_improvement(policy).astream()` drives the organs as one streaming controller — `observe → proposal → meta → reeval → canary → promote/rollback`. **`app.deploy`** promotes a prompt/policy live only on a no-regression canary verdict — an offline gated comparison *or* a live-traffic canary that ramps a fraction of real runs and auto-rolls-back a regression. Every promotion passes the same significance + safety + golden gates. |
| **Reflective optimization & the flywheel** | A GEPA-style `ReflectiveOptimizer` that reads eval failures, reflects on why a prompt lost, and proposes targeted edits, evolving a Pareto frontier under a hard rollout budget (plus MIPRO joint instruction+example proposal); a **distillation flywheel** (`app.export_training_set` / `vincio distill`) that curates grounded production traces into provider-ready fine-tuning JSONL and gates a cheaper student into the routing cascade only when it holds quality, with executed fine-tune jobs (OpenAI/Gemini/Anthropic); **learned prompt compression** (`LLMLinguaCompressor`) as a faithfulness-gated compiler pass; and reflective calibration of the optimizer's own LLM judge against κ-validated labels. |
| **On-policy reinforcement (RLVR)** | The learning loop closed on a *policy*, not just a prompt. A **`RewardModel`** turns the signals the platform already computes — the stateful-environment task-success oracle, the nine benchmark scorers, and a judge ensemble whose **disagreement down-weights itself** — into a verifiable reward; a **`TrajectoryAdvantage`** assigns step-level credit by Shapley counterfactual replay (the same kernel as causal regression attribution); and a GRPO-style **`TrajectoryOptimizer`** (**`app.learn`**) runs a group-relative update behind a **KL-to-reference clamp** and a **monotonic no-regression gate** — the served policy never regresses the baseline reward — emitting a fine-tune job through the existing flywheel under the same canary verdict a prompt deploy produces. The offline path runs a deterministic mock policy so the optimizer's math is fully tested without a GPU. |
| **Observability** | Every run yields a full trace span tree with sessions, threaded runs, user feedback, and eval scores on spans; JSONL and OpenTelemetry exporters (GenAI semantic conventions, including agentic spans); a local viewer (TUI + self-contained static HTML export + visual trace diff); traces become eval datasets in one command; a versioned prompt registry with tags, diffs, rollback, and eval links; per-run cost tracking. A **served, self-hosted observability & alerting plane** — an indexed trace/cost store with rollups, a stdlib dashboard (`serve_viewer`), and a rule engine (threshold / EWMA-anomaly / SRE burn-rate) over webhook/Slack/PagerDuty/Prometheus sinks — replaces O(n) JSONL scans, with prompt/completion content off by default at the export boundary. |
| **Security** | Deterministic PII / secret detection and redaction (with non-English locale packs for France/Germany/Spain/India/Singapore/Brazil/UK), prompt-injection defense, authority/provenance RAG-poisoning detection on retrieved evidence, and **provable injection containment** — a typed `TrustLabel` and `TaintedValue` that propagate taint from untrusted documents, a `DualPlaneExecutor` whose privileged planner sees only schema-validated extractions, and unforgeable `CapabilityToken`s minted from the user's request so an injected instruction provably cannot escalate to an unauthorized side effect; programmable input/output rails (topic / format / safety / custom) in the deterministic policy engine, RBAC / ABAC, tenant isolation, and a hash-chained, signed Merkle-checkpointed audit log with offline tamper verification (`vincio audit verify`) — all documented in a [threat model](docs/security/threat-model.md) and shipped with SBOM + SLSA provenance attestations. |
| **Governance & compliance** | Evidence generated from the running system, as files you own: machine-readable **model & system cards**, a **compliance coverage matrix** across OWASP LLM Top 10 (2025) / OWASP Agentic / NIST AI RMF / MITRE ATLAS / ISO IEC 42001 backed by red-team and eval evidence, an **AI-BOM** with SHA-256 model-hash verification, EU AI Act **synthetic-content marking** (media-aware, optionally signed + verifiable), an **EU AI Act conformity pack** (risk-tier + cited Annex IV + Article 27 FRIA), **provable erasure** (`app.erase_source` emits a signed, content-bound `ErasureProof` over the exact removed-id set across indexes, memory, and generated artifacts, anchored to the audit chain's Merkle root and verifiable offline), a **`ConsentLedger`** binding data to a GDPR purpose + lawful basis that access and recall enforce, **data lineage**, **data-residency-aware** egress refusal, and the non-English **token tax** surfaced per language/tenant — see the [governance guide](docs/guides/governance.md). |
| **Documents & media out (generation)** | The deliverable comes out under the same guarantees as text in. A `DocumentBuilder` renders a *validated* result into **cited DOCX/PDF/PPTX/HTML/Markdown**, structurally validated against a `DocumentContract` with formatting-only repair, plus template/form filling and tracked-change **redlines**. A `CitedReportBuilder` resolves `[E1]` markers to **footnotes + a bibliography** with sentence-level citation coverage and per-claim entailment. **Image generation/editing** and **TTS** are first-class output modalities where every asset is **C2PA-provenance-stamped, budget-metered, and audited**. Inputs get richer too: OCR auto-fallback for scanned PDFs, audio transcript ingestion, new-format loaders (PPTX/EPUB/RTF/ODT/Parquet/mbox), a real-parser HTML/JSON/YAML path, and offline forms/KYC extraction — see the [generation guide](docs/guides/generate-documents.md). |
| **Storage** | Pluggable metadata (in-memory / SQLite / Postgres, async-first with a psycopg3 pool), blob, analytics (DuckDB), vector (Qdrant / pgvector / Chroma / Pinecone / LanceDB / Weaviate / Milvus / Elasticsearch / OpenSearch / Vespa behind one `build_vector_index` factory), and graph (Neo4j) backends. Redis-backed shared state + a first-class `vincio serve` keep multi-worker deployments coherent. |
| **Providers** | OpenAI (Chat Completions + Responses API), Anthropic, Google, Mistral, any OpenAI-compatible endpoint (with hosted-gateway presets: groq, together, fireworks, openrouter, deepseek, perplexity, xai, nvidia), enterprise endpoints (**AWS Bedrock** SigV4, **Google Vertex**, **Azure OpenAI**) behind a pluggable `AuthStrategy` in the same registry, and a deterministic offline mock — all async-first with sync wrappers, pooled transport, retries, and **capability-aware failover**, with in-flight request coalescing. A data-driven `ModelRegistry` (capabilities, pricing, lifecycle) is the single source the cost table, capability guards, and rotation read from. Unified reasoning control (`reasoning_effort` / thinking budget) maps across OpenAI/Anthropic/Gemini, with thinking tokens recorded and billed. Opt-in **voice/realtime** sessions (OpenAI Realtime, Gemini Live) via `vincio.realtime`. Batteries-included **local neural models** (fastembed, SPLADE, ColBERT, a cross-encoder, and a llama.cpp **GGUF** in-process provider) give air-gapped/edge deployments true offline inference behind the same interfaces. |
| **Protocols & interoperability** | Speaks the standards in-process: **MCP** client *and* server (stdio / Streamable HTTP / in-process) — MCP tools run through the permissioned, sandboxed, audited, budgeted runtime; resources become cited evidence. **A2A** agent-to-agent — expose a crew/graph as an Agent Card + task lifecycle, and reach remote agents as bounded, traced crew delegates. **Agent Skills** — `SKILL.md` with progressive disclosure, bundled scripts as sandboxed tools. A governed **agent fabric** (`AgentDirectory` over A2A Agent Cards, AGNTCY/ACP, and the MCP Registry) resolves agents behind an `AllowListGate`, every resolution an audited access decision; the same gate governs a signed **community pack & skill registry** and a one-call **MCP-server marketplace bridge** that lands a discovered server's tools in the permissioned runtime. **Generative UI** streams a run as AG-UI events so an interactive frontend inherits the run's provenance, budget, and audit. |
| **Performance** | End-to-end streaming (`astream` + SSE) with incremental partial-JSON output and genuine provider token deltas, concurrent retrieval/memory/tool fan-out with cancellation propagation and hard latency deadlines, content-addressed compile/chunk/embedding caches, and zero-copy (slim) context packets. The compile hot path is **single-pass and allocation-light**: a vectorized candidate scorer (NumPy-optional, pure-Python fallback), a compiled-prompt render program and a warm candidate arena that reuse the stable prefix and prepared candidate set so a warm compile is dominated by scoring not allocation, streaming-first compilation that emits the prefix before scoring, speculative retrieval prefetch that warms the query embedding from the task classification, and a per-app resident-memory budget held by slim packets and evidence eviction. Quantized two-stage retrieval, a sub-millisecond warm compile, and a footprint regression gate — all held by CI-gated VincioBench performance budgets. |
| **Cost & reliability (FinOps)** | Production-traffic resilience in-process, not a proxy hop: **batch execution** at ~50% cost (OpenAI Batch + Anthropic Message Batches + Google/Vertex batch), **circuit breaking** + **health-aware failover** and **key pooling** with RPM+TPM rate limiting, **runtime model cascades** (start cheap, escalate on low confidence), **cost attribution** by tenant/feature with enforced **budget SLOs** (cap / degrade / queue-to-batch + `cost.anomaly`), provider-aware **prompt caching** with TTL choice and cache-hit telemetry, and **incremental** (content-hash) + **sharded** indexing at scale. |
| **Provider/model rotation & swap regression** | The migration safety net for the riskiest production change. A registry-backed **`Router`** picks the cheapest / fastest / least-busy *capable* model per request. A **`SwapGate`** replays golden traces and runs an eval + cost + latency + behavioral diff with statistical significance, PASS/FAILing the swap; **model-swap regression** swaps *only* the model and reports per-metric significance, the cost/latency trade, and the worst-regressed slices, with flake quarantine. A **shadow provider** and a capped **canary** qualify a candidate on live traffic with automatic rollback, and a **lifecycle watcher** proposes migrations off deprecated models. |
| **Connectors** | Pluggable data connectors — web, GitHub, SQL, S3, GCS, Notion, Confluence, Slack, **Jira, Linear, Google Drive, SharePoint, Salesforce, Zendesk, BigQuery, Snowflake**, plus custom via `register_connector` — feeding the document engine with full provenance: `app.add_source("kb", connector=connect("github", repo="acme/handbook"))`. The REST connectors ride the core `httpx` dependency; all accept an injected client so they round-trip offline. |
| **Plugins & ecosystem** | An entry-point **plugin contract** (`vincio plugins list`) so third-party providers, embedders, stores, connectors, chunkers, rerankers, judges, metrics, and packs register themselves on install, gated by a versioned plugin API; a signed, allow-list-gated, audited **`CommunityRegistry`** of opt-in domain packs and `SKILL.md` bundles (content-bound SHA-256 + HMAC/Ed25519 signatures, resolution as an audited access decision); and an **MCP-server marketplace bridge** (`app.add_mcp_from_registry`) that discovers a server from a registry, governs reachability, and lands its tools in the permissioned runtime in one call. |
| **Use-case coverage & verticals** | Go from primitives to a working app in one file. **Vertical packs** (`healthcare` / `ediscovery` / `kyc` / `customer_support` / `code_review`) preconfigure retrieval, scoped memory, deterministic rails, domain metrics, a data-residency posture, and a golden eval set in one `use_pack`, on top of the existing pack contract. A higher-level **`Assistant`** layer over `ContextApp` threads turns into a session, carries multi-turn state via memory write-back, and gates write tools behind an approval — a chat product in a few lines. An end-to-end **`VoiceAgent`** wires a realtime session to the deep-research agent, the memory OS, and the rails, so a spoken assistant inherits the same grounding, budget, and audit guarantees. A **cookbook** of task-shaped recipes (contract redlining, incident triage, data-room Q&A, multimodal RAG over slides/PDFs) ships as runnable, offline-gated examples. |
| **Integrations & DX** | LangChain + LlamaIndex + **Haystack + DSPy** interop (`vincio.interop`) for tools, retrievers, loaders, embedders, components, and compiled DSPy modules — both directions, duck-typed `from_*` (no heavy import); hosted rerankers/embedders (Cohere/Jina/Voyage, httpx-only) behind `build_reranker`/`build_embedder`; opt-in domain packs (support, engineering, finance, legal) via `app.use_pack(...)`; `vincio init` templates (rag/agent/eval) with a typed `vincio.yaml` JSON Schema for editor completion; notebook reprs (`enable_rich_reprs`) and an interactive `vincio tui` inspector. |
| **Stability & guarantees** | [Semantic Versioning](https://semver.org/spec/v2.0.0.html) on a frozen public surface (`vincio.__all__`) with a mechanical [deprecation policy](docs/reference/stability.md) (`@deprecated` / `stability_of`); published performance & quality [SLOs](docs/reference/slo.md) held by at-least-as-strict VincioBench budgets; CycloneDX SBOM + SLSA build-provenance attestations on every release. |
| **A trustworthy surface** | The public API is held to the same bar as the internals. Every `VincioError` carries a stable `.code`, a `.remediation` hint, and a `.docs_url` from a completeness-gated, internationalizable [error catalog](docs/reference/errors.md). The package ships [`py.typed`](docs/reference/typing.md) with a CI-enforced `mypy --strict` ladder, and a docstring-coverage gate keeps every public symbol documented. `vincio.yaml` files migrate forward automatically (`vincio config migrate`, in-memory upgrade on load), and `vincio doctor` reports any deprecated API a project still uses — its replacement and removal version straight from `stability_of`. |

Every extension point — providers, metrics, chunkers, rerankers, judges, validators, tools — accepts
your own implementation via a registry.

## Benchmarks

**VincioBench** ships in `benchmarks/` and runs fully offline (deterministic provider + deterministic
metrics) so results are reproducible. Each family compares the Vincio pipeline against a naive
baseline. Representative results on the bundled reference corpus:

| Family | Metric | Vincio | Naive baseline |
|---|---|--:|--:|
| **Context compression** | evidence tokens for the same task | **216** | 1,175 (stuff-everything) |
| | → token reduction | **−81.6%** | — |
| **Output recovery** | malformed model outputs successfully parsed | **5 / 5** | 3 / 5 (`json.loads`) |
| **Security** | prompt-injection detection rate | **100%** | — |
| | injection false-positive rate | **0%** | — |
| | PII coverage | **100%** | — |
| **Containment** | injection escalation rate (adversarial corpus) | **0** | — |
| **Retrieval** | recall@3 / MRR (known-answer corpus) | **1.00 / 1.00** | — |
| | per-mode recall@3 (sparse · late-interaction · PLAID · hybrid_full) | **1.00 each** | — |
| **Memory** | preference recall · contradiction supersede · tenant isolation | **pass** | — |
| **Tools** | runtime overhead, p50 | **0.02 ms** | — |
| **Agents** | adversarial infinite-loop model | **bounded** (budget) | unbounded |
| **Orchestration** | crew over-budget termination · delegation recorded | **pass** | — |
| | graph interrupt→resume and fork-replay vs straight run | **identical state** | — |
| | plan repair recovers a tool failure / contradiction / budget shock | **pass** | restart / dead-end |
| | cost-aware action selection vs always-strong | **−57%** | always-strong |
| | parallel sub-graph speedup over serial (4 workers) | **4.0×** | 1× (serial) |
| | durable timer survives restart → resumes when due | **pass** | timer lost |
| **Evals** | metric agreement on labeled examples | **100%** | — |
| | red-team detector coverage · guarded attack success | **100% · 0%** | naive target: 85% attacks succeed |
| | A/B significance (real shift detected / null ignored) | **pass** | — |
| **Reliability** | invalid output detected mid-stream → tokens saved | **98%** | 0% (validate at end) |
| | self-correction recovery rate (bounded cycles) | **3 / 3** | — |
| | rail catch rate · false positives on clean text | **100% · 0** | — |
| | schema routing / classification accuracy | **100%** | — |
| **Closed loop** | loop promotion fires · deterministic · gates block regressions | **pass** | — |
| | grounded facts written · ungrounded excluded | **pass** | — |
| | Pareto front excludes dominated · knee balanced · learned budgets promote | **pass** | — |
| **Cost & reliability** | prompt-cache hit rate · cost-attribution accuracy | **72% · 100%** | — |
| | cascade savings vs always-strong (escalate on low confidence) | **−70%** | — |
| | canary auto-rolls-back under load → serves primary after | **pass** | serves the regression |
| **Rotation & swap** | router picks cheapest capable · capability guard skips incapable failover | **pass** | routes to incapable/pricier |
| | swap gate blocks a regression · passes a safe swap (significance) | **pass** | swap on one noisy run |
| **Governance** | card/AI-BOM completeness · framework-mapping coverage | **pass · 79%** | — |
| | erasure correctness (removed = lineage) · audited · proof verifies | **pass** | — |
| | multilingual PII recall · RAG-poisoning detection (FP rate) | **100% · 100% (0%)** | English-only |

> **Honest by design.** These numbers come from a small, synthetic offline corpus and are meant to
> demonstrate the mechanisms, not to be quoted as universal gains. The context-compression
> hypothesis (a 20–40% reduction target) is *measured* per run, and VincioBench reports whether it
> was met on your data. Run `python benchmarks/vinciobench.py` against your own corpus — and trust
> only what that prints. See [`benchmarks/README.md`](benchmarks/README.md).

## How Vincio compares

Each ecosystem below is broad and capable in its own focus area. The table reflects **built-in,
in-library** capabilities — not what is reachable by bolting on a separate product or SaaS.

| Capability | **Vincio** | LangChain | LlamaIndex | DSPy | Ragas |
|---|:--:|:--:|:--:|:--:|:--:|
| Scored, budgeted **context compiler** | ✅ | ➖ | ➖ | ❌ | ❌ |
| Typed prompt **AST + lint + cache layout** | ✅ | ❌ | ❌ | ➖ | ❌ |
| Hybrid (BM25 + dense) **RAG** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Sparse + late-interaction + GraphRAG** in one fusion | ✅ | ➖ | ➖ | ❌ | ❌ |
| Layered **memory** (decay, conflicts, scopes, bi-temporal) | ✅ | ➖ | ➖ | ❌ | ❌ |
| **Permissioned** tool registry (RBAC/ABAC) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Bounded **agents** + deterministic workflows | ✅ | ✅ | ➖ | ➖ | ❌ |
| **Durable graphs** (checkpoint / resume / time-travel) + bounded crews | ✅ | ➖ | ❌ | ❌ | ❌ |
| Structured output + **structure-only repair** | ✅ | ➖ | ➖ | ✅ | ❌ |
| Built-in **evals + CI gates** | ✅ | ➖ | ➖ | ➖ | ✅ |
| **pytest assertions + red-teaming + synthetic data** | ✅ | ❌ | ❌ | ❌ | ➖ |
| Eval-driven **optimization** (gated promotion) | ✅ | ❌ | ❌ | ✅ | ❌ |
| Native **tracing + cost**, no account needed | ✅ | ➖ | ➖ | ❌ | ❌ |
| **Sessions, feedback, prompt registry, trace viewer** in-process | ✅ | ➖ | ❌ | ❌ | ❌ |
| **Deterministic security** (PII / injection / audit) | ✅ | ❌ | ❌ | ❌ | ❌ |
| **MCP** client *and* server + **A2A** + **Agent Skills** | ✅ | ➖ | ➖ | ➖ | ❌ |
| **In-process FinOps**: batch · circuit-break · cascades · cost attribution + budgets | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Capability-aware rotation + gated swap regression** | ✅ | ➖ | ❌ | ❌ | ❌ |
| **Governance evidence**: cards · framework mapping · AI-BOM · provable erasure · residency | ✅ | ❌ | ❌ | ❌ | ❌ |

<sub>✅ first-class in-library · ➖ partial or via a separate add-on/SaaS · ❌ not a focus. Ecosystems
evolve. Vincio is built to *interoperate* — it speaks MCP (client *and* server), A2A, and Agent Skills
in-process, `vincio.interop` brings LangChain, LlamaIndex, Haystack, and DSPy tools, retrievers,
loaders, embedders, components, and compiled modules in (and hands Vincio's back), first-party
connectors and an entry-point plugin system meet your data and tools where they live, and you can
point at any OpenAI-compatible model and the vector store you already run. See the [migration guides](docs/guides/migrate-from-langchain.md), the
[integrations guide](docs/guides/integrations.md), and the in-depth write-ups in
[`docs/comparisons/`](docs/comparisons).</sub>

## Use cases

| You want to… | Reach for | Example |
|---|---|---|
| Classify and route support tickets into typed labels | typed output | [`01_support_triage.py`](examples/01_support_triage.py) |
| Answer questions over your docs with real citations | hybrid RAG + grounding policy | [`02_document_qa.py`](examples/02_document_qa.py) |
| Review contracts clause-by-clause | end-to-end context app | [`03_contract_review.py`](examples/03_contract_review.py) |
| Extract structured fields from invoices | structured extraction + F1 eval | [`04_invoice_extraction.py`](examples/04_invoice_extraction.py) |
| Build a research agent with bounded budgets | ReAct agent + tools | [`05_research_agent.py`](examples/05_research_agent.py) |
| Automate a CRM agent with approval-gated writes | memory + permissioned tools | [`06_crm_agent.py`](examples/06_crm_agent.py) |
| Ask questions over a codebase | code-aware chunking + import graph | [`07_codebase_qa.py`](examples/07_codebase_qa.py) |
| Analyze spreadsheets with schema awareness | table chunking + quality checks | [`08_spreadsheet_analysis.py`](examples/08_spreadsheet_analysis.py) |
| Gate quality in CI | datasets, gates, baseline diff | [`09_eval_pipeline.py`](examples/09_eval_pipeline.py) |
| Tune prompts/context against an eval suite | optimization + gated promotion | [`10_optimization_run.py`](examples/10_optimization_run.py) |
| Stream answers token-by-token through the full pipeline | `astream` + partial-JSON + compile caches | [`11_streaming_performance.py`](examples/11_streaming_performance.py) |
| Push retrieval quality with the full toolkit | sparse+late-interaction fusion, HyDE, auto-merge, GraphRAG, connectors | [`12_advanced_rag.py`](examples/12_advanced_rag.py) |
| Personalize an app with governed memory | scoped remember/recall, consolidation, hygiene, memory evals | [`13_memory_personalization.py`](examples/13_memory_personalization.py) |
| Evaluate, test, and observe without a platform | quality metrics, synthetic data, red-teaming, experiments, prompt registry, sessions + trace viewer | [`14_evaluation_observability.py`](examples/14_evaluation_observability.py) |
| Run a multi-agent team with roles and delegation | crews + shared blackboard + budget guarantees | [`15_multi_agent_crew.py`](examples/15_multi_agent_crew.py) |
| Build an interruptible, auditable, resumable process | durable graphs + human gates + time-travel | [`16_durable_graph.py`](examples/16_durable_graph.py) |
| Guarantee output shape and guard every generation | signatures, constrained decoding, streaming validation, rails, self-correction, schema routing | [`17_reliable_structured_output.py`](examples/17_reliable_structured_output.py) |
| Improve the app from its own production traffic | the closed loop: traces→dataset→eval→optimize→promote, auto-memory, retrieval feedback, Pareto, learned budgets | [`18_closed_loop.py`](examples/18_closed_loop.py) |
| Reuse LangChain/LlamaIndex assets and any OpenAI-compatible model | framework interop + provider/vector-store breadth | [`19_framework_interop.py`](examples/19_framework_interop.py) |
| Configure an app for a domain in one line | opt-in domain packs (support/engineering/finance/legal) | [`20_domain_pack.py`](examples/20_domain_pack.py) |
| Govern PII, injection, access, and audit integrity | deterministic security primitives + tamper-evident audit | [`21_security_governance.py`](examples/21_security_governance.py) |
| Use MCP servers as tools/resources, or expose your app as one | MCP client + server | [`22_mcp_tools_and_resources.py`](examples/22_mcp_tools_and_resources.py) |
| Expose a crew over A2A and delegate across vendor agents | A2A agent card + task lifecycle + remote delegate | [`23_a2a_delegation.py`](examples/23_a2a_delegation.py) |
| Drop in portable `SKILL.md` knowledge with budgeting | Agent Skills + progressive disclosure | [`24_agent_skills.py`](examples/24_agent_skills.py) |
| Control reasoning effort across providers with honest cost | unified reasoning control + Responses API | [`25_reasoning_control.py`](examples/25_reasoning_control.py) |
| Score agents over their trajectory and live traffic | trajectory & tool-use metrics, multi-turn simulator, online eval + drift, annotation, A/B | [`26_agentic_eval.py`](examples/26_agentic_eval.py) |
| Survive outages and account for every dollar at scale | batch execution, circuit breaking + failover, key pooling, model cascades, cost attribution + budgets, prompt caching, sharded indexing | [`27_cost_and_reliability.py`](examples/27_cost_and_reliability.py) |
| Optimize prompts reflectively and distill traces into a cheaper model | GEPA/MIPRO reflective optimizer, distillation flywheel, learned compression, optimizer-judge calibration | [`28_reflective_optimization.py`](examples/28_reflective_optimization.py) |
| Shrink embeddings, retrieve across text+image, and add stores | Matryoshka truncation, contextual & multimodal embedders, new vector stores, layout-aware extraction, voice/realtime | [`29_multimodal_retrieval.py`](examples/29_multimodal_retrieval.py) |
| Generate compliance evidence and satisfy a data-erasure request | model/system cards, framework mapping, AI-BOM, lineage + erasure, residency, multilingual PII | [`30_governance_compliance.py`](examples/30_governance_compliance.py) |
| Enforce the honest, fast spine | hard budgets, the `ModelRegistry`, semantic scoring, cancellation, significance-gated promotion + replay | [`31_honest_fast_spine.py`](examples/31_honest_fast_spine.py) |
| Rotate providers/models safely | router, `SwapGate`, swap regression with quarantine, shadow + canary, lifecycle migrations | [`32_swap_regression.py`](examples/32_swap_regression.py) |
| Let documents and images flow out under the same guarantees | `DocumentBuilder` + `DocumentContract`, `CitedReportBuilder`, redlines, image/TTS with C2PA, richer inputs, EU AI Act pack | [`33_documents_and_media_out.py`](examples/33_documents_and_media_out.py) |
| Run a self-improving loop with agentic capabilities | online controller, GEPA reflector, experiment proposer, golden non-regression guard, deep-research agent, memory OS, computer-use + isolation | [`34_self_improving_loop_and_agents.py`](examples/34_self_improving_loop_and_agents.py) |
| Work with the multimodal-native packet and capability facades | facades, multimodal packet + cross-process materialize, structured `FilterSpec` pushdown + tenant scope, typed event catalog, enterprise endpoints, egress DLP + signed audit chain | [`35_multimodal_packet_and_facades.py`](examples/35_multimodal_packet_and_facades.py) |
| Scale out across workers, train a cheaper model, and serve a dashboard | distributed durable execution (lease/CAS + worker pool + `Send` map-reduce), swap-gated distillation, served observability + burn-rate alerting, quantized two-stage retrieval, in-process GGUF | [`36_distributed_scale_and_finetune.py`](examples/36_distributed_scale_and_finetune.py) |
| Score on the leaderboards, govern a fabric, and stream a UI | stateful-environment eval with a task-success oracle, the nine benchmark adapters, retrieval-eval + index regression, the `AgentDirectory` under an `AllowListGate`, AG-UI streaming | [`37_benchmarks_and_agent_fabric.py`](examples/37_benchmarks_and_agent_fabric.py) |
| Run continual self-improvement and prove a data-erasure request | one `SelfImprovementPolicy` (streaming proposal→meta→canary→promote/rollback), canary-gated `app.deploy`, signed & verifiable `ErasureProof`, a GDPR `ConsentLedger`, bi-temporal memory | [`38_self_improvement_and_erasure.py`](examples/38_self_improvement_and_erasure.py) |
| Plan deeper, recover from failure, and schedule fairly at scale | HTN hierarchical planning, in-place plan repair, cost-aware action selection, parallel sub-graph scheduling, durable timers | [`40_orchestrator_planner_depth.py`](examples/40_orchestrator_planner_depth.py) |
| Meet teams where their data and tools live | first-party connectors (Jira/Linear/Drive/SharePoint/Salesforce/Zendesk/BigQuery/Snowflake), the entry-point plugin system, a signed community pack/skill registry, Haystack + DSPy interop, the MCP-server marketplace bridge | [`41_ecosystem_and_integration.py`](examples/41_ecosystem_and_integration.py) |
| Configure a regulated domain in one line | full-stack vertical packs (healthcare/e-discovery/KYC-AML/support/code-review) with retrieval, memory, rails, metrics, residency, and a golden set | [`42_vertical_packs.py`](examples/42_vertical_packs.py) |
| Build a multi-turn chat product | the `Assistant` — session threading, memory write-back, tool approvals | [`43_assistant.py`](examples/43_assistant.py) |
| Ship a grounded, guarded voice assistant | the `VoiceAgent` — realtime wired to deep research, the memory OS, and rails | [`44_voice_agent.py`](examples/44_voice_agent.py) |
| Follow a task-shaped recipe | the cookbook — contract redlining, incident triage, data-room Q&A, multimodal RAG over slides/PDFs | [`45`](examples/45_recipe_contract_redlining.py) · [`46`](examples/46_recipe_incident_triage.py) · [`47`](examples/47_recipe_data_room_qa.py) · [`48`](examples/48_recipe_multimodal_rag.py) |

All examples in [`examples/`](examples) run **fully offline** with no API keys. Point them at a real
model with environment variables:

```bash
export VINCIO_PROVIDER=openai VINCIO_MODEL=gpt-5.2-mini OPENAI_API_KEY=sk-...
cd examples && python 02_document_qa.py
```

## Command line

```bash
vincio init my-project --template rag  # scaffold config + app + golden set (minimal|rag|agent|eval)
vincio config schema --output vincio.schema.json  # typed JSON Schema for editor completion
vincio config migrate            # upgrade vincio.yaml to the current schema (--check for CI)
vincio doctor                    # report deprecated-API usage and config schema drift
vincio packs list                # opt-in domain packs (support/engineering/finance/legal)
vincio tui                       # interactive inspector for runs, traces, and memory
vincio run app.py --input "..."  # run an app
vincio eval run golden.jsonl     # run an eval suite (with CI gates and baseline compare)
vincio eval dataset golden.jsonl --min-feedback 0.5  # curate traces into a dataset
vincio prompt lint prompts/      # lint prompt specs
vincio prompt push prompts/support.yaml --tag production  # version a prompt
vincio trace view trace_123      # TUI trace tree with scores + feedback
vincio trace export trace_123    # self-contained static HTML (also --session)
vincio trace diff a b --html diff.html  # visual side-by-side diff
vincio optimize run --target groundedness
vincio optimize reflective --app app.py --dataset golden.jsonl  # GEPA-style reflective optimization
vincio loop run --app app.py --min-feedback 0.5 --gate groundedness=">= 0.8"  # one closed-loop cycle
vincio distill --traces-dir .vincio/traces --output train.jsonl  # grounded fine-tuning JSONL
vincio index build ./docs        # build a retrieval index
vincio memory recall "answer style" --user u1  # scored hybrid recall
vincio audit verify              # verify the audit-log hash chain offline
vincio mcp serve app.py          # expose an app as an MCP server (stdio)
vincio serve --app app.py        # launch the HTTP API (health/readiness/metrics)
```

A FastAPI server (API-key + JWT auth, real-token SSE streaming, `/v1/health/ready`
and Prometheus `/v1/metrics`) is launched with `vincio serve --app app.py` or built
with `from vincio.server import create_app` — see [`docs/reference/api.md`](docs/reference/api.md).
For horizontal scale, point your process manager at it and configure `server.redis_url`
so rate-limit and idempotency state stays coherent across workers.

## Architecture

```text
                         ┌──────────────────────────────────────────────┐
   user input  ─────────▶│  Input engine   normalize · classify · scope  │
                         └───────────────┬──────────────────────────────┘
                                         ▼
        ┌──────────────┐        ┌────────────────┐        ┌──────────────┐
        │   Memory     │───────▶│    CONTEXT     │◀───────│  Retrieval   │
        │  L0…L5       │        │   COMPILER     │        │  hybrid RAG  │
        └──────────────┘        │ score·dedupe·  │        └──────────────┘
        ┌──────────────┐        │ conflict·      │        ┌──────────────┐
        │    Tools     │───────▶│ compress·budget│◀───────│   Prompt     │
        │ permissioned │        └───────┬────────┘        │  compiler    │
        └──────────────┘                ▼                 └──────────────┘
                              ┌────────────────────┐
                              │   Model execution  │   provider-neutral
                              └─────────┬──────────┘
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │ Output validation · Evals · Security ·   │
                    │ Trace + cost · Memory write-back         │
                    └─────────────────────────────────────────┘
```

The public surface is organized into lazy **capability facades** (`app.runs` / `.knowledge` /
`.governance` / `.optimization` / `.serving` / `.training`) over async-first stores and a typed,
versioned event catalog. See [`AGENTS.md`](AGENTS.md) for the package layout and
[`docs/concepts/`](docs/concepts) for a tour of each engine.

## Roadmap

Every subsystem above is implemented, tested offline, documented, and demonstrated by a runnable
example. The public API is frozen under [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with a mechanical [deprecation policy](docs/reference/stability.md); performance and quality targets
are [published as SLOs](docs/reference/slo.md) and gated by VincioBench; the
[threat model](docs/security/threat-model.md) is documented with offline audit-chain verification and
a resource-limited tool sandbox; and releases ship a CycloneDX SBOM with SLSA provenance attestations.

New capabilities are added without breaking working code: each one sits behind a new entry point or an
opt-in extra. Vincio adopts the ecosystem's standards — the MCP, A2A, and Agent Skills protocols, and
the OWASP LLM / OWASP Agentic / NIST AI RMF / MITRE ATLAS governance frameworks — *in your process*; it
never becomes a hosted service to do so.

See **[ROADMAP.md](ROADMAP.md)** for what ships today, what is planned next, and what is intentionally
out of scope.

Vincio is, and stays, a **library**. The building blocks for production operation (audit chain,
retention, tenant isolation, RBAC/ABAC, a server) ship in the package for you to deploy on your own
infrastructure. Hosted services and managed control planes are not part of this project.

## Documentation

- **[Getting started](docs/getting-started.md)** — install, your first app, offline development
- **Concepts** — [context packets](docs/concepts/context-packets.md) ·
  [prompt compiler](docs/concepts/prompt-compiler.md) · [memory](docs/concepts/memory.md) ·
  [retrieval](docs/concepts/retrieval.md) · [agents & workflows](docs/concepts/agents.md) ·
  [evaluation](docs/concepts/evals.md) · [observability](docs/concepts/observability.md)
- **Guides** — [build a RAG app](docs/guides/build-rag-app.md) ·
  [build a chat product (the Assistant)](docs/guides/assistant.md) ·
  [vertical packs](docs/guides/vertical-packs.md) · [cookbook recipes](docs/guides/cookbook.md) ·
  [connect data sources](docs/guides/connectors.md) ·
  [extend with plugins](docs/guides/plugins.md) ·
  [structured output](docs/guides/structured-output.md) ·
  [generate documents & media](docs/guides/generate-documents.md) ·
  [reliability & guardrails](docs/guides/reliability-guardrails.md) ·
  [add tools](docs/guides/add-tools.md) ·
  [orchestrate multi-agent systems](docs/guides/orchestrate-agents.md) ·
  [run evals](docs/guides/run-evals.md) · [test LLM apps](docs/guides/test-llm-apps.md) ·
  [optimize](docs/guides/optimize-context.md) · [close the loop](docs/guides/close-the-loop.md) ·
  [performance & streaming](docs/guides/performance.md) ·
  [cost, reliability & scale](docs/guides/cost-and-reliability.md) ·
  [integrations](docs/guides/integrations.md)
- **Agentic evaluation & continuous quality** —
  [trajectory metrics, simulator, online eval, drift & annotation](docs/guides/agentic-eval.md)
- **Protocols & interoperability** — [MCP client + server](docs/guides/mcp.md) ·
  [A2A agent-to-agent](docs/guides/a2a.md) · [Agent Skills](docs/guides/agent-skills.md) ·
  [reasoning control & Responses API](docs/guides/reasoning.md) ·
  [voice & realtime](docs/guides/realtime.md)
- **Migrating** — coming from [LangChain](docs/guides/migrate-from-langchain.md) ·
  [LlamaIndex](docs/guides/migrate-from-llamaindex.md) ·
  [Ragas](docs/guides/migrate-from-ragas.md) · [Mem0](docs/guides/migrate-from-mem0.md)
- **Reference** — [API](docs/reference/api.md) · [API index](docs/reference/api-generated.md) ·
  [CLI](docs/reference/cli.md) · [config](docs/reference/config.md) ·
  [errors](docs/reference/errors.md) · [typing](docs/reference/typing.md) ·
  [API stability & deprecation policy](docs/reference/stability.md) ·
  [performance & quality SLOs](docs/reference/slo.md)
- **Security & governance** — [threat model](docs/security/threat-model.md) ·
  [security policy](SECURITY.md) · [governance & compliance](docs/guides/governance.md)
- **Comparisons** — [LangChain](docs/comparisons/langchain.md) ·
  [LlamaIndex](docs/comparisons/llamaindex.md) · [RAGatouille](docs/comparisons/ragatouille.md) ·
  [Mem0](docs/comparisons/mem0.md) · [CrewAI](docs/comparisons/crewai.md) ·
  [OpenAI Agents SDK](docs/comparisons/openai-agents-sdk.md) · [DSPy](docs/comparisons/dspy.md) ·
  [Pydantic AI](docs/comparisons/pydantic-ai.md) · [Guardrails AI](docs/comparisons/guardrails.md) ·
  [NeMo Guardrails](docs/comparisons/nemo-guardrails.md) · [Ragas](docs/comparisons/ragas.md) ·
  [LiteLLM / gateways](docs/comparisons/litellm.md)

## Contributing

Contributions are welcome. The test suite runs fully offline and must stay green:

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q     # 1929 tests, no network or API keys required
ruff check vincio/ tests/
mypy vincio
```

See [`AGENTS.md`](AGENTS.md) for the codebase layout and engineering conventions.

## License

[Apache License 2.0](LICENSE) © Vincio Contributors.
