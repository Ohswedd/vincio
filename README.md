<p align="center">
  <img src="assets/banner.svg" alt="Vincio: the context engineering platform for AI applications" width="660">
</p>

<p align="center">
  <em>The scarce resource is not the model. It is the context you feed it.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/vincio/"><img src="https://img.shields.io/badge/vincio-5.5.0-B98B2E" alt="Vincio 5.5.0"></a>
  <a href="https://github.com/Ohswedd/vincio/actions/workflows/ci.yml"><img src="https://github.com/Ohswedd/vincio/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/pypi/pyversions/vincio?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-4C6EF5" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/tests-6421%20passing-2ea44f" alt="6421 tests passing">
  <img src="https://img.shields.io/badge/providers-OpenAI%20%C2%B7%20Anthropic%20%C2%B7%20Google%20%C2%B7%20Mistral%20%C2%B7%20local-B98B2E" alt="Providers: OpenAI, Anthropic, Google, Mistral, local, and OpenAI-compatible gateways">
</p>

---

**Vincio is a Python platform for building AI applications that you can trust in production.**
It takes everything that goes *into* a model (prompts, memory, retrieved evidence, tools, schemas,
and policies) and compiles it into an optimized, validated, observable **context packet**; then it
checks, measures, and traces everything that comes *out*. Named for **Leonardo da Vinci**,
it pairs engineering and craft in equal measure.

<p align="center">
  <img src="assets/pipeline.svg" alt="The run pipeline, governed end to end: raw input, normalize, redact and gate, retrieve and rank, compile context, call model, parse and validate, evaluate and guard, trace and cost, learn; with a governance layer across the whole run (policy and rails, PII redaction, injection defense, audit chain, EU AI Act, residency, cross-org)" width="840">
</p>

Most libraries help you *call* a model. Vincio governs the **boundary** between your application and
the model: what evidence is selected, how it is scored and budgeted, how the result is validated,
and what it cost. It runs on your model of choice across every major provider, with batching,
caching, failover, and cost tracking built in.

> **Try it in 30 seconds, no install:** open the
> [**quickstart notebook in Google Colab**](https://colab.research.google.com/github/Ohswedd/vincio/blob/main/examples/notebooks/01_quickstart.ipynb)
> — one `pip install`, runs offline on the bundled mock provider, no API key required.

<p align="center">
  <img src="assets/why.svg" alt="Why Vincio: offline dev and CI (deterministic mock, no key, no cost); deterministic (security and validation in code, not model output); measured (every run traced and costed, eval-gated); one system (input to output, not a bag of utilities)" width="840">
</p>

<details>
<summary><b>Why you'd reach for it, in one line each</b></summary>

- **Runs on any model.** Call OpenAI, Anthropic, Google, Mistral, a local model, or any OpenAI-compatible gateway through one interface, with batching, caching, failover, and cost tracking built in.
- **Develops and tests offline.** Pass the bundled deterministic `MockProvider` (it emits schema-valid output) and the whole pipeline — retrieval, validation, evals, traces, cost — runs in dev and CI with no network and no key. Flip one env var to point at a real model.
- **Deterministic where it counts.** Security, permissions, and validation are enforced in code, never gated on model output. The same input compiles to the same packet.
- **Measured, not asserted.** Every run is traced and costed; every change can be gated by an eval suite before it ships.
- **One coherent system** from input to output, not a bag of utilities you wire together yourself.

</details>

## Contents

[Install](#install) · [Quickstart](#quickstart) · [The one-line front door](#the-one-line-front-door) ·
[What you can build](#what-you-can-build) · [Providers](#providers--models) · [Features](#features) ·
[Benchmarks](#benchmarks) · [How Vincio compares](#how-vincio-compares) · [Examples](#examples) ·
[CLI](#command-line) · [Architecture](#architecture) · [Docs](#documentation)

## Install

```bash
pip install vincio                  # core (dependency-light: pydantic, httpx, pyyaml, typing-extensions)
pip install "vincio[openai]"        # + a provider (also: anthropic, google, mistral)
pip install "vincio[chroma]"        # + a vector store (also: pinecone, lancedb, pgvector, …)
pip install "vincio[server]"        # + the FastAPI server (vincio serve)
pip install "vincio[all]"           # every optional integration
```

Python 3.11+. Every heavy integration (vector stores, OCR, server, OpenTelemetry, charts, …) is an
opt-in extra; the core stays small and runs offline.

## Quickstart

```python
from vincio import ContextApp

# Configure a provider (the default is OpenAI — set a provider + key, or pass one explicitly).
app = ContextApp(name="docs_qa", provider="openai", model="gpt-4o-mini")
app.add_source("docs", path="./docs", retrieval="hybrid")
app.set_policy("answer_only_from_sources", True)

result = app.run("How do I configure SSO?")
print(result.output)      # the grounded answer
print(result.citations)   # the evidence it actually cited
print(result.trace_id)    # every run produces a full trace
print(result.cost_usd)    # …and a cost
```

**Run it offline — no key, no network.** Pass the bundled deterministic mock; it auto-generates
schema-valid output, so the whole pipeline runs in dev and CI for real:

```python
from vincio.providers import MockProvider
app = ContextApp(name="docs_qa", provider=MockProvider(), model="mock-1")
```

Set `VINCIO_PROVIDER` + the matching key in the environment (or pass `provider=`/`model=`) to point
the same code at OpenAI, Anthropic, Google, Mistral, a local model, or any OpenAI-compatible gateway.

## The one-line front door

For the five jobs you reach for most, the `vincio.tasks` namespace is one expression each — a
task-shaped constructor with sane governed defaults that **lowers to the exact same governed run** as
the verbose builder path (retrieval, grounding, validation, rails, budgets, tracing, and the audit
chain all apply unchanged). `.app` is the escape hatch to every deep method.

```python
from vincio import rag, extractor, tool_agent, evaluation, chat, Flow

rag("./docs").ask("How do I configure SSO?")          # grounded RAG Q&A, cited and eval-scored
extractor(Ticket).extract("I was charged twice")      # typed structured extraction
tool_agent(writes=[create_ticket]).run(task)          # an approval-gated tool agent
evaluation(dataset, gates={"groundedness": ">= 0.8"}).run()   # an offline eval
chat().send("What's my refund window?")               # a multi-turn assistant

# …or thread the whole pipeline fluently — the Vincio answer to LCEL:
Flow(provider=p, model=m).retrieve("./docs").ground().evaluate("groundedness").run(question)
```

These are `@experimental` while their shape settles. See
[`examples/00_one_liners.py`](examples/00_one_liners.py) and the
[ergonomic-surface concept](docs/concepts/ergonomic-surface.md) for how each one-liner maps to the
deep methods it composes.

## What you can build

**Typed output you can rely on**: declare a Pydantic schema, get a validated instance back:

```python
from pydantic import BaseModel
from vincio import ContextApp
from vincio.providers import MockProvider

class Triage(BaseModel):
    label: str
    confidence: float

# (provider=MockProvider() runs this offline; use a real provider in production)
app = ContextApp(name="triage", provider=MockProvider(), model="mock-1", output_schema=Triage)
app.run("The dashboard crashes after login").output.label   # → a validated Triage
```

**Agents with tools, memory, and hard budgets**: permissioned tools, approval-gated writes, and a
loop that cannot run away:

```python
app = ContextApp(name="support", output_schema=RefundDecision)
app.add_memory(scope="user", strategy="semantic")
app.add_tool(lookup_order, permissions=["orders:read"])
app.add_tool(issue_refund, permissions=["refunds:write"], approval_required=True)
app.run("Refund my duplicate charge")
```

**A real backend service** you can copy: `examples/applications/` ships a FastAPI grounded-RAG
service, a ticket-triage API, a structured-extraction service, and a CLI research agent — each
runnable fully offline. See **[Examples](#examples)**.

## Providers & models

Vincio calls real models in production. One interface routes to every major provider, with the
model-operations layer (reasoning control, half-cost batch, caching, failover, cost tracking) built
in. The deterministic mock is a development convenience, not the product: pass it to build and test
the whole pipeline with no key and no cost before you point it at a real model.

<p align="center">
  <img src="assets/providers.svg" alt="Providers and models: one interface over OpenAI, Anthropic, Google, Mistral, local models, and any OpenAI-compatible gateway, plus enterprise auth for Amazon Bedrock, Google Vertex, and Azure OpenAI. Model operations: unified reasoning control, batch at about half cost, prompt caching, circuit breaker and failover, key pool, and per-run cost tracking. With the bundled mock, the whole pipeline runs for dev, tests, and CI." width="840">
</p>

<details>
<summary><b>Providers, model operations, and the mock</b></summary>

- **Providers**: OpenAI, Anthropic, Google (Gemini), Mistral, local models, and any OpenAI-compatible gateway (Groq, Together, Fireworks, OpenRouter, and the like) through one `ModelProvider` interface.
- **Enterprise auth**: Amazon Bedrock, Google Vertex, and Azure OpenAI via pluggable auth strategies (SigV4, service-account, Azure AD / key).
- **Model operations**: unified reasoning/thinking control across providers, batch backends (~50% cost), prompt-cache strategy, a circuit breaker with health-aware failover, a key pool, and a data-driven `ModelRegistry` (capabilities, pricing, lifecycle) that drives capability guards and shadow / canary dispatch. Its shipped catalog prices the current lineup of every provider and is held by a coverage gate, so no current model silently bills $0.
- **The mock**: `MockProvider` is deterministic and emits schema-valid output, so the full pipeline (retrieval, validation, evals, traces, cost) runs offline in CI with no key and no cost. Pass it explicitly for development and tests; use a real provider in production.

```python
# point an app at a real model (or set VINCIO_PROVIDER / the API key in the environment)
app = ContextApp(name="docs_qa", provider="openai", model="gpt-4o-mini")
```

</details>

## Features

Everything below is implemented, tested offline, and demonstrated by a runnable example. Use the
high-level `ContextApp`, or reach for any engine directly.

<p align="center">
  <img src="assets/features.svg" alt="One platform, every layer: context and prompts; retrieval and memory; agents and orchestration; output and evaluation; the closed loop; security and governance; protocols and interop; cross-org economy, edge and federated reach" width="840">
</p>

<details>
<summary><b>Every engine, in detail</b></summary>

**Context & prompts**
- Prompt compiler: typed prompt ASTs with `${variables}`, lint rules, cache-aware stable-prefix layout, versioning, hashing, and diffing.
- Context compiler: scores every candidate (relevance, novelty, authority, freshness, provenance, token cost, leakage risk), deduplicates, resolves conflicts, compresses, and packs to a token budget, with an *excluded-context report* explaining every omission.
- Tabular evidence: a typed, columnar `Dataset` and a deterministic `DataEncoder` that renders it header-once — lossless, columnar-accurate in token cost, far cheaper than `json.dumps` or a Markdown table; `TableEvidence` scores and cites it like any other evidence.
- Governed text-to-query, a multi-step data-analysis agent, content- & data-bound charts, a streaming out-of-core path, and a governed semantic layer — the whole data & analytics plane, every answer citing the exact source cells and `verify()`-ing offline. See the [data analysis guide](docs/guides/analyze-data.md).

**Retrieval & memory**
- Hybrid RAG: BM25 + dense + learned-sparse + late-interaction fused in one weighted RRF; query understanding (HyDE, multi-query, decomposition); sentence-window / auto-merging chunking; GraphRAG; structured metadata filters with tenant scope; text + image + table + video evidence as first-class scored candidates.
- Layered memory: session → episodic → semantic → tenant → graph, with a guarded write pipeline, confidence decay, contradiction resolution, bi-temporal recall, per-memory ACLs, and audited GDPR-style edit/forget/export.

**Agents & orchestration**
- Tools: permissioned registry (RBAC + ABAC), schema-from-typehints, a resource-limited sandbox, idempotent write guardrails with approval callbacks, and a grounded computer-use action plane.
- Agents: bounded DAG execution with planners (ReAct / plan-and-execute / hierarchical HTN), in-place plan repair, cost-aware action selection, and a budgeted deep-research agent.
- Orchestration: multi-agent crews with a shared blackboard, durable stateful graphs (checkpoint / resume / time-travel / human-in-the-loop), deterministic workflows, and a distributed durable-execution backend.

**Output, evaluation & observability**
- Structured output: Pydantic contracts, constrained decoding, streaming validation with early abort, bounded self-correction that repairs structure only (never invents facts), and DSPy-style typed signatures.
- Evaluation: golden datasets, 30+ metrics, deterministic / model / G-Eval judges, synthetic data, red-teaming, trajectory & tool-use scoring, drift detection, regression gates, and a `pytest` plugin.
- Observability: full trace span trees, OpenTelemetry export, a local trace viewer, a versioned prompt registry, and per-run cost tracking — no account or hosted backend required.

**The closed loop**
- Optimization: one reproducible cycle (trace → dataset → eval → optimize → promote): a reflective GEPA/MIPRO optimizer, a distillation flywheel, on-policy reinforcement from verifiable rewards, and gated deploy with canary + rollback. No promotion ships without clearing the gates.

**Security & governance**
- Security: deterministic PII / secret redaction (multilingual), prompt-injection defense and provable containment (taint tracking + capability tokens), RBAC / ABAC, tenant isolation, and a hash-chained, signed audit log with offline tamper verification.
- Governance: model / system cards, an OWASP / NIST / MITRE / ISO compliance matrix, an AI-BOM, provable erasure, a consent ledger, data-residency enforcement, formal invariant verification, agent identity & delegation, verified-reasoning certificates, and continuous assurance cases.

**Interop**
- Protocols: MCP (client *and* server), A2A agent-to-agent, and Agent Skills, all in-process.
- Ecosystem: import/export LangChain, LlamaIndex, Haystack, and DSPy assets; first-party data connectors; and any OpenAI-compatible model or vector store you already run.

**Reach further:** a cross-organization agent economy (negotiation, contracts, durable sagas, metering, settlement, arbitration, reputation, collateral & solvency proofs), an edge / WASM in-process runtime, on-device LoRA adaptation, federated learning with a differential-privacy accountant, and per-run energy / carbon accounting. See [`ROADMAP.md`](ROADMAP.md).

</details>

## Benchmarks

Three suites ship in [`benchmarks/`](benchmarks), all reproducible on your own machine. Every number
is measured live from both sides; a missing competitor is reported as skipped, never assumed.

### Head-to-head vs. real libraries

[`competitive.py`](benchmarks/competitive.py) runs Vincio against the *actual* library a team would
otherwise use (Apple Silicon, Python 3.13; ratios are the portable signal, not wall-clock).

<p align="center">
  <img src="assets/benchmark-headtohead.svg" alt="Head-to-head vs. real libraries: 30 to 40 times faster BM25 at 20k docs vs rank_bm25; 60 percent fewer tokens for the same answer vs LangChain and LlamaIndex; 1.4 to 1.8 times faster token counting vs tiktoken; 4 of 8 vs 1 of 8 malformed JSON recovered vs stdlib json" width="840">
</p>

<details>
<summary><b>Show the full table</b></summary>

| Operation | Vincio | Competitor | Result |
|---|---|---|---|
| BM25 query @ **20k docs** | `BM25Index` | `rank_bm25` | **~30–40× faster**: identical top-1 ranking |
| **Context assembly**: tokens sent for the same retrieved set | context compiler | LangChain `stuff` / LlamaIndex `compact` | **~60% fewer tokens**: answer retained |
| **Tabular encoding**: tokens for a 50×5 table | `DataEncoder` | `json.dumps` / `pandas.to_markdown` / TOON | **~66% fewer tokens** than `json.dumps`, lossless, typed schema |
| **Fit a 5k-row table into the window** | `fit_to_window` | `json.dumps` all rows / `pandas.describe` | **~99% fewer tokens**: profile + representative sample, size invariant to row count |
| **Aggregate a 500k-row source** | `stream_aggregate` | materialize-then-aggregate / `pandas.groupby` | **~99% less peak memory**: one accumulator per group, footprint invariant to row count |
| Text chunking a 24k-word doc | `chunk_document` | LangChain / LlamaIndex splitters | **fastest**, chunks carry provenance |
| Token counting (~60k words) | `HeuristicTokenCounter` | `tiktoken` | **~1.4–1.8× faster**, zero-dependency, conservative |
| Malformed-JSON recovery | lenient parser | stdlib `json.loads` | **4/8 vs 1/8** recovered |
| Render with a missing variable | `PromptSpec.substitute` | `jinja2` | typed error vs. silently-empty render |

`rank_bm25` rescans every document per query; Vincio's inverted index only scans documents
containing a query term, so its lead grows with corpus size. The point isn't that every component
beats every specialist: a dedicated JSON-repair library recovers more than Vincio (by guessing,
which is unsafe for typed extraction). Vincio's edge is an **integrated, correct, governed**
pipeline, not a pile of single-purpose libraries.

</details>

### Orchestrator uplift: the same model, through Vincio

[`quality_uplift.py`](benchmarks/quality_uplift.py) measures what routing a model *through* Vincio
adds versus calling it directly, against real models on 15 company-specific policy questions a model
cannot know from pretraining (**4 models × 3 runs = 360 live calls**, OpenRouter, June 2026).

<p align="center">
  <img src="assets/benchmark-uplift.svg" alt="Grounded-answer accuracy, direct vs. through Vincio: gpt-4o-mini 2 to 100 percent; claude-3-haiku 0 to 91 percent; gemini-2.5-flash-lite 4 to 98 percent; llama-3.1-8b 2 to 89 percent; aggregate 2 to 95 percent" width="840">
</p>

<details>
<summary><b>Show the numbers and the honest read</b></summary>

**Deterministic mechanism metrics** (mechanical, so they hold for any model and run offline):

| Same model: direct vs. via Vincio | Direct | Via Vincio |
|---|--:|--:|
| Schema-valid object from realistic model outputs | 1/6 | **5/6** |
| Prompt-injection exfiltration via a tool call | compromised | **contained** |
| Context tokens to keep an early fact at 160 turns | 1,267 (lost) | **33 (retained)** |

**Grounded-answer quality on real models** (mean over runs, stochastic by a point or two):

| Model: direct vs. through Vincio | Direct correct | **Via Vincio correct** | Direct hallucinated | Cost per *correct* answer |
|---|--:|--:|--:|:--|
| `openai/gpt-4o-mini` | 2% | **100%** | 64% | **~62× cheaper** via Vincio |
| `anthropic/claude-3-haiku` | 0% | **91%** | 2%¹ | direct **never** correct (∞) |
| `google/gemini-2.5-flash-lite` | 4% | **98%** | 29% | **~67× cheaper** via Vincio |
| `meta-llama/llama-3.1-8b-instruct` | 2% | **89%** | 40% | **~29× cheaper** via Vincio |
| **Aggregate** | **2%** | **95%** | n/a | n/a |

<sub>¹ claude-3-haiku *abstains* (98% of the time) rather than guessing; better-aligned models say "I don't know," weaker ones confidently fabricate. Either way the model *alone* answers ~2%; the same model through Vincio's retrieval + grounding answers 89–100%, every answer cited.</sub>

The cost line is the honest punchline: a direct call is cheaper *per call*, but it answers almost
nothing correctly, so its cost **per correct answer** is 29–67× higher, or undefined when the model
gets *nothing* right on its own. Full per-metric breakdown is in
[`benchmarks/README.md`](benchmarks/README.md). Reproduce with `VINCIO_PROVIDER=openrouter … python
benchmarks/quality_uplift.py`.

</details>

### VincioBench: the internal regression suite

[`vinciobench.py`](benchmarks/vinciobench.py) is **not a competitive claim**: it is the deterministic
mechanism suite that gates CI. Its families assert that each engine still *works* on a bundled
synthetic corpus, so a regression fails the build. The scores saturate by design (a small corpus
built to exercise each mechanism), which proves *the mechanism is intact*, not real-world
performance. The credible performance evidence is the two sections above.

## How Vincio compares

Each ecosystem below is strong in its focus area. This reflects **built-in, in-library** capability,
not what's reachable by adding a separate product or SaaS.

<p align="center">
  <img src="assets/compare-matrix.svg" alt="Capability matrix comparing Vincio, LangChain, LlamaIndex, DSPy, and Ragas across twelve capabilities including the scored context compiler, sparse and late-interaction and GraphRAG fusion, layered memory, permissioned tools, durable graphs, structure-only repair, built-in evals and CI gates, eval-driven optimization, native tracing and cost, deterministic security, MCP and A2A and Skills, and governance evidence. Vincio is first-class across all twelve." width="840">
</p>

<details>
<summary><b>Show the full matrix</b></summary>

| Capability | **Vincio** | LangChain | LlamaIndex | DSPy | Ragas |
|---|:--:|:--:|:--:|:--:|:--:|
| Scored, budgeted **context compiler** | ✅ | ➖ | ➖ | ❌ | ❌ |
| **Sparse + late-interaction + GraphRAG** in one fusion | ✅ | ➖ | ➖ | ❌ | ❌ |
| Layered **memory** (decay, conflicts, bi-temporal) | ✅ | ➖ | ➖ | ❌ | ❌ |
| **Permissioned** tool registry (RBAC/ABAC) | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Durable graphs** + bounded crews | ✅ | ➖ | ❌ | ❌ | ❌ |
| Structured output + **structure-only repair** | ✅ | ➖ | ➖ | ✅ | ❌ |
| Built-in **evals + CI gates** | ✅ | ➖ | ➖ | ➖ | ✅ |
| Eval-driven **optimization** (gated promotion) | ✅ | ❌ | ❌ | ✅ | ❌ |
| Native **tracing + cost**, no account | ✅ | ➖ | ➖ | ❌ | ❌ |
| **Deterministic security** (PII / injection / audit) | ✅ | ❌ | ❌ | ❌ | ❌ |
| **MCP** client *and* server + **A2A** + **Skills** | ✅ | ➖ | ➖ | ➖ | ❌ |
| **Governance evidence** (cards · AI-BOM · erasure · residency) | ✅ | ❌ | ❌ | ❌ | ❌ |

✅ first-class in-library · ➖ partial or via an add-on/SaaS · ❌ not a focus. Ecosystems evolve, and
Vincio is built to *interoperate*: `vincio.interop` brings LangChain, LlamaIndex, Haystack, and DSPy
assets in (and hands Vincio's back). See the in-depth write-ups in
[`docs/comparisons/`](docs/comparisons).

</details>

## Examples

A three-tier on-ramp in [`examples/`](examples) — start in the browser, learn each subsystem, then
copy a real backend. Every tier runs **fully offline** on the bundled mock and points at a real model
with one env var; each is gated in CI so it can never drift.

### 1 · Notebooks — start in the browser

Five **Google Colab-ready** notebooks ([`examples/notebooks/`](examples/notebooks)), one `pip
install` and no setup:

| Notebook | Open in Colab |
|---|---|
| [Quickstart](examples/notebooks/01_quickstart.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohswedd/vincio/blob/main/examples/notebooks/01_quickstart.ipynb) |
| [RAG](examples/notebooks/02_rag.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohswedd/vincio/blob/main/examples/notebooks/02_rag.ipynb) |
| [Agents & tools](examples/notebooks/03_agents_and_tools.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohswedd/vincio/blob/main/examples/notebooks/03_agents_and_tools.ipynb) |
| [Evaluation](examples/notebooks/04_evaluation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohswedd/vincio/blob/main/examples/notebooks/04_evaluation.ipynb) |
| [Data analysis](examples/notebooks/05_data_analysis.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohswedd/vincio/blob/main/examples/notebooks/05_data_analysis.ipynb) |

### 2 · Feature tours — one program per subsystem

Twenty-three complete, heavily-commented programs; each runs offline and teaches a whole theme end to
end. Highlights (full index in [`examples/README.md`](examples/README.md)):

| # | Example | What it covers |
|--|---|---|
| 00 | [`one_liners`](examples/00_one_liners.py) | the `vincio.tasks` front door — `rag` / `extractor` / `tool_agent` / `evaluation` / `chat` / `Flow`, each lowering to the same governed run |
| 01 | [`quickstart`](examples/01_quickstart.py) | typed output · grounded QA with citations · trace & cost · a short conversation |
| 02 | [`retrieval_rag`](examples/02_retrieval_rag.py) | hybrid + sparse + late-interaction fusion · query understanding · GraphRAG · multimodal evidence |
| 04 | [`agents_and_tools`](examples/04_agents_and_tools.py) | permissioned tools · sandbox · planners · plan repair · deep research · computer-use |
| 07 | [`evaluation_observability`](examples/07_evaluation_observability.py) | datasets · metrics · judges · red-team · drift · tracing · prompt registry |
| 09 | [`security_governance`](examples/09_security_governance.py) | PII/injection/containment · audit · governance evidence · identity · verified reasoning · assurance |
| 12 | [`cross_org_economy`](examples/12_cross_org_economy.py) | negotiation · contracts · durable sagas · settlement · arbitration · solvency proofs |
| 13–20 | [data plane](examples/13_tabular_evidence.py) | tabular evidence · profiling · governed text-to-query · the analysis agent · cited charts · streaming · the semantic layer · the data engagement |
| 22 | [`connected_docs`](examples/22_connected_docs.py) | the capability map · Related cross-links · the learning path · the docs-graph check |

### 3 · Applications — real-world backends

Small, production-shaped apps to copy ([`examples/applications/`](examples/applications)): a FastAPI
**grounded-RAG service**, a **ticket-triage API** (typed output + scoped memory + an approval-gated
tool), a **structured-extraction service** (self-correcting), and a no-framework **CLI research
agent**. Each FastAPI app splits an offline-testable `core.py` from a thin FastAPI `main.py`.

```bash
cd examples && python 01_quickstart.py            # offline, no keys
export VINCIO_PROVIDER=openai OPENAI_API_KEY=sk-... && python 01_quickstart.py   # against a real model
pip install "vincio[server]" && cd examples/applications/rag_service && uvicorn main:app --reload
```

## Command line

```bash
vincio init my-project --template rag   # scaffold config + app + golden set
vincio run app.py --input "..."         # run an app
vincio eval run golden.jsonl            # run an eval suite with CI gates + baseline compare
vincio trace view trace_123             # TUI trace tree with scores + feedback
vincio loop run --app app.py --gate groundedness=">= 0.8"   # one closed-loop cycle
vincio docs check                       # gate the docs graph (links, coverage, llms.txt freshness)
vincio audit verify                     # verify the audit-log hash chain offline
vincio mcp serve app.py                 # expose an app as an MCP server
vincio serve --app app.py               # launch the HTTP API (health/readiness/metrics)
```

The full CLI is in the [CLI reference](docs/reference/cli.md). `vincio serve` launches a FastAPI
server (API-key + JWT auth, SSE streaming, Prometheus metrics); `from vincio.server import
create_app` embeds it.

## Architecture

One coherent pipeline from raw input to traced, validated result: the input engine normalizes and
scopes the request; memory, retrieval, tools, and the prompt compiler all feed the **context
compiler**, which scores, deduplicates, resolves conflicts, compresses, and budgets; the model runs
provider-neutral; and every output is validated, evaluated, secured, traced, costed, and written
back to memory.

<p align="center">
  <img src="assets/architecture.svg" alt="Vincio architecture: the input engine feeds the context compiler, which is also fed by memory, retrieval, tools, and the prompt compiler; the context compiler feeds provider-neutral model execution; the output is validated, evaluated, secured, traced, costed, and written back to memory" width="840">
</p>

See [`AGENTS.md`](AGENTS.md) for the package layout and [`docs/concepts/`](docs/concepts) for a tour
of each engine.

## Status

Vincio is **feature-complete and in long-term support** on the 5.x surface. The public API is frozen
under [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a mechanical
[deprecation policy](docs/reference/stability.md); performance and quality targets are
[published as SLOs](docs/reference/slo.md) and gated by VincioBench; releases ship a CycloneDX SBOM
with SLSA provenance. New capabilities are added behind opt-in extras, never by breaking working
code. The forward plan — a scheduled data & analytics extension line (real-time, federated,
forecasting/causal verifiers, notebook-native) — is in [`ROADMAP.md`](ROADMAP.md); upgrades in
[`MIGRATION.md`](MIGRATION.md).

Vincio is, and stays, a **library**. The building blocks for production (audit chain, retention,
tenant isolation, RBAC/ABAC, a server) ship in the package for you to deploy on your own
infrastructure. There is no hosted service.

## Documentation

The [documentation index](docs/README.md) maps every guide, concept, and reference page in a
reading order; the [learning path](docs/learning-path.md) is a staged route from your first app to
the full platform. Highlights:

- **[Getting started](docs/getting-started.md)**: install, your first app, offline development
- **Concepts**: [context packets](docs/concepts/context-packets.md) ·
  [prompt compiler](docs/concepts/prompt-compiler.md) · [memory](docs/concepts/memory.md) ·
  [retrieval](docs/concepts/retrieval.md) · [agents & workflows](docs/concepts/agents.md) ·
  [evaluation](docs/concepts/evals.md) · [observability](docs/concepts/observability.md)
- **Guides**: [build a RAG app](docs/guides/build-rag-app.md) ·
  [structured output](docs/guides/structured-output.md) ·
  [add tools](docs/guides/add-tools.md) · [analyze data](docs/guides/analyze-data.md) ·
  [orchestrate multi-agent systems](docs/guides/orchestrate-agents.md) ·
  [run evals](docs/guides/run-evals.md) · [close the loop](docs/guides/close-the-loop.md) ·
  [performance & streaming](docs/guides/performance.md) · [integrations](docs/guides/integrations.md)
- **Protocols**: [MCP client + server](docs/guides/mcp.md) · [A2A](docs/guides/a2a.md) ·
  [Agent Skills](docs/guides/agent-skills.md) · [reasoning control](docs/guides/reasoning.md)
- **Migrating**: from [LangChain](docs/guides/migrate-from-langchain.md) ·
  [LlamaIndex](docs/guides/migrate-from-llamaindex.md) · [Ragas](docs/guides/migrate-from-ragas.md)
- **Security & governance**: [threat model](docs/security/threat-model.md) ·
  [security policy](SECURITY.md) · [governance & compliance](docs/guides/governance.md)
- **Reference**: [API](docs/reference/api.md) · [capability map](docs/reference/capability-map.md) ·
  [CLI](docs/reference/cli.md) · [config](docs/reference/config.md) · [SLOs](docs/reference/slo.md) ·
  [stability & deprecation](docs/reference/stability.md)
- **Comparisons**: [LangChain](docs/comparisons/langchain.md) ·
  [LlamaIndex](docs/comparisons/llamaindex.md) · [DSPy](docs/comparisons/dspy.md) ·
  [CrewAI](docs/comparisons/crewai.md) · [Ragas](docs/comparisons/ragas.md) ·
  [and more](docs/comparisons)

## Contributing

Contributions are welcome. The test suite runs fully offline and must stay green:

```bash
pip install -e ".[dev]"
python -m pytest -q          # 6421 tests, no network or API keys required
ruff check vincio/ tests/
mypy vincio
```

See [`AGENTS.md`](AGENTS.md) for the codebase layout and engineering conventions.

## License

[Apache License 2.0](LICENSE) © Vincio Contributors.
