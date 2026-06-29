<p align="center">
  <img src="assets/banner.svg" alt="Vincio: the context engineering platform for AI applications" width="660">
</p>

<p align="center">
  <em>The scarce resource is not the model. It is the context you feed it.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/vincio/"><img src="https://img.shields.io/badge/vincio-5.0.0-B98B2E" alt="Vincio 5.0.0"></a>
  <a href="https://github.com/Ohswedd/vincio/actions/workflows/ci.yml"><img src="https://github.com/Ohswedd/vincio/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/pypi/pyversions/vincio?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-4C6EF5" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/tests-5858%20passing-2ea44f" alt="5858 tests passing">
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

<p align="center">
  <img src="assets/why.svg" alt="Why Vincio: offline dev and CI (deterministic mock, no key, no cost); deterministic (security and validation in code, not model output); measured (every run traced and costed, eval-gated); one system (input to output, not a bag of utilities)" width="840">
</p>

<details>
<summary><b>Why you'd reach for it, in one line each</b></summary>

- **Runs on any model, offline when you want.** Call OpenAI, Anthropic, Google, Mistral, a local model, or any OpenAI-compatible gateway through one interface. No key yet? A deterministic mock runs the whole pipeline (retrieval, validation, evals, traces) for dev, tests, and CI, with no network and no cost.
- **Deterministic where it counts.** Security, permissions, and validation are enforced in code, never gated on model output. The same input compiles to the same packet.
- **Measured, not asserted.** Every run is traced and costed; every change can be gated by an eval suite before it ships.
- **One coherent system** from input to output, not a bag of utilities you wire together yourself.

</details>

## Contents

[Install](#install) · [Quickstart](#quickstart) · [What you can build](#what-you-can-build) ·
[Providers](#providers--models) · [Features](#features) · [Benchmarks](#benchmarks) ·
[How Vincio compares](#how-vincio-compares) · [Examples](#examples) · [CLI](#command-line) ·
[Architecture](#architecture) · [Docs](#documentation)

## Install

```bash
pip install vincio                  # core (the offline mock provider is built in)
pip install "vincio[openai]"        # + OpenAI    (also: anthropic, google, mistral)
pip install "vincio[chroma]"        # + a vector store (also: pinecone, lancedb, pgvector, …)
pip install "vincio[all]"           # every optional integration
```

Python 3.11+. The core depends only on `pydantic`, `httpx`, `pyyaml`, and `typing-extensions`;
every heavy integration (vector stores, OCR, server, OpenTelemetry, …) is an opt-in extra.

## Quickstart

```python
from vincio import ContextApp

app = ContextApp(name="docs_qa")
app.add_source("docs", path="./docs", retrieval="hybrid")
app.set_policy("answer_only_from_sources", True)

result = app.run("How do I configure SSO?")
print(result.output)      # the grounded answer
print(result.citations)   # the evidence it actually cited
print(result.trace_id)    # every run produces a full trace
print(result.cost_usd)    # …and a cost
```

To use a real model, set a provider and key, for example `export VINCIO_PROVIDER=openai
OPENAI_API_KEY=sk-...`, or pass `provider=` and `model=` to `ContextApp`. The same code runs against
OpenAI, Anthropic, Google, Mistral, a local model, or any OpenAI-compatible gateway. No key yet? Out
of the box it runs on a deterministic mock that emits schema-valid output, so you can build and test
the whole pipeline offline in CI.

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

class Triage(BaseModel):
    label: str
    confidence: float

app = ContextApp(name="triage", output_schema=Triage)
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

**Evaluation as a CI gate**: measure quality and block a regression before it ships:

```python
from vincio import Dataset
from vincio.evals import EvalCase, EvalRunner

dataset = Dataset(name="golden", cases=[EvalCase(id="c1", input="…", expected="…")])
runner = EvalRunner(app, metrics=["groundedness", "citation_accuracy"],
                    gates={"groundedness": ">= 0.8"})
report = runner.run(dataset)
assert all(g["passed"] for g in report.gates.values())   # fail the build on a regression
```

See **[Examples](#examples)** for twelve complete, runnable programs that cover the whole platform.

## Providers & models

Vincio calls real models in production. One interface routes to every major provider, with the
model-operations layer (reasoning control, half-cost batch, caching, failover, cost tracking) built
in. The deterministic mock is a development convenience, not the product: it lets you build and test
the whole pipeline with no key and no cost before you point it at a real model.

<p align="center">
  <img src="assets/providers.svg" alt="Providers and models: one interface over OpenAI, Anthropic, Google, Mistral, local models, and any OpenAI-compatible gateway, plus enterprise auth for Amazon Bedrock, Google Vertex, and Azure OpenAI. Model operations: unified reasoning control, batch at about half cost, prompt caching, circuit breaker and failover, key pool, and per-run cost tracking. With no key, a deterministic mock runs the whole pipeline for dev, tests, and CI." width="840">
</p>

<details>
<summary><b>Providers, model operations, and the mock</b></summary>

- **Providers**: OpenAI, Anthropic, Google (Gemini), Mistral, local models, and any OpenAI-compatible gateway (Groq, Together, Fireworks, OpenRouter, and the like) through one `ModelProvider` interface.
- **Enterprise auth**: Amazon Bedrock, Google Vertex, and Azure OpenAI via pluggable auth strategies (SigV4, service-account, Azure AD / key).
- **Model operations**: unified reasoning/thinking control across providers, batch backends (~50% cost), prompt-cache strategy, a circuit breaker with health-aware failover, a key pool, and a data-driven `ModelRegistry` (capabilities, pricing, lifecycle) that drives capability guards and shadow / canary dispatch. Its shipped catalog prices the current lineup of every provider and is held by a coverage gate, so no current model silently bills $0.
- **The mock**: `MockProvider` is deterministic and emits schema-valid output, so the full pipeline (retrieval, validation, evals, traces, cost) runs offline in CI with no key and no cost. Use it for development and tests; use a real provider in production.

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
- Tabular evidence: a typed, columnar `Dataset` and a deterministic `DataEncoder` that renders it header-once — schema, types, and units declared once, cells as delimited rows — lossless, columnar-accurate in token cost, and far cheaper than `json.dumps` or a Markdown table; `TableEvidence` scores and cites it like any other evidence.
- Dataset profiling & quality: `profile_dataset` computes a deterministic, bounded-memory column profile (cardinality, percentiles, histograms, null rate, exemplars); reservoir/stratified sampling stands a representative sample in for the whole; `fit_to_window` fits a table far larger than the window — profile plus sample — under a fixed token budget; and `DataQualityRails` screen for schema violations, constraint breaks, anomalies, and PII on the deterministic rail path.
- Governed text-to-query: `app.query_data` turns a question over a registered dataset into a query that is verified *before* it runs — schema-grounded, **read-only by default** (a generated write, DDL, stacked statement, or an injection signal in the question is refused structurally), and cost-bounded — executed by the standard-library `sqlite3` engine where the data lives, not by pouring rows into the prompt. The answer **cites the exact source cells** it rests on (`sales#r0!revenue`), and `result.verify()` re-derives the answer and every cited cell from the bytes.
- Data-analysis agent: `app.analyze_data` runs a bounded, multi-step analysis over a dataset — it plans, queries through the read-only-verified query plane, inspects, and drills into the group that dominates — and returns a **cited analytical narrative** whose every finding points at the exact source cells and whose `verify(catalog)` re-derives the whole analysis from the bytes. Bounded by an explicit `AnalysisBudget`, audited, and deterministic offline; a DuckDB engine runs the same verified SQL at scale behind the `vincio[data]` extra.
- Charts & cited artifacts: `app.generate_chart` turns a cited query result into a spec-driven `Chart` that is **content-bound** (a C2PA data-driven credential bound to its rendered bytes, exactly the provenance a generated image carries) and **data-bound** (a back-reference to the exact source cells that `verify(catalog)` re-derives offline). The default renderer emits a portable Vega-Lite spec — no drawing library — and `MatplotlibRenderer` rasterizes the same spec to a PNG behind the `vincio[charts]` extra. The cited-report builder extends to figures, so a `Figure` embeds a chart or table into a deliverable that is per-claim entailed *and* per-figure data-bound.
- Streaming & out-of-core: a lazy, re-iterable `RowStream` processes a dataset far larger than memory in bounded passes — open a CSV / JSON-Lines file read line by line, iterate it in bounded chunks, and profile, fit, or sample it in a single pass whose footprint tracks columns, not rows. `stream_aggregate` is a bounded-memory group-by (one accumulator per group, never the rows); `encode_stream` renders the compact encoding header-once and gzip-compresses it; the context compiler's streaming candidate pre-filter (`max_candidates`) bounds a 10k+ evidence pool by a cheap relevance proxy before full scoring; and `app.map_stream` runs an analytical transform over a stream at scale through the `BatchRunner`.
- Semantic layer & governed metrics: a `SemanticLayer` defines measures, dimensions, and derived columns *once* (`revenue = price × qty`) so a question maps to a **governed metric** rather than a raw column — `app.query_metric` resolves a name or natural-language question to the measure, compiles it to one canonical read-only `SELECT`, and runs it through the same governed query plane, so the metric is computed **one way everywhere**, cell-cited, and `MetricResult.verify` proves the number is the governed one (an ad-hoc query is rejected). Column-level `app.metric_lineage` resolves a metric to its base columns and source, and a right-to-erasure sweep (`app.erase_source`) reaches the dataset plane — so a metric's provenance and a subject's erasure both reach structured data.
- Data engagement: `app.data_engagement` threads the whole analytics plane behind one governed, audited call-path — register → profile → sample → screen → query → analyze → chart → governed metric → cite — and seals it into a hash-chained, signed `DataNarrative`. The narrative `verify()`s offline from the bytes alone (a re-ordered stage, an edited digest, or a forged signature is caught), and — given the live catalog — every captured query, analysis, chart, and metric **re-executes against the content-hashed source and re-derives from the bytes**, so a tampered source is caught even when the chain is intact. Purely compositional: every step delegates to the same primitive a caller would use directly, each still usable on its own.

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
- Observability: full trace span trees, OpenTelemetry export, a local trace viewer, a versioned prompt registry, and per-run cost tracking, no account or hosted backend required.

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
gets *nothing* right on its own. Vincio is also faster per answer here (~1.3–1.6 s vs. ~1.7–2.5 s),
and token usage is roughly a wash. Full per-metric breakdown is in
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

Twelve complete, heavily-commented programs in [`examples/`](examples); each runs **fully offline**
and teaches a whole theme end to end.

| # | Example | What it covers |
|--|---|---|
| 01 | [`quickstart`](examples/01_quickstart.py) | typed output · grounded QA with citations · trace & cost · a short conversation |
| 02 | [`retrieval_rag`](examples/02_retrieval_rag.py) | hybrid + sparse + late-interaction fusion · query understanding · GraphRAG · multimodal evidence |
| 03 | [`memory`](examples/03_memory.py) | scoped remember/recall · bi-temporal · decay & contradictions · GDPR forget/export |
| 04 | [`agents_and_tools`](examples/04_agents_and_tools.py) | permissioned tools · sandbox · planners · plan repair · deep research · computer-use |
| 05 | [`orchestration`](examples/05_orchestration.py) | crews + blackboard · durable graphs · workflows · distributed execution |
| 06 | [`structured_output`](examples/06_structured_output.py) | contracts · constrained decoding · streaming validation · self-correction · signatures |
| 07 | [`evaluation_observability`](examples/07_evaluation_observability.py) | datasets · metrics · judges · red-team · drift · tracing · prompt registry |
| 08 | [`optimization_self_improvement`](examples/08_optimization_self_improvement.py) | the closed loop · reflective optimizer · RLVR · canary deploy · local & federated adaptation |
| 09 | [`security_governance`](examples/09_security_governance.py) | PII/injection/containment · audit · governance evidence · identity · verified reasoning · assurance |
| 10 | [`interop_and_protocols`](examples/10_interop_and_protocols.py) | MCP client+server · A2A · Agent Skills · framework interop · connectors · packs |
| 11 | [`advanced_context`](examples/11_advanced_context.py) | reasoning control · test-time compute · long-horizon · world-model · semantic cache · record-replay |
| 12 | [`cross_org_economy`](examples/12_cross_org_economy.py) | negotiation · contracts · durable sagas · settlement · arbitration · solvency proofs |
| 13 | [`tabular_evidence`](examples/13_tabular_evidence.py) | typed columnar `Dataset` · the compact, lossless `DataEncoder` · columnar token cost · `TableEvidence` in the compiler |
| 14 | [`dataset_profiling`](examples/14_dataset_profiling.py) | `profile_dataset` · reservoir/stratified sampling · `fit_to_window` under a token budget · `DataQualityRails` screening |
| 15 | [`governed_text_to_query`](examples/15_governed_text_to_query.py) | `app.query_data` · read-only-verified SQL · cell-level provenance (`cite_refs`) · offline `verify()` · the dataframe-op dialect |
| 16 | [`data_analysis_agent`](examples/16_data_analysis_agent.py) | `analyze_dataset` / `app.analyze_data` · plan → query → inspect → drill · cited analytical narrative · `AnalysisBudget` · offline `verify()` · injection refusal |
| 17 | [`charts_cited_artifacts`](examples/17_charts_cited_artifacts.py) | `generate_chart` / `app.generate_chart` · Vega-Lite spec (matplotlib PNG behind the extra) · C2PA-credentialed bytes · cell-level back-reference · offline `verify()` · per-figure data-bound cited reports |
| 18 | [`streaming_out_of_core`](examples/18_streaming_out_of_core.py) | `RowStream` over a source larger than memory · bounded chunks · `stream_aggregate` group-by in a fixed footprint · `encode_stream` (gzip) · the compiler's streaming candidate pre-filter · `app.map_stream` at scale on the `BatchRunner` |
| 19 | [`semantic_layer_governed_metrics`](examples/19_semantic_layer_governed_metrics.py) | `SemanticLayer` of measures / dimensions / derived columns defined once · `app.query_metric` governed metric computed one way everywhere · ratio metrics · cell-cited · `MetricResult.verify` (ad-hoc rejected) · column-level `app.metric_lineage` · `app.erase_source` reaching the dataset plane |
| 20 | [`data_engagement`](examples/20_data_engagement.py) | `app.data_engagement` threading the whole plane (register → profile → … → cite) into a hash-chained, signed `DataNarrative` · offline `verify()` · data-binding (every finding re-derives from the content-hashed source) · tamper detection · purely compositional |

```bash
cd examples && python 01_quickstart.py            # offline, no keys
export VINCIO_PROVIDER=openai OPENAI_API_KEY=sk-... && python 01_quickstart.py   # against a real model
```

## Command line

```bash
vincio init my-project --template rag   # scaffold config + app + golden set
vincio run app.py --input "..."         # run an app
vincio eval run golden.jsonl            # run an eval suite with CI gates + baseline compare
vincio trace view trace_123             # TUI trace tree with scores + feedback
vincio optimize run --target groundedness
vincio loop run --app app.py --gate groundedness=">= 0.8"   # one closed-loop cycle
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

Vincio 5.0 is **feature-complete and in long-term support**, with the data & analytics plane now
complete. The public API is frozen under
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a mechanical
[deprecation policy](docs/reference/stability.md); performance and quality targets are
[published as SLOs](docs/reference/slo.md) and gated by VincioBench; releases ship a CycloneDX SBOM
with SLSA provenance. New capabilities are added behind opt-in extras, never by breaking working
code. See [`ROADMAP.md`](ROADMAP.md) and [`MIGRATION.md`](MIGRATION.md).

Vincio is, and stays, a **library**. The building blocks for production (audit chain, retention,
tenant isolation, RBAC/ABAC, a server) ship in the package for you to deploy on your own
infrastructure. There is no hosted service.

## Documentation

The [documentation index](docs/README.md) maps every guide, concept, and reference page in a
reading order. Highlights:

- **[Getting started](docs/getting-started.md)**: install, your first app, offline development
- **Concepts**: [context packets](docs/concepts/context-packets.md) ·
  [prompt compiler](docs/concepts/prompt-compiler.md) · [memory](docs/concepts/memory.md) ·
  [retrieval](docs/concepts/retrieval.md) · [agents & workflows](docs/concepts/agents.md) ·
  [evaluation](docs/concepts/evals.md) · [observability](docs/concepts/observability.md)
- **Guides**: [build a RAG app](docs/guides/build-rag-app.md) ·
  [structured output](docs/guides/structured-output.md) ·
  [add tools](docs/guides/add-tools.md) ·
  [orchestrate multi-agent systems](docs/guides/orchestrate-agents.md) ·
  [run evals](docs/guides/run-evals.md) · [close the loop](docs/guides/close-the-loop.md) ·
  [performance & streaming](docs/guides/performance.md) · [integrations](docs/guides/integrations.md)
- **Protocols**: [MCP client + server](docs/guides/mcp.md) · [A2A](docs/guides/a2a.md) ·
  [Agent Skills](docs/guides/agent-skills.md) · [reasoning control](docs/guides/reasoning.md)
- **Migrating**: from [LangChain](docs/guides/migrate-from-langchain.md) ·
  [LlamaIndex](docs/guides/migrate-from-llamaindex.md) · [Ragas](docs/guides/migrate-from-ragas.md)
- **Security & governance**: [threat model](docs/security/threat-model.md) ·
  [security policy](SECURITY.md) · [governance & compliance](docs/guides/governance.md)
- **Reference**: [API](docs/reference/api.md) · [CLI](docs/reference/cli.md) ·
  [config](docs/reference/config.md) · [SLOs](docs/reference/slo.md) ·
  [stability & deprecation](docs/reference/stability.md)
- **Comparisons**: [LangChain](docs/comparisons/langchain.md) ·
  [LlamaIndex](docs/comparisons/llamaindex.md) · [DSPy](docs/comparisons/dspy.md) ·
  [CrewAI](docs/comparisons/crewai.md) · [Ragas](docs/comparisons/ragas.md) ·
  [and more](docs/comparisons)

## Contributing

Contributions are welcome. The test suite runs fully offline and must stay green:

```bash
pip install -e ".[dev]"
python -m pytest -q          # 5858 tests, no network or API keys required
ruff check vincio/ tests/
mypy vincio
```

See [`AGENTS.md`](AGENTS.md) for the codebase layout and engineering conventions.

## License

[Apache License 2.0](LICENSE) © Vincio Contributors.
