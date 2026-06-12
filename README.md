<p align="center">
  <img src="assets/banner.svg" alt="Vincio — the context engineering platform for AI applications" width="660">
</p>

<p align="center">
  <em>The scarce resource is not the model. It is the context you feed it.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/vincio/"><img src="https://img.shields.io/pypi/v/vincio?color=B98B2E" alt="PyPI version"></a>
  <a href="https://github.com/Ohswedd/vincio/actions/workflows/ci.yml"><img src="https://github.com/Ohswedd/vincio/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/pypi/pyversions/vincio?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-4C6EF5" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/tests-229%20passing-2ea44f" alt="229 tests passing">
  <img src="https://img.shields.io/badge/lint-ruff-D7FF64" alt="Ruff">
  <img src="https://img.shields.io/badge/typed-pydantic%20v2-E92063" alt="Pydantic v2">
  <img src="https://img.shields.io/badge/offline-first-555" alt="Offline-first">
</p>

---

**Vincio** is a Python platform for building **context-engineered** AI applications. It compiles
prompts, memory, retrieval, tools, schemas, and policies into optimized, testable, observable,
provider-neutral **context packets** — then validates and evaluates every output.

Most LLM frameworks help you call a model. Vincio governs the *boundary* between your application
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

[Why Vincio](#why-vincio) · [Install](#install) · [60-second quickstart](#60-second-quickstart) ·
[Features](#features) · [Benchmarks](#benchmarks) · [Comparison](#how-vincio-compares) ·
[Use cases](#use-cases) · [Examples](#more-examples) · [CLI](#command-line) ·
[Architecture](#architecture) · [Roadmap](#roadmap) · [Documentation](#documentation)

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
pip install "vincio[all]"           # every optional integration
```

Python 3.11+. Core dependencies are just `pydantic`, `httpx`, `pyyaml`, and `typing-extensions`;
every heavy integration (vector stores, OCR, server, OpenTelemetry, …) is an opt-in extra.

## 60-second quickstart

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

### Evaluation as a gate

```python
from vincio.evals import Dataset, EvalRunner

dataset = Dataset.load("golden/support_triage.jsonl")
report = EvalRunner(app).run(dataset)
report.print_summary()     # groundedness, citation accuracy, schema validity, cost — with CI exit codes
```

## Features

Vincio is organized into composable subsystems. Use the high-level `ContextApp` runtime, or reach
for any engine directly.

| Subsystem | What it does |
|---|---|
| **Prompt compiler** | Typed prompt ASTs with `${variables}`, lint rules, cache-aware stable-prefix layout, versioning, hashing, diffing, variant generation. |
| **Context compiler** | Scores every candidate (relevance, novelty, authority, freshness, provenance, token cost, leakage risk), deduplicates, resolves conflicts, compresses, and packs to a token budget — with an *excluded-context report* explaining every omission. |
| **Retrieval (RAG)** | Hybrid BM25 + dense retrieval, query planning with subqueries, rerankers, entity-graph and multi-hop retrieval, reasoning retrieval that reports missing fact types, citations. |
| **Memory** | Layered (session → episodic → semantic → tenant → graph) with a guarded write pipeline, confidence decay, contradiction resolution, and privacy scoping. |
| **Tools** | Permissioned registry (RBAC scopes + ABAC rules), schema derivation from type hints, sandboxing, reliability scoring, idempotent write-action guardrails with approval callbacks. |
| **Agents** | Bounded DAG execution with planners (direct / static / dynamic / ReAct / plan-and-execute), critics, validators, human gates, and hard budget enforcement. |
| **Workflows** | Deterministic DAGs with retries, branching, parallelism, compensation, and approval gates. |
| **Structured output** | Pydantic output contracts, robust parsers (fenced / embedded / lenient / streaming JSON), a validation pipeline, and **principled repair that fixes structure only — never invents facts**. |
| **Evaluation** | Golden JSONL datasets, 17+ task / grounding / retrieval / operational metrics, deterministic and model judges, regression gates, and baseline-diff reports. |
| **Optimization** | Prompt / context / routing / cache search driven by an eval-fitness function, with safety-gated promotion that blocks any candidate regressing schema validity or safety. |
| **Observability** | Every run yields a full trace span tree; JSONL and OpenTelemetry exporters; per-run cost tracking. |
| **Security** | Deterministic PII / secret detection and redaction, prompt-injection defense, RBAC / ABAC, tenant isolation, and a hash-chained audit log. |
| **Storage** | Pluggable metadata (in-memory / SQLite / Postgres), blob, analytics (DuckDB), vector (Qdrant / pgvector), and graph (Neo4j) backends behind one factory. |
| **Providers** | OpenAI, Anthropic, Google, Mistral, any OpenAI-compatible endpoint, and a deterministic offline mock — all async-first with sync wrappers, pooled transport, retries, failover, and in-flight request coalescing. |
| **Performance (0.2)** | End-to-end streaming (`astream` + SSE) with incremental partial-JSON output, concurrent retrieval/memory/tool fan-out with cancellation propagation and hard latency deadlines, content-addressed compile/chunk/embedding caches, zero-copy (slim) context packets, and CI-gated VincioBench performance budgets. |

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
| **Retrieval** | recall@3 / MRR (known-answer corpus) | **1.00 / 1.00** | — |
| **Memory** | preference recall · contradiction supersede · tenant isolation | **pass** | — |
| **Tools** | runtime overhead, p50 | **0.02 ms** | — |
| **Agents** | adversarial infinite-loop model | **bounded** (budget) | unbounded |

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
| Layered **memory** (decay, conflicts, scopes) | ✅ | ➖ | ➖ | ❌ | ❌ |
| **Permissioned** tool registry (RBAC/ABAC) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Bounded **agents** + deterministic workflows | ✅ | ✅ | ➖ | ➖ | ❌ |
| Structured output + **structure-only repair** | ✅ | ➖ | ➖ | ✅ | ❌ |
| Built-in **evals + CI gates** | ✅ | ➖ | ➖ | ➖ | ✅ |
| Eval-driven **optimization** (gated promotion) | ✅ | ❌ | ❌ | ✅ | ❌ |
| Native **tracing + cost**, no account needed | ✅ | ➖ | ➖ | ❌ | ❌ |
| **Deterministic security** (PII / injection / audit) | ✅ | ❌ | ❌ | ❌ | ❌ |

<sub>✅ first-class in-library · ➖ partial or via a separate add-on/SaaS · ❌ not a focus. Reflects
mid-2026; ecosystems evolve. Vincio is built to *interoperate* — wrap a LangChain component as a
tool, feed LlamaIndex-parsed documents into a source, use a DSPy program as a provider, or register
a Ragas metric with `@register_metric`. See the in-depth write-ups in
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

## More examples

All eleven examples in [`examples/`](examples) run **fully offline** with no API keys. Point them at
a real model with environment variables:

```bash
export VINCIO_PROVIDER=openai VINCIO_MODEL=gpt-5.2-mini OPENAI_API_KEY=sk-...
cd examples && python 02_document_qa.py
```

## Command line

```bash
vincio init my-project           # scaffold config, a starter app, and a golden dataset
vincio run app.py --input "..."  # run an app
vincio eval run golden.jsonl     # run an eval suite (with CI gates and baseline compare)
vincio prompt lint prompts/      # lint prompt specs
vincio trace show trace_123      # inspect a run's full trace
vincio optimize run --target groundedness
vincio index build ./docs        # build a retrieval index
vincio memory inspect --user u1  # inspect a user's memory
```

A FastAPI server (API-key + JWT auth, real-token SSE streaming) is available via
`from vincio.server import create_app` — see [`docs/reference/api.md`](docs/reference/api.md).

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

See [`AGENTS.md`](AGENTS.md) for the package layout and [`docs/concepts/`](docs/concepts) for a tour
of each engine.

## Roadmap

Vincio 0.1.0 shipped every in-scope subsystem above; 0.2.0 made the spine fast — streaming,
concurrency, compilation caches, and CI-gated performance budgets — with 229 offline tests, eleven
runnable examples, and full documentation. The public roadmap — what's shipped, what's next, and
what's intentionally out of scope — lives in **[ROADMAP.md](ROADMAP.md)**.

Vincio is, and stays, a **library**. The building blocks for production operation (audit chain,
retention, tenant isolation, RBAC/ABAC, a server) ship in the package for you to deploy on your own
infrastructure. Hosted services and managed control planes are not part of this project.

## Documentation

- **[Getting started](docs/getting-started.md)** — install, your first app, offline development
- **Concepts** — [context packets](docs/concepts/context-packets.md) ·
  [prompt compiler](docs/concepts/prompt-compiler.md) · [memory](docs/concepts/memory.md) ·
  [retrieval](docs/concepts/retrieval.md) · [agents & workflows](docs/concepts/agents.md) ·
  [evaluation](docs/concepts/evals.md)
- **Guides** — [build a RAG app](docs/guides/build-rag-app.md) ·
  [structured output](docs/guides/structured-output.md) · [add tools](docs/guides/add-tools.md) ·
  [run evals](docs/guides/run-evals.md) · [optimize](docs/guides/optimize-context.md) ·
  [performance & streaming](docs/guides/performance.md)
- **Reference** — [API](docs/reference/api.md) · [CLI](docs/reference/cli.md) ·
  [config](docs/reference/config.md)
- **Comparisons** — [LangChain](docs/comparisons/langchain.md) ·
  [LlamaIndex](docs/comparisons/llamaindex.md) · [DSPy](docs/comparisons/dspy.md) ·
  [Ragas](docs/comparisons/ragas.md)

## Contributing

Contributions are welcome. The test suite runs fully offline in a couple of seconds and must stay
green:

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q     # 229 tests, no network or API keys required
ruff check vincio/ tests/
```

See [`AGENTS.md`](AGENTS.md) for the codebase layout and engineering conventions.

## License

[Apache License 2.0](LICENSE) © Vincio Contributors.
