<p align="center">
  <img src="assets/banner.svg" alt="Vincio — the context engineering platform for AI applications" width="660">
</p>

<p align="center">
  <em>The scarce resource is not the model. It is the context you feed it.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/vincio/"><img src="https://img.shields.io/badge/vincio-4.0.0-B98B2E" alt="Vincio 4.0.0"></a>
  <a href="https://github.com/Ohswedd/vincio/actions/workflows/ci.yml"><img src="https://github.com/Ohswedd/vincio/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/pypi/pyversions/vincio?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-4C6EF5" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/tests-3288%20passing-2ea44f" alt="3288 tests passing">
  <img src="https://img.shields.io/badge/offline-first-555" alt="Offline-first">
</p>

---

**Vincio is a Python platform for building AI applications that you can trust in production.**
It takes everything that goes *into* a model — prompts, memory, retrieved evidence, tools, schemas,
and policies — and compiles it into an optimized, validated, observable **context packet**; then it
checks, measures, and traces everything that comes *out*.

Most libraries help you *call* a model. Vincio governs the **boundary** between your application and
the model: what evidence is selected, how it is scored and budgeted, how the result is validated,
and what it cost. Named for **Leonardo da Vinci** — engineering and craft in equal measure.

```text
Raw input → normalize → retrieve & rank evidence → compile context (score · dedupe · budget)
→ call model → parse & validate → evaluate & guard → trace & cost → learn
```

**Why you'd reach for it**

- **Runs offline, for real.** No API key needed — a deterministic mock provider emits schema-valid
  output, so your whole pipeline (retrieval, validation, evals, traces) runs in CI without network.
- **Deterministic where it counts.** Security, permissions, and validation are enforced in code,
  never gated on model output. The same input compiles to the same packet.
- **Measured, not asserted.** Every run is traced and costed; every change can be gated by an eval
  suite before it ships.
- **One coherent system** from input to output — not a bag of utilities you wire together yourself.

## Contents

[Install](#install) · [Quickstart](#quickstart) · [What you can build](#what-you-can-build) ·
[Features](#features) · [Benchmarks](#benchmarks) · [How Vincio compares](#how-vincio-compares) ·
[Examples](#examples) · [CLI](#command-line) · [Architecture](#architecture) ·
[Docs](#documentation)

## Install

```bash
pip install vincio                  # core — runs fully offline with the mock provider
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

No API key? It runs offline out of the box on a deterministic mock provider that emits schema-valid
output — so the whole pipeline runs for real in CI.

## What you can build

**Typed output you can rely on** — declare a Pydantic schema, get a validated instance back:

```python
from pydantic import BaseModel
from vincio import ContextApp

class Triage(BaseModel):
    label: str
    confidence: float

app = ContextApp(name="triage", output_schema=Triage)
app.run("The dashboard crashes after login").output.label   # → a validated Triage
```

**Agents with tools, memory, and hard budgets** — permissioned tools, approval-gated writes, and a
loop that cannot run away:

```python
app = ContextApp(name="support", output_schema=RefundDecision)
app.add_memory(scope="user", strategy="semantic")
app.add_tool(lookup_order, permissions=["orders:read"])
app.add_tool(issue_refund, permissions=["refunds:write"], approval_required=True)
app.run("Refund my duplicate charge")
```

**Evaluation as a CI gate** — measure quality and block a regression before it ships:

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

## Features

Everything below is implemented, tested offline, and demonstrated by a runnable example. Use the
high-level `ContextApp`, or reach for any engine directly.

**Context & prompts**
- Prompt compiler — typed prompt ASTs with `${variables}`, lint rules, cache-aware stable-prefix
  layout, versioning, hashing, and diffing.
- Context compiler — scores every candidate (relevance, novelty, authority, freshness, provenance,
  token cost, leakage risk), deduplicates, resolves conflicts, compresses, and packs to a token
  budget, with an *excluded-context report* explaining every omission.

**Retrieval & memory**
- Hybrid RAG — BM25 + dense + learned-sparse + late-interaction fused in one weighted RRF; query
  understanding (HyDE, multi-query, decomposition); sentence-window / auto-merging chunking;
  GraphRAG; structured metadata filters with tenant scope; text + image + table + video evidence as
  first-class scored candidates.
- Layered memory — session → episodic → semantic → tenant → graph, with a guarded write pipeline,
  confidence decay, contradiction resolution, bi-temporal recall, per-memory ACLs, and audited
  GDPR-style edit/forget/export.

**Agents & orchestration**
- Tools — permissioned registry (RBAC + ABAC), schema-from-typehints, a resource-limited sandbox,
  idempotent write guardrails with approval callbacks, and a grounded computer-use action plane.
- Agents — bounded DAG execution with planners (ReAct / plan-and-execute / hierarchical HTN),
  in-place plan repair, cost-aware action selection, and a budgeted deep-research agent.
- Orchestration — multi-agent crews with a shared blackboard, durable stateful graphs
  (checkpoint / resume / time-travel / human-in-the-loop), deterministic workflows, and a
  distributed durable-execution backend.

**Output, evaluation & observability**
- Structured output — Pydantic contracts, constrained decoding, streaming validation with early
  abort, bounded self-correction that repairs structure only (never invents facts), and DSPy-style
  typed signatures.
- Evaluation — golden datasets, 30+ metrics, deterministic / model / G-Eval judges, synthetic data,
  red-teaming, trajectory & tool-use scoring, drift detection, regression gates, and a `pytest`
  plugin.
- Observability — full trace span trees, OpenTelemetry export, a local trace viewer, a versioned
  prompt registry, and per-run cost tracking — no account or hosted backend required.

**The closed loop**
- Optimization — one reproducible cycle (trace → dataset → eval → optimize → promote): a reflective
  GEPA/MIPRO optimizer, a distillation flywheel, on-policy reinforcement from verifiable rewards,
  and gated deploy with canary + rollback. No promotion ships without clearing the gates.

**Security & governance**
- Security — deterministic PII / secret redaction (multilingual), prompt-injection defense and
  provable containment (taint tracking + capability tokens), RBAC / ABAC, tenant isolation, and a
  hash-chained, signed audit log with offline tamper verification.
- Governance — model / system cards, an OWASP / NIST / MITRE / ISO compliance matrix, an AI-BOM,
  provable erasure, a consent ledger, data-residency enforcement, formal invariant verification,
  agent identity & delegation, verified-reasoning certificates, and continuous assurance cases.

**Interop**
- Protocols — MCP (client *and* server), A2A agent-to-agent, and Agent Skills, all in-process.
- Ecosystem — import/export LangChain, LlamaIndex, Haystack, and DSPy assets; first-party data
  connectors; and any OpenAI-compatible model or vector store you already run.

> Reach further: a cross-organization agent economy (negotiation, contracts, durable sagas,
> metering, settlement, arbitration, reputation, collateral & solvency proofs), an edge / WASM
> in-process runtime, on-device LoRA adaptation, federated learning with a differential-privacy
> accountant, and per-run energy / carbon accounting. See [`ROADMAP.md`](ROADMAP.md).

## Benchmarks

Three suites ship in [`benchmarks/`](benchmarks), all reproducible on your own machine.

### Head-to-head vs. real libraries

[`competitive.py`](benchmarks/competitive.py) runs Vincio against the *actual* library a team would
otherwise use. **Every number is measured live from both sides** — a missing competitor is reported
as skipped, never assumed. Representative results (Apple Silicon, Python 3.13; *ratios* are the
portable signal, not wall-clock):

| Operation | Vincio | Competitor | Result |
|---|---|---|---|
| BM25 query @ **20k docs** | `BM25Index` | `rank_bm25` | **~30–40× faster**, identical top-1 ranking |
| **Context assembly** — tokens sent for the same retrieved set | context compiler | LangChain `stuff` / LlamaIndex `compact` | **~60% fewer tokens**, answer retained |
| Text chunking a 24k-word doc | `chunk_document` | LangChain / LlamaIndex splitters | **fastest**, chunks carry provenance |
| Token counting (~60k words) | `HeuristicTokenCounter` | `tiktoken` | **~1.4–1.8× faster**, zero-dependency, conservative |
| Malformed-JSON recovery | lenient parser | stdlib `json.loads` | **4/8 vs 1/8** recovered |
| Render with a missing variable | `PromptSpec.substitute` | `jinja2` | typed error vs. silently-empty render |

`rank_bm25` rescans every document per query; Vincio's inverted index only scans documents
containing a query term, so its lead grows with corpus size. The point isn't that every component
beats every specialist — a dedicated JSON-repair library recovers more than Vincio (by guessing,
which is unsafe for typed extraction). Vincio's edge is an **integrated, correct, governed**
pipeline, not a pile of single-purpose libraries.

### Orchestrator uplift — the same model, through Vincio

[`quality_uplift.py`](benchmarks/quality_uplift.py) measures what routing a model *through* Vincio
adds versus calling it directly — like every example and test, it runs deterministic on the mock and
**against a real model when a provider is configured** (`VINCIO_PROVIDER=openrouter …`). These
mechanism metrics are **deterministic** and measured offline:

| Same model — direct vs. via Vincio | Direct | Via Vincio |
|---|--:|--:|
| Schema-valid object from realistic model outputs | 1/6 | **5/6** |
| Prompt-injection exfiltration via a tool call | compromised | **contained** |
| Context tokens to retain an early fact at 80 turns | 640 (lost) | **33 (retained)** |

**Grounded-answer quality, measured on real models** (10 company-specific policy questions a model
cannot know from pretraining; OpenRouter, June 2026):

| Model (called directly vs. through Vincio) | Direct correct | Direct hallucinated | **Via Vincio correct** | Cited |
|---|--:|--:|--:|--:|
| `openai/gpt-4o-mini` | 1/10 | 8/10 | **10/10** | 10/10 |
| `anthropic/claude-3-haiku` | 0/10 | 0/10¹ | **8/10** | 10/10 |
| `google/gemini-2.5-flash-lite` | 0/10 | 4/10 | **9/10** | 10/10 |
| `meta-llama/llama-3.1-8b-instruct` | 0/10 | 8/10 | **9/10** | 9/10 |

<sub>¹ claude-3-haiku abstains rather than guessing — older/smaller models confidently fabricate
instead. Either way the model *alone* answers ~0/10; the same model through Vincio's retrieval +
grounding answers 8–10/10, every answer carrying a citation.</sub>

The token-usage and context-rot results are the heart of the deterministic story: a keep-everything
agent grows its context linearly until the early fact falls out of the window, while Vincio's
budgeted compiler and bounded recall hold the footprint flat and keep the relevant evidence in view —
the same mechanism that sends ~60% fewer tokens for the same retrieved set, on every call. Reproduce:
`VINCIO_PROVIDER=openrouter VINCIO_MODEL=openai/gpt-4o-mini OPENROUTER_API_KEY=… python
benchmarks/quality_uplift.py`.

### VincioBench — the internal regression suite

[`vinciobench.py`](benchmarks/vinciobench.py) is the third suite, but it is **not a competitive
claim** — it is the deterministic mechanism suite that gates CI. Its families assert that each engine
still *works* on a bundled synthetic corpus (the context compiler reduces tokens, the injection
detector fires, retrieval returns the known answer), so a regression fails the build. Because that
corpus is small and built to exercise each mechanism, the scores saturate — perfect recall, full
detection on a handful of cases against a naive baseline — which proves *the mechanism is intact*,
not real-world performance. **The credible performance evidence is the two sections above** (real
libraries, and real models). Run `python benchmarks/vinciobench.py` against your own corpus and trust
only what it prints; [`benchmarks/README.md`](benchmarks/README.md) documents what each family
measures and on what corpus size.

## How Vincio compares

Each ecosystem below is strong in its focus area. This reflects **built-in, in-library**
capability — not what's reachable by adding a separate product or SaaS.

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

<sub>✅ first-class in-library · ➖ partial or via an add-on/SaaS · ❌ not a focus. Ecosystems evolve,
and Vincio is built to *interoperate* — `vincio.interop` brings LangChain, LlamaIndex, Haystack, and
DSPy assets in (and hands Vincio's back). See the in-depth write-ups in
[`docs/comparisons/`](docs/comparisons).</sub>

## Examples

Twelve complete, heavily-commented programs in [`examples/`](examples) — each runs **fully offline**
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

## Status

Vincio 4.0 is **feature-complete and in long-term support**. The public API is frozen under
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a mechanical
[deprecation policy](docs/reference/stability.md); performance and quality targets are
[published as SLOs](docs/reference/slo.md) and gated by VincioBench; releases ship a CycloneDX SBOM
with SLSA provenance. New capabilities are added behind opt-in extras, never by breaking working
code. See [`ROADMAP.md`](ROADMAP.md) and [`MIGRATION.md`](MIGRATION.md).

Vincio is, and stays, a **library**. The building blocks for production (audit chain, retention,
tenant isolation, RBAC/ABAC, a server) ship in the package for you to deploy on your own
infrastructure. There is no hosted service.

## Documentation

- **[Getting started](docs/getting-started.md)** — install, your first app, offline development
- **Concepts** — [context packets](docs/concepts/context-packets.md) ·
  [prompt compiler](docs/concepts/prompt-compiler.md) · [memory](docs/concepts/memory.md) ·
  [retrieval](docs/concepts/retrieval.md) · [agents & workflows](docs/concepts/agents.md) ·
  [evaluation](docs/concepts/evals.md) · [observability](docs/concepts/observability.md)
- **Guides** — [build a RAG app](docs/guides/build-rag-app.md) ·
  [structured output](docs/guides/structured-output.md) ·
  [add tools](docs/guides/add-tools.md) ·
  [orchestrate multi-agent systems](docs/guides/orchestrate-agents.md) ·
  [run evals](docs/guides/run-evals.md) · [close the loop](docs/guides/close-the-loop.md) ·
  [performance & streaming](docs/guides/performance.md) · [integrations](docs/guides/integrations.md)
- **Protocols** — [MCP client + server](docs/guides/mcp.md) · [A2A](docs/guides/a2a.md) ·
  [Agent Skills](docs/guides/agent-skills.md) · [reasoning control](docs/guides/reasoning.md)
- **Migrating** — from [LangChain](docs/guides/migrate-from-langchain.md) ·
  [LlamaIndex](docs/guides/migrate-from-llamaindex.md) · [Ragas](docs/guides/migrate-from-ragas.md)
- **Security & governance** — [threat model](docs/security/threat-model.md) ·
  [security policy](SECURITY.md) · [governance & compliance](docs/guides/governance.md)
- **Reference** — [API](docs/reference/api.md) · [CLI](docs/reference/cli.md) ·
  [config](docs/reference/config.md) · [SLOs](docs/reference/slo.md) ·
  [stability & deprecation](docs/reference/stability.md)
- **Comparisons** — [LangChain](docs/comparisons/langchain.md) ·
  [LlamaIndex](docs/comparisons/llamaindex.md) · [DSPy](docs/comparisons/dspy.md) ·
  [CrewAI](docs/comparisons/crewai.md) · [Ragas](docs/comparisons/ragas.md) ·
  [and more](docs/comparisons)

## Contributing

Contributions are welcome. The test suite runs fully offline and must stay green:

```bash
pip install -e ".[dev]"
python -m pytest -q          # 3288 tests, no network or API keys required
ruff check vincio/ tests/
mypy vincio
```

See [`AGENTS.md`](AGENTS.md) for the codebase layout and engineering conventions.

## License

[Apache License 2.0](LICENSE) © Vincio Contributors.
