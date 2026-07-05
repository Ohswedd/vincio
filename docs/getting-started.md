# Getting started

Vincio is a Python platform for context-engineered AI applications. It compiles prompts, memory,
retrieval, tools, schemas, and policies into optimized, validated, observable **context packets**, then
checks, measures, and traces every output. This page takes you from install to a grounded, cited,
eval-gated app — most of it **without an API key**.

## Install

```bash
pip install vincio                # core — runs fully offline on the mock provider
pip install "vincio[openai]"      # + a provider (also: anthropic, google, mistral)
pip install "vincio[chroma]"      # + a vector store (also: pinecone, lancedb, postgres, …)
pip install "vincio[server]"      # + the FastAPI server (vincio serve)
pip install "vincio[all]"         # every optional integration
```

Python 3.11+. The core depends only on `pydantic`, `httpx`, `typing-extensions`, and `pyyaml`; every
heavy integration is an opt-in extra, so `import vincio` stays light and offline. See
[integrations](guides/integrations.md) for the full extras matrix.

## Scaffold a project

```bash
vincio init my-project --template rag   # or: minimal | agent | eval
cd my-project
```

`init` writes `vincio.yaml` (layered config, with a JSON Schema hint for editor completion), `app.py` (a
starter app), `vincio.schema.json`, and a starter golden eval dataset. Use `--provider groq` (or any
[OpenAI-compatible preset](guides/integrations.md)) to target a different model.

## Your first app

```python
from vincio import ContextApp

app = ContextApp(name="docs_qa", provider="openai", model="gpt-4o-mini")
app.add_source("docs", path="./docs", retrieval="hybrid")   # index a folder: chunk + hybrid retrieval
app.set_policy("answer_only_from_sources", True)            # the model may only use retrieved evidence

result = app.run("How do I configure SSO?")
print(result.output)        # the grounded answer
print(result.citations)     # the exact evidence it cited
print(result.trace_id)      # every run produces a full trace
print(result.cost_usd)      # …and a cost
```

That single `run()` is one coherent pipeline, not a prompt string. Summarized:

1. **Normalize & classify** — the input is normalized, language-detected, and typed by task.
2. **Gate** — policies run: injection detection, PII redaction (if enabled), scope checks.
3. **Recall & retrieve** — memory and retrieval produce candidate evidence.
4. **Compile context** — the **context compiler** scores every candidate (relevance, novelty, authority,
   freshness, provenance, token cost, duplication, leakage risk), dedupes, resolves conflicts,
   compresses, and packs the winners into a token budget — emitting an *excluded-context report* for
   everything it dropped.
5. **Compile prompt** — a cache-aware prompt (stable prefix, volatile suffix), lint-checked.
6. **Model** — runs provider-neutral, with a bounded tool loop if tools are registered.
7. **Validate** — parse, schema-validate, citation-check, policy-check, with principled repair that
   fixes *structure only, never facts*.
8. **Measure** — evaluators score the run; memory is updated; a hash-chained audit entry and a full
   trace are written.

> **Best practice — grounding.** `answer_only_from_sources` + a groundedness evaluator is what turns
> "the model said so" into "the evidence says so." Add `app.add_evaluator("groundedness")` and
> `app.add_evaluator("citation_accuracy")` so every run is scored, not just answered.

Prefer streaming? The identical pipeline streams end to end:

```python
async for event in app.astream("How do I configure SSO?"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
    elif event.type == "done":
        result = event.result
```

See the [performance & streaming guide](guides/performance.md) for stages, partial-JSON output, and the
server SSE endpoint.

## Develop offline — no key, no cost

This is the workflow that makes Vincio pleasant to build on: pass the bundled deterministic
`MockProvider` and the **whole pipeline** — retrieval, validation, evals, traces, cost — runs for real
with no network and no key.

```python
from vincio.providers import MockProvider

app = ContextApp(name="dev", provider=MockProvider(), model="mock-1")
```

With an `output_schema` set, the mock synthesizes a schema-valid instance, so validation and evals
exercise real code paths. When you're ready for a real model, change *nothing* in your logic — set
`VINCIO_PROVIDER` + the matching key in the environment (or pass `provider=`/`model=`). The same code you
tested offline is the code that ships.

> **Best practice — offline-first.** Build and test against the mock; keep it as your CI provider so the
> suite needs no secrets. Point at a real model only at the edges (a manual smoke test, staging).

## Typed output

Declare a Pydantic schema and get a validated instance back — attribute access, IDE completion, and a
validation error if the model strays off-schema:

```python
from pydantic import BaseModel

class TicketClassification(BaseModel):
    label: str
    confidence: float
    reason: str

app = ContextApp(name="triage", output_schema=TicketClassification)
result = app.run("The dashboard crashes after login")
print(result.output.label)   # a validated TicketClassification
```

## The one-liner shortcut

For the jobs you reach for most, `vincio.tasks` is one expression each — and it **lowers to the exact
same governed run** as the builder above (grounding, validation, rails, tracing all apply). `.app` is the
escape hatch to every deep method:

```python
from vincio import rag
rag("./docs").ask("How do I configure SSO?")   # grounded, cited, eval-scored — same packet as ContextApp
```

## Run an eval, gate on it

```bash
vincio eval run golden/basic.jsonl --app app.py \
    --metric groundedness --metric citation_accuracy \
    --gate "groundedness=>= 0.9" --output report.json
```

The gate makes quality a build check: the command exits non-zero if groundedness drops below the
threshold, so a regression fails CI instead of shipping. See [run evals](guides/run-evals.md) and
[test LLM apps with pytest](guides/test-llm-apps.md).

## Next steps

Follow the **[learning path](learning-path.md)** — a staged route from this first app through the core
model, building a real application, evaluation, orchestration, governance, and the cross-organization
economy. It is the spine; each stage links the concepts, guides, and examples in order.

For the exhaustive map see the [documentation index](README.md); for every `app.*` verb and the page
that documents it, the [capability map](reference/capability-map.md). Jumping straight in:

- [Concepts: context packets](concepts/context-packets.md) and [build a RAG app](guides/build-rag-app.md).
- [Add tools](guides/add-tools.md), [structured output](guides/structured-output.md), and [reliability & guardrails](guides/reliability-guardrails.md).
- [Run evals](guides/run-evals.md) and [test LLM apps with pytest](guides/test-llm-apps.md).
- [Reference: configuration](reference/config.md).
