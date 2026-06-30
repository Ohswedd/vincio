# Getting started

Vincio is a Python platform for context-engineered AI applications. It
compiles prompts, memory, retrieval, tools, schemas, and policies into
optimized, validated, observable **context packets**.

## Install

```bash
pip install vincio                # core (works fully offline with the mock provider)
pip install "vincio[openai]"      # OpenAI
pip install "vincio[anthropic]"   # Anthropic
pip install "vincio[all]"         # everything
```

Python 3.11+ is required.

## Initialize a project

```bash
vincio init my-project --template rag   # or: minimal | agent | eval
cd my-project
export OPENAI_API_KEY=sk-...
```

This creates `vincio.yaml` (configuration, with a JSON Schema hint for editor
completion), `app.py` (a starter app), `vincio.schema.json`, and a starter
golden eval dataset. Use `--provider groq` (or any
[OpenAI-compatible preset](guides/integrations.md)) to target a different model.

## First app

```python
from vincio import ContextApp

app = ContextApp(name="docs_qa")
app.add_source("docs", path="./docs", retrieval="hybrid")
app.set_policy("answer_only_from_sources", True)

result = app.run("How do I configure SSO?")
print(result.output)        # the answer, with citations
print(result.citations)     # evidence refs the answer cited
print(result.trace_id)      # every run has a trace
print(result.cost_usd)      # and a cost
```

What happened inside that single call — Vincio's 17-step run pipeline, summarized:

1. Your input was normalized, language-detected, and classified by task type.
2. Policies ran (injection detection, PII redaction if enabled).
3. Memory and retrieval produced candidate context.
4. The **context compiler** scored every candidate (relevance, novelty,
   authority, freshness, provenance, token cost, duplication, leakage risk),
   removed duplicates, resolved conflicts, compressed where needed, and
   packed the winners into a token budget.
5. The **prompt compiler** rendered a cache-friendly prompt (stable prefix,
   volatile suffix) with lint checks.
6. The model ran (with bounded tool loops if tools are registered).
7. Output was parsed, schema-validated, citation-checked, and policy-checked,
   with principled repair (structure only, never facts).
8. Evaluators scored the run, memory was updated, an audit entry and a full
   trace were written.

Prefer streaming? The same pipeline streams end to end:

```python
async for event in app.astream("How do I configure SSO?"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
    elif event.type == "done":
        result = event.result
```

See the [performance & streaming guide](guides/performance.md) for stages,
partial-JSON output, and the server SSE endpoint.

## Typed output

```python
from pydantic import BaseModel

class TicketClassification(BaseModel):
    label: str
    confidence: float
    reason: str

app = ContextApp(name="triage", output_schema=TicketClassification)
result = app.run("The dashboard crashes after login")
print(result.output.label)   # a validated TicketClassification instance
```

## Offline development

The deterministic mock provider lets you build and test with zero API calls:

```python
from vincio.providers import MockProvider
app = ContextApp(name="dev", provider=MockProvider(), model="mock-1")
```

With an `output_schema` set, the mock generates schema-valid instances, so
your whole pipeline (validation, evals, traces) runs for real.

## Run evals

```bash
vincio eval run golden/basic.jsonl --app app.py \
    --metric groundedness --metric citation_accuracy \
    --gate "groundedness=>= 0.9" --output report.json
```

## Next steps

Follow the **[learning path](learning-path.md)** — a staged route from this first
app through the core model, building a real application, evaluation, orchestration,
governance, and the cross-organization economy. It is the spine; each stage links
the concepts, guides, and examples in the order you need them.

For the exhaustive map see the [documentation index](README.md); for every
`app.*` verb and the page that documents it, the
[capability map](reference/capability-map.md). Jumping straight in:

- [Concepts: context packets](concepts/context-packets.md) and [build a RAG app](guides/build-rag-app.md).
- [Add tools](guides/add-tools.md), [structured output](guides/structured-output.md), and [reliability & guardrails](guides/reliability-guardrails.md).
- [Run evals](guides/run-evals.md) and [test LLM apps with pytest](guides/test-llm-apps.md).
- [Reference: configuration](reference/config.md).
