# The ergonomic surface — the one-line 'ad-hoc' front door

The platform is feature-complete, but its power is broad: a
[`ContextApp`](../../vincio/core/app.py) carries a couple hundred methods, and the
five jobs a newcomer actually has — grounded **RAG Q&A**, a **tool-using agent**,
**structured extraction**, an **eval**, and a **multi-step flow** — each take a
fistful of string-keyed builder calls. The canonical RAG path alone is six coupled
calls (`add_source(chunking=, retrieval=)` + `set_policy("answer_only_from_sources", True)`
+ two `add_evaluator(...)` + `run`), where an LCEL chain, a LlamaIndex query
engine, a DSPy module, or a Haystack pipeline each cost one line.

The [`vincio.tasks`](../../vincio/tasks/__init__.py) namespace is the missing *top
layer* — not a new capability. It is a small, discoverable set of task-shaped
constructors, each a **purely-compositional facade** in the proven
[`Assistant`](../../vincio/assistant.py) /
[`CrossOrgEngagement`](../../vincio/settlement/engagement.py) /
[`DataEngagement`](../../vincio/data/engagement.py) mold: it configures a
`ContextApp` with sane governed defaults using the *same* public builder calls a
caller would make by hand, so the one-liner **lowers to the exact same governed
`ContextApp.run` packet** as the verbose form. Retrieval, grounding, validation,
rails, budgets, tracing, and the audit chain all apply unchanged. The common case
is one expression; **`.app` is the escape hatch** to every deep method for the
complex case (nothing is shadowed, nothing is unreachable).

The symbols are [`@experimental`](../reference/stability.md) until their shape
settles. They are re-exported at the top level (`from vincio import rag, Flow`) and
live in `vincio.tasks`; the concrete facade types (`RagTask`, `Extractor`,
`ToolAgent`, `Evaluation`) live in the namespace. Everything is deterministic,
dependency-free, and offline — never a hosted playground, a managed quickstart, or
a GUI builder.

## The five one-liners

```python
from vincio import rag, extractor, tool_agent, evaluation, chat, Flow

# 1. Grounded RAG Q&A — retrieve, ground, cite, eval, in one expression.
answer = rag("./docs").ask("What is the refund window for the Pro plan?")
print(answer.output, answer.citations, answer.eval_scores)

# 2. Typed structured extraction — text in, a validated Pydantic object out.
ticket = extractor(TicketClassification).extract("I was charged twice this month.")

# 3. Approval-gated tool agent — writes are denied until you approve them.
agent = tool_agent(tools=[search_docs], writes=[create_ticket])
result = agent.run("Open a ticket for the duplicate charge")

# 4. An offline eval — metrics and gates bundled with the dataset.
report = evaluation(golden, metrics=["groundedness"], gates={"groundedness": ">= 0.8"}).run()

# 5. A multi-turn chat — a re-presentation of app.assistant.
bot = chat()
print(bot.send("What's my refund window? My plan is Pro.").text)
```

And the fluent, immutable [`Flow`](../../vincio/tasks/_flow.py) — the Vincio answer
to LCEL — threads the same pipeline as a value (every step returns a new Flow):

```python
answer = (
    Flow(provider=p, model=m)
    .retrieve("./docs", chunking="adaptive")
    .ground()
    .evaluate("groundedness", "citation_accuracy")
    .run("What is the refund window for the Pro plan?")
)
```

## Each one-liner maps to the deep methods it composes

The front door spells builder calls; it never adds behavior. This is the exact
desugaring of each constructor — the verbose path you can drop down to at any time
through `.app`.

| One-liner | Lowers to (the verbose `ContextApp` calls) |
|---|---|
| `rag(sources, evaluators=("groundedness", "citation_accuracy"))` | `ContextApp(...)` + `add_source("docs", path=…, chunking=…, retrieval=…)` + `set_policy("answer_only_from_sources", True)` + `add_evaluator("groundedness")` + `add_evaluator("citation_accuracy")`; `.ask(q)` → `app.run(q)` |
| `extractor(schema)` | `ContextApp(output_schema=schema)`; `.extract(text)` → `app.run(text).output` |
| `tool_agent(tools=[…], writes=[…], approve=[…])` | `add_tool(t)` per read tool + `add_tool(t, approval_required=True, side_effects="write")` per write tool, behind the same approval surface `app.assistant` installs; `.run(task)` → `app.run(task)` (the governed model+tool loop) |
| `evaluation(dataset, metrics=[…], gates={…})` | `add_evaluator(m)` per metric; `.run()` → `app.evaluate(dataset, gates=…)` |
| `chat(tools=[…], writes=[…], approve=[…])` | `add_tool(...)` for each tool + `app.assistant(auto_approve=…)` |
| `Flow.retrieve(...).ground().call(...).validate(...).evaluate(...)` | `add_source(...)` / `set_policy("answer_only_from_sources", …)` / `configure(...)` + model / `output_schema` contract + `set_policy("require_citations", …)` / `add_evaluator(...)`; `.run(input)` → `app.run(input)` |

Each constructor also accepts an optional persona (`role` / `objective` / `rules`,
applied via `app.configure(...)`) and an `app=` argument that layers the task onto a
pre-configured `ContextApp` — the escape hatch, inbound.

## No behavioral fork — the one-liner *is* the verbose form

Because a constructor only replays public builder calls, the ad-hoc form lowers to
a byte-identical packet and `RunResult`. The proof is mechanical: the shared
[`run_signature`](../../vincio/testing/lowering.py) harness (the same one the 5.2
single-pass feature arena uses to prove selection byte-identity) projects a run to
its packet `spec_hash` plus its stable outputs (output, citations, eval scores,
token usage), and the verbose form and the one-liner produce equal signatures when
given the same inputs:

```python
from vincio.testing import run_signature

verbose = run_signature(verbose_app, "What is the refund window?")
ad_hoc  = run_signature(rag(documents).app, "What is the refund window?")
assert verbose == ad_hoc   # same packet, same result — no behavioral fork
```

This is held by the **ErgonomicsBench** VincioBench family — a *conciseness* SLO
(each use case is one entry point, benchmarked head-to-head against LCEL, the
LlamaIndex query engine, DSPy modules, and Haystack pipelines in
`benchmarks/competitive.py`), a *compiles-byte-identical* SLO (the ad-hoc form
lowers to the verbose form's packet and result), and an *escape-hatch-total* SLO
(`.app` reaches every deep method).

## The escape hatch is total

Every facade exposes `.app`, the fully-configured `ContextApp`. Anything the
verbose path can do, the one-liner can reach — there is nothing the front door
hides:

```python
task = rag("./docs")
task.app.add_rail(name="no_pii", kind="safety", detectors=["pii"])
task.app.enable_self_correction(max_cycles=2)
task.app.add_memory()
answer = task.ask("What is the refund window?")   # all of the above now apply
```

## See also

- [`examples/00_one_liners.py`](../../examples/00_one_liners.py) — every one-liner, runnable and offline, before the quickstart.
- [Build a RAG app](../guides/build-rag-app.md) — the verbose RAG path `rag(...)` composes.
- [The Assistant](../guides/assistant.md) — what `chat(...)` re-presents.
- [Structured output](../guides/structured-output.md) — the contract `extractor(...)` builds on.
- [Run evals](../guides/run-evals.md) — the evaluation path `evaluation(...)` bundles.
- [API stability](../reference/stability.md) — what `@experimental` means for these symbols.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: Cookbook: task-shaped recipes](../guides/cookbook.md)
- [Example: 00_one_liners.py](../../examples/00_one_liners.py)
- [Concept: Prompt compiler](prompt-compiler.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
