# Observability

Every run produces a trace — in your process, in the same data model as the
runtime. There is no platform to send data to: traces export to JSONL,
memory, the console, or OpenTelemetry, and the viewer is a terminal renderer
plus a self-contained static HTML file.

## Traces and spans

A `Trace` covers one run; `Span`s cover pipeline stages (`input`, `retrieval`,
`context_compile`, `prompt_render`, `model_call`, `tool_call`,
`output_validation`, `eval`, ...). Nesting is implicit via `contextvars`, so
it is correct under asyncio concurrency:

```python
with tracer.trace(run_id="r1", session_id="sess_1", user_id="u1") as trace:
    with tracer.span("retrieval", type="retrieval") as span:
        span.set(query="termination clauses", top_k=8)
```

## Sessions and threaded runs

Traces carry `session_id` / `thread_id`; `ContextApp.run(..., session_id=...)`
threads them automatically. Sessions are a derived view — group any list of
traces, no second store to sync:

```python
from vincio.observability import sessions_from_traces

sessions = sessions_from_traces(exporter.load_all())
sessions[0].summary()   # runs, duration, error_rate, mean scores, feedback
```

## Scores and feedback

Eval scores attach to spans and traces (`span.add_score`, `trace.add_score`);
runtime evaluators do this automatically, so a trace shows *how good* the run
was, not just how long it took. User feedback is first-class:

```python
trace.add_feedback(score=1.0, comment="clear answer")          # in process
record_feedback(trace, score=1.0, exporter=exporter)           # persist update
```

```bash
vincio trace feedback <trace_id> --score 1.0 --comment "clear answer"
```

## Traces become datasets

The bridge from observability to evaluation is one call (or one command):

```python
golden = dataset_from_traces(exporter.load_all(), min_feedback_score=0.5)
```

```bash
vincio eval dataset golden.jsonl --min-feedback 0.5
```

Each case keeps full provenance: trace id, run id, session id, and the scores
the run earned.

## Local trace viewer

```bash
vincio trace view <trace_id>                 # TUI tree with scores + feedback
vincio trace export <trace_id>               # self-contained static HTML
vincio trace export <session_id> --session   # whole session as one page
vincio trace diff <a> <b> --html diff.html   # visual side-by-side diff
vincio trace sessions                        # session list with aggregates
```

The HTML is one file with inline CSS — no server, no account, no external
assets; mail it, attach it to a PR, or open it from CI artifacts.

## OpenTelemetry GenAI conventions

`OTelExporter` (extra: `vincio[otel]`) re-emits traces through any OTLP
backend. Model and tool spans follow the **GenAI semantic conventions** —
`chat {model}` / `execute_tool {tool}` span names, `gen_ai.request.model`,
`gen_ai.usage.input_tokens` / `output_tokens`, `gen_ai.response.finish_reasons`,
`gen_ai.conversation.id` for sessions — alongside the full `vincio.*`
attribute set, so GenAI-aware backends (Jaeger, Datadog, Grafana, Honeycomb)
render them natively.

## Costs

`CostTracker` prices model calls from a configurable price table; costs ride
on model spans and aggregate per run (`result.cost_usd`) and per report.
