# Observability

Every run produces a trace, in your process, in the same data model as the
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
threads them automatically. Sessions are a derived view; group any list of
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

## Online evaluators and drift

The same scores can be earned continuously, off the hot path. An online
evaluator scores a sampled fraction of live runs after the response is
finalized and writes each score as a time series to the metadata store
(kind `eval_results`); `OnlineEvaluator.series()` returns the rows, so a
metric becomes a trend you can watch, not just a per-run number:

```python
app.add_online_evaluator("goal_accuracy", sample_rate=0.2)
app.online_evaluators[0].series()   # the score time series
```

`DriftMonitor` reads those series against a baseline. When live scores or
embeddings move past threshold it raises a `drift.detected` event on the bus
and persists the comparison (kind `drift_baselines`); drift is itself an
observable signal, in the same store as everything else.

A traced agent run also carries a `Trajectory`, the steps, tool calls, and
termination it took to get there. Project it onto a `RunOutput` and the run
becomes scorable without re-instrumentation:

```python
run = RunOutput.from_agent_state(state)   # from app.agent(...).run(...)
```

so the trace that recorded *what happened* feeds the metrics that judge
*how good the path was*, the same scores/feedback narrative, now over the
whole trajectory.

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

The HTML is one file with inline CSS, no server, no account, no external
assets; mail it, attach it to a PR, or open it from CI artifacts.

## OpenTelemetry GenAI conventions

`OTelExporter` (extra: `vincio[otel]`) re-emits traces through any OTLP
backend. Model and tool spans follow the **GenAI semantic conventions**,
`chat {model}` / `execute_tool {tool}` span names, `gen_ai.request.model`,
`gen_ai.usage.input_tokens` / `output_tokens`, `gen_ai.response.finish_reasons`,
`gen_ai.conversation.id` for sessions, alongside the full `vincio.*`
attribute set, so GenAI-aware backends (Jaeger, Datadog, Grafana, Honeycomb)
render them natively.

## Costs

`CostTracker` prices model calls from a configurable price table; costs ride
on model spans and aggregate per run (`result.cost_usd`) and per report.

## Causal record-replay

A trace tells you *what* a run did; a **recording** lets you re-run it. A run is
deterministic except at its edges, every place it reads the outside world, so
the recorder captures exactly those: model responses, tool outputs, retrieval
hits, the capabilities a request was negotiated against, and the clock/seed,
each keyed to its trace span.

```python
from vincio.observability import Recorder, Replayer, BranchEdit

# Record: run normally while every non-deterministic edge is captured.
result, recording = await Recorder(app).record("What is the refund policy?")
recording.save(".vincio/recordings/refund.json")   # portable + content-addressed
assert recording.verify()                            # fidelity digest checks out

# Replay: serve every edge back so the run reproduces byte-for-byte.
replay = await Replayer(app).replay(recording)
assert replay.faithful and replay.output_identical

# Divergence: when the code changes, the recorded edge no longer matches.
report = await Replayer(changed_app).replay(recording)
for d in report.divergences:
    print(d.kind, d.detail)   # exactly where the run drifted

# Branch-and-edit: change one edge and re-execute only the affected suffix.
branch = await Replayer(app).branch(
    recording,
    edits=[BranchEdit(kind="tool_call", key=recording.tool_calls[0].key,
                      value={"call_id": "x", "tool_name": "lookup",
                             "status": "ok", "output": "EDITED"})],
)
```

Each edge is keyed by the same identity the live code computes (a model call by
its `ModelRequest.hash`, a tool call by its name + arguments), so a changed edge
is a cache miss, and a miss is a **divergence**, reported with the edge that
drifted instead of silently re-executed. In branch mode the unchanged prefix is
still served from the recording while only the affected suffix re-executes, so a
fix is validated against the exact failing run. Recordings are content-addressed
and verifiable (`recording.put(store)` / `Recording.from_store`,
`recording.verify()`); `vincio trace verify-recording <file>` checks one offline.
