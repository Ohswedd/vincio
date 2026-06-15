# Vincio vs LangSmith / Langfuse

LangSmith and Langfuse are observability platforms: tracing, prompt
management, datasets, experiments, and feedback — as a hosted (or
self-hosted) service you send data to.

**Where Vincio differs**

- **In your process, no platform** — traces, sessions, feedback, scores,
  prompt versions, datasets, and experiments live in the same library and
  data model as the runtime. Nothing leaves your machine unless you export
  it; everything works offline and in CI.
- **Sessions, feedback, scored spans** — traces carry `session_id` /
  `thread_id`, user feedback (`trace.add_feedback`, `vincio trace feedback`),
  and eval scores attached to spans and traces by the runtime evaluators.
- **A viewer you can attach to a PR** — `vincio trace export` writes one
  self-contained static HTML file (inline CSS, no server, no account) for a
  trace or a whole session; `vincio trace diff --html` renders two traces
  side by side with the structural diff highlighted.
- **Traces become datasets in one command** — `vincio eval dataset
  golden.jsonl --min-feedback 0.5` curates production runs (with provenance
  and scores) into an eval set; the eval loop, experiments with statistical
  significance, and the optimizer then run on the same objects.
- **Prompt registry with eval links** — versioned prompts with tags, diffs
  (field-level and rendered), rollback, and eval runs linked to the exact
  version they measured — file-backed, reviewable in git.
- **Provider-neutral export** — the OpenTelemetry exporter follows the GenAI
  semantic conventions, so any OTLP backend renders Vincio runs natively;
  you are not locked to one vendor's trace format.
- **Scoring runs in-process, not on a platform** — these platforms send your
  traces somewhere to score them; Vincio scores the trajectory in the same
  process and data model as the runtime. A captured trace becomes a scorable
  run with no re-instrumentation (`RunOutput.from_trace(trace)`,
  `RunOutput.from_agent_state(state)`), and the same metric object scores
  offline, gates CI, and — uniquely — becomes a runtime guardrail
  (`app.add_metric_rail("toxicity", threshold=0.0)`) or an optimizer fitness
  term.
- **Trajectory and tool-use metrics** — `tool_call_accuracy` / `tool_call_f1`,
  `goal_accuracy`, `plan_adherence`, `plan_quality`, `step_efficiency`, and
  `topic_adherence` score *how* a run reached its answer, not just the final
  text. `EvalReport.metric_families()` shows output-only and trajectory eval
  side by side, so you can see a run that answered right while taking the
  wrong path — something final-output scoring can't.
- **Online / continuous eval** — `app.add_online_evaluator("goal_accuracy",
  sample_rate=0.2)` scores a sampled fraction of live runs off the hot path
  and writes each score as a time series to the local store; no external
  mirroring, no second platform.
- **Drift detection** — `DriftMonitor` compares live scores and embeddings to
  a baseline (`check_scores` / `check_embeddings`), raises a `drift.detected`
  event, and gates in CI (`vincio eval drift baseline.json current.json`).
- **A/B with significance** — `app.experiment(...)` runs variants against a
  golden set and reports per-metric means, cost per variant, and a paired
  significance test (`exp.significance("goal_accuracy")`) — in your process.
- **Annotation queues with Cohen's κ** — `AnnotationQueue` pairs judge scores
  with human labels and reports `cohens_kappa`; an LLM judge only earns a
  gating weight once its calibrated κ clears the bar
  (`judge.gating_weight(threshold=0.6)`), so judges gate CI only after
  demonstrably agreeing with people. `vincio eval annotate labels.jsonl`.

**Where LangSmith / Langfuse are a fit:** hosted multi-team dashboards,
long-retention storage, and org-wide annotation queues. Vincio's OTel export
can feed the same backends those platforms read from.
