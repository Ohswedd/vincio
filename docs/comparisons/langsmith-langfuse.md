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

**Where LangSmith / Langfuse are a fit:** hosted multi-team dashboards,
long-retention storage, and org-wide annotation queues. Vincio's OTel export
can feed the same backends those platforms read from.
