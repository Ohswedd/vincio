# Vincio vs Ragas

Ragas is a focused, well-designed evaluation framework for RAG/LLM systems.

**Where Vincio differs**

- **Evaluation is wired into the runtime**, not a separate harness: every
  run can be scored (`app.add_evaluator("groundedness")`), every eval case
  runs through the same 17-step pipeline users hit in production, and every
  failing case links to a full trace.
- **Eval results drive optimization** — the evolution loop consumes the same
  reports and promotes prompt/context/routing changes only through safety
  gates.
- **Coverage beyond RAG**: agent metrics, tool reliability, memory quality
  (staleness, contradiction, privacy), schema validity, cost/latency — one
  report format, one gate mechanism, one CI command.
- **Deterministic metrics first**, model judges second (with repeated-sample
  calibration), keeping evaluation reliable and reproducible.

**Where Ragas is a fit:** research-grade RAG metric variants and synthetic
test-set generation. Ragas metrics can be wrapped as custom Vincio metrics
via `@register_metric` in a few lines.
