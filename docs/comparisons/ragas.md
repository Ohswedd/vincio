# Vincio vs Ragas

Ragas is a focused, well-designed evaluation framework for RAG/LLM systems.

**Where Vincio differs**

- **Metric parity, in-process** — faithfulness, answer relevance, context
  precision/recall, plus hallucination (with strict number checking),
  toxicity, bias, summarization quality, and conversational metrics — all
  deterministic and offline by default, with `GEvalJudge` for rubric-based
  LLM judging when you want it.
- **Synthetic test data with provenance** — `SyntheticGenerator` bootstraps
  golden sets from your own corpus with difficulty and coverage controls;
  every case records which source produced it and carries the source
  sentences as rubric facts, so grounding metrics work immediately.
- **Evaluation is wired into the runtime**, not a separate harness: every
  run can be scored (`app.add_evaluator("faithfulness")`), every eval case
  runs through the same 17-step pipeline users hit in production, and every
  failing case links to a full trace.
- **Eval results drive optimization** — the evolution loop consumes the same
  reports and promotes prompt/context/routing changes only through safety
  gates; experiments and A/Bs come with statistical significance built in.
- **Coverage beyond RAG**: agent metrics, tool reliability, memory quality
  (staleness, contradiction, privacy), schema validity, cost/latency — one
  report format, one gate mechanism, one CI command.
- **Deterministic metrics first**, model judges second (with repeated-sample
  calibration), keeping evaluation reliable and reproducible.

**Where Ragas is a fit:** research-grade RAG metric variants and a large
academic-lineage metric catalog. Ragas metrics can be wrapped as custom
Vincio metrics via `@register_metric` in a few lines.
