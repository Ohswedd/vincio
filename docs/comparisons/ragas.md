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
- **Eval results close the loop** — `ImprovementLoop` curates
  production traces into datasets, optimizes against them, and promotes the
  winner into the prompt registry in one audited, reproducible cycle;
  eval-scored relevance tunes retrieval fusion and reranker weights
  (`RetrievalFeedback`) and picks chunking configs (`recommend_chunking`).
  Ragas scores a system; Vincio's scores change the system — through gates.
- **Coverage beyond RAG**: agent metrics, tool reliability, memory quality
  (staleness, contradiction, privacy), schema validity, cost/latency — one
  report format, one gate mechanism, one CI command.
- **Trajectory and agentic metrics on top of RAG parity** —
  `tool_call_accuracy` / `tool_call_f1`, `goal_accuracy`, `plan_adherence`,
  `plan_quality`, `step_efficiency`, and `topic_adherence` score *how* an
  agent reached its answer. Build the trajectory from a finished run with no
  re-instrumentation (`RunOutput.from_agent_state(state)`,
  `from_crew_result`, `from_trace`), and `EvalReport.metric_families()` puts
  output-only and trajectory eval side by side.
- **Multi-turn conversation simulator** — `Simulator(seed=7).simulate(agent,
  Persona(name="sam", goal="reset password"))` drives a persona against your
  app and turns the thread into a golden case (`convo.to_eval_case(...)`);
  seed-deterministic by default, so simulated sessions work as CI goldens,
  and LLM-backed when given a provider. `conversation_outcome` and
  `intent_resolution` score the whole thread.
- **Online eval and drift** — `app.add_online_evaluator("faithfulness",
  sample_rate=0.1)` scores a sampled fraction of live runs off the hot path
  into a local time series, and `DriftMonitor` flags when live scores or
  embeddings move off baseline (`vincio eval drift ...`) — continuous quality
  without shipping traces to a platform.
- **Deterministic metrics first**, model judges second (with repeated-sample
  calibration), keeping evaluation reliable and reproducible.

**Where Ragas is a fit:** research-grade RAG metric variants and a large
academic-lineage metric catalog. Ragas metrics can be wrapped as custom
Vincio metrics via `@register_metric` in a few lines.
