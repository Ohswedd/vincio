# Vincio vs DeepEval

DeepEval brings unit-test-style assertions, G-Eval judging, and red-teaming
to LLM evaluation.

**Where Vincio differs**

- **The same assertions, offline by default**: `assert_eval`,
  `assert_grounded`, `assert_metric`, `assert_safe` run on deterministic
  metrics, so tests don't burn tokens or flake on judge variance; the pytest
  plugin (snapshot fixture, `--vincio-update-snapshots`) is auto-registered
  on install.
- **Metrics are runtime objects, not test-only**: the metric a test asserts
  (`hallucination`, `toxicity`, ...) is the same object the runtime attaches
  to every run (`app.add_evaluator`) and the optimizer uses as a fitness
  term. One definition, three uses.
- **Red-teaming judged deterministically**: attack probes carry a canary
  token and responses are judged by the security engine's detectors (secret
  scanner, PII detector, bias/toxicity metrics), so the adversarial suite
  gates CI without a judge model; it also reports input-side detector
  coverage, because Vincio ships the injection defense too.
- **G-Eval with calibration**: `GEvalJudge` auto-derives evaluation steps
  from criteria and fits a linear correction against human labels
  (`judge.calibrate(pairs)` → Pearson r), instead of trusting raw judge
  scores.
- **Snapshot tests for structure**: packets and traces snapshot with
  volatile fields normalized away, a layer DeepEval doesn't have because it
  doesn't own the runtime.
- **Trajectory and tool-use metrics**: alongside the assertion and red-team
  parity, `tool_call_accuracy` / `tool_call_f1`, `goal_accuracy`,
  `plan_adherence`, `plan_quality`, `step_efficiency`, and `topic_adherence`
  score *how* an agent reached its answer. Build the trajectory from a
  finished run with no re-instrumentation
  (`RunOutput.from_agent_state(state)`, `from_crew_result`, `from_trace`);
  `EvalReport.metric_families()` shows output-only and trajectory eval side
  by side.
- **Conversational metrics and a simulator**: `conversation_outcome` and
  `intent_resolution` score multi-turn threads, and
  `Simulator(seed=7).simulate(agent, Persona(...))` drives a persona against
  your app and emits a golden case (`convo.to_eval_case(...)`);
  seed-deterministic so simulated sessions are usable as CI goldens,
  LLM-backed when given a provider.
- **Online eval, in your process**: `app.add_online_evaluator("toxicity",
  sample_rate=0.1)` scores a sampled fraction of live runs off the hot path
  into a local time series, and `DriftMonitor` flags when scores or
  embeddings move off baseline, continuous quality without a hosted
  dashboard.
- **The metric is also a guardrail**: the same metric a test asserts becomes
  a runtime rail with `app.add_metric_rail("toxicity", threshold=0.0)` (or
  `metric_guardrail(metric, threshold=...)`) and an optimizer fitness term
  via `AGENTIC_OBJECTIVES`. One definition gates tests, blocks live requests,
  and drives optimization — a connection a test-only library can't offer.

**Where DeepEval is a fit:** a broad catalog of LLM-judged metric variants
and hosted dashboards via Confident AI. DeepEval metrics can be wrapped as
custom Vincio metrics via `@register_metric`.
