# Vincio vs DeepEval

DeepEval brings unit-test-style assertions, G-Eval judging, and red-teaming
to LLM evaluation.

**Where Vincio differs**

- **The same assertions, offline by default** — `assert_eval`,
  `assert_grounded`, `assert_metric`, `assert_safe` run on deterministic
  metrics, so tests don't burn tokens or flake on judge variance; the pytest
  plugin (snapshot fixture, `--vincio-update-snapshots`) is auto-registered
  on install.
- **Metrics are runtime objects, not test-only** — the metric a test asserts
  (`hallucination`, `toxicity`, ...) is the same object the runtime attaches
  to every run (`app.add_evaluator`) and the optimizer uses as a fitness
  term. One definition, three uses.
- **Red-teaming judged deterministically** — attack probes carry a canary
  token and responses are judged by the security engine's detectors (secret
  scanner, PII detector, bias/toxicity metrics), so the adversarial suite
  gates CI without a judge model; it also reports input-side detector
  coverage, because Vincio ships the injection defense too.
- **G-Eval with calibration** — `GEvalJudge` auto-derives evaluation steps
  from criteria and fits a linear correction against human labels
  (`judge.calibrate(pairs)` → Pearson r), instead of trusting raw judge
  scores.
- **Snapshot tests for structure** — packets and traces snapshot with
  volatile fields normalized away, a layer DeepEval doesn't have because it
  doesn't own the runtime.

**Where DeepEval is a fit:** a broad catalog of LLM-judged metric variants
and hosted dashboards via Confident AI. DeepEval metrics can be wrapped as
custom Vincio metrics via `@register_metric`.
