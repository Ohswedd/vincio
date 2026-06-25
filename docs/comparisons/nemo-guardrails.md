# Vincio vs NeMo Guardrails

NeMo Guardrails adds programmable rails to conversational systems via Colang
flows: input rails, output rails, dialog rails, and retrieval rails.

**Where Vincio differs**

- **Rails are policies, not a dialog language.** A Vincio rail is plain
  data (kind, direction, action, parameters) evaluated by the deterministic
  policy engine, no Colang runtime, no model-judged checks in the
  enforcement path, and the same engine that enforces every other run
  policy (citations, PII redaction, untrusted-instruction blocking).
- **Detection is shared with the security engine.** Safety rails reuse the
  PII detector, secret scanner, and injection detector that already screen
  inputs, retrieved evidence, and tool output, one set of detectors,
  hardened by the red-team suite, covering every path.
- **Every rail decision is evidence.** Violations are `PolicyViolation`
  records named `rail:<name>` on the trace and in the hash-chained audit
  log: "why was this blocked?" is always answerable, offline.
- **Output rails compose with the contract.** An output rail failure is a
  validation-pipeline step alongside schema, semantic, and citation checks,
  one report, one place to look, and redact-action rails can mask instead of
  block.

**Where NeMo Guardrails is a fit:** rich multi-turn dialog flows where the
conversation itself needs scripted structure. Vincio rails focus on run-level
input/output enforcement; a custom rail predicate can call into any external
checker, including a NeMo-style classifier.
