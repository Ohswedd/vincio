# Vincio vs Guardrails AI

Guardrails AI validates and corrects LLM output against a spec (RAIL/Pydantic)
with a hub of reusable validators and on-fail actions (reask, fix, filter).

**Where Vincio differs**

- **Guards are wired into a runtime, not wrapped around calls.** Output
  contracts, semantic validators, citation checks, and rails run inside the
  same 17-step pipeline that compiled the context — a validation failure is
  a trace span, an audit entry, and an eval signal, not just an exception.
- **Deterministic before model-based.** Vincio's repair ladder is ordered by
  trust: lenient parsing and safe structural coercion first (free,
  deterministic, auditable), bounded structure-only self-correction last —
  with cycle and cost ceilings, and with facts contractually out of bounds.
- **Input-side defense is the security engine.** PII, secrets, and
  prompt-injection rails reuse the same detectors that screen retrieved
  evidence and tool output everywhere else, so guard coverage doesn't drift
  between entry points.
- **Validation results close the loop.** `schema_validity` is a fitness term
  and a promotion gate in the optimizer: a prompt variant that improves
  accuracy but regresses validity is never auto-promoted.

**Where Guardrails AI is a fit:** the validator hub — a large catalog of
community validators. Any of them can be registered as a Vincio semantic
validator (`app.add_validator(name, fn)`); custom rails accept arbitrary
predicates the same way.
