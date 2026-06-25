# Vincio vs Pydantic AI

Pydantic AI brings Pydantic's validation ergonomics to agents: typed
dependencies, typed results, and retries when the model output fails
validation.

**Where Vincio differs**

- **Validation is a pipeline, not a type check.** A Vincio output contract
  runs parse → schema → semantic validators → citations → policy (including
  programmable rails), and every step's outcome is recorded in
  `result.validation`, on the trace, and in the hash-chained audit log.
- **Repair never invents facts.** Pydantic-style retries re-ask the model
  for a whole new answer; Vincio first applies deterministic structural
  repair (lenient parse, safe coercion, fill-optional), and its bounded
  self-correction loop is structure-only by contract, semantic and
  citation validators re-run every cycle, so a "fixed" answer with changed
  facts still fails.
- **Constrained decoding with a strict transform.** Schemas are
  strict-sanitized (`to_strict_json_schema`) for provider-native
  constrained decoders, negotiated per provider capability, with the robust
  parser as the universal fallback, and the decoding mode is on the trace.
- **Streaming validation.** Partial output is prefix-checked against the
  schema as it streams; a definite mismatch surfaces mid-generation so you
  can abort early instead of validating only at the end.
- **One system, not a library per concern.** The same schema objects feed
  retrieval citations, eval metrics (`schema_validity` gates releases and
  the optimizer), and multi-schema routing by task or content.

**Where Pydantic AI is a fit:** minimal typed agents in a codebase already
organized around Pydantic dependency injection. Vincio uses Pydantic v2 for
every contract, so models written for Pydantic AI drop into
`ContextApp(output_schema=...)` unchanged.
