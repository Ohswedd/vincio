# Vincio vs DSPy

DSPy pioneered "programming, not prompting": typed signatures, modules,
and automatic prompt/weight optimization.

**Where Vincio differs**

- **Typed signatures, validated end to end (0.7).** Vincio `Signature`s
  (class-based or the `"question, context -> answer"` string form) compile
  to a `PromptSpec` over the prompt AST; `Predict` executes them with
  provider-native constrained decoding and the full validation pipeline —
  schema, repair, citations, policy — not just a parse.
- **Signatures feed the optimizer.** `Signature.to_prompt_spec()` is a
  drop-in target for `PromptOptimizer`: format selection, example search,
  reasoning modes, and instruction rewrites all apply to signatures exactly
  as to hand-written prompts.
- **Optimization spans the full context lifecycle**, not just the LM
  program: prompt format/examples/reasoning-mode search, retrieval and
  context-budget tuning, model routing, and cache layout — all driven by
  the same fitness function and gated promotion rules.
- **Production runtime included**: storage, traces, audit logs, tenant
  isolation, permissioned tools, budgets, and a server — DSPy-style
  optimization without assembling the operational stack around it.
- **Safety-gated promotion**: candidates that improve quality but regress
  schema validity or safety are never auto-promoted, and
  optimization refuses to run on datasets too small to be trustworthy.

**Where DSPy is a fit:** research on LM program synthesis and
multi-stage pipeline optimization. A DSPy-optimized program can serve as a
Vincio provider or tool, and DSPy-style optimizers can plug into
`evolution_loop` as candidate generators.
