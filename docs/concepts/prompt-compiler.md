# Prompt compiler

Prompts are compiled, not concatenated. A `PromptSpec` is a typed,
declarative prompt definition; the compiler turns it into a `PromptAST` and
renders provider-neutral messages.

```python
from vincio.prompts import PromptSpec, PromptVariable, PromptCompiler, CompilerOptions

spec = PromptSpec(
    name="claims",
    role="insurance_claim_decision_engine",
    objective="Determine whether a claim for plan ${plan} is reimbursable",
    rules=["Use only provided documents"],
    citation_policy="Cite evidence IDs in square brackets.",
    insufficient_evidence_behavior="If evidence is missing, say so explicitly.",
    output_schema=ClaimDecision.model_json_schema(),
    variables=[PromptVariable(name="plan", type="str")],
)
compiled = PromptCompiler(CompilerOptions(format="markdown")).compile(
    spec, user_task="Is claim INV-9 reimbursable?", variables={"plan": "Gold"}
)
```

## Compiler passes

normalize → dedupe rules → conflict check → **stable-prefix layout** →
example selection (by quality, under budget) → schema render → context
block render → token budget validation → hashing.

## Cache-aware layout

Stable content (role, rules, definitions, schema, examples) forms the
prefix and gets `cache_hint=True`; volatile content (evidence, memory, the
user task) goes to the suffix. The compiled prompt reports:

```python
compiled.cacheability          # StablePrefixTokens / TotalInputTokens
compiled.stable_prefix_tokens
compiled.prompt_spec_hash      # version every spec
compiled.rendered_hash         # version every rendering
```

## Rendering formats

`markdown` (sections), `xml` (tags), `json`, `minimal`. Formats are an
optimization dimension; `vincio.optimize.PromptOptimizer` searches them.

## Lint rules

| Code | Meaning |
|---|---|
| PROMPT001 | vague role |
| PROMPT002 | duplicate instruction |
| PROMPT003 | conflicting constraints |
| PROMPT004 | missing insufficient-evidence behavior |
| PROMPT005 | schema requested in prose while structured output available |
| PROMPT006 | dynamic content placed before cacheable prefix |
| PROMPT007 | no citation policy for grounded task |
| PROMPT008 | excessive examples |
| PROMPT009 | hidden business rule only in user message |

Run them with `vincio prompt lint prompts/`.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: optimize prompts, context, and routing](../guides/optimize-context.md)
- [Example: 01_quickstart.py](../../examples/01_quickstart.py)
- [Concept: The ergonomic front door](ergonomic-surface.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
