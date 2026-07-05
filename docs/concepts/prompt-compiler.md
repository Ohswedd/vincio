# Prompt compiler

Prompts are compiled, not concatenated. String templates rot: the cache-defeating
order, the rule that quietly duplicates another, the schema asked for in prose while
the provider enforces it natively, the business rule that hides in a user message —
all invisible in an f-string. Vincio treats a prompt as a typed, declarative
`PromptSpec` that compiles — through a `PromptAST`, a lint pass, and a cache-aware
layout — into provider-neutral messages. Same spec + same inputs → same bytes, with
a hash to prove it.

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

## The spec is typed, and variables are checked

A `PromptSpec` is a Pydantic model with named fields for every prompt concern —
`role`, `objective`, `rules` (hard) and `soft_rules`, `definitions`,
`safety_policies`, `examples`, `output_schema` / `output_format` /
`output_instructions`, `citation_policy`, `insufficient_evidence_behavior`,
`reasoning_mode` — rather than a wall of prose. `${var}` placeholders interpolate
declared, type-checked `PromptVariable`s. `spec.substitute(values)` resolves them
and **raises** rather than silently substituting empty strings: a `${var}` that no
`PromptVariable` declares, or a required variable left unset, or a value of the
wrong `type`, is a `PromptError` at compile time, not a mystery blank in
production.

## How it works — the compile pipeline

```text
PromptSpec.substitute(vars)          # typed interpolation, fail-closed
   → build_stable_ast / build_volatile_ast   # partition into prefix / suffix nodes
   → normalize        # trim whitespace, drop empty nodes
   → dedupe           # drop identical (kind, text) rules / defs / policies
   → lint             # PROMPT001–009 (spec-level and AST-level)
   → select_examples  # keep the top max_examples by Example.quality
   → render sections  # per-format: markdown / xml / json / minimal
   → token budget      # PromptBudgetError if over max_prompt_tokens
   → hash              # prompt_spec_hash + rendered_hash
```

The spec first builds a `PromptAST` — an ordered tree of typed nodes
(`SystemRoleNode`, `RuleNode`, `EvidenceBlockNode`, `UserTaskNode`, …), each
carrying a `stable` flag and a `priority`. The AST, not the string, is what gets
normalized, deduped, linted, and ordered. `ast.ordered()` is the whole layout
policy in one line: **stable nodes sorted by priority, then volatile nodes sorted
by priority.** Priorities are fixed by node kind (role `0`, objective `10`, rules
`20`, safety `25`, definitions `30`, output contract `40`, examples `50`, then the
volatile block — memory `60`, evidence `70`, tool results `80`, user task `100`),
so the durable frame always leads and the per-call material always trails.

## Cache-aware layout, and how it stays warm

Stable content (role, rules, definitions, safety policies, schema, examples)
depends only on the spec, so it forms the prefix and the rendered system message
carries `cache_hint=True`; volatile content (memory, evidence, tool results, the
user task) travels in the user message as the suffix. Because the prefix is a pure
function of the spec, the compiler renders it **once** and reuses it: a
`ProgramCache` keyed by `(spec_hash, format, section_headers, max_examples,
include_schema, model, compiler_version)` holds the rendered stable prefix, so a
warm spec re-renders only the volatile suffix. The result is byte-identical to
compiling from scratch — a cache, not a shortcut — and `use_render_program` (on by
default) toggles it. Schema inclusion is resolved *per call*
(`include_schema_in_prompt and not provider_enforces_schema`), never by mutating
the shared `CompilerOptions`, so `compile()` is safe under concurrent use.

The compiled prompt reports its own cacheability:

```python
compiled.cacheability          # stable_prefix_tokens / token_count
compiled.stable_prefix_tokens  # tokens that a provider prompt-cache can reuse
compiled.token_count           # total input tokens
compiled.prompt_id             # "claims@a1b2c3d4" — name + spec-hash prefix
compiled.prompt_spec_hash      # versions the spec (over ordered node digests)
compiled.rendered_hash         # versions the exact rendered bytes
compiled.excluded_examples     # examples dropped over max_examples
```

## Rendering formats

`markdown` (`## Section` headers), `xml` (`<role>…</role>` tags), `json` (a keyed
object), `minimal` (blocks, no headers). The format changes only the surface, never
the node order or content, so `prompt_spec_hash` is stable across formats while
`rendered_hash` moves. Format is an optimization dimension:
`generate_variants(spec)` takes the Cartesian product over format × example-count ×
`reasoning_mode` × rule-ordering (capped at `max_variants`, default 24), and
`vincio.optimize.PromptOptimizer` evaluates the candidates and promotes winners
through eval gates.

## Versioning, diffing, and the registry

Two hashes give you version control for prompts: `prompt_spec_hash` identifies
*what you asked for*, `rendered_hash` identifies *the exact bytes sent*, and
`COMPILER_VERSION` is folded into the cache key so a compiler change invalidates
stale renders. `diff_specs(a, b)` returns a field-level delta; `diff_rendered(a, b)`
returns a unified diff of the rendered text.

`PromptRegistry` is a local, file-backed store (one JSON file per name under
`.vincio/prompts`, no hosted service). `push` is idempotent on content — re-pushing
an unchanged spec returns the existing version rather than minting a new one, keyed
by `spec_hash`; tags (`"production"`, `"candidate"`) move between versions;
`rollback` re-publishes an earlier version as the new head without losing history;
`link_eval` attaches an `EvalReport` summary to the exact version it measured, so a
regression traces to the spec change that caused it.

```python
from vincio.prompts import PromptRegistry

reg = PromptRegistry()
v = reg.push(spec, tags=["candidate"], message="tighten citation policy")
reg.diff("claims", 1, v.version, rendered=True)   # field + rendered diff
reg.tag("claims", v.version, "production")         # steals the tag from the old head
```

## Lint rules

`lint_spec` runs the spec-level checks; `lint_ast` runs the layout checks that need
the ordered tree (PROMPT006, PROMPT009). Findings carry a severity; with
`CompilerOptions(fail_on_lint_errors=True)`, any `error`-severity finding raises
`PromptLintError` at compile time.

| Code | Meaning | Severity |
|---|---|---|
| PROMPT001 | vague or missing role | warning |
| PROMPT002 | duplicate instruction | warning |
| PROMPT003 | conflicting constraints | **error** |
| PROMPT004 | grounded task missing insufficient-evidence behavior | warning |
| PROMPT005 | schema requested in prose while a structured schema is set | warning |
| PROMPT006 | dynamic content ordered before the cacheable prefix | **error** |
| PROMPT007 | grounded task without a citation policy | warning |
| PROMPT008 | excessive examples (>8) | warning |
| PROMPT009 | business rule hidden only in the user message | warning |

Run them over a directory of specs with `vincio prompt lint prompts/`, compile one
to inspect the rendered bytes with `vincio prompt compile prompt.yaml`, and manage
versions with `vincio prompt push` / `versions` / `diff` / `rollback`.

## Best practice

- **Put durable rules in the spec, not the task.** A rule in `rules` lives in the
  cacheable developer prefix and is linted; the same rule in the user task defeats
  the cache and trips PROMPT009. Keep the user message to the actual per-call task.
- **Let the provider enforce the schema.** Set `output_schema` and leave
  `provider_enforces_schema=True` on capable providers — PROMPT005 flags the
  redundant "reply in JSON" prose, and the schema is dropped from the prompt body.
- **Watch `cacheability`.** A low ratio means volatile content is bloating the
  suffix or a rule leaked into the task; a high, stable prefix is what a provider
  prompt-cache actually reuses across calls.

## Gotchas

- **A missing declared variable is a hard error, by design** — `substitute` raises
  `PromptError` rather than emitting a blank, so a typo can't silently ship a
  half-rendered prompt.
- **`rendered_hash` moves with the format even when `prompt_spec_hash` doesn't** —
  switching `markdown`→`xml` is a rendering change, not a spec change; diff on the
  hash that matches your intent.
- **The render program cache is per-`PromptCompiler` instance.** Reuse one compiler
  across calls that share a spec to get the warm-prefix win; a fresh compiler per
  call renders the prefix every time (still correct, just not cached).

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
