# The Big Brain

Every Vincio capability that improves a model's answer — adaptive reasoning,
internal planning, governed browsing, retrieval, deterministic verification and
bounded correction — converges in one orchestration core. We call it the Big
Brain: a provider-independent cognitive layer that any model, from a hosted
frontier system to a local 3B open-source model, runs through unchanged. The
model supplies language competence; the Big Brain supplies the architecture
around it, and every decision it makes is recorded in a typed receipt rather
than hidden in prose.

It is not an agent framework bolted on top of the run pipeline. It *is* the run
pipeline: `app.use_reasoning_engine()` makes ordinary `run`/`arun` calls flow
through it, every internal call is traced, DLP-screened, cost- and
energy-accounted, and every ceiling — passes, searches, pages, tokens, cost —
comes from the same budget the caller already set.

## One flow, seven stages

1. **Assess.** A zero-token deterministic router classifies the request: task
   taxonomy, mathematical/logical/causal/decision/temporal/spatial structure,
   constraints, supplied modalities, exact enabled-tool matches, and freshness.
   Non-English or uncertain-language requests receive one compact model-native
   classification instead — coverage follows the model, not a locale list.
2. **Plan.** Simple work stays a single pass. A deep, genuinely multi-step
   request triggers the internal plan mode: one bounded planning call returns a
   validated, typed decomposition (`PlannedStep` — goal, kind, dependency
   order, and the deterministic check each step's output should survive). The
   plan structures every candidate pass; it can never open the web, name an
   unregistered tool, or exceed its step and token ceilings, and low-confidence
   plans fall back to the heuristic decomposition.
3. **Gather.** Freshness, explicit fact-check requests and high-stakes topics
   deterministically trigger the governed browser when enabled. Pages become
   injection-screened, content-hashed, host-diversified untrusted evidence.
   Anchored sources and LAGER retrieval feed the same compiled context.
4. **Generate.** Bounded answer-only passes run through the normal pipeline —
   native reasoning effort where the provider has it, provider-neutral
   candidate passes where it doesn't. Prompts require private analysis; scratch
   work is neither requested nor recorded.
5. **Verify.** Offline kernels recompute arithmetic, units, dates, constraints
   and citation support; task-bound facts recomputed from the request reject an
   internally valid answer to the wrong question; and the fabrication check
   refutes any answer that attributes claims to a source present in neither the
   attached evidence nor the request.
6. **Correct.** A refuted, source-fabricating, leaking or materially disagreeing
   answer earns one bounded correction pass that sees only the answers,
   verdicts and governed evidence — and is accepted only if it verifies at
   least as well.
7. **Account.** The receipt on `result.metadata["universal_reasoning"]` records
   depth, strategy, plan steps, search decisions, web verification, per-pass
   verdicts, fabricated sources, model calls, tokens and cost. Multi-pass usage
   aggregates honestly onto the returned `RunResult`.

## Honesty is structural, not stylistic

The Big Brain's defining property is that it prefers no answer to a wrong or
fabricated one:

- a refuted answer cannot win candidate selection;
- an unsupported current-fact claim without live evidence is withheld unless it
  honestly carries uncertainty (`[UNVERIFIED]`);
- an invented source — a URL or "according to …" attribution matching nothing
  in the evidence or the request — refutes the answer outright, even when
  hedged;
- when every bounded attempt stays refuted, the engine returns a failed run
  with no output, or a minimal deterministic fallback containing only facts the
  local kernels already proved;
- a text fallback never replaces a structured Pydantic contract.

These guarantees are mechanically gated: `UniversalReasoningBench` pins plan
activation and skip, fabricated-source refusal, honest-citation precision,
kernel-refutation repair, leak blocking, pass ceilings and cost accounting in
CI, and `benchmarks/reasoning_uplift_live.py` measures the live uplift on real
OpenRouter models — including a plan-shaped decision case and a cite-a-source
honesty case.

## Using it

```python
import vincio

app = vincio.ContextApp(provider="openai", model="gpt-5.2-mini")
app.use_web_search(preset="research")
app.use_reasoning_engine()          # every run now flows through the Big Brain

result = app.run("Compare the two rollout plans under these constraints ...")
print(result.metadata["universal_reasoning"]["plan_steps"])

outcome = app.reason("...")         # full typed receipt for one call
print(outcome.plan.steps, outcome.answer_verification, outcome.confidence)
```

`UniversalReasoningPolicy` holds the levers: `plan_mode="auto"|"off"|"always"`,
`plan_max_steps`, `max_passes`, `web`, `require_citations_for_live_claims`,
`semantic_routing`, and the per-call token/timeout ceilings. Everything is
additive and experimental (`@experimental(since="7.10")`, plan mode since
`7.11`); an app that never calls `use_reasoning_engine()` is byte-identical to
before.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: Universal reasoning, native thinking & the Responses API](../guides/reasoning.md)
- [Example: 22_universal_reasoning.py](../../examples/22_universal_reasoning.py)
- [Concept: Prompt compiler](prompt-compiler.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
