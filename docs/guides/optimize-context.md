# Guide: optimize prompts, context, and routing

Optimization turns eval results into better configurations, always through
gates, never silently.

## Fitness

```text
Fitness = α·Accuracy + β·Groundedness + γ·SchemaValidity + δ·Safety
        − ε·Cost − ζ·Latency − η·RetryRate
```

## Prompt optimization

```bash
vincio optimize run --app app.py --dataset golden/contracts.jsonl \
  --target groundedness --budget 12 --output winning_prompt.yaml
```

Programmatically:

```python
from vincio.optimize import PromptOptimizer, FitnessWeights

optimizer = PromptOptimizer(evaluate_variant_fn,
    weights=FitnessWeights(groundedness=2.0),
    gates={"schema_validity": "== 1.0"})
result = await optimizer.optimize(app.prompt_spec, dataset, max_variants=12)
```

The evolution loop runs candidates on a screening subset, promotes the top
N to the full dataset, and **refuses to promote** when safety or schema
validity regress, cost exceeds budget, gates fail, or the dataset is too
small to be trustworthy.

## Context optimization

```python
from vincio.optimize import ContextOptimizer, ContextSearchSpace
space = ContextSearchSpace(top_k=[4, 8, 12], chunk_size_tokens=[200, 400, 800],
                           use_evidence_ledger=[False, True])
result = await ContextOptimizer(evaluate_config_fn).optimize(dataset, space=space, budget=12)
```

`strategy="hill_climb"` or `"anneal"` makes proposals condition on
subset scores already observed instead of sampling blindly, deterministic
under a seed, bounded by the budget, same gated promotion.

## The closed loop

`ImprovementLoop` runs the whole cycle, capture traces, curate a dataset,
evaluate, optimize, and promote the winner into the prompt registry, in
one call; `pareto_loop` keeps a cost/quality frontier instead of one score;
`RetrievalFeedback` tunes retrieval from eval relevance labels; and
`BudgetLearner` learns per-task budget allocation from eval outcomes. See
the [close the loop guide](close-the-loop.md).

## Model routing

```python
from vincio.optimize import RoutingPolicy, estimate_difficulty

policy = RoutingPolicy(cheap_model="gpt-5.2-nano", default_model="gpt-5.2-mini",
                       strong_model="gpt-5.2")
model = policy.route(difficulty=estimate_difficulty(query), risk="low")
```

`RoutingOptimizer` learns the difficulty thresholds offline from per-tier
eval reports; `EpsilonGreedyBandit`/`UCB1Bandit` support live routing,
behind offline gates only.

## Cache layout tuning

```python
from vincio.optimize import analyze_prompt_cacheability
report = analyze_prompt_cacheability(compiled_prompt)
for advice in report.advice:
    print(advice.code, advice.message)   # CACHE001 timestamp in prefix, ...
```

## How gated promotion works

Every optimizer here shares one shape: **propose → screen → gate → promote**.
Candidates are first scored on a cheap screening subset; only the top N advance
to the full dataset, so you don't pay to evaluate every dud at full cost. Then
the split between the objective and the constraints does the real work:

- **Fitness is the objective** — the weighted score the search *maximizes*.
- **Gates are hard constraints** — a variant that wins fitness but regresses
  safety or schema validity, exceeds the cost budget, fails a gate expression, or
  was measured on too small a dataset is **refused promotion**, never shipped.

That is why optimization "always through gates, never silently": the winner has
to beat the baseline on fitness *and* clear every constraint, or the incumbent
stays.

## Best practice

- **Optimize a signature's `PromptSpec`.** A typed [signature](structured-output.md)
  compiles to a `PromptSpec` that is a first-class search target — formats,
  examples, reasoning modes, rewrites — so you get optimization for free on
  anything you expressed as a signature.
- **Keep a held-out golden set** and let the gates reference it; a promotion
  judged on the same data it was tuned on is the classic way to ship a
  regression that "passed".
- **Prefer `hill_climb` / `anneal` for context search.** They condition
  proposals on subset scores already observed instead of sampling blindly, so a
  fixed budget goes further — and stay deterministic under a seed.

## Gotchas

- **Fitness weights are a tradeoff, not free wins.** Cranking `groundedness` can
  cost accuracy or latency; when one score isn't the whole story, run `pareto_loop`
  and keep the cost/quality frontier instead of collapsing to a single number.
- **Live bandits are gated too.** `EpsilonGreedyBandit` / `UCB1Bandit` support
  online routing, but only behind offline gates — you never ship an unbounded
  explorer that can drift into a bad arm in production.
- **`estimate_difficulty` is deterministic** (the same estimator the
  [reasoning controller](reasoning.md) uses), so routing decisions reproduce
  under a seed — good for tests, and a reminder that it is a heuristic, not a
  model call.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: Prompt compiler](../concepts/prompt-compiler.md)
- [Concept: Context packets & long-horizon governance](../concepts/context-packets.md)
- [Guide: Performance & streaming](performance.md)
- [Guide: close the loop](close-the-loop.md)
- [Example: 01_quickstart.py](../../examples/01_quickstart.py)
- [Example: 11_advanced_context.py](../../examples/11_advanced_context.py)
- [Example: 08_optimization_self_improvement.py](../../examples/08_optimization_self_improvement.py)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
