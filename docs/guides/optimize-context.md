# Guide: optimize prompts, context, and routing

Optimization turns eval results into better configurations — always through
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

`strategy="hill_climb"` or `"anneal"` (0.8) makes proposals condition on
subset scores already observed instead of sampling blindly — deterministic
under a seed, bounded by the budget, same gated promotion.

## The closed loop (0.8)

`ImprovementLoop` runs the whole cycle — capture traces, curate a dataset,
evaluate, optimize, and promote the winner into the prompt registry — in
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
eval reports; `EpsilonGreedyBandit`/`UCB1Bandit` support live routing —
behind offline gates only.

## Cache layout tuning

```python
from vincio.optimize import analyze_prompt_cacheability
report = analyze_prompt_cacheability(compiled_prompt)
for advice in report.advice:
    print(advice.code, advice.message)   # CACHE001 timestamp in prefix, ...
```
