# Evaluation

Evaluation is a native runtime capability — every subsystem is measurable.

## Datasets

JSONL cases with expected outputs, rubrics, tags, and difficulty:

```json
{"id": "case_001", "input": "Can I get a refund?", "expected": "...",
 "rubric": {"facts": ["..."], "relevant_ids": ["D1:C2"]},
 "tags": ["refund", "edge_case"], "difficulty": "medium"}
```

```python
from vincio.evals import Dataset
dataset = Dataset.load("golden/support_triage.jsonl")
dataset.filter(tags=["edge_case"]).sample(20)
```

## Metrics

- **Task** — `exact_match`, `semantic_similarity`, `classification_accuracy`, `extraction_f1`
- **Grounding** — `groundedness`, `unsupported_claim_rate`, `citation_accuracy`,
  `citation_recall`, `context_precision`, `context_recall`
- **Operational** — `cost`, `latency`, `input_tokens`, `output_tokens`, `retries`
- **Retrieval** — `recall_at_k`, `precision_at_k`, `mrr`, `ndcg`
- **Agent/memory** — via `AgentState.metrics()` and `MemoryEngine.stats()`

Register custom metrics with `@register_metric("name")`.

## Judges

`DeterministicJudge`, `ModelJudge` (rubric + structured score, calibrated by
repeated sampling), `EmbeddingJudge`, `HybridJudge` (weighted blend).

## Runner and gates

```python
report = app.evaluate("golden/contracts.jsonl",
    metrics=["groundedness", "citation_accuracy", "schema_validity", "cost"],
    concurrency=8,
    gates={"groundedness": ">= 0.95", "schema_validity": "== 1.0", "p95_latency": "<= 8000"})
report.print_summary()
report.diff(baseline_report)   # per-metric deltas + regressed cases
```

CI usage:

```bash
vincio eval run tests/golden/basic.jsonl --app app.py \
  --gate "groundedness=>= 0.95" --compare baseline.json --output report.json
```

The command exits non-zero when gates fail — wire it into CI directly.
