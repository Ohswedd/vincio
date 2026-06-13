# Guide: run evals

## 1. Build a golden dataset

```jsonl
{"id": "case_001", "input": "Can I get a refund?", "expected": "Refunds within 30 days.", "rubric": {"facts": ["refunds within 30 days"]}, "tags": ["refund"], "difficulty": "easy"}
```

Commit it under `tests/golden/` or `golden/` so it lives alongside your code.

## 2. Run

```python
report = app.evaluate("golden/refunds.jsonl",
    metrics=["groundedness", "citation_accuracy", "schema_validity", "cost", "latency"])
report.print_summary()
```

Or with the CLI:

```bash
vincio eval run golden/refunds.jsonl --app app.py \
  --metric groundedness --metric schema_validity \
  --gate "groundedness=>= 0.95" --gate "schema_validity=== 1.0" \
  --output reports/today.json --compare reports/baseline.json
```

## 3. Gate your CI

Gates use `metric` or `p95_metric`/`min_metric`/`max_metric` aggregates;
the CLI exits non-zero on failure:

```yaml
# .github/workflows/evals.yml (excerpt)
- run: |
    vincio eval run tests/golden/basic.jsonl --app app.py \
      --gate "groundedness=>= 0.95" --gate "schema_validity=== 1.0" \
      --gate "p95_latency=<= 8000" --compare baseline.json
```

## 4. Add model judges

```python
from vincio.evals import EvalRunner, ModelJudge
judge = ModelJudge(provider, model="gpt-5.2-mini",
                   rubric="Score correctness and completeness of the refund decision.",
                   samples=3)   # repeated scoring reduces judge noise
report = EvalRunner(app, metrics=["schema_validity"], judges=[judge]).run(dataset)
```

## 5. Track trends

```python
from vincio.storage.duckdb import DuckDBAnalytics   # pip install "vincio[duckdb]"
analytics = DuckDBAnalytics()
analytics.ingest_report(report)
analytics.metric_trend("groundedness")
```

## 6. Go further

- **Bootstrap a dataset from your corpus** — `SyntheticGenerator(seed=7).generate(documents, n=50)`
  (difficulty mix, source coverage, provenance), or curate production traces:
  `vincio eval dataset golden.jsonl --min-feedback 0.5`.
- **Judge with G-Eval** — `GEvalJudge(provider, model=..., criteria="...", samples=3)`;
  calibrate against human labels with `judge.calibrate(pairs)`.
- **Compare variants with significance** — `ExperimentTracker` + `ab_test(report_a, report_b, metric)`
  (paired/Welch t-test). See the [evaluation concepts](../concepts/evals.md).
- **Red-team the app** — `RedTeamSuite().run(app)`; gate `attack_success_rate` at 0.0.
- **Assert in pytest** — `assert_grounded(result)`, `assert_eval(result, metrics={...})`;
  see the [testing guide](test-llm-apps.md).
