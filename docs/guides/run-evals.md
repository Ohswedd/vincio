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

## 6. Score agent trajectories

Output-only metrics can't see that a run answered right while taking the wrong
path. Project a completed agent run onto a `RunOutput`, no re-instrumentation,
and score the trajectory it actually took:

```python
from vincio.evals import RunOutput
from vincio.evals.metrics import METRICS

# Score a single finished run directly with the metric functions:
state = app.agent(...).run(...)            # or RunOutput.from_crew_result / .from_trace
run = RunOutput.from_agent_state(state)
METRICS["tool_call_accuracy"](case, run).value

# ...or evaluate a whole golden set through the app, each run carries its
# trajectory automatically, so the trajectory metrics score every case:
report = app.evaluate("golden/agents.jsonl", metrics=[
    "goal_accuracy", "tool_call_accuracy", "plan_adherence", "step_efficiency",
])
```

Put the references the trajectory metrics need on the case rubric:

```jsonl
{"id": "a1", "input": "...", "expected": "...", "rubric": {"expected_tools": ["search", "summarize"], "plan": ["tool", "tool", "finalize"], "optimal_steps": 3, "topic": "..."}}
```

`expected_tools` entries may be names or `{"tool": name, "arguments": {...}}`.
Cases without a trajectory score a neutral `1.0`, so trajectory metrics sit
alongside output-only ones. To see both at once, split the report:

```python
report.metric_families()   # {"output": {...}, "trajectory": {...}}
```

The `trajectory` family is the seven names in `TRAJECTORY_METRICS`
(`tool_call_accuracy`, `tool_call_f1`, `goal_accuracy`, `plan_adherence`,
`plan_quality`, `step_efficiency`, `topic_adherence`).

## 7. Evaluate live traffic

Run the same metrics continuously, off the hot path. An online evaluator scores
a deterministic 1-in-N sample of live runs after each response is finalized and
writes a score time series to the store:

```python
app.add_online_evaluator("answer_relevance", sample_rate=0.1)
app.add_online_evaluator("goal_accuracy", sample_rate=0.2)

app.online_evaluators[0].series()   # the rows
await app.aflush_online()           # drain in-flight scoring (tests/shutdown)
```

Scoring is in-process, no traces leave the box, and emits an `eval.online`
event you can subscribe to.

## 8. Detect drift and calibrate judges

Watch a metric against a baseline. On drift the CLI exits non-zero, so it gates
like any other eval step:

```bash
vincio eval drift baseline.json current.json --metric goal_accuracy --threshold 0.1
```

```python
from vincio.evals import DriftMonitor
monitor = DriftMonitor(bus=app.events, store=app.store, score_threshold=0.1)
monitor.set_score_baseline("goal_accuracy", baseline_values)
report = monitor.check_scores("goal_accuracy", recent_values)
report.drifted, report.delta, report.z_score
```

Check that an LLM judge agrees with people before it gates anything. Score
agreement with Cohen's κ from a labels file:

```bash
vincio eval annotate labels.jsonl --threshold 0.6 --bins 2
```

```python
from vincio.evals import AnnotationQueue
q = AnnotationQueue(name="judge_cal")
item = q.add(run_id="r1", judge_score=0.9)
q.label(item.id, human_score=1.0)
q.agreement()                  # {"cohens_kappa": ..., "exact_agreement": ..., "n": ...}
q.judge_trusted(threshold=0.6)
```

## 9. Go further

- **Bootstrap a dataset from your corpus**: `SyntheticGenerator(seed=7).generate(documents, n=50)`
  (difficulty mix, source coverage, provenance), or curate production traces:
  `vincio eval dataset golden.jsonl --min-feedback 0.5`.
- **Judge with G-Eval**: `GEvalJudge(provider, model=..., criteria="...", samples=3)`;
  calibrate against human labels with `judge.calibrate(pairs)`.
- **Compare variants with significance**: `ExperimentTracker` + `ab_test(report_a, report_b, metric)`
  (paired/Welch t-test). See the [evaluation concepts](../concepts/evals.md).
- **Red-team the app**: `RedTeamSuite().run(app)`; gate `attack_success_rate` at 0.0.
- **Assert in pytest**: `assert_grounded(result)`, `assert_eval(result, metrics={...})`;
  see the [testing guide](test-llm-apps.md).

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: Evaluation & continuous quality](../concepts/evals.md)
- [Guide: test LLM apps with pytest](test-llm-apps.md)
- [Guide: agentic evaluation & continuous quality](agentic-eval.md)
- [Example: 07_evaluation_observability.py](../../examples/07_evaluation_observability.py)
- [Concept: Observability](../concepts/observability.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#optimization)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
