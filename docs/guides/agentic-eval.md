# Guide: agentic evaluation & continuous quality

Vincio already traces your crews, graphs, and tool loops. 1.2 makes it **score**
them — over the trajectory, across a multi-turn conversation, and on live traffic —
reusing the *same* metric objects offline, as runtime guardrails, and as optimizer
fitness. Everything runs in your process, offline and deterministic by default.

The new entry points (`app.add_online_evaluator`, `app.experiment`,
`app.add_metric_rail`) are `@experimental(since="1.2")`; the rest is stable
additive API. Import from `vincio.evals`.

## 1. Trajectory & tool-use metrics

Project a finished agent run onto a `RunOutput` that carries its trajectory —
no re-instrumentation — and score the seven trajectory metrics. References
(`expected_tools`, `plan`, `optimal_steps`, `topic`) live on the case rubric.
`metric_families()` shows output-only vs trajectory side by side: a run can
answer right while taking the wrong path, and output-only eval can't see that.

```python
from vincio.evals import EvalCase, RunOutput
from vincio.evals.metrics import METRICS

state = app.agent(tools=[search, summarize], planner="react").run(
    "What is the refund window for the Pro plan?"
)
run = RunOutput.from_agent_state(state)          # also: from_crew_result, from_trace
case = EvalCase(
    id="refund",
    input="What is the refund window for the Pro plan?",
    expected="Refunds on the Pro plan are available within 30 days.",
    rubric={
        "expected_tools": ["search", "summarize"],   # names, or {"tool": n, "arguments": {...}}
        "plan": ["tool", "tool", "finalize"],
        "optimal_steps": 3,
    },
)
for name in ("goal_accuracy", "tool_call_accuracy", "tool_call_f1",
             "plan_adherence", "plan_quality", "step_efficiency"):
    print(name, METRICS[name](case, run).value)
```

Across a golden dataset, run the eval through the app — each case's trajectory
comes for free from the run's tool calls — and read both families side by side:

```python
report = app.evaluate(dataset, metrics=[
    "lexical_overlap", "answer_relevance",       # output-only
    "goal_accuracy", "tool_call_accuracy", "step_efficiency",
])
report.metric_families()    # {"output": {...}, "trajectory": {...}}
```

The seven trajectory metrics (`TRAJECTORY_METRICS`) read a `Trajectory` carried
on the `RunOutput`. When a run has no trajectory they return a neutral `1.0`, so
they sit alongside output-only metrics without penalizing non-agentic cases.

## 2. Multi-turn & the Simulator

`Simulator` drives a synthetic user against your app for a whole thread. With no
provider it falls back to a **seed-deterministic** template — the same seed yields
the same conversation, which is what makes simulated sessions usable as CI goldens.
`to_eval_case` packs the thread into `context["messages"]` for the conversational
metrics.

```python
from vincio.evals import Simulator, Persona, RunOutput
from vincio.evals.metrics import METRICS

def assistant(messages: list[dict]) -> str:        # your app under test; sync or async
    return "Open Settings then Security then Reset. It takes 5 minutes."

convo = Simulator(seed=7).simulate(
    assistant, Persona(name="sam", goal="reset password", max_turns=3)
)
case = convo.to_eval_case(id="reset")
run = RunOutput(output=convo.turns[-1]["content"])
convo.goal_achieved, convo.rounds                  # await Simulator(...).asimulate(...) for async
METRICS["conversation_outcome"](case, run).value   # did the thread reach the goal
METRICS["intent_resolution"](case, run).value      # fraction of user turns addressed
```

Stitch real production traffic into multi-turn goldens the same shape:

```python
from vincio.evals import dataset_from_traces

golden = dataset_from_traces(exporter.load_all(), group_by_session=True)
```

## 3. Online / continuous eval

Score a sampled fraction of live runs *after* the response is finalized — scheduled
off the hot path with deterministic 1-in-N sampling. Each score is written as a time
series to the metadata store (kind `eval_results`); nothing is mirrored to any
external platform. Each call emits an `eval.online` event.

```python
app.add_online_evaluator("answer_relevance", sample_rate=0.1)
app.add_online_evaluator("goal_accuracy", sample_rate=0.2)

for q in ("How long are refunds?", "What is the fee?", "When does it renew?"):
    app.run(q)

await app.aflush_online()                          # drain in-flight scoring (tests/shutdown)
app.online_evaluators[0].series()                  # the recorded score rows, oldest first
```

## 4. Drift detection

`DriftMonitor` compares a recent window against a baseline — both on raw scores
(mean shift + z-score) and on embedding distributions. On drift it raises a
`drift.detected` event on the bus and persists baselines (kind `drift_baselines`).

```python
from vincio.evals import DriftMonitor

monitor = DriftMonitor(
    bus=app.events, store=app.store,
    score_threshold=0.1, embedding_threshold=0.15,
)
monitor.set_score_baseline("goal_accuracy", [0.90, 0.91, 0.89, 0.92])
report = monitor.check_scores("goal_accuracy", [0.60, 0.62, 0.58])
report.drifted, report.delta, report.z_score       # -> True, ...

monitor.set_embedding_baseline(golden_vectors)
monitor.check_embeddings(live_vectors)             # distribution-shift report
```

From CI — exits non-zero on drift:

```bash
vincio eval drift baseline.json current.json --threshold 0.1
```

## 5. Human annotation & Cohen's κ

An LLM judge should only gate CI once it has *demonstrably* agreed with people.
`AnnotationQueue` records judge↔human pairs and reports Cohen's κ; `judge_trusted`
and `gating_weight` turn that agreement into a CI-gating decision.

```python
from vincio.evals import AnnotationQueue, cohens_kappa

q = AnnotationQueue(name="judge_cal")
item = q.add(run_id="r1", judge_score=0.9)
q.label(item.id, human_score=1.0)
q.agreement()                          # {"cohens_kappa": ..., "exact_agreement": ..., "n": ...}
q.judge_trusted(threshold=0.6)         # bool; q.gating_weight(threshold=0.6) -> 0.0 | 1.0

cohens_kappa([(0.9, 1.0), (0.2, 0.0)], bins=2)
```

`GEvalJudge.calibrate(pairs)` now also returns `cohens_kappa`, and
`judge.gating_weight(threshold=0.6)` returns `1.0` only once calibrated κ clears
the bar. From the CLI:

```bash
vincio eval annotate labels.jsonl --threshold 0.6 --bins 2
```

## 6. A/B in production

`app.experiment` runs a production-style A/B over prompt/model/config variants of
the *same* app, comparing on eval metrics **and** cost with significance tests.
Variant dict keys: `model`, `prompt`, `apply` (a `callable(app)`), `params`.

```python
exp = app.experiment(
    "prompt_ab",
    variants={
        "baseline": {"model": "..."},
        "concise":  {"prompt": concise_spec},
    },
    dataset=golden,
    metrics=["goal_accuracy", "cost"],
)
exp.compare()                  # per-metric means by variant + best per metric
exp.cost()                     # total USD per variant
exp.significance("goal_accuracy")   # per-variant ab_test vs baseline (paired/Welch t-test)
```

## 7. One metric, three jobs

The differentiator: a trajectory metric is just a metric, so the very object that
gates a release offline also guards generations at run time and steers the
optimizer.

```python
# (a) the same metric as a runtime guardrail
app.add_metric_rail("toxicity", threshold=0.0)          # block; lower-is-better fires above
app.add_metric_rail("answer_relevance", threshold=0.3, action="warn")  # higher-is-better fires below

# or build the predicate directly: (text, params) -> message | None
from vincio.evals import metric_guardrail
guard = metric_guardrail("groundedness", threshold=0.8)

# (b) the same metric as optimizer fitness — trajectory metrics flow into
#     report.metric_values and onto the Pareto frontier
from vincio.optimize import AGENTIC_OBJECTIVES, pareto_loop   # goal_accuracy,
                                                              # tool_call_accuracy,
                                                              # step_efficiency, cost
result = await pareto_loop(candidates, evaluate_fn, dataset,
                           baseline=baseline, objectives=AGENTIC_OBJECTIVES)
```

Direction is inferred from `LOWER_IS_BETTER`: lower-is-better metrics fire when
the value exceeds the threshold, higher-is-better when it falls below. Pass
`evidence`/`expected`/`input` through the rail `params` for metrics that need them.

## Why in-process

LangSmith, Ragas, and DeepEval send your traces *out* to a platform to be scored.
Vincio scores the trajectory **in your process**, in the same model that runs the
agent at runtime — so the metric that gates a release offline is the identical
object that guards live generations and that the optimizer maximizes. Offline and
deterministic by default (mock provider, seed-deterministic simulator), with no
hosted dependency. See the [evaluation concepts](../concepts/evals.md) and the
[run evals guide](run-evals.md), and run
[`examples/26_agentic_eval.py`](../../examples/26_agentic_eval.py) end to end.
