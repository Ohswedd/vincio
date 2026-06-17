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

Datasets also come from **traces** (one command — see
[observability](observability.md)) and from **your own corpus**:

```python
from vincio.evals import SyntheticGenerator, dataset_from_traces

golden = SyntheticGenerator(seed=7).generate(documents, n=50)   # offline templates
golden = SyntheticGenerator(provider=p, model="gpt-5.2-mini").generate(documents, n=50)
production = dataset_from_traces(exporter.load_all(), min_feedback_score=0.5)
```

Synthetic cases carry difficulty (`easy` stated facts, `medium` cloze values,
`hard` multi-hop across sources), coverage (round-robin over sources, dedupe),
and provenance (`metadata.source_ids`, source sentences in `rubric.facts`).

## Metrics

- **Task** — `exact_match`, `lexical_overlap`, `classification_accuracy`, `extraction_f1`
- **Grounding** — `groundedness`, `unsupported_claim_rate`, `citation_accuracy`,
  `citation_recall`, `context_precision`, `context_recall`
- **Quality & safety** — `faithfulness`, `answer_relevance`, `hallucination`
  (strict number checking: "90 days" against evidence saying "30 days" fails),
  `toxicity`, `bias`, `summarization_quality`
- **Conversational** — `knowledge_retention` (flags re-asking for facts the
  user already gave), `conversation_relevance`, `conversation_outcome` (did the
  thread achieve the user's goal — `rubric["goal"]` or `rubric["goal_keywords"]`),
  `intent_resolution` (fraction of user turns the assistant addressed) — all read
  the session from `case.context["messages"]`
- **Trajectory & tool-use** — `tool_call_accuracy`, `tool_call_f1`,
  `goal_accuracy`, `plan_adherence`, `plan_quality`, `step_efficiency`,
  `topic_adherence` — they score *how* the agent got there, not just the final
  answer (see [Trajectory & tool-use metrics](#trajectory--tool-use-metrics))
- **Operational** — `cost`, `latency`, `input_tokens`, `output_tokens`, `retries`
- **Retrieval** — `recall_at_k`, `precision_at_k`, `mrr`, `ndcg`
- **Agent/memory** — via `AgentState.metrics()` and `MemoryEngine.stats()`

Register custom metrics with `@register_metric("name")`. All metrics are
deterministic and offline; the same objects run as eval metrics, runtime
evaluators (`app.add_evaluator`), and test assertions (`vincio.testing`).

## Trajectory & tool-use metrics

A run can answer right while taking the wrong path — output-only eval can't see
that. The seven `TRAJECTORY_METRICS` (`tool_call_accuracy`, `tool_call_f1`,
`goal_accuracy`, `plan_adherence`, `plan_quality`, `step_efficiency`,
`topic_adherence`) score the **trajectory** carried on the `RunOutput`. They have
the ordinary metric signature `(EvalCase, RunOutput) -> MetricResult` with a value
in `[0, 1]`, so they sit beside output-only metrics in the same run.

Attach a trajectory from a completed run — no re-instrumentation:

```python
from vincio.evals import RunOutput

run = RunOutput.from_agent_state(state)    # AgentState from app.agent(...).run(...)
run = RunOutput.from_crew_result(result)   # CrewResult
run = RunOutput.from_trace(trace)          # a captured observability Trace
```

Expected/optimal references live on the case `rubric`:

```python
rubric={"expected_tools": ["search", "summarize"],
        "plan": ["tool", "tool", "finalize"], "optimal_steps": 3, "topic": "..."}
```

`expected_tools` entries may be plain names or `{"tool": name, "arguments": {...}}`
(`tool_call_accuracy` checks the right tool, right args, in the right order;
`tool_call_f1` is an order-insensitive set F1 over tool names). When a `RunOutput`
has **no** trajectory, these metrics return a neutral `1.0`, so they don't
penalize output-only cases. `EvalReport.metric_families()` splits the two views:

```python
report.metric_families()    # {"output": {...}, "trajectory": {...}}
```

The `Trajectory` model (`vincio.evals.Trajectory`) holds `objective`,
`steps` (`TrajectoryStep`), `final_answer`, `success`, `termination_reason`, and
`usage`; `trajectory_from_agent_state` / `_crew_result` / `_trace` are the
underlying module functions.

## Multi-turn metrics & the Simulator

Conversational metrics read the whole thread from `case.context["messages"]`.
To produce that thread without a live user, drive your agent with the
**`Simulator`**:

```python
from vincio.evals import Simulator, Persona

def agent(messages: list[dict]) -> str: ...    # your app under test; sync or async

convo = Simulator(seed=7).simulate(
    agent, Persona(name="sam", goal="reset password", max_turns=3))
case = convo.to_eval_case(id="reset")          # context["messages"] = the full thread
convo.goal_achieved, convo.rounds, convo.turns
```

The simulator is LLM-backed when given a `provider` + `model`, otherwise it falls
back to a **seed-deterministic** template — the same seed yields an identical
conversation, which is what makes simulated sessions usable as CI goldens. Use
`await Simulator(...).asimulate(agent, persona)` for async agents. To turn real
traffic into multi-turn goldens, `dataset_from_traces(traces,
group_by_session=True)` stitches a session's traces into one case.

## Judges

`DeterministicJudge`, `ModelJudge` (rubric + structured score, calibrated by
repeated sampling), `EmbeddingJudge`, `HybridJudge` (weighted blend), and
**`GEvalJudge`** — rubric-based G-Eval: it derives explicit evaluation steps
from plain-language criteria once, scores on a 1–5 form-filling scale
(`samples > 1` approximates probability-weighted scoring), and calibrates
against human labels:

```python
judge = GEvalJudge(provider, model="gpt-5.2-mini",
                   criteria="The answer must be factually correct and cite its sources.",
                   samples=3)
judge.calibrate([(0.75, 0.9), (0.5, 0.7)])   # (judge, human) pairs → linear fit + r
```

### Human annotation & Cohen's κ

`calibrate()` now also returns `"cohens_kappa"` — an LLM judge should only gate CI
once it has *demonstrably* agreed with people. Collect the labels through an
`AnnotationQueue`, then read inter-rater agreement:

```python
from vincio.evals import AnnotationQueue, cohens_kappa

q = AnnotationQueue(name="judge_cal")
item = q.add(run_id="r1", judge_score=0.9)
q.label(item.id, human_score=1.0)
q.agreement()                  # {"cohens_kappa": ..., "exact_agreement": ..., "n": ...}
q.judge_trusted(threshold=0.6) # κ clears the bar?
q.gating_weight(threshold=0.6) # 1.0 once trusted, else 0.0

cohens_kappa(pairs, bins=2)    # pairs of (judge, human) scores in [0, 1]
```

`judge.gating_weight(threshold=0.6)` returns `1.0` only when the judge's
calibrated κ clears the bar. CLI:
`vincio eval annotate labels.jsonl [--threshold X] [--bins N]`.

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

## Online / continuous eval

Score a sampled fraction of **live** runs with the same metric objects. Scoring
runs after the response is finalized — scheduled off the hot path with
deterministic 1-in-N sampling — so it never adds latency to the request:

```python
app.add_online_evaluator("answer_relevance", sample_rate=0.1)
app.add_online_evaluator("goal_accuracy", sample_rate=0.2)

app.online_evaluators[0].series()    # the scored rows (time series)
await app.aflush_online()            # drain in-flight scoring (tests/shutdown)
```

Each score is written as a time series to the metadata store (kind
`eval_results`) and emits an `eval.online` event — no external mirroring.
`app.add_online_evaluator` is `@experimental(since="1.2")`.

## Drift detection

A `DriftMonitor` watches those live scores (and golden embeddings) against a
baseline and raises a `drift.detected` event when quality moves:

```python
from vincio.evals import DriftMonitor

monitor = DriftMonitor(bus=app.events, store=app.store,
                       score_threshold=0.1, embedding_threshold=0.15)
monitor.set_score_baseline("goal_accuracy", baseline_values)
report = monitor.check_scores("goal_accuracy", recent_values)
report.drifted, report.delta, report.z_score

monitor.set_embedding_baseline(golden_vectors)
monitor.check_embeddings(live_vectors)
```

On drift it persists baselines (kind `drift_baselines`) and emits the event. CLI:
`vincio eval drift baseline.json current.json [--threshold X]` (exits non-zero on
drift).

## Experiments & A/B significance

Reports log to a local experiment store (the same SQLite metadata store the
runtime uses); comparisons and ablations test for statistical significance
with a paired t-test when reports share case ids, Welch's t-test otherwise —
pure Python, no SciPy:

```python
from vincio.evals import ExperimentTracker, ab_test

tracker = ExperimentTracker(".vincio/experiments.db")
tracker.log("retrieval_ab", baseline_report, variant="baseline", params={"mode": "bm25"})
tracker.log("retrieval_ab", hybrid_report, variant="hybrid", params={"mode": "hybrid_full"})
tracker.compare("retrieval_ab")["best"]            # best variant per metric
tracker.ablation("retrieval_ab")                   # deltas + p-values vs baseline
ab_test(baseline_report, hybrid_report, "groundedness")  # {delta, p_value, significant, ...}
```

`app.experiment` runs variants of the live app over a golden dataset and tests
the same way (`@experimental(since="1.2")`):

```python
exp = app.experiment("prompt_ab",
    variants={"baseline": {"model": "..."}, "concise": {"prompt": concise_spec}},
    dataset=golden, metrics=["goal_accuracy", "cost"])
exp.compare()                       # per-metric means by variant + best per metric
exp.cost()                          # total USD per variant
exp.significance("goal_accuracy")   # per-variant ab_test vs baseline
```

Variant dict keys: `model`, `prompt`, `apply` (a `callable(app)`), `params`.

## Red-teaming

An adversarial suite (jailbreaks, prompt injections, PII/secret-leak probes,
bias and toxicity provocations) judged **deterministically** by the security
engine's detectors and the safety metrics — attack probes carry a canary
token, so an attack only "succeeds" if the output proves compliance:

```python
from vincio.evals import RedTeamSuite

report = RedTeamSuite().run(app)        # or any callable str -> str
report.attack_success_rate              # gate this at 0.0
report.detector_coverage                # input-side injection detection rate
report.by_category()                    # per-category breakdown
```

Custom probes extend the built-ins via `RedTeamProbe`; the suite runs offline
and gates CI like any other report.

## Every metric is also a guardrail and an optimizer term

The same metric object does three jobs, so quality criteria stay in one place:

```python
from vincio.evals import metric_guardrail
from vincio.optimize import AGENTIC_OBJECTIVES

app.add_metric_rail("toxicity", threshold=0.0)        # metric → runtime guardrail
rail = metric_guardrail(metric, threshold=...)        # (text, params) -> message | None
optimize(objectives=AGENTIC_OBJECTIVES)               # metric → fitness term
```

A metric-as-guardrail reads its direction from `LOWER_IS_BETTER`
(lower-is-better fires above the threshold, higher-is-better fires below); pass
`evidence` / `expected` / `input` via the rail params. Because trajectory metrics
are ordinary metrics, they flow into `report.metric_values` and the Pareto
frontier — the `AGENTIC_OBJECTIVES` preset is `goal_accuracy`,
`tool_call_accuracy`, `step_efficiency`, and `cost`, or pass your own
`ObjectiveSpec` list. Unlike platforms that ship traces out to score them,
Vincio scores the trajectory in-process, in the same model as the runtime, and
turns the very same metric into a guardrail and an optimization target.
`app.add_metric_rail` is `@experimental(since="1.2")`.

## Testing ergonomics

Unit-test LLM behavior with the `vincio.testing` assertions and the pytest
plugin (auto-registered on install) — see the
[testing guide](../guides/test-llm-apps.md):

```python
from vincio.testing import assert_eval, assert_grounded

def test_refund_answer():
    result = app.run("What is the refund window?")
    assert_grounded(result, threshold=0.8)
    assert_eval(result, "What is the refund window?",
                metrics={"answer_relevance": 0.5, "hallucination": 0.0})

def test_packet_shape(vincio_snapshot):      # plugin fixture
    vincio_snapshot.match_packet(compiled)   # refresh: pytest --vincio-update-snapshots
```
