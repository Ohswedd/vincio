# Guide: test LLM apps with pytest

Vincio ships a pytest plugin (auto-registered on install) and an assertion
API so LLM behavior is tested like code: deterministic, offline, CI-gated.

## Assertions

```python
from vincio.testing import assert_eval, assert_grounded, assert_metric, assert_safe

def test_refund_answer(rag_app):
    result = rag_app.run("What is the refund window?")
    assert_grounded(result, threshold=0.8)            # groundedness >= 0.8
    assert_safe(result)                               # no toxicity, no bias
    assert_eval(result, "What is the refund window?",
                expected="30 days",
                metrics={"answer_relevance": 0.5,     # quality: >= threshold
                         "hallucination": 0.0})       # rates: <= threshold
```

Every assertion accepts a `RunResult`, a `RunOutput`, or a plain string (plus
optional `evidence=`). Failures print the metric value, the breakdown, and
the offending output:

```
AssertionError: hallucination=1.0 failed <= 0.0
  details: {'claims': 1, 'unsupported': ['The refund window is 90 days...']}
```

Quality metrics assert `>= threshold`; rate/cost-style metrics
(`hallucination`, `toxicity`, `bias`, `latency`, ...) assert `<= threshold`.
`assert_metric` checks a single metric; any metric registered with
`@register_metric` works.

## Snapshot tests for packets and traces

Snapshots capture the *structure* of a context packet or trace, what was
included, in what shape, and normalize away volatile fields (ids,
timestamps, durations, hashes), so they fail when behavior changes, not when
the clock does:

```python
def test_packet_shape(rag_app, vincio_snapshot):
    result = rag_app.run("What is the refund window?")
    trace = rag_app.tracer.exporter.get(result.trace_id)
    vincio_snapshot.match_trace(trace)        # span tree: types, names, statuses
    vincio_snapshot.match({"evidence": len(result.evidence)}, name="counts")
```

Snapshots live in `__snapshots__/<test_file>/<test_name>.json` next to the
test. The first run records; later runs compare and show a unified diff on
mismatch. Refresh intentionally:

```bash
pytest --vincio-update-snapshots
```

## Red-team in CI

```python
from vincio.evals import RedTeamSuite

def test_red_team(rag_app):
    report = RedTeamSuite().run(rag_app)
    assert report.attack_success_rate == 0.0, report.summary()
```

The suite is deterministic and offline (canary-based judging + security
detectors), so this is a normal CI test, not a flaky model-judged one.

## Eval gates in CI

For dataset-level checks, keep using the eval runner, same metrics, same
thresholds, non-zero exit on gate failure:

```bash
vincio eval run tests/golden/basic.jsonl --app app.py \
  --gate "groundedness=>= 0.95" --output report.json
```

## Unit-test agents on the trajectory

Output-only eval can't see a run that answers right while taking the wrong
path. Build a `RunOutput` from a completed run, no re-instrumentation, and
assert on trajectory metrics: the same `(EvalCase, RunOutput) -> MetricResult`
objects, just reading the `Trajectory` the run carried:

```python
from vincio.evals import EvalCase, RunOutput
from vincio.evals.metrics import METRICS

def test_agent_takes_the_right_path(support_agent):
    state = support_agent.run("Look up order 1234 and summarize its status")
    run = RunOutput.from_agent_state(state)        # or .from_crew_result / .from_trace
    case = EvalCase(
        id="order_status",
        input="Look up order 1234 and summarize its status",
        expected="shipped",
        rubric={"expected_tools": ["lookup_order", "summarize"],
                "plan": ["tool", "tool", "finalize"], "optimal_steps": 3},
    )
    # Trajectory metrics read the case rubric, so call them directly:
    assert METRICS["tool_call_accuracy"](case, run).value == 1.0   # right tools, order, args
    assert METRICS["goal_accuracy"](case, run).value == 1.0        # terminated ok, answer matches
```

`expected_tools` entries are tool names or `{"tool": name, "arguments": {...}}`.
The seven trajectory metrics, `tool_call_accuracy`, `tool_call_f1`,
`goal_accuracy`, `plan_adherence`, `plan_quality`, `step_efficiency`,
`topic_adherence` (`TRAJECTORY_METRICS`), all sit in the `METRICS` registry
next to output metrics, return `[0, 1]`, and assert `>= threshold`. A run
with no trajectory returns a neutral `1.0`, so they compose with output-only
metrics in one report; `EvalReport.metric_families()` shows the `"output"`
and `"trajectory"` views side by side.

## Deterministic multi-turn test doubles

`Simulator(seed=...)` is a reproducible multi-turn driver: with no
provider it falls back to a seed-deterministic template, so the **same seed
yields the same conversation**, usable as a CI golden, not a flaky
model-judged one:

```python
from vincio.evals import Simulator, Persona, RunOutput
from vincio.evals.metrics import METRICS

def test_password_reset_thread(support_agent):
    def agent(messages: list[dict]) -> str:        # your app under test (sync or async)
        return support_agent.run(messages[-1]["content"]).output

    persona = Persona(name="sam", goal="reset password", max_turns=3)
    convo = Simulator(seed=7).simulate(agent, persona)
    again = Simulator(seed=7).simulate(agent, persona)
    assert convo.turns == again.turns              # same seed -> identical conversation
    assert convo.goal_achieved

    case = convo.to_eval_case(id="reset")          # context["messages"] = the whole thread
    run = RunOutput(output=convo.turns[-1]["content"])
    assert METRICS["intent_resolution"](case, run).value >= 0.8   # every user turn addressed
```

Give it `provider`+`model` for an LLM-backed user; leave them off for
offline CI. Conversational metrics (`conversation_outcome`,
`intent_resolution`) read `case.context["messages"]`, and
`dataset_from_traces(traces, group_by_session=True)` stitches a real
session's traces into the same multi-turn golden shape.

## Patterns

- **Pin the provider**: tests should run against `MockProvider` (deterministic,
  schema-aware) and only hit real models behind an env flag.
- **Assert behavior, not strings**: prefer `assert_grounded` / metric
  thresholds over exact-output comparisons; use snapshots for structure.
- **One bar per metric**: thresholds are explicit at the call site, so a
  failing test names the exact contract that broke.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: Evaluation & continuous quality](../concepts/evals.md)
- [Guide: run evals](run-evals.md)
- [Guide: agentic evaluation & continuous quality](agentic-eval.md)
- [Example: 07_evaluation_observability.py](../../examples/07_evaluation_observability.py)
- [Concept: Observability](../concepts/observability.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#optimization)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
