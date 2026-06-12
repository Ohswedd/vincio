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

Snapshots capture the *structure* of a context packet or trace — what was
included, in what shape — and normalize away volatile fields (ids,
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

For dataset-level checks, keep using the eval runner — same metrics, same
thresholds, non-zero exit on gate failure:

```bash
vincio eval run tests/golden/basic.jsonl --app app.py \
  --gate "groundedness=>= 0.95" --output report.json
```

## Patterns

- **Pin the provider**: tests should run against `MockProvider` (deterministic,
  schema-aware) and only hit real models behind an env flag.
- **Assert behavior, not strings**: prefer `assert_grounded` / metric
  thresholds over exact-output comparisons; use snapshots for structure.
- **One bar per metric**: thresholds are explicit at the call site, so a
  failing test names the exact contract that broke.
