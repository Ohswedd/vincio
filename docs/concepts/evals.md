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

- **Task** — `exact_match`, `semantic_similarity`, `classification_accuracy`, `extraction_f1`
- **Grounding** — `groundedness`, `unsupported_claim_rate`, `citation_accuracy`,
  `citation_recall`, `context_precision`, `context_recall`
- **Quality & safety (0.5)** — `faithfulness`, `answer_relevance`, `hallucination`
  (strict number checking: "90 days" against evidence saying "30 days" fails),
  `toxicity`, `bias`, `summarization_quality`
- **Conversational (0.5)** — `knowledge_retention` (flags re-asking for facts the
  user already gave), `conversation_relevance` — both read the session from
  `case.context["messages"]`
- **Operational** — `cost`, `latency`, `input_tokens`, `output_tokens`, `retries`
- **Retrieval** — `recall_at_k`, `precision_at_k`, `mrr`, `ndcg`
- **Agent/memory** — via `AgentState.metrics()` and `MemoryEngine.stats()`

Register custom metrics with `@register_metric("name")`. All metrics are
deterministic and offline; the same objects run as eval metrics, runtime
evaluators (`app.add_evaluator`), and test assertions (`vincio.testing`).

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
