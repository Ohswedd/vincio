# Coming from Ragas to Vincio

Ragas is eval-only, and its core ideas map cleanly onto Vincio: an
`EvaluationDataset` becomes a Vincio `Dataset`, Ragas metrics map onto
Vincio's built-in metrics, and `ragas.evaluate()` becomes `EvalRunner(...).run()`
or `vincio eval run`. Because Vincio evaluates the same app you ship, you can
adopt it incrementally, keep Ragas for ad-hoc analysis while moving CI scoring
and gating over case by case.

## Concept mapping

| Ragas | Vincio | Notes |
|---|---|---|
| `faithfulness` | `groundedness` | are claims supported by retrieved context |
| `answer_relevancy` / `answer_correctness` | `lexical_overlap` | answer vs `expected` |
| response / schema conformance | `schema_validity` | validates against `output_schema` |
| (not measured) | `cost`, `latency` | first-class metrics in every run |
| `EvaluationDataset` / test set | `Dataset` (JSONL) | `{id, input, expected, rubric, tags, difficulty}` |
| `ragas.evaluate(dataset, metrics=...)` | `EvalRunner(app, metrics=[...]).run("golden.jsonl")` | or `vincio eval run` |
| metric thresholds checked by hand | `gates={...}` / `--gate "groundedness=>= 0.8"` | non-zero exit on failure |
| `Result` / scores dataframe | `report.print_summary()` | per-case and aggregate |
| LLM-as-judge metrics | `ModelJudge(...)` | repeated-sample scoring to reduce noise |

## Bring your assets across

Ragas has no documents or tools to adapt, what carries over is your test set.
A Ragas record (question / ground-truth / contexts) becomes one line of a
Vincio `Dataset`, where `expected` holds the ground truth and `rubric.facts`
holds the supporting sentences grounding metrics check against:

```jsonl
{"id": "case_001", "input": "Can I get a refund?", "expected": "Refunds within 30 days.", "rubric": {"facts": ["refunds within 30 days"]}, "tags": ["refund"], "difficulty": "easy"}
```

Commit it under `tests/golden/` or `golden/` so it lives alongside your code.
Load it as a `Dataset` or pass the path straight to a runner:

```python
from vincio import Dataset

dataset = Dataset.load("golden/refunds.jsonl")
```

## In Vincio

A Ragas evaluation builds a dataset, names metrics, and calls `evaluate`:

```python
# before (Ragas)
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

result = evaluate(eval_dataset, metrics=[faithfulness, answer_relevancy])
print(result)
```

In Vincio the same scoring runs against the app you actually ship, so retrieval,
budgeting, and citation enforcement are part of what gets measured:

```python
# after (Vincio)
from vincio import ContextApp
from vincio.evals.runners import EvalRunner

app = ContextApp(name="kb", provider="openai", model="gpt-5.2")
app.add_source("kb", path="./docs", retrieval="hybrid")
app.set_policy("answer_only_from_sources", True)
app.set_policy("require_citations", True)

report = EvalRunner(
    app,
    metrics=["groundedness", "lexical_overlap", "schema_validity", "cost", "latency"],
    gates={"groundedness": ">= 0.8", "schema_validity": "== 1.0"},
).run("golden/refunds.jsonl")
report.print_summary()
```

The same run from the command line, with baseline comparison and a CI gate:

```bash
vincio eval run golden/refunds.jsonl --app app.py \
  --metric groundedness --metric lexical_overlap --metric schema_validity \
  --gate "groundedness=>= 0.8"
```

Every case that fails a gate links back to a full trace (`result.trace_id`), so
you can open `vincio trace view` and see the scored context packet that produced
the low score, not just the number.

## What Vincio adds

- **CI gates, not just scores**: `gates={...}` and `--gate` turn metrics into a
  pass/fail contract that exits non-zero, so a regression blocks the merge
  instead of landing silently.
- **Baseline diffing and regression detection**: compare a run against a stored
  baseline report to catch metric drift between versions, rather than eyeballing
  a dataframe each time.
- **Judges wired to the same report**: add a `ModelJudge` with repeated-sample
  scoring alongside deterministic metrics; one runner, one report format.
- **Trace-derived datasets**: production runs each write a trace, so you can
  curate golden cases from real traffic instead of hand-authoring every example.
- **Native, provider-neutral observability**: every eval case runs the same
  pipeline as production with cost and latency tracked as first-class metrics
  and a `trace_id` per case.
- **A closed improvement loop**: trace → dataset → eval → optimize → promote:
  scores feed context and prompt optimization behind safety gates, so an
  evaluation can change the system, not just measure it. Ragas stops at the
  score.

## Next steps

- [run evals](run-evals.md)
- [structured output](structured-output.md)
- [optimize context](optimize-context.md)
- [build a RAG app](build-rag-app.md)
- [orchestrate agents](orchestrate-agents.md)
- [evals concepts](../concepts/evals.md)
- [observability concepts](../concepts/observability.md)
- [retrieval concepts](../concepts/retrieval.md)
- [Vincio vs Ragas](../comparisons/ragas.md)

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

_Migrating from another library:_
- [Coming from LangChain / LangGraph to Vincio](migrate-from-langchain.md)
- [Coming from LlamaIndex to Vincio](migrate-from-llamaindex.md)
- [Coming from Mem0 to Vincio](migrate-from-mem0.md)
- [Reference: capability map](../reference/capability-map.md)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
