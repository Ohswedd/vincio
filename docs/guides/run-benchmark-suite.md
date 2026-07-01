# Guide: run a benchmark suite

The open evaluation plane runs the standard public model benchmarks (MMLU, GPQA,
GSM8K, HumanEval, IFEval, TruthfulQA, RULER, …) grouped by niche, with a
[provenance tier](../concepts/open-evaluation-plane.md) on every number. This guide
runs one, reads it, reports it, persists it, and extends it — fully offline on the
bundled Tier-S fixtures.

## Run a niche offline

`app.benchmark_suite` is the front door. The default tier `"static"` replays the
bundled fabricated fixtures, so it needs no key and no network.

```python
from vincio import ContextApp
from vincio.providers import MockProvider

app = ContextApp(name="eval", provider=MockProvider(), model="mock-1")

run = app.benchmark_suite("knowledge", tier="static")
run.overall()            # the mean primary score across the niche
run.niche_scores()       # mean primary per niche (the radar axes)
run.determinism_digest   # a content hash; identical across machines at Tier-S/R
for r in run.runs:
    print(r.benchmark_id, r.primary_metric, r.primary, "tier", r.tier.code)
```

`benchmarks` accepts a single id (`"knowledge.mmlu"`), a niche (`"knowledge"`),
`"all"`, or a list. `BenchmarkSuite` is the underlying engine if you want it
without an app:

```python
from vincio.evals.suite import BenchmarkSuite

suite = BenchmarkSuite(concurrency=8)
run = suite.run(["knowledge", "reasoning.gsm8k"], tier="static")
```

## The tiers, and why a fixture can't lie

The engine computes the tier a run may claim from its inputs and refuses to print a
higher one. A bundled fixture is fabricated, so its ceiling is `S`:

```python
from vincio.core.errors import TierViolationError

try:
    suite.run("knowledge.mmlu", tier="live")   # only a Tier-S fixture is available
except TierViolationError as exc:
    print(exc)   # cannot report tier L: the run's inputs only support tier S …
```

To run a **Recorded** or **Live** tier, supply the dataset yourself — a hash-pinned
slice (Recorded) or the full dataset against a live model (Live):

```python
from vincio.evals.suite import BenchmarkDataset, ProvenanceTier
from vincio.evals.suite.adapters import mmlu_tasks_from_export

# A hash-pinned recorded slice (replayed against recorded outputs → gates CI).
recorded = BenchmarkDataset.from_export(
    records, loader=mmlu_tasks_from_export,
    tier=ProvenanceTier.RECORDED, benchmark_id="knowledge.mmlu",
    pinned_hash="…",   # drift in the task set is caught against this pin
)
run = suite.run("knowledge.mmlu", tier="recorded",
                datasets={"knowledge.mmlu": recorded})

# Live: the full dataset against a live model (reported, never gated).
live = BenchmarkDataset.from_huggingface(            # needs vincio[eval-datasets]
    "cais/mmlu", loader=mmlu_tasks_from_export, split="test")
run = app.benchmark_suite("knowledge.mmlu", tier="live",
                          datasets={"knowledge.mmlu": live})
```

## Report it — every format, every number tiered

```python
from vincio.evals.suite import SuiteReport

report = SuiteReport(run)
report.to_markdown()        # also: to_html(), to_json(), to_csv()
report.save("report.html")  # format inferred from the suffix
report.save("report.pdf")   # PDF needs vincio[eval-pdf]
```

Every rendering carries the run's tier and cites the exact scored items (failing
tasks are listed by id). The Markdown / HTML / JSON / CSV bytes are a pure function
of the run, so a Tier-S report diffs cleanly across machines.

## Compare models — leaderboard, charts, and a run store

```python
from vincio.evals.suite import Leaderboard, RunStore, leaderboard_chart, radar_chart

board = Leaderboard.from_runs([run_a, run_b])   # ranked by overall primary
board.to_markdown()
leaderboard_chart(board).save("leaderboard.json")   # deterministic Vega-Lite
radar_chart(run_a).save("radar.json")               # .png needs vincio[eval-viz]

store = RunStore(".vincio/eval_runs.db")            # SQLite (stdlib)
store.save(run_a, version="1.0")
store.save(run_b, version="2.0")
store.compare_runs(run_a.run_id, run_b.run_id)      # per-benchmark delta
store.model_version_diff("my-model")                # did this version regress?
```

## Extend it — add a benchmark without touching the core

Register a `BenchmarkSpec` in-process, or ship it as a `vincio.benchmarks`
entry-point so installing your package is all it takes:

```python
from vincio.evals.suite import BenchmarkSpec, register_benchmark
from vincio.evals.benchmarks import BenchmarkAdapter, BenchmarkResult

class MyAdapter(BenchmarkAdapter):
    name = "my_eval"
    async def score(self, task, output):
        ok = str(output).strip() == str(task.gold).strip()
        return BenchmarkResult(task_id=task.id, success=ok, score=1.0 if ok else 0.0)

register_benchmark(BenchmarkSpec(
    id="custom.my_eval", niche="custom", title="My eval", adapter=MyAdapter,
    primary_metric="accuracy",
    static_tasks=[{"id": "t1", "prompt": "2+2?", "gold": "4", "recorded": "4"}],
))
```

## From the CLI

```bash
vincio eval suite run knowledge.mmlu --tier static          # text summary
vincio eval suite run all --format markdown --output report.md
vincio eval suite run knowledge --store runs.db             # persist the run
vincio eval suite leaderboard --store runs.db               # rank persisted runs
vincio eval suite report run.json --format pdf --output r.pdf
vincio eval suite compare <runA> <runB> --store runs.db
```

The CLI commands `run` / `leaderboard` / `report` / `compare` live under
`vincio eval suite`, beside the application-eval commands (`vincio eval run`,
`vincio eval report`). See the [reference](../reference/cli.md) for every flag, and
the [example](../../examples/16_open_evaluation_plane.py) for the full tour.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: The open evaluation plane](../concepts/open-evaluation-plane.md)
- [Example: 16_open_evaluation_plane.py](../../examples/16_open_evaluation_plane.py)
- [Concept: Observability](../concepts/observability.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#optimization)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
