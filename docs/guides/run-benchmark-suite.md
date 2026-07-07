# Guide: run a benchmark suite

Vincio's [benchmark platform](../concepts/open-evaluation-plane.md) answers three
questions as three **tracks**, each with a [provenance tier](../concepts/open-evaluation-plane.md)
on every number:

| Track | Question | CLI |
|---|---|---|
| **1 · Model** | how good is a *model* on the standard public benchmarks? | `vincio bench model` |
| **2 · Uplift** | how much does routing a model *through Vincio* change its scores? | `vincio bench uplift` |
| **3 · Feature** | how good is a Vincio *feature* vs a real competitor library? | `vincio bench feature` |

This guide runs each track offline, runs Track 1 and Track 3 with real numbers,
reads the reports, and registers a custom benchmark for every track. One command
drives all three — `vincio bench <track>` — and `python benchmarks/bench.py <track>`
is the equivalent script driver. The authoritative map is
[`benchmarks/PROVENANCE.md`](../../benchmarks/PROVENANCE.md).

## Live-first, honesty-gated

The platform is **Live-first**: the number that matters is the one produced
against a real model or a real competitor library on your hardware.
`vincio bench feature` runs Vincio head-to-head against the *installed*
competitor and grades both arms with the contest's own deterministic metric;
`vincio bench model --tier live` scores a real provider over a real dataset.

The Tier-S / mockup inputs are the **honesty rail, not a performance claim**.
Every number carries a [provenance tier](../concepts/open-evaluation-plane.md) —
`STATIC (S) < RECORDED (R) < LIVE (L)` — and only the *non-Live* tiers gate CI:
they saturate by design to prove the mechanism runs end to end offline,
deterministically, on every machine. A Tier-S score is never a real-world
result, and the suite refuses to print a tier its inputs cannot support
(raising `TierViolationError`), so a fabricated fixture can never masquerade as a
live win, and a `feature` contest reports **Live only when every declared
competitor actually executed** — an absent library leaves that contest at Tier-S,
never faked.

## See the whole platform

```bash
vincio bench list            # all three tracks' catalogs (--json for machine form)
```

It prints the 29 model benchmarks across ten niches, the 6 uplift benchmarks, and
the 8 feature contests — the whole system at a glance.

## Run each track offline (Tier-S)

Every track runs with no key and no network on its Tier-S / mockup inputs, which
gate CI. These numbers saturate by design — they prove the mechanism, not a
real-world score.

```bash
vincio bench model all --tier static     # track 1: fabricated fixtures, per-niche scores
vincio bench uplift                       # track 2: two recorded arms, direct vs Vincio
vincio bench feature                      # track 3: Vincio vs any installed competitor
```

Track 2 replays each benchmark's two recorded arms through the identical scorer and
reports the per-benchmark delta:

```text
uplift run uplift_… · tier S · overall 0.125 → 1.000 (+0.875)
  long_context.recall   needle_recall    0.000 -> 1.000  ^ +1.000
  output.schema_valid   valid_rate       0.000 -> 1.000  ^ +1.000
  rag.grounded          faithfulness     0.500 -> 1.000  ^ +0.500
  safety.injection      contained_rate   0.000 -> 1.000  ^ +1.000
```

Track 3 tiers each contest by whether its competitor actually ran — `[L]` when the
library is installed, `[S]` when it is absent (Vincio and the baseline still ran):

```text
feature run feat_… · suite tier S · 8 contests
  [L] retrieval.bm25    recall_at_1↑   winner: vincio
        vincio          1     0.60ms
        rank_bm25       1     2.05ms
  [S] memory.recall     current_fact_precision↑   winner: vincio
        vincio          1
        naive_keyword_store  0.5
```

## Track 3 with real numbers — live vs installed libraries

Track 3 is genuinely **Live** on this machine: install a competitor and the
head-to-head runs against it for real. The deterministic quality metric gates CI;
the latency ratio is real but machine-specific, so rerun it on your hardware.

```bash
pip install rank_bm25 tiktoken json-repair jinja2 pandas   # the real competitors
vincio bench feature                                       # now the contests report tier L
vincio bench feature retrieval.bm25 tokenization.count     # or a subset by id / capability
```

A contest with an uninstalled competitor is reported *skipped*, never fabricated —
so a partial install simply leaves those contests at Tier-S.

## Track 1 with real numbers — a live model

Track 1 Live needs a model key and a real dataset. The dedicated driver runs a
provider over hash-loaded datasets:

```bash
python benchmarks/eval_live.py --provider anthropic --model claude-opus-4-8 \
    --benchmarks knowledge.mmlu reasoning.gsm8k --tier live --dataset-dir ./datasets
```

The same track runs through the CLI over an app file (needed for `tier=live`):

```bash
vincio bench model knowledge.mmlu --tier live --app app.py --format markdown --output run.md
```

A bundled fixture is fabricated, so its ceiling is Tier-S; asking a fixture to print
a higher tier is refused rather than faked:

```python
from vincio.evals.suite import BenchmarkSuite
from vincio.core.errors import TierViolationError

suite = BenchmarkSuite(concurrency=8)
try:
    suite.run("knowledge.mmlu", tier="live")   # only a Tier-S fixture is available
except TierViolationError as exc:
    print(exc)   # cannot report tier L: the run's inputs only support tier S …
```

## Read the reports

Each track renders to Markdown from the CLI, and to JSON with `--json`:

```bash
vincio bench uplift --format markdown          # render_uplift_report under the hood
vincio bench feature --format markdown         # render_feature_report under the hood
vincio bench uplift --json                      # the whole run as JSON
```

In-process, the report renderers and the Track-1 `SuiteReport` are public:

```python
from vincio.evals.suite import (
    UpliftSuite, FeatureSuite, BenchmarkSuite,
    render_uplift_report, render_feature_report, SuiteReport,
)

print(render_uplift_report(UpliftSuite().run("all", tier="static")))
print(render_feature_report(FeatureSuite().run("all")))

run = BenchmarkSuite(concurrency=8).run(["knowledge", "reasoning.gsm8k"], tier="static")
report = SuiteReport(run)
report.to_markdown()          # also: to_html(), to_json(), to_csv()
report.save("report.html")    # format inferred from the suffix; PDF needs vincio[eval-pdf]
```

Every Track-1 rendering carries the run's tier and cites the exact scored items
(failing tasks by id), and its bytes are a pure function of the run, so a Tier-S
report diffs cleanly across machines. `app.benchmark_suite("knowledge",
tier="static")` is the in-process front door for Track 1; `Leaderboard` and
`RunStore` rank and persist Track-1 runs for model-version comparison.

## Register a custom benchmark for each track

Each track has a single extension point; installing your package or importing your
module is all it takes.

**Track 1 — a public-benchmark adapter** (`register_benchmark`), in-process or as a
`vincio.benchmarks` entry point:

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
# vincio bench model custom.my_eval --tier static
```

**Track 2 — a two-armed uplift benchmark** (`register_uplift_benchmark`): one
adapter, and tasks that carry both a `recorded` (direct) and a `recorded_vincio`
(routed) answer scored by the identical adapter:

```python
from vincio.evals.suite import UpliftBenchmark, register_uplift_benchmark
from vincio.evals.benchmarks import BenchmarkAdapter, BenchmarkResult

class ExactMatch(BenchmarkAdapter):
    name = "exact_match"
    async def score(self, task, output):
        ok = str(output).strip() == str(task.gold).strip()
        return BenchmarkResult(task_id=task.id, success=ok, score=1.0 if ok else 0.0)

register_uplift_benchmark(UpliftBenchmark(
    id="format.currency", title="Currency formatting", capability="output",
    adapter=ExactMatch, primary_metric="accuracy",
    tasks=[
        {"id": "c1", "prompt": "Format 1000 as USD.", "gold": "$1,000.00",
         "recorded": "1000 dollars",          # the direct arm gets it wrong
         "recorded_vincio": "$1,000.00"},     # routed through Vincio's output contract
    ],
))
# vincio bench uplift format.currency          # direct vs Vincio, per-benchmark delta
```

**Track 3 — a feature contest** (`register_feature_contest`): a `runner` returning
the ordered `Contender`s, each producing a `FeatureMeasurement`. A `competitor`
contender declares the modules it `requires`, so a missing library is skipped, never
faked:

```python
from vincio.evals.suite import (
    Contender, FeatureContest, FeatureMeasurement, register_feature_contest,
)

def _titlecase_contest() -> FeatureContest:
    corpus = ["the context engine", "grounded by design"]
    gold = ["The Context Engine", "Grounded By Design"]

    def score(fn) -> FeatureMeasurement:
        acc = sum(fn(t) == g for t, g in zip(corpus, gold)) / len(gold)
        return FeatureMeasurement(primary=round(acc, 4), metrics={"accuracy": round(acc, 4)})

    def vincio() -> FeatureMeasurement:
        return score(str.title)                 # the Vincio feature under test

    def competitor() -> FeatureMeasurement:
        import titlecase                          # only imported when installed
        return score(titlecase.titlecase)

    def runner() -> list[Contender]:
        return [
            Contender("vincio", vincio, kind="vincio"),
            Contender("titlecase", competitor, kind="competitor", requires=("titlecase",)),
        ]

    return FeatureContest(
        id="text.titlecase", title="Title-casing", capability="text",
        primary_metric="accuracy", runner=runner,
    )

register_feature_contest(_titlecase_contest())
# vincio bench feature text.titlecase          # Live iff `titlecase` is installed, else Tier-S
```

## The CLI surface

```bash
vincio bench list                                # all three tracks' catalogs
vincio bench model all --tier static             # track 1 (Tier-S; also --tier live --app app.py)
vincio bench model knowledge --format markdown --output r.md --store runs.db
vincio bench uplift [ids…] [--tier static] [--format markdown] [--json]
vincio bench feature [ids…] [--format markdown] [--json]
```

Track 1 is also the **open evaluation plane**: the same runs are available under
`vincio eval suite run|leaderboard|report|compare` and as `app.benchmark_suite(...)`
in-process. See the [CLI reference](../reference/cli.md) for every flag, and the
[example](../../examples/16_open_evaluation_plane.py) for the full Track-1 tour.

## Gotchas

- **Never quote a Tier-S number as a benchmark result.** It gates CI precisely
  because it saturates — it proves the plumbing, not the score. Cite a `[L]`
  (Live) number, and say what produced it.
- **Latency ratios are machine-specific.** A `feature` contest's quality metric
  is deterministic and portable; its timing is real but tied to your hardware, so
  rerun it where it matters and never compare timings across machines.
- **A partial competitor install silently lowers the tier.** If `rank_bm25` is
  present but `tiktoken` is not, `retrieval.bm25` reports `[L]` and
  `tokenization.count` drops to `[S]` — skipped, not fabricated. Install the full
  competitor set (`rank_bm25 tiktoken json-repair jinja2 pandas`) before reading
  Track 3 as Live.
- **Track-1 report bytes are a pure function of the run**, so a Tier-S report
  diffs cleanly across machines; a Live report will not, by design.

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
