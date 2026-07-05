# The benchmark platform

Vincio ships an evaluation subsystem for *your* application — golden datasets, 30+
metrics, calibrated judges, regression gates — and a three-tier internal benchmark
suite (VincioBench) that proves the library's own mechanisms. The **benchmark
platform** is the third thing: one coherent, pluggable harness that answers three
different questions as three **tracks**, under a single provenance-tier honesty
contract. Track 1, the **open evaluation plane**, runs the standard public model
benchmarks — MMLU, GPQA, GSM8K, HumanEval, IFEval, TruthfulQA, RULER, and more — and
is the oldest and largest of the three; Track 2 measures what routing a model
*through* Vincio changes; Track 3 pits a Vincio feature against the real library a
team would otherwise reach for.

It runs in your process over your own store. It is offline-reproducible exactly
where it claims to be, and it never becomes a hosted leaderboard.

## Three tracks, one question each

| Track | Question | Compares | Run |
|---|---|---|---|
| **1 · Model** | how good is a *model* on the standard public benchmarks? | a model vs the benchmark's verifiable gold | `vincio bench model` |
| **2 · Uplift** | how much does routing a model *through Vincio* change its scores? | the same model, Vincio-routed vs direct — per-benchmark delta | `vincio bench uplift` |
| **3 · Feature** | how good is a Vincio *feature* vs the same feature elsewhere? | a Vincio feature vs a real competitor library (+ a naive baseline) | `vincio bench feature` |

One command drives all three — `vincio bench <track>` — and every number, in every
track, carries the same provenance tier. The authoritative human map of the
platform, its tiers, and the internal gate is
[`benchmarks/PROVENANCE.md`](../../benchmarks/PROVENANCE.md); to run any of the
tracks, read [Run a benchmark suite](../guides/run-benchmark-suite.md).

## Provenance tiers — the honesty contract

Every number the platform prints carries a **provenance tier** that says,
structurally, how real it is. The tier is a property of the *execution*, not a
label a caller asserts, and the engine refuses to let a lower tier print a higher
tier's label — in every track, the tier means the same thing.

| Tier | What it is | Reproducible | Gates CI? |
|---|---|---|---|
| **S — Static / Mockup** | offline, reproducible — a fabricated fixture (Model), a recorded two-arm illustration (Uplift), or a competitor-absent / baseline comparison (Feature) | yes — byte-identical | yes |
| **R — Recorded** | a hash-pinned slice of the *real* thing, replayed against recorded outputs | yes — from the pin | yes |
| **L — Live** | the real thing ran end to end — a live model + provider (Model / Uplift), or the actual competitor library executed on this machine (Feature) | no — only from a real key or a real install | no — reported, never gated |

This makes the project's honesty culture structural: a Tier-S mechanism check can
never masquerade as a Tier-L score, and a request to print a tier the inputs do
not support raises a `TierViolationError`. And nothing is ever fabricated — a
missing model key or a missing competitor library is a *skip* or a refusal, never
an invented number.

```python
from vincio.evals.suite import resolve_tier, ProvenanceTier

# A fabricated fixture (ceiling S) asked for a Live label is refused.
resolve_tier(ProvenanceTier.LIVE, dataset_ceiling=ProvenanceTier.STATIC, solver_live=False)
# raises TierViolationError: a lower tier may not print a higher tier's label.
```

### How the tier is computed

The tier is not asserted — it is *derived* from the two things that actually
bound how real a run is, and the run may claim no higher:

```
dataset_ceiling   = STATIC   (fabricated fixture)
                  | RECORDED (hash-pinned real slice)
                  | LIVE     (full real dataset)
solver_ceiling    = LIVE if a live model ran else RECORDED   (a replay solver caps at R)

achievable = min(dataset_ceiling, solver_ceiling)     # the weaker input wins
```

`resolve_tier(requested, dataset_ceiling=…, solver_live=…)` returns
`achievable` when `requested` is `None`, honors any `requested` tier at or
below it (claiming a *more conservative* tier is always allowed), and raises
`TierViolationError` when `requested` exceeds it — with a reason that names the
weaker input (`the dataset is fabricated/static` vs `the solver replays
recorded outputs`). Because the ceiling is a `min` over both axes, a Live model
run over a fabricated dataset is still Tier-S, and a replay of a real dataset is
still Tier-R: you cannot borrow one axis's realness to launder the other. That
is the whole honesty contract in one function.

**Gotchas**

- A fully green Tier-S suite proves the *mechanism*, not the model — a mockup
  score *saturates by design*. Never quote a Tier-S number as a capability
  result; look for the `S` and read it as a gate, not a benchmark.
- Live is *reported, never gated*: it needs a real key (Model/Uplift) or a real
  install (Feature), so it is not byte-reproducible and cannot fail CI. CI bites
  on the Tier-S/R deterministic core underneath.
- A Track-3 contest silently drops from Live to Static when a competitor
  library is *not installed* — the head-to-head just didn't run. Check the
  `skipped` competitors before reading a Feature contest as a real comparison.

A **Tier-S / mockup** number *saturates by design* — it is a mechanism check, not a
real-world claim. Real-world evidence is the **Live** tier: Track 1 or 2 with a
model key, Track 3 with the competitor library actually installed.

## Track 1 — Model (the open evaluation plane)

`vincio bench model` scores a model on **29 standard public benchmarks across ten
niches**, each a `BenchmarkAdapter` that scores the benchmark's own verifiable
criterion — a normalized exact match, an extracted choice letter, a boxed-answer
equivalence, a per-test outcome, a verifiable instruction constraint — never a
model-judge proxy.

| Niche | Benchmarks | Primary metric |
|---|---|---|
| **Knowledge** | MMLU · GPQA · C-Eval · CMMLU · MMLU-Pro | choice accuracy |
| **Reasoning** | GSM8K · ARC · HellaSwag | exact-match / accuracy |
| **Math** | MATH | boxed-answer equivalence |
| **Coding** | HumanEval · MBPP · SWE-bench · LiveCodeBench · Spider · BIRD · DS-1000 | Pass@k / execution accuracy |
| **Instruction** | IFEval | verifiable-constraint rate |
| **Truthfulness** | TruthfulQA | accuracy |
| **Safety** | Prompt Injection | **contained vs compromised** |
| **RAG** | Faithfulness | grounded-claim rate |
| **Agent** | BFCL · τ-bench · GAIA · WebArena · AgentBench · ToolBench · InfiAgent-DABench · DABench | call-correctness / task-success |
| **Long Context** | RULER | needle recall at depth × length |

Two differentiators fall straight out of the reuse. The **Safety / Prompt
Injection** benchmark reports *contained vs compromised*, not merely attack-success
— Vincio's typed trust labels and dual-plane executor make containment
machine-checkable. And every **Long Context** benchmark is run twice, **with and
without the `ContextGovernor`**, so the long-context uplift is *measured*, never
assumed.

Under the hood this track is a composition of eight separated layers, each anchored
on a subsystem Vincio already ships — so it is part of the platform, not a parallel
stack beside it.

| Layer | Responsibility |
|---|---|
| **Core engine** | Deterministic, concurrent, resumable execution over a model or a Vincio-wrapped app — seeded sampling, checkpoint/resume, plugin dispatch (`BenchmarkSuite`). |
| **Benchmark registry** | A niche-grouped catalog of built-in, installable, and custom benchmarks, each declaring its dataset, metric, and provenance (`BenchmarkRegistry`, `register_benchmark`, the `vincio.benchmarks` entry-point group). |
| **Dataset layer** | Content-addressed Tier-S fixtures, optional Hugging Face fetch, and user datasets — hash-pinned so a silent task-set change is caught (`BenchmarkDataset`). |
| **Metrics engine** | Accuracy, F1, exact-match, Pass@k, and LLM-as-a-judge, reusing the existing metrics, judges, and ensembles. |
| **Reporting** | One run rendered to Markdown, HTML, JSON, CSV, and PDF, each citing the exact scored items (`SuiteReport`). |
| **Visualization** | Leaderboard, radar, heatmap, confusion-matrix, and trend charts (Vega-Lite by default; matplotlib PNG behind `vincio[eval-viz]`). |
| **Storage** | Run history and model-version comparison over SQLite (stdlib) or Postgres (`RunStore`). |
| **Plugin API** | Add a benchmark, metric, dataset, report format, or chart **without touching the core**, under the versioned plugin contract. |

The same track is also reachable as `app.benchmark_suite(...)` in-process and under
the older `vincio eval suite ...` commands, beside the application-eval commands.

## Track 2 — Uplift (the model, through Vincio vs direct)

Track 1 asks how good a model is; Track 2 asks what routing that model through
Vincio's infrastructure — grounding, rails, the context governor,
structured-output repair — *adds or removes*, benchmark by benchmark. Each
benchmark is scored **twice by the identical scorer**: once on the model's
**direct** answer and once on its **Vincio-routed** answer, and the per-benchmark
delta is the measured uplift (or regression), never assumed.

* **Mockup (offline, gates CI).** Each task carries two recorded outputs — a
  `recorded` direct answer and a `recorded_vincio` routed answer — both replayed
  through the real scorer, so the deterministic uplift is byte-identical.
* **Live.** The direct arm calls the model plainly; the Vincio arm calls the same
  model through a governed `ContextApp`. Same model, same scorer, two arms.

The four built-in benchmarks (`UpliftSuite`, `UpliftBenchmark`,
`register_uplift_benchmark`) cover the uplifts that hold for *any* model because
they are structural: grounding (`rag.grounded`), prompt-injection containment
(`safety.injection`), long-context needle recall via the governor
(`long_context.recall`), and structured-output validity (`output.schema_valid`).
On the offline mockup they move overall **direct 0.125 → Vincio 1.000 (+0.875)** —
grounding +0.5, injection containment +1.0 — a mechanism-level illustration that
saturates by design; the Live tier is where a real model's real delta is measured.

## Track 3 — Feature (a Vincio feature vs a real competitor)

Where Track 1 asks *how good is a model* and Track 2 asks *what does Vincio add to a
model*, Track 3 asks *how good is a Vincio feature — memory, retrieval, output
repair, context assembly, tabular encoding — against the real alternative a team
would otherwise reach for*. Every number is **measured on this machine from a live
run of both sides**: a `Contender` runs, is timed, and its quality is scored by the
contest's own deterministic metric.

A contest is **Live** only when every declared competitor actually executed; if a
competitor library is not installed it is reported **skipped** and the contest drops
to **Static** (Vincio and the baseline still ran, but the head-to-head did not). The
**deterministic quality metric — not wall-clock — is what gates CI**, so a re-run is
byte-identical on the part that matters; latency is recorded alongside as an
informational, machine-relative signal.

The eight built-in contests (`FeatureSuite`, `FeatureContest`, `Contender`,
`register_feature_contest`) span retrieval (`retrieval.bm25`, vs `rank_bm25`),
tokenization (`tokenization.count`, vs `tiktoken`), output repair
(`output.json_repair`, vs `json_repair`), prompt safety (`prompt.templating`, vs
`jinja2`), tabular encoding (`encoding.tabular`, vs `json.dumps` / `pandas`), context
assembly (`context.assembly`), layered memory (`memory.recall`, vs a naive keyword
store), and chunking (`chunking.split`). Representative measured results: retrieval
Vincio recall@1 **1.0 at ~10× the query speed** of `rank_bm25`; memory current-fact
precision **1.0** (it supersedes a stale fact) vs a naive store's **0.5**; tabular
encoding **~70% fewer tokens** than `json.dumps`; token counting **~2× faster** than
`tiktoken`, which is exact by definition.

## How it gates itself

Two mechanisms keep the platform honest. A VincioBench family, `eval_suite`, proves
the plane is honest the way `docs_conformance` proves the docs are connected:
**tier integrity** (a Tier-S fixture provably cannot emit a Tier-L label),
**metric correctness** (each aggregator matches a reference), **determinism** (a
Tier-S run is byte-identical), **registry completeness** (every catalog entry
resolves to a live adapter + dataset + metric + report), and **gate-bites** (a
mislabeled tier and a wrong metric are each caught), with companion published SLOs.

Separately, `benchmarks/vinciobench.py` is the internal mechanism / regression gate
— **not one of the three tracks** — and it CI-gates the deterministic core of all
three via the `families.bench_tracks.*` budgets. A missing key or a missing
competitor never breaks the gate: it degrades to a Tier-S mockup, which is exactly
what gates CI.

To run any track, read [Run a benchmark suite](../guides/run-benchmark-suite.md);
for the full tour of Track 1 see
[the example](../../examples/16_open_evaluation_plane.py).

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: run a benchmark suite](../guides/run-benchmark-suite.md)
- [Example: 16_open_evaluation_plane.py](../../examples/16_open_evaluation_plane.py)
- [Concept: Observability](observability.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#optimization)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
