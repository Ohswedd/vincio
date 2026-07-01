# The open evaluation plane

Vincio ships an evaluation subsystem for *your* application — golden datasets, 30+
metrics, calibrated judges, regression gates — and a three-tier internal benchmark
suite (VincioBench) that proves the library's own mechanisms. The **open
evaluation plane** is the third thing: one coherent, pluggable harness for running
the **standard public model benchmarks** — MMLU, GPQA, GSM8K, HumanEval, IFEval,
TruthfulQA, RULER, and more — grouped by niche, scored by reusable metrics, and
reported the same way for every model and every model *version*.

It runs in your process over your own store. It is offline-reproducible exactly
where it claims to be, and it never becomes a hosted leaderboard.

## Provenance tiers — the honesty contract

Every number the plane prints carries a **provenance tier** that says, structurally,
how real it is. The tier is a property of the *execution*, not a label a caller
asserts, and the engine refuses to let a lower tier print a higher tier's label.

| Tier | What it is | Reproducible | Gates CI? |
|---|---|---|---|
| **S — Static** | a small, bundled, *fabricated* fixture that exercises the adapter + metric end to end | yes — byte-identical | yes |
| **R — Recorded** | a hash-pinned slice of the *real* public dataset replayed against recorded model outputs | yes — from the pin | yes |
| **L — Live** | the full public dataset run against a live model and provider | no — only from a real key | no — reported, never gated |

This makes the project's honesty culture structural: a Tier-S mechanism check can
never masquerade as a Tier-L score, and a request to print a tier the inputs do
not support raises a `TierViolationError`.

```python
from vincio.evals.suite import resolve_tier, ProvenanceTier

# A fabricated fixture (ceiling S) asked for a Live label is refused.
resolve_tier(ProvenanceTier.LIVE, dataset_ceiling=ProvenanceTier.STATIC, solver_live=False)
# raises TierViolationError: a lower tier may not print a higher tier's label.
```

## Eight separated layers

The plane is eight layers, each anchored on a subsystem Vincio already ships, so it
is a composition of the platform — not a parallel stack beside it.

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

## The catalog — eleven niches, one contract

Every benchmark is a `BenchmarkAdapter` that scores the benchmark's own verifiable
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
| **Custom** | user-defined | user-supplied |

Two differentiators fall straight out of the reuse. The **Safety / Prompt
Injection** benchmark reports *contained vs compromised*, not merely
attack-success — Vincio's typed trust labels and dual-plane executor make
containment machine-checkable. And every **Long Context** benchmark is run twice,
**with and without the `ContextGovernor`**, so the long-context uplift is
*measured*, never assumed.

## How it gates itself

A VincioBench family `eval_suite` proves the plane is honest the way
`docs_conformance` proves the docs are connected: **tier integrity** (a Tier-S
fixture provably cannot emit a Tier-L label), **metric correctness** (each
aggregator matches a reference), **determinism** (a Tier-S run is byte-identical),
**registry completeness** (every catalog entry resolves to a live adapter +
dataset + metric + report), and **gate-bites** (a mislabeled tier and a wrong
metric are each caught), with companion published SLOs.

To run a benchmark, read [Run a benchmark suite](../guides/run-benchmark-suite.md);
for the full tour see [the example](../../examples/16_open_evaluation_plane.py).

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
