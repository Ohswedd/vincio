# Benchmark provenance — the three-track platform, and how real each number is

This is the map. Vincio's benchmark platform answers **three questions**, each as
its own **track**, and each supporting a **live** run and an offline **mockup**.
Every number carries a **provenance tier** so you never have to guess whether a
figure is `LIVE`, `STATIC/FABRICATED`, or a self-measurement — and a lower tier can
never print a higher tier's label.

The machine-readable source of truth is [`manifest.json`](manifest.json)
(regenerate with `python benchmarks/_manifest.py`; gated by
`tests/test_benchmark_manifest.py`). One command drives all three tracks:
`vincio bench <track>` (or `python benchmarks/bench.py <track>`).

## The three tracks

| Track | Question | Compares | Run |
|---|---|---|---|
| **1 · Model** | how good is a *model* on the standard public benchmarks? | a model vs the benchmark's verifiable gold | `vincio bench model` |
| **2 · Uplift** | how much does routing a model *through Vincio* change its scores? | the same model, Vincio-routed vs direct — per-benchmark delta | `vincio bench uplift` |
| **3 · Feature** | how good is a Vincio *feature* vs the same feature elsewhere? | a Vincio feature vs a real competitor library (+ a naive baseline) | `vincio bench feature` |

## The one contract: provenance tiers

| Tier | Name | Live? | Reproducible? | Gates CI? | What it means |
|:--:|---|:--:|:--:|:--:|---|
| **L** | Live | ✅ | ❌ | ❌ reported | the real thing ran end to end — a live model + provider (model / uplift), or the actual competitor library executed on this machine (feature) |
| **R** | Recorded | ➖ replay | ✅ from the pin | ✅ | a hash-pinned slice of the real thing, replayed against recorded outputs |
| **S** | Static / Mockup | ❌ | ✅ byte-identical | ✅ | offline & reproducible — a fabricated fixture (model), a recorded two-arm illustration (uplift), or a competitor-absent/baseline comparison (feature) |

> A **model** Tier-S score *saturates by design* (a mechanism check, not a
> real-world claim). A **feature** contest is **Live** only when a real competitor
> actually executed; a missing competitor is reported *skipped*, never fabricated.

## Track 1 — Model

`vincio bench model` scores a model on **29 standard public benchmarks across 10
niches** — MMLU, GPQA, GSM8K, MATH, HumanEval, MBPP, IFEval, TruthfulQA, ARC,
HellaSwag, SWE-bench, LiveCodeBench, Spider, BIRD, τ-bench, GAIA, RULER, and more.
Tier-S ships fabricated fixtures that gate CI; **Live** runs a real model over a
real dataset:

```bash
vincio bench model all --tier static                    # offline (Tier-S)
python benchmarks/eval_live.py --provider anthropic --model claude-opus-4-8 \
    --benchmarks knowledge.mmlu reasoning.gsm8k --tier live --dataset-dir ./datasets
```

Add a custom benchmark with `register_benchmark(BenchmarkSpec(...))`.

## Track 2 — Uplift

`vincio bench uplift` runs each benchmark **twice by the identical scorer** — the
model's **direct** answer vs its **Vincio-routed** answer — and reports the
per-benchmark delta (grounding, prompt-injection containment, long-context needle
recall via the governor, structured-output validity, web freshness, and universal
reasoning). Tier-S replays two recorded
arms (deterministic, gates CI); **Live** calls a real model plainly for the direct
arm and through a governed app for the Vincio arm. Add a custom uplift with
`register_uplift_benchmark(UpliftBenchmark(...))`.

The dynamic reasoning driver is `python benchmarks/reasoning_uplift_live.py`.
It compares native and non-native OpenRouter models on math, logic,
contradiction, live-source verification, overclaiming, tokens and cost — plus,
since 7.11, a plan-shaped decision case (does the internal plan mode activate
and produce a verified exact answer?) and a cite-a-source honesty case (are
fabricated source attributions ever delivered?); it is Tier-L and never runs
in CI.
`python benchmarks/reasoning_multilingual_live.py` is the companion Tier-L
router audit: Spanish, Japanese, Arabic, Swahili and Chinese prompts check that
language, depth, task, web/no-web intent and semantic-route accounting follow
the configured model rather than a finite local language list.
The dated 2026-07-08 snapshot records the small-sample outcomes rather than
generalizing them: GPT-4.1 mini moved from 4/6 exact direct answers to 5/6 via
Vincio with the internal plan mode active on both plan-shaped cases; Llama 3.2
3B moved from 0/6 to 3/6 (two verified, one bounded correction; its upstream
dropped one arithmetic call per arm, counted wrong for both). No fabricated
sources or overclaims were delivered; unverifiable answers were withheld.
GPT-4.1 mini routed 5/5 multilingual cases on 2026-07-07. The machine-readable
source is `reference/live_snapshot.json`; the published art is
`assets/benchmark-reasoning.svg` and `assets/big-brain.svg`.

## Track 3 — Feature

`vincio bench feature` runs a Vincio feature head-to-head against the real
competitor library — **measured live on this machine** — across retrieval (BM25 vs
`rank_bm25`), tokenization (vs `tiktoken`), output repair (vs `json_repair`), prompt
safety (vs `jinja2`), tabular encoding (vs `json.dumps`/`pandas`), context assembly,
layered memory (vs a naive keyword store), and chunking. A competitor that is not
installed is reported *skipped*. The deterministic quality metric — not wall-clock —
gates CI. Add a custom contest with `register_feature_contest(FeatureContest(...))`.

## The internal gate: VincioBench

`benchmarks/vinciobench.py` is **not one of the three tracks** — it is the internal
mechanism / regression gate. It proves each Vincio mechanism still works against a
naive baseline (offline, saturating), and **CI-gates the deterministic core of all
three tracks** via the `families.bench_tracks.*` budgets:

```bash
python benchmarks/vinciobench.py && python benchmarks/check_budgets.py
```

## The extended drivers

Two richer offline drivers predate the library platform and remain as each track's
*extended* driver: [`competitive.py`](competitive.py) (Track 3 — a few extra
micro-benchmarks) and [`quality_uplift.py`](quality_uplift.py) (Track 2 — the dated
real-model grounding harness with cost-per-correct-answer). New benchmarks belong in
the library, not in these scripts.

## The honesty rules

1. **Tier-S / mockup numbers are mechanism checks, not performance claims** — they
   saturate by design. Real-world evidence is the Live tier (Track 1/2 with a key,
   Track 3 with a real competitor installed).
2. **Live numbers are dated, environment-specific, and not reproducible offline.**
   Competitive latency ratios vary by machine; rerun on your hardware.
3. **Nothing is fabricated.** A missing model key or competitor library yields a
   *skip* or a refusal, never an invented number.
