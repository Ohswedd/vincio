# VincioBench

Benchmark suite for Vincio and baseline comparisons. Runs fully offline by
default (deterministic mock provider + deterministic metrics) so results
are reproducible; set `VINCIO_PROVIDER`/`VINCIO_MODEL` to benchmark a real
model.

```bash
python benchmarks/vinciobench.py             # all families
python benchmarks/vinciobench.py rag cost    # selected families
```

Results are printed as JSON and saved to `benchmarks/results/`.

## Families

| Family | Measures | Baseline |
|---|---|---|
| **PromptBench** | cacheability/tokens per render format, lint defect detection | naive string concatenation |
| **RAGBench** | recall@3 / MRR per retrieval mode (bm25, dense, sparse, late_interaction, PLAID-compressed, hybrid, hybrid_full, + query understanding); GraphRAG community building | single-index BM25 |
| **MemoryBench** | preference recall, contradiction supersede, cross-user isolation | — |
| **AgentBench** | budget adherence under an adversarial looping model, DAG success; crew over-budget termination, full-crew success, delegation recording; durable-graph interrupt→resume and fork-replay determinism; composition streaming coverage | unbounded loop |
| **ToolBench** | reliability, runtime overhead (p50 ms), invalid-arg rejection, cache hits | — |
| **OutputBench** | recovery rate over malformed model outputs; missing-required correctly rejected | raw `json.loads` |
| **ReliabilityBench (0.7)** | strict-schema closure for constrained decoding; mid-stream invalid detection + abort savings; self-correction recovery within cycle bounds; rail catch rate + false positives; signature prediction validity + optimizer variants; schema routing/classification accuracy | validate-at-end / unguarded output |
| **CostBench** | evidence-token reduction from the context compiler | stuff-everything context |
| **SecurityBench** | injection detection rate, false-positive rate, PII coverage | — |
| **EvalBench** | metric agreement on labeled examples; red-team judging (guarded vs naive target, detector coverage); synthetic-data determinism and coverage; A/B significance machinery; session grouping; HTML viewer self-containment; trace→dataset; G-Eval calibration | naive target (85% attack success) |
| **LoopBench (0.8)** | the closed loop end to end: promotion fires, decisions are deterministic, gates block regressions, the registry version is tagged + eval-linked; grounded auto-memory precision (grounded written, ungrounded excluded); retrieval-feedback improvement + gating; Pareto frontier correctness (dominated excluded, knee balanced); learned-budget promotion; guided-search budget bounds | ungated / single-score optimization |
| **PerfBench** | compile/retrieval/run latency (p50/p95), cache speedups, concurrent throughput, streaming TTFT | cold (uncached) paths |

## Benchmark gates (CI)

`budgets.json` defines hard budgets over the report (latency ceilings, cache
speedup floors, token-efficiency and quality floors). CI runs the suite and
gates the build:

```bash
python benchmarks/vinciobench.py      # produce the report
python benchmarks/check_budgets.py    # exit 1 on any budget breach
```

Latency budgets are deliberately loose (shared CI runners) and exist to catch
order-of-magnitude regressions; ratio and quality budgets are tight.

## Per-stage profiling

```bash
python benchmarks/profile_stages.py                  # stage breakdown from trace spans
python benchmarks/profile_stages.py --runs 50 --json
python benchmarks/profile_stages.py --cprofile vincio.prof
# flamegraph: snakeviz vincio.prof   (or: flameprof vincio.prof > flame.svg)
```

## A note on claims

Vincio's design targets are stated as improvement *hypotheses* (e.g. a 20–40%
token reduction through context compression). VincioBench measures them — the
report states whether each hypothesis was met on the benchmark corpus. Do not
market numbers beyond what a benchmark run on your own data shows.
