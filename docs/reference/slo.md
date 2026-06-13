# Reference: performance & quality SLOs

These are Vincio's published Service Level Objectives — the performance and
quality guarantees the engine is held to. They are not marketing numbers: each
SLO names a VincioBench metric and the CI **budget** that gates it, and the
budget is held *at least as strict* as the published target. A green build
therefore proves the SLO holds, with headroom. `tests/test_slos.py` enforces
that invariant, and the source of truth is
[`benchmarks/slos.json`](https://github.com/Ohswedd/vincio/blob/main/benchmarks/slos.json).

All numbers are measured on the deterministic offline suite (mock provider,
in-repo corpora). Reproduce them yourself — there is no hosted leaderboard:

```bash
python benchmarks/vinciobench.py     # produce results/vinciobench_latest.json
python benchmarks/check_budgets.py   # gate it (exit 1 on any breach)
```

See [benchmarks/METHODOLOGY.md](https://github.com/Ohswedd/vincio/blob/main/benchmarks/METHODOLOGY.md)
for how the suite works and the [performance guide](../guides/performance.md)
for tuning.

## Performance

| SLO | Target | VincioBench metric |
|---|---|---|
| Cold context compilation (p95) | ≤ 300 ms | `perf.context_compile.cold_p95_ms` |
| Compile cache speedup | ≥ 1.5× | `perf.context_compile.cache_speedup` |
| Retrieval latency (p95) | ≤ 150 ms | `perf.retrieval.p95_ms` |
| Cached end-to-end run (p50) | ≤ 300 ms | `perf.run.p50_ms` |
| Concurrent throughput | ≥ 50 runs/s | `perf.run.concurrent_runs_per_s` |
| Streaming TTFT | first token before done | `perf.streaming.ttft_before_done` |
| Tool runtime overhead (p50) | ≤ 50 ms | `tool.p50_overhead_ms` |

Absolute latencies are machine-relative (the suite runs on shared CI runners);
treat them as order-of-magnitude regression gates. Ratios and throughput are
portable.

## Cost & quality

| SLO | Target | VincioBench metric |
|---|---|---|
| Evidence-token reduction vs naive stuffing | ≥ 20% | `cost.token_reduction` |
| Hybrid retrieval recall@3 | ≥ 0.80 | `rag.recall_at_3.mean` |
| Self-correction recovery | 100% within cycle bound | `reliability.self_correction.recovery_rate` |

## Security

| SLO | Target | VincioBench metric |
|---|---|---|
| Prompt-injection detection rate | ≥ 0.80 | `security.injection_detection_rate` |
| Injection false-positive rate | ≤ 0.20 | `security.injection_false_positive_rate` |
| PII coverage | ≥ 0.80 | `security.pii_coverage` |

Quality and security floors describe behavior on the reference corpora; measure
on your own data with the same harness before depending on a number.
