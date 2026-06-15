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

## Protocols & interoperability

| SLO | Target | VincioBench metric |
|---|---|---|
| MCP tool schema fidelity | exact (1.0) | `protocols.mcp.schema_fidelity` |
| A2A budget-bounded delegation terminates | always | `protocols.a2a.terminates` |
| Skill progressive-disclosure token savings | ≥ 0.50 | `protocols.skills.disclosure_savings` |

A consumed MCP tool's input schema is preserved exactly so validation and
constrained decoding bind to the server's contract; a crew delegated over A2A
inherits the same termination guarantee as in-process; and an unused skill body
stays out of the budget.

## Continuous quality (agentic evaluation)

| SLO | Target | VincioBench metric |
|---|---|---|
| Trajectory-metric agreement with labeled traces | ≥ 0.90 | `agentic_evals.trajectory_agreement` |
| Trajectory eval flags runs output-only eval passes | always | `agentic_evals.trajectory_catches_more` |
| Simulator determinism (same seed → same conversation) | exact | `agentic_evals.simulator_determinism` |
| Drift detection sensitivity (real regressions caught) | ≥ 0.85 | `agentic_evals.drift_sensitivity` |
| Drift detection specificity (stable windows not alarmed) | ≥ 0.90 | `agentic_evals.drift_specificity` |
| Judge–human Cohen's κ before a judge earns gating weight | ≥ 0.75 | `agentic_evals.cohen_kappa_tracked` |

Trajectory scores must track ground truth or trajectory-gated releases would be
meaningless; a wrong path with a right-looking answer must be catchable;
simulated multi-turn cases must replay identically to serve as CI goldens; drift
must catch real regressions without crying wolf; and an LLM judge only earns CI
gating weight once it has demonstrably agreed with people.

## Cost & reliability (scale)

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| Every batched request is reconciled by custom id; results are never silently dropped. | true | `families.scale.batch.reconciled_ok` |
| A tripped circuit recovers through a half-open probe once the provider is healthy again. | true | `families.scale.circuit.half_open_recovers` |
| Provider-aware prompt caching achieves at least a 50% input-token hit rate on a warm, stable prefix. | ≥ 0.50 | `families.scale.cache.hit_rate` |
| Cost rolled up by tenant/feature equals the sum of the attributed per-call costs. | ≥ 0.99 | `families.scale.attribution.accuracy` |

Latency-tolerant batch work must return a result for every request — losing one
corrupts evals and bulk extraction; a breaker that opens but never closes turns
a transient outage into a permanent one; stable system/tool/context prefixes are
the bulk of input tokens, so caching them is the single biggest cost lever; and
FinOps decisions and per-tenant budgets are only trustworthy if attribution is
exact, not estimated.

Quality and security floors describe behavior on the reference corpora; measure
on your own data with the same harness before depending on a number.
