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
| Cold context compilation (p99) | ≤ 500 ms | `perf.context_compile.cold_p99_ms` |
| Warm compile hot path (p50, cache hit) | ≤ 10 ms (sub-ms on the reference corpus) | `perf.context_compile.cached_p50_ms` |
| Compile cache speedup | ≥ 1.5× | `perf.context_compile.cache_speedup` |
| Vectorized scoring equivalence | batched == per-candidate loop | `perf.vectorized_scoring.equivalent` |
| Render-program byte-identity | identical to from-scratch compile | `perf.render_program.byte_identical` |
| Warm candidate arena equivalence | warm reuse == cold compile | `perf.warm_arena.equivalent` |
| Streaming-first compilation | prefix before scoring | `perf.streaming_compile.prefix_before_scoring` |
| Speculative prefetch | warms the retrieval embed | `perf.prefetch.warms_cache` |
| Memory-footprint budget enforced | slim + evict to fit the ceiling | `perf.footprint.budget_enforced` |
| Resident footprint (reference corpus) | ≤ 6 KB | `perf.footprint.packet_bytes` |
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
| Matryoshka full output dimension recall@3 | ≥ 0.80 | `rag.mrl.full_recall_at_3.mean` |
| Matryoshka truncated dimension recall@3 (one-eighth of base) | ≥ 0.80 | `rag.mrl.recalls_by_dimension.64.recall_at_3.mean` |
| Unified text+image retrieval recall@3 | ≥ 0.80 | `rag.multimodal.recall_at_3.mean` |
| Self-correction recovery | 100% within cycle bound | `reliability.self_correction.recovery_rate` |

## Security

| SLO | Target | VincioBench metric |
|---|---|---|
| Prompt-injection detection rate | ≥ 0.80 | `security.injection_detection_rate` |
| Injection false-positive rate | ≤ 0.20 | `security.injection_false_positive_rate` |
| PII coverage | ≥ 0.80 | `security.pii_coverage` |
| Injection-containment escalation rate (adversarial corpus) | 0 | `containment.escalation_rate` |

## Protocols & interoperability

| SLO | Target | VincioBench metric |
|---|---|---|
| MCP tool schema fidelity | exact | `protocols.mcp.schema_fidelity` |
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

## Orchestrator & planner depth

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| On a tool failure, the agent repairs the plan in place (re-binds) and still finishes. | true | `families.agent.planner_depth.repair_rebind` |
| Under a budget shock, the agent drops the optional tail and finalizes inside the budget. | true | `families.agent.planner_depth.repair_budget_shock` |
| Cost-aware action selection cuts model spend vs always-strong. | ≥ 25% | `families.agent.planner_depth.cost_aware_savings` |
| Independent sub-graphs scheduled across workers reach a logical speedup over serial. | ≥ 1.5× | `families.scale.subgraph.speedup` |
| A graph paused on a durable timer survives a restart and resumes when due. | true | `families.agent.planner_depth.durable_timer_restart_safe` |

A failing dependency must not abort a run that can still succeed, and a budget is
a hard cap the planner converges toward rather than blows; reaching for the
strongest model on every step overpays, so the cheapest capable model earns the
easy steps; independent work should run concurrently, not serially; and a timer
whose wake condition did not survive a restart would silently never fire.

## Test-time compute & reasoning

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| At a fixed candidate budget, verifier-guided best-of-N beats the single-shot draw — a quality-per-dollar Pareto improvement. | ≥ +0.1 quality | `families.test_time_compute.pareto_quality_gain` |
| The best-of-N path returns quality per cent of inference spend above a floor. | ≥ 20 points/¢ | `families.test_time_compute.quality_per_cost_point` |
| A reasoning step's thinking budget never exceeds the controller's hard token ceiling at any difficulty. | ≤ 8192 tokens | `families.test_time_compute.max_thinking_budget` |

Test-time compute is only worth its spend if it lifts quality at the same budget;
early-exit returns the saved draws the moment the verifier clears the bar, and the
reasoning controller scales effort with difficulty but holds a hard token ceiling
so a hard task can never silently exhaust the run.

## Long-horizon context engineering

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| At 10× horizon, the governed resident context footprint stays within a bounded multiple of the 1× footprint. | ≤ 2× | `families.long_horizon.footprint_growth_ratio` |
| A fact compacted out of the live packet on a long run is still recalled at 10× horizon by paging it back from the content-addressed store. | ≥ 0.80 recall | `families.long_horizon.recall_at_horizon` |
| A governed long run stays inside its declared context budget (tokens / residency / KV-cache) at 10× horizon. | within budget | `families.long_horizon.within_budget_at_horizon` |

Naïve accumulation grows the context footprint ~linearly with the horizon (≈10×)
and lets stale spans rot quality; the `ContextGovernor` keeps it flat via intra-run
decay and provenance-preserving compaction, paging cold detail back on demand so
recall survives. The budgets gate 1.5× growth and full recall, below the published
promises.

## World-model / simulation-based planning

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| On a planning-favoring world, the imagined-rollout planner matches or beats a reactive (one-step) planner at a fixed action budget. | matches or beats | `families.world_model.planning_advantage` |
| A calibrated world model's predicted next states track the real environment, earning it planning weight. | ≥ 0.90 accuracy | `families.world_model.model_state_accuracy` |
| An uncalibrated world model is refused for planning. | true | `families.world_model.calibration_gate_enforced` |

Reacting to the live world one step at a time is trapped by a locally-attractive
shortcut that dead-ends; a planner that rolls a learned `WorldModel` forward sees
the dead end in imagination and commits the patient path instead — here it opens
the vault while the reactive planner is stuck. The model only earns planning
weight once its predictions are calibrated against the real environment, the way a
judge ensemble must earn its gating weight; the budget gates a strict planning win
and full prediction accuracy, above the published promises.

## Causal record-replay debugger

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| A recorded run replays byte-identically — the recording, not the live provider, serves the answer. | byte-identical | `families.record_replay.replay_faithful` |
| When the code under replay changes, the divergence is detected and reported. | true | `families.record_replay.divergence_detected` |
| A recording round-trips through a content-addressed store and verifies before replay. | true | `families.record_replay.fidelity_verified` |

A run is deterministic except at its edges — every place it reads the outside
world. The recorder captures those edges (model responses, tool outputs,
retrieval hits, the negotiated capabilities, clock/seed) keyed to the trace; the
replayer serves them back, so replay reproduces the run byte-for-byte against a
live provider that would answer differently. Because each edge is keyed by the
same identity the live code computes, a changed edge is a cache miss — and a miss
is a divergence, reported with the edge that drifted rather than silently
re-executed. Recordings are content-addressed and carry a fidelity digest, so a
recording shared across processes is verified before it is trusted for replay.

Quality and security floors describe behavior on the reference corpora; measure
on your own data with the same harness before depending on a number.
