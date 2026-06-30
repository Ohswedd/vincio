# Reference: performance & quality SLOs

These are Vincio's published Service Level Objectives, the performance and
quality guarantees the engine is held to. They are not marketing numbers: each
SLO names a VincioBench metric and the CI **budget** that gates it, and the
budget is held *at least as strict* as the published target. A green build
therefore proves the SLO holds, with headroom. `tests/test_slos.py` enforces
that invariant, and the source of truth is
[`benchmarks/slos.json`](https://github.com/Ohswedd/vincio/blob/main/benchmarks/slos.json).

All numbers are measured on the deterministic offline suite (mock provider,
in-repo corpora). Reproduce them yourself, there is no hosted leaderboard:

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
| Single-pass selection byte-identity | feature arena selects identical context | `perf.single_pass.selection_byte_identical` |
| Single-pass selection equivalence (large pool) | arena == per-pass baseline at scale | `perf.vectorized_selection.equivalent` |
| Single-pass compile speedup | ≥ 1.05× on a large pool | `perf.single_pass.compile_speedup` |
| Bounded BM25 top-k identity | nlargest == full-sort prefix | `perf.retrieval.topk_identical` |
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
| Compact table encoding vs `json.dumps` | ≥ 40% fewer tokens | `cost.table_encoding.reduction_vs_json` |
| Compact table encoding round-trips losslessly | true | `cost.table_encoding.lossless` |

## Model pricing & capability registry (5.1)

The data-driven `ModelRegistry` is the single source of truth the cost
`PriceTable`, the capability guard, the cost/latency router, the model cascades,
and the energy/carbon accounting all read from. These SLOs hold the shipped
`model_catalog.json` complete, honest, fresh, and routing-stable — proven offline
by `registry.coverage_report()` (run `vincio registry coverage`). Freshness is
evaluated against the catalog's **release date**, never the wall clock, so a
frozen release reports the same verdict forever.

| SLO | Target | VincioBench metric |
|---|---|---|
| Every provider default + capability family + openai_compat preset resolves to a non-sparse, priced profile | true | `registry_coverage.coverage_complete` |
| No price has drifted past the freshness horizon (vs release date) | true | `registry_coverage.no_stale_prices` |
| No GA billable model of a paid provider silently bills $0 | true | `registry_coverage.no_silent_zero` |

## Data & analytics plane

| SLO | Target | VincioBench metric |
|---|---|---|
| A table far larger than the window fits a fixed token budget (profile + representative sample), size invariant to row count | true | `data_plane.fit_in_window.within_budget` |
| The bounded-memory profile faithfully recovers a large table's extrema, count, cardinality, and central tendency | true | `data_plane.profile.faithful` |
| Data-quality rails catch every seeded defect class (type, range, allowed-set, anomaly, PII) deterministically | true | `data_plane.quality.detected_all` |
| Governed text-to-query reaches ≥ 0.9 execution accuracy on the Spider/BIRD-shaped battery (generated query's result set equals the gold's) | ≥ 0.9 | `data_plane.text_to_query.execution_accuracy` |
| Every generated write, DDL, stacked statement, or injection attempt is structurally refused before a query runs | true | `data_plane.text_to_query.read_only_enforced` |
| An analytical answer and its cited source cells re-derive from the bytes; a tampered source is caught | true | `data_plane.text_to_query.provenance_verifiable` |
| The data-analysis agent reaches the correct answer within its step budget on the DS-1000 / InfiAgent-DABench / DABench-shaped batteries | true | `data_plane.analysis.success_at_budget` |
| Every cell-traceable finding in a generated analytical narrative cites the exact source cells it rests on | true | `data_plane.analysis.narrative_cited` |
| An analytical narrative and its cited cells re-derive from the bytes; a tampered source or narrative is caught | true | `data_plane.analysis.verifiable` |
| The notebook-native reprs surface an artifact's real, verifiable facts — its content hash, exact cell citations, and audit id — and never a fabricated one; a tampered stage flips the repr's integrity verdict | true | `data_plane.notebook.repr_faithful` |
| An interactive register → query → analyze → chart → cite session seals into one signed, audited narrative whose verify() re-derives every inline finding from the bytes; a tampered source flips the verdict | true | `data_plane.notebook.session_verifies` |
| A generated chart re-derives from the rows it was built from; a tampered source is caught | true | `data_plane.charts.data_bound` |
| A generated chart cites the exact source cells it was built from, aggregates included | true | `data_plane.charts.figure_cited` |
| A generated chart carries a C2PA credential bound to its rendered bytes; an edited byte stream is caught | true | `data_plane.charts.content_bound` |
| Out-of-core processing sustains ≥ 20,000 rows/s (and ≥ 1,000,000 tokens/s through the streaming encoder) | ≥ 20,000 | `data_plane.streaming.throughput_rows_per_s` |
| The resident working set of a streaming group-by stays bounded as the dataset grows 100× (it tracks groups, not rows) | true | `data_plane.streaming.memory_bounded` |
| The context compiler's streaming candidate pre-filter bounds a 10k+ evidence pool before full scoring while keeping the relevant evidence | true | `data_plane.streaming.prefilter_bounds_pool` |
| Windowed analytics over an unbounded event stream are exact: the per-window group-by equals the brute-force ground truth of bucketing every event by its window | true | `data_plane.realtime.windowed_correct` |
| The resident working set of windowed analytics is invariant to the event volume (a 100× longer stream stays within a small factor — only the open window is held) | true | `data_plane.realtime.memory_bounded` |
| Every windowed result re-derives offline against its captured window, every cited event offset is in-window, and a tampered captured event is caught | true | `data_plane.realtime.provenance_sound` |
| A governed metric defined once compiles to one canonical read-only SELECT and returns the same number however the question is phrased | true | `data_plane.semantic_layer.governed_one_way` |
| A governed metric's result re-derives from the hashed source; an ad-hoc query passed off as the governed metric, or a tampered source, is rejected | true | `data_plane.semantic_layer.metric_verifiable` |
| A metric's column-level lineage resolves to its base columns and source, and a right-to-erasure sweep removes the dataset it rests on | true | `data_plane.semantic_layer.lineage_reaches_dataset` |
| A cross-org federated query moves no raw rows across a trust boundary: a per-row sentinel appears in neither the saga journal nor the sealed narrative — only group-by aggregates cross | true | `data_plane.federated_analytics.rows_never_cross` |
| Every finding in a federated engagement re-derives from each org's content-hashed source, the reconciled totals equal the brute-force totals over the pooled rows, and a tampered reconciliation is caught | true | `data_plane.federated_analytics.federated_data_binding` |
| Residency egress refusal, the consent ledger's analytics purpose, the differential-privacy budget, and the k-anonymity contributor floor each refuse a non-compliant federated round | true | `data_plane.federated_analytics.governance_preservation` |
| The plane composes end-to-end as one system: a single `DataEngagement` threads register → profile → … → cite into one content-bound, signed `DataNarrative` that verifies offline from the bytes alone, with one continuous hash-chained audit narrative | true | `families.data_analysis_conformance.conformance_verifies_offline` |
| Every analytical finding a composed engagement carries is data-bound: each captured query, analysis, chart, and metric re-executes against the content-hashed source and re-derives from the bytes | true | `families.data_analysis_conformance.conformance_artifacts_verify` |
| A tamper introduced anywhere in a composed engagement is caught from the bytes alone: a re-ordered stage breaks the hash chain, an edited digest or underlying artifact fails the digest check, a tampered source breaks data-binding, and a forged signature fails authentication | true | `families.data_analysis_conformance.conformance_tamper_caught` |

The fit-in-window guarantee is the headline of the profiling/sampling rung: a
full-fidelity column profile (computed over every row in bounded memory) plus a
representative sample sized to the remaining budget represent a table of any
height inside the same window. The budget gates that a 100k- and a 500k-row table
both fit, and that their representations stay within 100 tokens of each other.

The text-to-query SLOs gate the analyst rung three ways: execution accuracy holds
on a Spider/BIRD-shaped battery (the budget gates 0.95, stricter than the published
0.9), read-only enforcement is total (an entire battery of write / DDL / stacked /
injection attempts is refused, deterministically), and cell-level provenance is
offline-verifiable (a result and its cited cells re-derive from the bytes, and a
tampered source flips `verify()` to false).

The data-analysis-agent SLOs gate the multi-step EDA rung three ways: task success
at budget (the offline governed agent answers every DS-1000 / InfiAgent-DABench /
DABench-shaped battery task correctly within its step budget — correctness and
bounded exploration together), narrative citation completeness (every cell-traceable
finding carries the exact source cells it rests on, so a claim is never asserted
without its lineage), and offline verifiability (an analysis re-executes against its
hashed source and a tampered source or narrative flips `verify()` to false).

The data-analysis conformance SLOs are the plane's capstone: each rung above is
grounded, cited, and verifiable on its own, and these gate that they **compose**.
A single `DataEngagement` threads the whole plane (register → profile → … → cite)
into one signed, hash-chained `DataNarrative` — the budget gates that it verifies
offline from the bytes alone, that every captured finding is data-bound (re-executes
against the content-hashed source and re-derives), and that a tamper introduced
anywhere (a re-ordered stage, an edited digest, a tampered source, a forged
signature) is caught — so the plane is proven one verifiable system, the analytics
analogue of the cross-org conformance capstone.

## Security

| SLO | Target | VincioBench metric |
|---|---|---|
| Prompt-injection detection rate | ≥ 0.80 | `security.injection_detection_rate` |
| Injection false-positive rate | ≤ 0.20 | `security.injection_false_positive_rate` |
| PII coverage | ≥ 0.80 | `security.pii_coverage` |
| Injection-containment escalation rate (adversarial corpus) | 0 | `containment.escalation_rate` |

## Verified reasoning & statistical certificates

| SLO | Target | VincioBench metric |
|---|---|---|
| An answer carries a content-bound `Certificate` a deterministic kernel set confirms — including the statistical kernels that recompute a stated trend / correlation / interval / forecast from the cited cells — `verified` only when a claim recomputed and held, and a tampered verdict is caught from the bytes. | true | `families.verified_reasoning.certificate_soundness` |
| A correlation stated as causation with no controls or randomized design is refused, and a controlled claim whose association collapses once the declared confounder is partialled out is refuted, while a genuine controlled association and a randomized-design claim are verified. | true | `families.verified_reasoning.refutes_spurious_causation` |
| A behaviour shield wired into the tool runtime blocks a policy-violating action (an unapproved write) before it executes, while letting an approved one through. | true | `families.verified_reasoning.shield_prevents_violation` |

Correctness can be *certified*, not merely judged: the kernels recompute a claim
and refuse to emit a refuted one, the statistical kernels bind a statistic to its
cited cells and refute correlation stated as causation, and the shield refuses an
unsafe action at the boundary — all deterministic and offline.

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

Latency-tolerant batch work must return a result for every request, losing one
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
| At a fixed candidate budget, verifier-guided best-of-N beats the single-shot draw, a quality-per-dollar Pareto improvement. | ≥ +0.1 quality | `families.test_time_compute.pareto_quality_gain` |
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
the dead end in imagination and commits the patient path instead, here it opens
the vault while the reactive planner is stuck. The model only earns planning
weight once its predictions are calibrated against the real environment, the way a
judge ensemble must earn its gating weight; the budget gates a strict planning win
and full prediction accuracy, above the published promises.

## Causal record-replay debugger

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| A recorded run replays byte-identically, the recording, not the live provider, serves the answer. | byte-identical | `families.record_replay.replay_faithful` |
| When the code under replay changes, the divergence is detected and reported. | true | `families.record_replay.divergence_detected` |
| A recording round-trips through a content-addressed store and verifies before replay. | true | `families.record_replay.fidelity_verified` |

A run is deterministic except at its edges, every place it reads the outside
world. The recorder captures those edges (model responses, tool outputs,
retrieval hits, the negotiated capabilities, clock/seed) keyed to the trace; the
replayer serves them back, so replay reproduces the run byte-for-byte against a
live provider that would answer differently. Because each edge is keyed by the
same identity the live code computes, a changed edge is a cache miss, and a miss
is a divergence, reported with the edge that drifted rather than silently
re-executed. Recordings are content-addressed and carry a fidelity digest, so a
recording shared across processes is verified before it is trusted for replay.

## Learned semantic cache & near-miss KV reuse

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| An accepted near-miss served from the learned semantic cache is at-least-as-good as the live answer the same request would have produced, at a fixed budget. | ≥ 0.90 quality | `families.semantic_cache.accepted_near_miss_quality` |
| A near-miss below the calibrated acceptance threshold is never served. | true | `families.semantic_cache.below_bar_never_served` |
| The eval-replay regression gate passes a faithful cache and blocks a drifted one. | true | `families.semantic_cache.gate_blocks_drift` |

Exact-match caching serves a byte-identical request for free; the rung above it is
near-miss reuse, answering a request that is *semantically equivalent* to a recent
one straight from cache. The risk is serving a near-miss that is not actually
equivalent, so the cache never serves below a **calibrated** acceptance threshold:
the bar is fit from labelled trace pairs so accepted near-misses clear a precision
target, falling back to off rather than guessing when the target is unreachable.
Every accepted hit is auditable and reversible, and a cache whose calibration has
drifted is caught by the same eval-replay no-regression check that gates a model
swap. Cross-request KV-prefix reuse extends the warm-prefix layout from one request
to a family that shares a stable head, reporting the serving-engine KV the shared
head avoids recomputing, all held under the resident-memory budget. The budgets
gate full hit-quality and full precision, above the published floors.

## On-device fine-tuning & continual local adaptation

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| A LoRA-class adapter fit on-device is at-least-as-good as its base model on the held-out eval set before it is promoted. | true | `families.local_adaptation.at_least_as_good` |
| An on-device adapter reshapes only in-distribution traffic; an off-distribution request falls through to the base model untouched. | true | `families.local_adaptation.off_distribution_inert` |
| A regressing adapter is refused, never promoted or applied, the registry head unchanged. | true | `families.local_adaptation.regression_refused` |

The flywheel turns traces into hosted fine-tune jobs and the in-process GGUF
provider runs a model air-gapped; on-device adaptation closes the loop by fitting a
LoRA-class adapter **in your process** from the same grounded data, so an edge or
air-gapped deployment improves on its own traffic with no hosted round-trip. The
risk is shipping a local change that silently degrades quality, so a new adapter
version is promoted only behind the same no-regression gate a hosted fine-tune job
clears, the adapted model must be at-least-as-good as its base on a held-out set.
The adapter is bounded (inert off-distribution), every version is content-addressed
and versioned, and a regression is refused and rolled back.

## Federated / cross-org self-improvement

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| A federated contribution shares geometry, never raw traffic: no prompt or response appears in the wire artifact. | true | `families.federated.no_raw_traffic` |
| No single member's contribution is recoverable: a masked update is hidden on its own, yet the masks cancel exactly so the aggregate equals the true fleet sum. | true | `families.federated.secure_aggregation_masks_cancel` |
| A merged federated candidate is at-least-as-good as its base on the held-out eval set before any member adopts it, and is refused and reversible otherwise. | true | `families.federated.at_least_as_good` |

On-device adaptation improves a model on its own traffic within one trust boundary;
federated self-improvement lets a fleet improve **together without sharing the raw
traffic**. Each member contributes only the numeric, clipped, masked subspace
scatter of its local adapter geometry, never a prompt or response, and a secure
aggregation merges the fleet's contributions so no single member's update is ever
observed, refusing a round below the k-anonymity contributor floor. The risk is two
sided: leaking a member's data, or shipping a merged change that degrades quality.
Both are gated, the privacy SLOs hold that nothing but numeric aggregates crosses a
boundary and that individual updates are unrecoverable, while the no-regression SLO
holds that the adopting member's re-fit adapter clears the same at-least-as-good gate
a local promotion does, versioned and reversible.

## Cross-fleet reputation & weighting

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| A member's pull on the consensus geometry is weighted by a reputation earned only from how its contributions fared against the no-regression gate: discounting a regressing or adversarial member leans the merged subspace toward the reliable members, where an equal-weight merge would let it pull the consensus astray. | true | `families.reputation.discount_aligns_consensus` |
| Reputation only ever lowers a member's pull, never bypasses the quality bar: a reliability-weighted round still adopts only when at-least-as-good as base, so even a pristine-reputation member cannot push a regressing adapter through the gate. | true | `families.reputation.gate_not_bypassed` |

The federated round merged every member with equal weight, so a member whose
contributions repeatedly fail the gate pulled the shared consensus as hard as one
whose contributions consistently help. A reputation ledger earns a per-member
reliability score from the gate verdicts on the audit chain, never from raw traffic,
and the secure aggregator weights each member by it, discounting an unreliable or
adversarial member without singling it out. The discount-the-regressor SLO holds that
weighting measurably leans the consensus toward the reliable members; the
no-regression SLO holds that the discount is bounded (a weight never leaves
`[floor, 1]`) and reversible (adoption still clears the same gate), so reputation
changes only which geometry the fleet converges toward when every candidate already
passes the gate; it is never a way around it.

## Differential-privacy memory & training

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| A subject's cumulative (ε, δ) privacy loss composes across every consolidation and learning round their data touches, bounded more tightly than naively summing each step's ε. | true | `families.privacy.composes_across_rounds` |
| A consolidation or contribution that would exceed a subject's privacy budget is refused, the privacy analogue of a hard cost cap, or down-weighted to fit; an over-budget release never silently proceeds. | true | `families.privacy.budget_refused` |
| The spent privacy budget is a mechanical, auditable number: a per-subject report sits alongside the cost report, and every spend and refusal is on the verifiable audit chain. | true | `families.privacy.on_audit_chain` |

The federated round bounds one member's *per-round* influence, but a subject's data
is touched again and again, by every memory consolidation and learning round. A
Rényi/moments privacy accountant composes the cumulative (ε, δ) a subject has spent
and **refuses** once the budget is gone, the privacy analogue of a dollar budget. The
composition SLO holds that the accountant tracks loss across rounds (and tighter than
the naive sum); the refusal SLO holds that an over-budget release is refused or
down-weighted, never silently admitted; and the auditability SLO holds that the spent
budget is provable, reported per subject and recorded on the signed audit chain.

## Energy & carbon accounting

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| Every run yields a per-run energy (Wh) and carbon (gCO₂e) estimate, accrued deterministically from token accounting against a per-model (by-tier) intensity and a per-region grid factor, the energy analogue of the per-run dollar cost, on the same cost-report surface. | true | `families.energy.per_run_estimate` |
| A run that would push a scope's accrued energy or carbon over its sustainability envelope is refused, the energy analogue of a hard cost cap; an over-budget run never silently proceeds. | true | `families.energy.budget_refused` |
| The estimate is a mechanical, offline, auditable number: computed in-process from a deterministic intensity table (no external service), on the cost-report surface, with the per-run estimate and every refusal on the verifiable audit chain. | true | `families.energy.auditable_offline` |

The cost report makes a run's dollar spend an auditable number; this adds the
sustainability figure beside it. A run's energy is accrued from its own token
accounting against a per-model intensity (by tier, from the model registry) scaled by
a datacenter overhead factor, and its carbon from a per-region grid factor, all from
a built-in, deterministic table, so the estimate is reproducible and consults no
external service. The per-run-estimate SLO holds that an enabled run reports a
positive, mechanical figure; the budget-refusal SLO holds that an energy or carbon
envelope refuses an over-budget run the way a hard cost cap refuses spend; and the
auditable-offline SLO holds that the figure and every refusal are on the verifiable
audit chain, computed in-process. Accounting is off until explicitly enabled.

## Edge / WASM in-process runtime

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| The edge / WASM build is the same library under a build target, not a fork: an edge compile is byte-identical to a direct server compile over the same inputs, so a capability can never silently diverge between server and edge. | true | `families.edge.parity_byte_identical` |
| The edge runtime holds a bounded resident-memory profile: the compiled packet's footprint stays under the profile cap even as the candidate corpus grows 10×, held by slimming and evidence eviction the way the server's resident-memory budget is. | true | `families.edge.bounded_profile` |
| The edge core is WASM-buildable: every module on the compile/score/rail/pack path imports no native or optional dependency unconditionally, so the dependency-free core compiles for a browser or edge worker. | true | `families.edge.no_native_imports` |

The dependency-free core runs at the edge through a thin in-process boundary
(`EdgeRuntime`), bounded by an `EdgeProfile` that lowers to the *same*
`ContextCompilerOptions` the server compiler reads. The parity SLO holds that an
edge compile and a direct server compile produce a byte-identical packet, the
edge build is exercised by the same offline test suite, never a fork. The
bounded-profile SLO holds the resident footprint under the cap as the corpus
grows 10×, by the same eviction the server's memory budget uses. The
no-native-imports SLO holds, by a static scan, that the core path pulls nothing
native at import time (NumPy stays behind its guarded pure-Python fallback), so
the core is WASM-buildable.

## Cross-org settlement fabric, end-to-end conformance

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| The cross-org settlement & credit fabric composes end-to-end as one system: a single `CrossOrgEngagement` threads the whole pipeline (negotiate → contract → choreograph delivery → settle → net → prove solvency) into one content-bound, signed `EngagementNarrative` that verifies offline from the bytes alone, with every captured artifact re-verified and one continuous hash-chained audit narrative. | true | `families.cross_org_conformance.conformance_verifies_offline` |
| A tamper introduced anywhere in a composed engagement is caught from the bytes alone: a re-ordered stage breaks the hash chain, an edited stage digest or underlying artifact fails the digest check, and a forged signature fails authentication, so the engagement narrative is an end-to-end integrity proof, not merely a transcript. | true | `families.cross_org_conformance.conformance_tamper_caught` |

The twenty cross-org rungs (negotiation, settlement, netting, arbitration,
reputation portability, admission, collateral, solvency, insolvency) each publish
their own SLO in [`benchmarks/slos.json`](https://github.com/Ohswedd/vincio/blob/main/benchmarks/slos.json);
this capstone conformance family holds that they compose into one verifiable
system. With it, the cross-org settlement & credit surface is feature-complete and
frozen under the [stability policy](stability.md).

## Computer-use & embodied action plane

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| A computer-use agent driving the grounded action plane reaches a verified end state within its action budget, perceiving a screen as typed, addressable elements, grounding each intent to a stable role+name selector (never a brittle pixel), acting, and post-verifying the effect. | true | `families.computer_use.success_at_budget` |
| No destructive computer-use action ever executes without approval: a destructive or out-of-scope action is pre-gated like a write tool and refused unless explicitly approved, and a divergent action is rolled back. | true | `families.computer_use.no_unapproved_destructive` |

Computer-use and provider-hosted tools already shipped as a thin GUI adapter, an
agent acting on brittle pixel coordinates that cannot replay, cannot survive a
layout shift, and cannot tell whether its action took effect. The action plane
(`app.computer_use`) binds every action to a stable selector and closes a perceive →
ground → pre-gate → act → post-verify → undo loop, so success is a reconstructable
end-state the same trajectory metrics and test-time search already score, the
success-at-budget bar the agentic leaderboards judge on, held offline on a
deterministic WebArena/OSWorld-shaped app. The safety bar is structural, not
best-effort: a destructive or out-of-scope action is refused unless approved (the
gate makes an unapproved destructive action impossible, not discouraged), and a
post-verify divergence is undone, the computer-use analogue of saga compensation,
so the plane is reversible and accountable by construction. The budgets gate a strict
success-at-budget win and zero unapproved-destructive actions, above the published
promises.

## Agent identity, delegation & cryptographic accountability

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| An agent identity is portable and self-certifying, its DID derives from its Ed25519 public key, its document verifies from the bytes, and keys rotate along a signed chain so a rotated-away or revoked key cannot forge new history while its past signatures stay valid. | true | `families.identity.identity_integrity` |
| A signed delegation composes into a chain that verifies offline where each link only attenuates, never amplifies, so an over-reaching or tampered sub-delegation is refused from the bytes. | true | `families.identity.delegation_attenuation` |

The platform signed every artifact, but *who* a key belonged to was an out-of-band
`key_id` string, accountability was only as strong as that assumption. Identity
(`app.identity`) makes the key first-class: a DID **derived from** the public key
(self-certifying, offline-resolvable, no registry), a content-bound `IdentityDocument`,
and a `Keyring` that rotates along a **signed chain** rather than a swap, so a
compromised or superseded key cannot rewrite history while everything it legitimately
signed stays valid. Authority is delegated as a bounded `Grant` that only ever
attenuates: a `DelegationChain` verifies offline (each issuer's key resolves from its
DID) and refuses any link that widens capabilities, the budget, or the expiry, so a
tool call, contract, or saga handoff carries provenance of authority. Ed25519 runs in
pure Python (RFC 8032), with the native `cryptography` backend used automatically
behind `vincio[crypto]`. The budgets gate full identity integrity and delegation
attenuation, above the published promises.

## Autonomous skill acquisition & open-ended curriculum

| Promise | Target | VincioBench metric |
|---|---|---|
| A full propose → attempt → verify → distill → promote cultivation run ends **at least as capable** as it began, capability on a held-out frontier set never falls, each promotion clears the same gated no-regression check a deploy uses, dead weight is demoted, and a tampered capability number is caught from the bytes. | true | `families.skill_acquisition.capability_monotonicity` |
| Every self-proposed objective is gated **before** it is attempted: an objective a safety rail blocks, or any objective when the governance invariants do not hold, is refused and never run, and the proposal's content hash catches a refused objective relabelled as proposed. | true | `families.skill_acquisition.stay_in_policy_safety` |

The self-improvement loop, RLVR, and the distillation flywheel make an agent better
at *known* tasks; open-ended capability growth (Voyager / ADAS-shaped) is the apex of
that arc, and its risk is unbounded drift. `app.cultivate` proposes tasks at the
frontier of current competence, attempts each with a library-composing test-time
search, verifies against the task-success oracle, distills a winning trajectory into a
verified, content-addressed `LearnedSkill`, and promotes it only through the **same
no-regression gate** a prompt or policy promotion clears, so growth is reversible, not
runaway, and a skill that stops paying its way is demoted rather than silently kept.
The `AutoCurriculum` gates every proposed objective through the rails and the
governance verifier, so the autonomy stays inside the controls the platform already
enforces. The budgets gate full capability monotonicity and stay-in-policy safety,
offline against the deterministic reference environments.

## Connected docs & capability map

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| Every internal documentation link resolves — both the file path and, for a Markdown target, the heading anchor — across all concept, guide, and reference pages, so the docs are a connected graph rather than leaf pages bound by one index. | true | `families.docs_conformance.docs_link_integrity` |
| Every public `app.*` verb is bound to the concept, guide, example, and reference that document it (the generated capability map places all of them under the six facades, every verb is documented in `api.md`, and every concept reaches a guide + example + reference anchor). | true | `families.docs_conformance.docs_capability_map_coverage` |
| A reader traverses laterally: every concept and guide carries a current single-sourced Related block, the generated pages (capability map, learning path, `api.md` app-method index) are current, and no docs page is orphaned. | true | `families.docs_conformance.docs_navigation_reachability` |

The docs are ~80 leaf pages — a concept, a guide, a reference entry, and a
runnable example per subsystem. 5.4 adds the connective tissue: `vincio._docmap`
is a single source of truth that binds every public `app.*` verb to the page that
documents it and renders the [capability map](capability-map.md), the
[learning path](../learning-path.md), a Related cross-link block on every concept
and guide, and `llms.txt` (regenerated from `vincio.__all__`). Companion budgets
gate that every concept is connected, that no page is orphaned, that `llms.txt` is
current, and that the gate *bites* — a synthetic broken link, an unmapped verb,
and a stale block are each caught. Run `vincio docs check` to reproduce it
offline.

## Public-surface hygiene

| SLO | Target | VincioBench metric (enforced by) |
|---|---|---|
| The two-level public surface stays consistent: every public subpackage `__all__` resolves to a live attribute (no duplicate/malformed entries), the classified surface matches the committed manifest, and the gate provably bites on an injected dead symbol, duplicate, and malformed `__all__`. | true | `families.hygiene.surface_consistency` |
| No public subpackage `__all__` exports a name that resolves to no attribute, lists a name twice, or is malformed — there is no dead public surface. | true | `families.hygiene.surface_dead_symbol_free` |
| The classified two-level surface matches `docs/reference/subpackage-surface.txt` byte-for-byte, so any `__all__` change (a new symbol, a removed one, or a TOP/DUP/SUB reclassification) is a deliberate, reviewed edit. | true | `families.hygiene.surface_frozen` |
| The surface-consistency gate provably bites: an injected dead symbol, a duplicate entry, and a malformed `__all__` are each reported. | true | `families.hygiene.surface_gate_detects_tamper` |
| Error-contract conformance: every error raised on a public entry point derives from `VincioError` — the `ContextApp` (`app.*` verb) surface raises no bare built-in, the classified baseline of accepted public built-in raises matches the committed manifest, and the detector provably catches an injected leak. | true | `families.hygiene.error_contract_conformant` |
| No public method of the user-facing `ContextApp` facade raises a bare built-in exception; every raise on the `app.*` verb surface is a `VincioError` subclass. | true | `families.hygiene.error_contract_app_verbs_clean` |
| The error-contract detector provably bites: an injected bare built-in raise on a public def is reported while an encapsulated private-def raise is not. | true | `families.hygiene.error_contract_gate_detects_tamper` |
| Observable failure: no best-effort fallback swallows a broad exception silently — every broad `except` (or `contextlib.suppress(Exception)`) on a public module re-raises, records its failure observably (a logger call or `note_suppressed`), or carries a justifying `# noqa: BLE001`, and the detector provably catches an injected silent swallow while ignoring a logged one. | true | `families.hygiene.observable_failure_conformant` |
| Every public module is free of unmarked silent broad-except swallows: the lint reports none tree-wide. | true | `families.hygiene.observable_failure_clean` |
| The observable-failure detector provably bites: an injected silent broad swallow is reported while a logged one is not. | true | `families.hygiene.observable_failure_gate_detects_tamper` |
| Wire-or-retire: every public capability is reachable through a production path — each entry in a frozen ledger resolves to a live reach (an `app.*` verb, an engine method, a registration helper, or a public class member) and, for a wired one, is referenced by production code outside its defining module, and the detector provably bites on an unreachable reach and a wired symbol with no production caller. | true | `families.hygiene.wire_or_retire_conformant` |
| Every capability in the wire-or-retire ledger is reachable: the guard reports no unreachable reach and no wired symbol that has become dead surface. | true | `families.hygiene.wire_or_retire_clean` |
| The wire-or-retire detector provably bites: an injected unreachable reach and a wired symbol with no production caller are each reported. | true | `families.hygiene.wire_or_retire_gate_detects_tamper` |
| Docstring / behaviour parity: every docstring that advertises a behaviour either performs it or is corrected — the budget allocator exposes no reclaim it does not run, the compression tuner gates on the faithfulness metric its docstring names, the federated default-deny consent path refuses deterministically, and `delete`/`forget` share one body — each re-derived from the live code. | true | `families.hygiene.docstring_parity_conformant` |
| The token-budget allocator advertises no separate redistribute reclaim: the method is gone and the allocator hands every non-fixed token to the flexible blocks at allocation time. | true | `families.hygiene.docstring_parity_budgeting` |
| The learned-compression docstring matches the gate: `CompressionTuner` reads the `faithfulness` eval metric it names, and `compression_faithfulness` / `faithfulness_preserved` measure answer-bearing survival offline. | true | `families.hygiene.docstring_parity_compression` |
| The federated default-deny consent demonstration is deterministic: a store-less default-deny ledger refuses an ungranted subject and a grant flips it, regardless of any consent persisted from an earlier run. | true | `families.hygiene.docstring_parity_consent` |
| `MemoryEngine.delete` delegates to `forget` — one body — with the audit semantics preserved: a plain delete records no reason, `forget` records one. | true | `families.hygiene.docstring_parity_memory` |

`vincio.__all__` is the frozen top-level contract, but each public subpackage also
declares its own `__all__` — the return types, dataclasses, and helpers reached by
deep import. That surface had drifted: a few names were exported yet referenced
nowhere (dead surface that reads as supported API), and the gap to the top level
was real but undeclared. The hardening line's 6.0 phase removes the verified-dead
symbols and declares the subpackage-only public surface in
`docs/reference/subpackage-surface.txt`; `vincio._surface` classifies
each symbol TOP (re-exported in `vincio.__all__`) / DUP (an intentional name
collision) / SUB (subpackage-only) and freezes the result, so the interior surface
can only change on review. Run `python -m vincio._surface` to reproduce it offline.

The 6.1 phase makes the **error contract** mechanical the same way. Vincio's
contract is that every error it raises derives from `VincioError`, so one
`except VincioError` catches the family and `.code` is the stable branch key. A few
public entry points leaked a bare built-in (`ValueError` / `KeyError` /
`NotImplementedError`); those are converted, the `ContextApp` verb surface is held
to zero off-contract raises by an always-on check, and the classified baseline of
accepted public built-in raises (internal input-validation, abstract-base
placeholders, the `AttributeError` a `__getattr__` must raise) is frozen in
`docs/reference/error-contract.txt`. A new public bare-built-in raise must be
converted to a `VincioError` or deliberately reviewed into the baseline. Run
`python -m vincio._error_contract` to reproduce it offline.

The 6.2 phase makes **observable failure** mechanical the same way. A best-effort
fallback that catches a broad `Exception` and continues is correct policy, but one
that swallows it silently (no re-raise, no log, no metric) hides a real bug.
`vincio.core.diagnostics.note_suppressed` makes such a fallback observable — it logs
the suppression on a dedicated `vincio.suppressed` channel and counts it by label, so
an operator can watch the failures or scrape their rate — and `vincio._observable_failure`
holds the whole public tree to zero unmarked silent swallows: every broad `except`
(or `contextlib.suppress(Exception)`) must re-raise, record its failure, or carry a
justifying `# noqa: BLE001`. Run `python -m vincio._observable_failure` to reproduce
it offline.

The 6.3 phase makes **wire-or-retire** mechanical the same way. A capability that is
public but that nothing can reach — no `app.*` verb, no example, no internal caller —
reads as supported API while being dead. Those are wired to a production path
(`app.retrieve_facts`, `app.consolidate_memory`, `use_context_governor(blob_store=…)`,
and a provider-native token counter registered at provider init) or, where the
primitive is a deliberate advanced deep-import API (`ContextCompiler.compile_streaming`
/ `recompile` / `CompileStreamEvent`), documented as such; `vincio._wire_or_retire`
holds a frozen ledger of them, requiring each to resolve to a live reach and — for a
wired one — to be referenced by production code outside its defining module, so a
capability cannot silently become dead surface again. Run
`python -m vincio._wire_or_retire` to reproduce it offline.

The 6.4 phase makes **docstring / behaviour parity** mechanical the same way. A
docstring that advertises behaviour the code no longer performs is a quiet lie a
reader trusts. The reconciled claims are re-derived from the live code so they cannot
drift back: the budget allocator's module docstring no longer promises a separate
`redistribute` reclaim nothing invoked (the dead method is gone, and the allocator
hands the whole non-fixed remainder to the flexible blocks at allocation time); the
learned-compression docstring no longer claims the tuner calls the faithfulness
helpers directly (`CompressionTuner` gates adoption on the `faithfulness` eval metric,
while `compression_faithfulness` / `faithfulness_preserved` are the offline fidelity
measures); the federated default-deny consent demonstration refuses every run, not
only against a pristine store; and `MemoryEngine.delete` delegates to `forget` so the
deletion path has one body. The `families.hygiene.docstring_parity_*` budgets exercise
each of these behaviours, so a docstring and its code cannot silently diverge again.

Quality and security floors describe behavior on the reference corpora; measure
on your own data with the same harness before depending on a number.
