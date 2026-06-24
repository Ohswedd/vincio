# VincioBench methodology

VincioBench is the measurement system behind Vincio's published guarantees. It
is designed to be **reproducible by anyone, on their own machine, with no API
keys** — there is no hosted leaderboard and no number you have to take on
trust. This document explains what it measures, how, and why the thresholds are
where they are, so you can run it yourself and audit the claims.

## Principles

1. **Offline and deterministic.** Every family runs against the deterministic
   `MockProvider` and computes metrics from fixed, in-repo corpora. No network,
   no quota, no model-sampling variance. The same commit produces the same
   metrics on any machine (only per-family `_duration_ms` and absolute
   latencies vary with hardware).
2. **Hypotheses are measured, never asserted.** Each family compares the Vincio
   pipeline against a *named naive baseline* (e.g. string concatenation for
   prompts, single-index BM25 for retrieval, `json.loads` for output, "stuff
   everything" for cost) and reports the delta. A claim that isn't measured
   isn't made.
3. **Two tiers of thresholds.** CI gates on `budgets.json`; users are promised
   `slos.json`. The budget is always held *at least as strict* as the published
   SLO, so any green build necessarily honors the SLO. `tests/test_slos.py`
   enforces that invariant.
4. **Latency budgets are loose on purpose.** They run on shared CI runners and
   exist to catch order-of-magnitude regressions. Ratio, quality, and
   correctness budgets are tight. Treat absolute milliseconds as
   machine-relative; treat ratios and quality floors as portable.

## What each family measures

| Family | Question it answers | Naive baseline |
|---|---|---|
| **PromptBench** | Do compiled layouts cut tokens and raise cache prefix reuse vs hand-concatenation? Does the linter catch known defects? | string concatenation |
| **RAGBench** | recall@3 / MRR per retrieval mode; GraphRAG community building; (2.2) retrieval-eval recall@k/nDCG@k + index-version regression on recall deltas | single-index BM25 |
| **MemoryBench** | preference recall, contradiction superseding, cross-user isolation, staleness | — |
| **AgentBench** | budget adherence under adversarial loops, crew termination, durable-graph determinism; executor/crew token & tool-event streaming + AG-UI translation; orchestrator & planner depth (HTN decomposition, in-place plan repair, cost-aware action-selection savings, durable-timer restart safety) + parallel sub-graph scheduling (concurrency, speedup, fair-share budget, SLA-deadline partials) | unbounded loop / restart-on-failure / always-strong / serial |
| **ToolBench** | reliability, p50 runtime overhead, invalid-arg rejection, cache hits | — |
| **OutputBench** | recovery rate over malformed outputs vs raw parsing | `json.loads` |
| **ReliabilityBench** | constrained-decode closure, mid-stream abort savings, self-correction, rails | validate-at-end |
| **CostBench** | evidence-token reduction from the compiler | stuff-everything |
| **SecurityBench** | injection detection / false-positive rate, PII coverage | — |
| **EvalBench** | metric agreement, red-team judging, synthetic determinism, A/B significance | naive target |
| **AgenticEvalsBench** | trajectory/tool metric agreement, simulator determinism, drift sensitivity/specificity, κ tracking; (2.2) stateful-environment task-success oracle + deterministic hash-pinned replay of the nine benchmark adapters; κ-gated judge ensembles, Shapley causal regression attribution, verdict-preserving adaptive sampling | output-only eval |
| **LoopBench** | the closed loop end to end: promotion, gating, auto-memory, Pareto, learned budgets | ungated optimization |
| **ProtocolsBench** | MCP tool schema-fidelity + resource provenance, A2A delegation termination, Agent-Skill progressive-disclosure savings; (2.2) governed agent fabric (AGNTCY/ACP + MCP-registry discovery under the allow-list, audited resolution) | thin protocol adapter |
| **GovernanceBench** | card/AI-BOM completeness, OWASP/NIST/MITRE/ISO-42001 mapping coverage, erasure correctness + audit, multilingual PII recall, RAG-poisoning detection rate/FP, residency endpoint inference, signed-manifest verification | English-only / ungoverned |
| **GenerationBench** | document-contract validity (deficient rejected), cited-report coverage + per-claim entailment, media C2PA provenance binding + tamper rejection + disclosure, redline correctness, new-format ingestion recall, generated-media prompt safety | un-contracted / un-provenanced output |
| **PerfBench** | compile/retrieval/run latency, cache speedups, concurrent throughput, TTFT | cold paths |
| **IntegrationsBench** | every first-party connector round-trips offline against a recorded fixture with provenance; the plugin contract loads compatible plugins and gates incompatible majors; the signed community registry resolves under an allow-list (audited), verifies signatures, and detects tampering; Haystack/DSPy bridges adapt in; the MCP-server marketplace bridge discovers→governs→connects in one call | hand-wired loaders / ungoverned, unsigned registries |
| **LongHorizonBench (3.10)** | does a governed long run stay bounded as the horizon grows 10×? — resident/token growth ratio (flat vs ~linear naïve), recall preserved by paging a compacted needle back from the content-addressed store, provenance retained through compaction, and intra-run decay demoting stale spans | naïve accumulation (linear footprint, context rot) |
| **WorldModelBench (3.11)** | can an agent learn a model of its tools and plan against it? — next-state accuracy of the learned dynamics, the learned precondition (refund fails on a processing order, succeeds on a cancelled one), argument generalization, the calibration gate, and the planning-accuracy SLO (the imagined-rollout planner opens the vault while a reactive one-step planner is trapped at a fixed action budget) | reactive (one-step) planning |
| **AssuranceBench (3.49)** | can the platform's evidence be assembled into one continuously-checked safety argument? — assurance soundness (a fully-evidenced `assurance_case` holds and verifies offline, while a claim resting on missing, stale, or falsified evidence is pinpointed and fails) and assurance regression (re-checking on a change catches a previously-discharged claim now falsified and fails the build, a signed incident makes the case demand a remediation proof, and a certification report verifies from the bytes) | a point-in-time audit / a self-asserted fitness claim |

## Corpora and provenance

All inputs are committed to the repository (see `CORPUS` and `QA_CASES` in
`vinciobench.py` and the per-family fixtures). They are small, synthetic, and
hand-labeled — chosen so the suite runs in a couple of seconds and every metric
is checkable by eye. They are **not** claimed to be representative of any
production document distribution; RAG quality numbers describe behavior on this
reference corpus, and you should re-measure on your own data with the same
harness. Bring your own corpus by editing the family fixtures; the metric code
is unchanged.

The **agentic benchmark adapters** (SWE-bench Verified / τ-bench / GAIA /
WebArena / BFCL) read small recorded fixtures committed under
`benchmarks/fixtures/`. Each fixture declares a `task_set_hash` the adapter
recomputes and verifies on load, so a silent task-set change is caught; offline,
the adapters **replay a recorded agent output against each benchmark's own
verifiable scorer** (SWE-bench's fail-to-pass/pass-to-pass transition, τ-bench's
database end state via the environment oracle, GAIA's normalized exact match,
WebArena's functional check, BFCL's AST match) rather than cloning repos or
driving a browser.

Pointing an adapter at a **live task set** is a one-liner with the *identical*
scoring code: `adapter.run(solver)` solves each task fresh and scores it, where
`make_agent_solver(app_or_executor)` drives a real `ContextApp`/`AgentExecutor`
(`mode="text"` for an answer, `mode="calls"` for BFCL function calls captured
from the agent's event stream) and `make_env_solver(policy)` runs a policy through
the τ-bench world; `gaia_tasks_from_export` / `swebench_tasks_from_export` /
`bfcl_tasks_from_export` / `tasks_from_jsonl` load the official released formats.
The `agentic_evals` family exercises this live path (`adapters.live_run_scored`),
so the scorer is gated on both recorded and freshly-solved output.

## Reproducing the numbers

```bash
python benchmarks/vinciobench.py            # all families -> results/vinciobench_latest.json
python benchmarks/check_budgets.py          # gate the report against budgets.json (exit 1 on breach)
python benchmarks/profile_stages.py         # per-stage latency breakdown from trace spans
```

Every report carries an `environment` block — `vincio_version`,
`python_version`, `platform`, `schema_version`, and `provider` — so a saved
report is self-describing. The block intentionally omits wall-clock time so the
committed reference report stays diff-stable.

## Published SLOs vs CI budgets

`slos.json` is the human-readable promise; each entry names the `enforced_by`
budget key that gates it. The relationship is mechanical, not aspirational:

- `slo.threshold` is the public number (e.g. retrieval p95 ≤ 150 ms).
- `budgets.json` holds the same metric at least as strict (e.g. ≤ 100 ms).
- A passing CI run therefore proves the SLO holds, with headroom.

See `docs/reference/slo.md` for the published table and `docs/guides/performance.md`
for tuning guidance.

## Adding a family or metric

1. Write `async def bench_<name>() -> dict[str, Any]` returning a flat-ish dict
   of metrics plus any sub-dicts; register it in `FAMILIES`.
2. Add budget keys (dotted paths into the report) to `budgets.json`.
3. If the metric is a user-facing promise, add an `slos.json` entry referencing
   the budget key.
4. Document the family row above and in `benchmarks/README.md`.
