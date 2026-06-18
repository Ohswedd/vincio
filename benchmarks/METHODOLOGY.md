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
| **AgentBench** | budget adherence under adversarial loops, crew termination, durable-graph determinism; (2.2) executor/crew token & tool-event streaming + AG-UI translation | unbounded loop |
| **ToolBench** | reliability, p50 runtime overhead, invalid-arg rejection, cache hits | — |
| **OutputBench** | recovery rate over malformed outputs vs raw parsing | `json.loads` |
| **ReliabilityBench** | constrained-decode closure, mid-stream abort savings, self-correction, rails | validate-at-end |
| **CostBench** | evidence-token reduction from the compiler | stuff-everything |
| **SecurityBench** | injection detection / false-positive rate, PII coverage | — |
| **EvalBench** | metric agreement, red-team judging, synthetic determinism, A/B significance | naive target |
| **AgenticEvalsBench** | trajectory/tool metric agreement, simulator determinism, drift sensitivity/specificity, κ tracking; (2.2) stateful-environment task-success oracle + deterministic hash-pinned replay of the five benchmark adapters | output-only eval |
| **LoopBench** | the closed loop end to end: promotion, gating, auto-memory, Pareto, learned budgets | ungated optimization |
| **ProtocolsBench** | MCP tool schema-fidelity + resource provenance, A2A delegation termination, Agent-Skill progressive-disclosure savings; (2.2) governed agent fabric (AGNTCY/ACP + MCP-registry discovery under the allow-list, audited resolution) | thin protocol adapter |
| **GovernanceBench** | card/AI-BOM completeness, OWASP/NIST/MITRE/ISO-42001 mapping coverage, erasure correctness + audit, multilingual PII recall, RAG-poisoning detection rate/FP, residency endpoint inference, signed-manifest verification | English-only / ungoverned |
| **GenerationBench** | document-contract validity (deficient rejected), cited-report coverage + per-claim entailment, media C2PA provenance binding + tamper rejection + disclosure, redline correctness, new-format ingestion recall, generated-media prompt safety | un-contracted / un-provenanced output |
| **PerfBench** | compile/retrieval/run latency, cache speedups, concurrent throughput, TTFT | cold paths |

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
driving a browser. Point an adapter at a live task set to score a real run; the
scoring code is identical.

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
