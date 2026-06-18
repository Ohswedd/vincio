# VincioBench

Benchmark suite for Vincio and baseline comparisons. Runs fully offline and
deterministically (mock provider + deterministic metrics) so results are
reproducible across machines and gate CI without API keys, quota, or network
flakiness — the point is a stable, self-hostable measurement, not a hosted
leaderboard.

```bash
python benchmarks/vinciobench.py             # all families
python benchmarks/vinciobench.py rag cost    # selected families
```

Results are printed as JSON and saved to `benchmarks/results/`.

## Families

| Family | Measures | Baseline |
|---|---|---|
| **PromptBench** | cacheability/tokens per render format, lint defect detection | naive string concatenation |
| **RAGBench** | recall@3 / MRR per retrieval mode (bm25, dense, sparse, late_interaction, PLAID-compressed, hybrid, hybrid_full, + query understanding); GraphRAG community building; (1.5) MRL recall-vs-dimension (`families.rag.mrl`) and unified multimodal recall/MRR (`families.rag.multimodal`); (2.2) retrieval-eval recall@k/nDCG@k on a golden set + index-version regression caught on recall deltas with the swap significance test (`families.rag.retrieval_eval`) | single-index BM25 |
| **MemoryBench** | preference recall, contradiction supersede, cross-user isolation | — |
| **AgentBench** | budget adherence under an adversarial looping model, DAG success; crew over-budget termination, full-crew success, delegation recording; durable-graph interrupt→resume and fork-replay determinism; composition streaming coverage; (2.2) `AgentExecutor`/`Crew` token & tool-event streaming, AG-UI run-lifecycle translation, and an MCP-UI resource served (`families.agent.streaming`) | unbounded loop |
| **ToolBench** | reliability, runtime overhead (p50 ms), invalid-arg rejection, cache hits | — |
| **OutputBench** | recovery rate over malformed model outputs; missing-required correctly rejected | raw `json.loads` |
| **ReliabilityBench (0.7)** | strict-schema closure for constrained decoding; mid-stream invalid detection + abort savings; self-correction recovery within cycle bounds; rail catch rate + false positives; signature prediction validity + optimizer variants; schema routing/classification accuracy | validate-at-end / unguarded output |
| **CostBench** | evidence-token reduction from the context compiler | stuff-everything context |
| **SecurityBench** | injection detection rate, false-positive rate, PII coverage | — |
| **EvalBench** | metric agreement on labeled examples; red-team judging (guarded vs naive target, detector coverage); synthetic-data determinism and coverage; A/B significance machinery; session grouping; HTML viewer self-containment; trace→dataset; G-Eval calibration | naive target (85% attack success) |
| **AgenticEvalsBench (1.2)** | trajectory & tool-use metric agreement with labeled traces; trajectory eval flags runs that output-only eval passes; user-simulator determinism; drift sensitivity/specificity; Cohen's-κ judge-agreement tracking; (2.2) stateful-environment task-success oracle (verifies the end state, rejects policy violations) + deterministic replay of the five benchmark adapters with hash-pinned task sets (`families.agentic_evals.environment_eval`) | output-only evaluation / no drift detector |
| **LoopBench (0.8)** | the closed loop end to end: promotion fires, decisions are deterministic, gates block regressions, the registry version is tagged + eval-linked; grounded auto-memory precision (grounded written, ungrounded excluded); retrieval-feedback improvement + gating; Pareto frontier correctness (dominated excluded, knee balanced); learned-budget promotion; guided-search budget bounds | ungated / single-score optimization |
| **ProtocolsBench (1.1)** | MCP tool schema-fidelity + round-trip through the permissioned runtime + resource provenance (`origin: mcp:<server>`); A2A budget-bounded crew delegation terminates; Agent-Skill progressive-disclosure savings (off-topic bodies stay out of budget, index always present); (2.2) the governed agent fabric — allow-list-gated + audited resolution, capability discovery, AGNTCY/ACP roundtrip + discovery, and MCP-registry discovery with unlisted servers denied (`families.protocols.fabric`) | thin protocol adapter (no permissions/provenance/budget) |
| **GovernanceBench (1.6)** | model/system-card and AI-BOM completeness; compliance-framework mapping coverage (OWASP LLM 2025 / OWASP Agentic / NIST AI RMF / MITRE ATLAS / ISO IEC 42001); erasure-by-source correctness (chunks removed = lineage) + audit; multilingual PII recall + English-path intactness; RAG-poisoning detection rate + false-positive rate; residency endpoint-region inference + jurisdiction matching; signed-manifest verification (tamper/wrong-key fail closed) | English-only / ungoverned |
| **GenerationBench (1.9)** | document-contract validity (valid passes, deficient rejected — no invention); cited-report citation coverage + per-claim entailment + unresolved-marker detection; media C2PA provenance binding (image + audio) with tamper rejection and synthetic-content disclosure; redline correctness; new-format ingestion recall; generated-media prompt safety | un-contracted / un-provenanced output |
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

## Published SLOs vs CI budgets

There are two tiers. `slos.json` is the **public promise** — the
[performance & quality SLOs](../docs/reference/slo.md) users can rely on. Each
SLO names the `budgets.json` key that enforces it, and the budget is always held
**at least as strict** as the published target, so a green build provably honors
the SLO. `tests/test_slos.py` checks that invariant. Every report also carries an
`environment` block (Vincio/Python versions, platform, schema version) so a
saved report is self-describing and reproducible.

See **[METHODOLOGY.md](METHODOLOGY.md)** for what each family measures, its naive
baseline, corpus provenance, and how to reproduce every number on your own data.

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
