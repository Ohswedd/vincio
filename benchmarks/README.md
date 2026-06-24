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

## Three kinds of benchmark

| Suite | File | Compares against | Question it answers |
|---|---|---|---|
| **VincioBench** | `vinciobench.py` | a *naive in-house baseline* (stuff-everything, `json.loads`, single-index BM25, …) | does each Vincio mechanism do what it claims, deterministically? |
| **Competitive** | `competitive.py` | the *actual third-party library* a team would otherwise use (tiktoken, rank_bm25, LangChain, LlamaIndex, …) | where Vincio meets or beats a specialist library head-to-head |
| **Orchestrator uplift** | `quality_uplift.py` | calling the *same model directly* (no Vincio layer) | what routing a model through Vincio adds: output quality, token usage, context-rot resistance |

```bash
python benchmarks/competitive.py             # all head-to-head comparisons
python benchmarks/quality_uplift.py          # what the orchestration layer adds
pip install tiktoken rank-bm25 json-repair jinja2 langchain-core \
            langchain-text-splitters llama-index-core   # the competitors
```

### Competitive — head-to-head vs. real libraries

`competitive.py` runs Vincio against the real libraries on the operations where
an apples-to-apples comparison genuinely exists. **Every number it prints is
measured on your machine from a live run of both sides** — nothing is
hand-written, a missing competitor is reported as `skipped` rather than assumed,
and the report states it plainly where a specialist library wins on its own
narrow axis. Representative results (Apple Silicon, Python 3.13 — *ratios*, not
wall-clock, are the portable signal):

| Operation | Vincio | Competitor | Result |
|---|---|---|--|
| **BM25 query** @ 20k docs, selective queries | `BM25Index` | `rank_bm25` | **~32× faster**, 100% identical top-1 ranking |
| **BM25 query** @ 2k docs | `BM25Index` | `rank_bm25` | **~18× faster**, identical ranking |
| **Context assembly** — tokens sent for the same retrieved set | context compiler | LangChain `stuff` / LlamaIndex `compact` | **~60% fewer tokens**, answer retained (scores + dedups + budgets) |
| **Text chunking** a 24k-word doc | `chunk_document` | LangChain / LlamaIndex splitters | **fastest**, and chunks carry provenance the string-splitters don't |
| **Token counting** (~60k words) | `HeuristicTokenCounter` | `tiktoken` | **~1.5× faster**, zero-dependency, conservative (+~25%) |
| **Malformed-JSON recovery** | lenient parser | stdlib `json.loads` | **4/8 vs 1/8** (a dedicated repair lib recovers more, by guessing) |
| **Template render w/ a missing var** | `PromptSpec.substitute` | `jinja2` | raises a typed error vs silently rendering empty |

Why BM25 wins by so much: `rank_bm25` rescans *every* document on every query
(O(N) per query); Vincio's inverted postings only scan documents that contain a
query term, so the gap widens with corpus size. `rank_bm25`'s NumPy rescan can
win on a few hundred docs — the report shows both regimes. The honest takeaway is
not "every component beats every specialist": a dedicated repair library recovers
more malformed JSON than Vincio (by guessing, which is unsafe for typed
extraction). Vincio's edge is an *integrated, correct, governed* pipeline — one
BM25 mode fused beside dense/sparse/late-interaction, with provenance, tenant
scoping, and a token budget — not a pile of specialist libraries you wire
together yourself.

### Orchestrator uplift — Vincio vs. the raw model

`quality_uplift.py` asks the question the micro-benchmarks don't: take the *same
model* and call it directly (the way an agent harness or a web chat does) versus
through Vincio's pipeline — what changes? It keeps two regimes strictly separate
so nothing is overstated:

- **Deterministic (real numbers, below):** contributions that hold for *any*
  model because they are mechanical, measured offline on the deterministic mock.
- **Frontier-model quality (harness only):** the absolute lift in answer
  correctness needs a real model — the suite ships the harness and prints the
  exact `VINCIO_PROVIDER=…` command, and **does not print a quality number it did
  not measure.**

| Metric (same model, direct vs. via Vincio) | Direct | Via Vincio |
|---|--|--|
| Schema-valid object from realistic model outputs | 1/6 | **5/6** (structure-only repair) |
| Prompt-injection exfiltration via a tool call | compromised | **contained** (taint + capability token) |
| Context tokens to retain an early fact at 80 turns | 640 (needle falls out of window) | **33, needle retained** (bounded recall) |
| Grounded + cited answers (deterministic illustration) | 0/3 | **3/3** — real-model delta needs a provider key |

## Families

| Family | Measures | Baseline |
|---|---|---|
| **PromptBench** | cacheability/tokens per render format, lint defect detection | naive string concatenation |
| **RAGBench** | recall@3 / MRR per retrieval mode (bm25, dense, sparse, late_interaction, PLAID-compressed, hybrid, hybrid_full, + query understanding); GraphRAG community building; (1.5) MRL recall-vs-dimension (`families.rag.mrl`) and unified multimodal recall/MRR (`families.rag.multimodal`); (2.2) retrieval-eval recall@k/nDCG@k on a golden set + index-version regression caught on recall deltas with the swap significance test (`families.rag.retrieval_eval`) | single-index BM25 |
| **MemoryBench** | preference recall, contradiction supersede, cross-user isolation | — |
| **AgentBench** | budget adherence under an adversarial looping model, DAG success; crew over-budget termination, full-crew success, delegation recording; durable-graph interrupt→resume and fork-replay determinism; composition streaming coverage; `AgentExecutor`/`Crew` token & tool-event streaming, AG-UI run-lifecycle translation, and an MCP-UI resource served (`families.agent.streaming`); orchestrator & planner depth — HTN decomposition (parallel sub-goal on one level), in-place plan repair (re-bind / substitute / budget-shock drop), cost-aware action-selection savings vs always-strong, and durable-timer restart safety + event-wait resume (`families.agent.planner_depth`); plus parallel sub-graph scheduling — work-stealing concurrency, speedup vs serial, fair-share budget summing to the cap, and an SLA deadline returning partial results (`families.scale.subgraph`) | unbounded loop / restart-on-failure / always-strong / serial |
| **ToolBench** | reliability, runtime overhead (p50 ms), invalid-arg rejection, cache hits | — |
| **OutputBench** | recovery rate over malformed model outputs; missing-required correctly rejected | raw `json.loads` |
| **ReliabilityBench (0.7)** | strict-schema closure for constrained decoding; mid-stream invalid detection + abort savings; self-correction recovery within cycle bounds; rail catch rate + false positives; signature prediction validity + optimizer variants; schema routing/classification accuracy | validate-at-end / unguarded output |
| **CostBench** | evidence-token reduction from the context compiler | stuff-everything context |
| **SecurityBench** | injection detection rate, false-positive rate, PII coverage | — |
| **EvalBench** | metric agreement on labeled examples; red-team judging (guarded vs naive target, detector coverage); synthetic-data determinism and coverage; A/B significance machinery; session grouping; HTML viewer self-containment; trace→dataset; G-Eval calibration | naive target (85% attack success) |
| **AgenticEvalsBench (1.2)** | trajectory & tool-use metric agreement with labeled traces; trajectory eval flags runs that output-only eval passes; user-simulator determinism; drift sensitivity/specificity; Cohen's-κ judge-agreement tracking; (2.2) stateful-environment task-success oracle (verifies the end state, rejects policy violations) + deterministic replay of the nine benchmark adapters with hash-pinned task sets; judge ensembles whose disagreement is an uncertainty signal and whose calibration is κ-gated, Shapley causal regression attribution, and verdict-preserving adaptive sampling (`families.agentic_evals.environment_eval` / `families.agentic_evals.quality_frontier`) | output-only evaluation / no drift detector / un-attributed regressions |
| **LoopBench (0.8)** | the closed loop end to end: promotion fires, decisions are deterministic, gates block regressions, the registry version is tagged + eval-linked; grounded auto-memory precision (grounded written, ungrounded excluded); retrieval-feedback improvement + gating; Pareto frontier correctness (dominated excluded, knee balanced); learned-budget promotion; guided-search budget bounds | ungated / single-score optimization |
| **ProtocolsBench (1.1)** | MCP tool schema-fidelity + round-trip through the permissioned runtime + resource provenance (`origin: mcp:<server>`); A2A budget-bounded crew delegation terminates; Agent-Skill progressive-disclosure savings (off-topic bodies stay out of budget, index always present); (2.2) the governed agent fabric — allow-list-gated + audited resolution, capability discovery, AGNTCY/ACP roundtrip + discovery, and MCP-registry discovery with unlisted servers denied (`families.protocols.fabric`) | thin protocol adapter (no permissions/provenance/budget) |
| **GovernanceBench (1.6)** | model/system-card and AI-BOM completeness; compliance-framework mapping coverage (OWASP LLM 2025 / OWASP Agentic / NIST AI RMF / MITRE ATLAS / ISO IEC 42001); erasure-by-source correctness (chunks removed = lineage) + audit; multilingual PII recall + English-path intactness; RAG-poisoning detection rate + false-positive rate; residency endpoint-region inference + jurisdiction matching; signed-manifest verification (tamper/wrong-key fail closed) | English-only / ungoverned |
| **GenerationBench (1.9)** | document-contract validity (valid passes, deficient rejected — no invention); cited-report citation coverage + per-claim entailment + unresolved-marker detection; media C2PA provenance binding (image + audio) with tamper rejection and synthetic-content disclosure; redline correctness; new-format ingestion recall; generated-media prompt safety | un-contracted / un-provenanced output |
| **PerfBench** | compile/retrieval/run latency (p50/p95/p99), cache speedups, concurrent throughput, streaming TTFT; the compile hot-path families — vectorized-scoring equivalence, render-program byte-identity, warm-candidate-arena equivalence, streaming-first compilation, speculative-prefetch warm, a sub-millisecond warm compile, and a resident-footprint regression gate (`families.perf.{vectorized_scoring,render_program,warm_arena,streaming_compile,prefetch,footprint}`) | cold (uncached) paths |
| **LongHorizonBench (3.10)** | long-horizon context engineering — at 10× horizon the `ContextGovernor` holds resident/token footprint flat (vs ~linear naïve growth), preserves recall by paging a compacted needle back from the content-addressed store, retains provenance through hierarchical compaction, and demotes stale spans via intra-run decay (`families.long_horizon`) | naïve accumulation (linear footprint, context rot) |
| **WorldModelBench (3.11)** | world-model / simulation-based planning — a `WorldModel` fit offline from recorded transitions predicts next states accurately, learns a precondition (refund fails on a processing order, succeeds on a cancelled one) and generalizes over arguments, earns planning weight only once calibrated, and lets the `ModelPredictivePlanner` open the vault while a reactive one-step planner is trapped at a fixed action budget (`families.world_model`) | reactive (one-step) planning |
| **SemanticCacheBench (3.13)** | learned semantic cache & near-miss KV reuse — the acceptance threshold is calibrated from labelled trace pairs (precision-targeted, never serving below the bar), an accepted near-miss served through the run path is at-least-as-good as the live answer at a fixed budget, a below-bar match is never served, the eval-replay gate passes a faithful cache and blocks a drifted one, cross-request KV-prefix reuse reports the shared head's avoided recompute, and the cache stays inside the resident-memory budget (`families.semantic_cache`) | exact-match-only caching (no near-miss reuse) |
| **IntegrationsBench** | ecosystem & integration breadth — every first-party connector (Jira, Linear, Google Drive, SharePoint, Salesforce, Zendesk, BigQuery, Snowflake) round-trips offline against a recorded fixture with full provenance; the entry-point plugin contract loads a compatible plugin on install and gates an incompatible-major one; the signed community registry resolves a bundle under the allow-list (audited), verifies its signature, and denies a tampered one; the Haystack and DSPy bridges adapt assets in; and the MCP-server marketplace bridge discovers → governs (audited) → lands a server's tools in the permissioned runtime in one call, denying an unlisted server (`families.integrations`) | hand-wired loaders / ungoverned, unsigned registries |
| **EdgeBench (3.21)** | edge / WASM in-process runtime — an edge compile is byte-identical to a direct server compile over the same inputs (parity, not a fork) and the runtime delegates to the canonical compiler/rail-engine; the bounded edge profile holds the resident footprint under its cap as the candidate corpus grows 10× (eviction fires under load); the compile/score/rail/pack path imports nothing native unconditionally (WASM-buildable); the deterministic rails refuse a secret leaking from evidence into the rendered context; and it runs offline with no provider, store, or network (`families.edge`) | a forked/diverging edge build / unbounded footprint |
| **MCPAppsBench (3.22)** | MCP Apps & the evolving MCP spec — a consumed server's `ui://` UI resource is surfaced through the existing AG-UI channel as an `mcp.ui` event under the run's governance (untrusted-external provenance, token-metered so an oversized render is refused not streamed, audited); a server's mid-call elicitation is gated by the same approval + rail machinery a write tool passes, so a secret value is refused and an accepted value is tainted untrusted (contained end-to-end through a server tool); and evolving-spec parity — protocol-version negotiation honours a peer pinned to an older stable revision (`families.mcp_apps`) | an ungoverned UI side-channel / a trusted elicited value |
| **NegotiationBench (3.23)** | agent negotiation & contracting — a bounded offer/counter bargain over typed price/SLA/scope terms always **terminates** within its round/deadline budget (a deal when the parties' acceptable regions overlap, a clean no-deal when they do not, a partial result on a wall-clock deadline); the agreed **contract** is signed by both parties, verifies offline from the bytes alone, catches a tampered term, and enforces like a budget (`to_budget` / `check`); the counterparty's reputation **discounts** a regressing agent's offers without singling it out and selects the reputation-weighted best deal; and the bargain runs over the A2A fabric byte-for-byte the same as locally, every outcome on the audit chain (`families.negotiation`) | an unbounded haggle / a self-asserted, unverifiable agreement |
| **AssuranceBench (3.49)** | continuous assurance cases & production certification — `app.assurance_case` assembles the evidence the platform already emits (eval gates, the governance verifier, reasoning certificates, the signed audit chain, identity/delegation provenance, SBOM/SLSA) into a content-bound argument tree, and **no claim stands on missing or stale evidence**: a fully-discharged case holds and verifies offline while a leaf whose evidence is absent, expired past its freshness horizon, or falsified is pinpointed and the case fails (`families.assurance.assurance_soundness`); the case is re-checked on every change and `assurance_regression_gate` turns a previously-discharged claim now falsified into a build failure, with a signed incident making the case demand a remediation proof before it re-validates (`families.assurance.assurance_regression`); plus a portable certification report an auditor verifies from the bytes (`families.assurance`) | a point-in-time audit / a self-asserted fitness claim |

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
