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
| **BM25 query** @ 20k docs, selective queries | `BM25Index` | `rank_bm25` | **~30–40× faster**, 100% identical top-1 ranking |
| **BM25 query** @ 2k docs | `BM25Index` | `rank_bm25` | **~18–22× faster**, identical ranking |
| **Context assembly** — tokens sent for the same retrieved set | context compiler | LangChain `stuff` / LlamaIndex `compact` | **~60% fewer tokens**, answer retained (scores + dedups + budgets) |
| **Text chunking** a 24k-word doc | `chunk_document` | LangChain / LlamaIndex splitters | **fastest**, and chunks carry provenance the string-splitters don't |
| **Tabular encoding** — tokens for a 50×5 table | `DataEncoder` | `json.dumps` / `pandas.to_markdown` / a TOON reference | **~66% fewer tokens** than `json.dumps`, beats the TOON reference, round-trips losslessly with a typed schema |
| **Token counting** (~60k words) | `HeuristicTokenCounter` | `tiktoken` | **~1.4–1.8× faster**, zero-dependency, conservative (+~25%) |
| **Malformed-JSON recovery** | lenient parser | stdlib `json.loads` | **4/8 vs 1/8** (a dedicated repair lib recovers more, by guessing) |
| **Template render w/ a missing var** | `PromptSpec.substitute` | `jinja2` | raises a typed error vs silently rendering empty |
| **Conciseness** — grounded RAG Q&A expressed end-to-end | `rag("./docs").ask(q)` (1 line) | LCEL / LlamaIndex query engine / DSPy / Haystack | **1 line vs 3–9** — and the one Vincio line also grounds, cites, and runs two evals inline |

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
through Vincio's pipeline — what changes? Like every example and test, it runs
**deterministic on the mock and against a real model when a provider is
configured**:

```bash
python benchmarks/quality_uplift.py                            # deterministic, offline
VINCIO_PROVIDER=openrouter VINCIO_MODEL=openai/gpt-4o-mini \
  OPENROUTER_API_KEY=sk-or-... python benchmarks/quality_uplift.py   # real model
```

Set `VINCIO_UPLIFT_MODELS=a,b,c` to sweep several models and `VINCIO_UPLIFT_RUNS=k`
to repeat each for variance (defaults: the single `VINCIO_MODEL`, 3 runs on a real
provider). Cost is computed from live OpenRouter pricing; a provider error is
reported, never scored as a hallucination.

**Deterministic mechanism metrics** — hold for *any* model because they are
mechanical, measured offline:

| Metric (same model, direct vs. via Vincio) | Direct | Via Vincio |
|---|--|--|
| Schema-valid object from realistic model outputs | 1/6 | **5/6** (structure-only repair) |
| Prompt-injection exfiltration via a tool call | compromised | **contained** (taint + capability token) |
| Context tokens to keep an early fact across a growing chat | grows to 1,267 by turn 160 (needle falls out of a 256-tok window at turn 40) | **flat 33, needle always retained** |

Context-rot curve (deterministic; `window=256` tokens for illustration):

| Turns | Keep-everything buffer tokens | Needle in window? | Vincio recall tokens | Needle retained? |
|--:|--:|:--:|--:|:--:|
| 5 | 51 | ✅ | 33 | ✅ |
| 40 | 327 | ❌ | 33 | ✅ |
| 80 | 640 | ❌ | 33 | ✅ |
| 160 | 1,267 | ❌ | 33 | ✅ |

**Grounded-answer quality, measured on real models** — 15 company-specific policy
questions a model cannot know from pretraining (so the metric isolates the value
of *supplying and enforcing evidence*, not parametric memory). 4 models × 3 runs =
360 live calls (OpenRouter, June 2026); means over runs, stochastic by a point or
two.

*Quality* — fraction correct, and how the model fails when called directly:

| Model — direct → via Vincio | Direct correct | Via correct | Direct hallucinated | Direct abstained | Via cited |
|---|--:|--:|--:|--:|--:|
| `openai/gpt-4o-mini` | 2% | **100%** | 64% | 33% | 100% |
| `anthropic/claude-3-haiku` | 0% | **91%** | 2% | 98% | 100% |
| `google/gemini-2.5-flash-lite` | 4% | **98%** | 29% | 71% | 98% |
| `meta-llama/llama-3.1-8b-instruct` | 2% | **89%** | 40% | 60% | 100% |
| **aggregate** | **2%** | **95%** | — | — | — |

*Efficiency* — tokens, latency, and cost per answer; and the figure that matters,
cost per *correct* answer (µ$ = millionths of a dollar):

| Model | Tokens/ans (direct→via) | Latency ms (direct→via) | µ$/answer (direct→via) | µ$/**correct** answer (direct→via) |
|---|--:|--:|--:|--:|
| `openai/gpt-4o-mini` | 97 → 130 | 1693 → **1408** | 46 → 33 | 2081 → **33** (~62× cheaper) |
| `anthropic/claude-3-haiku` | 123 → 151 | 2495 → **1537** | 130 → 83 | ∞ → **91** (direct never correct) |
| `google/gemini-2.5-flash-lite` | 202 → 136 | 1893 → **1293** | 76 → 25 | 1720 → **26** (~67× cheaper) |
| `meta-llama/llama-3.1-8b-instruct` | 88 → 141 | 1706 → **1561** | 2.3 → 3.2 | 104 → **3.6** (~29× cheaper) |

The honest reading: called directly the model answers ~2% of company-specific
questions correctly (better-aligned models abstain, weaker ones hallucinate up to
64%); the same model through Vincio's retrieval + grounding answers 89–100%, every
answer cited. A direct call is cheaper *per call*, but because it gets almost
nothing right its cost *per correct answer* is **29–67× higher** — undefined for a
model that never answers correctly on its own. Vincio is also **faster per answer**
here (a concise cited reply beats a long wrong guess), and token usage is roughly a
wash (it adds the evidence, but the direct arm often rambles). Rerun the command
above on your own key to reproduce.

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
| **CostBench** | evidence-token reduction from the context compiler; the compact data encoder's tabular token efficiency (`families.cost.table_encoding`) and round-trip losslessness | stuff-everything context; `json.dumps` / Markdown table |
| **RegistryCoverageBench (5.1)** | the model pricing & capability registry held honest — every supported provider's default and capability-heuristic families and every openai_compat preset's headline model resolve to a non-sparse, priced profile (`families.registry_coverage.coverage_complete`), no GA billable model of a paid provider silently bills $0 (`…no_silent_zero`), no price has drifted past the `as_of`-deterministic freshness horizon evaluated against the catalog's release date (`…no_stale_prices`), and the canonical router/cascade picks are unchanged by the refresh — with each gate shown to bite on a deliberately broken catalog | a hand-bumped version string with no enforced horizon; a current model that resolves to nothing and bills $0 |
| **DataPlaneBench (4.2–5.9)** | dataset profiling, sampling & data-quality rails — a table far larger than the window is represented under a fixed token budget (profile + representative sample) with the representation's size invariant to row count (`families.data_plane.fit_in_window`), the bounded-memory profile faithfully recovers a large table's extrema/count/cardinality/central tendency (`families.data_plane.profile`), and the rails catch every seeded defect class — type, range, allowed-set, anomaly, PII — deterministically (`families.data_plane.quality`); **(4.3) governed text-to-query** — execution accuracy on a Spider/BIRD-shaped battery scored by the `SpiderAdapter` / `BIRDAdapter` (the generated query's result set equals the gold's; `families.data_plane.text_to_query.execution_accuracy`), every write / DDL / stacked / injection attempt structurally refused (`…read_only_enforced`), and a result and its cited source cells re-derive from the bytes with a tampered source caught (`…provenance_verifiable`); **(4.4) the data-analysis agent** — task success at budget on DS-1000 / InfiAgent-DABench / DABench-shaped batteries scored by the `DS1000Adapter` / `InfiAgentDABenchAdapter` / `DABenchAdapter` (the offline governed agent's answer matches the gold within its step budget; `families.data_plane.analysis.success_at_budget`), every cell-traceable finding cites its exact source cells (`…narrative_cited`), and the narrative and its cited cells re-derive from the bytes with a tampered source caught (`…verifiable`); **(5.9) notebook-native analysis surface** — the inline reprs of a query result, an analysis, a chart, and a sealed narrative surface the artifact's real, verifiable facts (content hash, exact cell citations, audit id) and never a fabricated one, with a tampered stage flipping the repr's integrity verdict (`families.data_plane.notebook.repr_faithful`), and an interactive register → query → analyze → chart → cite session seals into the same signed, audited narrative a script produces whose `verify()` re-derives every inline finding from the bytes, with a tampered source flipping the verdict (`…session_verifies`); **(5.7) cross-org / federated analytics** — one governed metric run across organizations moves no raw rows across a trust boundary (a per-row sentinel appears in neither the saga journal nor the sealed narrative; `families.data_plane.federated_analytics.rows_never_cross`), every reconciled finding re-derives from each org's content-hashed source and the reconciled totals equal the brute-force totals over the pooled rows with a tampered reconciliation caught (`…federated_data_binding`), and residency egress refusal / the consent ledger / the differential-privacy budget / the k-anonymity contributor floor each refuse a non-compliant round (`…governance_preservation`) | `json.dumps` all rows / compact all-rows encoding / `pandas.describe` (numeric-only, no sample); Spider/BIRD execution accuracy vs string match; a shared warehouse that pools raw rows |
| **SecurityBench** | injection detection rate, false-positive rate, PII coverage | — |
| **EvalBench** | metric agreement on labeled examples; red-team judging (guarded vs naive target, detector coverage); synthetic-data determinism and coverage; A/B significance machinery; session grouping; HTML viewer self-containment; trace→dataset; G-Eval calibration | an unguarded target where attacks succeed |
| **AgenticEvalsBench (1.2)** | trajectory & tool-use metric agreement with labeled traces; trajectory eval flags runs that output-only eval passes; user-simulator determinism; drift sensitivity/specificity; Cohen's-κ judge-agreement tracking; (2.2) stateful-environment task-success oracle (verifies the end state, rejects policy violations) + deterministic replay of the nine benchmark adapters with hash-pinned task sets; judge ensembles whose disagreement is an uncertainty signal and whose calibration is κ-gated, Shapley causal regression attribution, and verdict-preserving adaptive sampling (`families.agentic_evals.environment_eval` / `families.agentic_evals.quality_frontier`) | output-only evaluation / no drift detector / un-attributed regressions |
| **LoopBench (0.8)** | the closed loop end to end: promotion fires, decisions are deterministic, gates block regressions, the registry version is tagged + eval-linked; grounded auto-memory precision (grounded written, ungrounded excluded); retrieval-feedback improvement + gating; Pareto frontier correctness (dominated excluded, knee balanced); learned-budget promotion; guided-search budget bounds | ungated / single-score optimization |
| **ProtocolsBench (1.1)** | MCP tool schema-fidelity + round-trip through the permissioned runtime + resource provenance (`origin: mcp:<server>`); A2A budget-bounded crew delegation terminates; Agent-Skill progressive-disclosure savings (off-topic bodies stay out of budget, index always present); (2.2) the governed agent fabric — allow-list-gated + audited resolution, capability discovery, AGNTCY/ACP roundtrip + discovery, and MCP-registry discovery with unlisted servers denied (`families.protocols.fabric`) | thin protocol adapter (no permissions/provenance/budget) |
| **GovernanceBench (1.6)** | model/system-card and AI-BOM completeness; compliance-framework mapping coverage (OWASP LLM 2025 / OWASP Agentic / NIST AI RMF / MITRE ATLAS / ISO IEC 42001); erasure-by-source correctness (chunks removed = lineage) + audit; multilingual PII recall + English-path intactness; RAG-poisoning detection rate + false-positive rate; residency endpoint-region inference + jurisdiction matching; signed-manifest verification (tamper/wrong-key fail closed) | English-only / ungoverned |
| **GenerationBench (1.9)** | document-contract validity (valid passes, deficient rejected — no invention); cited-report citation coverage + per-claim entailment + unresolved-marker detection; media C2PA provenance binding (image + audio) with tamper rejection and synthetic-content disclosure; redline correctness; new-format ingestion recall; generated-media prompt safety | un-contracted / un-provenanced output |
| **PerfBench** | compile/retrieval/run latency (p50/p95/p99), cache speedups, concurrent throughput, streaming TTFT; the compile hot-path families — vectorized-scoring equivalence, render-program byte-identity, warm-candidate-arena equivalence, streaming-first compilation, speculative-prefetch warm, a sub-millisecond warm compile, and a resident-footprint regression gate; and the single-pass feature arena — selection byte-identity (`single_pass.selection_byte_identical`), large-pool selection equivalence (`vectorized_selection.equivalent`), bounded BM25 top-k identity (`retrieval.topk_identical`), and a `single_pass.compile_speedup` ratio floor so an erased win fails the build (`families.perf.{vectorized_scoring,single_pass,vectorized_selection,render_program,warm_arena,streaming_compile,prefetch,footprint}`) | cold (uncached) paths |
| **LongHorizonBench (3.10)** | long-horizon context engineering — at 10× horizon the `ContextGovernor` holds resident/token footprint flat (vs ~linear naïve growth), preserves recall by paging a compacted needle back from the content-addressed store, retains provenance through hierarchical compaction, and demotes stale spans via intra-run decay (`families.long_horizon`) | naïve accumulation (linear footprint, context rot) |
| **WorldModelBench (3.11)** | world-model / simulation-based planning — a `WorldModel` fit offline from recorded transitions predicts next states accurately, learns a precondition (refund fails on a processing order, succeeds on a cancelled one) and generalizes over arguments, earns planning weight only once calibrated, and lets the `ModelPredictivePlanner` open the vault while a reactive one-step planner is trapped at a fixed action budget (`families.world_model`) | reactive (one-step) planning |
| **SemanticCacheBench (3.13)** | learned semantic cache & near-miss KV reuse — the acceptance threshold is calibrated from labelled trace pairs (precision-targeted, never serving below the bar), an accepted near-miss served through the run path is at-least-as-good as the live answer at a fixed budget, a below-bar match is never served, the eval-replay gate passes a faithful cache and blocks a drifted one, cross-request KV-prefix reuse reports the shared head's avoided recompute, and the cache stays inside the resident-memory budget (`families.semantic_cache`) | exact-match-only caching (no near-miss reuse) |
| **IntegrationsBench** | ecosystem & integration breadth — every first-party connector (Jira, Linear, Google Drive, SharePoint, Salesforce, Zendesk, BigQuery, Snowflake) round-trips offline against a recorded fixture with full provenance; the entry-point plugin contract loads a compatible plugin on install and gates an incompatible-major one; the signed community registry resolves a bundle under the allow-list (audited), verifies its signature, and denies a tampered one; the Haystack and DSPy bridges adapt assets in; and the MCP-server marketplace bridge discovers → governs (audited) → lands a server's tools in the permissioned runtime in one call, denying an unlisted server (`families.integrations`) | hand-wired loaders / ungoverned, unsigned registries |
| **EdgeBench (3.21)** | edge / WASM in-process runtime — an edge compile is byte-identical to a direct server compile over the same inputs (parity, not a fork) and the runtime delegates to the canonical compiler/rail-engine; the bounded edge profile holds the resident footprint under its cap as the candidate corpus grows 10× (eviction fires under load); the compile/score/rail/pack path imports nothing native unconditionally (WASM-buildable); the deterministic rails refuse a secret leaking from evidence into the rendered context; and it runs offline with no provider, store, or network (`families.edge`) | a forked/diverging edge build / unbounded footprint |
| **MCPAppsBench (3.22)** | MCP Apps & the evolving MCP spec — a consumed server's `ui://` UI resource is surfaced through the existing AG-UI channel as an `mcp.ui` event under the run's governance (untrusted-external provenance, token-metered so an oversized render is refused not streamed, audited); a server's mid-call elicitation is gated by the same approval + rail machinery a write tool passes, so a secret value is refused and an accepted value is tainted untrusted (contained end-to-end through a server tool); and evolving-spec parity — protocol-version negotiation honours a peer pinned to an older stable revision (`families.mcp_apps`) | an ungoverned UI side-channel / a trusted elicited value |
| **NegotiationBench (3.23)** | agent negotiation & contracting — a bounded offer/counter bargain over typed price/SLA/scope terms always **terminates** within its round/deadline budget (a deal when the parties' acceptable regions overlap, a clean no-deal when they do not, a partial result on a wall-clock deadline); the agreed **contract** is signed by both parties, verifies offline from the bytes alone, catches a tampered term, and enforces like a budget (`to_budget` / `check`); the counterparty's reputation **discounts** a regressing agent's offers without singling it out and selects the reputation-weighted best deal; and the bargain runs over the A2A fabric byte-for-byte the same as locally, every outcome on the audit chain (`families.negotiation`) | an unbounded haggle / a self-asserted, unverifiable agreement |
| **AssuranceBench (3.49)** | continuous assurance cases & production certification — `app.assurance_case` assembles the evidence the platform already emits (eval gates, the governance verifier, reasoning certificates, the signed audit chain, identity/delegation provenance, SBOM/SLSA) into a content-bound argument tree, and **no claim stands on missing or stale evidence**: a fully-discharged case holds and verifies offline while a leaf whose evidence is absent, expired past its freshness horizon, or falsified is pinpointed and the case fails (`families.assurance.assurance_soundness`); the case is re-checked on every change and `assurance_regression_gate` turns a previously-discharged claim now falsified into a build failure, with a signed incident making the case demand a remediation proof before it re-validates (`families.assurance.assurance_regression`); plus a portable certification report an auditor verifies from the bytes (`families.assurance`) | a point-in-time audit / a self-asserted fitness claim |
| **ErgonomicsBench (5.3)** | the ergonomic 'ad-hoc' front door (`vincio.tasks`) made real, faithful, and uncapped — **conciseness** (each of the five jobs plus the fluent `Flow` is expressible through one entry point: `rag` / `extractor` / `tool_agent` / `evaluation` / `chat` / `Flow`, where the verbose path is a fistful of builder calls; benchmarked head-to-head against LCEL, the LlamaIndex query engine, DSPy, and Haystack in `competitive.py`) (`families.ergonomics.conciseness_one_entry_point`), **compiles-byte-identical** (the one-liner lowers to the same governed `ContextApp.run` packet and `RunResult` as the verbose six-call form — proven with the shared `vincio.testing.run_signature` harness, the same one the single-pass feature arena uses) (`families.ergonomics.compiles_byte_identical`), and **escape-hatch-total** (`.app` reaches every deep method; nothing shadowed or unreachable) (`families.ergonomics.escape_hatch_total`) | a fistful of string-keyed builder calls (the verbose `ContextApp` path) |
| **DocsConformanceBench (5.4)** | the documentation as one connected graph (mirroring `cross_org_conformance` / `data_analysis_conformance`) — **link integrity** (every internal docs link resolves, path and anchor, across all authored pages; `families.docs_conformance.docs_link_integrity`), **capability-map coverage** (every public `app.*` verb is placed in the generated capability map and documented in `api.md`, and every concept reaches a guide + example + reference anchor; `…docs_capability_map_coverage`), and **navigation reachability** (every concept and guide carries a current single-sourced Related block, the generated pages are current, and no page is orphaned; `…docs_navigation_reachability`); companion budgets gate that every concept is connected, that `llms.txt` is regenerated from `vincio.__all__` and current, and that the gate **bites** — a synthetic broken link and an unmapped verb are each caught (`…docs_gate_detects_tamper`) | a substring check that a subsystem name appears *somewhere* in the docs |
| **HygieneBench (6.0–6.3)** | the public surface kept consistent, every public raise typed, every best-effort failure observable, and every public capability reachable (mirroring `docs_conformance` / `registry_coverage`). **Surface (6.0):** **resolvable** (every public subpackage `__all__` name is a live attribute, with no duplicate or malformed entries — no dead surface; `families.hygiene.surface_dead_symbol_free`), **frozen** (the classified surface matches the committed `docs/reference/subpackage-surface.txt`, so any `__all__` change is a reviewed edit; `…surface_frozen`), and **gate bites** (an injected dead symbol, a duplicate, and a malformed `__all__` are each caught; `…surface_gate_detects_tamper`), folded into one headline `…surface_consistency`. **Error contract (6.1):** every error raised on a public entry point derives from `VincioError` — the `ContextApp` (`app.*` verb) surface raises no bare built-in (`…error_contract_app_verbs_clean`), the classified baseline of accepted public built-in raises matches the committed `docs/reference/error-contract.txt` (`…error_contract_frozen`), and the detector provably catches an injected public built-in raise while ignoring an encapsulated one (`…error_contract_gate_detects_tamper`), folded into one headline `…error_contract_conformant`. **Observable failure (6.2):** no public module swallows a broad exception silently — every broad `except` (or `contextlib.suppress(Exception)`) re-raises, records its failure (a logger call or `note_suppressed`), or carries a justifying `# noqa: BLE001` (`…observable_failure_clean`), and the detector provably flags an injected silent swallow while ignoring a logged one (`…observable_failure_gate_detects_tamper`), folded into one headline `…observable_failure_conformant`. **Wire-or-retire (6.3):** every formerly-unhooked public capability is reachable through a production path — each entry in a frozen ledger resolves to a live reach (an `app.*` verb, an engine method, a registration helper, or a public class member) and, for a wired one, is referenced by production code outside its defining module (`…wire_or_retire_clean`), and the detector provably bites on an unreachable reach and a wired symbol with no production caller (`…wire_or_retire_gate_detects_tamper`), folded into one headline `…wire_or_retire_conformant`; companion budgets gate the count of audited subpackages, the wired-capability count, and that the silent-swallow count is zero | a name in `__all__` that resolves to nothing reads as supported API while doing nothing; a bare `ValueError`/`KeyError` leaking off a public verb breaks `except VincioError` silently; a broad `except` that swallows its exception with no log or metric hides a real bug; a public capability with no `app.*` verb, example, or caller reads as supported API while being unreachable |

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

Three honesty rules this suite holds itself to:

1. **VincioBench numbers are mechanism checks, not performance claims.** Its
   corpus is small and synthetic, built to exercise each engine, so its scores
   *saturate* — e.g. recall@3 = 1.00 and 100% injection detection are perfect
   scores on a handful of cases (retrieval n=5, the bundled attack corpus n=6)
   against a naive in-house baseline. They prove the mechanism is intact and
   guard against regressions; they are **not** evidence of real-world quality.
   The load-bearing performance evidence is `competitive.py` (real libraries) and
   `quality_uplift.py` (real models). Context compression measures ~81% against a
   *stuff-everything strawman*; the figure to quote is the ~60% reduction vs. real
   assemblers (LangChain/LlamaIndex) in the competitive suite — the 20–40% design
   target is a conservative floor, not a ceiling.
2. **Competitive ratios vary by machine and run** (BM25 ~30–40×, token counting
   ~1.4–1.8× across runs here). The README quotes ranges; rerun on your hardware.
3. **Real-model rows are a dated external run** (OpenRouter, June 2026) — they are
   not reproducible from the bundled offline benchmarks, only from a live
   provider key. The offline harness ships a deterministic illustration.

Do not market numbers beyond what a benchmark run on your own data shows.
