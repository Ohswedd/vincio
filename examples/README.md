# Vincio examples

A three-tier on-ramp — start in the browser, learn each subsystem, then copy a real backend. Every
tier runs **fully offline** on the deterministic mock provider (no API keys, no network), and points
at a real model with one environment variable. Each tier is gated in CI so it can never drift.

```bash
cd examples
python 00_one_liners.py                 # the one-line front door
python 01_quickstart.py                 # then the five-minute tour

# Point any example at a real model instead of the mock:
export VINCIO_PROVIDER=openai VINCIO_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-...
python 02_retrieval_rag.py
```

## 1. Notebooks — start in the browser

Six **Google Colab-ready** notebooks: one `pip install`, no setup, offline by default. See
[`notebooks/`](notebooks/) for all six (quickstart, RAG, agents & tools, evaluation, data analysis,
notebook-native analysis), each with an *Open in Colab* badge.

## 2. Feature tours — one program per subsystem

Twenty-two focused programs (00–21), each a tight **syntax-explainer + best-practice** walkthrough of
one subsystem: a short module docstring states what it teaches, numbered inline comments explain *what
each API does and when/why you'd reach for it* (with the gotchas), and one meaningful print shows the
result that matters. Read them top to bottom, or jump to the one you need.

| # | Example | What it teaches |
|--|---|---|
| 00 | [`one_liners`](00_one_liners.py) | The ergonomic 'ad-hoc' front door (`vincio.tasks`): grounded RAG (`rag`), typed extraction (`extractor`), an approval-gated tool agent (`tool_agent`), an eval (`evaluation`), a chat (`chat`), and the fluent immutable `Flow` — each one expression that **lowers to the same governed `ContextApp.run` packet** as the verbose form (proven byte-identical), with `.app` as the escape hatch. |
| 01 | [`quickstart`](01_quickstart.py) | The five-minute tour — `ContextApp`, typed Pydantic output, grounded QA with citations, the trace_id + cost on every result, and a short multi-turn chat. |
| 02 | [`retrieval_rag`](02_retrieval_rag.py) | Hybrid retrieval (BM25 + dense + learned-sparse + late-interaction fused), query understanding (HyDE / multi-query / decomposition), chunking, GraphRAG, metadata filters, embedders, multimodal (image/table/video) evidence, and fact-grounded reasoning retrieval (`app.retrieve_facts`) that reports coverage gaps. |
| 03 | [`memory`](03_memory.py) | Scoped `remember`/`recall`, the guarded write pipeline, confidence decay, contradiction resolution, bi-temporal recall, per-memory ACLs, episodic→semantic consolidation, and audited GDPR edit/forget/export/erase. |
| 04 | [`agents_and_tools`](04_agents_and_tools.py) | A permissioned tool registry (RBAC/ABAC), the resource-limited sandbox, approval-gated writes, planners (ReAct / HTN) with in-place plan repair, cost-aware action selection, the deep-research agent, and the computer-use action plane. |
| 05 | [`orchestration`](05_orchestration.py) | Multi-agent crews with a shared blackboard, durable stateful graphs (checkpoint / resume / time-travel / human-in-the-loop), deterministic workflows, and the distributed durable-execution backend. |
| 06 | [`structured_output`](06_structured_output.py) | Pydantic contracts, constrained decoding, streaming validation with early abort, bounded self-correction (structure only, never invents facts), multi-schema routing, and typed signatures. |
| 07 | [`evaluation_observability`](07_evaluation_observability.py) | Golden datasets, metrics, deterministic/model/G-Eval judges, synthetic data, red-teaming, regression gates, trajectory & agentic eval, drift detection, trace span trees, and the prompt registry. |
| 08 | [`optimization_self_improvement`](08_optimization_self_improvement.py) | The closed loop, shown deeply on four capabilities: trace → dataset → eval → optimize → promote, the distillation flywheel, RLVR from verifiable rewards, and on-device LoRA adaptation — with a closing pointer to the rest (reflective GEPA/MIPRO, canary deploy, federated learning). |
| 09 | [`security_governance`](09_security_governance.py) | Deterministic security & governance, four capabilities deeply: PII/secret redaction, **provable** injection containment (taint + capability tokens), the hash-chained audit log verified offline, and an assurance case / certification — pointing to the rest (RBAC/ABAC, cards, AI-BOM, residency, verified reasoning). |
| 10 | [`interop_and_protocols`](10_interop_and_protocols.py) | MCP client *and* server, A2A agent-to-agent delegation, Agent Skills, framework interop (LangChain / LlamaIndex / Haystack / DSPy), first-party connectors, and vertical packs. |
| 11 | [`advanced_context`](11_advanced_context.py) | The advanced context runtime, four capabilities deeply: reasoning control, test-time compute (best-of-N / self-consistency), the long-horizon context governor, and the learned semantic cache — with a pointer to the rest (world-model planning, record-replay, energy accounting, the compile hot path). |
| 12 | [`cross_org_economy`](12_cross_org_economy.py) | One narrative of two organizations transacting through agents: negotiate a signed contract, run a durable compensating saga, meter and settle, arbitrate a dispute, prove solvency, and resolve an insolvency. |
| 13 | [`data_and_analytics`](13_data_and_analytics.py) | The data & analytics plane through the five capabilities that carry it, each deeply: tabular evidence & the compact header-once encoder; governed text-to-query with cell-level provenance (read-only guard + tamper detection); certified statistical claims recomputed from cited cells (causation refused); the data-engagement capstone (one signed, hash-chained, data-bound `DataNarrative`); and cross-org federated analytics (only aggregates cross — never the raw rows). A closing note points to the guide/concepts for profiling, charts, streaming, the semantic layer, and real-time windows. |
| 14 | [`model_pricing_registry`](14_model_pricing_registry.py) | The data-driven `ModelRegistry`: the shipped `model_catalog.json` prices the real current lineup of every provider, `priced_as_of` + an `as_of`-deterministic freshness horizon, `registry.coverage_report()` proving no silent $0 / no routing drift, and the review-only `vincio registry sync`. |
| 15 | [`connected_docs`](15_connected_docs.py) | The docs as one connected, verifiable graph: the capability map binding every `app.*` verb to its concept / guide / example / reference, the single-sourced Related blocks, the staged learning path, and the `vincio docs check` docs-graph gate. |
| 16 | [`open_evaluation_plane`](16_open_evaluation_plane.py) | The open evaluation plane: the standard public benchmarks (MMLU, GPQA, GSM8K, HumanEval, IFEval, TruthfulQA, RULER, …) grouped by niche behind one `BenchmarkAdapter` contract, with a **provenance tier** (Static / Recorded / Live) on every number that the engine refuses to let a lower tier inflate; deterministic Tier-S runs, the measured long-context governor uplift, Markdown / HTML / JSON / CSV reports, a ranked `Leaderboard` with charts, a SQLite `RunStore` with model-version diffs, and registering your own benchmark without touching the core. |
| 17 | [`compile_receipt`](17_compile_receipt.py) | The packet compile receipt: a compact, text-light manifest of *why* a context packet was compiled — inclusions, exclusions/supersessions, per-item scores, budget, privacy posture, and conflict winners, linked from the run trace. Read the receipt off a run, inspect the decision, verify it re-derives from its own bytes and carries no raw text, and diff a changed compile against a baseline as an explicit divergence. |
| 18 | [`ds4_local_inference`](18_ds4_local_inference.py) | DS4 local inference: a self-hosted DeepSeek V4 box (antirez's `ds4-server`) as a first-class provider. Runs offline by replaying a recorded DS4 exchange through the real `Ds4Provider` — chat with the thinking trace kept out of the answer, streaming with disk-KV usage, thinking modes driven by the reasoning controller (on with a budget, off when plain), fail-closed on-prem residency, and an honest self-hosted `$0` in the catalog with the coverage gate green. Point it at a real box with `VINCIO_PROVIDER=ds4`. |
| 19 | [`web_browser_search`](19_web_browser_search.py) | Universal web browsing & search: `app.use_web_search()` gives **any** model — native tool-calling or not — the same governed `web_search` / `web_read` tools, executed by Vincio against DuckDuckGo (offline here via an injected transport and the static backend). A page is read token-efficiently (only the passages relevant to the model's own query, ~74x cheaper than the full page), a local model without function calling runs the identical loop through the text protocol, private-host and denied-domain fetches are refused pre-egress, every search/read lands on the audit trail, and the whole session re-derives offline from its content-hashed snapshots. |
| 20 | [`context_anchors`](20_context_anchors.py) | Context anchors — keep a PRD / spec / brand frame across a whole coding task: mark a source `anchor=True` and Vincio distills it **once** into a compact, constraint-first, content-hash-cached brief injected as **pinned** evidence into every call, so the global frame is always present (even on a query that never mentions it, and under a tiny token window) at a flat few-hundred-token cost — ~26x cheaper than re-pasting the corpus — while on-demand detail still flows through normal retrieval. |
| 21 | [`lager_reasoning_retrieval`](21_lager_reasoning_retrieval.py) | LAGER — reasoning-driven retrieval: documents become byte-exact, offline-verifiable **Evidence Objects** in a typed knowledge graph, and retrieval runs as a **lazy** needs-driven loop (one round for an easy query, graph hops across a zero-lexical-overlap bridge for a why-question, honest abstention with uncovered needs named for an impossible one) instead of fixed top-k chunks; `app.use_lager()` swaps the loop into every run on the same screened, compiled, cited pipeline. |

`_shared.py` holds the small offline helpers every example imports (`example_provider`,
`json_responder`, `citing_responder`, `write_sample_docs`).

## 3. Applications — real-world backends

Small, production-shaped apps you can copy as a starting point, in
[`applications/`](applications/): a FastAPI **grounded-RAG service**, a **ticket-triage API** (typed
output + scoped memory + an approval-gated tool), a **structured-extraction service** (self-correcting),
and a no-framework **CLI research agent**. Each FastAPI app splits an offline-testable `core.py` from a
thin FastAPI `main.py`, so the Vincio logic runs with no web framework installed.

```bash
pip install "vincio[server]"
cd examples/applications/rag_service && uvicorn main:app --reload
```
