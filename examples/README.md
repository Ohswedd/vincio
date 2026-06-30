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

Five **Google Colab-ready** notebooks: one `pip install`, no setup, offline by default. See
[`notebooks/`](notebooks/) for all five (quickstart, RAG, agents & tools, evaluation, data analysis),
each with an *Open in Colab* badge.

## 2. Feature tours — one program per subsystem

Twenty-three complete, heavily-commented programs that exercise the whole platform. Each file is a
guided tour: a module docstring states what it teaches, and numbered sections each demonstrate one
capability and print a concrete result. Read them top to bottom.

| # | Example | What it teaches |
|--|---|---|
| 00 | [`one_liners`](00_one_liners.py) | The ergonomic 'ad-hoc' front door (`vincio.tasks`): grounded RAG (`rag`), typed extraction (`extractor`), an approval-gated tool agent (`tool_agent`), an eval (`evaluation`), a chat (`chat`), and the fluent immutable `Flow` — each one expression that **lowers to the same governed `ContextApp.run` packet** as the verbose form (proven byte-identical), with `.app` as the escape hatch. |
| 01 | [`quickstart`](01_quickstart.py) | The five-minute tour — `ContextApp`, typed Pydantic output, grounded QA with citations, the trace_id + cost on every result, and a short multi-turn chat. |
| 02 | [`retrieval_rag`](02_retrieval_rag.py) | Hybrid retrieval (BM25 + dense + learned-sparse + late-interaction fused), query understanding (HyDE / multi-query / decomposition), chunking, GraphRAG, metadata filters, embedders, and multimodal (image/table/video) evidence. |
| 03 | [`memory`](03_memory.py) | Scoped `remember`/`recall`, the guarded write pipeline, confidence decay, contradiction resolution, bi-temporal recall, per-memory ACLs, episodic→semantic consolidation, and audited GDPR edit/forget/export/erase. |
| 04 | [`agents_and_tools`](04_agents_and_tools.py) | A permissioned tool registry (RBAC/ABAC), the resource-limited sandbox, approval-gated writes, planners (ReAct / HTN) with in-place plan repair, cost-aware action selection, the deep-research agent, and the computer-use action plane. |
| 05 | [`orchestration`](05_orchestration.py) | Multi-agent crews with a shared blackboard, durable stateful graphs (checkpoint / resume / time-travel / human-in-the-loop), deterministic workflows, and the distributed durable-execution backend. |
| 06 | [`structured_output`](06_structured_output.py) | Pydantic contracts, constrained decoding, streaming validation with early abort, bounded self-correction (structure only, never invents facts), multi-schema routing, and typed signatures. |
| 07 | [`evaluation_observability`](07_evaluation_observability.py) | Golden datasets, metrics, deterministic/model/G-Eval judges, synthetic data, red-teaming, regression gates, trajectory & agentic eval, drift detection, trace span trees, and the prompt registry. |
| 08 | [`optimization_self_improvement`](08_optimization_self_improvement.py) | The closed loop (trace → dataset → eval → optimize → promote), the reflective optimizer, the distillation flywheel, RLVR, gated canary deploy, on-device LoRA, and federated learning with a privacy accountant. |
| 09 | [`security_governance`](09_security_governance.py) | PII/secret redaction, injection defense and provable containment, RBAC/ABAC, the hash-chained audit log, governance evidence (cards / AI-BOM / erasure / consent / residency), formal verification, agent identity, verified reasoning, and assurance cases. |
| 10 | [`interop_and_protocols`](10_interop_and_protocols.py) | MCP client *and* server, A2A agent-to-agent delegation, Agent Skills, framework interop (LangChain / LlamaIndex / Haystack / DSPy), first-party connectors, and vertical packs. |
| 11 | [`advanced_context`](11_advanced_context.py) | Reasoning control, test-time compute (best-of-N / self-consistency / beam), the long-horizon context governor, world-model planning, the learned semantic cache, the record-replay debugger, and energy/carbon accounting. |
| 12 | [`cross_org_economy`](12_cross_org_economy.py) | One narrative of two organizations transacting through agents: negotiate a signed contract, run a durable compensating saga, meter and settle, arbitrate a dispute, prove solvency, and resolve an insolvency. |
| 13 | [`tabular_evidence`](13_tabular_evidence.py) | Structured data as first-class evidence: a typed, columnar `Dataset`, the lossless `DataEncoder` that renders it header-once and token-cheap, columnar-accurate token accounting, and `TableEvidence` scored and cited by the context compiler. |
| 14 | [`dataset_profiling`](14_dataset_profiling.py) | Fitting a table far larger than the window into bounded, faithful evidence: `profile_dataset` for a fixed-size column profile, reservoir/stratified `sample_dataset`, `fit_to_window` under a fixed token budget, and `DataQualityRails` screening for schema/constraint/anomaly/PII defects. |
| 15 | [`governed_text_to_query`](15_governed_text_to_query.py) | Turning a question over a registered dataset into a schema-grounded, read-only-verified, cost-bounded query (`app.query_data`): the structural read-only guard refusing a write / DDL / injection, cell-level provenance (`result.cite_refs`) pointing at the exact source cells, offline `verify()` that catches a tampered source, and the deterministic dataframe-op dialect. |
| 16 | [`data_analysis_agent`](16_data_analysis_agent.py) | A bounded, multi-step analysis over a dataset (`analyze_dataset` / `app.analyze_data`): the agent plans, queries through the read-only-verified query plane, inspects, and drills into the dominant group, producing a cited analytical narrative whose every finding points at the exact source cells and whose `verify(catalog)` re-derives the whole analysis from the bytes — bounded by an explicit `AnalysisBudget`, audited, and refusing an injection-bearing objective. |
| 17 | [`charts_cited_artifacts`](17_charts_cited_artifacts.py) | Turning a cited query result into an analytical artifact (`generate_chart` / `app.generate_chart`): a spec-driven chart (a portable Vega-Lite v5 spec by default, a matplotlib PNG behind `vincio[charts]`) that is **content-bound** — a C2PA data-driven credential bound to its rendered bytes — and **data-bound** — a back-reference to the exact source cells (`sales#r0!revenue`) that `verify(catalog)` re-derives offline. The cited-report builder extends to figures, so a `Figure` embeds a chart or a table into a deliverable that is per-claim entailed *and* per-figure data-bound. |
| 18 | [`streaming_out_of_core`](18_streaming_out_of_core.py) | Processing a dataset far larger than memory in bounded passes (`RowStream` / `stream_aggregate` / `encode_stream` / `app.map_stream`): a lazy, re-iterable, schema-bearing stream iterated in bounded chunks and profiled in one pass; a deterministic group-by whose working set tracks the number of groups, not rows; the compact encoder applied header-once and gzip-compressed; the context compiler's streaming candidate pre-filter bounding a 10k+ evidence pool before full scoring; and a per-chunk transform dispatched at scale through the `BatchRunner`. |
| 19 | [`semantic_layer_governed_metrics`](19_semantic_layer_governed_metrics.py) | Defining measures, dimensions, and derived columns once with a `SemanticLayer` so a question maps to a **governed metric** (`app.query_metric`) compiled to one canonical read-only `SELECT` and computed the same way however it is phrased: a derived `revenue = price × qty`, a ratio `avg_order_value`, cell-level citations, `MetricResult.verify` proving the number is the governed one (an ad-hoc query is rejected), column-level `app.metric_lineage` resolving to base columns and source, and a right-to-erasure sweep (`app.erase_source`) reaching the dataset plane. |
| 20 | [`data_engagement`](20_data_engagement.py) | The data & analytics **capstone**: `app.data_engagement` threads the whole plane — register → profile → sample → screen → query → analyze → chart → governed metric → cite — behind one governed, audited call-path and seals it into a hash-chained, signed `DataNarrative`. The narrative `verify()`s offline from the bytes alone; given the live catalog, every captured query, analysis, chart, and metric **re-derives from the content-hashed source** (data-binding); a re-ordered stage, a forged signature, or a tampered source is caught; and every primitive stays usable on its own. The analytics analogue of the cross-org engagement (example 12). |
| 21 | [`model_pricing_registry`](21_model_pricing_registry.py) | The model pricing & capability registry made honest: the shipped `model_catalog.json` prices the real current lineup of every provider (OpenAI o-series / gpt-4.1, the openai_compat presets, …) so no current model silently bills $0; `priced_as_of` plus an `as_of`-deterministic freshness horizon evaluated against the *release* date (never the wall clock); `registry.coverage_report()` proving every provider default and capability-heuristic family resolves to a non-sparse priced profile with no silent $0 and no routing drift; the gates shown to bite on a broken catalog; and the review-only `vincio registry sync` diffing a provider's live model list into a candidate overlay without ever mutating the catalog. |
| 22 | [`connected_docs`](22_connected_docs.py) | The docs as one connected, verifiable graph: the **capability map** binding every public `app.*` verb to the concept, guide, example, and reference that document it (grouped by the six capability facades); the single-sourced **Related** cross-link block that lands on every concept and guide; the staged **learning path**; the **docs-graph check** (`vincio docs check`) — link integrity, capability-map coverage, navigation reachability, no orphans, and `llms.txt` freshness — that the `docs_conformance` VincioBench family gates; the gate shown to bite on a synthetic broken link and an unmapped verb; and `llms.txt` regenerated from `vincio.__all__`. All deterministic and dependency-free. |
| 23 | [`realtime_streaming_analytics`](23_realtime_streaming_analytics.py) | The profiling, query, governed-metric, and quality primitives re-expressed over an **unbounded event stream** with `StreamWindow` (`tumbling` / `sliding` / `session`): windowed `query` / `profile` / `query_metric` / `screen` / `aggregate` emitting one result per closed window, each citing the exact source **events** (`stream@<offset>`) it rests on and `verify()`-ing offline against its bounded captured window; memory shown to stay invariant as the stream grows 100×; and `app.stream_analytics` auditing every window and driving a **live** async feed (an alerting rule) exactly as it replays a log. |
| 24 | [`federated_analytics`](24_federated_analytics.py) | A governed metric run **across organizations** with `app.federated_data_engagement` — no shared warehouse. A `FederatedQuery` is negotiated as a `Contract`, choreographed as a `Saga` whose steps run each org's governed query plane **locally** and return only the aggregated, cell-cited `MetricResult` (the raw per-row data never crosses the trust boundary), reconciled into one signed, hash-chained `FederatedNarrative` whose every `FederatedFinding` re-derives from each org's content-hashed source; with residency egress refusal, the consent ledger's analytics purpose, the differential-privacy budget, and a k-anonymity contributor floor refusing a non-compliant round, and a tamper anywhere — chain, signature, or reconciliation — caught offline. The cross-org analogue of the data engagement (example 20). |

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
