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

Sixteen complete, heavily-commented programs (00–15) that exercise the whole platform. Each file is a
guided tour: a module docstring states what it teaches, and numbered sections each demonstrate one
capability and print a concrete result. Read them top to bottom.

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
| 08 | [`optimization_self_improvement`](08_optimization_self_improvement.py) | The closed loop (trace → dataset → eval → optimize → promote), the reflective optimizer, the distillation flywheel, RLVR, gated canary deploy, on-device LoRA, and federated learning with a privacy accountant. |
| 09 | [`security_governance`](09_security_governance.py) | PII/secret redaction, injection defense and provable containment, RBAC/ABAC, the hash-chained audit log, governance evidence (cards / AI-BOM / erasure / consent / residency), formal verification, agent identity, verified reasoning, and assurance cases. |
| 10 | [`interop_and_protocols`](10_interop_and_protocols.py) | MCP client *and* server, A2A agent-to-agent delegation, Agent Skills, framework interop (LangChain / LlamaIndex / Haystack / DSPy), first-party connectors, and vertical packs. |
| 11 | [`advanced_context`](11_advanced_context.py) | Reasoning control, test-time compute (best-of-N / self-consistency / beam), the long-horizon context governor (with blob-backed cross-process cold-span paging), world-model planning, the learned semantic cache, the record-replay debugger, energy/carbon accounting, and the compile hot path (cache, warm arena, streaming compile, and partial recompile). |
| 12 | [`cross_org_economy`](12_cross_org_economy.py) | One narrative of two organizations transacting through agents: negotiate a signed contract, run a durable compensating saga, meter and settle, arbitrate a dispute, prove solvency, and resolve an insolvency. |
| 13 | [`data_and_analytics`](13_data_and_analytics.py) | The whole data & analytics plane in one tour (eleven numbered sections): tabular evidence & the compact encoder; profiling / sampling / fit-in-window & quality rails; governed text-to-query with cell-level provenance; the bounded multi-step analysis agent; content- & data-bound charts and cited reports; streaming & out-of-core bulk processing; the semantic layer & governed metrics; the data-engagement capstone (one signed, hash-chained `DataNarrative`); real-time windowed analytics over an unbounded event stream; cross-org / federated analytics (the raw rows never cross the trust boundary); and certified statistical claims (trend / correlation / interval / forecast, with causation refused). |
| 14 | [`model_pricing_registry`](14_model_pricing_registry.py) | The data-driven `ModelRegistry`: the shipped `model_catalog.json` prices the real current lineup of every provider, `priced_as_of` + an `as_of`-deterministic freshness horizon, `registry.coverage_report()` proving no silent $0 / no routing drift, and the review-only `vincio registry sync`. |
| 15 | [`connected_docs`](15_connected_docs.py) | The docs as one connected, verifiable graph: the capability map binding every `app.*` verb to its concept / guide / example / reference, the single-sourced Related blocks, the staged learning path, and the `vincio docs check` docs-graph gate. |
| 16 | [`open_evaluation_plane`](16_open_evaluation_plane.py) | The open evaluation plane: the standard public benchmarks (MMLU, GPQA, GSM8K, HumanEval, IFEval, TruthfulQA, RULER, …) grouped by niche behind one `BenchmarkAdapter` contract, with a **provenance tier** (Static / Recorded / Live) on every number that the engine refuses to let a lower tier inflate; deterministic Tier-S runs, the measured long-context governor uplift, Markdown / HTML / JSON / CSV reports, a ranked `Leaderboard` with charts, a SQLite `RunStore` with model-version diffs, and registering your own benchmark without touching the core. |

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
