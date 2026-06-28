# Vincio examples

Fifteen complete, heavily-commented programs — together they exercise the whole platform. Every one
runs **fully offline** on the deterministic mock provider: no API keys, no network.

```bash
cd examples
python 01_quickstart.py                 # start here

# Point any example at a real model instead of the mock:
export VINCIO_PROVIDER=openai VINCIO_MODEL=gpt-5.2-mini OPENAI_API_KEY=sk-...
python 02_retrieval_rag.py
```

Each file is a guided tour: a module docstring states what it teaches, and numbered sections each
demonstrate one capability and print a concrete result. Read them top to bottom.

| # | Example | What it teaches |
|--|---|---|
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

`_shared.py` holds the small offline helpers every example imports (`example_provider`,
`json_responder`, `citing_responder`, `write_sample_docs`).
