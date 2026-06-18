# Vincio examples

All examples run **fully offline** by default using the deterministic mock
provider — no API keys needed. To run against a real model:

```bash
export VINCIO_PROVIDER=openai VINCIO_MODEL=gpt-5.2-mini OPENAI_API_KEY=sk-...
```

| Example | Shows |
|---|---|
| `01_support_triage.py` | typed (Pydantic) output, classification |
| `02_document_qa.py` | RAG with citations, grounding policy, per-run evaluators |
| `03_contract_review.py` | an end-to-end contract review app |
| `04_invoice_extraction.py` | structured extraction + extraction-F1 eval |
| `05_research_agent.py` | ReAct agent with tools, bounded budgets |
| `06_crm_agent.py` | memory + permissioned tools + approval-gated writes |
| `07_codebase_qa.py` | code-aware chunking, repository import graph |
| `08_spreadsheet_analysis.py` | table-aware chunking, schema inference, quality checks |
| `09_eval_pipeline.py` | datasets, gates, reports, baseline diff |
| `10_optimization_run.py` | prompt-variant search with gated promotion |
| `11_streaming_performance.py` | end-to-end streaming, partial-JSON output, compile caches, zero-copy packets |
| `12_advanced_rag.py` | sparse+late-interaction fusion, query understanding, auto-merging, GraphRAG, live indexes, SQL connector |
| `13_memory_personalization.py` | scoped remember/recall, hybrid vector+graph recall, consolidation with provenance, GDPR-style hygiene, memory eval harness |
| `14_evaluation_observability.py` | quality/safety metrics, eval assertions, synthetic datasets, experiments with significance, red-teaming, prompt registry, sessions + feedback, trace viewer, traces→datasets |
| `15_multi_agent_crew.py` | crews with roles, shared blackboard, sequential/parallel/hierarchical processes, delegation records |
| `16_durable_graph.py` | durable stateful graphs, checkpoints, human-in-the-loop interrupts, edit-and-resume, time-travel forks, composition |
| `17_reliable_structured_output.py` | typed signatures, constrained decoding, streaming validation, rails, self-correction, multi-schema routing |
| `18_closed_loop.py` | the closed loop: traces→dataset→eval→optimize→promote, auto-memory from runs, retrieval feedback, Pareto frontier, learned budgeting |
| `19_framework_interop.py` | LangChain/LlamaIndex interop, OpenAI-compatible providers, `build_embedder`/`build_vector_index` breadth |
| `20_domain_pack.py` | opt-in domain packs (`use_pack`), pack schema + golden eval set |
| `21_security_governance.py` | PII/secret redaction, injection defense, RBAC/ABAC access control, programmable rails, tamper-evident audit log |
| `22_mcp_tools_and_resources.py` | MCP client + server: register a server's tools/resources, expose the app as an MCP server |
| `23_a2a_delegation.py` | A2A: expose a crew as an agent (Agent Card + task lifecycle), remote agent as a bounded crew delegate |
| `24_agent_skills.py` | Agent Skills: load `SKILL.md` with progressive disclosure, bundled scripts as sandboxed tools |
| `25_reasoning_control.py` | unified `reasoning_effort` across providers, thinking-token cost accounting, Responses API adapter |
| `26_agentic_eval.py` | trajectory & tool-use metrics, multi-turn `Simulator`, online eval + drift, Cohen's-κ annotation, A/B + metric-as-guardrail |
| `27_cost_and_reliability.py` | batch execution at ~50% cost, circuit breaking + health-aware failover, key pooling with RPM+TPM limits, runtime model cascades, cost attribution + budget SLOs, prompt-cache strategy, incremental + sharded indexing |
| `28_reflective_optimization.py` | GEPA-style reflective optimizer + MIPRO joint proposal, distillation flywheel (grounded fine-tuning JSONL + gated teacher→student), learned prompt compression (faithfulness-gated), optimizer-judge calibration |
| `29_multimodal_retrieval.py` | Matryoshka (MRL) embeddings, query/document input-type hints, contextual (Voyage context-3) & multimodal (Cohere v4 / Voyage) embedders, new vector stores (Weaviate/Milvus/Elasticsearch/OpenSearch/Vespa), layout-aware PDF extraction, and the optional voice/realtime module |
| `30_governance_compliance.py` | model/system cards, OWASP/NIST/MITRE compliance mapping, AI-BOM with model-hash verification, EU AI Act content marking, data lineage + erasure-by-source, data-residency routing, multilingual PII + token tax, RAG-poisoning detection |
| `31_honest_fast_spine.py` | enforced full Budget (hard caps + opt-out), the data-driven `ModelRegistry` (capabilities/pricing/lifecycle), semantic context scoring + value-level contradiction, `RunHandle` cooperative cancellation, and significance-gated promotion + the trace-replay executor |
| `32_swap_regression.py` | provider/model rotation: capability-aware routing, the `SwapGate` (replay + eval + cost/latency/behavioral diff with significance), model-swap regression with flake quarantine, shadow + canary with auto-rollback, the lifecycle watcher's migration proposals, and Google/Vertex batch parity |
| `33_documents_and_media_out.py` | documents & images flow OUT (1.9): the `DocumentBuilder` with a structural `DocumentContract`, the `CitedReportBuilder` (resolved `[E1]` footnotes + bibliography + per-claim entailment), redline generation, image generation and TTS with C2PA provenance + budget + audit, richer inputs (PPTX/transcript/forms), and the EU AI Act conformity pack (risk tier + Annex IV + FRIA) |
| `34_continual_loop_and_agentic_frontier.py` | the loop closes itself (1.10): the online improvement controller (drift → gated re-eval / re-optimization / rollback), the real provider-backed GEPA reflector, the autonomous experiment proposer + held-out growing golden regression suite, the budgeted citation-gated deep-research agent, the self-editing memory OS, and computer-use + hosted tools behind a hardened `IsolationBackend` |
| `35_breaking_window_2_0.py` | the one breaking window (2.0): capability facades over the decomposed `ContextApp`, the multimodal-native Context Packet (image/table evidence + cross-process `materialize()` from a content-addressed store), a structured `FilterSpec` with native pushdown + tenant scope, the typed/versioned event catalog, enterprise endpoints (Bedrock/Vertex/Azure) behind a pluggable auth strategy, and mandatory egress DLP + a signed, Merkle-checkpointed audit chain |
| `36_scale_out_and_train.py` | scale out & train for real (2.1): distributed durable execution (a TTL lease + checkpoint-version CAS so two workers can't double-execute, a worker-pool batch fan-out, and `Send` map-reduce super-steps), executed swap-gated distillation (a `StudentTrainer` that trains + registers a model, promoted only past the significance gate), a served observability + alerting plane over an `IndexedTraceStore` (percentiles, cost-by-tenant, a `ViewerApp` dashboard, SRE burn-rate alerts), quantized Matryoshka two-stage retrieval (`TwoStageIndex`), and batteries-included local neural models (an in-process `GGUFProvider` + a `FastEmbedEmbedder`) |
| `37_benchmarks_and_fabric.py` | prove it on the world's benchmarks + the agent fabric (2.2): a stateful-environment eval harness (`Environment` reset/step/observe/verify with a task-success oracle that scores the database end state), the five agentic benchmark adapters (SWE-bench Verified / τ-bench / GAIA / WebArena / BFCL) pinned by a task-set hash and replayed offline against each benchmark's verifiable scorer, a retrieval-eval harness with index-version regression gated on recall deltas using the swap significance test, the governed agent fabric (`AgentDirectory` over A2A + AGNTCY/ACP + the MCP registry under an `AllowListGate`, every resolution an audited access decision), and generative UI (an agent run streamed as AG-UI events) |
| `38_self_improvement_and_provable_erasure.py` | the 3.0 breaking culmination: one `SelfImprovementPolicy` driving a streaming controller (observe → proposal → meta-optimization via successive-halving → re-optimization → canary → promote/rollback), canary-gated `app.deploy` (promote live only on a no-regression verdict, else roll back), provable erasure (`app.erase_source` emits a signed, content-bound `ErasureProof` over the exact removed-id set across indexes/memory/artifacts, that *verifies*), a `ConsentLedger` binding data to a GDPR purpose/lawful basis (withdraw consent → recall stops), and bi-temporal memory (as-of recall after a correction, plus per-memory ACLs for team-shared memory) |

Run any of them:

```bash
cd examples && python 02_document_qa.py
```
