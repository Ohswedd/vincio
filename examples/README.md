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

Run any of them:

```bash
cd examples && python 02_document_qa.py
```
