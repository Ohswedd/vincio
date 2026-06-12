# Vincio vs LlamaIndex

LlamaIndex is excellent at the data layer: ingestion, indexing, RAG, and
document workflows.

**Where Vincio differs**

- **Retrieval is one subsystem of the context compiler**, not the product.
  Retrieved chunks compete with memory, tool results, and instructions for
  a scored token budget, with conflict resolution and deduplication across
  all of them.
- **The full lifecycle is one consistent model**: input routing, memory with
  decay and privacy scopes, permissioned tools, bounded agents, output
  contracts with principled repair, evals, optimization, tracing, audit.
- **Grounding is enforced end-to-end** — citation policies are compiled into
  prompts, citations are validated against real evidence ids, and
  groundedness is measured per run.
- **Reasoning retrieval** retrieves by required fact types and reports
  missing facts, feeding insufficient-evidence behavior.

**Where LlamaIndex is a fit:** very broad loader/index integrations for
exotic data sources. Vincio's `Document`/`Chunk` contracts make it easy to
feed LlamaIndex-parsed content into `app.add_source(documents=...)`.
