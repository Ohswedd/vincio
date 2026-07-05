# Vincio vs LlamaIndex

LlamaIndex is excellent at the data layer: ingestion, indexing, RAG, and
document workflows.

**Where Vincio differs**

- **Retrieval is one subsystem of the context compiler**, not the product.
  Retrieved chunks compete with memory, tool results, and instructions for
  a scored token budget, with conflict resolution and deduplication across
  all of them.
- **The advanced-indexing playbook ships natively**: sentence-window,
  parent-document/auto-merging (`AutoMergingIndex`), contextual chunk
  prefixes, HyDE / multi-query / decomposition / step-back query
  strategies, and GraphRAG (communities + hierarchical summaries with
  global/local routing), every one fused through the same weighted RRF and
  measured by the same eval loop, alongside learned-sparse and
  late-interaction indexes LlamaIndex delegates to plugins.
- **Live corpora without rebuilds**: `LiveIndex` upserts, TTL expiry, and
  embedding migrations (`VectorIndex.migrate`), with freshness surfaced in
  evidence metadata.
- **The full lifecycle is one consistent model**: input routing, memory with
  decay and privacy scopes, permissioned tools, bounded agents, output
  contracts with principled repair, evals, optimization, tracing, audit.
- **Grounding is enforced end-to-end**: citation policies are compiled into
  prompts, citations are validated against real evidence ids, and
  groundedness is measured per run.
- **Reasoning-driven retrieval (LAGER).** `app.use_lager()` swaps fixed
  top-k for a lazy loop over a typed evidence graph: it plans the fact types
  an answer needs, acquires source-span-exact evidence incrementally while the
  marginal information gain justifies it, and reports uncovered needs so an
  unanswerable query abstains instead of guessing.

**Where LlamaIndex is a fit:** very broad loader/index integrations for
exotic data sources. Vincio's connector hub covers the common ones (web,
GitHub, SQL, S3, GCS, Notion, Confluence, Slack, plus custom connectors
via `register_connector`), and for anything else `vincio.interop`
converts LlamaIndex readers, retrievers, tools, and embeddings directly:
`from_llamaindex_reader(reader)` → `app.add_source(documents=...)`,
`from_llamaindex_retriever(r)`, `add_llamaindex_tool(app, t)`,
`from_llamaindex_embedding(e)` (and `to_llamaindex_*` with
`vincio[llamaindex]`). Vector stores, Chroma, Pinecone, LanceDB,
Weaviate, Milvus, Elasticsearch, OpenSearch, Vespa, join Qdrant and
pgvector behind one `build_vector_index` factory, and `build_embedder`
spans local, jina, voyage, cohere, and openai plus Matryoshka dimension
truncation (`dimensions=`), contextual (`voyage-context`), and multimodal
(`voyage-multimodal` / `cohere-multimodal`) variants. Document extraction
includes a layout-aware PDF path that recovers reading order, tables, and
figures. See
[Coming from LlamaIndex to Vincio](../guides/migrate-from-llamaindex.md).
