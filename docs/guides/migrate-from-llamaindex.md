# Coming from LlamaIndex to Vincio

LlamaIndex's data layer maps cleanly onto Vincio: readers become documents,
`VectorStoreIndex` becomes a scored source, retrievers and node
postprocessors become rerankers, and the query engine becomes `app.run`.
The `vincio.interop` adapters let you bring those assets across one at a
time, so you can migrate incrementally without rewriting your ingestion.

## Concept mapping

| LlamaIndex | Vincio | Notes |
|---|---|---|
| `SimpleDirectoryReader` / readers | `from_llamaindex_reader(reader)` → `app.add_source(documents=...)` | or `app.add_source("kb", path="./docs")` directly |
| `VectorStoreIndex.from_documents` | `app.add_source(..., retrieval="hybrid")` | hybrid (dense + sparse) by default |
| vector store integrations | `build_vector_index("qdrant"|"chroma"|"pinecone"|"pgvector"|"lancedb"|"weaviate"|"milvus"|"elasticsearch"|"opensearch"|"vespa", embedder)` | `pgvector` needs `dsn=...` |
| `index.as_retriever()` | `from_llamaindex_retriever(li_retriever)` | read-only index with async `.search()` |
| node postprocessors / rerankers | `build_reranker("cohere"|"jina"|"voyage"|"heuristic"|"recency")` | set `retrieval.reranker` in `vincio.yaml` |
| embedding models | `from_llamaindex_embedding(li_embedding)` / `build_embedder(...)` | local, jina, voyage, cohere, openai; Matryoshka truncation (`dimensions=`), contextual (`voyage-context`), multimodal (`voyage-multimodal`/`cohere-multimodal`) |
| `index.as_query_engine().query(...)` | `result = app.run("question")` | returns `output`, `citations`, `cost_usd`, `trace_id` |
| `FunctionTool` | `add_llamaindex_tool(app, li_tool)` | registers and enables the tool |
| `Settings.llm` / model config | `ContextApp(provider=..., model=...)` | provider-neutral; OpenAI-compatible presets too |

## Bring your assets across

Readers — convert LlamaIndex-parsed content into Vincio `Document`s and
ingest them through the document engine:

```python
from llama_index.core import SimpleDirectoryReader
from vincio import ContextApp
from vincio.interop import from_llamaindex_reader

app = ContextApp(name="kb", provider="openai", model="gpt-5.2")

reader = SimpleDirectoryReader("./docs")
docs = from_llamaindex_reader(reader)          # -> list[Document]
app.add_source("kb", documents=docs, retrieval="hybrid")
```

Already have nodes or documents in hand? Convert them directly:

```python
from vincio.interop import from_llamaindex_documents, from_llamaindex_document

docs = from_llamaindex_documents(li_documents)
one = from_llamaindex_document(li_node)
```

Retrievers and embeddings — wrap an existing LlamaIndex retriever as a
read-only source, or reuse its embedding model:

```python
from vincio.interop import from_llamaindex_retriever, from_llamaindex_embedding

index = from_llamaindex_retriever(li_retriever)   # async .search()
embedder = from_llamaindex_embedding(li_embedding)
```

Tools — register a `FunctionTool` so an agent can call it:

```python
from vincio.interop import add_llamaindex_tool

add_llamaindex_tool(app, li_function_tool)
```

Going the other way (Vincio → LlamaIndex) is also supported with
`to_llamaindex_tool`, `to_llamaindex_retriever`, `to_llamaindex_embedding`,
and `to_llamaindex_documents` — install `vincio[llamaindex]` for the `to_*`
direction.

## In Vincio

Once the data is across, drop the index/query-engine ceremony. A LlamaIndex
RAG pipeline:

```python
# before (LlamaIndex)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

docs = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(docs)
engine = index.as_query_engine()
answer = engine.query("how do refunds work?")
```

becomes a source plus a run — retrieval, reranking, budgeting, and citation
enforcement are handled by the compiler:

```python
# after (Vincio)
from vincio import ContextApp

app = ContextApp(name="kb", provider="openai", model="gpt-5.2")
app.add_source("kb", path="./docs", retrieval="hybrid")
app.set_policy("answer_only_from_sources", True)
app.set_policy("require_citations", True)

result = app.run("how do refunds work?")
print(result.output)
print(result.citations, result.cost_usd, result.trace_id)
```

Swap the in-memory index for a production vector store without touching the
query side — the connector hub and embedder feed the same scored retrieval:

```python
from vincio.retrieval import build_embedder, build_reranker
from vincio.storage import build_vector_index

embedder = build_embedder("voyage", model="voyage-3")
index = build_vector_index("qdrant", embedder, collection="kb")
reranker = build_reranker("cohere")        # or set retrieval.reranker in vincio.yaml
```

The same factories span the wider breadth: `build_vector_index` also targets
Weaviate, Milvus, Elasticsearch, OpenSearch, and Vespa, and `build_embedder`
adds Matryoshka dimension truncation (`build_embedder(kind, dimensions=N)`),
contextual (`voyage-context`), and multimodal (`voyage-multimodal` /
`cohere-multimodal`) embedders — all behind the same `Embedder` interface and
feeding the same scored retrieval.

## What Vincio adds

- **A context compiler, not just a query engine** — retrieved chunks compete
  with memory, tool results, and instructions for a scored token budget, with
  conflict resolution and deduplication across all of them.
- **Scored, budgeted context packets** — every piece of evidence is ranked
  and fitted to a budget, so you see exactly what went into each answer.
- **Built-in evals, gates, and an optimization loop** — `groundedness`,
  `lexical_overlap`, `schema_validity`, `cost`, and `latency` metrics with
  CI gates, no separate harness to wire up.
- **Native, provider-neutral observability** — every run writes a trace with
  cost tracking and a `trace_id`; view it with `vincio trace view`.
- **Deterministic security** — permissioned tools (`permission="read_only"`)
  and citation policies compiled into the prompt and validated against real
  evidence ids.
- **A closed improvement loop** — feedback and traces feed back into evals and
  context optimization, all in one consistent model.

## Next steps

- [build a RAG app](build-rag-app.md)
- [connect external data sources](connectors.md)
- [run evals](run-evals.md)
- [structured output](structured-output.md)
- [orchestrate agents](orchestrate-agents.md)
- [optimize context](optimize-context.md)
- [retrieval concepts](../concepts/retrieval.md)
- [memory concepts](../concepts/memory.md)
- [evals concepts](../concepts/evals.md)
- [observability concepts](../concepts/observability.md)
- [agents concepts](../concepts/agents.md)
- [Vincio vs LlamaIndex](../comparisons/llamaindex.md)
