# Integrations: providers, vector stores, and frameworks

Vincio's breadth sits behind interfaces that already exist, so adding a model
gateway, embedder, reranker, or vector store changes nothing downstream — the
context compiler, budgeting, evals, traces, and security apply unchanged. This
guide covers those integration adapters; for end-to-end migrations see the
"coming from" guides linked at the bottom, and for the MCP/A2A/Skills
interoperability protocols see their dedicated guides.

## Any OpenAI-compatible model

`OpenAICompatibleProvider` speaks the OpenAI Chat Completions dialect, so it
reaches any compatible endpoint. Named presets cover the popular hosted
gateways and resolve their key from the conventional `<NAME>_API_KEY` env var:

```python
from vincio import ContextApp
from vincio.providers import openai_compatible

groq = openai_compatible("groq")                    # GROQ_API_KEY
together = openai_compatible("together")            # TOGETHER_API_KEY
custom = openai_compatible(base_url="https://my-gateway/v1", api_key="...")

app = ContextApp(name="fast", provider=groq, model="llama-3.3-70b-versatile")
```

Presets: `groq`, `together`, `fireworks`, `openrouter`, `deepseek`,
`perplexity`, `xai`, `nvidia`. They are also registered by name, so config and
`build_provider` work too:

```python
from vincio.providers import build_provider

provider = build_provider("groq")     # picks up GROQ_API_KEY from the env
```

```yaml
# vincio.yaml
provider:
  default: groq
  model: llama-3.3-70b-versatile
```

The local/self-hosted path (Ollama, vLLM, llama.cpp, LM Studio) is unchanged —
use the `local`, `ollama`, or `vllm` provider names.

## Embedders

`build_embedder` returns an embedder for local hashing, a hosted embedding API
(httpx only — no SDK), or any provider that supports embeddings:

```python
from vincio.retrieval import build_embedder

local = build_embedder("local")                       # deterministic, offline
jina = build_embedder("jina", api_key="...", dimensions=512)  # Matryoshka truncation
openai = build_embedder("openai", model="text-embedding-3-small")
context = build_embedder("voyage-context")            # per-chunk vectors carry doc context
mm = build_embedder("voyage-multimodal")              # unified text+image vector space
```

`dimensions=N` truncates each output vector to N dims and L2-renormalizes
(the leading dimensions carry the most signal — `retrieval.embedding_dimensions`
in config); hosted embedders request the shorter vector natively, others wrap
in `MatryoshkaEmbedder`. All built-in embedders accept an `input_type`
(`"document"` | `"query"`); `VectorIndex` passes the right one on add/search.

| Kind | Backend | Dependency |
|---|---|---|
| `local` | deterministic hash embedder | none |
| `jina` / `voyage` / `cohere` | hosted embedding API | none (core `httpx`) |
| `voyage-context` | contextual hosted embedder (`voyage-context-3`) | none (core `httpx`) |
| `voyage-multimodal` / `cohere-multimodal` | multimodal text+image hosted embedder | none (core `httpx`) |
| `openai` / `google` / `mistral` / preset names | provider `embed` | the provider's extra |

`voyage-context` (model `voyage-context-3`) embeds each chunk with surrounding
document context — it complements the `contextualize_chunks` LLM-prefix path
rather than replacing it, with `embed_grouped(documents)` for explicit
per-document grouping. The multimodal embedders — `voyage-multimodal`
(`voyage-multimodal-3`) and `cohere-multimodal` (alias `cohere-v4`, model
`embed-v4.0`) — share one text+image vector space; pass
`MultimodalInput(text=..., image=ImageRef(path=...|url=...))` to
`embedder.embed_multimodal([...])`.

## Rerankers

`build_reranker` covers the offline heuristic rerankers and hosted
cross-encoders. Set `retrieval.reranker` in `vincio.yaml` to apply one to every
retrieve:

```python
from vincio.retrieval import build_reranker

reranker = build_reranker("cohere", api_key="...")    # also jina | voyage
heuristic = build_reranker("heuristic")               # offline, no key
```

```yaml
# vincio.yaml
retrieval:
  reranker: cohere
```

Kinds: `heuristic`, `recency`, `authority`, `llm`, `cohere`, `jina`, `voyage`.
The hosted rerankers (`cohere`/`jina`/`voyage`) ride the core `httpx`
dependency — no SDK to install.

## Vector stores

Every backend implements the retrieval `Index` protocol, so one factory swaps
the store without touching the query side. The in-memory backend has no
dependencies; the rest import their client lazily and raise a clear error when
the extra is missing.

```python
from vincio.retrieval import build_embedder
from vincio.storage import build_vector_index

embedder = build_embedder("local")

mem = build_vector_index("memory", embedder)                       # no deps
qdrant = build_vector_index("qdrant", embedder, url="http://localhost:6333")
chroma = build_vector_index("chroma", embedder, path="./chroma")
pinecone = build_vector_index("pinecone", embedder, api_key="...")
lancedb = build_vector_index("lancedb", embedder, uri=".vincio/lancedb")
pg = build_vector_index("pgvector", embedder, dsn="postgresql://localhost/vincio")
weaviate = build_vector_index("weaviate", embedder, url="http://localhost:8080")
milvus = build_vector_index("milvus", embedder, uri="http://localhost:19530")
es = build_vector_index("elasticsearch", embedder, url="http://localhost:9200")
opensearch = build_vector_index("opensearch", embedder, url="http://localhost:9200")
vespa = build_vector_index("vespa", embedder, url="http://localhost", port=8080)
```

| Backend | Extra |
|---|---|
| `memory` | none |
| `qdrant` | `vincio[retrieval]` |
| `pgvector` | `vincio[postgres]` (needs `dsn=`) |
| `chroma` | `vincio[chroma]` |
| `pinecone` | `vincio[pinecone]` |
| `lancedb` | `vincio[lancedb]` |
| `weaviate` | `vincio[weaviate]` |
| `milvus` | `vincio[milvus]` |
| `elasticsearch` (alias `es`) | `vincio[elasticsearch]` |
| `opensearch` | `vincio[opensearch]` |
| `vespa` | `vincio[vespa]` |

## Layout-aware extraction

The dependency-free pypdf text path stays the default, but
`load_document(path, layout=True)` (or `load_pdf(path, layout=True)` /
`extract_pdf_layout(path)`) recovers column-aware reading order, tables (with
bounding boxes), and figures via `vincio[pdf-layout]` (pdfplumber). The pure
helpers `group_words_into_lines`, `order_blocks`, and `assemble_layout` are
available for custom pipelines.

## LangChain & LlamaIndex assets

`vincio.interop` brings LangChain and LlamaIndex tools, retrievers,
loaders/readers, and embeddings into Vincio (and hands Vincio's back). The
`from_*` direction is duck-typed and imports nothing heavy; the `to_*` direction
needs `vincio[langchain]` / `vincio[llamaindex]`.

```python
from vincio.interop import add_langchain_tool, from_llamaindex_reader

add_langchain_tool(app, my_langchain_tool)        # register + enable
docs = from_llamaindex_reader(my_reader)          # -> list[Document]
app.add_source("kb", documents=docs, retrieval="hybrid")
```

See the dedicated guides:
[coming from LangChain](migrate-from-langchain.md),
[coming from LlamaIndex](migrate-from-llamaindex.md).

## Domain packs

Opt-in, dependency-free bundles configure an app for a domain in one call:

```python
from vincio import ContextApp, available_packs

app = ContextApp(name="helpdesk").use_pack("support")   # also engineering | finance | legal
available_packs()                                       # ['engineering', 'finance', 'legal', 'support']
```

Each pack sets a role/objective/rules prompt, a structured output schema,
recommended policies and evaluators, and ships a golden eval set
(`load_pack("support").dataset()`). Inspect them with `vincio packs list` and
`vincio packs show <name>`.

## Next steps

- [connect external data sources](connectors.md)
- [build a RAG app](build-rag-app.md)
- coming from [LangChain](migrate-from-langchain.md), [LlamaIndex](migrate-from-llamaindex.md),
  [Ragas](migrate-from-ragas.md), [Mem0](migrate-from-mem0.md)
- [retrieval concepts](../concepts/retrieval.md)
