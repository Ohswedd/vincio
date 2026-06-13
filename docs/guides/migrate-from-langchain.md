# Coming from LangChain / LangGraph to Vincio

Most LangChain concepts have a direct Vincio counterpart: a chain becomes a
`ContextApp`, a retriever becomes a source, an agent graph becomes
`app.crew()` / `app.graph()`. The `vincio.interop` adapters are duck-typed, so
you can pull your existing LangChain tools, retrievers, loaders, and embeddings
in (and hand Vincio components back) without rewriting everything at once.

## Concept mapping

| LangChain / LangGraph | Vincio | Notes |
|---|---|---|
| Chain / LCEL `Runnable` | `ContextApp` + `app.run("...")` | one compiled, traced, budgeted call |
| `@tool` / `StructuredTool` | `add_langchain_tool(app, t)` or `app.add_tool(fn)` | duck-typed import-free wrap |
| `VectorStoreRetriever` | `from_langchain_retriever(r)` or `app.add_source(retrieval="hybrid")` | bring yours, or let Vincio index |
| `DocumentLoader` | `from_langchain_loader(loader)` | returns `list[Document]` |
| `Embeddings` | `from_langchain_embeddings(e)` or `build_embedder(...)` | reuse or swap |
| `ConversationBufferMemory` | `app.remember(...)` / `app.recall(...)` | scoped, decaying, audited |
| `AgentExecutor` | `app.crew(...)` | roles, delegation, bounded rounds |
| `LangGraph` `StateGraph` | `app.graph(...)` | conditional edges, checkpoints, resume |
| `PydanticOutputParser` | `output_schema=Model` | validated, repaired output |
| LangSmith callbacks / tracing | native traces + evals | JSONL/OTEL, no account |
| `with_structured_output` retries | `set_policy(...)` + validator | enforced in code |

Interop is incremental: keep a LangChain component in place, wrap it, and
replace it with the native equivalent when you're ready.

## Bring your assets across

Tools — register a LangChain tool directly (no `langchain` import needed):

```python
from vincio import ContextApp
from vincio.interop import add_langchain_tool, from_langchain_tool

app = ContextApp(name="support", provider="openai", model="gpt-5.2")

add_langchain_tool(app, my_lc_tool)            # register + enable in one call

spec = from_langchain_tool(my_lc_tool)         # or inspect the adapter
# {name, description, input_schema, handler}
```

Retrievers and loaders — reuse what you already built:

```python
from vincio.interop import from_langchain_retriever, from_langchain_loader

index = from_langchain_retriever(my_lc_retriever)   # read-only, async .search()
docs = from_langchain_loader(my_lc_loader)          # -> list[Document]
app.add_source("kb", documents=docs)                # ingest converted Documents
```

Embeddings — wrap a LangChain embeddings object, or switch to a native one:

```python
from vincio.interop import from_langchain_embeddings
from vincio.retrieval import build_embedder

embedder = from_langchain_embeddings(my_lc_embeddings)
embedder = build_embedder("openai")                 # or local | jina | voyage | cohere
```

Hand Vincio back to LangChain — the `to_*` direction (needs
`pip install "vincio[langchain]"`):

```python
from vincio.interop import to_langchain_tool, to_langchain_retriever

lc_tool = to_langchain_tool(vincio_tool)
lc_retriever = to_langchain_retriever(vincio_index)
# also to_langchain_embeddings / to_langchain_documents
```

## In Vincio

A LCEL retrieval chain — retriever, prompt, parser, model wired by hand —
collapses into a source plus a grounding policy:

```python
# Before (LangChain): loader -> splitter -> vectorstore -> retriever
#                      -> prompt | model | StrOutputParser, with manual citing.

# After (Vincio): one app that loads, chunks, indexes, grounds, and cites.
from vincio import ContextApp

app = ContextApp(name="docs_qa", provider="openai", model="gpt-5.2")
app.add_source("kb", path="./docs", retrieval="hybrid")
app.set_policy("answer_only_from_sources", True)
app.set_policy("require_citations", True)

result = app.run("How do I configure SSO?")
print(result.output, result.citations, result.cost_usd, result.trace_id)
```

A LangGraph `StateGraph` maps to `app.graph()` — conditional edges and
checkpoints included; multi-specialist `AgentExecutor` setups map to
`app.crew()`:

```python
# Before (LangGraph): StateGraph + add_conditional_edges + a checkpointer.
graph = app.graph("escalation")
graph.add_node("classify", classify)
graph.add_node("reply", draft_reply)
graph.add_conditional_edge("classify", lambda s: s["severity"],
                           {"high": "reply", "low": "reply"})
flow = graph.compile()
done = flow.invoke({"ticket": ticket_text})

# Before (AgentExecutor with tools): a crew with roles and delegation.
crew = app.crew(name="triage", members=[
    {"name": "billing", "description": "invoices, refunds"},
    {"name": "writer", "goal": "draft the customer reply"},
])
answer = crew.run("Customer disputes invoice INV-77")
```

LangChain memory becomes scoped, native memory:

```python
app.add_memory()
app.remember("prefers email over phone", user_id="u1")
hits = app.recall("contact preference", user_id="u1")
```

## What Vincio adds

- **A context compiler, not string templates** — prompts, evidence, memory,
  tools, and policies compile into one scored, budgeted, provenance-aware
  packet with an excluded-context report for every omission.
- **Built-in evals + gates + optimization** — `groundedness`,
  `semantic_similarity`, `schema_validity`, `cost`, and `latency` ship in the
  core library and gate CI; no separate eval SaaS.
- **Native, provider-neutral observability** — every run writes a full trace
  (JSONL or OTEL) with cost tracking and `result.trace_id`; no LangSmith
  account required.
- **Deterministic security** — answer-only-from-sources, citation enforcement,
  tool permissions (`app.add_tool(fn, permission="read_only")`), and audit logs
  are enforced in code, not by the model.
- **Provider breadth without rewrites** — `openai_compatible("groq")` and
  `build_provider("groq")` reach groq, together, fireworks, openrouter,
  deepseek, perplexity, xai, nvidia, or any OpenAI-compatible gateway.
- **A closed improvement loop** — eval-scored runs feed dataset curation and
  gated prompt/context/routing optimization, so scores change the system.

## Next steps

- [build a RAG app](build-rag-app.md)
- [connect external data sources](connectors.md)
- [run evals](run-evals.md)
- [structured output](structured-output.md)
- [orchestrate agents](orchestrate-agents.md)
- [optimize context](optimize-context.md)
- concepts: [retrieval](../concepts/retrieval.md), [memory](../concepts/memory.md),
  [evals](../concepts/evals.md), [observability](../concepts/observability.md),
  [agents](../concepts/agents.md)
- [Vincio vs LangChain / LangGraph](../comparisons/langchain.md)
