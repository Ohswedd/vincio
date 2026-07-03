# Guide: build a RAG app

A grounded document-QA app in under 30 lines. In a hurry? The whole grounded path
below is one expression with `rag(...)` — see the
[ergonomic front door](../concepts/ergonomic-surface.md):
`rag("./docs").ask("What is the refund window?")` lowers to the exact same governed
run. This guide shows the verbose path it composes, so you can tune any step.

```python
from vincio import ContextApp

app = ContextApp(name="docs_qa")

# 1. Sources: load → chunk (adaptive) → index (BM25 + dense, RRF-merged).
app.add_source("docs", path="./docs", chunking="adaptive", retrieval="hybrid")

# 2. Grounding policy: adds citation rules, requires citations in output,
#    and sets insufficient-evidence behavior.
app.set_policy("answer_only_from_sources", True)

# 3. Built-in evaluators score every run.
app.add_evaluator("groundedness")
app.add_evaluator("citation_accuracy")

result = app.run("How do I configure SSO?")
print(result.output)            # answer with [citation] refs
print(result.citations)         # verified citation refs
print(result.eval_scores)       # {"groundedness": 1.0, "citation_accuracy": 1.0}
print(result.excluded_context)  # why items were excluded
```

## Tuning

```yaml
# vincio.yaml
retrieval:
  top_k: 8
  chunk_size_tokens: 400
  chunk_overlap_tokens: 50
  chunking: adaptive          # fixed | recursive | semantic | heading_aware | table_aware | code_aware | sentence_window | hierarchical | contextual
  reranker: heuristic         # heuristic | recency | authority | llm | null
  embedder: local             # local | jina | voyage | cohere | voyage-context | voyage-multimodal | cohere-multimodal | openai | google | mistral
  embedding_dimensions: null  # Matryoshka output-dimension truncation; null keeps the native dimension
  query_strategies: []        # hyde | multi_query | decompose | step_back
```

## Pushing retrieval quality

When hybrid BM25+dense isn't enough, escalate the index mix and the query
side, see [retrieval concepts](../concepts/retrieval.md) for each technique:

```python
# Fuse BM25 + dense + learned sparse + late interaction in one RRF.
app.add_source("docs", path="./docs", retrieval="hybrid_full")
```

To pull from live systems instead of local files, use the
[connector hub](connectors.md): `app.add_source("kb", connector=connect("github", repo="acme/handbook"))`.

## Per-run files

```python
result = app.run("Which termination clauses are risky?",
                 files=["msa.pdf", "order_form.pdf"], tenant_id="acme")
```

Files are loaded (`pip install "vincio[pdf]"` for PDF), chunked, indexed,
and offered to the context compiler alongside source evidence.

## Multi-tenancy

Pass `tenant_id=` on every run. With `security.tenant_isolation: true`
(default), retrieval filters chunks by tenant, memory scopes are enforced,
and cross-tenant access raises `TenantIsolationError`.

## Hallucination defense in depth

1. Retrieval only surfaces real chunks with provenance.
2. The context compiler excludes low-relevance evidence (reported).
3. The prompt carries the citation policy in the stable prefix.
4. The output validator rejects citations that don't match real evidence ids.
5. The `groundedness` evaluator measures supported-claim ratio per run.

## Keep a task frame across many calls (context anchors)

Some documents are not "look it up when asked" evidence — they are the *frame*
of the whole task (a PRD, coding standards, a brand guide). Mark them
`anchor=True` and Vincio keeps that frame present on **every** call at a flat,
tiny cost, instead of re-pasting the corpus or losing it to a lexical miss:

```python
app.add_source("spec", documents=[prd, brand, standards], anchor=True, brief_tokens=300)
print(app.task_brief())            # the compact, constraint-first frame

# The frame is present here even though the query never mentions the brand rule:
report = app.run("add a settings panel")
```

The anchor docs are still chunked and indexed, so a call that needs a specific
section retrieves it normally — the frame just guarantees the global context is
always there. See [context anchors](../concepts/context-anchors.md) for how it
is distilled, budget-reserved, and guaranteed.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: Retrieval (RAG)](../concepts/retrieval.md)
- [Concept: Context anchors (task frame)](../concepts/context-anchors.md)
- [Guide: Connect external data sources](connectors.md)
- [Example: 02_retrieval_rag.py](../../examples/02_retrieval_rag.py)
- [Example: 20_context_anchors.py](../../examples/20_context_anchors.py)
- [Concept: Context packets & long-horizon governance](../concepts/context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
