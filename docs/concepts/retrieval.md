# Retrieval

Retrieval in Vincio finds the **evidence required for the task**, not just
similar text, and it stays one scored, budgeted subsystem of the context
compiler, not the center of gravity.

## Pipeline

```text
query_understanding (strategies) → query_rewrite → candidate_generation
(multi-index) → hybrid_merge (weighted RRF) → rerank → deduplicate → evidence
```

## Index types

All indexes implement one `Index` protocol (`add` / `search` / `delete`), so
any mix fuses in a single weighted reciprocal-rank-fusion merge:

- **BM25**: pure-python Okapi BM25.
- **Dense**: vector index (local hash embeddings offline; provider or hosted
  embeddings, `build_embedder("local"|"jina"|"voyage"|"cohere"|<provider>)`,
  with Qdrant, pgvector, Chroma, Pinecone, LanceDB, Weaviate, Milvus,
  Elasticsearch/OpenSearch, or Vespa in production, all behind one
  `build_vector_index(kind, embedder, ...)` factory). `VectorIndex.migrate()`
  re-embeds in place for embedding-model migrations. Embedders support
  Matryoshka dimension truncation (`build_embedder(..., dimensions=N)`),
  contextual chunk embeddings (`voyage-context`), and unified text+image
  multimodal embeddings (`voyage-multimodal` / `cohere-multimodal`), plus
  query-vs-document `input_type` hints applied automatically on add/search.
- **Learned sparse**: `SparseIndex` over SPLADE-style impact vectors:
  the offline `LocalImpactEncoder` (sublinear tf + morphological expansion)
  or any served model via `CallableSparseEncoder`.
- **Late interaction**: `LateInteractionIndex` scores per-token MaxSim
  (ColBERT-style); `compressed=True` adds PLAID-style centroid candidate
  generation with exact rerank for scale.
- **Graph**: `EntityGraph` walks entity co-occurrence paths
  (Customer → Plan → RefundPolicy → Evidence).

App modes: `bm25`, `dense`, `sparse`, `late_interaction`, `hybrid`
(BM25+dense), `hybrid_full` (BM25+dense+sparse+late-interaction), `graph`,
`hybrid_graph`, e.g. `app.add_source(..., retrieval="hybrid_full")`.

## How hybrid fusion works

No single signal decides the ranking. `RetrievalEngine.retrieve` fans out
every `(query, index)` pair concurrently under one concurrency bound, then
folds the ranked lists into one order with **weighted reciprocal-rank fusion**
(`reciprocal_rank_fusion`), so a chunk that ranks well in *several* signals
beats one that spikes in a single index:

```text
score(chunk) = Σ_lists  weight_list / (k + rank_in_list + 1)      # k = 60
```

Rank position, not raw index score, is what fuses — that is what lets a
BM25 hit and a cosine hit combine without ever comparing incompatible score
scales. The weights are the query plan made concrete: the rewritten main query
carries weight `1.0`, each decomposed subquery `0.6`, and each
query-understanding expansion its strategy weight (`decompose` 0.6,
`multi_query`/`hyde` 0.5, `step_back` 0.3), each multiplied by the per-index
`index_weights`.

Precision comes *after* recall, in a fixed order:

```text
plan → over-fetch (candidate_multiplier × top_k per search)
     → weighted RRF merge → [multi-hop re-fuse] → [rerank top_k×2]
     → dedup (near_duplicate_score ≥ duplicate_threshold, default 0.9)
     → top_k evidence
```

- **Over-fetch**: each index search pulls `candidate_multiplier` (default 4) ×
  `top_k` candidates, so the merge has depth to work with before the cut.
- **Multi-hop** (`multi_hop=True`): entities from the current top hits seed
  follow-up queries, re-fused into the running order (held evidence weighted
  `2.0`, each hop list `0.4`), bounded by `max_hops`.
- **Rerank** (optional): a `Reranker` re-scores the merged top `top_k × 2`.
- **Deduplicate**: near-identical chunks (word-shingle Jaccard or containment ≥
  `duplicate_threshold`) collapse, and selection stops at `top_k`.
- **Grow-only adaptive k** (`retrieval.adaptive_top_k`, off by default): a
  broad or multi-part query raises `top_k` up to `adaptive_top_k_ceiling`
  (default 20), never below the configured floor — k is the *recall* knob;
  precision stays governed downstream by the relevance gate, min-score flush,
  dedup, and the context budget.

Every merge, rewrite, and chosen-k is recorded on the plan and in traces, so
the fused order is reconstructable, not a black box.

## When to use which mode

- **`bm25`** — exact terms, code identifiers, IDs, short corpora where an
  embedder buys nothing; zero model calls, fully offline.
- **`dense`** — paraphrase-heavy questions where the answer shares meaning but
  not words with the query.
- **`hybrid`** (the default) — the safe general choice: lexical anchors the
  precise terms, dense recovers the paraphrases.
- **`hybrid_full`** — highest recall (adds learned-sparse + late-interaction),
  at more index cost; reach for it on hard multi-aspect corpora.
- **`graph` / `hybrid_graph`** — entity-centric questions that walk
  relationships (Customer → Plan → RefundPolicy) rather than match text.

## Query understanding

Planner strategies expand the query before fusion; each expansion is
recorded on the plan and in traces. LLM-written with a provider,
deterministic heuristics offline:

- **HyDE**: a hypothetical answer passage used as a search probe.
- **Multi-query**: paraphrase rewrites of the same intent.
- **Decomposition**: self-contained subquestions for multi-hop questions.
- **Step-back**: broader questions about the underlying concepts.

```python
engine = RetrievalEngine(indexes, query_strategies=["hyde", "multi_query"])
# or per call: engine.retrieve(q, strategies=["decompose"])
# or app-wide: config retrieval.query_strategies
```

## Advanced indexing

- **Sentence-window** (`chunking="sentence_window"`): score the precise
  sentence, hand the model the surrounding window.
- **Hierarchical / parent-document** (`chunking="hierarchical"`): small
  children indexed for precision; `AutoMergingIndex` merges sibling hits
  back into their parent so the model sees one coherent unit.
- **Contextual retrieval** (`chunking="contextual"`): every chunk gets a
  situating prefix (title, section path, document lead);
  `contextualize_chunks(doc, chunks, provider=..., model=...)` upgrades the
  prefixes to LLM-written context.

## GraphRAG

`GraphRAG` clusters the entity graph into communities (deterministic label
propagation), writes hierarchical community summaries (extractive offline,
LLM-written with a provider), and routes queries: **local** questions walk
entity paths; **global** questions ("main themes across…") retrieve
community summaries, each carrying provenance to its member chunks.

```python
rag = GraphRAG(app.entity_graph)
await rag.build()
evidence = await rag.retrieve("What are the main themes across these contracts?")
```

## Live indexes

`LiveIndex` wraps any index for corpora that change: `upsert()` replaces in
place, per-entry TTLs expire stale content, `purge_expired()` reclaims it,
and every chunk is stamped `indexed_at`, surfaced as `indexed_at` /
`age_days` in evidence metadata for freshness-aware scoring.

## Multi-hop & reasoning retrieval

- **Multi-hop**: entities from the first hits seed follow-up queries.
- **Reasoning retrieval**: declare the facts a task needs
  (`FactSchema`), retrieve per missing fact, and get a coverage report:

```python
schema = FactSchema.from_names("refund_decision",
    ["customer_plan", "payment_status", "refund_policy", "dispute_status"])
evidence, coverage, report = await ReasoningRetriever(engine).retrieve(query, schema)
report["missing_facts"]   # feeds insufficient-evidence behavior
```

## Chunking strategies

`fixed`, `recursive`, `semantic` (lexical cohesion), `heading_aware`
(section-path prefixed), `table_aware` (tables stay intact), `code_aware`
(symbol boundaries), `sentence_window`, `hierarchical` / `parent_document`,
`contextual`, `adaptive` (auto-select per document).

Layout-aware PDF extraction (`load_document(path, layout=True)`) recovers
column-aware reading order, tables, and figures before chunking; the
dependency-free text path stays the default.

## Rerankers

`heuristic` (lexical + structure priors), `recency`, `authority`,
`llm` (batched model scoring), hosted cross-encoders (`cohere`, `jina`,
`voyage`, httpx-only) via `build_reranker(kind, api_key=..., model=...)`, or any
custom cross-encoder via `CrossEncoderReranker(score_fn)`.

## Connectors

The connector hub feeds the document engine from external systems: web,
GitHub, SQL, S3, GCS, Notion, Confluence, Slack, or anything custom via
`register_connector`. See the [connectors guide](../guides/connectors.md).

## Gotchas

- **Fusion is rank-based, not score-based.** Raising one index's raw scores
  does nothing; change its `index_weights` (or the per-query weights) to shift
  its influence on the merged order.
- **`candidate_multiplier` is the recall ceiling.** If the right chunk never
  enters the top `candidate_multiplier × top_k` of *any* index, no reranker can
  recover it — widen the multiplier or add a signal, don't just rerank harder.
- **`adaptive_top_k` only grows.** It never trims a query below the configured
  `top_k`; it is a recall lever, and precision is still the downstream gate's job.
- **Dedup is lexical.** Two chunks that say the same thing in different words
  both survive `near_duplicate_score`; semantic near-dupes are pruned later, in
  the context compiler, not here.
- **Freshness needs a `LiveIndex`.** `indexed_at` / `age_days` only reach
  evidence metadata when content was stamped on ingest; a plain index has no age.

## Evaluation

Retrieval metrics ship in `vincio.evals.metrics`: `recall_at_k`,
`precision_at_k`, `mrr`, `ndcg`, `context_precision`, `context_recall`.
The VincioBench `rag` family compares every retrieval mode (BM25, dense,
sparse, late interaction, PLAID, hybrid, hybrid_full, query understanding,
GraphRAG) on recall@3/MRR, gated in CI by `benchmarks/budgets.json`.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: build a RAG app](../guides/build-rag-app.md)
- [Guide: Connect external data sources](../guides/connectors.md)
- [Example: 02_retrieval_rag.py](../../examples/02_retrieval_rag.py)
- [Concept: Context packets & long-horizon governance](context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
