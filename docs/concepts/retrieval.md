# Retrieval

Retrieval in Vincio finds the **evidence required for the task**, not just
similar text.

## Pipeline

```text
query_understanding → query_rewrite → candidate_generation (multi-index)
→ hybrid_merge (weighted RRF) → rerank → deduplicate → evidence
```

## Modes

- **BM25** — pure-python Okapi BM25.
- **Dense** — vector index (local hash embeddings offline; provider
  embeddings, Qdrant, or pgvector in production).
- **Hybrid** — both indexes merged with reciprocal rank fusion.
- **Graph** — `EntityGraph` walks entity co-occurrence paths
  (Customer → Plan → RefundPolicy → Evidence).
- **Multi-hop** — entities from the first hits seed follow-up queries.
- **Reasoning retrieval** — declare the facts a task needs
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
(symbol boundaries), `adaptive` (auto-select per document).

## Rerankers

`heuristic` (lexical + structure priors), `recency`, `authority`,
`llm` (batched model scoring), or any cross-encoder via
`CrossEncoderReranker(score_fn)`.

## Evaluation

Retrieval metrics ship in `vincio.evals.metrics`: `recall_at_k`,
`precision_at_k`, `mrr`, `ndcg`, `context_precision`, `context_recall`.
