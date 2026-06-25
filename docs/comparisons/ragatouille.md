# Vincio vs RAGatouille / ColBERT

RAGatouille makes late-interaction retrieval (ColBERT) easy to use: train,
index, and search multi-vector representations with PLAID compression.

**Where Vincio differs**

- **Late interaction is one fused signal, not the whole retriever.**
  `LateInteractionIndex` implements the same `Index` protocol as BM25,
  dense, learned-sparse, and graph retrieval, so MaxSim scores merge with
  every other signal in one weighted reciprocal-rank fusion
  (`retrieval="hybrid_full"`).
- **PLAID-style scale without a model server.** `compressed=True` clusters
  token vectors into centroids, generates candidates over inverted centroid
  lists, and exact-reranks survivors. Token embeddings come from any
  `Embedder`, the offline hash embedder for tests, a served ColBERT
  checkpoint in production. The same `Embedder` interface and `build_embedder`
  also reach contextual (`voyage-context-3`) and multimodal (Cohere v4 /
  Voyage) embedders, plus Matryoshka dimension truncation (`dimensions=`).
- **Learned sparse rides along.** `SparseIndex` covers the
  SPLADE/uniCOIL-style impact-weighted family (offline approximation
  built in, any served encoder via `CallableSparseEncoder`), so you can fuse
  late interaction *and* learned sparse, something a ColBERT-only stack
  doesn't do.
- **Retrieval ends in a compiled packet.** Whatever the index mix, results
  become provenance-tracked evidence that is scored, deduplicated,
  conflict-resolved, **budgeted, and cited** by the context compiler, and
  measured by the same eval/trace loop as the rest of the run.

**Where RAGatouille is a fit:** training and fine-tuning ColBERT models
themselves. Serve the resulting encoder behind Vincio's `Embedder`
protocol and keep the fusion, budgeting, and evals.
