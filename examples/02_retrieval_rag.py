"""Retrieval & RAG depth.

Retrieval is where a context engine earns its keep: the right evidence, fused
from complementary signals, found despite vocabulary mismatch, filtered by
structured metadata, and — crucially — not limited to text. This example walks
the whole stack on the deterministic offline mock: four-way hybrid fusion
(BM25 + dense + learned-sparse + late-interaction in one RRF), query
understanding (HyDE / multi-query / decomposition), chunking strategies,
GraphRAG global-vs-local routing, serializable FilterSpec metadata pushdown,
Matryoshka/multimodal embedders, and image/table/video as first-class scored
evidence in a single packet. No API keys, no network.
"""

from __future__ import annotations

import asyncio

from _shared import example_provider

from vincio.context.compiler import ContextCompiler
from vincio.core.types import Chunk, Document, EvidenceItem, ImageRef
from vincio.documents import MockVideoAnalyzer, video_evidence_items
from vincio.retrieval import (
    BM25Index,
    EntityGraph,
    FilterSpec,
    GraphRAG,
    LateInteractionIndex,
    LocalHashEmbedder,
    MatryoshkaEmbedder,
    RetrievalEngine,
    SparseIndex,
    VectorIndex,
    and_,
    build_embedder,
    chunk_document,
    eq,
)

# A small, self-contained corpus reused across the sections below. The mix of
# exact terms ("uptime"), stems ("refund" vs "refunding"), and named entities
# ("Acme Corp", "Beta LLC") is deliberate — it exercises each retrieval signal.
DOCS = [
    Document(
        title="Refund Policy",
        text="Customers on the Pro plan may request refunds within 30 days of purchase. "
        "Basic plan refunds incur a $5 processing fee and must be requested within 14 days. "
        "Refunds are processed by the billing team within five business days.",
    ),
    Document(
        title="Terms of Service",
        text="The subscription renews automatically unless terminated 60 days before the "
        "renewal date. The initial term is 24 months. Acme Corp contracts Beta LLC to "
        "operate the billing platform under a Master Service Agreement.",
    ),
    Document(
        title="SLA",
        text="The service level agreement guarantees 99.9 percent monthly uptime. Credits of "
        "10 percent apply for each hour of downtime beyond the threshold.",
    ),
]


def banner(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


# ── 1. Four-way hybrid fusion ────────────────────────────────────────────────
async def hybrid_fusion() -> None:
    """BM25 + dense + learned-sparse + late-interaction, fused in one RRF.

    Each index captures a different signal: BM25 = exact lexical overlap, dense
    = semantic similarity, learned-sparse (SPLADE-style) = term expansion via
    stems/synonyms, late-interaction (ColBERT-style) = fine-grained token
    matching. RetrievalEngine runs them concurrently and merges with reciprocal
    rank fusion, so a query that matches *no* index perfectly can still rank the
    right chunk first. The query "refunding a purchase" shares no exact token
    with "refunds within 30 days" — only the non-lexical signals bridge it.
    """
    banner("1. Four-way hybrid fusion (BM25 + dense + sparse + late-interaction)")
    chunks = [c for doc in DOCS for c in chunk_document(doc, strategy="recursive", size=60)]
    bm25 = BM25Index()
    sparse = SparseIndex()
    late = LateInteractionIndex()
    dense = VectorIndex(LocalHashEmbedder(dim=128))  # offline, deterministic vectors
    for index in (bm25, sparse, late, dense):
        await index.add(chunks)

    engine = RetrievalEngine([bm25, sparse, late, dense])
    # use_planner=False isolates the *fusion* behaviour from query rewriting.
    result = await engine.retrieve("refunding a purchase", top_k=2, use_planner=False)
    print(f"  fused 4 signals -> {len(result.evidence)} chunks for a no-exact-match query:")
    for item in result.evidence:
        print(f"    [{item.id}] {(item.text or '')[:64]}...")


# ── 2. Query understanding ───────────────────────────────────────────────────
async def query_understanding() -> None:
    """HyDE, multi-query, and decomposition turn one query into many.

    The planner rewrites the user's question into several retrieval queries
    *before* fusion. HyDE drafts a hypothetical answer and searches with it;
    multi-query paraphrases; decompose splits a compound question into parts.
    All rewrites are retrieved and their results fused, widening recall without
    asking the user to phrase things "search-engine style".
    """
    banner("2. Query understanding (HyDE / multi-query / decomposition)")
    chunks = [c for doc in DOCS for c in chunk_document(doc, strategy="recursive", size=60)]
    bm25 = BM25Index()
    await bm25.add(chunks)

    engine = RetrievalEngine([bm25], query_strategies=["hyde", "multi_query", "decompose"])
    result = await engine.retrieve(
        "Compare the refund windows across plans and the SLA credit", top_k=2
    )
    for expansion in result.plan.expansions:
        # Each strategy emits one or more derived queries.
        print(f"  {expansion.strategy:<12} -> {expansion.queries}")
    print(f"  {result.metadata['queries']} total queries fused -> top hit: {result.evidence[0].id}")


# ── 3. Chunking strategies ───────────────────────────────────────────────────
def chunking_strategies() -> None:
    """Chunk granularity is a retrieval knob, not an afterthought.

    The same document yields very different evidence units depending on the
    strategy. 'recursive' respects sentence/paragraph boundaries; 'hierarchical'
    emits parent+child chunks (small for precise matching, large parents for
    context); table/code-aware strategies (used via app.add_source) keep
    structured units intact. We print chunk counts so the trade-off is visible.
    """
    banner("3. Chunking strategies — granularity is a retrieval knob")
    [refund_doc] = [d for d in DOCS if d.title == "Refund Policy"]
    for strategy, size in (("recursive", 60), ("recursive", 30), ("hierarchical", 20)):
        chunks = chunk_document(refund_doc, strategy=strategy, size=size)
        levels = sorted({c.metadata.get("level", "flat") for c in chunks})
        print(f"  {strategy:<12} size={size:<3} -> {len(chunks):>2} chunks  levels={levels}")


# ── 4. GraphRAG ──────────────────────────────────────────────────────────────
async def graphrag() -> None:
    """Entity graph + community summaries, routed global vs. local.

    GraphRAG extracts entities/relations into a graph, clusters them into
    communities, and summarizes each. A *global* question ("main themes")
    answers from community summaries; a *local* question ("what did Acme agree
    with Beta") traverses the graph around named entities. rag.route() picks the
    mode automatically — the same index serves both shapes of question.
    """
    banner("4. GraphRAG — communities + global/local routing")
    graph = EntityGraph()
    for doc in DOCS:
        graph.add_chunks(chunk_document(doc, strategy="recursive", size=40))
    rag = GraphRAG(graph)
    communities = await rag.build()
    print(f"  detected {len(communities)} communities")
    for query in (
        "What are the main themes across these documents?",
        "What did Acme Corp agree with Beta LLC?",
    ):
        mode = rag.route(query)  # 'global' (summaries) vs 'local' (entity traversal)
        evidence = await rag.retrieve(query, top_k=2)
        head = (evidence[0].text or "")[:60] if evidence else "(none)"
        print(f"  [{mode:<6}] {query}\n           -> {head}...")


# ── 5. Structured metadata FilterSpec ────────────────────────────────────────
async def metadata_filter() -> None:
    """Serializable FilterSpec: hard metadata constraints pushed into search.

    Relevance is not the only constraint — tenancy, document kind, and field
    values are *hard* filters. FilterSpec is a serializable boolean tree
    (and_/or_/eq...) that round-trips as data and compiles to native predicates
    for hosted stores (Pinecone, pgvector, ...) while also filtering local
    indexes via `where=`. Here we retrieve refunds but restrict to the Pro plan.
    """
    banner("5. Structured metadata FilterSpec — hard constraints + pushdown")
    chunks = [
        Chunk(document_id="kb", text="Pro plan refunds within 30 days.", index=0,
              metadata={"plan": "Pro", "category": "policy"}),
        Chunk(document_id="kb", text="Basic plan refunds within 14 days with a $5 fee.", index=1,
              metadata={"plan": "Basic", "category": "policy"}),
    ]
    index = BM25Index()
    await index.add(chunks)

    # Unqualified field names resolve from chunk metadata (top-level Chunk fields
    # like `kind`/`tenant_id` are reserved, so we use a custom key, `category`).
    spec: FilterSpec = and_(eq("plan", "Pro"), eq("category", "policy"))
    # Same spec compiles to a hosted store's native predicate language...
    print("  FilterSpec -> Pinecone:", spec.to_pinecone())
    # ...and round-trips losslessly as JSON (so it can cross a process boundary).
    revived = FilterSpec.model_validate_json(spec.model_dump_json())
    hits = await index.search("refunds", top_k=5, where=revived)
    print(f"  refunds query under plan=Pro filter -> {len(hits)} hit(s): {hits[0].chunk.text!r}")


# ── 6. Embedders: Matryoshka + multimodal ────────────────────────────────────
async def embedders() -> None:
    """Matryoshka (truncatable) embeddings trade dimension for storage/latency.

    A Matryoshka embedder front-ends any base embedder and lets you truncate the
    output to fewer dimensions — smaller vectors, less storage and faster search,
    for a little recall. We vary the dimension and show byte cost while the top
    hit stays correct. build_embedder(..., dimensions=) wires the same MRL trick
    for hosted (incl. multimodal) embedders.
    """
    banner("6. Embedders — Matryoshka dimension/recall trade-off")
    facts = [
        "Pro plan refunds are available within 30 days of purchase.",
        "The subscription renews automatically unless cancelled 60 days ahead.",
        "All data is encrypted at rest with AES-256 and in transit with TLS 1.3.",
    ]
    chunks = [Chunk(document_id="kb", text=t, index=i) for i, t in enumerate(facts)]
    for dimensions in (256, 128, 64, 32):
        index = VectorIndex(MatryoshkaEmbedder(LocalHashEmbedder(dim=256), dimensions))
        await index.add(chunks)
        hits = await index.search("How is data encrypted at rest and in transit?", top_k=1)
        print(f"  dim={dimensions:>3}  {dimensions * 4:>4} B/vec  top hit: {hits[0].chunk.text[:34]!r}")
    # The factory wires MRL for hosted embedders the same way.
    shrunk = build_embedder("local", dimensions=64)
    print(f"  build_embedder('local', dimensions=64) -> dim {shrunk.dim}")


# ── 7. Multimodal evidence in one packet ─────────────────────────────────────
async def multimodal_packet() -> None:
    """Image, table, and video are first-class scored candidates beside text.

    Retrieval is not text-only. The context compiler treats an image (with
    caption), a table, and a sampled video clip as evidence items with their own
    token cost — scored, budgeted, and citable in the *same* packet as text. A
    video is sampled into a segmented timeline by MockVideoAnalyzer and lowered
    into typed evidence whose citation preserves the time range.
    """
    banner("7. Multimodal evidence — image / table / video in one packet")
    # Sample a clip into a timeline, then lower it into typed video evidence.
    analysis = await MockVideoAnalyzer(segment_seconds=5.0, frames_per_segment=2).analyze("demo.mp4")
    video_ev = video_evidence_items(analysis, source_id="DEMO", video_path="/clips/demo.mp4")

    mixed = [
        EvidenceItem(source_id="d1", text="The Pro plan annual fee is $99.", relevance=0.9),
        EvidenceItem(source_id="d2", modality="image", source_type="image",
                     image=ImageRef(path="/pricing.png", detail="high",
                                    metadata={"caption": "Pro plan pricing chart"})),
        EvidenceItem(source_id="d3", modality="table",
                     table={"columns": ["plan", "fee"], "rows": [["Pro", 99]],
                            "markdown": "| Pro | 99 |"}),
        *video_ev,
    ]
    # _collect lowers heterogeneous evidence into uniformly-scorable candidates.
    candidates = ContextCompiler()._collect(evidence=mixed, memory=[], tool_results=[])
    modalities = sorted({c.modality for c in candidates})
    print(f"  one packet, {len(candidates)} candidates spanning modalities: {modalities}")
    for modality in modalities:
        first = next(c for c in candidates if c.modality == modality)
        print(f"    {modality:<6} token_cost={first.token_cost}")
    # The video citation carries the moment, not just the document.
    print(f"  video clip cites a time range, e.g. {video_ev[-1].citation_ref}")


async def main() -> None:
    # Constructed once so the example is honestly "offline mock provider".
    example_provider()
    await hybrid_fusion()
    await query_understanding()
    chunking_strategies()
    await graphrag()
    await metadata_filter()
    await embedders()
    await multimodal_packet()
    print("\nOne retrieval stack: complementary signals fused, queries understood,")
    print("metadata enforced, and every modality scored and cited in the same packet.")


if __name__ == "__main__":
    asyncio.run(main())
