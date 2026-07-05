"""Retrieval & RAG depth — the whole stack on one offline corpus.

Retrieval is where a context engine earns its keep. This tour walks it end to
end on the deterministic mock: four-way hybrid fusion (BM25 + dense +
learned-sparse + late-interaction in one RRF), query understanding
(HyDE / multi-query / decompose), chunking as a retrieval knob, GraphRAG
global/local routing, serializable FilterSpec metadata pushdown, Matryoshka
embedders, image/table/video as first-class scored evidence, and reasoning
retrieval that retrieves by the facts a decision needs and reports the gaps.
No API keys, no network.
"""

from __future__ import annotations

import asyncio

from vincio import ContextApp
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

# A small corpus whose mix of exact terms ("uptime"), stems ("refund"/"refunding")
# and entities ("Acme Corp"/"Beta LLC") deliberately exercises every signal.
DOCS = [
    Document(title="Refund Policy",
             text="Customers on the Pro plan may request refunds within 30 days of purchase. "
             "Basic plan refunds incur a $5 processing fee and must be requested within 14 days. "
             "Refunds are processed by the billing team within five business days."),
    Document(title="Terms of Service",
             text="The subscription renews automatically unless terminated 60 days before the "
             "renewal date. The initial term is 24 months. Acme Corp contracts Beta LLC to "
             "operate the billing platform under a Master Service Agreement."),
    Document(title="SLA",
             text="The service level agreement guarantees 99.9 percent monthly uptime. Credits of "
             "10 percent apply for each hour of downtime beyond the threshold."),
]


async def main() -> None:
    chunks = [c for doc in DOCS for c in chunk_document(doc, strategy="recursive", size=60)]

    # 1. Four-way hybrid fusion. Each index captures a different signal: BM25 =
    #    exact lexical, dense = semantic, learned-sparse = term expansion,
    #    late-interaction = fine-grained token match. RetrievalEngine runs them
    #    concurrently and merges with reciprocal-rank fusion, so a query that
    #    matches NO index perfectly can still rank the right chunk first.
    #    use_planner=False isolates fusion from query rewriting.
    indexes = [BM25Index(), SparseIndex(), LateInteractionIndex(),
               VectorIndex(LocalHashEmbedder(dim=128))]  # offline deterministic vectors
    for index in indexes:
        await index.add(chunks)
    fused = await RetrievalEngine(indexes).retrieve("refunding a purchase", top_k=2, use_planner=False)
    print("1. hybrid fusion (no exact match):", [it.id for it in fused.evidence])

    # 2. Query understanding rewrites one question into many BEFORE fusion — HyDE
    #    drafts a hypothetical answer to search with, multi-query paraphrases,
    #    decompose splits a compound question. All rewrites are retrieved and
    #    fused, widening recall without "search-engine style" phrasing.
    bm25 = BM25Index()
    await bm25.add(chunks)
    engine = RetrievalEngine([bm25], query_strategies=["hyde", "multi_query", "decompose"])
    result = await engine.retrieve("Compare the refund windows across plans and the SLA credit", top_k=2)
    print("2. query understanding:",
          {e.strategy: e.queries for e in result.plan.expansions},
          f"-> {result.metadata['queries']} queries fused")

    # 3. Chunking is a retrieval knob, not an afterthought: the same document
    #    yields different evidence units per strategy. 'recursive' respects
    #    sentence/paragraph boundaries; 'hierarchical' emits parent+child chunks
    #    (small for precise matching, large parents for context).
    [refund_doc] = [d for d in DOCS if d.title == "Refund Policy"]
    print("3. chunking granularity:")
    for strategy, size in (("recursive", 60), ("recursive", 30), ("hierarchical", 20)):
        cs = chunk_document(refund_doc, strategy=strategy, size=size)
        levels = sorted({c.metadata.get("level", "flat") for c in cs})
        print(f"     {strategy:<12} size={size:<3} -> {len(cs):>2} chunks levels={levels}")

    # 4. GraphRAG extracts entities/relations into a graph, clusters them into
    #    communities and summarizes each. A GLOBAL question answers from community
    #    summaries; a LOCAL one traverses the graph around named entities.
    #    route() picks the mode automatically — one index serves both shapes.
    graph = EntityGraph()
    for doc in DOCS:
        graph.add_chunks(chunk_document(doc, strategy="recursive", size=40))
    rag = GraphRAG(graph)
    await rag.build()
    print("4. GraphRAG routing:")
    for query in ("What are the main themes across these documents?",
                  "What did Acme Corp agree with Beta LLC?"):
        evidence = await rag.retrieve(query, top_k=2)
        head = (evidence[0].text or "")[:48] if evidence else "(none)"
        print(f"     [{rag.route(query):<6}] {head}...")

    # 5. FilterSpec is a serializable boolean tree (and_/or_/eq...) of HARD
    #    metadata constraints — tenancy, kind, field values. It compiles to a
    #    hosted store's native predicate language AND round-trips as JSON, so the
    #    same filter crosses a process boundary and filters local indexes via where=.
    kb = [Chunk(document_id="kb", text="Pro plan refunds within 30 days.", index=0,
                metadata={"plan": "Pro", "category": "policy"}),
          Chunk(document_id="kb", text="Basic plan refunds within 14 days with a $5 fee.", index=1,
                metadata={"plan": "Basic", "category": "policy"})]
    index = BM25Index()
    await index.add(kb)
    spec: FilterSpec = and_(eq("plan", "Pro"), eq("category", "policy"))
    revived = FilterSpec.model_validate_json(spec.model_dump_json())  # JSON round-trip
    hits = await index.search("refunds", top_k=5, where=revived)
    print("5. FilterSpec pushdown (plan=Pro):", spec.to_pinecone(), "->", hits[0].chunk.text)

    # 6. Matryoshka embeddings are truncatable: front-end any base embedder and
    #    cut the output to fewer dimensions — smaller vectors, less storage, faster
    #    search, for a little recall. The top hit stays correct as bytes shrink.
    facts = ["Pro plan refunds are available within 30 days of purchase.",
             "The subscription renews automatically unless cancelled 60 days ahead.",
             "All data is encrypted at rest with AES-256 and in transit with TLS 1.3."]
    mrl_chunks = [Chunk(document_id="kb", text=t, index=i) for i, t in enumerate(facts)]
    print("6. Matryoshka dimension/recall:")
    for dimensions in (256, 64):
        idx = VectorIndex(MatryoshkaEmbedder(LocalHashEmbedder(dim=256), dimensions))
        await idx.add(mrl_chunks)
        top = await idx.search("How is data encrypted at rest and in transit?", top_k=1)
        print(f"     dim={dimensions:>3} ({dimensions * 4:>4}B/vec) -> {top[0].chunk.text[:32]!r}")
    print("     build_embedder('local', dimensions=64).dim =", build_embedder("local", dimensions=64).dim)

    # 7. Multimodal: image (with caption), table and a sampled video clip are
    #    scored, budgeted, citable evidence in the SAME packet as text, each with
    #    its own token cost. A video's citation preserves the time range, not just
    #    the document. _collect lowers heterogeneous evidence to scorable candidates.
    analysis = await MockVideoAnalyzer(segment_seconds=5.0, frames_per_segment=2).analyze("demo.mp4")
    video_ev = video_evidence_items(analysis, source_id="DEMO", video_path="/clips/demo.mp4")
    mixed = [
        EvidenceItem(source_id="d1", text="The Pro plan annual fee is $99.", relevance=0.9),
        EvidenceItem(source_id="d2", modality="image", source_type="image",
                     image=ImageRef(path="/pricing.png", detail="high",
                                    metadata={"caption": "Pro plan pricing chart"})),
        EvidenceItem(source_id="d3", modality="table",
                     table={"columns": ["plan", "fee"], "rows": [["Pro", 99]], "markdown": "| Pro | 99 |"}),
        *video_ev,
    ]
    candidates = ContextCompiler()._collect(evidence=mixed, memory=[], tool_results=[])
    print("7. one packet, modalities:", sorted({c.modality for c in candidates}),
          "| video cites a moment:", video_ev[-1].citation_ref)

    # 8. Reasoning retrieval inverts top-k: a top-k by similarity can return four
    #    passages restating the same fact while the decision still hangs on one
    #    nobody retrieved. Here a fact list declares what the task needs;
    #    retrieve_facts runs a targeted retrieval per uncovered fact and reports a
    #    `complete` flag that stays False while any required fact is missing.
    app = ContextApp("retrieval-demo", provider="mock")
    app.add_source("kb", documents=DOCS)
    facts_result = app.retrieve_facts(
        "Can a Pro-plan customer get a refund, and what SLA credit applies?",
        facts=["refund_policy", "sla_credit", "dispute_status"],  # last one is uncovered
    )
    covered = {c.fact: c.covered for c in facts_result.coverage}
    print("8. reasoning retrieval coverage:", covered,
          f"| complete={facts_result.complete} missing={facts_result.missing_facts}")


if __name__ == "__main__":
    asyncio.run(main())
