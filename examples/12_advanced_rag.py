"""Advanced retrieval: learned sparse + late interaction fused with
BM25/dense in one RRF, sentence-window and auto-merging retrieval, query
understanding (HyDE / multi-query / step-back), GraphRAG with global vs
local routing, a live index with TTL, and a SQL connector — fully offline.
"""

import asyncio
import sqlite3

from _shared import citing_responder, example_provider

from vincio import ContextApp
from vincio.connectors import connect
from vincio.core.types import Document
from vincio.retrieval import (
    AutoMergingIndex,
    BM25Index,
    EntityGraph,
    GraphRAG,
    LateInteractionIndex,
    LiveIndex,
    RetrievalEngine,
    SparseIndex,
    chunk_document,
)

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


async def four_way_fusion() -> None:
    print("— learned sparse + late interaction fused with BM25/dense (one RRF) —")
    chunks = [c for doc in DOCS for c in chunk_document(doc, strategy="recursive", size=60)]
    bm25, sparse, late = BM25Index(), SparseIndex(), LateInteractionIndex()
    for index in (bm25, sparse, late):
        await index.add(chunks)
    engine = RetrievalEngine([bm25, sparse, late])
    # "refunding" has no exact token match — the sparse stem expansion and
    # late-interaction token matching still land on the refund policy.
    result = await engine.retrieve("refunding a purchase", top_k=2, use_planner=False)
    for item in result.evidence:
        print(f"  [{item.id}] {(item.text or '')[:80]}...")


async def query_understanding() -> None:
    print("\n— query understanding: HyDE / multi-query / step-back as fusion inputs —")
    chunks = [c for doc in DOCS for c in chunk_document(doc, strategy="recursive", size=60)]
    bm25 = BM25Index()
    await bm25.add(chunks)
    engine = RetrievalEngine([bm25], query_strategies=["hyde", "multi_query", "step_back"])
    result = await engine.retrieve("What is the refund window for the Pro plan?", top_k=2)
    for expansion in result.plan.expansions:
        print(f"  {expansion.strategy}: {expansion.queries}")
    print(f"  fused queries={result.metadata['queries']} -> top hit: {result.evidence[0].id}")


async def auto_merging() -> None:
    print("\n— hierarchical chunks with auto-merging (parent-document retrieval) —")
    [doc] = [d for d in DOCS if d.title == "Refund Policy"]
    chunks = chunk_document(doc, strategy="hierarchical", size=20)
    index = AutoMergingIndex(BM25Index())
    await index.add(chunks)
    hits = await index.search("refunds", top_k=3)
    for hit in hits[:2]:
        level = hit.chunk.metadata.get("level", "child")
        print(f"  ({hit.source}, {level}) {hit.chunk.text[:80]}...")


async def graphrag() -> None:
    print("\n— GraphRAG: communities, summaries, global vs local routing —")
    graph = EntityGraph()
    for doc in DOCS:
        graph.add_chunks(chunk_document(doc, strategy="recursive", size=40))
    rag = GraphRAG(graph)
    communities = await rag.build()
    print(f"  communities={len(communities)}")
    for query in (
        "What are the main themes across these documents?",
        "What did Acme Corp agree with Beta LLC?",
    ):
        mode = rag.route(query)
        evidence = await rag.retrieve(query, top_k=2)
        head = (evidence[0].text or "")[:70] if evidence else "(none)"
        print(f"  [{mode}] {query}\n        -> {head}...")


async def live_index() -> None:
    print("\n— live index: upsert + TTL expiry + freshness metadata —")
    chunks = [c for doc in DOCS for c in chunk_document(doc, strategy="recursive", size=60)]
    live = LiveIndex(BM25Index())
    await live.upsert(chunks)
    flash = chunk_document(
        Document(title="Status", text="Maintenance window tonight; uptime credits paused."),
        strategy="recursive",
        size=60,
    )
    await live.upsert(flash, ttl_seconds=0)  # already stale: vanishes on next search
    hits = await live.search("uptime credits", top_k=3)
    print(f"  after TTL purge: {len(live)} chunks, top hit: {hits[0].chunk.text[:60]}...")
    print(f"  freshness stamp: indexed_at={hits[0].chunk.metadata['indexed_at'][:19]}")


def sql_connector_app() -> None:
    print("\n— SQL connector feeding a full app (hybrid_full retrieval) —")
    # check_same_thread=False: app.add_source loads connectors off-thread.
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.execute("CREATE TABLE faq (id INTEGER, question TEXT, answer TEXT)")
    connection.executemany(
        "INSERT INTO faq VALUES (?, ?, ?)",
        [
            (1, "What is the refund window?", "Pro plan refunds within 30 days."),
            (2, "How is uptime guaranteed?", "The SLA guarantees 99.9 percent monthly uptime."),
        ],
    )
    provider, model = example_provider(
        citing_responder("The Pro plan refund window is 30 days. [{ref}]")
    )
    app = ContextApp(name="advanced_rag", provider=provider, model=model)
    app.add_source(
        "faq",
        connector=connect(
            "sql",
            query="SELECT * FROM faq",
            connection=connection,
            id_column="id",
            title_column="question",
        ),
        retrieval="hybrid_full",
    )
    app.set_policy("answer_only_from_sources", True)
    result = app.run("What is the refund window for the Pro plan?")
    print(f"  answer: {result.output}")
    print(f"  citations: {result.citations}")


async def main() -> None:
    await four_way_fusion()
    await query_understanding()
    await auto_merging()
    await graphrag()
    await live_index()
    sql_connector_app()


if __name__ == "__main__":
    asyncio.run(main())
