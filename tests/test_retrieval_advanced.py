"""Retrieval features: learned sparse, late interaction, hierarchical
indexing, query understanding, GraphRAG, live indexes."""

import pytest

from vincio.core.types import Chunk, Document
from vincio.core.utils import utcnow
from vincio.providers import MockProvider
from vincio.retrieval import (
    AutoMergingIndex,
    BM25Index,
    EntityGraph,
    GraphRAG,
    LateInteractionIndex,
    LiveIndex,
    LocalHashEmbedder,
    LocalImpactEncoder,
    QueryUnderstanding,
    RetrievalEngine,
    SparseIndex,
    VectorIndex,
    build_filter,
    chunk_document,
    contextualize_chunks,
    detect_communities,
)


def make_chunks() -> list[Chunk]:
    texts = [
        "Customers on the Pro plan may request refunds within 30 days.",
        "The subscription renews automatically unless terminated 60 days before renewal.",
        "Acme Corp signed a Master Service Agreement with Beta LLC covering billing.",
        "Bananas are rich in potassium and grow in tropical climates.",
    ]
    return [
        Chunk(id=f"c{i}", document_id="d1", text=text, index=i, entities=[])
        for i, text in enumerate(texts)
    ]


class TestSparse:
    def test_encoder_expands_morphology(self):
        encoder = LocalImpactEncoder()
        vector = encoder.encode_one("refunds terminated")
        assert "refunds" in vector
        assert "refund" in vector  # stem expansion
        assert vector["refund"] < vector["refunds"]

    async def test_ranking_matches_variants(self):
        index = SparseIndex()
        await index.add(make_chunks())
        # No surface-form overlap with "refunds": the stem expansion matches.
        hits = await index.search("refunding", top_k=2)
        assert hits and hits[0].chunk.id == "c0"

    async def test_delete_and_filter(self):
        chunks = make_chunks()
        chunks[0].tenant_id = "acme"
        index = SparseIndex()
        await index.add(chunks)
        hits = await index.search("refund", top_k=4, where=build_filter(tenant_id="other"))
        assert all(h.chunk.tenant_id in (None, "other") for h in hits)
        assert await index.delete(["c0"]) == 1
        assert len(index) == 3
        hits = await index.search("refunds", top_k=4)
        assert all(h.chunk.id != "c0" for h in hits)

    async def test_reindex_same_id_replaces(self):
        index = SparseIndex()
        await index.add(make_chunks())
        updated = Chunk(id="c0", document_id="d1", text="Refund window is now 45 days.", index=0)
        await index.add([updated])
        assert len(index) == 4
        hits = await index.search("refund window", top_k=1)
        assert "45 days" in hits[0].chunk.text


class TestLateInteraction:
    async def test_maxsim_ranking(self):
        index = LateInteractionIndex()
        await index.add(make_chunks())
        hits = await index.search("refund window for the Pro plan", top_k=2)
        assert hits[0].chunk.id == "c0"

    async def test_compressed_matches_exact_top1(self):
        chunks = make_chunks()
        exact = LateInteractionIndex()
        compressed = LateInteractionIndex(compressed=True, n_centroids=4)
        await exact.add(chunks)
        await compressed.add(chunks)
        for query in ("refund window Pro plan", "automatic renewal termination", "Acme billing agreement"):
            top_exact = (await exact.search(query, top_k=1))[0]
            top_compressed = (await compressed.search(query, top_k=1))[0]
            assert top_exact.chunk.id == top_compressed.chunk.id, query

    async def test_delete_invalidates_codes(self):
        index = LateInteractionIndex(compressed=True, n_centroids=4)
        await index.add(make_chunks())
        await index.search("refund", top_k=2)  # builds codes
        assert await index.delete(["c0"]) == 1
        hits = await index.search("refund window Pro plan", top_k=4)
        assert all(h.chunk.id != "c0" for h in hits)

    async def test_fuses_in_engine(self):
        chunks = make_chunks()
        bm25, late = BM25Index(), LateInteractionIndex()
        await bm25.add(chunks)
        await late.add(chunks)
        engine = RetrievalEngine([bm25, late])
        result = await engine.retrieve("refund window Pro plan", top_k=2, use_planner=False)
        assert result.evidence
        assert "refund" in result.evidence[0].text.lower()


class TestSentenceWindow:
    def test_chunker_attaches_windows(self):
        doc = Document(
            text="Alpha is first. Beta follows alpha. Gamma is third. Delta ends the doc."
        )
        chunks = chunk_document(doc, strategy="sentence_window")
        assert len(chunks) == 4
        middle = chunks[1]
        assert middle.text == "Beta follows alpha."
        assert "Alpha is first." in middle.metadata["window_text"]
        assert "Delta ends the doc." in middle.metadata["window_text"]

    async def test_engine_returns_window(self):
        doc = Document(
            text="Plans differ by tier. Refunds are allowed within 30 days. Contact support to start one."
        )
        chunks = chunk_document(doc, strategy="sentence_window")
        bm25 = BM25Index()
        await bm25.add(chunks)
        engine = RetrievalEngine([bm25])
        result = await engine.retrieve("refunds allowed", top_k=1, use_planner=False)
        [item] = result.evidence
        assert "Contact support" in (item.text or "")  # window, not just the sentence
        assert item.metadata["matched_sentence"] == "Refunds are allowed within 30 days."


class TestHierarchical:
    def test_chunker_links_children_to_parents(self):
        doc = Document(text=" ".join(f"Sentence number {i} about policies." for i in range(60)))
        chunks = chunk_document(doc, strategy="hierarchical", size=40)
        parents = [c for c in chunks if c.metadata.get("level") == "parent"]
        children = [c for c in chunks if c.metadata.get("level") == "child"]
        assert parents and children
        parent_ids = {p.id for p in parents}
        assert all(c.metadata["parent_id"] in parent_ids for c in children)

    async def test_auto_merge_returns_parent(self):
        parent = Chunk(
            id="p1", document_id="d1", index=0,
            text="Refund policy overview. Refunds within 30 days. Refunds require receipts.",
            metadata={"level": "parent"},
        )
        children = [
            Chunk(id=f"k{i}", document_id="d1", index=i + 1, text=text,
                  metadata={"level": "child", "parent_id": "p1"})
            for i, text in enumerate(
                ["Refund policy overview.", "Refunds within 30 days.", "Refunds require receipts."]
            )
        ]
        other = Chunk(id="x1", document_id="d1", index=9, text="Office plants are watered weekly.",
                      metadata={"level": "child", "parent_id": "p_missing"})
        index = AutoMergingIndex(BM25Index())
        await index.add([parent, *children, other])
        hits = await index.search("refunds", top_k=4)
        assert hits[0].chunk.id == "p1"
        assert hits[0].source == "auto_merge"
        # merged children are absorbed into the parent hit
        assert all(h.chunk.id not in {"k0", "k1", "k2"} for h in hits)

    async def test_no_merge_below_threshold(self):
        parent = Chunk(id="p1", document_id="d1", index=0, text="A B C D",
                       metadata={"level": "parent"})
        children = [
            Chunk(id=f"k{i}", document_id="d1", index=i + 1, text=text,
                  metadata={"level": "child", "parent_id": "p1"})
            for i, text in enumerate(["refunds here", "renewal terms", "uptime credits", "billing dates"])
        ]
        index = AutoMergingIndex(BM25Index(), merge_threshold=0.5)
        await index.add([parent, *children])
        hits = await index.search("refunds", top_k=3)
        assert hits and hits[0].chunk.id == "k0"  # one child of four: no merge


class TestContextual:
    def test_contextual_chunker_prefixes(self):
        doc = Document(
            title="Service Terms",
            text="The plan includes support. " * 30,
        )
        chunks = chunk_document(doc, strategy="contextual", size=40)
        assert chunks
        assert all(c.text.startswith("[Service Terms") for c in chunks)
        assert all("original_text" in c.metadata for c in chunks)

    async def test_llm_contextualization(self):
        doc = Document(title="Policy", text="Refunds within 30 days. Renewal after 24 months.")
        chunks = chunk_document(doc, strategy="recursive", size=10)
        provider = MockProvider(responder=lambda req: {"context": "From the refund policy."})
        await contextualize_chunks(doc, chunks, provider=provider, model="mock-1")
        assert all(c.text.startswith("[From the refund policy.]") for c in chunks)
        assert all(c.metadata["contextualized"] == "llm" for c in chunks)
        # Idempotent: a second pass leaves chunks alone.
        before = [c.text for c in chunks]
        await contextualize_chunks(doc, chunks, provider=provider, model="mock-1")
        assert [c.text for c in chunks] == before

    async def test_heuristic_fallback_without_provider(self):
        doc = Document(title="Policy", text="Refunds within 30 days. Renewal after 24 months.")
        chunks = chunk_document(doc, strategy="recursive", size=10)
        await contextualize_chunks(doc, chunks)
        assert all(c.metadata.get("contextualized") == "heuristic" for c in chunks)


class TestQueryUnderstanding:
    async def test_heuristic_strategies(self):
        qu = QueryUnderstanding()
        expansions = await qu.expand(
            "What is the refund window for the Pro plan?",
            ["hyde", "multi_query", "step_back"],
        )
        by_strategy = {e.strategy: e for e in expansions}
        assert by_strategy["hyde"].hypothetical.startswith("The refund window")
        assert by_strategy["multi_query"].queries
        assert by_strategy["step_back"].queries
        assert "Pro" not in by_strategy["step_back"].queries[0]

    async def test_decompose_splits_compounds(self):
        qu = QueryUnderstanding()
        [expansion] = await qu.expand(
            "What is the refund window and how does renewal work?", ["decompose"]
        )
        assert len(expansion.queries) == 2

    async def test_unknown_strategy_raises(self):
        qu = QueryUnderstanding()
        with pytest.raises(ValueError):
            await qu.expand("anything", ["nope"])

    async def test_llm_backed_expansion(self):
        def responder(request):
            if request.output_schema_name == "hyde_passage":
                return {"passage": "Refunds are available within 30 days on the Pro plan."}
            return {"queries": ["pro plan refund period", "refund eligibility window"]}

        qu = QueryUnderstanding(MockProvider(responder=responder), "mock-1")
        expansions = await qu.expand("refund window?", ["hyde", "multi_query"])
        assert expansions[0].hypothetical.startswith("Refunds are available")
        assert expansions[1].queries == ["pro plan refund period", "refund eligibility window"]

    async def test_engine_applies_strategies(self):
        chunks = make_chunks()
        bm25 = BM25Index()
        await bm25.add(chunks)
        engine = RetrievalEngine([bm25], query_strategies=["hyde", "multi_query"])
        result = await engine.retrieve("What is the refund window for the Pro plan?", top_k=2)
        assert result.metadata["strategies"] == ["hyde", "multi_query"]
        assert result.plan.expansions
        assert result.evidence and "refund" in result.evidence[0].text.lower()


def _contract_graph() -> EntityGraph:
    doc_a = Document(
        text="Acme Corp signed a Master Service Agreement with Beta LLC. "
        "Acme Corp pays Beta LLC monthly. Beta LLC operates billing for Acme Corp."
    )
    doc_b = Document(
        text="Tropical Fruits Ltd ships Cavendish Bananas. "
        "Cavendish Bananas grow for Tropical Fruits Ltd in coastal farms."
    )
    graph = EntityGraph()
    for doc in (doc_a, doc_b):
        graph.add_chunks(chunk_document(doc, strategy="recursive", size=25))
    return graph


class TestGraphRAG:
    def test_communities_separate_clusters(self):
        graph = _contract_graph()
        labels = detect_communities(graph)
        assert labels["acme corp"] == labels["beta llc"]
        assert labels["cavendish bananas"] == labels["tropical fruits ltd"]
        assert labels["acme corp"] != labels["cavendish bananas"]

    async def test_build_and_global_retrieval(self):
        rag = GraphRAG(_contract_graph())
        communities = await rag.build()
        assert communities
        assert all(c.summary for c in communities)
        evidence = await rag.retrieve("What are the main themes across the documents?", top_k=4)
        assert evidence
        assert all(e.metadata["graphrag_mode"] == "global" for e in evidence)
        assert all(e.metadata["member_chunk_ids"] for e in evidence)

    async def test_routing(self):
        rag = GraphRAG(_contract_graph())
        await rag.build()
        assert rag.route("Summarize the main themes across all documents") == "global"
        assert rag.route("What did acme corp agree with beta llc?") == "local"
        local = await rag.retrieve("What did acme corp agree with beta llc?", top_k=4)
        assert local and all(e.metadata["graphrag_mode"] == "local" for e in local)

    async def test_llm_summaries(self):
        provider = MockProvider(responder=lambda req: {"summary": "Acme and Beta share a billing MSA."})
        rag = GraphRAG(_contract_graph(), provider=provider, model="mock-1")
        await rag.build()
        assert any(c.summary == "Acme and Beta share a billing MSA." for c in rag.communities.values())


class TestLiveIndex:
    async def test_upsert_replaces(self):
        live = LiveIndex(BM25Index())
        await live.upsert(make_chunks())
        updated = Chunk(id="c0", document_id="d1", text="Refund window is 45 days now.", index=0)
        await live.upsert([updated])
        assert len(live) == 4
        hits = await live.search("refund window", top_k=1)
        assert "45 days" in hits[0].chunk.text

    async def test_ttl_expiry(self):
        live = LiveIndex(BM25Index())
        chunks = make_chunks()
        await live.upsert(chunks[:1], ttl_seconds=0)
        await live.upsert(chunks[1:])
        hits = await live.search("refund renewal", top_k=4)
        assert all(h.chunk.id != "c0" for h in hits)
        assert len(live) == 3

    async def test_freshness_flows_to_evidence(self):
        live = LiveIndex(BM25Index())
        chunks = make_chunks()
        for chunk in chunks:
            chunk.created_at = utcnow()
        await live.upsert(chunks)
        engine = RetrievalEngine([live])
        result = await engine.retrieve("refunds Pro plan", top_k=1, use_planner=False)
        [item] = result.evidence
        assert "indexed_at" in item.metadata
        assert item.metadata["age_days"] < 1.0

    async def test_vector_migrate_reembeds(self):
        index = VectorIndex(LocalHashEmbedder(dim=256))
        await index.add(make_chunks())
        assert all(len(v) == 256 for v in index.vectors.values())
        migrated = await index.migrate(LocalHashEmbedder(dim=64))
        assert migrated == 4
        assert all(len(v) == 64 for v in index.vectors.values())
        hits = await index.search("refund window Pro plan", top_k=1)
        assert hits[0].chunk.id == "c0"
