"""Retrieval engine unit tests (retriever merge)."""

import pytest

from vincio.core.types import Chunk, Document
from vincio.retrieval import (
    BM25Index,
    EntityGraph,
    FactSchema,
    HeuristicReranker,
    LocalHashEmbedder,
    ReasoningRetriever,
    RetrievalEngine,
    VectorIndex,
    build_filter,
    chunk_document,
    cosine,
    reciprocal_rank_fusion,
)
from vincio.retrieval.indexes import SearchHit


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


class TestChunking:
    def test_strategies_produce_chunks(self, sample_document):
        for strategy in ("fixed", "recursive", "semantic", "adaptive"):
            chunks = chunk_document(sample_document, strategy=strategy, size=40)
            assert chunks, strategy
            assert all(c.document_id == sample_document.id for c in chunks)

    def test_heading_chunker_uses_sections(self):
        doc = Document(
            text="ignored",
            sections=[
                {"title": "Intro", "level": 1, "path": ["Intro"], "text": "Welcome text.", "start_line": 0},
                {"title": "Refunds", "level": 2, "path": ["Intro", "Refunds"], "text": "30 day window.", "start_line": 5},
            ],
        )
        chunks = chunk_document(doc, strategy="heading_aware")
        assert any("Intro > Refunds" in c.text for c in chunks)

    def test_code_chunker_splits_symbols(self):
        doc = Document(
            text="import os\n\ndef alpha():\n    return 1\n\nclass Beta:\n    def method(self):\n        return 2\n",
            media_type="text/x-python",
            metadata={"language": "python"},
        )
        chunks = chunk_document(doc, strategy="code_aware")
        symbols = {c.metadata.get("symbol") for c in chunks}
        assert "alpha" in symbols

    def test_unknown_strategy_raises(self, sample_document):
        with pytest.raises(ValueError):
            chunk_document(sample_document, strategy="nope")


class TestIndexes:
    @pytest.mark.asyncio
    async def test_bm25_ranking(self):
        index = BM25Index()
        await index.add(make_chunks())
        hits = await index.search("refund window Pro plan", top_k=2)
        assert hits[0].chunk.id == "c0"

    @pytest.mark.asyncio
    async def test_vector_ranking(self):
        index = VectorIndex(LocalHashEmbedder())
        await index.add(make_chunks())
        hits = await index.search("automatic renewal termination", top_k=2)
        assert hits[0].chunk.id == "c1"

    @pytest.mark.asyncio
    async def test_delete(self):
        index = BM25Index()
        await index.add(make_chunks())
        assert await index.delete(["c0"]) == 1
        hits = await index.search("refund", top_k=4)
        assert all(h.chunk.id != "c0" for h in hits)

    @pytest.mark.asyncio
    async def test_metadata_filter(self):
        chunks = make_chunks()
        chunks[0].tenant_id = "acme"
        chunks[1].tenant_id = "other"
        index = BM25Index()
        await index.add(chunks)
        hits = await index.search("refund renewal", top_k=4, where=build_filter(tenant_id="acme"))
        assert all(h.chunk.tenant_id in (None, "acme") for h in hits)

    def test_embedder_semantics(self):
        embedder = LocalHashEmbedder()
        related = cosine(
            embedder.embed_one("terminate the contract"), embedder.embed_one("contract termination")
        )
        unrelated = cosine(
            embedder.embed_one("terminate the contract"), embedder.embed_one("banana smoothie")
        )
        assert related > unrelated


class TestHybrid:
    def test_rrf_merges(self):
        chunks = make_chunks()
        list_a = [SearchHit(chunk=chunks[0], score=5.0), SearchHit(chunk=chunks[1], score=4.0)]
        list_b = [SearchHit(chunk=chunks[1], score=0.9), SearchHit(chunk=chunks[2], score=0.8)]
        merged = reciprocal_rank_fusion([list_a, list_b])
        assert merged[0].chunk.id == "c1"  # appears in both lists

    @pytest.mark.asyncio
    async def test_engine_end_to_end(self):
        chunks = make_chunks()
        bm25, vector = BM25Index(), VectorIndex(LocalHashEmbedder())
        await bm25.add(chunks)
        await vector.add(chunks)
        engine = RetrievalEngine([bm25, vector], reranker=HeuristicReranker())
        result = await engine.retrieve("What is the refund window for the Pro plan?", top_k=2)
        assert result.evidence
        assert result.evidence[0].source_id == "d1"
        assert "refund" in result.evidence[0].text.lower()

    @pytest.mark.asyncio
    async def test_dedup_in_engine(self):
        chunks = make_chunks()
        duplicate = chunks[0].model_copy(update={"id": "dup0"})
        bm25 = BM25Index()
        await bm25.add(chunks + [duplicate])
        engine = RetrievalEngine([bm25])
        result = await engine.retrieve("refund window Pro plan", top_k=4)
        texts = [e.text for e in result.evidence]
        assert len(texts) == len(set(texts))


class TestGraphAndReasoning:
    def test_entity_graph_retrieval(self):
        doc = Document(
            text="Acme Corp signed a Master Service Agreement with Beta LLC. "
            "The MSA includes a termination clause requiring 60 days notice. "
            "Beta LLC operates the billing platform for Acme Corp."
        )
        chunks = chunk_document(doc, strategy="recursive", size=30)
        graph = EntityGraph()
        graph.add_chunks(chunks)
        assert len(graph) > 0
        evidence = graph.retrieve("What did Acme Corp agree with Beta LLC?")
        assert evidence
        paths = graph.paths_between("Acme Corp", "Beta LLC")
        assert paths

    @pytest.mark.asyncio
    async def test_reasoning_retrieval_reports_missing_facts(self):
        chunks = make_chunks()
        bm25 = BM25Index()
        await bm25.add(chunks)
        engine = RetrievalEngine([bm25])
        retriever = ReasoningRetriever(engine)
        schema = FactSchema.from_names(
            "refund_decision", ["refund_policy", "dispute_status"]
        )
        evidence, coverages, report = await retriever.retrieve(
            "Can the customer get a refund?", schema
        )
        coverage_map = {c.fact: c.covered for c in coverages}
        assert coverage_map["refund_policy"] is True
        assert coverage_map["dispute_status"] is False
        assert report["missing_facts"] == ["dispute_status"]
