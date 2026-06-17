"""Quantization + two-stage retrieval and batteries-included local neural models
(2.1). The two-stage index and quantization functions are exercised exactly; the
optional-dependency local models are exercised through injection and the offline
fallback path (the heavy deps are not installed in CI)."""

from __future__ import annotations

import importlib.util

import pytest

from vincio.core.errors import ConfigError
from vincio.core.types import Chunk, Message, ModelRequest
from vincio.providers import GGUFProvider
from vincio.retrieval import (
    ColBERTTokenEmbedder,
    FastEmbedEmbedder,
    LocalCrossEncoderReranker,
    LocalHashEmbedder,
    SearchHit,
    SpladeEncoder,
    TwoStageIndex,
    VectorIndex,
    binary_similarity,
    build_embedder,
    build_reranker,
    eq,
    quantize_binary,
    quantize_scalar,
    scalar_similarity,
)

_DOCS = [
    Chunk(id="c0", document_id="d", text="The refund window is 30 days for the Pro plan."),
    Chunk(id="c1", document_id="d", text="Password reset links expire after one hour."),
    Chunk(id="c2", document_id="d", text="Enterprise contracts include a dedicated account manager."),
    Chunk(id="c3", document_id="d", text="Our refund policy covers thirty days from purchase."),
]


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------


class TestQuantization:
    def test_scalar_quantization(self):
        q = quantize_scalar([1.0, -1.0, 0.5, 0.0])
        assert q == [127, -127, 64, 0]
        assert scalar_similarity(q, q) > 0

    def test_binary_quantization(self):
        a = quantize_binary([0.3, -0.1, 0.0, -2.0])
        assert a == [1, 0, 1, 0]
        assert binary_similarity(a, a) == 1.0
        assert binary_similarity(a, [0, 1, 0, 1]) == 0.0

    def test_unknown_quantization_rejected(self):
        with pytest.raises(ValueError, match="quantization"):
            TwoStageIndex(quantization="float4")


# ---------------------------------------------------------------------------
# Two-stage retrieval
# ---------------------------------------------------------------------------


class TestTwoStageIndex:
    async def test_matches_exact_when_all_candidates_reranked(self):
        embedder = LocalHashEmbedder(dim=128)
        exact = VectorIndex(embedder=embedder)
        two_stage = TwoStageIndex(embedder=embedder, quantization="scalar", rerank_factor=10)
        await exact.add(_DOCS)
        await two_stage.add(_DOCS)
        for query in ["refund window", "password reset", "account manager"]:
            top_exact = (await exact.search(query, top_k=1))[0]
            top_two = (await two_stage.search(query, top_k=1))[0]
            assert top_exact.chunk.id == top_two.chunk.id  # recall preserved

    async def test_matryoshka_coarse_then_full_rerank(self):
        two_stage = TwoStageIndex(
            embedder=LocalHashEmbedder(dim=128), coarse_dims=32, quantization="binary", rerank_factor=3
        )
        await two_stage.add(_DOCS)
        hits = await two_stage.search("refund policy", top_k=2)
        assert hits and hits[0].chunk.id in {"c0", "c3"}  # the refund chunks
        assert all(isinstance(h, SearchHit) for h in hits)

    async def test_filter_pushdown(self):
        chunks = [
            Chunk(id="a", document_id="d", text="alpha refund", tenant_id="t1"),
            Chunk(id="b", document_id="d", text="alpha refund", tenant_id="t2"),
        ]
        index = TwoStageIndex(quantization="scalar")
        await index.add(chunks)
        hits = await index.search("alpha refund", top_k=5, where=eq("tenant_id", "t1"))
        assert {h.chunk.id for h in hits} == {"a"}

    async def test_delete(self):
        index = TwoStageIndex()
        await index.add(_DOCS)
        assert len(index) == 4
        assert await index.delete(["c0"]) == 1
        assert len(index) == 3


# ---------------------------------------------------------------------------
# Local neural models: injection + offline fallback
# ---------------------------------------------------------------------------


class TestLocalEmbedders:
    async def test_fastembed_injected_model(self):
        emb = FastEmbedEmbedder(encode_fn=lambda texts: [[float(len(t))] * 4 for t in texts])
        vecs = await emb.embed(["abc"])
        assert vecs == [[3.0, 3.0, 3.0, 3.0]]

    async def test_fastembed_fallback_offline(self):
        emb = FastEmbedEmbedder(dim=16, fallback=True)
        vecs = await emb.embed(["hello world"])
        assert len(vecs) == 1 and len(vecs[0]) == 16

    def test_fastembed_requires_dep_without_fallback(self):
        if importlib.util.find_spec("fastembed") is not None:
            pytest.skip("fastembed is installed")
        with pytest.raises(ConfigError):
            FastEmbedEmbedder()._ensure()

    async def test_colbert_token_embedder_fallback(self):
        emb = ColBERTTokenEmbedder(dim=8)  # fallback=True by default
        vecs = await emb.embed(["token one", "token two"])
        assert len(vecs) == 2 and len(vecs[0]) == 8

    def test_build_embedder_fastembed(self):
        emb = build_embedder("fastembed", fallback=True)
        assert isinstance(emb, FastEmbedEmbedder)


class TestSpladeEncoder:
    async def test_injected(self):
        enc = SpladeEncoder(encode_fn=lambda texts, is_query: [{"refund": 2.0} for _ in texts])
        out = await enc.encode(["refund please"])
        assert out == [{"refund": 2.0}]

    async def test_fallback_offline(self):
        enc = SpladeEncoder(fallback=True)
        out = await enc.encode(["refund window thirty days"])
        assert out and "refund" in out[0]

    def test_requires_dep_without_fallback(self):
        if importlib.util.find_spec("transformers") is not None:
            pytest.skip("transformers is installed")
        with pytest.raises(ConfigError):
            SpladeEncoder()._ensure()


class TestLocalCrossEncoder:
    async def test_injected_score_fn(self):
        async def score(query, passages):
            return [float(len(p)) for p in passages]

        reranker = LocalCrossEncoderReranker(score_fn=score)
        hits = [
            SearchHit(chunk=Chunk(id="a", document_id="d", text="short"), score=0.1),
            SearchHit(chunk=Chunk(id="b", document_id="d", text="a much longer passage"), score=0.1),
        ]
        ranked = await reranker.rerank("q", hits, top_k=2)
        assert ranked[0].chunk.id == "b"  # longest scored highest

    async def test_fallback_to_heuristic(self):
        reranker = LocalCrossEncoderReranker(fallback=True)
        hits = [SearchHit(chunk=Chunk(id="a", document_id="d", text="refund policy"), score=0.5)]
        ranked = await reranker.rerank("refund", hits, top_k=1)
        assert ranked and ranked[0].chunk.id == "a"

    def test_build_reranker_local(self):
        reranker = build_reranker("cross-encoder", fallback=True)
        assert isinstance(reranker, LocalCrossEncoderReranker)


# ---------------------------------------------------------------------------
# GGUF / llama.cpp in-process provider
# ---------------------------------------------------------------------------


class _FakeLlama:
    def create_chat_completion(self, messages, **kwargs):
        self.last_messages = messages
        return {
            "choices": [{"message": {"content": "hi from gguf"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

    def embed(self, text):
        return [float(len(text))] * 8


class TestGGUFProvider:
    async def test_generate(self):
        provider = GGUFProvider(llama=_FakeLlama())
        resp = await provider.generate(
            ModelRequest(model="local-gguf", messages=[Message(role="user", content="hello")])
        )
        assert resp.text == "hi from gguf"
        assert resp.usage.input_tokens == 3 and resp.usage.output_tokens == 2
        assert resp.provider == "gguf"

    async def test_on_device_embedding(self):
        provider = GGUFProvider(llama=_FakeLlama())
        vecs = await provider.embed(["abc"])
        assert vecs == [[3.0] * 8]

    def test_requires_dep_or_injection(self):
        if importlib.util.find_spec("llama_cpp") is not None:
            pytest.skip("llama_cpp is installed")
        with pytest.raises(ConfigError):
            GGUFProvider()._ensure()
