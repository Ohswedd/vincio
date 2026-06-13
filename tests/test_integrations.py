"""0.9 integration breadth: OpenAI-compatible providers, hosted rerankers /
embedders, and the vector-store factory. All offline via httpx MockTransport."""

from __future__ import annotations

import httpx
import pytest

from vincio.core.errors import ConfigError, StorageError
from vincio.core.types import Chunk
from vincio.providers import build_provider, default_registry, openai_compatible
from vincio.providers.base import RetryingProvider
from vincio.providers.openai_compat import PRESETS, OpenAICompatibleProvider
from vincio.retrieval import (
    CohereEmbedder,
    CohereReranker,
    JinaEmbedder,
    LocalHashEmbedder,
    VoyageEmbedder,
    VoyageReranker,
    build_embedder,
)
from vincio.retrieval.indexes import SearchHit
from vincio.retrieval.rerankers import build_reranker
from vincio.storage import VECTOR_BACKENDS, build_vector_index


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _hits(texts: list[str]) -> list[SearchHit]:
    return [
        SearchHit(chunk=Chunk(document_id="d", text=text, index=i), score=0.1, source="bm25")
        for i, text in enumerate(texts)
    ]


# -- OpenAI-compatible providers ------------------------------------------------


def test_all_presets_registered():
    names = set(default_registry().names)
    assert set(PRESETS) <= names
    assert "openai_compat" in names


def test_openai_compatible_preset_sets_base_url_and_name():
    provider = openai_compatible("groq", api_key="k")
    assert provider.name == "groq"
    assert provider.base_url == "https://api.groq.com/openai/v1"


def test_openai_compatible_requires_base_url_without_preset():
    with pytest.raises(ConfigError):
        openai_compatible()
    with pytest.raises(ConfigError):
        OpenAICompatibleProvider(api_key="k")  # no base_url


def test_openai_compatible_unknown_preset():
    with pytest.raises(ConfigError):
        openai_compatible("not-a-real-gateway")


def test_build_provider_resolves_preset_env_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-key")
    provider = build_provider("groq")
    assert isinstance(provider, RetryingProvider)
    inner = provider.inner
    assert inner.name == "groq"
    assert inner.api_key == "secret-key"
    assert inner.base_url == "https://api.groq.com/openai/v1"


# -- hosted rerankers -----------------------------------------------------------


@pytest.mark.asyncio
async def test_cohere_reranker_reorders_by_relevance():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"results": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4}]},
        )

    reranker = CohereReranker(api_key="k", client=_mock_client(handler))
    out = await reranker.rerank("q", _hits(["a", "b", "c"]), top_k=2)
    assert [h.chunk.text for h in out] == ["c", "a"]
    assert seen["payload"]["top_n"] == 2
    assert seen["payload"]["model"] == "rerank-v3.5"


@pytest.mark.asyncio
async def test_voyage_reranker_uses_data_key_and_top_k():
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content)
        assert "top_k" in payload  # Voyage names the field top_k, not top_n
        return httpx.Response(200, json={"data": [{"index": 1, "relevance_score": 0.8}]})

    reranker = VoyageReranker(api_key="k", client=_mock_client(handler))
    out = await reranker.rerank("q", _hits(["a", "b"]), top_k=1)
    assert [h.chunk.text for h in out] == ["b"]


@pytest.mark.asyncio
async def test_reranker_empty_hits_short_circuits():
    reranker = CohereReranker(api_key="k")
    assert await reranker.rerank("q", [], top_k=5) == []


def test_build_reranker_dispatches_http():
    assert isinstance(build_reranker("cohere", api_key="k"), CohereReranker)
    with pytest.raises(ValueError):
        build_reranker("nope")


# -- hosted embedders -----------------------------------------------------------


@pytest.mark.asyncio
async def test_jina_embedder_openai_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1, 0.2]}, {"index": 1, "embedding": [0.3, 0.4]}]},
        )

    embedder = JinaEmbedder(api_key="k", client=_mock_client(handler))
    vectors = await embedder.embed(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert embedder.dim == 2


@pytest.mark.asyncio
async def test_cohere_embedder_v2_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content)
        assert payload["input_type"] == "search_document"
        return httpx.Response(200, json={"embeddings": {"float": [[1.0, 2.0, 3.0]]}})

    embedder = CohereEmbedder(api_key="k", client=_mock_client(handler))
    assert await embedder.embed(["x"]) == [[1.0, 2.0, 3.0]]


@pytest.mark.asyncio
async def test_voyage_embedder_and_empty():
    embedder = VoyageEmbedder(api_key="k")
    assert await embedder.embed([]) == []


def test_build_embedder_variants():
    assert isinstance(build_embedder("local"), LocalHashEmbedder)
    assert isinstance(build_embedder("jina", api_key="k"), JinaEmbedder)
    with pytest.raises(ConfigError):
        build_embedder("definitely-not-a-provider")


# -- vector-store factory -------------------------------------------------------


def test_vector_factory_memory_backend():
    index = build_vector_index("memory", LocalHashEmbedder())
    assert len(index) == 0
    assert index.name == "vector"


@pytest.mark.parametrize("backend", ["chroma", "pinecone", "lancedb"])
def test_vector_factory_missing_dep_is_helpful(backend):
    with pytest.raises(StorageError):
        build_vector_index(backend, LocalHashEmbedder())


def test_vector_factory_pgvector_needs_dsn():
    with pytest.raises(ConfigError):
        build_vector_index("pgvector", LocalHashEmbedder())


def test_vector_factory_unknown_backend():
    with pytest.raises(ConfigError):
        build_vector_index("weaviate", LocalHashEmbedder())
    assert "memory" in VECTOR_BACKENDS
