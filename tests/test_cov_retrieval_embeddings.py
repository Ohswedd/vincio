"""Real-behavior coverage tests for vincio.retrieval.embeddings.

Everything is deterministic and offline: the LocalHashEmbedder and the
deterministic MockProvider drive model interaction; hosted HTTP embedders are
exercised against an in-process httpx.MockTransport. No mocks/patches.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from vincio.core.errors import ConfigError, ProviderAuthError, ProviderError
from vincio.core.types import ImageRef
from vincio.providers.mock import MockProvider
from vincio.retrieval.embeddings import (
    BatchingEmbedder,
    CachedEmbedder,
    CohereEmbedder,
    CohereMultimodalEmbedder,
    ColBERTTokenEmbedder,
    FastEmbedEmbedder,
    JinaEmbedder,
    LocalHashEmbedder,
    MatryoshkaEmbedder,
    MultimodalInput,
    ProviderEmbedder,
    VoyageContextualEmbedder,
    VoyageEmbedder,
    VoyageMultimodalEmbedder,
    build_embedder,
    cosine,
    embed_texts,
    mrl_truncate,
)


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------- #
# cosine / mrl_truncate                                                        #
# --------------------------------------------------------------------------- #


def test_cosine_zero_on_length_mismatch_and_empty():
    assert cosine([1.0, 2.0], [1.0]) == 0.0
    assert cosine([], [1.0]) == 0.0
    assert cosine([1.0], []) == 0.0


def test_cosine_zero_when_a_vector_is_all_zeros():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_identical_vectors_is_one():
    assert cosine([3.0, 4.0], [3.0, 4.0]) == pytest.approx(1.0)


def test_mrl_truncate_noop_when_dimensions_cover_whole_vector():
    v = [0.1, 0.2, 0.3]
    assert mrl_truncate(v, 3) == v
    assert mrl_truncate(v, 99) == v
    assert mrl_truncate(v, 0) == v


def test_mrl_truncate_keeps_head_and_l2_renormalizes():
    out = mrl_truncate([3.0, 4.0, 99.0, 99.0], 2)
    assert out == pytest.approx([0.6, 0.8])
    assert sum(x * x for x in out) == pytest.approx(1.0)


def test_mrl_truncate_zero_head_avoids_divide_by_zero():
    assert mrl_truncate([0.0, 0.0, 5.0], 2) == [0.0, 0.0]


# --------------------------------------------------------------------------- #
# LocalHashEmbedder                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_local_hash_is_unit_norm_and_deterministic():
    emb = LocalHashEmbedder(dim=64)
    a = await emb.embed(["hello world"])
    b = await emb.embed(["hello world"])
    assert a == b
    assert sum(x * x for x in a[0]) == pytest.approx(1.0)


def test_local_hash_features_drop_non_alnum_tokens():
    emb = LocalHashEmbedder(dim=16)
    # "!!!" has no alnum characters -> contributes no features.
    assert emb._features("!!!") == []
    feats = emb._features("Cat")
    assert "w:cat" in feats
    assert any(f.startswith("t:") for f in feats)


def test_local_hash_empty_text_returns_zero_then_unit_fallback():
    emb = LocalHashEmbedder(dim=8)
    out = emb.embed_one("")
    assert out == [0.0] * 8  # norm fell back to 1.0, vector stays zero


# --------------------------------------------------------------------------- #
# ProviderEmbedder                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_provider_embedder_empty_returns_empty_no_dim_change():
    pe = ProviderEmbedder(MockProvider(embedding_dim=64), dim=1536)
    assert await pe.embed([]) == []
    assert pe.dim == 1536  # untouched on empty input


@pytest.mark.asyncio
async def test_provider_embedder_single_batch_updates_dim():
    pe = ProviderEmbedder(MockProvider(embedding_dim=32), dim=1536, batch_size=64)
    out = await pe.embed(["a", "b"])
    assert len(out) == 2
    assert pe.dim == 32  # learned from the returned vectors


@pytest.mark.asyncio
async def test_provider_embedder_splits_into_batches_preserving_order():
    pe = ProviderEmbedder(MockProvider(embedding_dim=16), batch_size=2, concurrency=2)
    texts = ["t0", "t1", "t2", "t3", "t4"]
    out = await pe.embed(texts)
    # Order preserved across the concurrent batch fan-out.
    direct = await MockProvider(embedding_dim=16).embed(texts)
    assert out == direct
    assert pe.dim == 16


def test_provider_embedder_clamps_batch_and_concurrency_floor():
    pe = ProviderEmbedder(MockProvider(), batch_size=0, concurrency=0)
    assert pe.batch_size == 1
    assert pe.concurrency == 1


# --------------------------------------------------------------------------- #
# CachedEmbedder                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cached_embedder_hit_miss_accounting_and_dim():
    cache = CachedEmbedder(LocalHashEmbedder(dim=32))
    assert cache.dim == 32  # property delegates to inner
    first = await cache.embed(["alpha", "beta"])
    assert cache.misses == 2
    assert cache.hits == 0
    second = await cache.embed(["alpha", "beta"])
    assert second == first
    assert cache.hits == 2  # both served from cache
    assert cache.misses == 2


@pytest.mark.asyncio
async def test_cached_embedder_dedupes_repeated_text_in_one_call():
    cache = CachedEmbedder(LocalHashEmbedder(dim=16))
    out = await cache.embed(["dup", "dup", "dup"])
    assert out[0] == out[1] == out[2]
    assert cache.misses == 1  # only the first unique text was a miss
    assert cache.hits == 2


@pytest.mark.asyncio
async def test_cached_embedder_skips_already_resolved_key_in_same_call():
    cache = CachedEmbedder(LocalHashEmbedder(dim=8))
    warm = (await cache.embed(["dup"]))[0]  # prime the cache
    cache.hits = cache.misses = 0
    out = await cache.embed(["dup", "dup"])  # first cache-hit, second hits `key in resolved`
    assert out == [warm, warm]
    assert cache.misses == 0  # nothing re-embedded
    assert cache.hits == 2


@pytest.mark.asyncio
async def test_cached_embedder_lru_evicts_oldest():
    cache = CachedEmbedder(LocalHashEmbedder(dim=8), max_entries=2)
    await cache.embed(["one"])
    await cache.embed(["two"])
    await cache.embed(["three"])  # evicts "one"
    assert len(cache._cache) == 2
    cache.hits = cache.misses = 0
    await cache.embed(["one"])  # re-embedded -> miss, proving eviction happened
    assert cache.misses == 1


@pytest.mark.asyncio
async def test_cached_embedder_persistent_backend_round_trip():
    class DictBackend:
        def __init__(self) -> None:
            self.store: dict[str, list[float]] = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value, tags=None):
            self.store[key] = value

    backend = DictBackend()
    cold = CachedEmbedder(LocalHashEmbedder(dim=8), backend=backend)
    vec = (await cold.embed(["persisted"]))[0]
    assert any(k.startswith("emb:") for k in backend.store)  # written through

    warm = CachedEmbedder(LocalHashEmbedder(dim=8), backend=backend)
    out = await warm.embed(["persisted"])
    assert out[0] == vec
    assert warm.misses == 0  # served from the shared backend, not re-embedded
    assert warm.hits == 1


@pytest.mark.asyncio
async def test_cached_embedder_input_type_folds_into_key_when_supported():
    class TypedInner:
        dim = 4
        supports_input_type = True

        def __init__(self) -> None:
            self.calls: list[tuple] = []

        async def embed(self, texts, *, input_type=None):
            self.calls.append((tuple(texts), input_type))
            base = 1.0 if input_type == "query" else 0.0
            return [[base, 0.0, 0.0, 0.0] for _ in texts]

    inner = TypedInner()
    cache = CachedEmbedder(inner)
    assert cache.supports_input_type is True
    doc = await cache.embed(["x"], input_type="document")
    qry = await cache.embed(["x"], input_type="query")
    assert doc != qry  # distinct cache keys, never aliased
    assert len(inner.calls) == 2


def test_cached_embed_sync_runs_event_loop():
    cache = CachedEmbedder(LocalHashEmbedder(dim=8))
    out = cache.embed_sync(["sync"])
    assert len(out) == 1
    assert len(out[0]) == 8


# --------------------------------------------------------------------------- #
# BatchingEmbedder                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_batching_embedder_empty_and_passthrough_props():
    be = BatchingEmbedder(LocalHashEmbedder(dim=16))
    assert be.dim == 16
    assert be.supports_input_type is False
    assert await be.embed([]) == []


@pytest.mark.asyncio
async def test_batching_embedder_coalesces_concurrent_calls_into_one_flush():
    class CountingInner:
        dim = 8

        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        async def embed(self, texts):
            self.batches.append(list(texts))
            return [[float(len(t))] * 8 for t in texts]

    inner = CountingInner()
    be = BatchingEmbedder(inner, max_batch=64, window_ms=5.0)
    res = await asyncio.gather(be.embed(["aa"]), be.embed(["bbb"]), be.embed(["aa"]))
    await be.aclose()
    assert be.flushes == 1  # all three coalesced into a single provider call
    # Duplicate "aa" sent once within the flush.
    assert inner.batches == [["aa", "bbb"]]
    assert res[0][0][0] == 2.0
    assert res[1][0][0] == 3.0
    assert res[2][0][0] == 2.0  # duplicate got the shared vector


@pytest.mark.asyncio
async def test_batching_embedder_flushes_when_batch_fills():
    inner = LocalHashEmbedder(dim=8)
    be = BatchingEmbedder(inner, max_batch=2, window_ms=1000.0)
    # Two texts in one call hits max_batch -> immediate flush, no timer wait.
    out = await be.embed(["x", "y"])
    assert be.flushes == 1
    assert len(out) == 2
    await be.aclose()


@pytest.mark.asyncio
async def test_batching_embedder_delayed_flush_via_timer():
    inner = LocalHashEmbedder(dim=8)
    be = BatchingEmbedder(inner, max_batch=64, window_ms=2.0)
    out = await be.embed(["solo"])  # under max_batch -> timer flush
    assert len(out) == 1
    assert be.flushes == 1


@pytest.mark.asyncio
async def test_batching_embedder_propagates_inner_error_to_waiters():
    class Boom:
        dim = 4

        async def embed(self, texts):
            raise ProviderError("embed blew up", provider="boom")

    be = BatchingEmbedder(Boom(), max_batch=2, window_ms=1000.0)
    with pytest.raises(ProviderError, match="embed blew up"):
        await be.embed(["a", "b"])


@pytest.mark.asyncio
async def test_batching_embedder_flush_reraises_cancelled_error():
    class Cancels:
        dim = 4

        async def embed(self, texts):
            raise asyncio.CancelledError

    be = BatchingEmbedder(Cancels(), max_batch=1, window_ms=1000.0)
    with pytest.raises(asyncio.CancelledError):
        await be.embed(["x"])  # flush re-raises CancelledError after notifying waiters


@pytest.mark.asyncio
async def test_batching_embedder_input_type_groups_never_share_vector():
    class TypedInner:
        dim = 2
        supports_input_type = True

        async def embed(self, texts, *, input_type=None):
            mark = 1.0 if input_type == "query" else 0.0
            return [[mark, 0.0] for _ in texts]

    be = BatchingEmbedder(TypedInner(), max_batch=64, window_ms=5.0)
    doc, qry = await asyncio.gather(
        be.embed(["same"], input_type="document"),
        be.embed(["same"], input_type="query"),
    )
    await be.aclose()
    assert doc[0] == [0.0, 0.0]
    assert qry[0] == [1.0, 0.0]  # separate group, separate vector


@pytest.mark.asyncio
async def test_batching_embedder_timer_fires_on_already_drained_pending():
    inner = LocalHashEmbedder(dim=8)
    be = BatchingEmbedder(inner, max_batch=2, window_ms=3.0)
    # First call (< max_batch) schedules a timer. Second call fills max_batch
    # and drains pending synchronously, so the timer later wakes to an empty
    # queue (the `if batch:` guard in _delayed_flush is false).
    f1 = asyncio.ensure_future(be.embed(["solo"]))  # schedules timer, 1 pending
    await asyncio.sleep(0)  # let it register
    out2 = await be.embed(["fill"])  # now 2 pending -> fills, flushes, drains both
    out1 = await f1
    await asyncio.sleep(0.01)  # stale timer wakes to empty pending (no-op flush)
    assert len(out1) == 1
    assert len(out2) == 1
    await be.aclose()


@pytest.mark.asyncio
async def test_provider_embedder_empty_vectors_for_nonempty_input_keeps_dim():
    class EmptyProvider(MockProvider):
        async def embed(self, texts, model=None):
            return []

    pe = ProviderEmbedder(EmptyProvider(), dim=512, batch_size=64)
    out = await pe.embed(["a"])  # single-batch path, provider returns []
    assert out == []
    assert pe.dim == 512  # `if vectors:` false -> dim unchanged


@pytest.mark.asyncio
async def test_batching_embedder_aclose_flushes_pending():
    inner = LocalHashEmbedder(dim=8)
    be = BatchingEmbedder(inner, max_batch=64, window_ms=10_000.0)
    fut = asyncio.ensure_future(be.embed(["pending"]))
    await asyncio.sleep(0)  # let it register but not flush (huge window)
    await be.aclose()  # cancels timer and flushes
    out = await fut
    assert len(out) == 1
    assert be.flushes == 1


# --------------------------------------------------------------------------- #
# MatryoshkaEmbedder                                                           #
# --------------------------------------------------------------------------- #


def test_matryoshka_rejects_non_positive_dimensions():
    with pytest.raises(ConfigError, match="dimensions > 0"):
        MatryoshkaEmbedder(LocalHashEmbedder(dim=64), 0)


@pytest.mark.asyncio
async def test_matryoshka_truncates_to_target_dimension():
    me = MatryoshkaEmbedder(LocalHashEmbedder(dim=64), 8)
    assert me.dim == 8
    assert me.supports_input_type is False
    out = await me.embed(["truncate me"])
    assert len(out[0]) == 8
    assert sum(x * x for x in out[0]) == pytest.approx(1.0)


def test_matryoshka_embed_sync():
    me = MatryoshkaEmbedder(LocalHashEmbedder(dim=64), 4)
    out = me.embed_sync(["a"])
    assert len(out[0]) == 4


@pytest.mark.asyncio
async def test_matryoshka_passes_input_type_to_typed_inner():
    class TypedInner:
        dim = 6
        supports_input_type = True

        async def embed(self, texts, *, input_type=None):
            base = 9.0 if input_type == "query" else 1.0
            return [[base, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in texts]

    me = MatryoshkaEmbedder(TypedInner(), 3)
    assert me.supports_input_type is True
    out = await me.embed(["q"], input_type="query")
    assert out[0] == [1.0, 0.0, 0.0]  # head renormalized; magnitude on dim 0


# --------------------------------------------------------------------------- #
# embed_texts dispatch                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_embed_texts_skips_hint_for_plain_embedder():
    # LocalHashEmbedder has no supports_input_type -> hint silently ignored.
    out = await embed_texts(LocalHashEmbedder(dim=8), ["x"], input_type="query")
    assert len(out[0]) == 8


# --------------------------------------------------------------------------- #
# HTTPEmbedder family (offline via MockTransport)                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_http_embedder_empty_short_circuits():
    emb = JinaEmbedder(api_key="k", client=_mock_client(lambda r: httpx.Response(500)))
    assert await emb.embed([]) == []


def test_http_embedder_missing_key_raises_auth_error():
    emb = JinaEmbedder()  # no api_key
    with pytest.raises(ProviderAuthError, match="missing API key for embedder 'jina'"):
        emb._headers()


@pytest.mark.asyncio
async def test_http_embedder_error_status_raises_provider_error():
    def handler(request):
        return httpx.Response(429, text="rate limited")

    emb = JinaEmbedder(api_key="k", client=_mock_client(handler))
    with pytest.raises(ProviderError, match="error 429: rate limited"):
        await emb.embed(["hi"])


@pytest.mark.asyncio
async def test_jina_sets_task_and_dimensions_and_truncates():
    captured = {}

    def handler(request):
        import json

        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0, 4.0, 9.0]}]})

    emb = JinaEmbedder(api_key="k", dimensions=2, client=_mock_client(handler))
    out = await emb.embed(["doc"], input_type="document")
    assert captured["json"]["task"] == "retrieval.passage"
    assert captured["json"]["dimensions"] == 2
    assert len(out[0]) == 2  # client-side MRL truncation enforced
    assert emb.dim == 2


@pytest.mark.asyncio
async def test_http_embedder_owns_client_when_not_injected():
    # No injected client -> HTTPEmbedder builds and closes its own. Point it at
    # a local transport via base_url is not enough, so we drive the parse path
    # by injecting through a subclass that returns a canned response.
    def handler(request):
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]})

    emb = JinaEmbedder(api_key="k", client=_mock_client(handler))
    out = await emb.embed(["a", "b"])
    # _parse sorts by index, so index 0 (vector [2.0]) comes first.
    assert out == [[2.0], [1.0]]
    assert emb.dim == 1


@pytest.mark.asyncio
async def test_http_embedder_empty_data_response_leaves_dim_untouched():
    def handler(request):
        return httpx.Response(200, json={"data": []})

    emb = JinaEmbedder(api_key="k", dim=777, client=_mock_client(handler))
    out = await emb.embed(["x"])
    assert out == []
    assert emb.dim == 777  # no vectors -> dim stays the constructor value


@pytest.mark.asyncio
async def test_voyage_payload_uses_output_dimension_and_input_type():
    captured = {}

    def handler(request):
        import json

        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    emb = VoyageEmbedder(api_key="k", dimensions=4, client=_mock_client(handler))
    await emb.embed(["q"], input_type="query")
    assert captured["json"]["output_dimension"] == 4
    assert captured["json"]["input_type"] == "query"


@pytest.mark.asyncio
async def test_cohere_payload_and_dict_parse():
    captured = {}

    def handler(request):
        import json

        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": {"float": [[0.1, 0.2, 0.3]]}})

    emb = CohereEmbedder(api_key="k", dimensions=3, client=_mock_client(handler))
    out = await emb.embed(["text"], input_type="document")
    assert captured["json"]["input_type"] == "search_document"
    assert captured["json"]["embedding_types"] == ["float"]
    assert captured["json"]["output_dimension"] == 3
    assert out == [[0.1, 0.2, 0.3]]


def test_voyage_payload_omits_input_type_when_none():
    emb = VoyageEmbedder(api_key="k")
    payload = emb._payload(["x"], None)
    assert "input_type" not in payload


def test_cohere_multimodal_content_text_only_omits_image_part():
    emb = CohereMultimodalEmbedder(api_key="k")
    parts = emb._content(MultimodalInput(text="just text"))
    assert parts == [{"type": "text", "text": "just text"}]


def test_cohere_multimodal_content_includes_image_url_part():
    emb = CohereMultimodalEmbedder(api_key="k")
    parts = emb._content(MultimodalInput(image=ImageRef(url="https://x/y.png")))
    assert parts == [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]


def test_cohere_parse_handles_bare_list_legacy_shape():
    emb = CohereEmbedder(api_key="k")
    assert emb._parse({"embeddings": [[1.0, 2.0]]}) == [[1.0, 2.0]]


def test_cohere_default_input_type_used_when_no_hint():
    emb = CohereEmbedder(api_key="k", input_type="search_query")
    payload = emb._payload(["x"], None)
    assert payload["input_type"] == "search_query"


# --------------------------------------------------------------------------- #
# VoyageContextualEmbedder                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_voyage_contextual_empty_inputs():
    emb = VoyageContextualEmbedder(api_key="k", client=_mock_client(lambda r: httpx.Response(500)))
    assert await emb.embed([]) == []
    assert await emb.embed_grouped([]) == []


@pytest.mark.asyncio
async def test_voyage_contextual_grouped_round_trip_and_truncation():
    def handler(request):
        import json

        body = json.loads(request.content)
        assert body["inputs"] == [["c0", "c1"], ["d0"]]
        assert body["input_type"] == "document"
        assert body["output_dimension"] == 2
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "index": 0,
                        "data": [
                            {"index": 0, "embedding": [3.0, 4.0, 5.0]},
                            {"index": 1, "embedding": [1.0, 0.0, 9.0]},
                        ],
                    },
                    {"index": 1, "data": [{"index": 0, "embedding": [0.0, 1.0, 0.0]}]},
                ]
            },
        )

    emb = VoyageContextualEmbedder(api_key="k", dimensions=2, client=_mock_client(handler))
    groups = await emb.embed_grouped([["c0", "c1"], ["d0"]], input_type="document")
    assert len(groups) == 2
    assert len(groups[0]) == 2
    assert len(groups[0][0]) == 2  # truncated to dimensions
    assert emb.dim == 2


@pytest.mark.asyncio
async def test_voyage_contextual_grouped_error_status():
    def handler(request):
        return httpx.Response(503, text="down")

    emb = VoyageContextualEmbedder(api_key="k", client=_mock_client(handler))
    with pytest.raises(ProviderError, match="error 503: down"):
        await emb.embed_grouped([["a"]])


@pytest.mark.asyncio
async def test_voyage_contextual_embed_delegates_to_grouped():
    def handler(request):
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "data": [{"index": 0, "embedding": [7.0, 8.0]}]}]},
        )

    emb = VoyageContextualEmbedder(api_key="k", client=_mock_client(handler))
    out = await emb.embed(["just one chunk"])
    assert out == [[7.0, 8.0]]


# --------------------------------------------------------------------------- #
# Multimodal embedders                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_multimodal_embed_empty_and_text_only():
    def handler(request):
        import json

        body = json.loads(request.content)
        # text-only embed() wraps each text into a MultimodalInput.
        assert body["inputs"] == [{"content": [{"type": "text", "text": "hi"}]}]
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    emb = VoyageMultimodalEmbedder(api_key="k", client=_mock_client(handler))
    assert await emb.embed([]) == []
    assert await emb.embed_multimodal([]) == []
    out = await emb.embed(["hi"])
    assert out == [[1.0, 2.0]]


@pytest.mark.asyncio
async def test_voyage_multimodal_image_url_passthrough():
    captured = {}

    def handler(request):
        import json

        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.5]}]})

    emb = VoyageMultimodalEmbedder(api_key="k", dimensions=1, client=_mock_client(handler))
    item = MultimodalInput(text="a cat", image=ImageRef(url="https://example.com/cat.png"))
    await emb.embed_multimodal([item], input_type="query")
    parts = captured["json"]["inputs"][0]["content"]
    assert parts[0] == {"type": "text", "text": "a cat"}
    assert parts[1] == {"type": "image_url", "image_url": "https://example.com/cat.png"}
    assert captured["json"]["input_type"] == "query"
    assert captured["json"]["output_dimension"] == 1


@pytest.mark.asyncio
async def test_voyage_multimodal_local_image_uses_base64_part(tmp_path):
    captured = {}
    png = tmp_path / "pix.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfakebody")  # bytes irrelevant; data URL only

    def handler(request):
        import json

        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0]}]})

    emb = VoyageMultimodalEmbedder(api_key="k", client=_mock_client(handler))
    # image-only item exercises the `if item.text:` false branch and the
    # `data:` base64 branch of _content.
    item = MultimodalInput(image=ImageRef(path=str(png)))
    await emb.embed_multimodal([item])
    parts = captured["json"]["inputs"][0]["content"]
    assert len(parts) == 1  # no text part emitted
    assert parts[0]["type"] == "image_base64"
    assert parts[0]["image_base64"].startswith("data:")


def test_multimodal_encode_image_requires_path_or_url():
    emb = VoyageMultimodalEmbedder(api_key="k")
    with pytest.raises(ConfigError, match="needs a path or url"):
        emb._encode_image(ImageRef())


@pytest.mark.asyncio
async def test_cohere_multimodal_payload_and_parse():
    captured = {}

    def handler(request):
        import json

        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": {"float": [[0.1, 0.2]]}})

    emb = CohereMultimodalEmbedder(api_key="k", dimensions=2, client=_mock_client(handler))
    item = MultimodalInput(text="logo", image=ImageRef(url="https://x/y.png"))
    out = await emb.embed_multimodal([item], input_type="document")
    body = captured["json"]
    assert body["input_type"] == "search_document"
    assert body["embedding_types"] == ["float"]
    assert body["output_dimension"] == 2
    content = body["inputs"][0]["content"]
    assert content[0] == {"type": "text", "text": "logo"}
    assert content[1] == {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
    assert out == [[0.1, 0.2]]


def test_cohere_multimodal_parse_bare_list_and_default_input_type():
    emb = CohereMultimodalEmbedder(api_key="k", input_type="search_query")
    assert emb._parse({"embeddings": [[9.0]]}) == [[9.0]]
    payload = emb._multimodal_payload([MultimodalInput(text="t")], None)
    assert payload["input_type"] == "search_query"


# --------------------------------------------------------------------------- #
# FastEmbedEmbedder / ColBERTTokenEmbedder (offline paths)                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fastembed_encode_fn_path():
    def encode(texts):
        return [[1, 2, 3] for _ in texts]  # ints coerced to floats

    emb = FastEmbedEmbedder(encode_fn=encode, dim=3)
    out = await emb.embed(["a", "b"])
    assert out == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]


@pytest.mark.asyncio
async def test_fastembed_injected_model_executor_path():
    class FakeModel:
        def embed(self, texts):
            return [[float(len(t))] for t in texts]

    emb = FastEmbedEmbedder(model=FakeModel(), dim=1)
    out = await emb.embed(["xyz"])
    assert out == [[3.0]]


@pytest.mark.asyncio
async def test_fastembed_fallback_to_local_hash():
    emb = FastEmbedEmbedder(model_name="missing-model", dim=16, fallback=True)
    out = await emb.embed(["fallback please"])
    assert len(out[0]) == 16
    assert isinstance(emb._fallback_embedder, LocalHashEmbedder)


def test_fastembed_missing_dependency_raises_config_error():
    # No encode_fn, no model, no fallback, and fastembed not installed offline.
    emb = FastEmbedEmbedder(model_name="definitely-not-installed-xyz", fallback=False)
    with pytest.raises(ConfigError, match=r"vincio\[fastembed\]"):
        emb._ensure()


@pytest.mark.asyncio
async def test_colbert_fallback_and_encode_fn():
    emb = ColBERTTokenEmbedder(dim=12)  # fallback=True by default
    out = await emb.embed(["token vectors"])
    assert len(out[0]) == 12

    def encode(texts):
        return [[5, 6] for _ in texts]

    custom = ColBERTTokenEmbedder(encode_fn=encode, dim=2)
    assert await custom.embed(["x"]) == [[5.0, 6.0]]


def test_colbert_no_fallback_no_encode_raises():
    emb = ColBERTTokenEmbedder(fallback=False)
    with pytest.raises(ConfigError, match="requires a local model"):
        emb._ensure()


# --------------------------------------------------------------------------- #
# build_embedder dispatch                                                      #
# --------------------------------------------------------------------------- #


def test_build_embedder_local_default():
    emb = build_embedder("local", dim=128)
    assert isinstance(emb, LocalHashEmbedder)
    assert emb.dim == 128


def test_build_embedder_fastembed_and_colbert_aliases():
    assert isinstance(build_embedder("fastembed"), FastEmbedEmbedder)
    assert isinstance(build_embedder("onnx"), FastEmbedEmbedder)
    assert isinstance(build_embedder("colbert"), ColBERTTokenEmbedder)


def test_build_embedder_http_passes_dimensions_natively():
    emb = build_embedder("voyage", api_key="k", dimensions=256)
    assert isinstance(emb, VoyageEmbedder)
    assert emb.dimensions == 256  # native MRL, not wrapped
    assert emb.dim == 256


def test_build_embedder_wraps_local_with_matryoshka_for_dimensions():
    emb = build_embedder("local", dim=64, dimensions=8)
    # LocalHashEmbedder lacks supports_dimensions -> Matryoshka wrapper.
    assert isinstance(emb, MatryoshkaEmbedder)
    assert emb.dim == 8


def test_build_embedder_http_without_dimensions_keeps_default_dim():
    emb = build_embedder("jina", api_key="k")
    assert isinstance(emb, JinaEmbedder)
    assert emb.dimensions is None
    assert emb.dim == 1024  # default, no Matryoshka wrap


def test_build_embedder_provider_path_with_mock():
    # "mock" resolves through build_provider -> ProviderEmbedder.
    emb = build_embedder("mock", model="text-embed", dim=32)
    assert isinstance(emb, ProviderEmbedder)
    assert emb.model == "text-embed"


def test_build_embedder_dispatches_to_discovered_entry_point():
    import vincio.retrieval.embeddings as mod

    class PluginEmbedder:
        supports_dimensions = True

        def __init__(self, *, dimensions=None, **kwargs):
            self.dim = dimensions or 11
            self.dimensions = dimensions

    saved = mod._DISCOVERED_EMBEDDERS
    mod._DISCOVERED_EMBEDDERS = {"my-plugin": PluginEmbedder}
    try:
        emb = build_embedder("my-plugin", dimensions=7)
        assert isinstance(emb, PluginEmbedder)
        assert emb.dimensions == 7  # dimensions forwarded into the plugin
    finally:
        mod._DISCOVERED_EMBEDDERS = saved


def test_build_embedder_unknown_kind_raises_config_error():
    with pytest.raises(ConfigError, match="unknown embedder 'totally-not-real'"):
        build_embedder("totally-not-real")
