"""Retrieval breadth: Matryoshka + input-type embeddings, contextual and
multimodal embedders, the new vector-store adapters, and layout-aware PDF
extraction. All offline via httpx MockTransport and injected fake clients."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import httpx
import pytest

from vincio.core.errors import LoaderError
from vincio.core.types import Chunk, ImageRef
from vincio.documents import (
    LayoutBlock,
    LayoutFigure,
    LayoutWord,
    PageLayout,
    assemble_layout,
    extract_pdf_layout,
    group_words_into_lines,
    order_blocks,
)
from vincio.retrieval import (
    CohereMultimodalEmbedder,
    LocalHashEmbedder,
    MatryoshkaEmbedder,
    MultimodalInput,
    VectorIndex,
    VoyageContextualEmbedder,
    VoyageMultimodalEmbedder,
    build_embedder,
    build_filter,
    build_filter_spec,
    embed_texts,
    mrl_truncate,
)
from vincio.retrieval.embeddings import (
    BatchingEmbedder,
    CachedEmbedder,
    JinaEmbedder,
    VoyageEmbedder,
)
from vincio.storage import VECTOR_BACKENDS, build_vector_index


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _chunks() -> list[Chunk]:
    return [
        Chunk(document_id="d", text="cats are small domestic felines", index=0),
        Chunk(document_id="d", text="the refund policy allows 30 day returns", index=1),
        Chunk(document_id="d", text="quarterly revenue grew twelve percent", index=2),
    ]


def _tenant_chunks() -> list[Chunk]:
    return [
        Chunk(document_id="d", text="cats are small domestic felines", index=0, tenant_id="acme"),
        Chunk(document_id="d", text="the refund policy allows 30 day returns", index=1, tenant_id="other"),
        Chunk(document_id="d", text="quarterly revenue grew twelve percent", index=2, tenant_id="acme"),
    ]


class _InputTypeAwareEmbedder:
    """An embedder whose vectors differ by input_type, to prove wrappers
    forward and key on the hint."""

    supports_input_type = True
    dim = 2

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str | None]] = []

    async def embed(self, texts, *, input_type=None):
        self.calls.append((tuple(texts), input_type))
        marker = 1.0 if input_type == "query" else 2.0
        return [[marker, float(len(t))] for t in texts]


@pytest.mark.asyncio
async def test_batching_embedder_groups_by_input_type():
    inner = _InputTypeAwareEmbedder()
    batcher = BatchingEmbedder(inner)
    # Two concurrent embeds of the same text but different input_type, coalesced
    # into one flush — they must NOT share a vector.
    doc, qry = await asyncio.gather(
        batcher.embed(["same"], input_type="document"),
        batcher.embed(["same"], input_type="query"),
    )
    assert doc[0][0] == 2.0 and qry[0][0] == 1.0  # not aliased across input types
    assert len(inner.calls) == 2  # one inner call per input type
    assert {input_type for _texts, input_type in inner.calls} == {"document", "query"}


# -- Matryoshka (MRL) -----------------------------------------------------------


def test_mrl_truncate_shrinks_and_renormalizes():
    v = [3.0, 4.0, 5.0, 12.0]
    out = mrl_truncate(v, 2)
    assert len(out) == 2
    assert math.isclose(math.sqrt(sum(x * x for x in out)), 1.0, rel_tol=1e-9)


def test_mrl_truncate_noop_when_dimensions_cover_vector():
    v = [1.0, 2.0, 3.0]
    assert mrl_truncate(v, 5) == v
    assert mrl_truncate(v, 0) == v


@pytest.mark.asyncio
async def test_matryoshka_embedder_truncates_any_embedder():
    base = LocalHashEmbedder(dim=128)
    mrl = MatryoshkaEmbedder(base, 32)
    assert mrl.dim == 32
    [vec] = await mrl.embed(["hello world"])
    assert len(vec) == 32
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-6)


def test_matryoshka_rejects_nonpositive_dimensions():
    from vincio.core.errors import ConfigError

    with pytest.raises(ConfigError):
        MatryoshkaEmbedder(LocalHashEmbedder(), 0)


@pytest.mark.asyncio
async def test_build_embedder_dimensions_wraps_local():
    emb = build_embedder("local", dimensions=16)
    assert isinstance(emb, MatryoshkaEmbedder)
    [vec] = await emb.embed(["x"])
    assert len(vec) == 16


@pytest.mark.asyncio
async def test_jina_native_dimensions_sent_and_enforced():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]}]})

    emb = build_embedder("jina", api_key="k", dimensions=2, client=_mock_client(handler))
    assert not isinstance(emb, MatryoshkaEmbedder)  # native support, not wrapped
    [vec] = await emb.embed(["a"])
    assert seen["payload"]["dimensions"] == 2
    assert len(vec) == 2  # enforced client-side too


# -- input-type plumbing (query vs document) ------------------------------------


@pytest.mark.asyncio
async def test_vector_index_passes_input_type_to_aware_embedder():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body.get("task"))
        n = len(body["input"])
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [float(i), 1.0]} for i in range(n)]})

    embedder = JinaEmbedder(api_key="k", client=_mock_client(handler))
    index = VectorIndex(embedder)
    await index.add(_chunks())
    await index.search("revenue", top_k=2)
    assert "retrieval.passage" in seen  # document side
    assert "retrieval.query" in seen  # query side


@pytest.mark.asyncio
async def test_voyage_input_type_in_payload():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    embedder = VoyageEmbedder(api_key="k", client=_mock_client(handler))
    await embed_texts(embedder, ["q"], input_type="query")
    assert seen["payload"]["input_type"] == "query"


@pytest.mark.asyncio
async def test_cached_embedder_keys_on_input_type():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        it = json.loads(request.content).get("input_type")
        # different vector per input_type so we can tell them apart
        value = 1.0 if it == "query" else 2.0
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [value, 0.0]}]})

    cached = CachedEmbedder(VoyageEmbedder(api_key="k", client=_mock_client(handler)))
    doc = await cached.embed(["same text"], input_type="document")
    qry = await cached.embed(["same text"], input_type="query")
    again = await cached.embed(["same text"], input_type="document")
    assert doc != qry  # not aliased across input types
    assert again == doc  # document re-read is a cache hit
    assert calls["n"] == 2  # only the two distinct input types hit the wire


# -- contextual embeddings ------------------------------------------------------


@pytest.mark.asyncio
async def test_voyage_contextual_embedder_groups_and_parses():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "index": 0,
                        "data": [
                            {"index": 0, "embedding": [0.1, 0.2]},
                            {"index": 1, "embedding": [0.3, 0.4]},
                        ],
                    }
                ]
            },
        )

    embedder = VoyageContextualEmbedder(api_key="k", client=_mock_client(handler))
    vectors = await embedder.embed(["chunk one", "chunk two"], input_type="document")
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert seen["payload"]["inputs"] == [["chunk one", "chunk two"]]  # one document group
    assert seen["payload"]["input_type"] == "document"


@pytest.mark.asyncio
async def test_voyage_contextual_embed_grouped_returns_per_document():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "data": [{"index": 0, "embedding": [1.0]}]},
                    {"index": 1, "data": [{"index": 0, "embedding": [2.0]}]},
                ]
            },
        )

    embedder = VoyageContextualEmbedder(api_key="k", client=_mock_client(handler))
    groups = await embedder.embed_grouped([["a"], ["b"]])
    assert groups == [[[1.0]], [[2.0]]]


# -- multimodal embeddings ------------------------------------------------------


@pytest.mark.asyncio
async def test_voyage_multimodal_embeds_text_and_image(tmp_path):
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.5, 0.5]}]})

    embedder = VoyageMultimodalEmbedder(api_key="k", client=_mock_client(handler))
    item = MultimodalInput(text="a bar chart", image=ImageRef(path=str(image_path)))
    [vec] = await embedder.embed_multimodal([item])
    assert vec == [0.5, 0.5]
    content = seen["payload"]["inputs"][0]["content"]
    assert content[0] == {"type": "text", "text": "a bar chart"}
    assert content[1]["type"] == "image_base64"
    assert content[1]["image_base64"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_cohere_multimodal_shape_and_text_only():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": {"float": [[1.0, 2.0, 3.0]]}})

    embedder = CohereMultimodalEmbedder(api_key="k", client=_mock_client(handler))
    out = await embedder.embed(["unified text"], input_type="query")
    assert out == [[1.0, 2.0, 3.0]]
    assert seen["payload"]["model"] == "embed-v4.0"
    assert seen["payload"]["input_type"] == "search_query"
    assert seen["payload"]["inputs"][0]["content"][0]["text"] == "unified text"


@pytest.mark.asyncio
async def test_cohere_multimodal_image_content_shape(tmp_path):
    image_path = tmp_path / "c.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": {"float": [[0.1, 0.2]]}})

    embedder = CohereMultimodalEmbedder(api_key="k", client=_mock_client(handler))
    await embedder.embed_multimodal(
        [MultimodalInput(text="a chart", image=ImageRef(path=str(image_path)))]
    )
    content = seen["payload"]["inputs"][0]["content"]
    # Cohere v4 nests the image as {"image_url": {"url": ...}} — distinct from Voyage.
    assert content[0] == {"type": "text", "text": "a chart"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_multimodal_encode_image_url_passthrough():
    embedder = VoyageMultimodalEmbedder(api_key="k")
    assert embedder._encode_image(ImageRef(url="https://example.com/x.png")) == "https://example.com/x.png"
    # A url image rides the image_url branch, not image_base64.
    parts = embedder._content(MultimodalInput(image=ImageRef(url="https://example.com/x.png")))
    assert parts[0] == {"type": "image_url", "image_url": "https://example.com/x.png"}


def test_multimodal_image_needs_path_or_url():
    from vincio.core.errors import ConfigError

    embedder = VoyageMultimodalEmbedder(api_key="k")
    with pytest.raises(ConfigError):
        embedder._encode_image(ImageRef())


# -- new vector stores (round-trip via injected fakes) --------------------------


def _es_match(rec: dict, q: dict) -> bool:
    """Apply a compiled Elasticsearch bool/term/terms/range/exists filter to a
    stored document's flat fields — proves the pushed-down filter is server-side."""
    if "bool" in q:
        b = q["bool"]
        if any(not _es_match(rec, c) for c in b.get("must", [])):
            return False
        if b.get("should") and not any(_es_match(rec, c) for c in b["should"]):
            return False
        if any(_es_match(rec, c) for c in b.get("must_not", [])):
            return False
        return True
    if "term" in q:
        (f, v), = q["term"].items()
        return rec.get(f) == v
    if "terms" in q:
        (f, vs), = q["terms"].items()
        return rec.get(f) in vs
    if "exists" in q:
        return rec.get(q["exists"]["field"]) is not None
    if "range" in q:
        (f, conds), = q["range"].items()
        val = rec.get(f)
        if val is None:
            return False
        return all(
            (op == "gt" and val > b) or (op == "gte" and val >= b)
            or (op == "lt" and val < b) or (op == "lte" and val <= b)
            for op, b in conds.items()
        )
    return True


def _pinecone_match(rec: dict, f: dict) -> bool:
    """Apply a compiled Pinecone mongo-style metadata filter to stored metadata."""
    if "$and" in f:
        return all(_pinecone_match(rec, c) for c in f["$and"])
    if "$or" in f:
        return any(_pinecone_match(rec, c) for c in f["$or"])
    for field, cond in f.items():
        if field == "$nor":
            if any(_pinecone_match(rec, c) for c in cond):
                return False
            continue
        if field in ("$and", "$or"):
            continue
        actual = rec.get(field)
        for op, val in cond.items():
            if op == "$eq" and actual != val:
                return False
            if op == "$ne" and actual == val:
                return False
            if op == "$in" and actual not in val:
                return False
            if op == "$nin" and actual in val:
                return False
            if op == "$exists" and (actual is not None) != val:
                return False
            if op == "$gt" and not (actual is not None and actual > val):
                return False
            if op == "$gte" and not (actual is not None and actual >= val):
                return False
            if op == "$lt" and not (actual is not None and actual < val):
                return False
            if op == "$lte" and not (actual is not None and actual <= val):
                return False
    return True


def _weaviate_match(rec: dict, w: dict) -> bool:
    """Apply a compiled Weaviate `where` filter to stored properties."""
    op = w.get("operator")
    if op in ("And", "Or"):
        results = [_weaviate_match(rec, o) for o in w["operands"]]
        return all(results) if op == "And" else any(results)
    if op == "Not":
        return not _weaviate_match(rec, w["operands"][0])
    path = w["path"][0]
    actual = rec.get(path)
    if op == "IsNull":
        return (actual is None) == w.get("valueBoolean", True)
    val = next((w[k] for k in w if k.startswith("value")), None)
    if op == "Equal":
        return actual == val
    if op == "NotEqual":
        return actual != val
    if op == "ContainsAny":
        wanted = val if isinstance(val, list) else [val]
        return actual in wanted or (isinstance(actual, list) and any(x in actual for x in wanted))
    if op == "GreaterThan":
        return actual is not None and actual > val
    if op == "GreaterThanEqual":
        return actual is not None and actual >= val
    if op == "LessThan":
        return actual is not None and actual < val
    if op == "LessThanEqual":
        return actual is not None and actual <= val
    return True


class _FakeSearchEngine:
    """Minimal Elasticsearch/OpenSearch-shaped client for offline round trips."""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.created = False

        engine = self

        class _Indices:
            def exists(self, index):
                return engine.created

            def create(self, index, mappings=None, body=None):
                engine.created = True

        self.indices = _Indices()

    def index(self, index, id, document, refresh=False):
        self.docs[id] = document

    def count(self, index):
        return {"count": len(self.docs)}

    def delete(self, index, id, refresh=False):
        if id not in self.docs:
            raise KeyError(id)
        del self.docs[id]

    def search(self, index, knn=None, size=10, body=None):
        if knn is not None:
            query_vector, k = knn["query_vector"], knn["k"]
            native = knn.get("filter")
        else:
            knn_q = body["query"]["knn"]["vector"]
            query_vector, k = knn_q["vector"], knn_q["k"]
            native = knn_q.get("filter")
        self.last_filter = native  # recorded so tests can assert pushdown
        candidates = [d for d in self.docs.values() if native is None or _es_match(d, native)]
        ranked = sorted(candidates, key=lambda d: _cos(d["vector"], query_vector), reverse=True)
        hits = [
            {"_source": {"json": d["json"]}, "_score": _cos(d["vector"], query_vector)}
            for d in ranked[:k]
        ]
        return {"hits": {"hits": hits}}


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["elasticsearch", "opensearch"])
async def test_elastic_family_round_trip(backend):
    chunks = _chunks()
    index = build_vector_index(backend, LocalHashEmbedder(), client=_FakeSearchEngine())
    await index.add(chunks)
    assert len(index) == 3
    hits = await index.search("refund returns", top_k=1)
    assert hits[0].chunk.index == 1  # the refund chunk ranks first
    assert hits[0].source == backend
    assert await index.delete([chunks[0].id]) == 1
    assert len(index) == 2


class _FakeWeaviate:
    def __init__(self) -> None:
        store: dict[str, dict] = {}
        outer = self

        class _Meta:
            def __init__(self, distance):
                self.distance = distance

        class _Obj:
            def __init__(self, properties, distance):
                self.properties = properties
                self.metadata = _Meta(distance)

        class _Result:
            def __init__(self, objects):
                self.objects = objects

        class _Data:
            def insert(self, properties, vector, uuid):
                store[uuid] = {"properties": properties, "vector": vector}

            def delete_by_id(self, uuid):
                if uuid not in store:
                    raise KeyError(uuid)
                del store[uuid]

        class _Query:
            def near_vector(self, near_vector, limit, return_metadata=None, filters=None):
                outer.last_filter = filters  # recorded so tests can assert pushdown
                candidates = [
                    r for r in store.values()
                    if filters is None or _weaviate_match(r["properties"], filters)
                ]
                ranked = sorted(
                    candidates, key=lambda r: _cos(r["vector"], near_vector), reverse=True
                )
                objs = [
                    _Obj(r["properties"], 1.0 - _cos(r["vector"], near_vector)) for r in ranked[:limit]
                ]
                return _Result(objs)

        class _Aggregate:
            def over_all(self, total_count=False):
                class _Total:
                    pass

                t = _Total()
                t.total_count = len(store)
                return t

        class _Collection:
            data = _Data()
            query = _Query()
            aggregate = _Aggregate()

        class _Collections:
            def exists(self, name):
                return getattr(outer, "_created", False)

            def create(self, name):
                outer._created = True

            def get(self, name):
                return _Collection()

        self.collections = _Collections()


@pytest.mark.asyncio
async def test_weaviate_round_trip():
    chunks = _chunks()
    index = build_vector_index("weaviate", LocalHashEmbedder(), client=_FakeWeaviate())
    await index.add(chunks)
    assert len(index) == 3
    hits = await index.search("quarterly revenue", top_k=1)
    assert hits[0].chunk.index == 2
    assert 0.0 <= hits[0].score <= 1.0
    assert await index.delete([chunks[0].id]) == 1
    assert len(index) == 2


class _FakeMilvus:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.created = False

    def has_collection(self, collection_name):
        return self.created

    def create_collection(self, **kwargs):
        self.created = True

    def insert(self, collection_name, data):
        for row in data:
            self.rows[row["id"]] = row

    def delete(self, collection_name, ids):
        for cid in ids:
            self.rows.pop(cid, None)

    def get_collection_stats(self, collection_name):
        return {"row_count": len(self.rows)}

    def search(self, collection_name, data, limit, output_fields, filter=None):  # noqa: A002
        self.last_filter = filter  # the compiled Milvus expr, recorded for assertions
        query = data[0]
        ranked = sorted(self.rows.values(), key=lambda r: _cos(r["vector"], query), reverse=True)
        return [
            [
                {"id": r["id"], "distance": _cos(r["vector"], query), "entity": {"json": r["json"]}}
                for r in ranked[:limit]
            ]
        ]


@pytest.mark.asyncio
async def test_milvus_round_trip():
    chunks = _chunks()
    index = build_vector_index("milvus", LocalHashEmbedder(), client=_FakeMilvus())
    await index.add(chunks)
    assert len(index) == 3
    hits = await index.search("domestic cats", top_k=1)
    assert hits[0].chunk.index == 0
    assert await index.delete([chunks[2].id]) == 1
    assert len(index) == 2


class _FakeVespa:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}

    def feed_data_point(self, schema, data_id, fields):
        self.docs[data_id] = fields

    def delete_data(self, schema, data_id):
        if data_id not in self.docs:
            raise KeyError(data_id)
        del self.docs[data_id]

    def query(self, body):
        outer = self

        class _Resp:
            def __init__(self, hits, total):
                self.hits = hits
                self.json = {"root": {"fields": {"totalCount": total}}}

        if body.get("hits") == 0:
            return _Resp([], len(outer.docs))
        query_vector = body["input.query(q)"]
        ranked = sorted(
            outer.docs.values(), key=lambda f: _cos(f["embedding"], query_vector), reverse=True
        )
        hits = [{"fields": f, "relevance": _cos(f["embedding"], query_vector)} for f in ranked]
        return _Resp(hits, len(outer.docs))


@pytest.mark.asyncio
async def test_vespa_round_trip():
    chunks = _chunks()
    index = build_vector_index("vespa", LocalHashEmbedder(), app=_FakeVespa())
    await index.add(chunks)
    assert len(index) == 3
    hits = await index.search("refund policy", top_k=1)
    assert hits[0].chunk.index == 1
    assert await index.delete([chunks[1].id]) == 1
    assert len(index) == 2


def test_new_backends_registered():
    for backend in ("weaviate", "milvus", "elasticsearch", "opensearch", "vespa"):
        assert backend in VECTOR_BACKENDS


@pytest.mark.asyncio
async def test_new_vector_stores_apply_where_filter():
    """Each new store over-fetches and applies the metadata filter client-side;
    a tenant filter must exclude the other tenant's chunk even when it is the
    best lexical match."""
    builders = [
        lambda: build_vector_index("elasticsearch", LocalHashEmbedder(), client=_FakeSearchEngine()),
        lambda: build_vector_index("opensearch", LocalHashEmbedder(), client=_FakeSearchEngine()),
        lambda: build_vector_index("weaviate", LocalHashEmbedder(), client=_FakeWeaviate()),
        lambda: build_vector_index("milvus", LocalHashEmbedder(), client=_FakeMilvus()),
        lambda: build_vector_index("vespa", LocalHashEmbedder(), app=_FakeVespa()),
    ]
    where = build_filter(tenant_id="acme")
    for build in builders:
        index = build()
        await index.add(_tenant_chunks())
        # "refund returns" best-matches the index-1 chunk, which is tenant "other".
        hits = await index.search("refund policy returns", top_k=3, where=where)
        assert hits, index.name
        assert all(h.chunk.tenant_id == "acme" for h in hits), index.name


class _FakePinecone:
    """Minimal Pinecone-shaped client that applies the metadata filter."""

    def __init__(self) -> None:
        self.vectors: dict[str, dict] = {}
        self.last_filter = None
        outer = self

        class _Index:
            def upsert(self, vectors, namespace=""):
                for v in vectors:
                    outer.vectors[v["id"]] = v

            def delete(self, ids, namespace=""):
                for i in ids:
                    outer.vectors.pop(i, None)

            def describe_index_stats(self):
                return {"total_vector_count": len(outer.vectors)}

            def query(self, vector, top_k, include_metadata=True, namespace="", filter=None):  # noqa: A002
                outer.last_filter = filter
                cands = [
                    v for v in outer.vectors.values()
                    if filter is None or _pinecone_match(v["metadata"], filter)
                ]
                ranked = sorted(cands, key=lambda v: _cos(v["values"], vector), reverse=True)
                return {
                    "matches": [
                        {"metadata": v["metadata"], "score": _cos(v["values"], vector)}
                        for v in ranked[:top_k]
                    ]
                }

        self._index = _Index()

    def list_indexes(self):
        # Report the index as existing so the adapter skips the create path
        # (which imports ServerlessSpec from the real SDK).
        return [{"name": "vincio-chunks"}]

    def create_index(self, **kwargs):  # pragma: no cover - create path not exercised
        pass

    def Index(self, name):  # noqa: N802 - mirrors the Pinecone SDK method name
        return self._index


@pytest.mark.asyncio
async def test_native_pushdown_passes_compiled_filter_to_backend():
    """A FilterSpec is compiled and pushed into each backend's native filter
    (recorded by the fake), and tenant scope returns only the tenant's rows."""
    cases = [
        ("elasticsearch", lambda f: build_vector_index("elasticsearch", LocalHashEmbedder(), client=f), _FakeSearchEngine),
        ("weaviate", lambda f: build_vector_index("weaviate", LocalHashEmbedder(), client=f), _FakeWeaviate),
        ("milvus", lambda f: build_vector_index("milvus", LocalHashEmbedder(), client=f), _FakeMilvus),
        ("pinecone", lambda f: build_vector_index("pinecone", LocalHashEmbedder(), client=f), _FakePinecone),
    ]
    scope = build_filter_spec(tenant_id="acme")  # FilterSpec, not a callable
    for name, build, fake_cls in cases:
        fake = fake_cls()
        index = build(fake)
        await index.add(_tenant_chunks())
        hits = await index.search("refund policy returns", top_k=3, where=scope)
        assert hits, name
        assert all(h.chunk.tenant_id == "acme" for h in hits), name
        # The compiled native filter actually reached the backend (pushdown),
        # not just a client-side post-filter.
        assert fake.last_filter is not None, name


@pytest.mark.asyncio
async def test_native_pushdown_stores_flat_filter_fields():
    """add() persists flat filterable fields alongside the chunk blob so the
    native filter has something to match server-side."""
    fake = _FakeSearchEngine()
    index = build_vector_index("elasticsearch", LocalHashEmbedder(), client=fake)
    await index.add(_tenant_chunks())
    stored = next(iter(fake.docs.values()))
    assert stored["tenant_id"] in {"acme", "other"}
    assert "document_id" in stored and "kind" in stored


# -- layout-aware extraction ----------------------------------------------------


def test_group_words_into_lines():
    words = [
        LayoutWord(text="Hello", x0=0, top=0, x1=20, bottom=10),
        LayoutWord(text="world", x0=22, top=1, x1=44, bottom=10),
        LayoutWord(text="second", x0=0, top=20, x1=30, bottom=30),
    ]
    blocks = group_words_into_lines(words)
    assert [b.text for b in blocks] == ["Hello world", "second"]


def test_order_blocks_two_column_reading_order():
    blocks = [
        LayoutBlock(text="L1", x0=10, top=10, x1=90, bottom=20),
        LayoutBlock(text="R1", x0=110, top=12, x1=190, bottom=22),
        LayoutBlock(text="L2", x0=10, top=30, x1=90, bottom=40),
        LayoutBlock(text="R2", x0=110, top=32, x1=190, bottom=42),
    ]
    ordered = order_blocks(blocks, page_width=200)
    assert [b.text for b in ordered] == ["L1", "L2", "R1", "R2"]


def test_order_blocks_single_column_top_to_bottom():
    blocks = [
        LayoutBlock(text="b", x0=10, top=30, x1=190, bottom=40),
        LayoutBlock(text="a", x0=10, top=10, x1=190, bottom=20),
    ]
    ordered = order_blocks(blocks, page_width=200)
    assert [b.text for b in ordered] == ["a", "b"]


def test_assemble_layout_builds_document_with_tables_and_figures():
    from vincio.documents import TableData

    page = PageLayout(
        page_number=1,
        width=200,
        height=300,
        blocks=[LayoutBlock(text="Intro", x0=10, top=10, x1=190, bottom=20)],
        tables=[TableData(columns=["a", "b"], rows=[["1", "2"]])],
        figures=[LayoutFigure(page=1, x0=5, top=5, x1=50, bottom=50, caption="fig 1")],
    )
    doc = assemble_layout([page], title="report", source_uri="report.pdf")
    assert doc.metadata["extractor"] == "layout"
    assert doc.metadata["page_count"] == 1
    assert doc.metadata["table_count"] == 1
    assert doc.metadata["figure_count"] == 1
    assert doc.tables[0]["page"] == 1
    assert "Intro" in doc.text
    assert doc.tables[0]["inferred_schema"]  # schema/quality computed during assembly


def test_extract_pdf_layout_missing_dependency_is_helpful(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pdfplumber":
            raise ImportError("no pdfplumber")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(LoaderError, match="pdf-layout"):
        extract_pdf_layout(Path("does-not-matter.pdf"))
