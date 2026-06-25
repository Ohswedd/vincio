"""Real-behavior coverage for the LlamaIndex interop bridge.

The ``from_llamaindex_*`` adapters are duck-typed, so they are exercised with
small hand-built fakes that mimic LlamaIndex's surface (the same approach as
tests/test_interop.py). The ``to_llamaindex_*`` exporters build genuine
``llama_index.core`` objects -- this environment has llama_index installed, so
those round trips are tested end to end against the real classes.
"""

from __future__ import annotations

import builtins
import sys

import pytest

from vincio import ContextApp
from vincio.core.errors import ConfigError
from vincio.core.types import Chunk, Document
from vincio.interop import llamaindex as li
from vincio.providers import MockProvider
from vincio.retrieval.indexes import SearchHit
from vincio.tools import ToolRegistry

# The export (``to_llamaindex_*``) tests build real ``llama_index.core`` objects;
# each guards its own body with
# ``pytest.importorskip("llama_index.core", exc_type=ImportError)``. Using
# importorskip (which actually imports) rather than ``find_spec`` is deliberate:
# under ``coverage run`` this environment's numpy C-extension can raise on
# llama_index's import even when the spec is found, and ``exc_type=ImportError``
# makes importorskip skip on that real ImportError too. The dep-free ``from_*``
# tests still collect and run.


# -- duck-typed fakes for the from_* (import-free) path -------------------------


class _MetadataModeNode:
    """A node whose ``get_content()`` rejects a no-arg call, forcing the
    ``metadata_mode="none"`` fallback in ``_node_text`` (lines 44-46)."""

    def __init__(self, text: str, metadata: dict | None = None) -> None:
        self._text = text
        self.metadata = metadata or {}

    def get_content(self, *, metadata_mode: str | None = None) -> str:
        if metadata_mode is None:
            raise TypeError("get_content() requires metadata_mode")
        return self._text


class _TextOnlyNode:
    """No ``get_content`` -- exercises the ``getattr(node, 'text', ...)`` branch."""

    def __init__(self, text: str, metadata: dict | None = None) -> None:
        self.text = text
        self.metadata = metadata or {}


class _FnSchema:
    @staticmethod
    def model_json_schema() -> dict:
        return {"type": "object", "properties": {"q": {"type": "string"}}}


class _SchemaMeta:
    name = "schema_tool"
    description = "Tool with a pydantic fn_schema."
    fn_schema = _FnSchema


class _SchemaTool:
    metadata = _SchemaMeta()

    def __call__(self, **kwargs):  # no .call -> exercises the __call__ branch
        return {"content": f"echo {kwargs.get('q')}"}


class _RaisingParamsMeta:
    name = "raisey"
    description = "params extraction blows up"
    fn_schema = None

    def get_parameters_dict(self):
        raise RuntimeError("boom")


class _RaisingParamsTool:
    metadata = _RaisingParamsMeta()

    def call(self, **kwargs):  # noqa: ARG002
        return "ok"


class _NamelessMeta:
    name = None
    description = ""
    fn_schema = None


class _NamelessTool:
    metadata = _NamelessMeta()

    def call(self, **kwargs):  # noqa: ARG002
        return "raw"


class _ContentOutput:
    """A ToolOutput with no ``raw_output`` -> unwrap falls back to ``content``."""

    def __init__(self, content):
        self.content = content


class _ContentTool:
    class _Meta:
        name = "content_tool"
        description = "returns a content-bearing output"
        fn_schema = None

    metadata = _Meta()

    def call(self, **kwargs):
        return _ContentOutput(f"hi {kwargs['who']}")


class _LINode:
    def __init__(self, text, metadata=None):
        self._text = text
        self.metadata = metadata or {}

    def get_content(self):
        return self._text


class _Scored:
    def __init__(self, node, score):
        self.node = node
        self.score = score


class _Retriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query):  # noqa: ARG002
        return self._results


class _BatchEmbedding:
    def get_text_embedding_batch(self, texts):
        return [[float(len(t)), 0.5, 0.5] for t in texts]


class _SingleEmbedding:
    """No batch method -> per-text loop branch (line 201)."""

    def get_text_embedding(self, text):
        return [float(len(text)), 9.0]


# -- _node_text fallbacks -------------------------------------------------------


def test_node_text_metadata_mode_fallback():
    doc = li.from_llamaindex_document(_MetadataModeNode("payload", {"source": "x"}))
    assert doc.text == "payload"


def test_node_text_text_attribute_branch():
    doc = li.from_llamaindex_document(_TextOnlyNode("plain", {"title": "T"}))
    assert doc.text == "plain"
    assert doc.title == "T"


def test_from_document_prefers_file_name_when_no_title():
    doc = li.from_llamaindex_document(_LINode("b", {"file_name": "readme.md"}))
    assert doc.title == "readme.md"
    assert doc.source_uri is None  # no file_path / source key


def test_from_documents_maps_each():
    docs = li.from_llamaindex_documents([_LINode("a", {}), _LINode("b", {})])
    assert [d.text for d in docs] == ["a", "b"]


def test_from_reader_runs_load_data_with_kwargs():
    class Reader:
        def load_data(self, *, limit):
            return [_LINode(f"n{i}", {}) for i in range(limit)]

    docs = li.from_llamaindex_reader(Reader(), limit=3)
    assert [d.text for d in docs] == ["n0", "n1", "n2"]


# -- tool schema extraction -----------------------------------------------------


def test_tool_input_schema_from_fn_schema():
    adapter = li.from_llamaindex_tool(_SchemaTool())
    assert adapter["name"] == "schema_tool"
    assert adapter["input_schema"]["properties"]["q"]["type"] == "string"


def test_tool_input_schema_swallows_extraction_error():
    # get_parameters_dict raises -> best-effort returns {} (lines 88-90).
    adapter = li.from_llamaindex_tool(_RaisingParamsTool())
    assert adapter["input_schema"] == {}


def test_tool_handler_call_branch_and_raw_output():
    adapter = li.from_llamaindex_tool(_SchemaTool())
    # _SchemaTool has no .call -> goes through li_tool(**kwargs); output is a
    # dict, no raw_output/content attr -> handler returns the dict itself.
    assert adapter["handler"](q="ping") == {"content": "echo ping"}


def test_tool_handler_unwraps_content_field():
    adapter = li.from_llamaindex_tool(_ContentTool())
    assert adapter["handler"](who="ada") == "hi ada"


def test_tool_defaults_name_when_metadata_name_missing():
    adapter = li.from_llamaindex_tool(_NamelessTool())
    assert adapter["name"] == "li_tool"
    assert adapter["description"] == ""
    assert adapter["handler"].__name__ == "li_tool"


def test_tool_metadata_missing_entirely():
    class Bare:
        def call(self, **kwargs):  # noqa: ARG002
            return "z"

    adapter = li.from_llamaindex_tool(Bare())
    assert adapter == {
        "name": "li_tool",
        "description": "",
        "input_schema": {},
        "handler": adapter["handler"],
    }


# -- registry / app wiring (lines 120-134) --------------------------------------


def test_register_llamaindex_tool_with_overrides():
    registry = ToolRegistry()
    name = li.register_llamaindex_tool(registry, _ContentTool())
    assert name == "content_tool"
    assert "content_tool" in registry
    assert registry.get("content_tool").handler(who="bo") == "hi bo"


def test_register_llamaindex_tool_name_override():
    registry = ToolRegistry()
    name = li.register_llamaindex_tool(registry, _ContentTool(), name="weather", description="d")
    assert name == "weather"
    assert "weather" in registry
    assert registry.get("weather").spec.description == "d"


def test_add_llamaindex_tool_enables_on_app():
    app = ContextApp(name="t", provider=MockProvider(), model="mock-1")
    li.add_llamaindex_tool(app, _ContentTool(), side_effects="readonly")
    assert "content_tool" in app.enabled_tools
    assert app.tool_registry.get("content_tool").spec.side_effects == "readonly"


# -- retriever read-only contract + filtering -----------------------------------


@pytest.mark.asyncio
async def test_retriever_len_add_delete_are_inert():
    r = li.from_llamaindex_retriever(_Retriever([]), top_k=3)
    assert len(r) == 0
    assert await r.add([Chunk(document_id="d", text="ignored")]) is None
    assert await r.delete(["a", "b"]) == 0
    assert r.top_k == 3
    assert r.name == "llamaindex"


@pytest.mark.asyncio
async def test_retriever_reciprocal_rank_for_unscored_nodes():
    # bare nodes (no .node, no .score) -> reciprocal-rank fallback (line 176).
    results = [_LINode("first", {}), _LINode("second", {})]
    r = li.from_llamaindex_retriever(_Retriever(results))
    hits = await r.search("q")
    assert [h.chunk.text for h in hits] == ["first", "second"]
    assert hits[0].score == 1.0  # 1/(0+1)
    assert hits[1].score == 0.5  # 1/(1+1)
    assert hits[0].chunk.document_id == "llamaindex"  # no source metadata


@pytest.mark.asyncio
async def test_retriever_where_filter_excludes(monkeypatch):
    results = [
        _Scored(_LINode("keep", {"source": "ok", "tag": "yes"}), 0.8),
        _Scored(_LINode("drop", {"source": "no", "tag": "no"}), 0.7),
    ]
    r = li.from_llamaindex_retriever(_Retriever(results))
    hits = await r.search("q", where=lambda chunk: chunk.metadata.get("tag") == "yes")
    assert [h.chunk.text for h in hits] == ["keep"]
    assert hits[0].chunk.document_id == "ok"
    assert hits[0].chunk.source_uri == "ok"


@pytest.mark.asyncio
async def test_retriever_truncates_to_top_k():
    results = [_Scored(_LINode(f"n{i}", {}), 1.0 - i * 0.1) for i in range(5)]
    r = li.from_llamaindex_retriever(_Retriever(results))
    hits = await r.search("q", top_k=2)
    assert [h.chunk.text for h in hits] == ["n0", "n1"]


# -- embedder branches ----------------------------------------------------------


@pytest.mark.asyncio
async def test_embedder_empty_returns_empty_without_calling():
    embedder = li.from_llamaindex_embedding(_BatchEmbedding(), dim=7)
    assert await embedder.embed([]) == []
    assert embedder.dim == 7  # untouched on empty input


@pytest.mark.asyncio
async def test_embedder_batch_updates_dim():
    embedder = li.from_llamaindex_embedding(_BatchEmbedding(), dim=99)
    vectors = await embedder.embed(["ab", "abcd"])
    assert vectors == [[2.0, 0.5, 0.5], [4.0, 0.5, 0.5]]
    assert embedder.dim == 3  # re-derived from first vector


@pytest.mark.asyncio
async def test_embedder_per_text_fallback():
    embedder = li.from_llamaindex_embedding(_SingleEmbedding())
    vectors = await embedder.embed(["xyz"])
    assert vectors == [[3.0, 9.0]]
    assert embedder.dim == 2


@pytest.mark.asyncio
async def test_embedder_keeps_dim_when_backend_returns_no_vectors():
    # Non-empty input but the backend yields nothing -> the 203->205 branch
    # skips the dim update and dim stays at its configured value.
    class _EmptyBatch:
        def get_text_embedding_batch(self, texts):  # noqa: ARG002
            return []

    embedder = li.from_llamaindex_embedding(_EmptyBatch(), dim=42)
    assert await embedder.embed(["something"]) == []
    assert embedder.dim == 42


# -- _unwrap_tool (pure-python, no llama_index needed) --------------------------


def test_unwrap_tool_from_registered_tool():
    registry = ToolRegistry()

    def ping() -> str:
        return "pong"

    registry.register(ping, name="ping", description="returns pong")
    spec, handler = li._unwrap_tool(registry.get("ping"))
    assert spec.name == "ping"
    assert handler() == "pong"


def test_unwrap_tool_from_tuple():
    from vincio.core.types import ToolSpec

    spec = ToolSpec(name="t", description="d")

    def fn():
        return 1

    out_spec, out_handler = li._unwrap_tool((spec, fn))
    assert out_spec is spec
    assert out_handler is fn


def test_unwrap_tool_rejects_unknown():
    with pytest.raises(ConfigError, match="RegisteredTool or"):
        li._unwrap_tool(object())


def test_unwrap_tool_rejects_short_tuple():
    with pytest.raises(ConfigError, match="RegisteredTool or"):
        li._unwrap_tool(("solo",))


# -- export guard when llama_index is absent (covers the lazy-import branches) ---


def test_export_helpers_require_extra_when_import_blocked(monkeypatch):
    """Force every lazy ``import llama_index...`` to fail and assert each
    exporter converts the ImportError into the install-hint ConfigError.

    This exercises the absent-dependency branch even though llama_index is
    installed, by hiding it from the import system for the duration.
    """
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "llama_index" or name.startswith("llama_index."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    for mod in list(sys.modules):
        if mod == "llama_index" or mod.startswith("llama_index."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    for call in (
        lambda: li.to_llamaindex_document(Document(text="x")),
        lambda: li.to_llamaindex_documents([Document(text="x")]),
        lambda: li.to_llamaindex_tool((object(), object())),
        lambda: li.to_llamaindex_retriever(object()),
        lambda: li.to_llamaindex_embedding(object()),
    ):
        with pytest.raises(ConfigError, match=r"vincio\[llamaindex\]"):
            call()


def test_missing_returns_config_error_with_install_hint():
    err = li._missing(ImportError("nope"))
    assert isinstance(err, ConfigError)
    assert 'pip install "vincio[llamaindex]"' in str(err)


# -- export: Vincio -> real LlamaIndex objects ----------------------------------


def test_to_llamaindex_document_real():
    pytest.importorskip("llama_index.core", exc_type=ImportError)
    from llama_index.core import Document as LIDocument

    node = li.to_llamaindex_document(
        Document(text="hello world", metadata={"k": "v"}, source_uri="s3://bucket/a")
    )
    assert isinstance(node, LIDocument)
    assert node.get_content() == "hello world"
    assert node.metadata["k"] == "v"
    assert node.metadata["source"] == "s3://bucket/a"  # source_uri injected


def test_to_llamaindex_document_keeps_explicit_source():
    pytest.importorskip("llama_index.core", exc_type=ImportError)
    node = li.to_llamaindex_document(
        Document(text="t", metadata={"source": "explicit"}, source_uri="other")
    )
    # setdefault must not overwrite an existing "source".
    assert node.metadata["source"] == "explicit"


def test_to_llamaindex_documents_accepts_chunks():
    pytest.importorskip("llama_index.core", exc_type=ImportError)
    docs = li.to_llamaindex_documents(
        [Document(text="d1"), Chunk(document_id="d", text="c1", source_uri="file://c")]
    )
    assert [d.get_content() for d in docs] == ["d1", "c1"]
    assert docs[1].metadata["source"] == "file://c"


def test_to_llamaindex_tool_from_registered_tool():
    pytest.importorskip("llama_index.core", exc_type=ImportError)
    from llama_index.core.tools import FunctionTool

    registry = ToolRegistry()

    def greet(name: str) -> str:
        return f"hi {name}"

    registry.register(greet, name="greet", description="Greets someone.")
    fn_tool = li.to_llamaindex_tool(registry.get("greet"))
    assert isinstance(fn_tool, FunctionTool)
    assert fn_tool.metadata.name == "greet"
    assert fn_tool.metadata.description == "Greets someone."
    out = fn_tool.call(name="ada")
    assert "hi ada" in str(out)


def test_to_llamaindex_tool_from_spec_handler_tuple():
    pytest.importorskip("llama_index.core", exc_type=ImportError)
    from vincio.core.types import ToolSpec

    spec = ToolSpec(name="adder", description="adds")

    def add(a: int, b: int) -> int:
        return a + b

    fn_tool = li.to_llamaindex_tool((spec, add))
    assert fn_tool.metadata.name == "adder"
    assert int(str(fn_tool.call(a=2, b=3))) == 5


def test_to_llamaindex_tool_rejects_unknown_shape():
    pytest.importorskip("llama_index.core", exc_type=ImportError)
    # The FunctionTool import succeeds, so we reach _unwrap_tool's guard.
    with pytest.raises(ConfigError, match="RegisteredTool or"):
        li.to_llamaindex_tool(object())


def test_to_llamaindex_tool_rejects_wrong_length_tuple():
    pytest.importorskip("llama_index.core", exc_type=ImportError)
    with pytest.raises(ConfigError, match="RegisteredTool or"):
        li.to_llamaindex_tool(("only-one",))


# -- export: retriever & embedding bridges (round trips) ------------------------


class _StaticSearchable:
    """A minimal Vincio-style searchable with an async ``search``."""

    def __init__(self, hits):
        self._hits = hits

    async def search(self, query, *, top_k=8, where=None):  # noqa: ARG002
        return self._hits[:top_k]


def test_to_llamaindex_retriever_round_trips_hits():
    pytest.importorskip("llama_index.core", exc_type=ImportError)
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import QueryBundle

    hits = [
        SearchHit(chunk=Chunk(document_id="d", text="alpha", metadata={"m": 1}), score=0.9, source="s"),
        SearchHit(chunk=Chunk(document_id="d", text="beta", metadata={}), score=0.4, source="s"),
    ]
    retriever = li.to_llamaindex_retriever(_StaticSearchable(hits), top_k=5)
    assert isinstance(retriever, BaseRetriever)
    nodes = retriever.retrieve(QueryBundle(query_str="anything"))
    assert [n.node.get_content() for n in nodes] == ["alpha", "beta"]
    assert [n.score for n in nodes] == [0.9, 0.4]
    assert nodes[0].node.metadata == {"m": 1}


def test_to_llamaindex_retriever_accepts_plain_string_query():
    pytest.importorskip("llama_index.core", exc_type=ImportError)
    hits = [SearchHit(chunk=Chunk(document_id="d", text="x"), score=1.0, source="s")]
    retriever = li.to_llamaindex_retriever(_StaticSearchable(hits))
    nodes = retriever.retrieve("plain string")  # not a QueryBundle -> str() branch
    assert nodes[0].node.get_content() == "x"


class _VincioEmbedder:
    async def embed(self, texts):
        return [[float(len(t)), 1.0, 2.0] for t in texts]


def test_to_llamaindex_embedding_sync_and_async():
    pytest.importorskip("llama_index.core", exc_type=ImportError)
    from llama_index.core.embeddings import BaseEmbedding

    emb = li.to_llamaindex_embedding(_VincioEmbedder())
    assert isinstance(emb, BaseEmbedding)
    assert emb._get_query_embedding("ab") == [2.0, 1.0, 2.0]
    assert emb._get_text_embedding("abcd") == [4.0, 1.0, 2.0]
    assert emb._get_text_embeddings(["a", "bb"]) == [[1.0, 1.0, 2.0], [2.0, 1.0, 2.0]]


@pytest.mark.asyncio
async def test_to_llamaindex_embedding_async_query():
    pytest.importorskip("llama_index.core", exc_type=ImportError)
    emb = li.to_llamaindex_embedding(_VincioEmbedder())
    assert await emb._aget_query_embedding("hey") == [3.0, 1.0, 2.0]
