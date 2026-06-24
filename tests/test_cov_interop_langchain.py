"""Real-behavior coverage for vincio.interop.langchain.

The ``from_langchain_*`` (import) direction is duck-typed and exercised with
plain fakes that expose only the methods LangChain objects expose. The
``to_langchain_*`` (export) direction builds real ``langchain_core`` objects,
which are installed in this environment, so those round-trips are exercised for
real (no mocking). The graceful-absence branch is covered without uninstalling
the package by calling the private ``_missing`` helper directly.
"""

from __future__ import annotations

import importlib.util

import pytest

from vincio.core.errors import ConfigError
from vincio.core.types import Chunk, Document, ToolSpec
from vincio.interop import langchain as lc
from vincio.providers import MockProvider
from vincio.retrieval.indexes import SearchHit
from vincio.tools import ToolRegistry

_HAS_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None
_needs_lc = pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain_core not installed")


# -- duck-typed fakes (import direction) ----------------------------------------


class _SchemaModel:
    """Stands in for a pydantic args_schema with model_json_schema()."""

    @staticmethod
    def model_json_schema() -> dict:
        return {"type": "object", "properties": {"q": {"type": "string"}}, "title": "Args"}


class _ToolWithSchema:
    name = "schema_tool"
    description = "Has a pydantic args_schema."
    args_schema = _SchemaModel


class _ToolRunOnly:
    """No invoke(); only run() — covers the second handler branch."""

    name = "run_tool"
    description = "run-only"

    def run(self, payload: dict) -> str:
        return f"ran {payload['x']}"


class _ToolCallable:
    """No invoke()/run(); plain callable — covers the third handler branch."""

    name = "callable_tool"
    description = "callable"

    def __call__(self, **kwargs) -> str:
        return f"called {kwargs['y']}"


class _NamelessTool:
    """No name/description/args at all — exercises the fallbacks."""

    def invoke(self, payload: dict) -> str:  # noqa: D401
        return "ok"


class _GetRelevantRetriever:
    """Old-style retriever: no invoke(), only get_relevant_documents()."""

    def __init__(self, docs):
        self._docs = docs

    def get_relevant_documents(self, query: str):
        return self._docs


class _LCDoc:
    def __init__(self, page_content="", metadata=None):
        self.page_content = page_content
        self.metadata = metadata


# -- from_langchain_document / documents / loader -------------------------------


def test_from_document_maps_text_metadata_source_title():
    doc = lc.from_langchain_document(
        _LCDoc("hello", {"source": "s3://x", "title": "Doc", "extra": 1})
    )
    assert doc.text == "hello"
    assert doc.source_uri == "s3://x"
    assert doc.title == "Doc"
    assert doc.metadata["extra"] == 1


def test_from_document_handles_missing_attrs_and_none_metadata():
    # An object with no page_content/metadata at all -> empty text, empty metadata.
    class _Bare:
        pass

    doc = lc.from_langchain_document(_Bare())
    assert doc.text == ""
    assert doc.metadata == {}
    assert doc.source_uri is None
    assert doc.title is None
    # And a doc whose metadata is explicitly None coerces to {}.
    doc2 = lc.from_langchain_document(_LCDoc("x", None))
    assert doc2.metadata == {}


def test_from_documents_converts_every_element():
    docs = lc.from_langchain_documents([_LCDoc("a", {}), _LCDoc("b", {})])
    assert [d.text for d in docs] == ["a", "b"]


def test_from_loader_runs_load_and_converts():
    class _Loader:
        def load(self):
            return [_LCDoc("one", {"source": "u"}), _LCDoc("two", {})]

    docs = lc.from_langchain_loader(_Loader())
    assert [d.text for d in docs] == ["one", "two"]
    assert docs[0].source_uri == "u"


# -- _tool_input_schema ---------------------------------------------------------


def test_input_schema_from_args_dict_branch():
    class _ToolArgsDict:
        name = "t"
        description = "d"
        args = {"city": {"type": "string"}}

        def invoke(self, payload):
            return payload

    schema = lc.from_langchain_tool(_ToolArgsDict())["input_schema"]
    assert schema == {"type": "object", "properties": {"city": {"type": "string"}}}


def test_handler_prefers_invoke_when_present():
    class _ToolInvoke:
        name = "inv"
        description = "d"

        def invoke(self, payload):
            return f"invoked {payload['k']}"

        def run(self, payload):  # should NOT be called — invoke wins
            raise AssertionError("run() must not be used when invoke() exists")

    handler = lc.from_langchain_tool(_ToolInvoke())["handler"]
    assert handler(k="v") == "invoked v"


def test_input_schema_from_pydantic_args_schema():
    schema = lc.from_langchain_tool(_ToolWithSchema())["input_schema"]
    assert schema["properties"]["q"]["type"] == "string"
    assert schema["title"] == "Args"


def test_input_schema_empty_when_no_args():
    # args is None and there is no args_schema -> empty dict (line 76 branch).
    assert lc.from_langchain_tool(_NamelessTool())["input_schema"] == {}


def test_nameless_tool_uses_fallback_name_and_description():
    adapter = lc.from_langchain_tool(_NamelessTool())
    assert adapter["name"] == "lc_tool"
    assert adapter["description"] == ""


# -- _tool_handler dispatch branches --------------------------------------------


def test_handler_prefers_run_when_no_invoke():
    handler = lc.from_langchain_tool(_ToolRunOnly())["handler"]
    assert handler(x="hi") == "ran hi"
    assert handler.__name__ == "run_tool"
    assert handler.__doc__ == "run-only"


def test_handler_falls_through_to_call():
    handler = lc.from_langchain_tool(_ToolCallable())["handler"]
    assert handler(y="z") == "called z"


def test_handler_name_falls_back_when_tool_has_no_name():
    handler = lc.from_langchain_tool(_NamelessTool())["handler"]
    assert handler.__name__ == "lc_tool"


# -- LangChainRetriever ---------------------------------------------------------


@pytest.mark.asyncio
async def test_retriever_len_and_noop_mutations():
    r = lc.from_langchain_retriever(_GetRelevantRetriever([]))
    assert len(r) == 0
    assert await r.add([Chunk(document_id="d", text="t", index=0)]) is None
    assert await r.delete(["a", "b"]) == 0


@pytest.mark.asyncio
async def test_retriever_uses_get_relevant_documents_branch():
    docs = [_LCDoc("alpha", {"source": "s1"}), _LCDoc("beta", {"source": "s2"})]
    r = lc.from_langchain_retriever(_GetRelevantRetriever(docs))
    hits = await r.search("q")
    assert [h.chunk.text for h in hits] == ["alpha", "beta"]
    # Reciprocal-rank scoring: 1/(0+1) then 1/(1+1).
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.5)
    assert hits[0].chunk.document_id == "s1"


@pytest.mark.asyncio
async def test_retriever_prefers_invoke_branch():
    class _InvokeRetriever:
        def __init__(self, docs):
            self._docs = docs

        def invoke(self, query):
            return self._docs

        def get_relevant_documents(self, query):  # must not be reached
            raise AssertionError("invoke() should win over get_relevant_documents()")

    docs = [_LCDoc("first", {"source": "a"}), _LCDoc("second", {"source": "b"})]
    r = lc.from_langchain_retriever(_InvokeRetriever(docs))
    hits = await r.search("q", top_k=2)
    assert [h.chunk.text for h in hits] == ["first", "second"]


@pytest.mark.asyncio
async def test_retriever_missing_source_defaults_document_id():
    r = lc.from_langchain_retriever(_GetRelevantRetriever([_LCDoc("x", {})]))
    hit = (await r.search("q"))[0]
    assert hit.chunk.document_id == "langchain"
    assert hit.chunk.source_uri is None


@pytest.mark.asyncio
async def test_retriever_where_filter_excludes_nonmatching():
    docs = [_LCDoc("keep", {"source": "s1", "kind": "k"}), _LCDoc("drop", {"source": "s2"})]
    r = lc.from_langchain_retriever(_GetRelevantRetriever(docs))
    hits = await r.search("q", where=lambda chunk: chunk.metadata.get("kind") == "k")
    assert [h.chunk.text for h in hits] == ["keep"]


@pytest.mark.asyncio
async def test_retriever_top_k_truncates():
    docs = [_LCDoc(str(i), {"source": str(i)}) for i in range(5)]
    r = lc.from_langchain_retriever(_GetRelevantRetriever(docs))
    hits = await r.search("q", top_k=2)
    assert [h.chunk.text for h in hits] == ["0", "1"]


# -- LangChainEmbedder ----------------------------------------------------------


@pytest.mark.asyncio
async def test_embedder_empty_input_returns_empty_and_keeps_dim():
    embedder = lc.from_langchain_embeddings(_FakeEmbeddings(), dim=7)
    assert await embedder.embed([]) == []
    assert embedder.dim == 7  # unchanged (early return before dim update)


@pytest.mark.asyncio
async def test_embedder_updates_dim_from_first_vector():
    embedder = lc.from_langchain_embeddings(_FakeEmbeddings(), dim=99)
    vectors = await embedder.embed(["ab", "cdef"])
    assert vectors == [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    assert embedder.dim == 3  # learned from the actual returned width


class _FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(t)), 0.0, 0.0] for t in texts]


# -- _missing / graceful-absence branch -----------------------------------------


def test_missing_returns_config_error_with_install_hint():
    err = lc._missing(ImportError("boom"))
    assert isinstance(err, ConfigError)
    assert 'pip install "vincio[langchain]"' in str(err)


# -- export direction (real langchain_core) -------------------------------------


@_needs_lc
def test_to_langchain_document_round_trips_metadata():
    doc = Document(text="body", metadata={"k": "v"}, source_uri="s3://x", title="T")
    lc_doc = lc.to_langchain_document(doc)
    assert lc_doc.page_content == "body"
    assert lc_doc.metadata["k"] == "v"
    assert lc_doc.metadata["source"] == "s3://x"
    assert lc_doc.metadata["title"] == "T"


@_needs_lc
def test_to_langchain_document_does_not_overwrite_existing_source():
    doc = Document(text="b", metadata={"source": "kept"}, source_uri="other")
    lc_doc = lc.to_langchain_document(doc)
    assert lc_doc.metadata["source"] == "kept"  # setdefault, not overwrite


@_needs_lc
def test_to_langchain_document_no_source_or_title_keys():
    doc = Document(text="b")
    lc_doc = lc.to_langchain_document(doc)
    assert "source" not in lc_doc.metadata
    assert "title" not in lc_doc.metadata


@_needs_lc
def test_to_langchain_documents_maps_each():
    docs = [Document(text="a"), Document(text="b")]
    out = lc.to_langchain_documents(docs)
    assert [d.page_content for d in out] == ["a", "b"]


# -- _unwrap_tool ---------------------------------------------------------------


def test_unwrap_tool_from_registered_tool():
    registry = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    registry.register(add, name="add", description="adds")
    spec, handler = lc._unwrap_tool(registry.get("add"))
    assert spec.name == "add"
    assert handler(2, 3) == 5


def test_unwrap_tool_from_tuple():
    spec = ToolSpec(name="t", description="d")

    def h():
        return 1

    out_spec, out_handler = lc._unwrap_tool((spec, h))
    assert out_spec is spec
    assert out_handler is h


def test_unwrap_tool_rejects_other_shapes():
    with pytest.raises(ConfigError, match="RegisteredTool or"):
        lc._unwrap_tool(("only-one-element",))


# -- to_langchain_tool (sync + async handlers) ----------------------------------


@_needs_lc
def test_to_langchain_tool_sync_handler():
    spec = ToolSpec(name="greet", description="say hi")

    def greet(name: str) -> str:
        return f"hi {name}"

    tool = lc.to_langchain_tool((spec, greet))
    assert tool.name == "greet"
    assert tool.description == "say hi"
    assert tool.invoke({"name": "vincio"}) == "hi vincio"


@pytest.mark.asyncio
@_needs_lc
async def test_to_langchain_tool_async_handler_uses_coroutine_path():
    spec = ToolSpec(name="aecho", description="async echo")

    async def aecho(value: str) -> str:
        return f"echo:{value}"

    tool = lc.to_langchain_tool((spec, aecho))
    assert tool.name == "aecho"
    result = await tool.ainvoke({"value": "x"})
    assert result == "echo:x"


# -- to_langchain_retriever -----------------------------------------------------


class _StubIndex:
    """Minimal Vincio-style searchable with an async search()."""

    def __init__(self, hits):
        self._hits = hits
        self.last_top_k = None

    async def search(self, query, *, top_k=8, where=None):
        self.last_top_k = top_k
        return self._hits


@_needs_lc
def test_to_langchain_retriever_converts_hits_to_documents():
    hits = [
        SearchHit(chunk=Chunk(document_id="d1", text="one", index=0, metadata={"m": 1}), score=0.9),
        SearchHit(chunk=Chunk(document_id="d2", text="two", index=1, metadata={}), score=0.5),
    ]
    index = _StubIndex(hits)
    retriever = lc.to_langchain_retriever(index, top_k=3)
    docs = retriever.invoke("anything")
    assert [d.page_content for d in docs] == ["one", "two"]
    assert docs[0].metadata == {"m": 1}
    assert index.last_top_k == 3  # top_k threaded through


# -- to_langchain_embeddings ----------------------------------------------------


class _StubEmbedder:
    async def embed(self, texts):
        return [[float(len(t)), 1.0] for t in texts]


@_needs_lc
def test_to_langchain_embeddings_documents_and_query():
    emb = lc.to_langchain_embeddings(_StubEmbedder())
    assert emb.embed_documents(["ab", "cde"]) == [[2.0, 1.0], [3.0, 1.0]]
    # embed_query takes the first (and only) vector.
    assert emb.embed_query("abcd") == [4.0, 1.0]


# -- a small end-to-end safety check on the import-side registration ------------


def test_add_langchain_tool_marks_external_side_effects():
    from vincio import ContextApp

    app = ContextApp(name="t", provider=MockProvider(), model="mock-1")
    lc.add_langchain_tool(app, _ToolRunOnly())
    assert "run_tool" in app.enabled_tools
    assert app.tool_registry.get("run_tool").spec.side_effects == "external"
