"""Framework interop: LangChain + LlamaIndex bridges, exercised with
duck-typed fakes so the suite never needs the real frameworks installed."""

from __future__ import annotations

import importlib.util

import pytest

from vincio import ContextApp
from vincio.core.errors import ConfigError
from vincio.core.types import Document
from vincio.interop import langchain as lc
from vincio.interop import llamaindex as li
from vincio.providers import MockProvider
from vincio.tools import ToolRegistry

_HAS_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None
_HAS_LLAMAINDEX = importlib.util.find_spec("llama_index") is not None


# -- fakes ----------------------------------------------------------------------


class FakeLCDoc:
    def __init__(self, page_content: str, metadata: dict) -> None:
        self.page_content = page_content
        self.metadata = metadata


class FakeLCTool:
    name = "web_search"
    description = "Search the web for a query."
    args = {"query": {"type": "string"}}

    def invoke(self, payload: dict) -> str:
        return f"results for {payload['query']}"


class FakeLCRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query: str):
        return self._docs


class FakeLCEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


class _LIMeta:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.fn_schema = None

    def get_parameters_dict(self):
        return {"type": "object", "properties": {"city": {"type": "string"}}}


class _LIToolOutput:
    def __init__(self, raw_output):
        self.raw_output = raw_output


class FakeLITool:
    def __init__(self):
        self.metadata = _LIMeta("get_weather", "Look up the weather.")

    def call(self, **kwargs):
        return _LIToolOutput(f"sunny in {kwargs['city']}")


class FakeLINode:
    def __init__(self, text, metadata):
        self._text = text
        self.metadata = metadata

    def get_content(self):
        return self._text


class FakeLINodeWithScore:
    def __init__(self, node, score):
        self.node = node
        self.score = score


class FakeLIRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query):
        return self._results


class FakeLIEmbedding:
    def get_text_embedding_batch(self, texts):
        return [[float(len(t)), 2.0] for t in texts]


# -- LangChain from_* -----------------------------------------------------------


def test_from_langchain_document_maps_fields():
    doc = lc.from_langchain_document(FakeLCDoc("hello", {"source": "s3://x", "title": "Doc"}))
    assert doc.text == "hello"
    assert doc.source_uri == "s3://x"
    assert doc.title == "Doc"


def test_from_langchain_loader():
    class Loader:
        def load(self):
            return [FakeLCDoc("a", {}), FakeLCDoc("b", {})]

    docs = lc.from_langchain_loader(Loader())
    assert [d.text for d in docs] == ["a", "b"]


def test_from_langchain_tool_handler_and_schema():
    adapter = lc.from_langchain_tool(FakeLCTool())
    assert adapter["name"] == "web_search"
    assert adapter["input_schema"]["properties"]["query"]["type"] == "string"
    assert adapter["handler"](query="vincio") == "results for vincio"


def test_register_langchain_tool_on_registry():
    registry = ToolRegistry()
    name = lc.register_langchain_tool(registry, FakeLCTool())
    assert name == "web_search"
    assert "web_search" in registry
    assert registry.get("web_search").spec.description.startswith("Search the web")


def test_add_langchain_tool_to_app():
    app = ContextApp(name="t", provider=MockProvider(), model="mock-1")
    lc.add_langchain_tool(app, FakeLCTool())
    assert "web_search" in app.enabled_tools
    assert app.tool_registry.get("web_search").spec.side_effects == "external"


@pytest.mark.asyncio
async def test_from_langchain_retriever_search():
    docs = [FakeLCDoc("first", {"source": "a"}), FakeLCDoc("second", {"source": "b"})]
    retriever = lc.from_langchain_retriever(FakeLCRetriever(docs))
    hits = await retriever.search("q", top_k=2)
    assert [h.chunk.text for h in hits] == ["first", "second"]
    assert hits[0].score > hits[1].score  # reciprocal-rank ordering


@pytest.mark.asyncio
async def test_from_langchain_embeddings():
    embedder = lc.from_langchain_embeddings(FakeLCEmbeddings())
    vectors = await embedder.embed(["ab", "abcd"])
    assert vectors == [[2.0, 1.0], [4.0, 1.0]]
    assert embedder.dim == 2


@pytest.mark.skipif(_HAS_LANGCHAIN, reason="langchain_core is installed")
def test_to_langchain_requires_extra():
    with pytest.raises(ConfigError):
        lc.to_langchain_documents([Document(text="x")])  # triggers the lazy import guard


# -- LlamaIndex from_* ----------------------------------------------------------


def test_from_llamaindex_document():
    doc = li.from_llamaindex_document(FakeLINode("body", {"file_path": "/tmp/x.md"}))
    assert doc.text == "body"
    assert doc.source_uri == "/tmp/x.md"


def test_from_llamaindex_reader():
    class Reader:
        def load_data(self):
            return [FakeLINode("one", {}), FakeLINode("two", {})]

    docs = li.from_llamaindex_reader(Reader())
    assert [d.text for d in docs] == ["one", "two"]


def test_from_llamaindex_tool():
    adapter = li.from_llamaindex_tool(FakeLITool())
    assert adapter["name"] == "get_weather"
    assert adapter["input_schema"]["properties"]["city"]["type"] == "string"
    assert adapter["handler"](city="Rome") == "sunny in Rome"


@pytest.mark.asyncio
async def test_from_llamaindex_retriever_uses_scores():
    results = [
        FakeLINodeWithScore(FakeLINode("hot", {"source": "a"}), 0.9),
        FakeLINodeWithScore(FakeLINode("cold", {"source": "b"}), 0.3),
    ]
    retriever = li.from_llamaindex_retriever(FakeLIRetriever(results))
    hits = await retriever.search("q", top_k=2)
    assert [round(h.score, 1) for h in hits] == [0.9, 0.3]


@pytest.mark.asyncio
async def test_from_llamaindex_embedding():
    embedder = li.from_llamaindex_embedding(FakeLIEmbedding())
    assert await embedder.embed(["xy"]) == [[2.0, 2.0]]


@pytest.mark.skipif(_HAS_LLAMAINDEX, reason="llama_index is installed")
def test_to_llamaindex_requires_extra():
    with pytest.raises(ConfigError):
        li.to_llamaindex_documents([Document(text="x")])
