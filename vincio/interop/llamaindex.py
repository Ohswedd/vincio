"""LlamaIndex interop.

Bring LlamaIndex tools, retrievers, readers (loaders), and embeddings into
Vincio — and hand Vincio's back. Like the LangChain bridge, ``from_llamaindex_*``
is duck-typed (no import of ``llama_index``); ``to_llamaindex_*`` builds real
LlamaIndex objects and needs ``pip install "vincio[llamaindex]"``.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import ConfigError
from ..core.types import Chunk, Document, ToolSpec
from ..retrieval.indexes import SearchFilter, SearchHit

__all__ = [
    "from_llamaindex_document",
    "from_llamaindex_documents",
    "from_llamaindex_reader",
    "from_llamaindex_tool",
    "register_llamaindex_tool",
    "add_llamaindex_tool",
    "from_llamaindex_retriever",
    "LlamaIndexRetriever",
    "from_llamaindex_embedding",
    "LlamaIndexEmbedder",
    "to_llamaindex_document",
    "to_llamaindex_documents",
    "to_llamaindex_tool",
    "to_llamaindex_retriever",
    "to_llamaindex_embedding",
]


def _missing(exc: ImportError) -> ConfigError:
    return ConfigError('LlamaIndex export requires: pip install "vincio[llamaindex]"')


def _node_text(node: Any) -> str:
    if hasattr(node, "get_content"):
        try:
            return node.get_content() or ""
        except TypeError:
            return node.get_content(metadata_mode="none") or ""
    return getattr(node, "text", "") or ""


# -- documents / readers --------------------------------------------------------


def from_llamaindex_document(node: Any) -> Document:
    """Convert a LlamaIndex ``Document``/``TextNode`` to a Vincio document."""
    metadata = dict(getattr(node, "metadata", {}) or {})
    source = metadata.get("file_path") or metadata.get("source")
    return Document(
        text=_node_text(node),
        metadata=metadata,
        source_uri=source,
        title=metadata.get("title") or metadata.get("file_name"),
    )


def from_llamaindex_documents(nodes: Any) -> list[Document]:
    return [from_llamaindex_document(node) for node in nodes]


def from_llamaindex_reader(reader: Any, **load_kwargs: Any) -> list[Document]:
    """Run a LlamaIndex reader (``.load_data()``) and convert its output."""
    return from_llamaindex_documents(reader.load_data(**load_kwargs))


# -- tools ----------------------------------------------------------------------


def _tool_metadata(li_tool: Any) -> Any:
    return getattr(li_tool, "metadata", None)


def _tool_input_schema(li_tool: Any) -> dict[str, Any]:
    metadata = _tool_metadata(li_tool)
    fn_schema = getattr(metadata, "fn_schema", None)
    if fn_schema is not None and hasattr(fn_schema, "model_json_schema"):
        return fn_schema.model_json_schema()
    if metadata is not None and hasattr(metadata, "get_parameters_dict"):
        try:
            return metadata.get_parameters_dict()
        except Exception:  # noqa: BLE001 - best-effort schema extraction
            return {}
    return {}


def _tool_handler(li_tool: Any):
    metadata = _tool_metadata(li_tool)
    name = getattr(metadata, "name", None) or "li_tool"

    def handler(**kwargs: Any) -> Any:
        output = li_tool.call(**kwargs) if hasattr(li_tool, "call") else li_tool(**kwargs)
        # FunctionTool returns a ToolOutput; unwrap to the raw value.
        return getattr(output, "raw_output", getattr(output, "content", output))

    handler.__name__ = name
    handler.__doc__ = getattr(metadata, "description", "") or name
    return handler


def from_llamaindex_tool(li_tool: Any) -> dict[str, Any]:
    """Adapt a LlamaIndex tool to a registration kwargs dict (see
    :func:`vincio.interop.langchain.from_langchain_tool`)."""
    metadata = _tool_metadata(li_tool)
    return {
        "name": getattr(metadata, "name", None) or "li_tool",
        "description": getattr(metadata, "description", "") or "",
        "input_schema": _tool_input_schema(li_tool),
        "handler": _tool_handler(li_tool),
    }


def register_llamaindex_tool(registry: Any, li_tool: Any, **overrides: Any) -> str:
    adapter = from_llamaindex_tool(li_tool)
    name = overrides.pop("name", adapter["name"])
    registry.register(
        adapter["handler"],
        name=name,
        description=overrides.pop("description", adapter["description"]),
        input_schema=overrides.pop("input_schema", adapter["input_schema"]) or None,
        **overrides,
    )
    return name


def add_llamaindex_tool(app: Any, li_tool: Any, *, side_effects: str = "external", **overrides: Any) -> Any:
    name = register_llamaindex_tool(app.tool_registry, li_tool, **overrides)
    return app.add_tool(name, side_effects=side_effects)


# -- retrievers -----------------------------------------------------------------


class LlamaIndexRetriever:
    """Wrap a LlamaIndex retriever (``.retrieve``) as a read-only Vincio index."""

    name = "llamaindex"

    def __init__(self, li_retriever: Any, *, top_k: int = 10) -> None:
        self.retriever = li_retriever
        self.top_k = top_k

    def __len__(self) -> int:
        return 0

    async def add(self, chunks: list[Chunk]) -> None:
        return None

    async def delete(self, chunk_ids: list[str]) -> int:
        return 0

    async def search(
        self, query: str, *, top_k: int = 10, where: SearchFilter | None = None
    ) -> list[SearchHit]:
        results = self.retriever.retrieve(query)
        hits: list[SearchHit] = []
        for index, scored in enumerate(results):
            node = getattr(scored, "node", scored)
            metadata = dict(getattr(node, "metadata", {}) or {})
            chunk = Chunk(
                document_id=str(metadata.get("file_path") or metadata.get("source") or "llamaindex"),
                text=_node_text(node),
                index=index,
                metadata=metadata,
                source_uri=metadata.get("file_path") or metadata.get("source"),
            )
            if where is not None and not where(chunk):
                continue
            score = getattr(scored, "score", None)
            score = float(score) if score is not None else 1.0 / (index + 1)
            hits.append(SearchHit(chunk=chunk, score=score, source=self.name))
        return hits[:top_k]


def from_llamaindex_retriever(li_retriever: Any, *, top_k: int = 10) -> LlamaIndexRetriever:
    return LlamaIndexRetriever(li_retriever, top_k=top_k)


# -- embeddings -----------------------------------------------------------------


class LlamaIndexEmbedder:
    """Adapt a LlamaIndex ``BaseEmbedding`` to a Vincio embedder."""

    def __init__(self, li_embedding: Any, *, dim: int = 1536) -> None:
        self.li = li_embedding
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if hasattr(self.li, "get_text_embedding_batch"):
            vectors = self.li.get_text_embedding_batch(list(texts))
        else:
            vectors = [self.li.get_text_embedding(text) for text in texts]
        vectors = [list(v) for v in vectors]
        if vectors:
            self.dim = len(vectors[0])
        return vectors


def from_llamaindex_embedding(li_embedding: Any, *, dim: int = 1536) -> LlamaIndexEmbedder:
    return LlamaIndexEmbedder(li_embedding, dim=dim)


# -- export (Vincio -> LlamaIndex) ----------------------------------------------


def to_llamaindex_document(doc: Document | Chunk) -> Any:
    try:
        from llama_index.core import Document as LIDocument
    except ImportError as exc:
        raise _missing(exc) from exc
    metadata = {**dict(getattr(doc, "metadata", {}) or {})}
    if getattr(doc, "source_uri", None):
        metadata.setdefault("source", doc.source_uri)
    return LIDocument(text=doc.text, metadata=metadata)


def to_llamaindex_documents(docs: list[Document | Chunk]) -> list[Any]:
    return [to_llamaindex_document(doc) for doc in docs]


def _unwrap_tool(tool: Any) -> tuple[ToolSpec, Any]:
    if hasattr(tool, "spec") and hasattr(tool, "handler"):
        return tool.spec, tool.handler
    if isinstance(tool, tuple) and len(tool) == 2:
        return tool
    raise ConfigError("to_llamaindex_tool expects a RegisteredTool or (ToolSpec, handler)")


def to_llamaindex_tool(tool: Any) -> Any:
    """Build a LlamaIndex ``FunctionTool`` from a Vincio registered tool."""
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError as exc:
        raise _missing(exc) from exc
    spec, handler = _unwrap_tool(tool)
    return FunctionTool.from_defaults(fn=handler, name=spec.name, description=spec.description)


def to_llamaindex_retriever(searchable: Any, *, top_k: int = 8) -> Any:
    """Expose a Vincio index/engine (async ``search``) as a LlamaIndex retriever."""
    try:
        from llama_index.core.retrievers import BaseRetriever
        from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
    except ImportError as exc:
        raise _missing(exc) from exc
    from ..providers.base import run_sync

    class _VincioRetriever(BaseRetriever):
        def __init__(self) -> None:
            super().__init__()
            self._searchable = searchable
            self._k = top_k

        def _retrieve(self, query_bundle: Any) -> list[Any]:
            query = query_bundle.query_str if isinstance(query_bundle, QueryBundle) else str(query_bundle)
            hits = run_sync(self._searchable.search(query, top_k=self._k))
            return [
                NodeWithScore(
                    node=TextNode(text=hit.chunk.text, metadata=dict(hit.chunk.metadata)),
                    score=hit.score,
                )
                for hit in hits
            ]

    return _VincioRetriever()


def to_llamaindex_embedding(embedder: Any) -> Any:
    """Expose a Vincio embedder as a LlamaIndex ``BaseEmbedding``."""
    try:
        from llama_index.core.embeddings import BaseEmbedding
    except ImportError as exc:
        raise _missing(exc) from exc
    from ..providers.base import run_sync

    class _VincioEmbedding(BaseEmbedding):
        def _get_query_embedding(self, query: str) -> list[float]:
            return run_sync(embedder.embed([query]))[0]

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return (await embedder.embed([query]))[0]

        def _get_text_embedding(self, text: str) -> list[float]:
            return run_sync(embedder.embed([text]))[0]

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return run_sync(embedder.embed(list(texts)))

    return _VincioEmbedding()
