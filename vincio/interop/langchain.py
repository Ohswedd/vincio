"""LangChain interop.

Bring LangChain tools, retrievers, document loaders, and embeddings into
Vincio — and hand Vincio's back to LangChain. The ``from_langchain_*``
direction is **duck-typed**: it never imports ``langchain`` (it just calls the
methods LangChain objects expose), so existing assets drop in without adding a
dependency. The ``to_langchain_*`` direction constructs real LangChain objects
and therefore needs ``pip install "vincio[langchain]"``.
"""

from __future__ import annotations

import inspect
from typing import Any

from ..core.errors import ConfigError
from ..core.types import Chunk, Document, ToolSpec
from ..retrieval.indexes import SearchFilter, SearchHit

__all__ = [
    "from_langchain_document",
    "from_langchain_documents",
    "from_langchain_loader",
    "from_langchain_tool",
    "register_langchain_tool",
    "add_langchain_tool",
    "from_langchain_retriever",
    "LangChainRetriever",
    "from_langchain_embeddings",
    "LangChainEmbedder",
    "to_langchain_document",
    "to_langchain_documents",
    "to_langchain_tool",
    "to_langchain_retriever",
    "to_langchain_embeddings",
]


def _missing(exc: ImportError) -> ConfigError:
    return ConfigError('LangChain export requires: pip install "vincio[langchain]"')


# -- documents / loaders --------------------------------------------------------


def from_langchain_document(lc_doc: Any) -> Document:
    """Convert a LangChain ``Document`` (``.page_content`` + ``.metadata``)."""
    metadata = dict(getattr(lc_doc, "metadata", {}) or {})
    return Document(
        text=getattr(lc_doc, "page_content", "") or "",
        metadata=metadata,
        source_uri=metadata.get("source"),
        title=metadata.get("title"),
    )


def from_langchain_documents(lc_docs: Any) -> list[Document]:
    return [from_langchain_document(doc) for doc in lc_docs]


def from_langchain_loader(loader: Any) -> list[Document]:
    """Run a LangChain document loader (``.load()``) and convert its output."""
    return from_langchain_documents(loader.load())


# -- tools ----------------------------------------------------------------------


def _tool_input_schema(lc_tool: Any) -> dict[str, Any]:
    args_schema = getattr(lc_tool, "args_schema", None)
    if args_schema is not None and hasattr(args_schema, "model_json_schema"):
        return args_schema.model_json_schema()
    args = getattr(lc_tool, "args", None)
    if isinstance(args, dict):
        return {"type": "object", "properties": args}
    return {}


def _tool_handler(lc_tool: Any):
    name = getattr(lc_tool, "name", None) or "lc_tool"

    def handler(**kwargs: Any) -> Any:
        if hasattr(lc_tool, "invoke"):
            return lc_tool.invoke(kwargs)
        if hasattr(lc_tool, "run"):
            return lc_tool.run(kwargs)
        return lc_tool(**kwargs)

    handler.__name__ = name
    handler.__doc__ = getattr(lc_tool, "description", "") or name
    return handler


def from_langchain_tool(lc_tool: Any) -> dict[str, Any]:
    """Adapt a LangChain tool to a registration kwargs dict.

    Returns ``{"name", "description", "input_schema", "handler"}`` — pass it to
    :meth:`vincio.tools.ToolRegistry.register`, or use
    :func:`register_langchain_tool` / :func:`add_langchain_tool`.
    """
    return {
        "name": getattr(lc_tool, "name", None) or "lc_tool",
        "description": getattr(lc_tool, "description", "") or "",
        "input_schema": _tool_input_schema(lc_tool),
        "handler": _tool_handler(lc_tool),
    }


def register_langchain_tool(registry: Any, lc_tool: Any, **overrides: Any) -> str:
    """Register a LangChain tool on a :class:`~vincio.tools.ToolRegistry`."""
    adapter = from_langchain_tool(lc_tool)
    name = overrides.pop("name", adapter["name"])
    registry.register(
        adapter["handler"],
        name=name,
        description=overrides.pop("description", adapter["description"]),
        input_schema=overrides.pop("input_schema", adapter["input_schema"]) or None,
        **overrides,
    )
    return name


def add_langchain_tool(app: Any, lc_tool: Any, *, side_effects: str = "external", **overrides: Any) -> Any:
    """Register *and enable* a LangChain tool on a :class:`ContextApp`.

    Defaults to ``side_effects="external"`` (the safe assumption for an opaque
    third-party tool — it is gated by the ``allow_external_tools`` policy).
    """
    name = register_langchain_tool(app.tool_registry, lc_tool, **overrides)
    return app.add_tool(name, side_effects=side_effects)


# -- retrievers -----------------------------------------------------------------


class LangChainRetriever:
    """Wrap a LangChain retriever as a read-only Vincio index.

    Exposes ``search`` (so it slots into a hybrid :class:`RetrievalEngine` as
    one source) plus no-op ``add``/``delete`` — ingestion stays on the
    LangChain side; this only queries.
    """

    name = "langchain"

    def __init__(self, lc_retriever: Any, *, top_k: int = 10) -> None:
        self.retriever = lc_retriever
        self.top_k = top_k

    def __len__(self) -> int:
        return 0

    def _invoke(self, query: str) -> list[Any]:
        if hasattr(self.retriever, "invoke"):
            return self.retriever.invoke(query)
        return self.retriever.get_relevant_documents(query)

    async def add(self, chunks: list[Chunk]) -> None:  # read-only adapter
        return None

    async def delete(self, chunk_ids: list[str]) -> int:
        return 0

    async def search(
        self, query: str, *, top_k: int = 10, where: SearchFilter | None = None
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for index, lc_doc in enumerate(self._invoke(query)):
            metadata = dict(getattr(lc_doc, "metadata", {}) or {})
            chunk = Chunk(
                document_id=str(metadata.get("source") or "langchain"),
                text=getattr(lc_doc, "page_content", "") or "",
                index=index,
                metadata=metadata,
                source_uri=metadata.get("source"),
            )
            if where is not None and not where(chunk):
                continue
            # Retrievers return ranked, not scored, documents — use reciprocal rank.
            hits.append(SearchHit(chunk=chunk, score=1.0 / (index + 1), source=self.name))
        return hits[:top_k]


def from_langchain_retriever(lc_retriever: Any, *, top_k: int = 10) -> LangChainRetriever:
    return LangChainRetriever(lc_retriever, top_k=top_k)


# -- embeddings -----------------------------------------------------------------


class LangChainEmbedder:
    """Adapt LangChain ``Embeddings`` (``.embed_documents``) to a Vincio embedder."""

    def __init__(self, lc_embeddings: Any, *, dim: int = 1536) -> None:
        self.lc = lc_embeddings
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.lc.embed_documents(list(texts))
        vectors = [list(v) for v in vectors]
        if vectors:
            self.dim = len(vectors[0])
        return vectors


def from_langchain_embeddings(lc_embeddings: Any, *, dim: int = 1536) -> LangChainEmbedder:
    return LangChainEmbedder(lc_embeddings, dim=dim)


# -- export (Vincio -> LangChain) -----------------------------------------------


def to_langchain_document(doc: Document | Chunk) -> Any:
    try:
        from langchain_core.documents import Document as LCDocument
    except ImportError as exc:
        raise _missing(exc) from exc
    metadata = {**dict(getattr(doc, "metadata", {}) or {})}
    source_uri = getattr(doc, "source_uri", None)
    if source_uri:
        metadata.setdefault("source", source_uri)
    title = getattr(doc, "title", None)
    if title:
        metadata.setdefault("title", title)
    return LCDocument(page_content=doc.text, metadata=metadata)


def to_langchain_documents(docs: list[Document | Chunk]) -> list[Any]:
    return [to_langchain_document(doc) for doc in docs]


def _unwrap_tool(tool: Any) -> tuple[ToolSpec, Any]:
    if hasattr(tool, "spec") and hasattr(tool, "handler"):
        return tool.spec, tool.handler
    if isinstance(tool, tuple) and len(tool) == 2:
        return tool
    raise ConfigError("to_langchain_tool expects a RegisteredTool or (ToolSpec, handler)")


def to_langchain_tool(tool: Any) -> Any:
    """Build a LangChain ``StructuredTool`` from a Vincio registered tool."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise _missing(exc) from exc
    spec, handler = _unwrap_tool(tool)
    if inspect.iscoroutinefunction(handler):
        return StructuredTool.from_function(
            coroutine=handler, name=spec.name, description=spec.description
        )
    return StructuredTool.from_function(func=handler, name=spec.name, description=spec.description)


def to_langchain_retriever(searchable: Any, *, top_k: int = 8) -> Any:
    """Expose a Vincio index/engine (anything with async ``search``) as a
    LangChain ``BaseRetriever``."""
    try:
        from langchain_core.documents import Document as LCDocument
        from langchain_core.retrievers import BaseRetriever
    except ImportError as exc:
        raise _missing(exc) from exc
    from ..providers.base import run_sync

    class _VincioRetriever(BaseRetriever):
        model_config = {"arbitrary_types_allowed": True}
        searchable: Any = None
        k: int = 8

        def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Any]:
            hits = run_sync(self.searchable.search(query, top_k=self.k))
            return [
                LCDocument(page_content=hit.chunk.text, metadata=dict(hit.chunk.metadata))
                for hit in hits
            ]

    return _VincioRetriever(searchable=searchable, k=top_k)


def to_langchain_embeddings(embedder: Any) -> Any:
    """Expose a Vincio embedder as a LangChain ``Embeddings``."""
    try:
        from langchain_core.embeddings import Embeddings
    except ImportError as exc:
        raise _missing(exc) from exc
    from ..providers.base import run_sync

    class _VincioEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return run_sync(embedder.embed(list(texts)))

        def embed_query(self, text: str) -> list[float]:
            return run_sync(embedder.embed([text]))[0]

    return _VincioEmbeddings()
