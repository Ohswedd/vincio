"""Haystack interop.

Bring Haystack 2.x documents, retrievers, embedders, and components into
Vincio — and hand Vincio documents back. The ``from_haystack_*`` direction is
**duck-typed**: it never imports ``haystack`` (it just calls the methods
Haystack objects expose — components run via ``.run(**kwargs)``), so existing
assets drop in without adding a dependency. ``to_haystack_*`` constructs real
Haystack objects and needs ``pip install "vincio[haystack]"``.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import ConfigError
from ..core.types import Chunk, Document
from ..retrieval.indexes import SearchFilter, SearchHit

__all__ = [
    "from_haystack_document",
    "from_haystack_documents",
    "from_haystack_retriever",
    "from_haystack_component",
    "register_haystack_component",
    "add_haystack_component",
    "HaystackRetriever",
    "from_haystack_embedder",
    "HaystackEmbedder",
    "to_haystack_document",
    "to_haystack_documents",
]


# -- documents ------------------------------------------------------------------


def from_haystack_document(hs_doc: Any) -> Document:
    """Convert a Haystack ``Document`` (``.content`` + ``.meta``)."""
    meta = dict(getattr(hs_doc, "meta", {}) or {})
    return Document(
        text=getattr(hs_doc, "content", "") or "",
        metadata=meta,
        source_uri=meta.get("source") or meta.get("url") or meta.get("file_path"),
        title=meta.get("title"),
    )


def from_haystack_documents(hs_docs: Any) -> list[Document]:
    return [from_haystack_document(doc) for doc in hs_docs]


# -- components (as tools) ------------------------------------------------------


def _component_handler(component: Any):
    name = getattr(component, "name", None) or type(component).__name__

    def handler(**kwargs: Any) -> Any:
        return component.run(**kwargs)

    handler.__name__ = name
    handler.__doc__ = getattr(component, "description", "") or f"Haystack component {name}"
    return handler


def from_haystack_component(component: Any, *, name: str | None = None) -> dict[str, Any]:
    """Adapt a Haystack component (anything with ``.run(**kwargs)``) to a
    registration kwargs dict (``{"name", "description", "input_schema", "handler"}``)."""
    resolved = name or getattr(component, "name", None) or type(component).__name__
    return {
        "name": resolved,
        "description": getattr(component, "description", "") or "",
        "input_schema": {},
        "handler": _component_handler(component),
    }


def register_haystack_component(registry: Any, component: Any, **overrides: Any) -> str:
    """Register a Haystack component on a :class:`~vincio.tools.ToolRegistry`."""
    adapter = from_haystack_component(component, name=overrides.pop("name", None))
    registry.register(
        adapter["handler"],
        name=adapter["name"],
        description=overrides.pop("description", adapter["description"]),
        input_schema=overrides.pop("input_schema", adapter["input_schema"]) or None,
        **overrides,
    )
    return adapter["name"]


def add_haystack_component(app: Any, component: Any, *, side_effects: str = "external", **overrides: Any) -> Any:
    """Register *and enable* a Haystack component as a tool on a :class:`ContextApp`."""
    name = register_haystack_component(app.tool_registry, component, **overrides)
    return app.add_tool(name, side_effects=side_effects)


# -- retrievers -----------------------------------------------------------------


class HaystackRetriever:
    """Wrap a Haystack retriever as a read-only Vincio index.

    Haystack retrievers are components: calling ``.run(query=...)`` returns
    ``{"documents": [...]}``. This adapter exposes async ``search`` (so it slots
    into a hybrid engine as one source) plus no-op ``add``/``delete``.
    """

    name = "haystack"

    def __init__(self, retriever: Any, *, top_k: int = 10, query_param: str = "query") -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.query_param = query_param

    def __len__(self) -> int:
        return 0

    def _run(self, query: str) -> list[Any]:
        result = self.retriever.run(**{self.query_param: query})
        if isinstance(result, dict):
            return result.get("documents", [])
        return list(result or [])

    async def add(self, chunks: list[Chunk]) -> None:  # read-only adapter
        return None

    async def delete(self, chunk_ids: list[str]) -> int:
        return 0

    async def search(
        self, query: str, *, top_k: int = 10, where: SearchFilter | None = None
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for index, hs_doc in enumerate(self._run(query)):
            meta = dict(getattr(hs_doc, "meta", {}) or {})
            chunk = Chunk(
                document_id=str(meta.get("source") or meta.get("file_path") or "haystack"),
                text=getattr(hs_doc, "content", "") or "",
                index=index,
                metadata=meta,
                source_uri=meta.get("source") or meta.get("url"),
            )
            if where is not None and not where(chunk):
                continue
            # Haystack retrievers usually attach a similarity score; fall back to
            # reciprocal rank when they rank without scoring.
            score = getattr(hs_doc, "score", None)
            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=float(score) if score is not None else 1.0 / (index + 1),
                    source=self.name,
                )
            )
        return hits[:top_k]


def from_haystack_retriever(retriever: Any, *, top_k: int = 10) -> HaystackRetriever:
    return HaystackRetriever(retriever, top_k=top_k)


# -- embedders ------------------------------------------------------------------


class HaystackEmbedder:
    """Adapt a Haystack text embedder (``.run(text=...) -> {"embedding": [...]}``)."""

    def __init__(self, embedder: Any, *, dim: int = 768) -> None:
        self.embedder = embedder
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for text in texts:
            result = self.embedder.run(text=text)
            embedding = result["embedding"] if isinstance(result, dict) else result
            vectors.append([float(x) for x in embedding])
        if vectors:
            self.dim = len(vectors[0])
        return vectors


def from_haystack_embedder(embedder: Any, *, dim: int = 768) -> HaystackEmbedder:
    return HaystackEmbedder(embedder, dim=dim)


# -- export (Vincio -> Haystack) ------------------------------------------------


def to_haystack_document(doc: Document | Chunk) -> Any:
    try:
        from haystack import Document as HSDocument
    except ImportError:  # pragma: no cover - exercised only without the extra
        try:
            from haystack.dataclasses import Document as HSDocument
        except ImportError as exc:
            raise ConfigError('Haystack export requires: pip install "vincio[haystack]"') from exc
    meta = {**dict(getattr(doc, "metadata", {}) or {})}
    source_uri = getattr(doc, "source_uri", None)
    if source_uri:
        meta.setdefault("source", source_uri)
    title = getattr(doc, "title", None)
    if title:
        meta.setdefault("title", title)
    return HSDocument(content=doc.text, meta=meta)


def to_haystack_documents(docs: list[Document | Chunk]) -> list[Any]:
    return [to_haystack_document(doc) for doc in docs]
