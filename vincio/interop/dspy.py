"""DSPy interop.

Bring compiled DSPy modules, signatures, and retrievers into Vincio — and
expose a Vincio provider to DSPy as a language model. The ``from_dspy_*``
direction is **duck-typed**: it never imports ``dspy`` (it calls the methods
DSPy objects expose — a ``dspy.Module`` is callable and returns a
``Prediction``), so an optimized DSPy program drops in as a Vincio tool without
adding a dependency.
"""

from __future__ import annotations

from typing import Any

from ..core.types import Chunk
from ..retrieval.indexes import SearchFilter, SearchHit

__all__ = [
    "from_dspy_signature",
    "from_dspy_module",
    "register_dspy_module",
    "add_dspy_module",
    "from_dspy_retriever",
    "DSPyRetriever",
    "to_dspy_lm",
]


# -- signatures -----------------------------------------------------------------


def _field_desc(field: Any) -> str:
    extra = getattr(field, "json_schema_extra", None) or {}
    if isinstance(extra, dict):
        return str(extra.get("desc") or extra.get("description") or "")
    return ""


def from_dspy_signature(signature: Any) -> dict[str, Any]:
    """Summarize a DSPy ``Signature`` as ``{instructions, inputs, outputs}``.

    ``inputs`` / ``outputs`` map field name → description. Works on a signature
    class or instance via its ``input_fields`` / ``output_fields``.
    """
    inputs = {name: _field_desc(f) for name, f in (getattr(signature, "input_fields", {}) or {}).items()}
    outputs = {name: _field_desc(f) for name, f in (getattr(signature, "output_fields", {}) or {}).items()}
    instructions = getattr(signature, "instructions", "") or (getattr(signature, "__doc__", "") or "")
    return {"instructions": str(instructions).strip(), "inputs": inputs, "outputs": outputs}


def _input_schema_from_signature(signature: Any) -> dict[str, Any]:
    fields = getattr(signature, "input_fields", {}) or {}
    if not fields:
        return {}
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "description": _field_desc(field)}
            for name, field in fields.items()
        },
        "required": list(fields),
    }


# -- modules (as tools) ---------------------------------------------------------


def _prediction_to_dict(prediction: Any) -> Any:
    if hasattr(prediction, "toDict"):
        return prediction.toDict()
    if hasattr(prediction, "items"):
        return dict(prediction.items())
    if hasattr(prediction, "_store") and isinstance(prediction._store, dict):
        return dict(prediction._store)
    return prediction


def _module_handler(module: Any):
    name = getattr(module, "name", None) or type(module).__name__

    def handler(**kwargs: Any) -> Any:
        return _prediction_to_dict(module(**kwargs))

    handler.__name__ = name
    handler.__doc__ = getattr(module, "__doc__", "") or f"DSPy module {name}"
    return handler


def from_dspy_module(module: Any, *, name: str | None = None) -> dict[str, Any]:
    """Adapt a (compiled) DSPy module to a registration kwargs dict.

    A ``dspy.Module`` is callable and returns a ``Prediction``; the handler
    returns the prediction's fields as a dict. The input schema is derived from
    the module's ``signature`` when available.
    """
    signature = getattr(module, "signature", None)
    resolved = name or getattr(module, "name", None) or type(module).__name__
    description = ""
    if signature is not None:
        summary = from_dspy_signature(signature)
        description = summary["instructions"]
    return {
        "name": resolved,
        "description": description,
        "input_schema": _input_schema_from_signature(signature) if signature is not None else {},
        "handler": _module_handler(module),
    }


def register_dspy_module(registry: Any, module: Any, **overrides: Any) -> str:
    """Register a DSPy module on a :class:`~vincio.tools.ToolRegistry`."""
    adapter = from_dspy_module(module, name=overrides.pop("name", None))
    registry.register(
        adapter["handler"],
        name=adapter["name"],
        description=overrides.pop("description", adapter["description"]),
        input_schema=overrides.pop("input_schema", adapter["input_schema"]) or None,
        **overrides,
    )
    return adapter["name"]


def add_dspy_module(app: Any, module: Any, *, side_effects: str = "pure", **overrides: Any) -> Any:
    """Register *and enable* a DSPy module as a tool on a :class:`ContextApp`.

    Defaults to ``side_effects="pure"`` — a DSPy program is a deterministic
    transformation over its inputs (it may call an LM, but performs no external
    writes).
    """
    name = register_dspy_module(app.tool_registry, module, **overrides)
    return app.add_tool(name, side_effects=side_effects)


# -- retrievers -----------------------------------------------------------------


def _passage_text(passage: Any) -> str:
    for attr in ("long_text", "text", "content"):
        value = getattr(passage, attr, None)
        if value:
            return str(value)
    if isinstance(passage, dict):
        return str(passage.get("long_text") or passage.get("text") or passage.get("content") or "")
    return str(passage)


class DSPyRetriever:
    """Wrap a DSPy retrieval model (``rm(query, k=...)`` → passages) as a
    read-only Vincio index."""

    name = "dspy"

    def __init__(self, rm: Any, *, top_k: int = 10) -> None:
        self.rm = rm
        self.top_k = top_k

    def __len__(self) -> int:
        return 0

    def _run(self, query: str, k: int) -> list[Any]:
        try:
            result = self.rm(query, k=k)
        except TypeError:
            result = self.rm(query)
        # dspy.Retrieve returns a Prediction with a `.passages` list.
        passages = getattr(result, "passages", None)
        return list(passages if passages is not None else result or [])

    async def add(self, chunks: list[Chunk]) -> None:  # read-only adapter
        return None

    async def delete(self, chunk_ids: list[str]) -> int:
        return 0

    async def search(
        self, query: str, *, top_k: int = 10, where: SearchFilter | None = None
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for index, passage in enumerate(self._run(query, top_k)):
            chunk = Chunk(document_id="dspy", text=_passage_text(passage), index=index)
            if where is not None and not where(chunk):
                continue
            score = getattr(passage, "score", None)
            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=float(score) if score is not None else 1.0 / (index + 1),
                    source=self.name,
                )
            )
        return hits[:top_k]


def from_dspy_retriever(rm: Any, *, top_k: int = 10) -> DSPyRetriever:
    return DSPyRetriever(rm, top_k=top_k)


# -- export (Vincio -> DSPy) ----------------------------------------------------


class _VincioDSPyLM:
    """A duck-typed DSPy LM backed by a Vincio provider.

    DSPy calls an LM as ``lm(prompt=...)`` or ``lm(messages=[...])`` and expects
    a list of completion strings. This adapter is intentionally import-free so it
    works across DSPy versions; register it with ``dspy.configure(lm=...)``.
    """

    def __init__(self, provider: Any, *, model: str, **defaults: Any) -> None:
        self.provider = provider
        self.model = model
        self.kwargs = defaults
        self.history: list[dict[str, Any]] = []

    def __call__(
        self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> list[str]:
        from ..core.types import Message, ModelRequest
        from ..providers.base import run_sync

        if messages:
            msgs = [Message(role=m.get("role", "user"), content=m.get("content", "")) for m in messages]
        else:
            msgs = [Message(role="user", content=prompt or "")]
        request = ModelRequest(model=self.model, messages=msgs)
        response = run_sync(self.provider.generate(request))
        self.history.append({"prompt": prompt, "messages": messages, "response": response.text})
        return [response.text]


def to_dspy_lm(provider_or_app: Any, *, model: str | None = None, **defaults: Any) -> _VincioDSPyLM:
    """Expose a Vincio provider (or app) as a DSPy-compatible language model."""
    provider = getattr(provider_or_app, "resolve_provider", None)
    if callable(provider):  # a ContextApp
        app = provider_or_app
        return _VincioDSPyLM(app.resolve_provider(), model=model or app.model, **defaults)
    if model is None:
        raise ValueError("to_dspy_lm requires model= when given a provider")
    return _VincioDSPyLM(provider_or_app, model=model, **defaults)
