"""The single source of truth for *lowering* task config to builder calls.

Both front doors — the task-shaped constructors in :mod:`._facades` and the
fluent :class:`~vincio.tasks._flow.Flow` — turn a small, ergonomic description of
a job into a configured :class:`~vincio.core.app.ContextApp`. The guarantee that a
one-liner **lowers to the exact same governed run packet** as the verbose builder
form only holds if *both* front doors emit the *same* builder calls. That is what
this module is: each capability's lowering is written **once** here, and the
facades and the flow both call it — so the byte-identical guarantee holds by
construction, not by parallel maintenance across two files.

Every function here is a thin, deliberately boring wrapper over a *public*
``ContextApp`` builder call (``add_source`` / ``set_policy`` / ``configure`` /
``add_evaluator`` / ``add_tool`` / ``output_schema``). The one place that reaches a
non-public method — building the output contract — lives in
:func:`apply_output_schema`, so nothing else has to.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.app import ContextApp
    from ..core.config import VincioConfig
    from ..providers.base import ModelProvider

__all__ = [
    "resolve_app",
    "apply_output_schema",
    "apply_persona",
    "apply_grounding",
    "apply_require_citations",
    "add_evaluators",
    "add_source",
    "add_sources",
    "register_tools",
]


# --------------------------------------------------------------------------- #
# App resolution & the output contract — the only reach past the public API.
# --------------------------------------------------------------------------- #
def resolve_app(
    app: ContextApp | None,
    *,
    name: str,
    provider: ModelProvider | str | None,
    model: str | None,
    output_schema: Any | None = None,
    config: VincioConfig | str | None = None,
) -> ContextApp:
    """Return the app to configure: the supplied escape-hatch app, or a fresh one.

    When ``app`` is supplied it is used as-is (the escape hatch — provider / model
    / config are taken to already be set), with only an explicit ``output_schema``
    layered on so a typed task can re-shape an existing app's contract.
    """
    from ..core.app import ContextApp

    if app is not None:
        if output_schema is not None:
            apply_output_schema(app, output_schema)
        return app
    return ContextApp(
        name=name,
        provider=provider,
        model=model,
        output_schema=output_schema,
        config=config,
    )


def apply_output_schema(app: ContextApp, schema: Any) -> None:
    """Set the typed output contract from ``schema`` (the one private-API reach).

    Mirrors ``ContextApp(output_schema=...)`` exactly so the contract is
    byte-for-byte what the constructor would build.
    """
    app.output_contract = app._build_contract(schema)


# --------------------------------------------------------------------------- #
# Persona, grounding, citations, evaluators — public-builder one-liners.
# --------------------------------------------------------------------------- #
def apply_persona(
    app: ContextApp,
    *,
    role: str | None,
    objective: str | None,
    rules: Sequence[str] | None,
) -> None:
    """Apply an optional role / objective / rules persona via ``app.configure``."""
    if role is None and objective is None and rules is None:
        return
    app.configure(
        role=role,
        objective=objective,
        rules=list(rules) if rules is not None else None,
    )


def apply_grounding(app: ContextApp, only_from_sources: bool = True) -> None:
    """Answer only from retrieved sources, with citations (the grounding policy)."""
    app.set_policy("answer_only_from_sources", bool(only_from_sources))


def apply_require_citations(app: ContextApp, value: bool) -> None:
    """Require every answer to carry citations (the citation policy)."""
    app.set_policy("require_citations", bool(value))


def add_evaluators(app: ContextApp, evaluators: Sequence[str | Callable[..., Any]]) -> None:
    """Register each evaluator on the app (``ContextApp.add_evaluator``)."""
    for evaluator in evaluators:
        app.add_evaluator(evaluator)


# --------------------------------------------------------------------------- #
# Knowledge sources.
# --------------------------------------------------------------------------- #
def add_source(
    app: ContextApp,
    name: str,
    *,
    path: str | None = None,
    documents: Sequence[Any] | None = None,
    connector: Any | None = None,
    loader: str | None = None,
    chunking: str | None = None,
    retrieval: str = "hybrid",
) -> None:
    """Register exactly one knowledge source (one ``ContextApp.add_source`` call)."""
    app.add_source(
        name,
        path=path,
        documents=list(documents) if documents is not None else None,
        connector=connector,
        loader=loader,
        chunking=chunking,
        retrieval=retrieval,
    )


def _add_one_source(
    app: ContextApp, name: str, spec: Any, *, chunking: str | None, retrieval: str
) -> None:
    if isinstance(spec, (str, Path)):
        app.add_source(name, path=str(spec), chunking=chunking, retrieval=retrieval)
    elif isinstance(spec, Mapping):
        kwargs: dict[str, Any] = dict(spec)
        kwargs.setdefault("chunking", chunking)
        kwargs.setdefault("retrieval", retrieval)
        app.add_source(name, **kwargs)
    elif isinstance(spec, Sequence):
        app.add_source(name, documents=list(spec), chunking=chunking, retrieval=retrieval)
    else:
        raise TypeError(
            f"source {name!r} must be a path, a list of documents, or an add_source "
            f"kwargs mapping; got {type(spec).__name__}"
        )


def add_sources(app: ContextApp, sources: Any, *, chunking: str | None, retrieval: str) -> None:
    """Register knowledge sources from the ergonomic ``sources`` argument.

    Accepts a single path, a list of paths and/or in-memory documents, or a
    ``name -> spec`` mapping (spec = a path, a document list, or add_source
    kwargs). A single path indexes one ``"docs"`` source — exactly the canonical
    ``add_source("docs", path=..., chunking=..., retrieval=...)`` call.
    """
    if sources is None:
        return
    if isinstance(sources, (str, Path)):
        app.add_source("docs", path=str(sources), chunking=chunking, retrieval=retrieval)
        return
    if isinstance(sources, Mapping):
        for src_name, spec in sources.items():
            _add_one_source(app, str(src_name), spec, chunking=chunking, retrieval=retrieval)
        return
    if isinstance(sources, Sequence):
        paths = [s for s in sources if isinstance(s, (str, Path))]
        documents = [s for s in sources if not isinstance(s, (str, Path))]
        if documents:
            app.add_source("docs", documents=documents, chunking=chunking, retrieval=retrieval)
        single = len(paths) == 1 and not documents
        for index, path in enumerate(paths):
            src_name = "docs" if single else f"docs_{index}"
            app.add_source(src_name, path=str(path), chunking=chunking, retrieval=retrieval)
        return
    raise TypeError(
        "sources must be a path, a list of paths/documents, or a name->spec mapping; "
        f"got {type(sources).__name__}"
    )


# --------------------------------------------------------------------------- #
# Tools.
# --------------------------------------------------------------------------- #
def register_tools(
    app: ContextApp,
    tools: Sequence[str | Callable[..., Any]],
    writes: Sequence[str | Callable[..., Any]],
) -> None:
    """Enable read tools and approval-gated write tools (least privilege)."""
    for tool in tools:
        app.add_tool(tool)
    for tool in writes:
        app.add_tool(tool, approval_required=True, side_effects="write")
