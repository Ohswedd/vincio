"""Docstring-driven API reference for the frozen public surface.

Introspects ``vincio.__all__`` — the exact set SemVer applies to — and renders a
complete, alphabetized reference from each symbol's signature and docstring
summary. The curated narrative lives in ``docs/reference/api.md``; this module
generates the *exhaustive index* (``docs/reference/api-generated.md``) so the
two stay in sync mechanically, and exposes :func:`undocumented_symbols` for the
docstring-coverage gate that keeps every public symbol documented.
"""

from __future__ import annotations

import inspect
from typing import Any

__all__ = [
    "public_symbols",
    "docstring_summary",
    "symbol_kind",
    "symbol_signature",
    "undocumented_symbols",
    "render_api_index",
]

# Public names that are values, not introspectable callables/classes.
_DUNDER = {"__version__"}


def public_symbols() -> list[tuple[str, Any]]:
    """Return ``(name, object)`` for every public symbol, alphabetized."""
    import vincio

    return [
        (name, getattr(vincio, name))
        for name in sorted(vincio.__all__)
        if name not in _DUNDER
    ]


def docstring_summary(obj: Any) -> str:
    """Return the first paragraph of an object's docstring (one line)."""
    doc = inspect.getdoc(obj)
    if not doc:
        return ""
    summary: list[str] = []
    for line in doc.strip().splitlines():
        if not line.strip():
            break
        summary.append(line.strip())
    return " ".join(summary)


def symbol_kind(obj: Any) -> str:
    """Classify a public symbol as ``class``, ``function``, or ``data``."""
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj):
        return "function"
    if callable(obj) and not isinstance(obj, type):
        # Decorated callables / partials still read as functions to users.
        if inspect.isroutine(obj):
            return "function"
    return "data"


def symbol_signature(name: str, obj: Any) -> str:
    """Best-effort ``name(signature)`` string; falls back to the bare name."""
    target = obj
    if inspect.isclass(obj):
        init = getattr(obj, "__init__", None)
        if init is not None and init is not object.__init__:
            target = init
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return name
    params = [p for p in sig.parameters.values() if p.name != "self"]
    rendered = ", ".join(_render_param(p) for p in params)
    return f"{name}({rendered})"


def _render_param(param: inspect.Parameter) -> str:
    if param.kind is inspect.Parameter.VAR_POSITIONAL:
        return f"*{param.name}"
    if param.kind is inspect.Parameter.VAR_KEYWORD:
        return f"**{param.name}"
    if param.kind is inspect.Parameter.KEYWORD_ONLY:
        # Collapse keyword-only markers; the leading `*` is implied once.
        return f"{param.name}=…" if param.default is not inspect.Parameter.empty else param.name
    if param.default is not inspect.Parameter.empty:
        return f"{param.name}=…"
    return param.name


def undocumented_symbols() -> list[str]:
    """Return public symbol names whose docstring is missing or empty.

    This is the docstring-coverage gate: the list must be empty so no public
    symbol ships undocumented.
    """
    missing: list[str] = []
    for name, obj in public_symbols():
        if not docstring_summary(obj):
            missing.append(name)
    return missing


def render_api_index() -> str:
    """Render the exhaustive, docstring-driven public API index (Markdown)."""
    by_kind: dict[str, list[tuple[str, Any]]] = {"class": [], "function": [], "data": []}
    for name, obj in public_symbols():
        by_kind[symbol_kind(obj)].append((name, obj))

    total = sum(len(v) for v in by_kind.values())
    lines: list[str] = [
        "# Reference: public API index",
        "",
        "This page is generated from `vincio.__all__` — the exact set of names",
        "[Semantic Versioning](https://semver.org/spec/v2.0.0.html) applies to —",
        "with each symbol's signature and docstring summary. It is gated for",
        "docstring coverage: no public symbol ships undocumented. For the curated,",
        "grouped narrative see [api.md](api.md).",
        "",
        f"**{total}** public symbols.",
        "",
    ]
    headings = (
        ("class", "Classes"),
        ("function", "Functions"),
        ("data", "Values"),
    )
    for kind, heading in headings:
        entries = by_kind[kind]
        if not entries:
            continue
        lines.append(f"## {heading}")
        lines.append("")
        for name, obj in entries:
            if kind == "data":
                lines.append(f"### `{name}`")
            else:
                lines.append(f"### `{symbol_signature(name, obj)}`")
            lines.append("")
            summary = docstring_summary(obj)
            lines.append(summary if summary else "_(undocumented)_")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
