"""Shared static-scan helpers for ``vincio doctor`` and ``vincio migrate``.

Both tools statically match uses of renamed / deprecated public symbols. A
symbol can be reached through more shapes than a ``from vincio import old``
binding or a literal ``vincio.old`` attribute: ``import vincio.data`` followed
by ``vincio.data.old(...)``, ``import vincio.data as vd`` followed by
``vd.old(...)``, and ``from vincio import data`` followed by ``data.old(...)``
are all documented usage forms. These helpers resolve a local name (or a
dotted attribute chain) to the vincio module it denotes, so the scanners can
recognise every one of those shapes instead of only attribute access on the
bare name ``vincio``.

Everything here is static: user source is parsed with :mod:`ast`, never
imported. Submodule existence is answered from the installed package's own
file tree, not by importing anything.
"""

from __future__ import annotations

import ast
from functools import cache, lru_cache
from pathlib import Path

__all__ = ["is_vincio_module", "resolve_attr_module", "vincio_module_aliases"]


@lru_cache(maxsize=1)
def _package_root() -> Path:
    import vincio

    return Path(vincio.__file__).resolve().parent


@cache
def is_vincio_module(dotted: str) -> bool:
    """Whether ``dotted`` (``vincio`` or ``vincio.x[.y]``) names a real module.

    Answered from the installed package's file tree (``x/__init__.py`` or
    ``x.py``), so scanning never imports user code or extra vincio modules.
    """
    if dotted == "vincio":
        return True
    if not dotted.startswith("vincio."):
        return False
    parts = dotted.split(".")[1:]
    base = _package_root().joinpath(*parts)
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def vincio_module_aliases(tree: ast.AST) -> dict[str, str]:
    """Map each local name in *tree* to the vincio module it is bound to.

    Covers ``import vincio`` / ``import vincio as v`` (bind ``vincio`` / ``v``),
    ``import vincio.data`` (binds the top name ``vincio``; the dotted chain is
    resolved attribute-by-attribute), ``import vincio.data as vd`` (binds
    ``vd`` to ``vincio.data``), and ``from vincio[.sub] import mod [as m]``
    when ``mod`` is a submodule rather than a symbol. The bare name ``vincio``
    always denotes the package even with no import in the scanned file — a
    module object can arrive by re-export (``from myproject.compat import
    vincio``), and the pre-7.5 scanners matched ``vincio.X`` unconditionally.
    """
    aliases: dict[str, str] = {"vincio": "vincio"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name != "vincio" and not name.startswith("vincio."):
                    continue
                if alias.asname is not None:
                    aliases[alias.asname] = name
                else:
                    aliases["vincio"] = "vincio"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or (module != "vincio" and not module.startswith("vincio.")):
                continue
            for alias in node.names:
                candidate = f"{module}.{alias.name}"
                if is_vincio_module(candidate):
                    aliases[alias.asname or alias.name] = candidate
    return aliases


def resolve_attr_module(value: ast.expr, aliases: dict[str, str]) -> str | None:
    """The dotted vincio module *value* denotes, or ``None``.

    ``vd`` → ``vincio.data``; ``vincio.data`` → ``vincio.data``;
    ``vincio.evals.suite`` → ``vincio.evals.suite`` (chains resolve left to
    right, and every prefix must be a real vincio module).
    """
    if isinstance(value, ast.Name):
        return aliases.get(value.id)
    if isinstance(value, ast.Attribute):
        base = resolve_attr_module(value.value, aliases)
        if base is None:
            return None
        dotted = f"{base}.{value.attr}"
        return dotted if is_vincio_module(dotted) else None
    return None
