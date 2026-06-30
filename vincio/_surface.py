"""Two-level public-surface consistency.

``vincio.__all__`` is the *frozen top-level contract* (the exact set SemVer is
applied to). Each public subpackage *also* declares its own ``__all__`` — its
return types, dataclasses, enums, and helpers that callers reach by deep import
(``from vincio.evals import ...``). Historically those two levels drifted: a
subpackage exported far more than the top level re-exported, and that surface was
real but **undeclared as such**, with no guard against a name in an ``__all__``
that resolved to nothing (dead surface that reads as supported API).

This module reconciles the two levels and is the engine behind the
surface-consistency gate (``tests/test_surface_consistency.py``). It enumerates
every public subpackage's ``__all__``, classifies each symbol as ``TOP``
(re-exported in ``vincio.__all__`` — the *same object*), ``DUP`` (the name also
exists at the top level but as a *different* object: an intentional collision,
e.g. the tabular :class:`vincio.data.Dataset` beside the eval
:class:`vincio.Dataset`), or ``SUB`` (subpackage-only public), and proves two
invariants:

* **resolvable** — every name in a subpackage ``__all__`` is a live attribute of
  that module (a dead/renamed export fails the build), with no duplicate or
  malformed entries;
* **frozen** — the whole classified surface matches the committed manifest
  (``docs/reference/subpackage-surface.txt``), so any change to a subpackage
  ``__all__`` — a new symbol, a removed one, or a ``TOP``↔``DUP``↔``SUB``
  reclassification (a re-export that started or stopped shadowing the top level) —
  is a deliberate, reviewed edit, exactly as
  ``docs/reference/public-surface.txt`` freezes the top level.

The manifest is the standing *declaration* of the subpackage-only public surface
that was previously undeclared. Regenerate it with
``python -m vincio._surface --freeze`` and review the diff.
"""

from __future__ import annotations

import importlib
import os.path
import pkgutil

__all__ = [
    "public_subpackages",
    "subpackage_surface",
    "surface_rows",
    "surface_problems",
    "render_surface",
    "load_surface",
]


def public_subpackages() -> list[str]:
    """Return the names of every public first-level subpackage/module, sorted.

    Underscore-prefixed modules (private tooling like :mod:`vincio._apiref` and
    this module) are excluded — the two-level contract covers the *public*
    surface only.
    """
    import vincio

    names = [
        info.name
        for info in pkgutil.iter_modules(vincio.__path__)
        if not info.name.startswith("_")
    ]
    return sorted(names)


def _top_objects() -> dict[str, object]:
    """The live top-level public objects, keyed by name (``__version__`` aside)."""
    import vincio

    return {name: getattr(vincio, name) for name in vincio.__all__ if name != "__version__"}


def _classify(symbol: str, module: object, top: dict[str, object]) -> str:
    """Classify one subpackage symbol against the frozen top-level surface.

    ``TOP`` — re-exported at the top level (the *same object* as
    ``vincio.<symbol>``). ``DUP`` — the name also exists at the top level but
    resolves to a *different* object: an intentional name collision (e.g. the
    tabular :class:`vincio.data.Dataset` beside the eval :class:`vincio.Dataset`),
    so the subpackage symbol is subpackage-only public under a shared name.
    ``SUB`` — the name is not in the top-level surface at all.
    """
    if symbol not in top:
        return "SUB"
    if hasattr(module, symbol) and getattr(module, symbol) is top[symbol]:
        return "TOP"
    return "DUP"


def subpackage_surface() -> dict[str, list[tuple[str, str]]]:
    """Map each public subpackage to its classified ``__all__``.

    The value is ``[(symbol, tag), ...]`` sorted by symbol, where ``tag`` is one
    of ``"TOP"`` / ``"DUP"`` / ``"SUB"`` (see :func:`_classify`). Subpackages
    without an ``__all__`` are omitted.
    """
    top = _top_objects()
    out: dict[str, list[tuple[str, str]]] = {}
    for name in public_subpackages():
        module = importlib.import_module(f"vincio.{name}")
        exported = getattr(module, "__all__", None)
        if not exported:
            continue
        out[name] = [(symbol, _classify(symbol, module, top)) for symbol in sorted(exported)]
    return out


def surface_rows() -> list[tuple[str, str, str]]:
    """Flatten the surface to ``(subpackage, tag, symbol)`` rows, sorted.

    The deterministic ordering (by subpackage, then symbol) is the manifest's
    canonical line order, so a single reclassification or rename is a one-line
    diff.
    """
    rows: list[tuple[str, str, str]] = []
    for subpackage, symbols in subpackage_surface().items():
        for symbol, tag in symbols:
            rows.append((subpackage, tag, symbol))
    rows.sort(key=lambda row: (row[0], row[2]))
    return rows


def _module_problems(modname: str, exported: object, attrs: object) -> list[str]:
    """Invariant violations for one module's ``__all__`` (pure, injectable).

    ``exported`` is the module's ``__all__`` value and ``attrs`` is the object
    whose attributes back it (the module itself in production; a stand-in in
    tests). Returns one message per violation: a malformed ``__all__`` (not a
    list/tuple, or a non-string entry), a duplicate entry, or a name that resolves
    to no attribute (dead surface). An intentional name collision (``DUP``) is
    *not* a violation — it is classified and frozen in the manifest, so a *new*
    one is a reviewed diff rather than a silent shadow.
    """
    problems: list[str] = []
    if not isinstance(exported, (list, tuple)):
        problems.append(f"vincio.{modname}.__all__ is not a list/tuple")
        return problems
    seen: set[str] = set()
    for symbol in exported:
        if not isinstance(symbol, str):
            problems.append(f"vincio.{modname}.__all__ contains a non-string entry {symbol!r}")
            continue
        if symbol in seen:
            problems.append(f"vincio.{modname}.__all__ lists {symbol!r} more than once")
            continue
        seen.add(symbol)
        if not hasattr(attrs, symbol):
            problems.append(
                f"vincio.{modname}.__all__ exports {symbol!r} but the module has no such "
                f"attribute (dead surface)"
            )
    return problems


def surface_problems() -> list[str]:
    """Return every two-level surface violation (empty == consistent).

    This is the always-on integrity half of the gate: it never silently passes a
    dead or malformed export, independent of whether the frozen manifest is
    current.
    """
    problems: list[str] = []
    for name in public_subpackages():
        module = importlib.import_module(f"vincio.{name}")
        exported = getattr(module, "__all__", None)
        if exported is None:
            continue
        problems.extend(_module_problems(name, exported, module))
    return problems


# --- Frozen subpackage-surface manifest ------------------------------------

_SURFACE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "docs",
    "reference",
    "subpackage-surface.txt",
)

_SURFACE_HEADER = (
    "# Vincio two-level public surface.\n"
    "# Every public subpackage's __all__, one symbol per line, sorted by\n"
    "# (subpackage, symbol), as: <subpackage> <tag> <symbol>. The tag classifies\n"
    "# the symbol against the frozen top-level surface (docs/reference/public-\n"
    "# surface.txt): TOP = re-exported in vincio.__all__ (the same object);\n"
    "# DUP = the name also exists at the top level but as a different object (an\n"
    "# intentional collision, e.g. the tabular data.Dataset beside the eval\n"
    "# Dataset) so the subpackage symbol is reachable only by deep import;\n"
    "# SUB = subpackage-only public. Re-freeze deliberately: regenerate with\n"
    "# `python -m vincio._surface --freeze` and review the diff. Guarded by\n"
    "# tests/test_surface_consistency.py.\n"
)


def render_surface() -> str:
    """Render the frozen subpackage-surface manifest from the live ``__all__``s."""
    lines = [f"{subpackage} {tag} {symbol}" for subpackage, tag, symbol in surface_rows()]
    return _SURFACE_HEADER + "\n".join(lines) + "\n"


def load_surface(path: str | None = None) -> str:
    """Read the committed subpackage-surface manifest verbatim."""
    target = path or _SURFACE_FILE
    with open(target, encoding="utf-8") as handle:
        return handle.read()


def _freeze(path: str | None = None) -> None:  # pragma: no cover - dev tool
    target = path or _SURFACE_FILE
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(render_surface())


if __name__ == "__main__":  # pragma: no cover - dev tool
    import sys

    if "--freeze" in sys.argv[1:]:
        _freeze()
        rows = surface_rows()
        subs = {row[0] for row in rows}
        print(f"froze {len(rows)} symbols across {len(subs)} subpackages → {_SURFACE_FILE}")
    else:
        problems = surface_problems()
        if problems:
            print("\n".join(problems))
            sys.exit(1)
        print("subpackage surface consistent")
