"""Error-contract conformance: every public raise is a ``VincioError``.

Vincio's stated contract is that **every error it raises derives from**
:class:`~vincio.core.errors.VincioError`, so an application catches the whole
family with one ``except VincioError`` and branches on the stable ``.code``
(:mod:`vincio.core.error_catalog`). Most built-in raises inside the package are
legitimate *internal input-validation* — a numeric helper that needs at least
two points, an abstract base method placeholder, the ``AttributeError`` a
``__getattr__`` must raise to make ``hasattr`` work — and those stay as they are.
The violations the hardening line pays down are the ones that **leaked a bare
built-in off a public entry point**, so a caller saw a raw ``ValueError`` /
``KeyError`` / ``NotImplementedError`` instead of a typed, catalog-coded
``VincioError``.

This module is the *mechanical* half of that contract — the lint behind
``tests/test_error_contract.py`` and the ``hygiene`` VincioBench family, built on
exactly the freeze-and-gate idiom :mod:`vincio._surface` uses for the two-level
public surface. It statically scans every **public** module under ``vincio/``
(no underscore-prefixed, non-dunder path component) — plus, deliberately, the
private ``vincio/core/_app_*.py`` modules holding the ``_*Verbs`` mixins that
``ContextApp`` composes, so the decomposed ``app.*`` verb surface stays guarded —
for a ``raise`` of a bare built-in exception that sits on a **public entry
point** (every enclosing function and class is public — non-underscore, dunders
like ``__init__`` count; a ``_*Verbs`` mixin class counts as public inside those
whitelisted modules), and enforces two invariants:

* **app-verb purity (always-on, no manifest)** — no public method of the
  user-facing :class:`~vincio.core.app.ContextApp` facade (the ``app.*`` verb
  surface) raises a bare built-in. This is the canonical public entry point and
  is held to **zero** off-contract raises, the way
  :func:`vincio._surface.surface_problems` is the always-on integrity half of the
  surface gate. It bites automatically — no allowlist to keep current.
* **frozen baseline (drift-gated)** — the full classified set of public built-in
  raises matches the committed manifest (``docs/reference/error-contract.txt``),
  so a *new* one is a deliberate, reviewed edit: either convert it to a
  ``VincioError`` (it drops out of the baseline) or, when it is genuinely
  internal input-validation, regenerate the manifest and review the diff. The
  baseline is the standing, honest inventory of the accepted built-in raises that
  were previously undeclared.

Regenerate the manifest with ``python -m vincio._error_contract --freeze`` and
review the diff.
"""

from __future__ import annotations

import ast
import builtins
import os.path

__all__ = [
    "BUILTIN_EXCEPTION_NAMES",
    "public_modules",
    "contract_raises_in_source",
    "contract_rows",
    "app_verb_violations",
    "render_manifest",
    "load_manifest",
]

# The exception classes the contract treats as *off-contract* when raised: every
# built-in that derives from ``BaseException``. A ``raise`` whose callable name is
# one of these is a bare built-in; a ``VincioError`` subclass (or any other name)
# is on-contract. Resolved from ``builtins`` so the set tracks the running Python.
BUILTIN_EXCEPTION_NAMES: frozenset[str] = frozenset(
    name
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type)
    and issubclass(getattr(builtins, name), BaseException)
)

# The package root (the directory that holds this module).
_PACKAGE_DIR = os.path.dirname(__file__)
_PACKAGE_NAME = os.path.basename(_PACKAGE_DIR)


def _is_private_component(name: str) -> bool:
    """A path/identifier component is private if it is underscore-prefixed but not a dunder.

    ``_ed25519`` / ``_flow`` / ``_helper`` are private; ``__init__`` / ``__call__``
    / ``__post_init__`` are dunders and count as public reachable entry points.
    """
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _module_name(rel_path: str) -> str:
    """Dotted module name for a file path relative to the package's parent."""
    parts = rel_path[: -len(".py")].split(os.sep)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_app_mixin_module(module: str) -> bool:
    """Whether ``module`` is a ContextApp verb-mixin module (``vincio.core._app_*``).

    The standing-guard whitelist: ``ContextApp``'s verb surface is decomposed into
    private ``vincio/core/_app_*.py`` mixin modules (the ``_*Verbs`` classes the
    app composes). Those files would normally drop out of the public-module scan
    as underscore-prefixed, silently un-guarding the ``app.*`` verb bodies — so
    they are deliberately kept in scope, here and in
    :mod:`vincio._observable_failure` / :mod:`vincio._assert_robustness`.
    """
    return module.startswith(f"{_PACKAGE_NAME}.core._app_")


def _public_module_paths() -> list[tuple[str, str]]:
    """``(module_name, abs_path)`` for every public module under the package, sorted.

    A module is *public* when no component of its path — package dirs and the file
    stem — is private (underscore-prefixed and not a dunder). Private tooling like
    :mod:`vincio._surface`, this module, and ``vincio/security/_ed25519.py`` is
    excluded: the contract covers the public surface only. One deliberate
    exception: the ContextApp verb mixins (:func:`_is_app_mixin_module`) stay in
    scope so the decomposed ``app.*`` surface remains guarded.
    """
    parent = os.path.dirname(_PACKAGE_DIR)
    out: list[tuple[str, str]] = []
    for root, _dirs, files in os.walk(_PACKAGE_DIR):
        for filename in files:
            if not filename.endswith(".py"):
                continue
            abs_path = os.path.join(root, filename)
            rel = os.path.relpath(abs_path, parent)
            stems = [
                part[: -len(".py")] if part.endswith(".py") else part
                for part in rel.split(os.sep)
            ]
            name = _module_name(rel)
            if any(_is_private_component(stem) for stem in stems) and not _is_app_mixin_module(
                name
            ):
                continue
            out.append((name, abs_path))
    out.sort()
    return out


def public_modules() -> list[str]:
    """Return the dotted name of every public module under the package, sorted."""
    return [name for name, _path in _public_module_paths()]


def _raised_builtin(node: ast.Raise) -> str | None:
    """The bare-built-in exception name a ``raise`` raises, or ``None``.

    ``raise ValueError(...)`` / ``raise KeyError`` → the name; ``raise`` (bare
    re-raise, ``node.exc is None``), ``raise SomeVincioError(...)``, or a
    non-:class:`ast.Name` callable → ``None`` (on-contract or unclassifiable).
    """
    exc = node.exc
    if exc is None:
        return None
    callee = exc.func if isinstance(exc, ast.Call) else exc
    if isinstance(callee, ast.Name) and callee.id in BUILTIN_EXCEPTION_NAMES:
        return callee.id
    return None


def contract_raises_in_source(source: str, *, app_mixin: bool = False) -> list[tuple[str, str]]:
    """Bare-built-in raises on a *public entry point* in one module's source.

    Pure and injectable (mirrors :func:`vincio._surface._module_problems`): parses
    ``source`` and returns sorted ``(qualname, exception)`` pairs for every
    ``raise <Builtin>`` whose enclosing chain is entirely public — there is at
    least one enclosing function and **no** enclosing function or class is private
    (underscore-prefixed, dunders excepted). A raise inside a nested private
    helper (``_step``) of a public method is *encapsulated*, not surface, so it is
    not reported. Duplicate ``(qualname, exception)`` pairs (the same function
    raising the same built-in twice) collapse to one row, so the baseline tracks
    *where* a built-in can leak, stably across refactors that move lines.

    ``app_mixin=True`` (set for the whitelisted ``vincio/core/_app_*.py`` modules,
    :func:`_is_app_mixin_module`) makes a private ``_*Verbs`` mixin class count as
    public: its methods are ``ContextApp`` verbs, so a raise inside one is surface,
    not encapsulated.
    """
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    rows: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        exception = _raised_builtin(node)
        if exception is None:
            continue
        chain: list[str] = []
        has_function = False
        private = False
        cursor: ast.AST | None = parents.get(node)
        while cursor is not None:
            if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                has_function = True
                chain.append(cursor.name)
                if _is_private_component(cursor.name):
                    private = True
            elif isinstance(cursor, ast.ClassDef):
                chain.append(cursor.name)
                if _is_private_component(cursor.name) and not (
                    app_mixin and cursor.name.endswith("Verbs")
                ):
                    private = True
            cursor = parents.get(cursor)
        if not has_function or private:
            continue
        qualname = ".".join(reversed(chain))
        rows.add((qualname, exception))
    return sorted(rows)


def contract_rows() -> list[tuple[str, str, str]]:
    """Every public bare-built-in raise tree-wide as ``(module, qualname, exception)``.

    Sorted by ``(module, qualname, exception)`` — the manifest's canonical line
    order, so a single conversion or a new raise is a one-line diff. This is the
    full classified baseline the committed manifest freezes.
    """
    rows: list[tuple[str, str, str]] = []
    for module, path in _public_module_paths():
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        for qualname, exception in contract_raises_in_source(
            source, app_mixin=_is_app_mixin_module(module)
        ):
            rows.append((module, qualname, exception))
    rows.sort()
    return rows


def app_verb_violations(rows: list[tuple[str, str, str]] | None = None) -> list[str]:
    """Off-contract raises on the :class:`~vincio.core.app.ContextApp` verb surface.

    The always-on integrity half of the gate, independent of the frozen baseline:
    every public method of ``ContextApp`` — the user-facing ``app.*`` entry points,
    whether defined on ``ContextApp`` itself or on one of the ``_*Verbs`` mixin
    classes it composes from the whitelisted ``vincio/core/_app_*.py`` modules —
    must raise only ``VincioError`` subclasses. Returns one
    ``"<module>.<qualname> raises <Exception>"`` message per violation; an empty
    list means the verb surface is clean. It needs no allowlist, so it bites on a
    *new* leak the moment it lands.

    ``rows`` defaults to the live :func:`contract_rows`; pass an explicit set to
    exercise the classification in isolation (pure and injectable, the way
    :func:`contract_raises_in_source` is).
    """
    if rows is None:
        rows = contract_rows()
    problems = [
        f"{module}.{qualname} raises {exception}"
        for module, qualname, exception in rows
        if qualname.split(".", 1)[0] == "ContextApp"
        or (_is_app_mixin_module(module) and qualname.split(".", 1)[0].endswith("Verbs"))
    ]
    return sorted(problems)


# --- Frozen error-contract manifest ----------------------------------------

_MANIFEST_FILE = os.path.join(
    os.path.dirname(_PACKAGE_DIR),
    "docs",
    "reference",
    "error-contract.txt",
)

_MANIFEST_HEADER = (
    "# Vincio error-contract baseline.\n"
    "# Every bare built-in exception raised on a PUBLIC entry point under vincio/,\n"
    "# one per line, sorted by (module, qualname, exception), as:\n"
    "#   <module> <qualname> <Exception>\n"
    "# A public entry point is a raise whose every enclosing function and class is\n"
    "# public (non-underscore; dunders like __init__ count). These are the ACCEPTED,\n"
    "# reviewed built-in raises: internal input-validation (a numeric helper that\n"
    "# needs two points), abstract-base placeholders, and the AttributeError a\n"
    "# __getattr__ must raise so hasattr() works. They stay built-ins on purpose.\n"
    "#\n"
    "# The contract is that every error Vincio raises derives from VincioError. A\n"
    "# NEW public built-in raise must therefore be either converted to a VincioError\n"
    "# (it drops out of this baseline) or, when it is genuinely internal input-\n"
    "# validation, added here deliberately: regenerate with\n"
    "# `python -m vincio._error_contract --freeze` and review the diff. The ContextApp\n"
    "# (app.* verb) surface carries ZERO entries and is held there by an always-on\n"
    "# gate. Guarded by tests/test_error_contract.py and the hygiene VincioBench family.\n"
)


def render_manifest() -> str:
    """Render the frozen error-contract manifest from the live source tree."""
    lines = [f"{module} {qualname} {exception}" for module, qualname, exception in contract_rows()]
    return _MANIFEST_HEADER + "\n".join(lines) + "\n"


def load_manifest(path: str | None = None) -> str:
    """Read the committed error-contract manifest verbatim."""
    target = path or _MANIFEST_FILE
    with open(target, encoding="utf-8") as handle:
        return handle.read()


def _freeze(path: str | None = None) -> None:  # pragma: no cover - dev tool
    target = path or _MANIFEST_FILE
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(render_manifest())


if __name__ == "__main__":  # pragma: no cover - dev tool
    import sys

    if "--freeze" in sys.argv[1:]:
        _freeze()
        rows = contract_rows()
        modules = {row[0] for row in rows}
        print(
            f"froze {len(rows)} public built-in raises across {len(modules)} modules "
            f"→ {_MANIFEST_FILE}"
        )
    else:
        verb_problems = app_verb_violations()
        if verb_problems:
            print("ContextApp verb surface raises bare built-ins:")
            print("\n".join(verb_problems))
            sys.exit(1)
        committed = load_manifest()
        rendered = render_manifest()
        if committed != rendered:
            print(
                "error-contract baseline drifted from the frozen manifest; if "
                "intentional, regenerate with `python -m vincio._error_contract "
                "--freeze` and review the diff."
            )
            sys.exit(1)
        print("error contract conformant")
