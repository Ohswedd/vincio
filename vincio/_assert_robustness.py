"""``-O`` robustness: every shipped ``assert`` is a reviewed, marked invariant.

Python strips every ``assert`` statement under ``python -O`` (and ``-OO``). An
``assert`` that carries real control-flow weight — narrowing a value the code then
dereferences, or checking a precondition a public operation depends on — therefore
*vanishes* in an optimized deployment, turning a caught invariant into an opaque
downstream ``TypeError`` / ``AttributeError`` far from its cause. The hardening
line's 6.5 phase replaces each such *load-bearing* ``assert`` with an explicit guard
that raises the appropriate :class:`~vincio.core.errors.VincioError`, so an optimized
run fails loudly and correctly; a genuine *never-happens* invariant — a value
guaranteed non-``None`` by an adjacent caller guard, a model validator, or the
immediately-preceding assignment — stays an ``assert`` but is **documented and
marked**.

This module is the *static* half: the lint behind ``tests/test_assert_robustness.py``
and the ``hygiene`` VincioBench family, built on exactly the scan idiom
:mod:`vincio._observable_failure` uses for the broad-except contract and
:mod:`vincio._error_contract` uses for the error contract. It scans every **public**
module under ``vincio/`` for an ``assert`` statement and flags any that does **not**
carry a justifying ``# noqa: S101`` on its line — ``S101`` being the standard
"use of ``assert``" code, the same inline-marker convention the codebase uses with
``# noqa: BLE001`` for a reviewed broad ``except``. Marking an ``assert`` is the
reviewer's affirmation that it is a genuine never-happens invariant kept for
documentation and a cheap defensive check, not a load-bearing test that should be a
guard.

Like the observable-failure gate, there is **no frozen manifest**: the inline
``# noqa: S101`` is the per-site accepted marker, so the check is *always-on with
zero tolerance* — the live tree carries no unmarked ``assert``, and a new one fails
the build the moment it lands (convert it to a guard that raises a ``VincioError``,
or mark it if it is a real invariant). The scan covers public *and* private functions
of a public module (a stripped ``assert`` in a ``_helper`` breaks just as silently
under ``-O``), and naturally ignores an ``assert`` that appears only inside a
docstring example, since that is a string literal, not an ``ast.Assert`` node.

Run ``python -m vincio._assert_robustness`` to reproduce the check offline.
"""

from __future__ import annotations

import ast
import os.path

__all__ = [
    "ASSERT_NOQA_CODE",
    "public_modules",
    "asserts_in_source",
    "unmarked_asserts_in_source",
    "unmarked_asserts",
    "unmarked_assert_count",
    "marked_assert_count",
]

# The inline marker that accepts an ``assert`` as a reviewed never-happens invariant
# (the standard flake8-bandit "use of assert" code), mirroring the ``BLE001`` marker
# the observable-failure gate uses for a reviewed broad ``except``.
ASSERT_NOQA_CODE = "S101"

_PACKAGE_DIR = os.path.dirname(__file__)


def _is_private_component(name: str) -> bool:
    """A path/identifier component is private if underscore-prefixed but not a dunder."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _module_name(rel_path: str) -> str:
    """Dotted module name for a file path relative to the package's parent."""
    parts = rel_path[: -len(".py")].split(os.sep)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _public_module_paths() -> list[tuple[str, str]]:
    """``(module_name, abs_path)`` for every public module under the package, sorted.

    A module is *public* when no component of its path is private (underscore-prefixed
    and not a dunder), so private tooling like this module, :mod:`vincio._surface`,
    :mod:`vincio._observable_failure`, and ``vincio/security/_ed25519.py`` is excluded
    — matching the scope of :mod:`vincio._error_contract` and
    :mod:`vincio._observable_failure`.
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
            if any(_is_private_component(stem) for stem in stems):
                continue
            out.append((_module_name(rel), abs_path))
    out.sort()
    return out


def public_modules() -> list[str]:
    """Return the dotted name of every public module under the package, sorted."""
    return [name for name, _path in _public_module_paths()]


def _line_has_noqa(line: str) -> bool:
    """Whether a source line carries the justifying ``# noqa: S101`` (or a bare ``# noqa``)."""
    if "noqa" not in line:
        return False
    return ASSERT_NOQA_CODE in line or line.rstrip().endswith("# noqa")


def _qualname(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """The enclosing function/class chain for a node, or ``"<module>"`` at module scope."""
    chain: list[str] = []
    cursor: ast.AST | None = parents.get(node)
    while cursor is not None:
        if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            chain.append(cursor.name)
        cursor = parents.get(cursor)
    return ".".join(reversed(chain)) if chain else "<module>"


def asserts_in_source(source: str) -> list[tuple[str, int, bool]]:
    """Every ``assert`` statement in one module's source as ``(qualname, lineno, marked)``.

    Pure and injectable (mirrors
    :func:`vincio._observable_failure.silent_swallows_in_source`): parses ``source``
    and returns one sorted row per ``ast.Assert`` — its enclosing qualname, its line,
    and whether that line carries the justifying ``# noqa: S101`` marker. An
    ``assert`` that appears only inside a docstring (a string literal) is not an
    ``ast.Assert`` node and is correctly ignored.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def line_at(lineno: int) -> str:
        return lines[lineno - 1] if 0 <= lineno - 1 < len(lines) else ""

    rows: list[tuple[str, int, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            marked = _line_has_noqa(line_at(node.lineno))
            rows.append((_qualname(node, parents), node.lineno, marked))
    return sorted(rows)


def unmarked_asserts_in_source(source: str) -> list[tuple[str, int]]:
    """The ``(qualname, lineno)`` of every ``assert`` in ``source`` lacking ``# noqa: S101``.

    An empty list means every ``assert`` in the source is either a guard already
    (there is none) or a marked, reviewed never-happens invariant.
    """
    return [(qualname, lineno) for qualname, lineno, marked in asserts_in_source(source) if not marked]


def unmarked_asserts() -> list[str]:
    """Every unmarked ``assert`` tree-wide as a sorted ``"<module>:<lineno> ..."`` message.

    The always-on gate: an empty list means no public module carries an ``assert``
    that would silently vanish under ``python -O`` without a reviewer having marked it
    a genuine never-happens invariant. Each message names the module, line, and
    enclosing qualname, so a violation is a one-line locate-and-fix.
    """
    out: list[str] = []
    for module, path in _public_module_paths():
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        for qualname, lineno in unmarked_asserts_in_source(source):
            out.append(
                f"{module}:{lineno} ({qualname}) is a bare `assert`, stripped under "
                f"`python -O`; convert it to a guard that raises a VincioError, or mark "
                f"a genuine never-happens invariant with `# noqa: {ASSERT_NOQA_CODE}`"
            )
    return sorted(out)


def unmarked_assert_count() -> int:
    """The number of unmarked ``assert``s tree-wide (``0`` when the gate is clean)."""
    return len(unmarked_asserts())


def marked_assert_count() -> int:
    """The number of ``assert``s tree-wide carrying the ``# noqa: S101`` invariant marker."""
    total = 0
    for _module, path in _public_module_paths():
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        total += sum(1 for _qualname, _lineno, marked in asserts_in_source(source) if marked)
    return total


if __name__ == "__main__":  # pragma: no cover - dev tool
    import sys

    problems = unmarked_asserts()
    if problems:
        print("unmarked `assert`s (convert to a guard that raises a VincioError, or add a")
        print(f"justifying `# noqa: {ASSERT_NOQA_CODE}` for a genuine never-happens invariant):")
        print("\n".join(problems))
        sys.exit(1)
    print("assert robustness conformant")
