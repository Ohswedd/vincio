"""Observable failure: no broad ``except`` swallows its exception silently.

A best-effort fallback that catches a broad ``Exception`` and continues is correct
policy, but one that swallows the exception *silently* — no re-raise, no log, no
metric — hides a real bug inside itself (see :mod:`vincio.core.diagnostics` for the
runtime half of this contract). This module is the *static* half: the lint behind
``tests/test_observable_failure.py`` and the ``hygiene`` VincioBench family, built
on exactly the scan idiom :mod:`vincio._error_contract` and :mod:`vincio._surface`
use for the error contract and the two-level surface.

It scans every **public** module under ``vincio/`` — plus, deliberately, the
private ``vincio/core/_app_*.py`` ContextApp verb mixins, so the decomposed
``app.*`` surface stays guarded — for a *broad* exception handler
— ``except Exception`` / ``except BaseException`` / a bare ``except:`` / a tuple
containing one — and for ``contextlib.suppress(Exception)`` (which is *always*
silent). Such a handler is a **silent swallow** unless its body either:

* **re-raises** (any ``raise`` in the handler — a translated or conditional
  re-raise still surfaces the failure), or
* **records the failure observably** — a call to a logger method (``debug`` /
  ``info`` / ``warning`` / ``error`` / ``exception`` / ``critical`` / ``log`` /
  ``warn``), the event bus (``emit`` / ``emit_async`` / ``publish`` /
  ``add_event``), or :func:`vincio.core.diagnostics.note_suppressed`, or
* carries a justifying **``# noqa: BLE001``** on the ``except`` line — the same
  inline marker the codebase already uses for a reviewed, deliberately-silent
  swallow, with the reason in the trailing comment.

Unlike the error-contract baseline, there is **no frozen manifest**: the inline
``# noqa: BLE001`` is the per-site accepted marker, so the check is *always-on with
zero tolerance* — the live tree carries no unmarked silent swallow, and a new one
fails the build the moment it lands. The handler scan covers public *and* private
functions of a public module (a silent swallow in a ``_helper`` hides a bug just as
much), unlike the error contract, which is a public-entry-point surface concern.

Run ``python -m vincio._observable_failure`` to reproduce the check offline.
"""

from __future__ import annotations

import ast
import os.path

from ._guard_scope import is_app_mixin_module as _is_app_mixin_module

__all__ = [
    "BROAD_EXCEPTION_NAMES",
    "LOG_METHOD_NAMES",
    "public_modules",
    "silent_swallows_in_source",
    "silent_swallows",
    "silent_swallow_count",
]

# The exception names that make a handler *broad* (a "blind except"). A handler
# catching one of these, a bare ``except:``, or a tuple containing one is held to
# the observable-or-justified rule; a narrow ``except ValueError`` is not.
BROAD_EXCEPTION_NAMES: frozenset[str] = frozenset({"Exception", "BaseException"})

# Method names that count as observably recording a failure: the standard logging
# levels, plus the event-bus verbs the runtime already publishes through.
LOG_METHOD_NAMES: frozenset[str] = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
)
_EMIT_METHOD_NAMES: frozenset[str] = frozenset({"emit", "emit_async", "publish", "add_event"})
# The runtime helper that makes a fallback observable (a log + a counter).
_NOTE_FUNCTION_NAMES: frozenset[str] = frozenset({"note_suppressed"})

_PACKAGE_DIR = os.path.dirname(__file__)
_PACKAGE_NAME = os.path.basename(_PACKAGE_DIR)


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
    and ``vincio/security/_ed25519.py`` is excluded — matching the scope of
    :mod:`vincio._error_contract`, including its one deliberate exception: the
    ContextApp verb mixins (:func:`_is_app_mixin_module`) stay in scope.
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


def _is_broad_handler(handler: ast.ExceptHandler) -> str | None:
    """The broad-exception spelling a handler catches, or ``None`` if it is narrow.

    A bare ``except:`` → ``"bare except"``; ``except Exception`` /
    ``except BaseException`` (or a tuple containing one) → that name. A narrow
    handler (``except ValueError``) → ``None``.
    """
    node = handler.type
    if node is None:
        return "bare except"
    if isinstance(node, ast.Name) and node.id in BROAD_EXCEPTION_NAMES:
        return node.id
    if isinstance(node, ast.Tuple):
        for element in node.elts:
            if isinstance(element, ast.Name) and element.id in BROAD_EXCEPTION_NAMES:
                return element.id
    return None


def _broad_suppress_arg(call: ast.expr) -> str | None:
    """The broad-exception name a ``suppress(...)`` call silences, or ``None``.

    Matches ``contextlib.suppress(Exception)`` / ``suppress(BaseException)`` — a
    context manager that swallows its body's exception with *no* handler body to log
    in, so it is always silent and held to the same rule.
    """
    if not isinstance(call, ast.Call):
        return None
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
    if name != "suppress":
        return None
    for arg in call.args:
        if isinstance(arg, ast.Name) and arg.id in BROAD_EXCEPTION_NAMES:
            return arg.id
    return None


def _body_is_observable(body: list[ast.stmt]) -> bool:
    """Whether a handler body re-raises or records its failure observably.

    True if the body contains any ``raise`` (a re-raise still surfaces the failure)
    or a call that logs / emits an event / notes the suppression. Walks the whole
    body, so a re-raise or a log nested in an ``if`` still counts.
    """
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Raise):
                return True
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and (
                    func.attr in LOG_METHOD_NAMES
                    or func.attr in _EMIT_METHOD_NAMES
                    or func.attr in _NOTE_FUNCTION_NAMES
                ):
                    return True
                if isinstance(func, ast.Name) and func.id in _NOTE_FUNCTION_NAMES:
                    return True
    return False


def _line_has_noqa(line: str) -> bool:
    """Whether a source line carries the justifying ``# noqa: BLE001`` (or a bare ``# noqa``)."""
    if "noqa" not in line:
        return False
    return "BLE001" in line or line.rstrip().endswith("# noqa")


def _qualname(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """The enclosing function/class chain for a node, or ``"<module>"`` at module scope."""
    chain: list[str] = []
    cursor: ast.AST | None = parents.get(node)
    while cursor is not None:
        if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            chain.append(cursor.name)
        cursor = parents.get(cursor)
    return ".".join(reversed(chain)) if chain else "<module>"


def silent_swallows_in_source(source: str) -> list[tuple[str, int, str]]:
    """Silent broad swallows in one module's source as ``(qualname, lineno, detail)``.

    Pure and injectable (mirrors :func:`vincio._error_contract.contract_raises_in_source`):
    parses ``source`` and returns one sorted row per broad ``except`` (or
    ``contextlib.suppress(Exception)``) whose body neither re-raises nor records its
    failure observably and that carries no ``# noqa: BLE001``. ``detail`` names what
    is swallowed (``"Exception"`` / ``"BaseException"`` / ``"bare except"`` /
    ``"suppress(...)"``). A handler that logs, emits, notes, or re-raises — or a
    narrow handler — is not a swallow and is not reported.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def line_at(lineno: int) -> str:
        return lines[lineno - 1] if 0 <= lineno - 1 < len(lines) else ""

    rows: set[tuple[str, int, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            spelling = _is_broad_handler(node)
            if spelling is None:
                continue
            if _body_is_observable(node.body):
                continue
            if _line_has_noqa(line_at(node.lineno)):
                continue
            rows.add((_qualname(node, parents), node.lineno, spelling))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                suppressed = _broad_suppress_arg(item.context_expr)
                if suppressed is None:
                    continue
                if _line_has_noqa(line_at(node.lineno)):
                    continue
                rows.add((_qualname(node, parents), node.lineno, f"suppress({suppressed})"))
    return sorted(rows)


def silent_swallows() -> list[str]:
    """Every silent broad swallow tree-wide as a sorted ``"<module>:<lineno> ..."`` message.

    The always-on gate: an empty list means no public module swallows a broad
    exception silently. Each message names the module, line, enclosing qualname, and
    what is swallowed, so a violation is a one-line locate-and-fix.
    """
    out: list[str] = []
    for module, path in _public_module_paths():
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        for qualname, lineno, detail in silent_swallows_in_source(source):
            out.append(f"{module}:{lineno} ({qualname}) swallows {detail} with no log, metric, or re-raise")
    return sorted(out)


def silent_swallow_count() -> int:
    """The number of silent broad swallows tree-wide (``0`` when the gate is clean)."""
    return len(silent_swallows())


if __name__ == "__main__":  # pragma: no cover - dev tool
    import sys

    problems = silent_swallows()
    if problems:
        print("silent broad-except swallows (add a log / note_suppressed / re-raise, or a")
        print("justifying `# noqa: BLE001` with the reason):")
        print("\n".join(problems))
        sys.exit(1)
    print("observable failure conformant")
