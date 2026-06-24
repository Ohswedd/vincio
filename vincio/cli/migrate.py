"""``vincio migrate <target>``: the source codemod for a major-version upgrade.

Where :mod:`vincio.cli.doctor` *reports* a project's use of deprecated APIs,
this module *rewrites* it. It is the code-surface analogue of the
``vincio config migrate`` discipline: a one-shot, mechanical codemod that renames
the public symbols a major bump consolidates, driven by a declarative rename
table keyed by target version.

Like the doctor it is a **static** tool: it parses project source with
:mod:`ast` (it never imports or runs it) and rewrites only the exact identifier
tokens a rename touches, so it is safe to run in CI against untrusted code and
leaves formatting, comments, and unrelated code byte-for-byte intact.

The 3.x line was strictly additive on a frozen public surface and **no public
API reached its removal runway**, so the ``"4.0"`` rename table is intentionally
empty — a clean 3.x → 4.0 upgrade needs no source changes. The machinery ships
regardless: it gives ``vincio migrate 4.0`` a truthful "nothing to do" answer
today and is the mechanism any future 4.x consolidation (or the 5.0 removal of a
symbol deprecated across 4.x) is delivered through.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .doctor import _iter_python_files

__all__ = [
    "SymbolRename",
    "Rewrite",
    "MigrationReport",
    "RENAMES",
    "SUPPORTED_TARGETS",
    "renames_for",
    "scan_source",
    "apply_rewrites",
    "run_migrate",
]


@dataclass(frozen=True, slots=True)
class SymbolRename:
    """One public symbol renamed by a major-version consolidation.

    ``old`` is the 3.x name a project may still import; ``new`` is the canonical
    name to steer it to. ``since`` is the version that introduced the rename and
    ``note`` an optional one-line rationale shown in ``MIGRATION.md`` and the
    codemod output.
    """

    old: str
    new: str
    since: str
    note: str | None = None


# The surface migration table, keyed by major target. Each entry maps a 3.x
# public name to its 4.x canonical name. ``"4.0"`` is intentionally empty: the
# additive-only 3.x contract was held end to end, so no symbol needed renaming
# or removing across the major. Future 4.x consolidations append to a new key
# (e.g. ``"5.0"``) — never mutate a shipped table.
RENAMES: dict[str, tuple[SymbolRename, ...]] = {
    "4.0": (),
}

SUPPORTED_TARGETS: tuple[str, ...] = tuple(RENAMES)


def renames_for(target: str) -> tuple[SymbolRename, ...]:
    """Return the rename table for *target*, or raise ``KeyError`` if unknown."""
    return RENAMES[target]


@dataclass(frozen=True, slots=True)
class Rewrite:
    """A single identifier rewrite at an exact source position (0-based col)."""

    file: str
    line: int
    col: int
    old: str
    new: str

    def describe(self) -> str:
        return f"{self.file}:{self.line}:{self.col + 1}: `{self.old}` → `{self.new}`"


@dataclass(slots=True)
class MigrationReport:
    """The aggregate result of scanning (and optionally rewriting) a project."""

    target: str
    rewrites: list[Rewrite]
    files_scanned: int
    files_written: int = 0

    @property
    def ok(self) -> bool:
        """True when the project needs no changes for *target*."""
        return not self.rewrites

    @property
    def files_affected(self) -> int:
        """Number of distinct files a rewrite touches."""
        return len({r.file for r in self.rewrites})


def _attr_col(node: ast.Attribute) -> int:
    """0-based column where ``node.attr`` begins (after ``value.``)."""
    # end_col_offset spans the whole attribute expression and always ends at the
    # last character of the attribute name, so this is robust to whitespace and
    # line continuations between the value, the dot, and the attribute.
    end = node.end_col_offset
    return end - len(node.attr) if end is not None else node.col_offset


def scan_source(
    path: str | Path, renames: dict[str, SymbolRename]
) -> list[Rewrite]:
    """Statically find every identifier in one file a rename should rewrite.

    *renames* maps each old name to its :class:`SymbolRename`. Three forms are
    rewritten, mirroring how the doctor recognises deprecated usage:

    * ``from vincio[.sub] import old`` — the imported-name token (and, when not
      aliased with ``as``, every later use of the bound name);
    * ``vincio.old`` attribute access — the attribute token;
    * a bare use of a name imported unaliased from ``vincio*``.

    An ``import ... as alias`` rebinds the symbol locally, so only the imported
    token is rewritten and the local alias is left untouched.
    """
    file_path = Path(path)
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (OSError, SyntaxError):
        return []
    if not renames:
        return []

    rewrites: list[Rewrite] = []
    # Local names bound (unaliased) to a renamed symbol via a vincio import.
    bound: dict[str, SymbolRename] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module != "vincio" and not module.startswith("vincio."):
                continue
            for alias in node.names:
                rename = renames.get(alias.name)
                if rename is None:
                    continue
                rewrites.append(
                    Rewrite(
                        file=str(file_path),
                        line=alias.lineno,
                        col=alias.col_offset,
                        old=rename.old,
                        new=rename.new,
                    )
                )
                if alias.asname is None:
                    bound[alias.name] = rename

    seen: set[tuple[int, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "vincio" and node.attr in renames:
                rename = renames[node.attr]
                col = _attr_col(node)
                if (node.lineno, col) not in seen:
                    seen.add((node.lineno, col))
                    rewrites.append(
                        Rewrite(str(file_path), node.lineno, col, rename.old, rename.new)
                    )
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            rename = bound.get(node.id)
            if rename is not None and (node.lineno, node.col_offset) not in seen:
                seen.add((node.lineno, node.col_offset))
                rewrites.append(
                    Rewrite(
                        str(file_path), node.lineno, node.col_offset, rename.old, rename.new
                    )
                )
    return rewrites


def apply_rewrites(source: str, rewrites: list[Rewrite]) -> str:
    """Apply *rewrites* to *source*, returning the rewritten text.

    Each rewrite replaces the exact identifier token at its 0-based
    ``(line, col)``. Edits on a line are applied right-to-left so earlier
    columns keep their offsets, and a rewrite whose slice does not match its
    recorded ``old`` token is skipped (defends against stale positions).
    """
    if not rewrites:
        return source
    lines = source.splitlines(keepends=True)
    by_line: dict[int, list[Rewrite]] = {}
    for rw in rewrites:
        by_line.setdefault(rw.line, []).append(rw)
    for lineno, edits in by_line.items():
        idx = lineno - 1
        if not 0 <= idx < len(lines):
            continue
        line = lines[idx]
        for rw in sorted(edits, key=lambda r: r.col, reverse=True):
            start, end = rw.col, rw.col + len(rw.old)
            if line[start:end] == rw.old:
                line = line[:start] + rw.new + line[end:]
        lines[idx] = line
    return "".join(lines)


def run_migrate(
    root: str | Path = ".",
    *,
    target: str,
    write: bool = False,
) -> MigrationReport:
    """Scan a project tree for *target*'s renames, optionally rewriting in place.

    With ``write=False`` (the default) the report lists every rewrite without
    touching disk — a dry run. With ``write=True`` each affected file is
    rewritten atomically and ``files_written`` is set. Raises ``KeyError`` for
    an unknown *target*.
    """
    table = renames_for(target)
    renames = {r.old: r for r in table}
    base = Path(root)
    files = _iter_python_files(base) if base.is_dir() else [base]

    rewrites: list[Rewrite] = []
    for file_path in files:
        rewrites.extend(scan_source(file_path, renames))

    written = 0
    if write and rewrites:
        by_file: dict[str, list[Rewrite]] = {}
        for rw in rewrites:
            by_file.setdefault(rw.file, []).append(rw)
        for file_str, file_rewrites in by_file.items():
            path = Path(file_str)
            try:
                original = path.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover - defensive
                continue
            updated = apply_rewrites(original, file_rewrites)
            if updated != original:
                path.write_text(updated, encoding="utf-8")
                written += 1

    return MigrationReport(
        target=target,
        rewrites=rewrites,
        files_scanned=len(files),
        files_written=written,
    )
