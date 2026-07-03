"""``vincio doctor``: report a project's use of deprecated APIs and config drift.

The doctor is driven by the same :func:`~vincio.stability.stability_of` metadata
the library uses to mark its own surface, so a symbol that is deprecated with
:func:`~vincio.stability.deprecated` is reported with its replacement and removal
version automatically — no separate list to maintain. It also flags a
``vincio.yaml`` that predates the current schema and points at
``vincio config migrate``.

It is a static check: it parses project source with :mod:`ast` (it does not
import or run it), so it is safe to run in CI against untrusted code.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.diagnostics import note_suppressed
from ..stability import StabilityLevel, stability_of
from ._symbol_scan import resolve_attr_module, vincio_module_aliases

__all__ = [
    "Deprecation",
    "Finding",
    "DoctorReport",
    "collect_deprecations",
    "scan_source",
    "scan_config",
    "run_doctor",
]

# Directories never worth scanning for project source.
_SKIP_DIRS = {
    ".venv",
    "venv",
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "site-packages",
    "build",
    "dist",
    ".vincio",
}


@dataclass(frozen=True, slots=True)
class Deprecation:
    """A deprecated public symbol and its migration metadata."""

    name: str
    since: str | None
    removed_in: str | None
    alternative: str | None

    def remediation(self) -> str:
        parts: list[str] = []
        if self.alternative:
            parts.append(f"use {self.alternative} instead")
        if self.removed_in:
            parts.append(f"removed in {self.removed_in}")
        return "; ".join(parts) if parts else "see the deprecation policy"


@dataclass(frozen=True, slots=True)
class Finding:
    """One actionable issue the doctor found."""

    kind: str  # "deprecated_api" | "config_drift"
    file: str
    line: int
    message: str
    remediation: str


@dataclass(slots=True)
class DoctorReport:
    """The aggregate doctor result for a project."""

    findings: list[Finding]
    files_scanned: int
    deprecations_known: int

    @property
    def ok(self) -> bool:
        """True when no actionable issues were found."""
        return not self.findings


def _record_to_deprecation(name: str, record: dict[str, Any]) -> Deprecation:
    return Deprecation(
        name=name,
        since=record.get("since"),
        removed_in=record.get("removed_in"),
        alternative=record.get("alternative"),
    )


def collect_deprecations(package: str = "vincio") -> dict[str, Deprecation]:
    """Map every deprecated public symbol name to its migration metadata.

    Walks the top-level ``__all__`` of *package* plus the ``__all__`` of each
    immediate subpackage/module, keeping any symbol whose
    :func:`stability_of` level is :attr:`StabilityLevel.DEPRECATED`.
    """
    deprecations: dict[str, Deprecation] = {}
    try:
        root = importlib.import_module(package)
    except Exception:
        note_suppressed("doctor.import_package")
        return deprecations

    def consider(name: str, obj: Any) -> None:
        record = stability_of(obj)
        if record.get("level") is StabilityLevel.DEPRECATED:
            deprecations[name] = _record_to_deprecation(name, record)

    for name in getattr(root, "__all__", ()):  # top-level surface
        consider(name, getattr(root, name, None))

    search_path = getattr(root, "__path__", None)
    if search_path is not None:
        for info in pkgutil.iter_modules(search_path):
            try:
                module = importlib.import_module(f"{package}.{info.name}")
            except Exception:
                note_suppressed("doctor.import_module")
                continue
            for name in getattr(module, "__all__", ()):
                consider(name, getattr(module, name, None))
    return deprecations


def _imported_deprecated_names(
    tree: ast.AST, deprecations: dict[str, Deprecation]
) -> tuple[dict[str, Deprecation], list[Finding]]:
    """Find names imported from ``vincio*`` that are deprecated.

    Returns the local-name -> Deprecation map (honoring ``as`` aliases) and a
    finding for each deprecated import site.
    """
    bound: dict[str, Deprecation] = {}
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # A relative `from .vincio import ...` names the project's own
            # local module, never this library — leave it alone.
            if node.level or (module != "vincio" and not module.startswith("vincio.")):
                continue
            for alias in node.names:
                dep = deprecations.get(alias.name)
                if dep is None:
                    continue
                local = alias.asname or alias.name
                bound[local] = dep
                findings.append(
                    Finding(
                        kind="deprecated_api",
                        file="",
                        line=node.lineno,
                        message=(
                            f"imports deprecated `{alias.name}` from `{module}`"
                            + (f" (since {dep.since})" if dep.since else "")
                        ),
                        remediation=dep.remediation(),
                    )
                )
    return bound, findings


# Keyword-argument runways: an old keyword accepted-and-warned on the way to
# removal. Statically flagged only on calls whose function provably resolves
# to this library (a name imported from ``vincio*``, or an attribute of a
# vincio module) — receiver-typed method calls (``book.attest(verify_with=)``,
# ``credential.verify(at=)``) can't be resolved without type inference and are
# covered by the runtime ``VincioDeprecationWarning`` instead.
_DEPRECATED_KWARGS: dict[str, tuple[str, str, str]] = {
    # old keyword -> (replacement, since, removed_in)
    "verify_with": ("verifier", "7.5", "8.0"),
}


def _vincio_imported_names(tree: ast.AST) -> set[str]:
    """Every local name bound by a ``from vincio[.sub] import ...`` statement."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or (module != "vincio" and not module.startswith("vincio.")):
                continue
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def scan_source(path: str | Path, deprecations: dict[str, Deprecation]) -> list[Finding]:
    """Statically scan one Python file for deprecated-API usage."""
    file_path = Path(path)
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (OSError, SyntaxError):
        return []
    if not deprecations:
        return []

    bound, findings = _imported_deprecated_names(tree, deprecations)
    aliases = vincio_module_aliases(tree)
    from_vincio = _vincio_imported_names(tree)
    seen: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        # Attribute access on vincio or any vincio module, however it is
        # reached: `vincio.old`, `vincio.data.old`, `import vincio.data as vd;
        # vd.old`, `from vincio import data; data.old`.
        if isinstance(node, ast.Attribute) and node.attr in deprecations:
            module = resolve_attr_module(node.value, aliases)
            if module is not None:
                dep = deprecations[node.attr]
                # Report at the attribute token's own line (the value and the
                # dot may sit lines above in a parenthesized chain).
                line = node.end_lineno if node.end_lineno is not None else node.lineno
                key = (line, node.attr)
                if key not in seen:
                    seen.add(key)
                    findings.append(
                        Finding(
                            kind="deprecated_api",
                            file="",
                            line=line,
                            message=f"uses deprecated `{module}.{node.attr}`",
                            remediation=dep.remediation(),
                        )
                    )
        # A reference to a locally-bound deprecated import.
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            bound_dep = bound.get(node.id)
            if bound_dep is not None:
                key = (node.lineno, node.id)
                if key not in seen:
                    seen.add(key)
                    findings.append(
                        Finding(
                            kind="deprecated_api",
                            file="",
                            line=node.lineno,
                            message=f"uses deprecated `{bound_dep.name}`",
                            remediation=bound_dep.remediation(),
                        )
                    )
        # A deprecated keyword on a call that provably targets this library.
        elif isinstance(node, ast.Call) and node.keywords:
            func = node.func
            is_vincio_call = (
                isinstance(func, ast.Name) and func.id in from_vincio
            ) or (
                isinstance(func, ast.Attribute)
                and resolve_attr_module(func.value, aliases) is not None
            )
            if not is_vincio_call:
                continue
            for kw in node.keywords:
                spec = _DEPRECATED_KWARGS.get(kw.arg or "")
                if spec is None:
                    continue
                replacement, since, removed_in = spec
                key = (kw.value.lineno, f"{kw.arg}=")
                if key not in seen:
                    seen.add(key)
                    findings.append(
                        Finding(
                            kind="deprecated_api",
                            file="",
                            line=kw.value.lineno,
                            message=f"passes deprecated keyword `{kw.arg}=` (since {since})",
                            remediation=f"use {replacement}= instead; removed in {removed_in}",
                        )
                    )
    return [
        Finding(f.kind, str(file_path), f.line, f.message, f.remediation) for f in findings
    ]


def scan_config(root: str | Path) -> list[Finding]:
    """Flag a project ``vincio.yaml`` that is behind the current schema.

    Looks only at the config file in *root* itself (the doctor scans a project,
    so it does not walk up out of the project tree to an unrelated config).
    """
    import yaml

    from ..core.config import CONFIG_FILENAMES
    from ..core.config_migrations import CONFIG_SCHEMA_VERSION, detect_version, migrate

    base = Path(root)
    config_path: Path | None = None
    for name in CONFIG_FILENAMES:
        candidate = base / name
        if candidate.is_file():
            config_path = candidate
            break
    if config_path is None:
        return []
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(raw, dict):
        return []
    result = migrate(raw)
    if not result.steps:
        return []
    current = detect_version(raw)
    detail = "; ".join(result.notes) if result.notes else "stamp schema version"
    return [
        Finding(
            kind="config_drift",
            file=str(config_path),
            line=1,
            message=(
                f"config schema is at version {current}, current is "
                f"{CONFIG_SCHEMA_VERSION} ({detail})"
            ),
            remediation="run `vincio config migrate` to upgrade the file",
        )
    ]


def _iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def run_doctor(
    root: str | Path = ".",
    *,
    deprecations: dict[str, Deprecation] | None = None,
) -> DoctorReport:
    """Scan a project tree for deprecated-API usage and config drift."""
    base = Path(root)
    known = collect_deprecations() if deprecations is None else deprecations
    findings: list[Finding] = []
    files = _iter_python_files(base) if base.is_dir() else [base]
    for file_path in files:
        findings.extend(scan_source(file_path, known))
    findings.extend(scan_config(base if base.is_dir() else base.parent))
    return DoctorReport(
        findings=findings,
        files_scanned=len(files),
        deprecations_known=len(known),
    )
