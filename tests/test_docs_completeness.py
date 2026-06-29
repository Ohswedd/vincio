"""Docs-completeness gate: every subsystem and example stays documented.

Operationalizes the "a guide and a tested example for every subsystem and
every public API" goal so coverage can't silently regress. From 5.4 the gate
deepens from a substring check into a docs-*graph* check: the structural
guarantees (links resolve, every concept reaches a guide/example/reference,
every ``app.*`` verb is documented, no orphans, ``llms.txt`` is current) live in
``tests/test_docs_graph.py`` driving ``vincio._docmap``; the bridge test below
ties that graph into this completeness gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "vincio"
DOCS = ROOT / "docs"
EXAMPLES = ROOT / "examples"

# Internal/implementation packages that are not part of the public, documented
# surface (no public entry points the docs need to advertise).
_PRIVATE = {"__pycache__"}


def _public_subsystems() -> list[str]:
    names: list[str] = []
    for child in PKG.iterdir():
        if child.name.startswith("_"):
            continue
        if child.is_dir() and child.name not in _PRIVATE:
            names.append(child.name)
        elif child.suffix == ".py" and child.name != "__init__.py":
            names.append(child.stem)
    return sorted(names)


def _all_docs_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in DOCS.rglob("*.md"))


@pytest.mark.parametrize("subsystem", _public_subsystems())
def test_subsystem_is_documented(subsystem):
    text = _all_docs_text()
    assert f"vincio.{subsystem}" in text, (
        f"vincio.{subsystem} is not referenced anywhere under docs/ — "
        f"add it to docs/reference/api.md or a concept/guide page."
    )


def test_reference_pages_exist():
    for page in (
        "api.md",
        "api-generated.md",
        "cli.md",
        "config.md",
        "stability.md",
        "slo.md",
        "errors.md",
        "typing.md",
    ):
        assert (DOCS / "reference" / page).is_file(), f"missing docs/reference/{page}"


def test_docstring_coverage_of_public_surface():
    """The docs-completeness gate extends to docstring coverage: no public
    symbol in ``vincio.__all__`` may ship without a docstring."""
    from vincio._apiref import undocumented_symbols

    missing = undocumented_symbols()
    assert not missing, f"undocumented public symbols: {missing}"


def test_error_catalog_completeness():
    """The docs-completeness gate extends to the error catalog: every error
    code is documented with a remediation in the reference page."""
    from vincio.core.error_catalog import ERROR_CATALOG, render_error_reference

    page = (DOCS / "reference" / "errors.md").read_text(encoding="utf-8")
    assert page == render_error_reference(), "docs/reference/errors.md is stale"
    for code, entry in ERROR_CATALOG.items():
        assert f"### {code}" in page
        assert entry.remediation


def test_py_typed_marker_ships():
    """Strict-typing deliverable: the PEP 561 marker must ship so downstream
    type-checkers see Vincio's inline contract."""
    assert (PKG / "py.typed").is_file(), "missing vincio/py.typed (PEP 561 marker)"


def test_threat_model_documented():
    assert (DOCS / "security" / "threat-model.md").is_file()


def test_every_example_is_listed_in_readme():
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    for example in sorted(EXAMPLES.glob("[0-9]*.py")):
        # README links by filename.
        assert example.name in readme, f"{example.name} missing from examples/README.md"


def test_public_api_names_resolve():
    import vincio

    # Every advertised name must import and the stability surface must be public.
    for name in vincio.__all__:
        assert hasattr(vincio, name), name
    for name in ("deprecated", "experimental", "stability_of", "StabilityLevel"):
        assert name in vincio.__all__


def test_docs_graph_is_intact():
    """The deepened gate: the docs graph (links, capability-map coverage,
    navigation reachability, no orphans, llms.txt freshness) must hold.

    Detailed, per-page assertions live in tests/test_docs_graph.py; this is the
    one-line bridge that fails the completeness gate if any of them regress.
    """
    from vincio import _docmap

    failed = [c.name for c in _docmap.docs_graph_report() if not c.ok]
    assert not failed, (
        f"docs-graph checks failed: {failed} — run `vincio docs check` for detail "
        f"and `vincio docs map` to regenerate"
    )
