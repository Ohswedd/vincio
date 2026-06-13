"""Docs-completeness gate: every subsystem and example stays documented.

Operationalizes the 1.0 "a guide and a tested example for every subsystem and
every public API" goal so coverage can't silently regress.
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
    for page in ("api.md", "cli.md", "config.md", "stability.md", "slo.md"):
        assert (DOCS / "reference" / page).is_file(), f"missing docs/reference/{page}"


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
