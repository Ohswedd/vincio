"""Gate: docstring coverage of the public surface + generated API index."""

from __future__ import annotations

from pathlib import Path

from vincio._apiref import (
    docstring_summary,
    public_symbols,
    render_api_index,
    symbol_signature,
    undocumented_symbols,
)

ROOT = Path(__file__).resolve().parent.parent


def test_no_public_symbol_ships_undocumented():
    missing = undocumented_symbols()
    assert not missing, (
        f"public symbols without a docstring: {missing} — every name in "
        f"vincio.__all__ must be documented (the docstring-coverage gate)"
    )


def test_public_symbols_cover_all():
    import vincio

    names = {n for n, _ in public_symbols()}
    expected = set(vincio.__all__) - {"__version__"}
    assert names == expected


def test_signature_rendering_is_robust():
    # Every public symbol renders a signature/name without raising.
    for name, obj in public_symbols():
        rendered = symbol_signature(name, obj)
        assert rendered.startswith(name)


def test_generated_api_index_is_current():
    page = (ROOT / "docs" / "reference" / "api-generated.md").read_text(encoding="utf-8")
    assert page == render_api_index(), (
        "docs/reference/api-generated.md is stale — regenerate it from "
        "vincio._apiref.render_api_index()"
    )


def test_generated_index_lists_headline_symbols():
    page = (ROOT / "docs" / "reference" / "api-generated.md").read_text(encoding="utf-8")
    for headline in ("ContextApp", "Workflow", "MemoryEngine", "VincioError"):
        assert headline in page


def test_summaries_are_single_line():
    for _name, obj in public_symbols():
        assert "\n" not in docstring_summary(obj)
