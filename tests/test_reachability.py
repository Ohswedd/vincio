"""Gate: public-symbol reachability (vincio._reachability).

:mod:`vincio._surface` proves every name in a public ``__all__`` *resolves*; it
cannot tell a live, *used* symbol from one that resolves yet is referenced
nowhere — the dead-but-resolvable surface the hardening line's 6.0 phase removed
by a one-time manual reference check. 6.6 mechanizes that rubric: every public
symbol is either *used* somewhere in the code corpus or *declared*, with its
structural reason, in the frozen reachability baseline
(``docs/reference/reachability.txt``). This gate proves the tree is conformant,
proves the baseline is exactly the live unreferenced set (no missing or stale
entry), and proves the detector bites — so a new dead public symbol cannot
silently ship. Reproduce offline with ``python -m vincio._reachability``.
"""

from __future__ import annotations

from vincio import _reachability as r

# --- the always-on gate ------------------------------------------------------


def test_reachability_conformant_tree_wide():
    """Every public symbol is used in the corpus or declared in the baseline.

    A new dead-but-resolvable public symbol fails here the moment it lands:
    exercise it with a test (preferred), or declare it BASE/OPTDEP/WIRING in
    ``vincio._reachability.REACHABILITY_BASELINE``.
    """
    problems = r.reachability_problems()
    assert problems == [], "reachability violations:\n" + "\n".join(problems)


def test_baseline_is_exactly_the_unreferenced_set():
    """No missing entry (an undeclared dead symbol) and no stale one (a baselined
    symbol that has since gained a reference)."""
    assert set(r.REACHABILITY_BASELINE) == set(r.unreferenced_public_symbols())


def test_baseline_categories_are_valid():
    assert set(r.REACHABILITY_BASELINE.values()) <= {"BASE", "OPTDEP", "WIRING"}


def test_manifest_is_frozen():
    """The rendered manifest matches the committed file byte-for-byte, so any
    baseline edit is a reviewed diff."""
    assert r.render_manifest() == r.load_manifest()


def test_surface_covers_top_level_and_subpackages():
    surface = r.public_surface()
    # A top-level re-export carries both its top and subpackage level.
    assert surface["serve_viewer"] == ["vincio", "vincio.observability"]
    # A subpackage-only public symbol carries just its subpackage.
    assert surface["BlobStore"] == ["vincio.storage"]
    # ``__version__`` is not a symbol.
    assert "__version__" not in surface


def test_home_subpackage_picks_the_owning_subpackage():
    assert r.home_subpackage(["vincio", "vincio.observability"]) == "vincio.observability"
    assert r.home_subpackage(["vincio"]) == "vincio"


# --- the detector bites ------------------------------------------------------


def test_detector_flags_unreferenced_symbol_missing_from_baseline():
    """An empty baseline leaves every legitimately-unexercised symbol undeclared."""
    problems = r.reachability_problems(baseline={})
    assert any("BlobStore" in p and "referenced nowhere" in p for p in problems)


def test_detector_flags_stale_baseline_entry():
    """A baselined symbol that is actually referenced (a live ``app`` facade) is stale."""
    problems = r.reachability_problems(baseline={"ContextApp": "BASE"})
    assert any("ContextApp" in p and "stale" in p for p in problems)


def test_detector_flags_invalid_category():
    problems = r.reachability_problems(baseline={**r.REACHABILITY_BASELINE, "BlobStore": "NOPE"})
    assert any("invalid reachability category" in p for p in problems)


# --- the reference scan distinguishes a use from a declaration ---------------


def test_scan_counts_a_load_as_a_use():
    used = r.referenced_symbols_in_source("x = Widget()\n", {"Widget"})
    assert used == {"Widget"}


def test_scan_counts_an_aliased_import_use():
    """``from m import Sym as alias`` then a load of ``alias`` credits ``Sym`` —
    the case that made ``selection_signature`` look dead before the fix."""
    source = "from m import Sym as _Sym\n\n_Sym()\n"
    assert r.referenced_symbols_in_source(source, {"Sym"}) == {"Sym"}


def test_scan_ignores_a_bare_reexport_and_all_entry():
    """A re-export binds a name without loading it, and an ``__all__`` string is a
    constant, not a reference — neither is a use."""
    source = "from .sub import Widget\n\n__all__ = ['Widget']\n"
    assert r.referenced_symbols_in_source(source, {"Widget"}) == set()


def test_scan_counts_attribute_access():
    assert r.referenced_symbols_in_source("vincio.serve_viewer(store)\n", {"serve_viewer"}) == {
        "serve_viewer"
    }


def test_private_lints_are_outside_the_public_surface():
    surface = r.public_surface()
    assert "reachability_problems" not in surface
    assert "surface_problems" not in surface
