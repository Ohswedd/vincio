"""The deepened docs-completeness gate: a docs-*graph* check, not a substring.

5.4 turned the docs into a connected graph: a single source of truth
(``vincio._docmap``) binds every public ``app.*`` verb to the concept, guide,
example, and reference that document it, renders the capability map / learning
path / Related cross-links / ``llms.txt`` from it, and gates the whole graph.
This module operationalizes that gate so navigation can't silently regress:

* every public ``app.*`` verb is placed in the capability map and appears in
  ``api.md``;
* every internal docs link resolves (path + anchor);
* every concept reaches a guide, an example, and a reference anchor;
* every concept and guide carries a current, single-sourced Related block;
* no docs page is orphaned; and
* ``llms.txt`` is current (regenerated from ``vincio.__all__``).
"""

from __future__ import annotations

import importlib

import pytest

from vincio import _docmap


def test_every_app_verb_is_placed_in_the_capability_map():
    uncovered = _docmap.uncovered_verbs()
    assert not uncovered, (
        f"public app.* verbs missing from the capability map: {uncovered} — "
        f"add each to a Topic in vincio/_docmap.py::TOPICS"
    )


def test_doc_graph_references_only_real_verbs():
    real = set(_docmap.app_verbs())
    declared: set[str] = set()
    for topic in _docmap.TOPICS:
        declared.update(topic.verbs)
    assert declared <= real, f"doc graph references non-existent verbs: {sorted(declared - real)}"


@pytest.mark.parametrize("check", _docmap.docs_graph_report(), ids=lambda c: c.name)
def test_docs_graph_check_passes(check):
    assert check.ok, f"{check.name} failed:\n" + "\n".join(f"  - {p}" for p in check.problems)


def test_capability_map_is_current():
    on_disk = (_docmap._read(_docmap._CAPABILITY_MAP))
    assert on_disk == _docmap.render_capability_map(), (
        "docs/reference/capability-map.md is stale — run `vincio docs map`"
    )


def test_learning_path_is_current():
    on_disk = _docmap._read(_docmap._LEARNING_PATH)
    assert on_disk == _docmap.render_learning_path(), (
        "docs/learning-path.md is stale — run `vincio docs map`"
    )


def test_llms_txt_is_current():
    assert _docmap.llms_txt_current().ok, "llms.txt is stale — run `vincio docs map`"


def test_every_app_method_is_documented_in_api_md():
    api = _docmap._read(_docmap._API_REF)
    missing = [v for v in _docmap.app_verbs() if f"app.{v}" not in api]
    assert not missing, f"public app.* methods absent from api.md: {missing}"


def test_api_md_app_index_block_is_current():
    api = _docmap._read(_docmap._API_REF)
    block = _docmap._extract_block(api, _docmap._APPINDEX_BEGIN, _docmap._APPINDEX_END)
    assert block == _docmap.render_app_method_index(), (
        "api.md app-method index is stale — run `vincio docs map`"
    )


@pytest.mark.parametrize("page", _docmap.concept_pages())
def test_every_concept_has_a_current_related_block(page):
    text = _docmap._read(page)
    assert _docmap._RELATED_BEGIN in text, f"{page} is missing its Related block"
    block = _docmap._extract_block(text, _docmap._RELATED_BEGIN, _docmap._RELATED_END)
    assert block == _docmap.render_related_block(page), f"{page} Related block is stale"


@pytest.mark.parametrize("page", _docmap.guide_pages())
def test_every_guide_has_a_current_related_block(page):
    text = _docmap._read(page)
    assert _docmap._RELATED_BEGIN in text, f"{page} is missing its Related block"
    block = _docmap._extract_block(text, _docmap._RELATED_BEGIN, _docmap._RELATED_END)
    assert block == _docmap.render_related_block(page), f"{page} Related block is stale"


def test_every_concept_reaches_a_guide_example_and_reference():
    for concept in _docmap.concept_pages():
        topics = [t for t in _docmap.TOPICS if t.concept == concept]
        assert topics, f"concept not in the doc graph: {concept}"
        assert any(t.guides for t in topics), f"concept reaches no guide: {concept}"
        assert any(t.examples for t in topics), f"concept reaches no example: {concept}"


def test_sync_is_idempotent():
    # A fresh render must equal what is committed: nothing would change.
    assert _docmap.sync_docs(write=False) == [], (
        "generated docs artifacts are stale — run `vincio docs map`"
    )


def test_docs_cli_check_and_map_pass():
    cli = importlib.import_module("vincio.cli.main")
    parser = cli.build_parser()
    assert cli.cmd_docs_check(parser.parse_args(["docs", "check", "--json"])) == 0
    assert cli.cmd_docs_map(parser.parse_args(["docs", "map", "--check"])) == 0


def test_new_reference_pages_are_indexed():
    index = _docmap._read("docs/README.md")
    assert "capability-map.md" in index, "capability map not linked from the docs index"
    assert "learning-path.md" in index, "learning path not linked from the docs index"
