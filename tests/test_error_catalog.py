"""Gate: every error has a stable code, remediation, and docs link.

Operationalizes the "internationalized, actionable errors" goal — the catalog is
the single source of truth, complete over the whole hierarchy, and the error
reference page is generated from it so docs links never dangle.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

import vincio
import vincio.core.errors as errors_mod
from vincio.core.error_catalog import (
    DOCS_BASE_URL,
    ERROR_CATALOG,
    PROTOCOL_ERROR_CLASSES,
    available_error_locales,
    catalog_entry,
    docs_url_for,
    register_error_locale,
    remediation_for,
    render_error_reference,
    set_default_error_locale,
    title_for,
)
from vincio.core.errors import ProviderError, VincioError

ROOT = Path(__file__).resolve().parent.parent


def _all_error_classes() -> set[type[VincioError]]:
    # Import every submodule so all VincioError subclasses are defined.
    for module in pkgutil.walk_packages(vincio.__path__, "vincio."):
        try:
            importlib.import_module(module.name)
        except Exception:  # optional deps / heavy modules
            continue

    def subclasses(cls: type) -> set[type]:
        out: set[type] = set()
        for sub in cls.__subclasses__():
            out.add(sub)
            out |= subclasses(sub)
        return out

    return {errors_mod.VincioError, *subclasses(errors_mod.VincioError)}


def test_every_string_coded_error_has_a_catalog_entry():
    missing: list[str] = []
    for cls in _all_error_classes():
        code = cls.code
        if isinstance(code, str):
            if code not in ERROR_CATALOG:
                missing.append(f"{cls.__name__}({code})")
    assert not missing, f"errors with no catalog entry: {missing}"


def test_no_orphan_catalog_entries():
    live_codes = {c.code for c in _all_error_classes() if isinstance(c.code, str)}
    orphans = set(ERROR_CATALOG) - live_codes
    assert not orphans, f"catalog codes with no error class: {orphans}"


def test_non_string_codes_are_exactly_the_known_protocol_errors():
    non_string = {c.__name__ for c in _all_error_classes() if not isinstance(c.code, str)}
    assert non_string == set(PROTOCOL_ERROR_CLASSES), (
        "a new error uses a non-string .code; add it to PROTOCOL_ERROR_CLASSES "
        "or give it a stable string code with a catalog entry"
    )


def test_catalog_entries_are_well_formed():
    for code, entry in ERROR_CATALOG.items():
        assert entry.code == code
        assert entry.title and entry.title[0].isupper()
        assert len(entry.remediation) > 20, f"{code} remediation too thin"


def test_instance_exposes_remediation_and_docs_url():
    exc = VincioError("boom")
    assert exc.code == "VINCIO_ERROR"
    assert exc.remediation == remediation_for("VINCIO_ERROR")
    assert exc.docs_url == f"{DOCS_BASE_URL}#vincio_error"
    payload = exc.to_dict()
    assert payload["code"] == "VINCIO_ERROR"
    assert payload["remediation"]
    assert payload["docs_url"].startswith(DOCS_BASE_URL)


def test_subclass_resolves_its_own_code():
    exc = ProviderError("down", provider="openai")
    assert exc.code == "PROVIDER_ERROR"
    assert "FailoverChain" in (exc.remediation or "")
    assert exc.docs_url == docs_url_for("PROVIDER_ERROR")


def test_instance_hint_override_wins():
    exc = VincioError("boom", hint="do the thing", docs_url="https://example/x")
    assert exc.remediation == "do the thing"
    assert exc.docs_url == "https://example/x"


def test_docs_url_is_none_for_unknown_code():
    assert docs_url_for("NOPE") is None
    assert catalog_entry("NOPE") is None


def test_internationalization_lookup_and_fallback():
    register_error_locale(
        "xx",
        {"VINCIO_ERROR": ("Erreur", "Faites la chose"), "CONFIG_ERROR": ("Config", "Réparez")},
    )
    assert "xx" in available_error_locales()
    assert title_for("VINCIO_ERROR", locale="xx") == "Erreur"
    assert remediation_for("CONFIG_ERROR", locale="xx") == "Réparez"
    # A code without a translation falls back to English.
    assert remediation_for("PROVIDER_AUTH", locale="xx") == remediation_for("PROVIDER_AUTH")
    # Unknown code in a locale registration is rejected.
    with pytest.raises(KeyError):
        register_error_locale("xx", {"NOPE": ("a", "b")})


def test_default_locale_switch(monkeypatch):
    register_error_locale("zz", {"VINCIO_ERROR": ("ZZ", "zz remediation")})
    try:
        set_default_error_locale("zz")
        assert remediation_for("VINCIO_ERROR") == "zz remediation"
    finally:
        set_default_error_locale("en")
    assert remediation_for("VINCIO_ERROR") == ERROR_CATALOG["VINCIO_ERROR"].remediation


def test_error_reference_page_is_current():
    page = (ROOT / "docs" / "reference" / "errors.md").read_text(encoding="utf-8")
    assert page == render_error_reference(), (
        "docs/reference/errors.md is stale — regenerate it from "
        "vincio.core.error_catalog.render_error_reference()"
    )


def test_error_reference_documents_every_code():
    page = (ROOT / "docs" / "reference" / "errors.md").read_text(encoding="utf-8")
    for code in ERROR_CATALOG:
        assert f"### {code}" in page, f"{code} missing from errors.md"
