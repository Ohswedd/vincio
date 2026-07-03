"""Gate: error-contract conformance (vincio._error_contract).

Vincio's contract is that every error it raises derives from ``VincioError`` so
an application catches the family with one ``except VincioError`` and branches on
the stable ``.code``. The hardening line's 6.1 phase converted the off-contract
built-in exceptions that leaked off public entry points and made the contract
mechanical: this gate freezes the classified baseline of accepted public built-in
raises (``docs/reference/error-contract.txt``), holds the ``ContextApp`` verb
surface to *zero* off-contract raises with an always-on check, proves the
detector bites on an injected leak, and pins the three converted sites so they
cannot regress.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import vincio
from vincio import _error_contract as ec
from vincio.core.error_catalog import register_error_locale
from vincio.core.errors import ConfigError, InputError, VincioError
from vincio.providers.mock import MockProvider
from vincio.retrieval.embeddings import MultimodalEmbedder, MultimodalInput

ROOT = Path(__file__).resolve().parent.parent


# --- the frozen baseline -----------------------------------------------------


def test_error_contract_manifest_is_frozen():
    """The live public built-in raises must match the committed manifest.

    A new public bare-built-in raise must be a deliberate edit: convert it to a
    ``VincioError`` (it drops out) or, when it is genuinely internal
    input-validation, regenerate with ``python -m vincio._error_contract --freeze``
    and review the diff.
    """
    committed = ec.load_manifest()
    rendered = ec.render_manifest()
    if committed != rendered:
        committed_rows = set(committed.splitlines())
        rendered_rows = set(rendered.splitlines())
        added = sorted(rendered_rows - committed_rows)
        removed = sorted(committed_rows - rendered_rows)
        raise AssertionError(
            "error-contract baseline drifted from the frozen manifest; if "
            "intentional, regenerate with `python -m vincio._error_contract "
            f"--freeze` and review the diff. added={added} removed={removed}"
        )


def test_manifest_matches_committed_file():
    """The committed manifest file is the one this module reads (no path drift)."""
    on_disk = (ROOT / "docs" / "reference" / "error-contract.txt").read_text(encoding="utf-8")
    assert on_disk == ec.load_manifest()


def test_render_is_idempotent():
    """Rendering the baseline twice is byte-identical (deterministic ordering)."""
    assert ec.render_manifest() == ec.render_manifest()


def test_baseline_rows_are_well_formed():
    """Every baseline row names a public module, a qualname, and a built-in exception."""
    rows = ec.contract_rows()
    assert rows, "no baseline rows discovered"
    public = set(ec.public_modules())
    for module, qualname, exception in rows:
        assert module in public, module
        assert qualname and not qualname.startswith("."), qualname
        assert exception in ec.BUILTIN_EXCEPTION_NAMES, exception


# --- the always-on app-verb gate (no manifest) -------------------------------


def test_app_verb_surface_raises_no_builtin():
    """No public ``ContextApp`` (``app.*``) method raises a bare built-in."""
    problems = ec.app_verb_violations()
    assert problems == [], "\n".join(problems)


def test_app_verb_gate_classifies_injected_rows():
    """The verb gate flags a ContextApp row and ignores a non-ContextApp one (injectable)."""
    rows = [
        ("vincio.core.app", "ContextApp.some_verb", "ValueError"),
        ("vincio.evals.judges", "build_judge", "ValueError"),
    ]
    assert ec.app_verb_violations(rows) == ["vincio.core.app.ContextApp.some_verb raises ValueError"]


# --- the three converted sites cannot regress --------------------------------


def test_app_test_time_search_unknown_strategy_is_input_error(offline_config):
    """app.py:5791 — an unknown test-time search strategy raises InputError, not ValueError."""
    app = vincio.ContextApp(
        name="ec-ttc", provider=MockProvider(), model="mock-1", config=offline_config
    )
    with pytest.raises(InputError) as info:
        app.test_time_search("q", strategy="does-not-exist")
    assert info.value.code == "INPUT_ERROR"
    assert isinstance(info.value, VincioError)
    # A valid strategy must NOT trip the guard (regression sanity).
    result = app.test_time_search("q", strategy="self_consistency", n=2)
    assert result.best is not None


def test_register_error_locale_unknown_code_is_config_error():
    """error_catalog.py:785 — an unknown code in a locale registration raises ConfigError."""
    with pytest.raises(ConfigError) as info:
        register_error_locale("ec-xx", {"NOPE_NOT_A_CODE": ("a", "b")})
    assert info.value.code == "CONFIG_ERROR"
    assert isinstance(info.value, VincioError)


def test_multimodal_embedder_base_raises_config_error():
    """embeddings.py:675 — the unimplemented payload encoder raises ConfigError, not NotImplementedError."""
    embedder = MultimodalEmbedder()
    with pytest.raises(ConfigError) as info:
        embedder._multimodal_payload([MultimodalInput(text="x")], None)
    assert info.value.code == "CONFIG_ERROR"
    assert isinstance(info.value, VincioError)
    # The public embed_multimodal() path surfaces the same typed error.
    import asyncio

    with pytest.raises(ConfigError):
        asyncio.run(embedder.embed_multimodal([MultimodalInput(text="x")]))


# --- the detector bites ------------------------------------------------------


def test_detector_reports_public_builtin_raise():
    """A bare built-in raise on a public def is reported (the gate bites)."""
    source = (
        "class Widget:\n"
        "    def build(self):\n"
        "        raise ValueError('boom')\n"
    )
    rows = ec.contract_raises_in_source(source)
    assert ("Widget.build", "ValueError") in rows


def test_detector_ignores_private_and_oncontract_raises():
    """A private-def raise, an encapsulated nested helper, and a VincioError raise are not reported."""
    source = (
        "from vincio.core.errors import ConfigError\n"
        "class Widget:\n"
        "    def _private(self):\n"            # private method -> not surface
        "        raise ValueError('a')\n"
        "    def public_ok(self):\n"
        "        raise ConfigError('b')\n"      # on-contract -> not a builtin
        "    def public_with_helper(self):\n"
        "        def _step():\n"                # encapsulated nested private def
        "            raise KeyError('c')\n"
        "        return _step\n"
        "def _module_private():\n"             # private module-level func
        "    raise TypeError('d')\n"
    )
    rows = ec.contract_raises_in_source(source)
    assert rows == []


def test_detector_handles_bare_reraise_and_attribute_callables():
    """A bare ``raise`` and a non-Name exception callable are not classified as built-ins."""
    source = (
        "def public_fn():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception:\n"
        "        raise\n"                       # bare re-raise -> node.exc is None
        "def other():\n"
        "    raise errors.ValueError('x')\n"    # attribute callable -> not a bare Name
    )
    assert ec.contract_raises_in_source(source) == []


def test_dunder_methods_count_as_public_surface():
    """A built-in raise in a dunder (e.g. __init__) is on the public surface."""
    source = (
        "class Widget:\n"
        "    def __init__(self):\n"
        "        raise ValueError('bad config')\n"
    )
    rows = ec.contract_raises_in_source(source)
    assert ("Widget.__init__", "ValueError") in rows


# --- the builtin set is sane -------------------------------------------------


def test_builtin_exception_set_is_sane():
    for name in ("ValueError", "KeyError", "TypeError", "NotImplementedError", "RuntimeError"):
        assert name in ec.BUILTIN_EXCEPTION_NAMES, name
    # A VincioError subclass name is never a built-in (it is on-contract).
    assert "VincioError" not in ec.BUILTIN_EXCEPTION_NAMES
    assert "ConfigError" not in ec.BUILTIN_EXCEPTION_NAMES
    assert "InputError" not in ec.BUILTIN_EXCEPTION_NAMES


def test_private_modules_are_excluded():
    """Underscore-prefixed modules (e.g. the ed25519 backend) are out of scope."""
    modules = ec.public_modules()
    assert "vincio.security._ed25519" not in modules
    assert "vincio._error_contract" not in modules
    assert "vincio._surface" not in modules
    # Public modules are present.
    assert "vincio.core.app" in modules
    assert "vincio.core.error_catalog" in modules


# --- the standing-guard whitelist covers the decomposed app verb surface -----


def test_app_mixin_modules_stay_in_scope():
    """The private ContextApp verb mixins are whitelisted into the scan.

    ``vincio/core/_app_*.py`` (the ``_*Verbs`` classes ``ContextApp`` composes)
    would normally drop out of the public-module scan as underscore-prefixed;
    the app.py split must not silently un-guard the ``app.*`` verb bodies.
    """
    modules = ec.public_modules()
    for name in (
        "vincio.core._app_config",
        "vincio.core._app_knowledge",
        "vincio.core._app_settlement",
        "vincio.core._app_data",
    ):
        assert name in modules, name
    # The whitelist is surgical: other private modules stay out of scope.
    assert "vincio.security._ed25519" not in modules
    assert "vincio.tasks._flow" not in modules


def test_detector_bites_inside_a_mixin_module():
    """An injected bare built-in raise in a ``_*Verbs`` verb is still reported."""
    source = (
        "class _SettlementVerbs:\n"
        "    def settle(self):\n"
        "        raise ValueError('boom')\n"
        "    def _helper(self):\n"          # private methods stay encapsulated
        "        raise KeyError('x')\n"
    )
    assert ec.contract_raises_in_source(source, app_mixin=True) == [
        ("_SettlementVerbs.settle", "ValueError")
    ]
    # Without the whitelist flag the private class would encapsulate the raise —
    # the flag is exactly what keeps the moved verb surface guarded.
    assert ec.contract_raises_in_source(source) == []


def test_app_verb_gate_flags_injected_mixin_row():
    """The always-on verb gate treats a ``_*Verbs`` mixin verb as a ContextApp verb."""
    rows = [
        ("vincio.core._app_settlement", "_SettlementVerbs.settle", "ValueError"),
        # A non-Verbs private class in a mixin module is not the verb surface.
        ("vincio.core._app_support", "_AgentHandle.run", "ValueError"),
        # A Verbs-shaped qualname outside the whitelisted modules is not either.
        ("vincio.evals.judges", "_FakeVerbs.build", "ValueError"),
    ]
    assert ec.app_verb_violations(rows) == [
        "vincio.core._app_settlement._SettlementVerbs.settle raises ValueError"
    ]
