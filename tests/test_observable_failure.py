"""Gate: observable failure (vincio._observable_failure + vincio.core.diagnostics).

A best-effort fallback that catches a broad ``Exception`` and continues is correct
policy, but one that swallows it *silently* — no re-raise, no log, no metric — hides
a real bug. The hardening line's 6.2 phase makes such a fallback observable
(:func:`vincio.core.diagnostics.note_suppressed` logs it on the dedicated
``vincio.suppressed`` channel and counts it) and adds a lint
(:mod:`vincio._observable_failure`) that holds the whole public tree to **zero**
unmarked silent swallows: every broad ``except`` must re-raise, record its failure,
or carry a justifying ``# noqa: BLE001``. This gate proves the tree is clean, proves
the detector bites, and proves the runtime helper logs-and-counts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from vincio import _observable_failure as of
from vincio.core import diagnostics

# --- the always-on, zero-tolerance gate --------------------------------------


def test_no_silent_swallows_tree_wide():
    """No public module swallows a broad exception silently.

    A new silent broad ``except`` must be a deliberate edit: record the failure
    (a logger call or ``note_suppressed``), re-raise, or add a justifying
    ``# noqa: BLE001`` with the reason. Reproduce offline with
    ``python -m vincio._observable_failure``.
    """
    problems = of.silent_swallows()
    assert problems == [], "silent broad-except swallows found:\n" + "\n".join(problems)


def test_silent_swallow_count_is_zero():
    assert of.silent_swallow_count() == 0


def test_public_modules_are_discovered_and_private_excluded():
    modules = of.public_modules()
    assert "vincio.core.runtime" in modules
    assert "vincio.core.app" in modules
    # Private tooling (this lint, the surface gate, the ed25519 backend) is out of scope.
    assert "vincio._observable_failure" not in modules
    assert "vincio._surface" not in modules
    assert "vincio.security._ed25519" not in modules


def test_app_mixin_modules_stay_in_scope():
    """The private ContextApp verb mixins (``vincio/core/_app_*.py``) are whitelisted.

    The app.py split moved the ``app.*`` verb bodies into underscore-prefixed
    modules; the standing guard deliberately keeps scanning them, so an injected
    silent swallow inside a mixin verb still fails the build.
    """
    modules = of.public_modules()
    assert "vincio.core._app_optimize" in modules
    assert "vincio.core._app_serving" in modules
    # The whitelist is surgical: other private modules stay out of scope.
    assert "vincio.tasks._flow" not in modules


def test_detector_bites_inside_a_mixin_class():
    """An injected silent swallow in a ``_*Verbs`` verb body is flagged.

    Together with :func:`test_app_mixin_modules_stay_in_scope` this proves the
    whitelisted module's violation reaches :func:`of.silent_swallows`.
    """
    source = (
        "class _ServingVerbs:\n"
        "    def add_tool(self):\n"
        "        try:\n"
        "            g()\n"
        "        except Exception:\n"
        "            pass\n"
    )
    rows = of.silent_swallows_in_source(source)
    assert [q for q, _ln, _d in rows] == ["_ServingVerbs.add_tool"]


def test_moved_marked_swallow_stays_scanned():
    """The one reviewed swallow (``_score_online``) moved into the optimize mixin;
    the whitelist keeps its module scanned and its ``# noqa: BLE001`` still holds."""
    import vincio

    path = Path(vincio.__file__).resolve().parent / "core" / "_app_optimize.py"
    source = path.read_text(encoding="utf-8")
    assert "# noqa: BLE001" in source
    assert of.silent_swallows_in_source(source) == []


# --- the detector bites on a silent swallow ----------------------------------


def test_detector_flags_bare_pass():
    """A broad ``except`` whose body just passes is a silent swallow (the gate bites)."""
    source = "def f():\n    try:\n        g()\n    except Exception:\n        pass\n"
    rows = of.silent_swallows_in_source(source)
    assert [(q, d) for q, _ln, d in rows] == [("f", "Exception")]


def test_detector_flags_silent_return():
    source = "def f():\n    try:\n        return g()\n    except Exception:\n        return None\n"
    rows = of.silent_swallows_in_source(source)
    assert [(q, d) for q, _ln, d in rows] == [("f", "Exception")]


def test_detector_flags_bare_except():
    source = "def f():\n    try:\n        g()\n    except:\n        pass\n"
    rows = of.silent_swallows_in_source(source)
    assert [d for _q, _ln, d in rows] == ["bare except"]


def test_detector_flags_base_exception():
    source = "def f():\n    try:\n        g()\n    except BaseException:\n        pass\n"
    rows = of.silent_swallows_in_source(source)
    assert [d for _q, _ln, d in rows] == ["BaseException"]


def test_detector_flags_tuple_containing_exception():
    source = "def f():\n    try:\n        g()\n    except (ValueError, Exception):\n        pass\n"
    rows = of.silent_swallows_in_source(source)
    assert [d for _q, _ln, d in rows] == ["Exception"]


def test_detector_flags_broad_contextlib_suppress():
    """``contextlib.suppress(Exception)`` is always silent and is held to the rule."""
    source = (
        "import contextlib\n"
        "def f():\n"
        "    with contextlib.suppress(Exception):\n"
        "        g()\n"
    )
    rows = of.silent_swallows_in_source(source)
    assert [d for _q, _ln, d in rows] == ["suppress(Exception)"]


def test_qualname_reports_enclosing_class_and_method():
    source = (
        "class Widget:\n"
        "    def build(self):\n"
        "        try:\n"
        "            g()\n"
        "        except Exception:\n"
        "            pass\n"
    )
    rows = of.silent_swallows_in_source(source)
    assert [q for q, _ln, _d in rows] == ["Widget.build"]


def test_module_level_swallow_reports_module_qualname():
    source = "try:\n    import maybe\nexcept Exception:\n    pass\n"
    rows = of.silent_swallows_in_source(source)
    assert [q for q, _ln, _d in rows] == ["<module>"]


def test_swallow_in_private_helper_is_still_flagged():
    """A silent swallow in a ``_helper`` hides a bug too — the scan covers private defs."""
    source = "def _helper():\n    try:\n        g()\n    except Exception:\n        return False\n"
    rows = of.silent_swallows_in_source(source)
    assert [q for q, _ln, _d in rows] == ["_helper"]


# --- the detector ignores observable, re-raising, narrow, and marked handlers -


def test_detector_ignores_logged_handler():
    source = (
        "def f():\n"
        "    try:\n"
        "        g()\n"
        "    except Exception:\n"
        "        logger.debug('boom', exc_info=True)\n"
    )
    assert of.silent_swallows_in_source(source) == []


def test_detector_ignores_note_suppressed_handler():
    source = (
        "def f():\n"
        "    try:\n"
        "        g()\n"
        "    except Exception:\n"
        "        note_suppressed('x.y')\n"
        "        return None\n"
    )
    assert of.silent_swallows_in_source(source) == []


def test_detector_ignores_event_emit_handler():
    source = (
        "def f(self):\n"
        "    try:\n"
        "        g()\n"
        "    except Exception:\n"
        "        self.events.emit('failed')\n"
    )
    assert of.silent_swallows_in_source(source) == []


def test_detector_ignores_reraise_handler():
    source = "def f():\n    try:\n        g()\n    except Exception:\n        raise\n"
    assert of.silent_swallows_in_source(source) == []


def test_detector_ignores_translated_reraise():
    source = (
        "def f():\n"
        "    try:\n"
        "        g()\n"
        "    except Exception as exc:\n"
        "        raise ConfigError('bad') from exc\n"
    )
    assert of.silent_swallows_in_source(source) == []


def test_detector_ignores_narrow_handler():
    source = "def f():\n    try:\n        g()\n    except ValueError:\n        pass\n"
    assert of.silent_swallows_in_source(source) == []


def test_detector_ignores_narrow_suppress():
    source = (
        "import contextlib\n"
        "def f():\n"
        "    with contextlib.suppress(ValueError):\n"
        "        g()\n"
    )
    assert of.silent_swallows_in_source(source) == []


def test_detector_respects_noqa_marker():
    source = (
        "def f():\n"
        "    try:\n"
        "        g()\n"
        "    except Exception:  # noqa: BLE001 - justified silence\n"
        "        pass\n"
    )
    assert of.silent_swallows_in_source(source) == []


def test_detector_respects_noqa_on_suppress():
    source = (
        "import contextlib\n"
        "def f():\n"
        "    with contextlib.suppress(Exception):  # noqa: BLE001 - probe only\n"
        "        g()\n"
    )
    assert of.silent_swallows_in_source(source) == []


# --- the runtime helper logs and counts --------------------------------------


def test_note_suppressed_increments_counter():
    diagnostics.reset_suppressed_failures()
    try:
        raise RuntimeError("boom")
    except Exception:
        diagnostics.note_suppressed("test.counter")
    assert diagnostics.suppressed_failure_counts().get("test.counter") == 1


def test_note_suppressed_aggregates_by_label():
    diagnostics.reset_suppressed_failures()
    for _ in range(3):
        try:
            raise RuntimeError("boom")
        except Exception:
            diagnostics.note_suppressed("test.repeat")
    assert diagnostics.suppressed_failure_counts()["test.repeat"] == 3


def test_note_suppressed_logs_on_dedicated_channel(caplog):
    diagnostics.reset_suppressed_failures()
    with caplog.at_level(logging.DEBUG, logger=diagnostics.SUPPRESSED_LOGGER_NAME):
        try:
            raise RuntimeError("boom")
        except Exception:
            diagnostics.note_suppressed("test.logged")
    messages = [r.getMessage() for r in caplog.records if r.name == diagnostics.SUPPRESSED_LOGGER_NAME]
    assert any("test.logged" in m for m in messages)
    # The active exception's traceback is captured (exc_info), so the symptom is traceable.
    assert any(r.exc_info is not None for r in caplog.records if r.name == diagnostics.SUPPRESSED_LOGGER_NAME)


def test_note_suppressed_honors_level_and_detail(caplog):
    diagnostics.reset_suppressed_failures()
    with caplog.at_level(logging.WARNING, logger=diagnostics.SUPPRESSED_LOGGER_NAME):
        try:
            raise RuntimeError("boom")
        except Exception:
            diagnostics.note_suppressed("test.warned", level=logging.WARNING, detail="RuntimeError")
    record = next(r for r in caplog.records if r.name == diagnostics.SUPPRESSED_LOGGER_NAME)
    assert record.levelno == logging.WARNING
    assert "RuntimeError" in record.getMessage()


def test_suppressed_failure_counts_returns_a_copy():
    diagnostics.reset_suppressed_failures()
    try:
        raise RuntimeError("boom")
    except Exception:
        diagnostics.note_suppressed("test.copy")
    snapshot = diagnostics.suppressed_failure_counts()
    snapshot["test.copy"] = 999
    assert diagnostics.suppressed_failure_counts()["test.copy"] == 1


def test_reset_clears_the_counters():
    try:
        raise RuntimeError("boom")
    except Exception:
        diagnostics.note_suppressed("test.cleared")
    diagnostics.reset_suppressed_failures()
    assert diagnostics.suppressed_failure_counts() == {}


# --- a converted site is wired end-to-end ------------------------------------


def test_converted_fallback_is_observable_end_to_end():
    """A real converted fallback logs-and-counts through ``note_suppressed``.

    ``DataEngagement._safe_bind`` re-executes a data-binder and treats any failure
    as not-bound; the failure is now counted under its label and the method still
    returns ``False`` — the fallback stays best-effort *and* observable.
    """
    from vincio.data.engagement import DataEngagement

    def boom_binder(catalog):
        raise RuntimeError("re-bind failed")

    diagnostics.reset_suppressed_failures()
    assert DataEngagement._safe_bind(boom_binder, None) is False
    assert diagnostics.suppressed_failure_counts().get("data.engagement.rebind") == 1


@pytest.fixture(autouse=True)
def _reset_counters_after_each_test():
    yield
    diagnostics.reset_suppressed_failures()
