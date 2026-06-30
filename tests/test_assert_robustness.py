"""Gate: ``-O`` robustness (vincio._assert_robustness).

Python strips every ``assert`` under ``python -O``. An ``assert`` that carries real
control-flow weight therefore *vanishes* in an optimized deployment, turning a caught
invariant into an opaque downstream error. The hardening line's 6.5 phase replaces
each such load-bearing ``assert`` with an explicit guard that raises the appropriate
:class:`~vincio.core.errors.VincioError`, and adds a lint
(:mod:`vincio._assert_robustness`) that holds the whole public tree to **zero**
unmarked ``assert``s: a genuine never-happens invariant kept as an ``assert`` carries
a justifying ``# noqa: S101``. This gate proves the tree is clean, proves the
detector bites, and proves two representative converted guards fire under the very
condition their stripped ``assert`` used to cover.
"""

from __future__ import annotations

import asyncio

import pytest

from vincio import _assert_robustness as ar

# --- the always-on, zero-tolerance gate --------------------------------------


def test_no_unmarked_asserts_tree_wide():
    """No public module carries an ``assert`` that vanishes under ``python -O`` unmarked.

    A new ``assert`` must be a deliberate edit: convert it to a guard that raises a
    ``VincioError``, or add a justifying ``# noqa: S101`` for a genuine never-happens
    invariant. Reproduce offline with ``python -m vincio._assert_robustness``.
    """
    problems = ar.unmarked_asserts()
    assert problems == [], "unmarked asserts found:\n" + "\n".join(problems)


def test_unmarked_assert_count_is_zero():
    assert ar.unmarked_assert_count() == 0


def test_marked_invariants_remain():
    """The genuine never-happens invariants are kept (and marked), not deleted."""
    assert ar.marked_assert_count() > 0


def test_public_modules_are_discovered_and_private_excluded():
    modules = ar.public_modules()
    assert "vincio.core.runtime" in modules
    assert "vincio.mcp.transport" in modules
    # Private tooling (this lint, the other gates, the ed25519 backend) is out of scope.
    assert "vincio._assert_robustness" not in modules
    assert "vincio._observable_failure" not in modules
    assert "vincio.security._ed25519" not in modules


# --- the detector bites on an unmarked assert --------------------------------


def test_detector_flags_bare_assert():
    """A bare ``assert`` with no marker is reported (the gate bites)."""
    source = "def f(x):\n    assert x is not None\n    return x\n"
    rows = ar.unmarked_asserts_in_source(source)
    assert [(q, ln) for q, ln in rows] == [("f", 2)]


def test_detector_flags_assert_in_private_helper():
    """A stripped ``assert`` in a ``_helper`` breaks under ``-O`` too — private defs are scanned."""
    source = "def _helper(x):\n    assert x\n"
    rows = ar.unmarked_asserts_in_source(source)
    assert [q for q, _ln in rows] == ["_helper"]


def test_qualname_reports_enclosing_class_and_method():
    source = "class Widget:\n    def build(self, x):\n        assert x is not None\n"
    rows = ar.unmarked_asserts_in_source(source)
    assert [q for q, _ln in rows] == ["Widget.build"]


def test_module_level_assert_reports_module_qualname():
    source = "import os\nassert os is not None\n"
    rows = ar.unmarked_asserts_in_source(source)
    assert [q for q, _ln in rows] == ["<module>"]


# --- the detector respects the marker and ignores non-asserts ----------------


def test_detector_respects_noqa_marker():
    source = "def f(x):\n    assert x is not None  # noqa: S101 - invariant\n    return x\n"
    assert ar.unmarked_asserts_in_source(source) == []


def test_detector_respects_marker_on_multiline_assert():
    """The marker on the ``assert`` line covers a message that spans continuation lines."""
    source = (
        "def f(a, b):\n"
        "    assert a == b, (  # noqa: S101 - intentional\n"
        "        f'mismatch: {a} != {b}'\n"
        "    )\n"
    )
    assert ar.unmarked_asserts_in_source(source) == []


def test_detector_respects_bare_noqa():
    source = "def f(x):\n    assert x  # noqa\n"
    assert ar.unmarked_asserts_in_source(source) == []


def test_detector_ignores_assert_inside_docstring():
    """An ``assert`` that appears only in a docstring example is a string, not a statement."""
    source = 'def f():\n    """Example::\n\n        assert result.verify()\n    """\n    return 1\n'
    assert ar.unmarked_asserts_in_source(source) == []


def test_asserts_in_source_reports_marked_flag():
    source = (
        "def f(x, y):\n"
        "    assert x is not None  # noqa: S101 - kept\n"
        "    assert y is not None\n"
    )
    rows = ar.asserts_in_source(source)
    assert rows == [("f", 2, True), ("f", 3, False)]


# --- representative converted guards fire under the -O condition --------------


def test_compiler_footprint_guard_raises_without_ceiling():
    """``ContextCompiler._enforce_footprint`` raises ``ContextCompileError`` (not a
    stripped ``AssertionError``) when invoked without a ``max_resident_bytes`` ceiling
    — the ROADMAP's flagship load-bearing assert, now an explicit guard."""
    from vincio.context.compiler import ContextCompiler
    from vincio.core.errors import ContextCompileError, VincioError

    compiler = ContextCompiler()
    assert compiler.options.max_resident_bytes is None
    with pytest.raises(ContextCompileError) as excinfo:
        compiler._enforce_footprint([], [], False, [])
    # The contract: it is catchable as the whole family, with a stable typed code.
    assert isinstance(excinfo.value, VincioError)
    assert excinfo.value.code == "CONTEXT_COMPILE"


def test_mcp_transport_guard_raises_when_not_started():
    """``StdioTransport`` raises ``MCPError`` (not a stripped ``AssertionError``) when
    asked to answer a server request before the subprocess (and its stdin pipe) exist."""
    from vincio.core.errors import VincioError
    from vincio.mcp.protocol import MCPError
    from vincio.mcp.transport import StdioTransport

    transport = StdioTransport(["true"])
    assert transport._proc is None
    with pytest.raises(MCPError):
        asyncio.run(transport._answer_server_request({"method": "ping", "id": 1}))
    assert issubclass(MCPError, VincioError)
