"""Tests for the API stability contract (vincio.stability)."""

from __future__ import annotations

import warnings

import pytest

import vincio
from vincio.stability import (
    API_VERSION,
    StabilityLevel,
    VincioDeprecationWarning,
    VincioExperimentalWarning,
    deprecated,
    deprecated_alias,
    experimental,
    public_api,
    stability_of,
)


def test_version_and_api_contract():
    assert vincio.__version__ == "3.19.0"
    # API_VERSION is the frozen public-API contract; it bumps only on a MAJOR
    # release, independent of the package patch level.
    assert API_VERSION == "3.0"


def test_public_api_is_stable_surface():
    names = public_api()
    # The frozen surface must include the headline entry points.
    for required in ("ContextApp", "Workflow", "Rail", "MemoryEngine", "OutputContract"):
        assert required in names
    # public_api() mirrors __all__ exactly.
    assert set(names) == set(vincio.__all__)
    # Every advertised name is actually importable from the package.
    for name in names:
        assert hasattr(vincio, name), name


def test_deprecated_function_warns_and_forwards():
    @deprecated(since="1.1", removed_in="2.0", alternative="new_fn")
    def old_fn(x):
        return x * 2

    with pytest.warns(VincioDeprecationWarning, match="removed in 2.0"):
        assert old_fn(3) == 6

    record = stability_of(old_fn)
    assert record["level"] is StabilityLevel.DEPRECATED
    assert record["removed_in"] == "2.0"
    assert record["alternative"] == "new_fn"
    assert "[DEPRECATED]" in (old_fn.__doc__ or "")


def test_deprecated_class_warns_on_instantiation():
    @deprecated(since="1.1", removed_in="2.0")
    class OldThing:
        def __init__(self, v):
            self.v = v

    with pytest.warns(VincioDeprecationWarning):
        obj = OldThing(7)
    assert obj.v == 7
    assert stability_of(OldThing)["level"] is StabilityLevel.DEPRECATED


def test_experimental_warns_once():
    @experimental(since="1.0", note="shape may change")
    def beta_fn():
        return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        beta_fn()
        beta_fn()
    exp = [w for w in caught if issubclass(w.category, VincioExperimentalWarning)]
    assert len(exp) == 1  # one-time per symbol
    assert stability_of(beta_fn)["level"] is StabilityLevel.EXPERIMENTAL


def test_deprecated_alias_forwards_to_target():
    def new_name(a, b):
        return a + b

    old_name = deprecated_alias(new_name, old_name="old_name", since="1.1", removed_in="2.0")
    with pytest.warns(VincioDeprecationWarning, match="old_name"):
        assert old_name(2, 3) == 5
    assert old_name.__name__ == "old_name"


def test_unmarked_symbol_is_stable_by_default():
    def plain():
        return 1

    assert stability_of(plain)["level"] is StabilityLevel.STABLE


def test_deprecation_warning_can_be_escalated_to_error():
    @deprecated(since="1.1", removed_in="2.0")
    def doomed():
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", VincioDeprecationWarning)
        with pytest.raises(VincioDeprecationWarning):
            doomed()
