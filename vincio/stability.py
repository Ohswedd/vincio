"""API stability: semantic-versioning guarantees and the deprecation policy.

From 1.0 onward Vincio follows `Semantic Versioning 2.0.0
<https://semver.org/spec/v2.0.0.html>`_ on its **public API** — everything
re-exported from the top-level :mod:`vincio` package (see :data:`vincio.__all__`)
plus the documented entry points of each subpackage. Within a major version:

* **PATCH** (``1.0.x``) — bug fixes only; no public behaviour changes.
* **MINOR** (``1.x.0``) — additive: new symbols, new optional parameters with
  defaults. Existing code keeps working.
* **MAJOR** (``x.0.0``) — may remove or change deprecated public API.

Anything not in the public surface — names prefixed with ``_``, modules under a
``_``-prefixed path, and symbols marked :func:`experimental` — may change in any
release. The deprecation contract is mechanical, not just documented:

* A public symbol is removed only after at least one MINOR release in which it
  is marked with :func:`deprecated` and emits :class:`VincioDeprecationWarning`.
* The warning names the version it was deprecated in, the version it is
  scheduled for removal in, and the replacement.
* :func:`stability_of` lets tools and tests introspect any symbol's contract.

Use :func:`deprecated` / :func:`experimental` to annotate functions and
classes, and :func:`deprecated_alias` to keep an old name working while steering
callers to the new one.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from enum import StrEnum
from typing import Any, TypeVar

__all__ = [
    "API_VERSION",
    "StabilityLevel",
    "VincioDeprecationWarning",
    "VincioExperimentalWarning",
    "deprecated",
    "experimental",
    "deprecated_alias",
    "stability_of",
    "public_api",
]

# The public-API contract version. Bumped only on a MAJOR release; it is the
# promise SemVer is applied against, independent of the package patch level.
# The public surface is frozen and evolves only under the mechanical
# deprecation runway described above. 5.0 is the second long-term-support major:
# it re-freezes the surface expanded additively across the 4.x data & analytics
# plane (4.1–5.0) and declares that plane feature-complete and frozen.
API_VERSION = "5.0"

_STABILITY_ATTR = "__vincio_stability__"

F = TypeVar("F", bound=Callable[..., Any])


class StabilityLevel(StrEnum):
    """Stability contract for a public symbol."""

    STABLE = "stable"
    """Covered by SemVer guarantees; removed only across a MAJOR bump."""

    BETA = "beta"
    """Public and supported, but signature may still change in a MINOR."""

    EXPERIMENTAL = "experimental"
    """No stability guarantee; may change or vanish in any release."""

    DEPRECATED = "deprecated"
    """Scheduled for removal; emits a warning and names its replacement."""


class VincioDeprecationWarning(DeprecationWarning):
    """Emitted when a deprecated Vincio API is used.

    Subclasses :class:`DeprecationWarning` so it is silenced by default outside
    ``__main__`` and test runs, yet can be turned into an error in CI with
    ``warnings.simplefilter("error", VincioDeprecationWarning)``.
    """


class VincioExperimentalWarning(UserWarning):
    """Emitted on first use of an :func:`experimental` API."""


def _stability_record(
    level: StabilityLevel,
    *,
    since: str | None = None,
    removed_in: str | None = None,
    alternative: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "level": level,
        "since": since,
        "removed_in": removed_in,
        "alternative": alternative,
        "note": note,
    }


def _attach(obj: Any, record: dict[str, Any]) -> None:
    try:
        setattr(obj, _STABILITY_ATTR, record)
    except (AttributeError, TypeError):  # pragma: no cover - exotic objects
        pass


def deprecated(
    *,
    since: str,
    removed_in: str,
    alternative: str | None = None,
) -> Callable[[F], F]:
    """Mark a function or class as deprecated.

    Emits a :class:`VincioDeprecationWarning` on each call/instantiation that
    names ``since``, ``removed_in``, and the suggested ``alternative``. The
    contract is also recorded on the object for :func:`stability_of`.

    >>> @deprecated(since="1.1", removed_in="2.0", alternative="new_fn")
    ... def old_fn(): ...
    """

    def decorate(obj: F) -> F:
        message = (
            f"{obj.__name__}() is deprecated since Vincio {since} and will be "
            f"removed in {removed_in}."
        )
        if alternative:
            message += f" Use {alternative} instead."
        record = _stability_record(
            StabilityLevel.DEPRECATED,
            since=since,
            removed_in=removed_in,
            alternative=alternative,
        )

        if isinstance(obj, type):
            orig_init = obj.__init__  # type: ignore[misc]  # intentional __init__ wrapping

            @functools.wraps(orig_init)
            def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(message, VincioDeprecationWarning, stacklevel=2)
                orig_init(self, *args, **kwargs)

            obj.__init__ = __init__  # type: ignore[misc]
            _attach(obj, record)
            obj.__doc__ = f"[DEPRECATED] {message}\n\n{obj.__doc__ or ''}".rstrip()
            return obj

        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, VincioDeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        _attach(wrapper, record)
        wrapper.__doc__ = f"[DEPRECATED] {message}\n\n{wrapper.__doc__ or ''}".rstrip()
        return wrapper  # type: ignore[return-value]

    return decorate


def experimental(*, since: str, note: str | None = None) -> Callable[[F], F]:
    """Mark a function or class as experimental (no stability guarantee).

    Emits a one-time :class:`VincioExperimentalWarning` per process per symbol,
    so it is visible without being noisy.
    """

    def decorate(obj: F) -> F:
        message = (
            f"{obj.__name__}() is experimental (since Vincio {since}); its API "
            f"may change in any release."
        )
        if note:
            message += f" {note}"
        record = _stability_record(StabilityLevel.EXPERIMENTAL, since=since, note=note)
        warned = {"done": False}

        def _warn() -> None:
            if not warned["done"]:
                warnings.warn(message, VincioExperimentalWarning, stacklevel=3)
                warned["done"] = True

        if isinstance(obj, type):
            orig_init = obj.__init__  # type: ignore[misc]  # intentional __init__ wrapping

            @functools.wraps(orig_init)
            def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
                _warn()
                orig_init(self, *args, **kwargs)

            obj.__init__ = __init__  # type: ignore[misc]
            _attach(obj, record)
            return obj

        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _warn()
            return obj(*args, **kwargs)

        _attach(wrapper, record)
        return wrapper  # type: ignore[return-value]

    return decorate


def deprecated_alias(
    target: Callable[..., Any],
    *,
    old_name: str,
    since: str,
    removed_in: str,
) -> Callable[..., Any]:
    """Build a forwarding wrapper for a renamed public symbol.

    The returned callable forwards to ``target`` and emits a
    :class:`VincioDeprecationWarning` pointing at the new name. Bind it to the
    old name in the module that used to export it::

        new_name = _impl
        old_name = deprecated_alias(new_name, old_name="old_name",
                                    since="1.1", removed_in="2.0")
    """

    message = (
        f"{old_name}() is deprecated since Vincio {since} and will be removed in "
        f"{removed_in}. Use {getattr(target, '__name__', target)!r} instead."
    )

    @functools.wraps(target)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(message, VincioDeprecationWarning, stacklevel=2)
        return target(*args, **kwargs)

    wrapper.__name__ = old_name
    _attach(
        wrapper,
        _stability_record(
            StabilityLevel.DEPRECATED,
            since=since,
            removed_in=removed_in,
            alternative=getattr(target, "__name__", None),
        ),
    )
    return wrapper


def stability_of(obj: Any) -> dict[str, Any]:
    """Return the stability record for ``obj``.

    Defaults to ``StabilityLevel.STABLE`` for any public symbol without an
    explicit marker — the contract is "stable unless stated otherwise".
    """

    record = getattr(obj, _STABILITY_ATTR, None)
    if record is None:
        return _stability_record(StabilityLevel.STABLE)
    return dict(record)


def public_api() -> tuple[str, ...]:
    """Return the frozen public-API surface (the top-level ``__all__``).

    This is the exact set of names SemVer guarantees apply to. Tests use it to
    detect accidental additions/removals to the public surface.
    """

    import vincio

    return tuple(vincio.__all__)
