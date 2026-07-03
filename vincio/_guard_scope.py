"""Shared scope rule for the standing-guard lints.

``ContextApp``'s verb surface is decomposed into private
``vincio/core/_app_*.py`` mixin modules (the ``_*Verbs`` classes the app
composes). Those files would normally drop out of the public-module scans as
underscore-prefixed, silently un-guarding the ``app.*`` verb bodies — so
:mod:`vincio._error_contract`, :mod:`vincio._observable_failure`, and
:mod:`vincio._assert_robustness` all deliberately keep them in scope through
this one predicate, defined once so the three guards can never drift apart on
what "the app surface" means.
"""

from __future__ import annotations

__all__ = ["is_app_mixin_module"]


def is_app_mixin_module(module: str) -> bool:
    """Whether ``module`` is a ContextApp verb-mixin module (``vincio.core._app_*``)."""
    return module.startswith("vincio.core._app_")
