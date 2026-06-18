"""Live model discovery reconciled into the registry.

Optional runtime discovery from a provider's model-list endpoint
(:meth:`~vincio.providers.base.ModelProvider.list_models`), reconciled into the
:class:`~vincio.providers.registry.ModelRegistry` so a local or gateway
deployment can stay current without a release. Offline-safe: a provider that
exposes no list endpoint returns an empty list, leaving the shipped catalog
authoritative.
"""

from __future__ import annotations

from typing import Any

__all__ = ["discover_models"]


async def discover_models(
    provider: Any,
    *,
    registry: Any | None = None,
    mark_missing_deprecated: bool = False,
    as_of: Any = None,
) -> dict[str, list[str]]:
    """Fetch *provider*'s live model list and reconcile it into the registry.

    Returns the reconcile summary (``added`` / ``updated`` / ``deprecated_missing``).
    With ``mark_missing_deprecated`` a catalog model of this provider that has
    vanished from the live list is flagged deprecated — providers retire models
    silently, and this surfaces it as a rotation signal.
    """
    from .registry import default_model_registry

    registry = registry or default_model_registry()
    profiles = await provider.list_models()
    return registry.reconcile(
        profiles,
        provider=getattr(provider, "name", None),
        mark_missing_deprecated=mark_missing_deprecated,
        as_of=as_of,
    )
