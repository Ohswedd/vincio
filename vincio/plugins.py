"""Entry-point plugin discovery — the stable, versioned plugin contract.

Third-party packages extend Vincio by advertising entry points; installing the
package is all it takes for the extension to register itself. Each kind maps to
one entry-point group, and the loaded object's shape is the contract:

==================  ====================  ==================================
Entry-point group   Kind                  Loaded object
==================  ====================  ==================================
``vincio.providers``   provider           a provider factory ``(config) -> ModelProvider``
``vincio.embedders``   embedder           an :class:`~vincio.retrieval.embeddings.Embedder`
``vincio.stores``      store              a vector-store factory
``vincio.connectors``  connector          a connector factory ``(**opts) -> Connector``
``vincio.chunkers``    chunker            a chunking strategy ``(document, size, overlap)``
``vincio.rerankers``   reranker           a reranker factory ``(**opts) -> Reranker``
``vincio.metrics``     metric             a metric ``(case, output) -> MetricResult``
``vincio.judges``      judge              a judge factory ``(**opts) -> Judge``
``vincio.packs``       pack               a :class:`~vincio.packs.Pack` (or factory)
==================  ====================  ==================================

The contract is versioned by :data:`PLUGIN_API_VERSION`. A plugin distribution
may declare the major it targets with a ``vincio.plugins`` entry point named
``api_version`` (resolving to a version string); a major mismatch is reported
and the plugin is **not** loaded, so an incompatible plugin fails loud, not
silently.

    from vincio.plugins import installed_plugins, load_plugins

    for p in installed_plugins():
        print(p.name, p.kind, p.distribution, p.status)
    load_plugins()                 # register every compatible plugin

The ``vincio plugins list`` CLI prints the same table. ``providers`` /
``embedders`` / ``stores`` self-register at their own first use; the other kinds
register through :func:`load_plugins` (which :func:`connect` and
:func:`~vincio.packs.load_pack` call on a name miss, so an installed connector
or pack simply works).
"""

from __future__ import annotations

import importlib.metadata as _md
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

__all__ = [
    "PLUGIN_API_VERSION",
    "PLUGIN_GROUPS",
    "PluginInfo",
    "discover_plugins",
    "installed_plugins",
    "load_plugins",
    "ensure_loaded",
]

# The plugin-API contract version. Bumped only on a breaking change to a group's
# expected object shape; the major is what compatibility is checked against.
PLUGIN_API_VERSION = "1.0"

# Group used by a distribution to declare the plugin-API version it targets.
_COMPAT_GROUP = "vincio.plugins"
_COMPAT_NAME = "api_version"

# Entry-point group -> human-readable kind. Order is the display order.
PLUGIN_GROUPS: dict[str, str] = {
    "vincio.providers": "provider",
    "vincio.embedders": "embedder",
    "vincio.stores": "store",
    "vincio.connectors": "connector",
    "vincio.chunkers": "chunker",
    "vincio.rerankers": "reranker",
    "vincio.metrics": "metric",
    "vincio.judges": "judge",
    "vincio.packs": "pack",
}

# Groups whose objects this module actively registers in :func:`load_plugins`.
# providers/embedders/stores self-register at their own first-use sites; we only
# report them so they are visible alongside the rest.
_SELF_REGISTERING = {"vincio.providers", "vincio.embedders", "vincio.stores"}


class PluginInfo(BaseModel):
    """A discovered plugin entry point and its registration status."""

    name: str
    group: str
    kind: str
    distribution: str = ""
    version: str = ""
    declared_api: str | None = None
    # available | loaded | incompatible | error
    status: str = "available"
    detail: str = ""


@dataclass
class _EP:
    """A normalized entry point, decoupled from importlib.metadata specifics."""

    name: str
    group: str
    distribution: str = ""
    version: str = ""
    load: Callable[[], Any] = field(default=lambda: None)


def _iter_entry_points() -> Iterator[_EP]:
    """Yield every installed entry point paired with its distribution metadata.

    Iterating distributions (rather than ``entry_points(group=...)``) gives the
    distribution name and version for free across Python 3.11–3.13.
    """
    seen: set[tuple[str, str, str]] = set()
    try:
        distributions = list(_md.distributions())
    except Exception:  # pragma: no cover - importlib.metadata edge cases
        return
    for dist in distributions:
        try:
            dist_name = dist.name or ""
            version = dist.version or ""
            eps = list(dist.entry_points)
        except Exception:  # noqa: BLE001 - a broken dist must not break discovery
            continue
        for ep in eps:
            # A package installed twice (e.g. editable + wheel) can list the same
            # entry point under two distributions; de-dup so it appears once.
            key = (ep.group, ep.name, dist_name)
            if key in seen:
                continue
            seen.add(key)
            yield _EP(
                name=ep.name,
                group=ep.group,
                distribution=dist_name,
                version=version,
                load=ep.load,
            )


def _major(version: str) -> str:
    return (version or "").split(".", 1)[0].strip()


def _declared_api_versions(eps: list[_EP]) -> dict[str, str]:
    """Map distribution -> declared plugin-API version (best-effort)."""
    declared: dict[str, str] = {}
    for ep in eps:
        if ep.group == _COMPAT_GROUP and ep.name == _COMPAT_NAME:
            try:
                value = ep.load()
            except Exception:  # noqa: BLE001 - never let a probe break discovery
                continue
            declared[ep.distribution] = str(value() if callable(value) else value)
    return declared


def discover_plugins(
    *,
    groups: Iterable[str] | None = None,
    entry_points: Iterable[_EP] | None = None,
) -> list[PluginInfo]:
    """List installed Vincio plugins without registering them.

    ``groups`` restricts the report to specific entry-point groups. Pass
    ``entry_points`` (an iterable of normalized entry points) to test against a
    fixed set instead of the live environment. The target objects are **not**
    loaded — only compatibility is resolved — so this is safe and cheap.
    """
    eps = list(entry_points) if entry_points is not None else list(_iter_entry_points())
    declared = _declared_api_versions(eps)
    wanted = set(groups) if groups is not None else set(PLUGIN_GROUPS)
    infos: list[PluginInfo] = []
    for ep in eps:
        if ep.group not in PLUGIN_GROUPS or ep.group not in wanted:
            continue
        api = declared.get(ep.distribution)
        info = PluginInfo(
            name=ep.name,
            group=ep.group,
            kind=PLUGIN_GROUPS[ep.group],
            distribution=ep.distribution,
            version=ep.version,
            declared_api=api,
        )
        if api is not None and _major(api) != _major(PLUGIN_API_VERSION):
            info.status = "incompatible"
            info.detail = (
                f"targets plugin API {api}; this Vincio provides {PLUGIN_API_VERSION}"
            )
        infos.append(info)
    group_order = list(PLUGIN_GROUPS)
    infos.sort(key=lambda i: (group_order.index(i.group), i.name))
    return infos


def installed_plugins() -> list[PluginInfo]:
    """All installed Vincio plugins across every group (alias for discovery)."""
    return discover_plugins()


# -- registration --------------------------------------------------------------


def _register_connector(name: str, obj: Any) -> None:
    from .connectors.base import register_connector

    register_connector(name)(obj)


def _register_chunker(name: str, obj: Any) -> None:
    from .retrieval.chunking import CHUNKERS

    CHUNKERS[name] = obj


def _register_reranker(name: str, obj: Any) -> None:
    from .retrieval.rerankers import register_reranker

    register_reranker(name, obj)


def _register_metric(name: str, obj: Any) -> None:
    from .evals.metrics import METRICS

    METRICS[name] = obj


def _register_judge(name: str, obj: Any) -> None:
    from .evals.judges import register_judge

    register_judge(name, obj)


def _register_pack(name: str, obj: Any) -> None:
    from .packs import Pack, register_pack

    pack = obj() if (callable(obj) and not isinstance(obj, Pack)) else obj
    if not isinstance(pack, Pack):
        raise TypeError(f"pack plugin {name!r} did not resolve to a Pack")
    register_pack(pack)


_REGISTRARS: dict[str, Callable[[str, Any], None]] = {
    "vincio.connectors": _register_connector,
    "vincio.chunkers": _register_chunker,
    "vincio.rerankers": _register_reranker,
    "vincio.metrics": _register_metric,
    "vincio.judges": _register_judge,
    "vincio.packs": _register_pack,
}

# Track what we have already loaded so load is idempotent and cheap to re-call.
_loaded_keys: set[tuple[str, str, str]] = set()
_loaded_groups: set[str] = set()


def load_plugins(
    *,
    groups: Iterable[str] | None = None,
    entry_points: Iterable[_EP] | None = None,
) -> list[PluginInfo]:
    """Register every compatible installed plugin into its registry.

    Idempotent: a plugin already registered is left untouched and reported as
    ``loaded``. Incompatible plugins (declared-API major mismatch) are skipped.
    A plugin that fails to import or register is isolated and reported as
    ``error`` — one broken plugin never breaks the rest. Returns the status of
    every plugin considered.
    """
    eps_list = list(entry_points) if entry_points is not None else list(_iter_entry_points())
    by_key = {(ep.group, ep.name, ep.distribution): ep for ep in eps_list}
    infos = discover_plugins(groups=groups, entry_points=eps_list)
    for info in infos:
        if info.group not in _REGISTRARS:
            continue  # self-registering kinds are report-only here
        if info.status == "incompatible":
            continue
        key = (info.group, info.name, info.distribution)
        if key in _loaded_keys:
            info.status = "loaded"
            continue
        ep = by_key.get(key)
        if ep is None:  # pragma: no cover - discovery/lookup mismatch
            continue
        try:
            obj = ep.load()
            _REGISTRARS[info.group](info.name, obj)
        except Exception as exc:  # noqa: BLE001 - isolate a broken plugin
            info.status = "error"
            info.detail = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                f"failed to load plugin {info.name!r} from {info.group!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        _loaded_keys.add(key)
        info.status = "loaded"
    return infos


def ensure_loaded(group: str) -> None:
    """Load plugins for ``group`` once (used on a registry-name miss)."""
    if group in _loaded_groups:
        return
    _loaded_groups.add(group)
    if group not in _REGISTRARS:
        return
    try:
        load_plugins(groups=[group])
    except Exception:  # noqa: BLE001 - discovery must never break the caller
        pass
