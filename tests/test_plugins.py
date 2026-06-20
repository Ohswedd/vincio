"""Entry-point plugin discovery — exercised against injected entry points so the
suite never needs a real third-party distribution installed."""

from __future__ import annotations

import pytest

from vincio.plugins import (
    _EP,
    PLUGIN_API_VERSION,
    PLUGIN_GROUPS,
    discover_plugins,
    ensure_loaded,
    installed_plugins,
    load_plugins,
)


@pytest.fixture(autouse=True)
def _isolate_plugin_state():
    """Plugin load-state is process-global; snapshot and reset it around each test
    so order (and examples that load plugins) can't leak in."""
    import vincio.plugins as pl

    keys, groups = set(pl._loaded_keys), set(pl._loaded_groups)
    pl._loaded_keys.clear()
    pl._loaded_groups.clear()
    try:
        yield
    finally:
        pl._loaded_keys.clear()
        pl._loaded_keys.update(keys)
        pl._loaded_groups.clear()
        pl._loaded_groups.update(groups)


def _connector_factory(**opts):
    from vincio.core.types import Document

    class _C:
        name = "acme"

        async def load(self):
            return [Document(text="acme")]

    return _C()


def _metric(case, output):
    from vincio.evals.metrics import MetricResult

    return MetricResult(name="acme_metric", value=1.0)


def _pack():
    from vincio.packs import Pack

    return Pack(name="acme_pack", description="An installed pack", role="r", objective="o")


def _eps(api_version: str = "1.0", dist: str = "acme-vincio"):
    return [
        _EP("acme", "vincio.connectors", dist, "0.1.0", lambda: _connector_factory),
        _EP("acme_metric", "vincio.metrics", dist, "0.1.0", lambda: _metric),
        _EP("acme_pack", "vincio.packs", dist, "0.1.0", lambda: _pack),
        _EP("api_version", "vincio.plugins", dist, "0.1.0", lambda: api_version),
    ]


def test_contract_is_versioned_and_groups_are_stable():
    assert PLUGIN_API_VERSION  # documented contract version
    # The headline kinds are all part of the contract.
    kinds = set(PLUGIN_GROUPS.values())
    assert {"provider", "connector", "metric", "chunker", "reranker", "judge", "pack"} <= kinds


def test_discover_does_not_load_targets():
    # A loader that raises proves discovery never imports the target object.
    eps = [_EP("boom", "vincio.connectors", "boom-dist", "1.0.0", lambda: (_ for _ in ()).throw(RuntimeError("loaded!")))]
    infos = discover_plugins(entry_points=eps)
    assert [i.name for i in infos] == ["boom"]
    assert infos[0].status == "available"
    assert infos[0].kind == "connector"


def test_discover_reports_incompatible_major():
    eps = _eps(api_version="2.0", dist="future-dist")
    infos = {i.name: i for i in discover_plugins(entry_points=eps)}
    assert infos["acme"].status == "incompatible"
    assert "2.0" in infos["acme"].detail


def test_load_registers_into_registries():
    from vincio.connectors import CONNECTORS
    from vincio.evals.metrics import METRICS
    from vincio.packs import available_packs
    from vincio.packs.base import _CACHE

    eps = _eps()
    try:
        infos = {i.name: i for i in load_plugins(entry_points=eps)}
        assert infos["acme"].status == "loaded"
        assert infos["acme_metric"].status == "loaded"
        assert infos["acme_pack"].status == "loaded"
        assert "acme" in CONNECTORS
        assert "acme_metric" in METRICS
        assert "acme_pack" in available_packs()
    finally:
        CONNECTORS.pop("acme", None)
        METRICS.pop("acme_metric", None)
        _CACHE.pop("acme_pack", None)


def test_incompatible_plugin_is_not_loaded():
    from vincio.retrieval.chunking import CHUNKERS

    eps = [
        _EP("future_chunker", "vincio.chunkers", "future", "9.0.0", lambda: (_ for _ in ()).throw(RuntimeError("nope"))),
        _EP("api_version", "vincio.plugins", "future", "9.0.0", lambda: "2.0"),
    ]
    infos = {i.name: i for i in load_plugins(entry_points=eps)}
    assert infos["future_chunker"].status == "incompatible"
    assert "future_chunker" not in CHUNKERS


def test_broken_plugin_is_isolated():
    eps = [_EP("broken", "vincio.connectors", "broken-dist", "1.0.0", lambda: (_ for _ in ()).throw(ValueError("kaboom")))]
    with pytest.warns(RuntimeWarning):
        infos = {i.name: i for i in load_plugins(entry_points=eps)}
    assert infos["broken"].status == "error"
    assert "kaboom" in infos["broken"].detail


def test_load_is_idempotent():
    from vincio.connectors import CONNECTORS

    eps = [_EP("idem", "vincio.connectors", "idem-dist", "1.0.0", lambda: _connector_factory)]
    try:
        first = {i.name: i.status for i in load_plugins(entry_points=eps)}
        second = {i.name: i.status for i in load_plugins(entry_points=eps)}
        assert first["idem"] == "loaded"
        assert second["idem"] == "loaded"  # not re-registered, still reported loaded
    finally:
        CONNECTORS.pop("idem", None)


def test_pack_plugin_resolves_through_load_pack():
    # An installed pack plugin must resolve via the public load_pack() on a miss.
    import vincio.plugins as plugins_mod
    from vincio.connectors import connect
    from vincio.packs import load_pack
    from vincio.packs.base import _CACHE

    eps = _eps(dist="lazy-dist")
    # Simulate "installed environment" by patching the live entry-point iterator.
    original = plugins_mod._iter_entry_points
    plugins_mod._iter_entry_points = lambda: iter(eps)  # type: ignore[assignment]
    plugins_mod._loaded_groups.discard("vincio.packs")
    plugins_mod._loaded_groups.discard("vincio.connectors")
    try:
        pack = load_pack("acme_pack")
        assert pack.name == "acme_pack"
        # connector plugin also resolves lazily through connect()
        conn = connect("acme")
        assert conn.name == "acme"
    finally:
        plugins_mod._iter_entry_points = original  # type: ignore[assignment]
        _CACHE.pop("acme_pack", None)
        from vincio.connectors import CONNECTORS

        CONNECTORS.pop("acme", None)
        plugins_mod._loaded_groups.discard("vincio.packs")
        plugins_mod._loaded_groups.discard("vincio.connectors")
        plugins_mod._loaded_keys.clear()


def test_installed_plugins_smoke():
    # Against the real environment there may be zero plugins; just ensure it runs.
    assert isinstance(installed_plugins(), list)


def test_ensure_loaded_is_safe_to_call():
    ensure_loaded("vincio.connectors")  # idempotent, never raises
    ensure_loaded("vincio.connectors")
