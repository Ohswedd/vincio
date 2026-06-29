# Plugins: extend Vincio from a separate package

Vincio is built on registries, providers, connectors, metrics, chunkers,
rerankers, judges, and packs are all looked up by name. The `vincio.plugins`
module turns those registries into a **discoverable, versioned plugin contract**:
a third-party package advertises entry points, and installing it is all it takes
for the extension to register itself. Nothing in your application code changes.

## The contract

Each extension kind maps to one entry-point group, and the shape of the loaded
object is the contract:

| Entry-point group | Kind | Loaded object |
|---|---|---|
| `vincio.providers` | provider | a provider factory `(config) -> ModelProvider` |
| `vincio.embedders` | embedder | an `Embedder` |
| `vincio.stores` | store | a vector-store factory |
| `vincio.connectors` | connector | a connector factory `(**opts) -> Connector` |
| `vincio.chunkers` | chunker | a chunking strategy `(document, size, overlap)` |
| `vincio.rerankers` | reranker | a reranker factory `(**opts) -> Reranker` |
| `vincio.metrics` | metric | a metric `(case, output) -> MetricResult` |
| `vincio.judges` | judge | a judge factory `(**opts) -> Judge` |
| `vincio.packs` | pack | a `Pack` (or a zero-arg factory returning one) |

The contract is versioned by `vincio.plugins.PLUGIN_API_VERSION`. It bumps only
on a breaking change to a group's expected object shape, never for the package
patch level.

## Publishing a plugin

Declare entry points in your distribution's `pyproject.toml`. Optionally declare
the plugin-API major you target so Vincio can fail loud on a mismatch instead of
loading an incompatible plugin:

```toml
# my-vincio-plugin/pyproject.toml
[project.entry-points."vincio.connectors"]
acme = "my_plugin.connectors:make_acme_connector"

[project.entry-points."vincio.packs"]
acme_support = "my_plugin.packs:ACME_SUPPORT_PACK"

# Optional: declare the plugin-API major this distribution was built against.
[project.entry-points."vincio.plugins"]
api_version = "my_plugin:PLUGIN_API_VERSION"   # resolves to e.g. "1.0"
```

A connector factory is just a callable returning anything with an async
`load() -> list[Document]`; a pack entry point resolves to a `vincio.packs.Pack`.

## Discovery and loading

```python
from vincio.plugins import installed_plugins, load_plugins

for p in installed_plugins():
    print(p.kind, p.name, p.distribution, p.version, p.status)

load_plugins()   # register every compatible plugin into its registry
```

`installed_plugins()` (alias `discover_plugins()`) lists what is installed
**without importing the target objects**, so a broken or heavy plugin never
slows discovery, and an incompatible-major plugin is reported as `incompatible`.
`load_plugins()` actually imports and registers each compatible plugin; it is
idempotent and isolates a plugin that fails to import (reported as `error`,
never breaking the rest).

You rarely call `load_plugins()` directly: `connect("acme")` and
`load_pack("acme_support")` trigger discovery for their group on a name miss, so
an installed connector or pack simply resolves. Providers, embedders, and stores
self-register at their own first use.

From the CLI:

```bash
vincio plugins list
```

```text
vincio plugin API: 1.0
KIND       NAME                   DISTRIBUTION           VERSION    STATUS
connector  acme                   my-vincio-plugin       0.3.1      available
pack       acme_support           my-vincio-plugin       0.3.1      available
chunker    legacy                 old-plugin             9.0.0      incompatible  (targets plugin API 2.0; this Vincio provides 1.0)
```

## Compatibility gating

When a distribution declares its targeted plugin-API major via the
`vincio.plugins` / `api_version` entry point, Vincio compares it to
`PLUGIN_API_VERSION`. A major mismatch marks every plugin from that distribution
`incompatible` and skips loading it, a plugin written for a future contract
never half-loads against an older runtime, and vice-versa. A distribution that
doesn't declare a version is treated as compatible.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

_Recipes & extending Vincio:_
- [Cookbook: task-shaped recipes](cookbook.md)
- [Integrations: providers, vector stores, and frameworks](integrations.md)
- [Vertical packs: a regulated domain in one line](vertical-packs.md)
- [Reference: capability map](../reference/capability-map.md)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
