# Reference: API stability & deprecation policy

From **1.0**, Vincio follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html)
on its public API. This page is the contract: what's covered, what isn't, and
exactly how things get removed.

## What is "public API"

The public surface is:

- Every symbol re-exported from the top-level `vincio` package, i.e.
  `vincio.__all__`, also returned by `vincio.stability.public_api()`.
- The documented entry points of each subpackage listed in the
  [API reference](api.md).
- The `vincio` CLI commands documented in the [CLI reference](cli.md).
- The config schema (`vincio config schema`) field names and semantics.

Everything else is internal and may change at any time without notice:

- Any name beginning with an underscore (`_helper`, `vincio.core._runtime`).
- Modules and attributes not listed in the API reference.
- Symbols marked [`@experimental`](#experimental-apis).
- The exact wording of log lines, trace span internals, and error messages
  (error *types* and `.code` values are stable; message strings are not).

## Versioning guarantees

Within a major version (`1.x.y`):

| Bump | Example | Guarantee |
|---|---|---|
| **PATCH** | `1.0.0 → 1.0.1` | Bug fixes only. No public behavior changes. |
| **MINOR** | `1.0.0 → 1.1.0` | Additive only: new symbols, new **optional** parameters with defaults. Existing code keeps working. |
| **MAJOR** | `1.x → 2.0.0` | May remove or change public API, but only after the deprecation contract below. |

`vincio.API_VERSION` (`"5.0"`) is the contract version SemVer is applied
against; it changes only on a major bump. **5.0 is the current long-term-support
major:** the public surface — expanded additively across the 4.x data & analytics
plane — is re-frozen for the 5.x line (see
[The 5.0 long-term-support contract](#the-50-long-term-support-contract)).

## Deprecation contract

The policy is mechanical, not just documented:

1. A public symbol is **never removed in a minor or patch release.** It is first
   marked deprecated in a **minor** release.
2. While deprecated, calling it emits a `VincioDeprecationWarning` naming the
   version it was deprecated in, the version scheduled for removal, and the
   replacement.
3. Removal happens no earlier than the **next major** release.

```python
from vincio import deprecated, experimental, stability_of

@deprecated(since="1.2", removed_in="2.0", alternative="new_api")
def old_api(...): ...

@experimental(since="1.0", note="signature may change")
def fresh_api(...): ...

stability_of(old_api)
# {'level': StabilityLevel.DEPRECATED, 'since': '1.2',
#  'removed_in': '2.0', 'alternative': 'new_api', 'note': None}
```

Run `vincio doctor` to scan a project for any deprecated public API it still
uses, each finding names the symbol, its replacement, and its removal version,
read straight from the `stability_of` metadata above (the doctor also flags a
`vincio.yaml` that is behind the current schema). See the
[CLI reference](cli.md).

`VincioDeprecationWarning` subclasses the built-in `DeprecationWarning`, so it's
silent by default but visible under `python -W` and in test runs. To make
deprecations hard errors in your CI:

```python
import warnings
from vincio import VincioDeprecationWarning

warnings.simplefilter("error", VincioDeprecationWarning)
```

### Renamed symbols

A renamed function keeps its old name working via a forwarding alias for one
major cycle:

```python
from vincio.stability import deprecated_alias

new_name = _impl
old_name = deprecated_alias(new_name, old_name="old_name",
                            since="1.2", removed_in="2.0")
```

When a major bump renames public symbols, `vincio migrate <major>` rewrites a
project's source to the new names, the code-surface analogue of
`vincio config migrate`. It is a **static** codemod (it parses with `ast`, never
imports or runs your code) driven by a declarative, per-major rename table:

```bash
vincio migrate 4.0            # dry run: print the rewrite plan (default)
vincio migrate 4.0 --write    # apply the rewrites in place
vincio migrate 4.0 --check    # CI gate: exit non-zero if a migration is available
```

The `4.0` table is **empty**, the 3.x line reached no removal runway, so a clean
3.x → 4.0 upgrade needs no source changes (the codemod truthfully reports "no
source changes required"). See [MIGRATION.md](https://github.com/Ohswedd/vincio/blob/main/MIGRATION.md).

## Experimental APIs

Symbols marked `@experimental` are public and usable but carry **no stability
guarantee**, they may change or be removed in any release, including a minor.
They emit a one-time `VincioExperimentalWarning` per process so their status is
visible without being noisy. Use them, but pin your Vincio version if you
depend on their exact shape.

The public API is currently **stable end to end**, no shipped symbol is marked
`@experimental`. Future, unproven surface will carry the marker and emit the
warning until its shape settles, at which point it graduates to stable. The
marker is part of the contract, not a permanent state.

## The breaking-window contract

`API_VERSION` (returned by `vincio.stability.API_VERSION`) tracks the frozen
public-API contract and bumps only on a deliberate breaking window, announced in
advance and shipped through the mechanical deprecation runway above. Nothing
breaks *outside* such a window: across a minor or patch release, upgrading never
breaks working code.

## The 5.0 long-term-support contract

5.0 is the second long-term-support major. Like 4.0 before it, it **broke
nothing**: every release from 1.0 → 5.0 was additive on a frozen surface, the
deprecation policy above was followed mechanically, and no public API ever reached
its `removed_in` runway, so the deprecation sweep removed nothing and a project
that tracked 4.x cleanly upgrades with zero source changes. 5.0 concludes the data
& analytics plane (built additively across 4.1 → 5.0) and declares it
**feature-complete and frozen**.

From 5.0 the public surface is **re-frozen for the 5.x line** and the freeze is
mechanical, not just documented: the exact surface is pinned in
[`docs/reference/public-surface.txt`](public-surface.txt) and a build gate fails
the moment `vincio.__all__` drifts from it (regenerate deliberately with
`python -m vincio._apiref --freeze` and review the diff). Removal still takes a
major: a symbol deprecated across a 5.x minor is removed no earlier than 6.0, with
`vincio doctor` reporting any usage and `vincio migrate 6.0` rewriting it. (4.0 was
the first LTS major; `vincio migrate 4.0` and `vincio migrate 5.0` both report "no
source changes required".)

## Currently deprecated

No public APIs are currently deprecated. When one is, it appears here with its
replacement and removal version, and `stability_of(symbol)` reports the same
contract programmatically so tooling can detect it.

## Supported versions

See [SECURITY.md](https://github.com/Ohswedd/vincio/blob/main/SECURITY.md) for
which release lines receive security fixes.
