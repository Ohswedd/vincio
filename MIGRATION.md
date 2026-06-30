# Upgrading Vincio

**Upgrading is clean and mechanical, not a rewrite.** Every Vincio release has been
additive on a frozen public surface — new symbols and new optional parameters with
defaults, never a breaking rename of an API in active use — so a project that tracks
the library upgrades with **zero source changes**. This guide explains why, and how to
confirm it for your project.

## TL;DR

```bash
pip install --upgrade vincio        # upgrade to the latest release
vincio migrate <major>             # codemod: reports "no source changes required"
vincio doctor                      # confirm a clean tree on the new version
```

If `vincio migrate` reports no changes (it will, for any project tracking a recent
line) you are done.

## Why upgrades need no source changes

Everything Vincio has shipped is **additive on a frozen public surface**. The
[deprecation policy](docs/reference/stability.md) is followed mechanically, the surface
is held consistent by a build gate, and no public API in active use has ever reached its
`removed_in` runway. The "deprecation sweep" a major release usually performs therefore
removes nothing — the discipline pays off, and a clean upgrade is the normal case.

`vincio.API_VERSION` (`"5.0"`) is the frozen public-API **contract** version. The package
is versioned independently under SemVer; `API_VERSION` bumps only when the contract
surface that working code depends on changes, so it stays stable across additive releases.

## The `vincio migrate` codemod

`vincio migrate <target>` is the code-surface analogue of `vincio config migrate`: a
one-shot, **static** codemod (it parses your source with `ast`, never imports or runs it)
that rewrites the public symbols a breaking window renames, driven by a declarative
per-major rename table.

```bash
vincio migrate <major> [path]      # dry run: print the plan (default)
vincio migrate <major> --write     # apply the rewrites in place
vincio migrate <major> --check     # CI gate: exit non-zero if a migration is available
vincio migrate <major> --json      # machine-readable plan
```

The rename tables are currently **empty**, so the codemod reports *"no source changes are
required for this release"* on any project. The machinery ships anyway: it gives the
upgrade a truthful, automatable answer today, and it is the mechanism any future
consolidation — or the removal of a deprecated symbol — would be delivered through.

## Deprecations and the breaking-window contract

Removal always takes a deliberate, announced breaking window:

1. A public symbol is **never removed in a minor or patch release.** It is first marked
   deprecated in a **minor**, emitting a `VincioDeprecationWarning` that names the version
   it was deprecated in, the version scheduled for removal, and the replacement.
2. While deprecated, `vincio doctor` reports any project usage, naming the symbol, its
   replacement, and its removal version.
3. Removal happens no earlier than the **next major**, applied by `vincio migrate`.

No such window is currently open, so no public API is deprecated and no project needs to
change anything to stay current.

## Pinning

Pin a major range (for example `vincio>=6,<7`) to stay on a stable surface and pick up
bug-fix, security, and additive releases without surprises. Every guarantee carries
forward unchanged across the range: the published SLOs held by at-least-as-strict
VincioBench budgets, the CycloneDX SBOM and SLSA build-provenance attestation on every
release, the strict-typing ladder, and the completeness-gated error and API references.

## Troubleshooting

- **`vincio migrate` says nothing changed** — expected. Run `vincio doctor` to confirm
  the tree is clean.
- **A `VincioDeprecationWarning` appears after upgrading** — you are on a minor that
  deprecated something; the warning names the replacement and the removal version. Apply
  the named replacement, or run `vincio migrate <next-major>` when one is available.
- **Confirming integrity** — `vincio doctor` also flags a `vincio.yaml` that is behind
  the current config schema; run `vincio config migrate` to upgrade it.
