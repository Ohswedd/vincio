# Migrating to Vincio 4.0

Vincio **4.0** is the long-term-support major: the one announced breaking window
where the deprecation runway is collected and the public surface is re-frozen for
the 4.x line. This guide names every change and exactly how to apply it.

**The upgrade is clean and mechanical, not a rewrite.** A project that built on
3.x and fixed every deprecation warning needs **zero source changes** to run on
4.0 — see [Why there are no renames](#why-there-are-no-renames) below.

## TL;DR

```bash
pip install --upgrade vincio        # 3.x → 4.0.0
vincio migrate 4.0                  # codemod: reports "no source changes required"
vincio doctor                       # confirm a clean tree on the new major
```

If `vincio migrate 4.0` reports no changes (it will, for a project that tracked
3.x cleanly) you are done.

## What changed in 4.0

| Area | 3.x | 4.0 | Action |
| --- | --- | --- | --- |
| Package version | `3.49.0` | `4.0.0` | none — `pip install --upgrade vincio` |
| `vincio.API_VERSION` | `"3.0"` | `"4.0"` | none — read it if you branch on the contract version |
| Public surface (`vincio.__all__`) | 481 symbols | **same 481 symbols** | none — re-frozen identically |
| Deprecated APIs removed | — | **none** (the runway was empty) | none |
| Deprecation policy | next-major removal | unchanged | none |
| Support window | 3.x | latest 4.x | upgrade off pre-4.0 for security fixes |

No capability was removed and no symbol was renamed. The surface that was
additive from 1.0 → 3.49 is re-frozen unchanged under the 4.0 SemVer contract.

## The `vincio migrate 4.0` codemod

`vincio migrate <target>` is the code-surface analogue of `vincio config migrate`:
a one-shot, **static** codemod (it parses with `ast`, never imports or runs your
code) that rewrites the public symbols a major bump renames, driven by a
declarative rename table.

```bash
vincio migrate 4.0 [path]      # dry run: print the plan (default)
vincio migrate 4.0 --write     # apply the rewrites in place
vincio migrate 4.0 --check     # CI gate: exit non-zero if a migration is available
vincio migrate 4.0 --json      # machine-readable plan
```

For 4.0 the rename table is **empty**, so the codemod reports
*“no source changes are required for this release”* on any project. The machinery
ships anyway: it gives the upgrade a truthful, automatable answer today, and it is
the mechanism any future 4.x consolidation — or the 5.0 removal of a symbol
deprecated across the 4.x line — will be delivered through.

## Why there are no renames

Everything shipped from 1.0 through 3.49 was **additive on a frozen public
surface**: new symbols and new optional parameters with defaults, never a removal
or a breaking rename. Because the [deprecation policy](docs/reference/stability.md)
was followed mechanically and the surface was kept consistent across 40+ themes,
**no public API ever reached its `removed_in` runway** and no entry point drifted
badly enough to justify a breaking rename. The "deprecation sweep" a major usually
performs therefore removes nothing — the discipline paid off.

4.0 is consequently a *consolidation* major in the strict sense the
[roadmap](ROADMAP.md) set out: it re-freezes the surface and promotes the
contract version, without breaking working code.

## The 4.0 long-term-support contract

From 4.0 the public surface is re-frozen under
[Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html), carrying every
guarantee forward unchanged:

- **The frozen surface is mechanical.** `docs/reference/public-surface.txt` pins
  the exact 481-name surface; a test fails the build if `vincio.__all__` drifts
  from it, so no SemVer-significant change lands silently.
- **Removal still takes a major.** A symbol is deprecated in a 4.x minor (emitting
  `VincioDeprecationWarning` naming `since`, `removed_in`, and the replacement)
  and removed no earlier than 5.0 — `vincio doctor` reports any usage, and
  `vincio migrate 5.0` will rewrite it.
- **Provenance is unchanged.** Every release still ships a CycloneDX SBOM and SLSA
  build-provenance attestation; the published SLOs are held by at-least-as-strict
  VincioBench budgets; the strict-typing ladder and completeness-gated error and
  API references still gate the build.

After 4.0 the platform is in long-term support: bug-fix, security, and
standards-tracking releases on a stable 4.x surface.

## Troubleshooting

- **`vincio migrate` says nothing changed** — expected for 4.0. Run
  `vincio doctor` to confirm the tree is clean.
- **A `VincioDeprecationWarning` appears after upgrading** — you are on a 4.x
  minor that deprecated something; the warning names the replacement and the
  removal version. Run `vincio migrate <next-major>` when one is available, or
  apply the named replacement.
- **Pinning** — pin `vincio>=4,<5` to stay on the long-term-stable surface.
