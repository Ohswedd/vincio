# Migrating to Vincio 5.0

Vincio **5.0** is the second long-term-support major. It concludes the data &
analytics plane built additively across the 4.x line (4.1 → 5.0), re-freezes the
expanded public surface for the 5.x line, and promotes the contract version. This
guide names every change and exactly how to apply it.

**The upgrade is clean and mechanical, not a rewrite.** A project that built on
4.x needs **zero source changes** to run on 5.0 — the entire 4.x line was additive
on a frozen surface, so nothing was renamed or removed. See
[Why there are no renames](#why-there-are-no-renames) below.

## TL;DR

```bash
pip install --upgrade vincio        # 4.x → 5.0.0
vincio migrate 5.0                  # codemod: reports "no source changes required"
vincio doctor                       # confirm a clean tree on the new major
```

If `vincio migrate 5.0` reports no changes (it will, for any project on 4.x) you
are done.

## What changed in 5.0

| Area | 4.x | 5.0 | Action |
| --- | --- | --- | --- |
| Package version | `4.7.0` | `5.0.0` | none — `pip install --upgrade vincio` |
| `vincio.API_VERSION` | `"4.0"` | `"5.0"` | none — read it if you branch on the contract version |
| Public surface (`vincio.__all__`) | additive across 4.1–4.7 | **+5 symbols** (the data-engagement capstone) | none — purely additive |
| New entry point | — | `app.data_engagement` → `DataEngagement` / `DataNarrative` | opt-in; existing code is unaffected |
| Deprecated APIs removed | — | **none** (the runway was empty) | none |
| Deprecation policy | next-major removal | unchanged | none |
| Support window | 4.x | latest 5.x | upgrade off pre-5.0 for security fixes |

No capability was removed and no symbol was renamed. Every symbol added across the
4.x data & analytics plane — and the 5.0 `DataEngagement` capstone — is additive,
behind a new entry point or an opt-in extra. The data & analytics plane is now
**feature-complete and frozen**.

The five new public symbols in 5.0 (all re-exported from `vincio` and `vincio.data`):
`DataEngagement`, `DataNarrative`, `DataStage`, `DataEngagementSignature`, and
`DataEngagementVerification`.

## The `vincio migrate 5.0` codemod

`vincio migrate <target>` is the code-surface analogue of `vincio config migrate`:
a one-shot, **static** codemod (it parses with `ast`, never imports or runs your
code) that rewrites the public symbols a major bump renames, driven by a
declarative rename table.

```bash
vincio migrate 5.0 [path]      # dry run: print the plan (default)
vincio migrate 5.0 --write     # apply the rewrites in place
vincio migrate 5.0 --check     # CI gate: exit non-zero if a migration is available
vincio migrate 5.0 --json      # machine-readable plan
```

For 5.0 the rename table is **empty**, so the codemod reports
*“no source changes are required for this release”* on any project. The machinery
ships anyway: it gives the upgrade a truthful, automatable answer today, and it is
the mechanism any future consolidation — or the removal of a symbol deprecated
across the 5.x line — will be delivered through.

## Why there are no renames

Everything shipped from 1.0 through 5.0 was **additive on a frozen public
surface**: new symbols and new optional parameters with defaults, never a removal
or a breaking rename. Because the [deprecation policy](docs/reference/stability.md)
was followed mechanically and the surface was kept consistent across 50+ themes,
**no public API ever reached its `removed_in` runway** and no entry point drifted
badly enough to justify a breaking rename. The "deprecation sweep" a major usually
performs therefore removes nothing — the discipline paid off.

5.0 is consequently a *consolidation* major in the strict sense the
[roadmap](ROADMAP.md) set out: it re-freezes the (additively expanded) surface and
promotes the contract version, without breaking working code.

## The 5.0 long-term-support contract

From 5.0 the public surface is re-frozen under
[Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html), carrying every
guarantee forward unchanged:

- **The frozen surface is mechanical.** `docs/reference/public-surface.txt` pins
  the exact 508-name surface; a test fails the build if `vincio.__all__` drifts
  from it, so no SemVer-significant change lands silently.
- **Removal still takes a major.** A symbol is deprecated in a 5.x minor (emitting
  `VincioDeprecationWarning` naming `since`, `removed_in`, and the replacement)
  and removed no earlier than 6.0 — `vincio doctor` reports any usage, and
  `vincio migrate 6.0` will rewrite it.
- **Provenance is unchanged.** Every release still ships a CycloneDX SBOM and SLSA
  build-provenance attestation; the published SLOs are held by at-least-as-strict
  VincioBench budgets; the strict-typing ladder and completeness-gated error and
  API references still gate the build.

After 5.0 the platform is in long-term support: bug-fix, security, and
standards-tracking releases on a stable 5.x surface.

## Upgrading from 3.x

4.0 was the first long-term-support major — the one announced breaking window,
which broke nothing (the 1.0 → 3.49 surface was re-frozen unchanged). A project
still on 3.x upgrades the same clean, mechanical way: `pip install --upgrade
vincio`, `vincio migrate 4.0` then `vincio migrate 5.0` (both report "no source
changes required"), `vincio doctor`. Pin `vincio>=5,<6` to stay on the
long-term-stable surface.

## Troubleshooting

- **`vincio migrate` says nothing changed** — expected for 5.0. Run
  `vincio doctor` to confirm the tree is clean.
- **A `VincioDeprecationWarning` appears after upgrading** — you are on a 5.x
  minor that deprecated something; the warning names the replacement and the
  removal version. Run `vincio migrate <next-major>` when one is available, or
  apply the named replacement.
- **Pinning** — pin `vincio>=5,<6` to stay on the long-term-stable surface.
