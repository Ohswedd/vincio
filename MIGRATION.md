# Upgrading Vincio

**Upgrading is clean and mechanical, not a rewrite.** Every Vincio release has been
additive on a frozen public surface — new symbols and new optional parameters with
defaults, never a breaking rename of an API in active use — so a project that tracks
the library upgrades with **zero source changes**. This guide explains why, and how to
confirm it for your project.

## TL;DR

```bash
pip install --upgrade vincio        # upgrade to the latest release
vincio doctor                       # list any deprecated API your project still uses
vincio migrate 8.0                  # dry-run the one-shot rename codemod (--write applies)
```

Upgrading to `7.6` still requires **zero source changes** — every old name keeps
working, and `7.6` (universal web browsing & search) is purely additive.
`7.5` did open the first **scheduled** breaking window: ten factory symbols
were renamed to `build_*` and the old names deprecated, for removal in `8.0`.
`vincio doctor` tells you whether your project uses any of them, and
`vincio migrate 8.0` rewrites them in one shot whenever you choose — now, or any time
before you move to `8.0`.

## Why upgrades need no source changes

Everything Vincio has shipped is **additive on a frozen public surface**. The
[deprecation policy](docs/reference/stability.md) is followed mechanically, the surface
is held consistent by a build gate, and no public API is ever removed before it has
spent at least one minor release deprecated, warning, and codemod-covered. The
"deprecation sweep" a major release performs is therefore a one-command rewrite, not
an investigation.

`vincio.API_VERSION` (`"5.0"`) is the frozen public-API **contract** version. The package
is versioned independently under SemVer; `API_VERSION` bumps only when the contract
surface that working code depends on changes, so it stays stable across additive releases.

## The `7.5` deprecations (removal scheduled for `8.0`)

Old and new names are both exported until `8.0`; the old ones emit a
`VincioDeprecationWarning` naming the replacement. The CI professionalism budget pins
the open runway at exactly this set — an unplanned deprecation fails the build.

| Deprecated (works until 8.0)                  | Use instead                                   |
| --------------------------------------------- | --------------------------------------------- |
| `vincio.evals.make_retail_environment`        | `vincio.evals.build_retail_environment`       |
| `vincio.evals.make_counter_environment`       | `vincio.evals.build_counter_environment`      |
| `vincio.evals.make_vault_environment`         | `vincio.evals.build_vault_environment`        |
| `vincio.evals.make_agent_solver`              | `vincio.evals.build_agent_solver`             |
| `vincio.evals.make_env_solver`                | `vincio.evals.build_env_solver`               |
| `vincio.providers.make_finetune_backend`      | `vincio.providers.build_finetune_backend`     |
| `vincio.data.make_query_contract`             | `vincio.data.build_query_contract`            |
| `vincio.tools.make_web_checkout`              | `vincio.tools.build_web_checkout`             |
| `vincio.skills.make_script_handler`           | `vincio.skills.build_script_handler`          |
| `vincio.storage.create_metadata_store`        | `vincio.storage.build_metadata_store`         |

(`vincio.server.create_app` is deliberately **not** renamed — it is the ASGI-factory
idiom servers and frameworks look for.)

Alongside the symbol renames, three keyword/accessor spellings gained canonical forms
with the same warn-until-8.0 runway — these are **keyword** changes `vincio doctor`'s
warning stream surfaces at call time:

- `verify_with=` → **`verifier=`** on the settlement/netting/arbitration/attestation
  verification functions (methods already used `verifier=`).
- `at=` → **`as_of=`** on `vincio.security.identity` validity checks, matching the
  rest of the platform.
- `DocumentArtifact.sha256()`, `Recording.compute_digest()`, and the
  `PromptNode.content_hash` property → **`.digest()`**; the canonical content-address
  *field* read is **`.content_hash`** (signed artifacts keep their wire bytes frozen
  and expose it as a read-only property).

## The `vincio migrate` codemod

`vincio migrate <target>` is the code-surface analogue of `vincio config migrate`: a
one-shot, **static** codemod (it parses your source with `ast`, never imports or runs it)
that rewrites the public symbols a breaking window renames, driven by a declarative
per-major rename table. It knows three targets — `4.0` and `5.0` (both empty; those
consolidations were additive) and **`8.0`**, which carries the ten symbol renames above:

```bash
vincio migrate 8.0 [path]      # dry run: print the plan (default)
vincio migrate 8.0 --write     # apply the rewrites in place
vincio migrate 8.0 --check     # CI gate: exit non-zero if a migration is available
vincio migrate 8.0 --json      # machine-readable plan
```

Keyword renames (`verifier=`, `as_of=`) and method renames (`digest()`) are not
symbol-table rewrites; the deprecation warnings pinpoint each call site, and both
spellings work until `8.0`. `vincio doctor` additionally flags `verify_with=`
statically on any call it can resolve to the library (a `from vincio` import or a
vincio-module attribute); receiver-typed method calls are covered by the runtime
warning.

## Deprecations and the breaking-window contract

Removal always takes a deliberate, announced breaking window:

1. A public symbol is **never removed in a minor or patch release.** It is first marked
   deprecated in a **minor**, emitting a `VincioDeprecationWarning` that names the version
   it was deprecated in, the version scheduled for removal, and the replacement.
2. While deprecated, `vincio doctor` reports any project usage, naming the symbol, its
   replacement, and its removal version.
3. Removal happens no earlier than the **next major**, applied by `vincio migrate`.

The `7.5` window above is the first to reach step 2. Nothing is removed today; a project
that ignores it entirely keeps working through every `7.x` release.

## Pinning

Pin a major range (for example `vincio>=7,<8`) to stay on a stable surface and pick up
bug-fix, security, and additive releases without surprises. Every guarantee carries
forward unchanged across the range: the published SLOs held by at-least-as-strict
VincioBench budgets, the CycloneDX SBOM and SLSA build-provenance attestation on every
release, the strict-typing ladder, and the completeness-gated error and API references.

## Troubleshooting

- **`vincio migrate` says nothing changed** — your project doesn't use any renamed
  symbol. Run `vincio doctor` to confirm the tree is clean.
- **A `VincioDeprecationWarning` appears after upgrading** — you are on a minor that
  deprecated something; the warning names the replacement and the removal version. Apply
  the named replacement, or run `vincio migrate 8.0 --write` to rewrite the symbol
  renames in one shot.
- **Confirming integrity** — `vincio doctor` also flags a `vincio.yaml` that is behind
  the current config schema; run `vincio config migrate` to upgrade it.
