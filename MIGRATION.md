# Upgrading Vincio

**Upgrading is clean and mechanical, not a rewrite.** Every Vincio release has been additive on a
**frozen public surface** — new symbols and new optional parameters with defaults, never a breaking
rename of an API in active use — so a project that tracks the library upgrades with **zero source
changes**. This guide explains why, and how to confirm it for yours.

## Upgrade in 60 seconds

```bash
pip install --upgrade vincio        # move to the latest release
vincio doctor                       # list any deprecated API your project still uses (and a stale config)
vincio migrate 8.0                  # dry-run the one-shot rename codemod for the next major (--write applies)
```

If `vincio doctor` reports nothing, you are done — the tree is clean and every guarantee carries
forward. Upgrading to **7.10** still requires zero source changes: `7.6` (universal web browsing &
search), `7.7` (context anchors), `7.8` (LAGER), and `7.9` (the LAGER dense-signal tightening — new
optional `LazyOptions` fields and an `EvidenceIndex.semantic_similarity` method, all defaulting to the
byte-identical pure-stdlib path) are purely additive. `7.10` adds the opt-in experimental universal
reasoning engine (`app.use_reasoning_engine()` / `app.reason()`); existing `run` behavior is unchanged
until the engine is installed.

## The guarantee, and why it holds

Two version numbers, decoupled on purpose:

- **`vincio.__version__`** (`7.10.0`) — the release version, bumped every ship.
- **`vincio.API_VERSION`** (`"5.0"`) — the frozen public-API **contract** version. It bumps *only* when
  the surface working code depends on changes — so it stays stable across additive releases.

The public surface is `vincio.__all__` (plus documented subpackage entry points), held consistent by a
build gate that freezes it to `docs/reference/public-surface.txt`. No public API is ever removed before
it has spent at least one minor release **deprecated, warning, and codemod-covered**. A major's
"deprecation sweep" is therefore a one-command rewrite, not an investigation. The full policy is in
[stability & deprecation](docs/reference/stability.md).

## The open deprecation runway (removal scheduled for 8.0)

`7.5` opened the first scheduled breaking window: ten factory symbols were renamed to `build_*` and the
old names deprecated. Both spellings are exported until `8.0`; the old ones emit a
`VincioDeprecationWarning` naming the replacement. The CI professionalism budget pins the open runway at
*exactly* this set — an unplanned deprecation fails the build.

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

(`vincio.server.create_app` is deliberately **not** renamed — it is the ASGI-factory idiom servers and
frameworks look for.)

Three keyword/accessor spellings gained canonical forms on the same warn-until-8.0 runway — these are
**keyword** changes `vincio doctor` surfaces at call time:

- `verify_with=` → **`verifier=`** on the settlement/netting/arbitration/attestation verification
  functions.
- `at=` → **`as_of=`** on `vincio.security.identity` validity checks, matching the rest of the platform.
- `DocumentArtifact.sha256()`, `Recording.compute_digest()`, and `PromptNode.content_hash` (property) →
  **`.digest()`**; the canonical content-address *field* read stays **`.content_hash`** (signed
  artifacts keep their wire bytes frozen and expose it read-only).

## The `vincio migrate` codemod

`vincio migrate <target>` is a one-shot, **static** codemod: it parses your source with `ast` — never
imports or runs it — and rewrites the public symbols a breaking window renames, from a declarative
per-major table. Targets are `4.0` and `5.0` (both empty; those consolidations were additive) and
**`8.0`** (the ten renames above):

```bash
vincio migrate 8.0 [path]      # dry run: print the plan (default)
vincio migrate 8.0 --write     # apply the rewrites in place
vincio migrate 8.0 --check     # CI gate: exit non-zero if a migration is available
vincio migrate 8.0 --json      # machine-readable plan
```

Keyword and method renames (`verifier=`, `as_of=`, `.digest()`) are not symbol-table rewrites; both
spellings work until `8.0`, and the deprecation warnings pinpoint each call site. `vincio doctor` also
flags `verify_with=` statically on any call it can resolve to the library.

## The breaking-window contract

Removal always takes a deliberate, announced window:

1. A public symbol is **never removed in a minor or patch.** It is first marked deprecated in a
   **minor**, emitting a warning that names the version it was deprecated in, the version scheduled for
   removal, and the replacement.
2. While deprecated, `vincio doctor` reports any project usage — the symbol, its replacement, and its
   removal version.
3. Removal happens no earlier than the **next major**, applied by `vincio migrate`.

The `7.5` window above is the first to reach step 2. Nothing is removed today; a project that ignores it
entirely keeps working through every `7.x` release.

## Pinning & troubleshooting

Pin a major range (e.g. `vincio>=7,<8`) to pick up bug-fix, security, and additive releases without
surprises. Every guarantee carries forward unchanged across the range: the published SLOs held by
at-least-as-strict VincioBench budgets, the CycloneDX SBOM + SLSA provenance on every release, the
strict-typing ladder, and the completeness-gated error and API references.

- **`vincio migrate` says nothing changed** — your project uses no renamed symbol. Run `vincio doctor`
  to confirm the whole tree is clean.
- **A `VincioDeprecationWarning` appeared after upgrading** — you're on a minor that deprecated
  something; the warning names the replacement and removal version. Apply it, or run
  `vincio migrate 8.0 --write`.
- **A `vincio.yaml` is behind the schema** — `vincio doctor` flags it; run `vincio config migrate`.
